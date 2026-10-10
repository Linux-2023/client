"""ROS-independent pyAgxArm adapter. All public positions use radians/meters."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time

SDK_ROOT = Path(__file__).resolve().parents[2] / "third_party/pyAgxArm"
SDK_COMMIT = "841a625f5f4920e776f20b934eb13048b747e6d0"
JOINT_NAMES = tuple(f"joint{i}" for i in range(7))
STATUS_FIELDS = ("ctrl_mode", "arm_status", "mode_feedback", "teach_status",
                 "motion_status", "trajectory_num", "err_code")


def error_flags(code):
    """Piper protocol: low byte = communication, high byte = angle limits."""
    flags = {}
    for index in range(1, 7):
        flags[f"communication_status_joint_{index}"] = bool(code & (1 << (index - 1)))
        flags[f"joint_{index}_angle_limit"] = bool(code & (1 << (index + 7)))
    return flags


def fault_details(fields):
    flags = error_flags(fields["err_code"])
    communication = [i for i in range(1, 7) if flags[f"communication_status_joint_{i}"]]
    limits = [i for i in range(1, 7) if flags[f"joint_{i}_angle_limit"]]
    return (f"err_code=0x{fields['err_code']:04X}, communication_fault_joints={communication}, "
            f"angle_limit_joints={limits}")


def load_sdk():
    # Prefer the reviewed checkout; no installation into system Python required.
    if SDK_ROOT.is_dir():
        sys.path.insert(0, str(SDK_ROOT))
    import pyAgxArm
    return pyAgxArm


def finite_vector(values, length):
    if len(values) != length or any(isinstance(v, bool) for v in values):
        raise ValueError(f"expected {length} numeric values")
    result = [float(v) for v in values]
    if not all(math.isfinite(v) for v in result):
        raise ValueError("values must be finite")
    return result


class Arm:
    def __init__(self, robot, gripper, limits, *, feedback_timeout=0.5, gripper_max=None):
        self.robot, self.gripper, self.limits = robot, gripper, limits
        self.feedback_timeout = feedback_timeout
        self.gripper_max = gripper_max
        self.command = None
        self._speed_percent = None
        self._speed_update_at = None
        self.closed = False

    @classmethod
    def create(cls, channel, firmware, *, model="piper", interface="socketcan", **kwargs):
        sdk = load_sdk()
        config = sdk.create_agx_arm_config(
            robot=model, firmeware_version=firmware, channel=channel,
            interface=interface, auto_connect=False,
        )
        robot = sdk.AgxArmFactory.create_arm(config)
        try:
            gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
            robot.connect()
        except BaseException:
            robot.disconnect()
            raise
        limits = [config["joint_limits"][name] for name in config["joint_names"]]
        return cls(robot, gripper, limits, **kwargs)

    def fresh(self, sample, label, now):
        if sample is None:
            raise RuntimeError(f"missing {label}")
        age = now - float(sample.timestamp)
        if not math.isfinite(age) or not -0.1 <= age <= self.feedback_timeout:
            raise RuntimeError(f"stale {label} (age={age:.3f}s)")
        return sample.msg

    def read(self, *, require_enabled=False, allow_idle_standby=False, now=None):
        now = time.time() if now is None else now
        if self.robot.has_comm_error():
            raise RuntimeError(f"CAN error: {self.robot.get_comm_error()}")
        # The reviewed SDK's composite timestamp reflects only the last joint
        # pair. Check each underlying frame so a stale pair cannot look fresh.
        for pair in ("joint_12", "joint_34", "joint_56"):
            self.fresh(getattr(self.robot._parser, pair, None), pair, now)
        joints = self.fresh(self.robot.get_joint_angles(), "joint feedback", now)
        grip = self.fresh(self.gripper.get_gripper_status(), "gripper feedback", now)
        status = self.fresh(self.robot.get_arm_status(), "arm status", now)
        fields = {key: int(getattr(status, key)) for key in STATUS_FIELDS}
        if fields["arm_status"] or fields["err_code"]:
            raise RuntimeError(f"arm fault: {fields}; {fault_details(fields)}")
        if grip.mode != "width":
            raise RuntimeError("gripper feedback must use width mode (meters)")
        for index in range(1, 7):
            driver = self.fresh(self.robot.get_driver_states(index), f"driver {index}", now)
            if require_enabled and not driver.foc_status.driver_enable_status:
                raise RuntimeError(f"joint {index} disabled")
        idle_standby = (allow_idle_standby and fields["ctrl_mode"] == 0
                        and fields["mode_feedback"] in (0, 1) and fields["teach_status"] == 0)
        # Official set_speed_percent() sends move_mode=255 before move_j()
        # sends mode=1. Firmware may report that intermediate value briefly.
        speed_transition = (self._speed_update_at is not None
                            and 0 <= time.monotonic() - self._speed_update_at <= 0.1
                            and fields["ctrl_mode"] == 1 and fields["mode_feedback"] == 255
                            and fields["teach_status"] == 0)
        if require_enabled and not idle_standby and not speed_transition and (fields["ctrl_mode"] != 1 or fields["mode_feedback"] != 1):
            raise RuntimeError(f"expected CAN control / MOVE J: {fields}")
        state = finite_vector([*joints, grip.value], 7)
        return state, fields

    def validate_command(self, positions, velocity=(), effort=()):
        values = finite_vector(positions, 7)
        # Leave joint range handling to the official move_j implementation.
        # Its model-specific limits are disabled by default; do not impose a
        # stricter adapter policy that blocks a return from the measured pose.
        if self.gripper_max is not None and not 0 <= values[6] <= self.gripper_max:
            raise ValueError(f"gripper width {values[6]} m outside explicitly configured 0..{self.gripper_max} m")
        speed = 30
        if len(velocity):
            speeds = finite_vector(velocity, 7)
            if any(speeds):
                speed = min(100, max(30, round(speeds[6])))
        force = 1.0
        if len(effort):
            force = min(3.0, max(0.5, finite_vector(effort, 7)[6]))
        return values, speed, force

    def send(self, validated):
        values, speed, force = validated
        if speed != self._speed_percent:
            self._speed_update_at = time.monotonic()
            self.robot.set_speed_percent(speed)
            self._speed_percent = speed
        self.robot.move_j(values[:6])
        self.gripper.move_gripper_m(value=values[6], force=force)
        self.command = values.copy()

    def close(self):
        if not self.closed:
            self.closed = True
            self.robot.disconnect()


class DualController:
    """One fault locks both arms until process restart."""

    def __init__(self, arms, *, enabled=False, command_timeout=5.0):
        self.arms = arms
        self.enabled = enabled
        self.command_timeout = command_timeout
        self.last_command = {}
        self.command_count = {side: 0 for side in arms}
        self.fault = None

    def stop(self, reason):
        if self.fault is not None:
            return
        self.fault = str(reason)
        if self.enabled:
            for side, arm in self.arms.items():
                try:
                    arm.robot.electronic_emergency_stop()
                except Exception as exc:
                    self.fault += f"; {side} stop failed: {exc}"

    def poll(self):
        if self.fault is not None:
            raise RuntimeError(self.fault)
        try:
            now = time.monotonic()
            for side, stamp in self.last_command.items():
                if now - stamp > self.command_timeout:
                    raise RuntimeError(f"{side} command watchdog expired")
            snapshots = {}
            for side, arm in self.arms.items():
                try:
                    snapshots[side] = arm.read(
                        require_enabled=self.enabled, allow_idle_standby=not self.last_command,
                    )
                except Exception as exc:
                    count = self.command_count.get(side, 0)
                    if not self.last_command:
                        phase = "waiting for first action"
                    else:
                        age = now - self.last_command[side]
                        phase = f"action stream started, watchdog age={age:.3f}s"
                    raise RuntimeError(f"{side}: {exc}; {phase}; commands_sent={count}") from exc
            return snapshots
        except Exception as exc:
            self.stop(exc)
            raise

    def finish_zero(self):
        """End an explicitly completed zero task, retaining monitoring in idle."""
        if not self.enabled or not self.last_command:
            raise RuntimeError("no active enabled task")
        snapshots = self.poll()
        for side, (state, _) in snapshots.items():
            command = self.arms[side].command
            if command is None or any(abs(v) > 1e-6 for v in command):
                raise RuntimeError(f"{side}: last target is not zero")
            if any(abs(v) > 0.02 for v in state[:6]) or abs(state[6]) > 0.002:
                raise RuntimeError(f"{side}: zero feedback not converged")
        self.last_command.clear()

    def apply(self, side, positions, velocity=(), effort=()):
        if not self.enabled:
            return
        try:
            snapshots = self.poll()  # Check BOTH arms before either sends a command.
            command = self.arms[side].validate_command(positions, velocity, effort)
            if not self.last_command:
                self._prepare_first_action(snapshots)
            self.arms[side].send(command)
            self.command_count[side] = self.command_count.get(side, 0) + 1
            now = time.monotonic()
            if not self.last_command:
                self.last_command = {name: now for name in self.arms}
            self.last_command[side] = now
        except Exception as exc:
            self.stop(exc)
            raise

    def _prepare_first_action(self, snapshots):
        """Leave pre-action standby only on a valid command, with feedback ACK.

        Never reset, re-enable motors, or send a position to establish the mode.
        After the first action, any loss of mode remains a latched fault.
        """
        requested = {}
        for side, (_, status) in snapshots.items():
            if status["ctrl_mode"] == 0:
                requested[side] = time.time()
                self.arms[side].robot.set_motion_mode("j")
        deadline = time.monotonic() + 0.5
        while requested:
            current = self.poll()
            for side in list(requested):
                status = current[side][1]
                sample = self.arms[side].robot.get_arm_status()
                if (status["ctrl_mode"] == 1 and status["mode_feedback"] == 1
                        and sample is not None and sample.timestamp > requested[side]):
                    del requested[side]
            if not requested:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"MOVE J confirmation timed out before first action: {list(requested)}")
            time.sleep(0.005)
        # Recheck both arms strictly before transmitting any target.
        for side, arm in self.arms.items():
            try:
                arm.read(require_enabled=True)
            except Exception as exc:
                raise RuntimeError(f"{side}: not ready for first action: {exc}") from exc
