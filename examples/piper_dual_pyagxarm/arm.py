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
    def __init__(self, robot, gripper, limits, *, feedback_timeout=0.5, motion_mode="j"):
        if motion_mode not in ("j", "js"):
            raise ValueError("motion_mode must be j or js")
        self.motion_mode = motion_mode
        self.robot, self.gripper, self.limits = robot, gripper, limits
        self.feedback_timeout = feedback_timeout
        self.command = None
        self._speed_percent = None
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
        joint_sample = self.robot.get_joint_angles()
        if joint_sample is None:
            raw = []
            for pair, indices in (("joint_12", (1, 2)), ("joint_34", (3, 4)), ("joint_56", (5, 6))):
                message = self.fresh(getattr(self.robot._parser, pair, None), pair, now)
                raw.extend(getattr(message, f"joint_{i}") for i in indices)
            raise RuntimeError(f"official get_joint_angles() returned None; decoded joints(rad)={raw}; "
                               "SDK feedback validation requires each joint within +/-2pi")
        joints = self.fresh(joint_sample, "joint feedback", now)
        grip = self.fresh(self.gripper.get_gripper_status(), "gripper feedback", now)
        status = self.fresh(self.robot.get_arm_status(), "arm status", now)
        fields = {key: int(getattr(status, key)) for key in STATUS_FIELDS}
        if grip.mode != "width":
            raise RuntimeError("gripper feedback must use width mode (meters)")
        state = finite_vector([*joints, grip.value], 7)
        return state, fields

    def validate_command(self, positions, velocity=(), effort=()):
        values = finite_vector(positions, 7)
        # Leave joint range handling to the official move_j implementation.
        # Its model-specific limits are disabled by default; do not impose a
        # stricter adapter policy that blocks a return from the measured pose.
        speed = 30
        if len(velocity):
            speeds = finite_vector(velocity, 7)
            speed = round(speeds[6])
        force = 1.0
        if len(effort):
            force = finite_vector(effort, 7)[6]
        return values, speed, force

    def send(self, validated):
        values, speed, force = validated
        if speed != self._speed_percent:
            self.robot.set_speed_percent(speed)
            self._speed_percent = speed
        if self.motion_mode == "js":
            self.robot.move_js(values[:6])
        else:
            self.robot.move_j(values[:6])
        self.gripper.move_gripper_m(value=values[6], force=force)
        self.command = values.copy()

    def close(self):
        if not self.closed:
            self.closed = True
            self.robot.disconnect()


class DualController:
    """ROS dispatch only; motor control and range handling belong to the SDK."""

    def __init__(self, arms, *, enabled=False):
        self.arms = arms
        self.enabled = enabled
        self.command_count = {}
        self.feedback_errors = {}

    def poll(self):
        snapshots = {}
        for side, arm in self.arms.items():
            try:
                snapshots[side] = arm.read()
                self.feedback_errors.pop(side, None)
            except RuntimeError as exc:
                self.feedback_errors[side] = str(exc)
                # Do not publish stale data as a new ROS measurement.
                continue
        return snapshots

    def finish_zero(self):
        # Compatibility notification; no watchdog or motion gate to release.
        pass

    def apply(self, side, positions, velocity=(), effort=()):
        if self.enabled:
            arm = self.arms[side]
            arm.send(arm.validate_command(positions, velocity, effort))
            self.command_count[side] = self.command_count.get(side, 0) + 1
