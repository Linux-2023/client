"""Direct ROS 2 adapter for unchanged ``control_your_robot`` PiperController."""

from __future__ import annotations

import argparse
import math
import numbers
import signal
import sys
from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
import rclpy
from piper_msgs.msg import PiperStatusMsg
from rclpy.node import Node
from sensor_msgs.msg import JointState

JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper")
GRIPPER_MAX_METERS = 0.07
STATUS_FIELDS = ("ctrl_mode", "arm_status", "mode_feed", "motion_status", "err_code")


def _load_piper_controller() -> type:
    """Import the third-party controller only after the explicit enable gate."""
    from robot.controller.Piper_controller import PiperController

    return PiperController


def _validate_action(values: Sequence[float]) -> list[float]:
    if len(values) != 7:
        raise ValueError("direct SDK action must contain exactly seven values")
    converted: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"direct SDK action value {index} must be a real numeric scalar")
        if not isinstance(value, (numbers.Real, np.integer, np.floating)):
            raise ValueError(f"direct SDK action value {index} must be a real numeric scalar")
        converted.append(float(value))
    if not all(math.isfinite(value) for value in converted):
        raise ValueError("direct SDK action values must be finite")
    gripper = converted[6]
    if not 0.0 <= gripper <= GRIPPER_MAX_METERS:
        raise ValueError(f"direct SDK gripper meters must be within 0.0..{GRIPPER_MAX_METERS}")
    return converted


class DirectSdkArm:
    """Small injectable wrapper around one unchanged third-party PiperController."""

    def __init__(self, name: str, can_port: str, *, controller_factory: Callable[[str], Any]) -> None:
        self.name = name
        self.can_port = can_port
        self._controller_factory = controller_factory
        self.controller: Any | None = None
        self.locked_reason: str | None = None
        self._shutdown_complete = False

    def setup(self) -> None:
        self.controller = self._controller_factory(self.name)
        self.controller.set_up(self.can_port)

    def read_state(self) -> list[float]:
        controller = self._require_controller()
        state = controller.get_state()
        joints = [float(value) for value in state["joint"]]
        if len(joints) != 6 or not all(math.isfinite(value) for value in joints):
            raise RuntimeError(f"{self.name} SDK returned invalid joint state")
        gripper = float(state["gripper"])
        if not math.isfinite(gripper):
            raise RuntimeError(f"{self.name} SDK returned invalid gripper state")
        return joints + [gripper * GRIPPER_MAX_METERS]

    def apply_action(self, values: Sequence[float]) -> list[float]:
        if self.locked_reason is not None:
            raise RuntimeError(f"{self.name} direct SDK actions locked: {self.locked_reason}")
        controller = self._require_controller()
        action = _validate_action(values)
        controller.set_joint(np.asarray(action[:6], dtype=float))
        controller.set_gripper(action[6] / GRIPPER_MAX_METERS)
        return action

    def read_status(self) -> tuple[dict[str, int], str | None]:
        controller = self._require_controller()
        raw = getattr(controller, "controller", None)
        fields = {field: 0 for field in STATUS_FIELDS}
        if raw is None:
            return fields, self._lock("missing raw SDK controller")
        try:
            status_holder = raw.GetArmStatus()
            status = status_holder.arm_status
        except Exception as exc:  # noqa: BLE001 - status projection must surface SDK shape problems.
            return fields, self._lock(f"missing SDK status object: {exc}")
        missing_fields: list[str] = []
        for field in STATUS_FIELDS:
            try:
                fields[field] = int(getattr(status, field))
            except AttributeError:
                missing_fields.append(field)
            except Exception as exc:  # noqa: BLE001 - malformed fields are unsafe status data.
                missing_fields.append(f"{field}: {exc}")
        if missing_fields:
            fields["ctrl_mode"] = 0
            if len(missing_fields) == 1:
                return fields, self._lock(f"missing SDK status field {missing_fields[0]}")
            return fields, self._lock(f"missing SDK status fields {', '.join(missing_fields)}")
        try:
            enabled = tuple(raw.GetArmEnableStatus())
        except Exception as exc:  # noqa: BLE001 - status projection must surface SDK shape problems.
            fields["ctrl_mode"] = 0
            return fields, self._lock(f"missing SDK enable status: {exc}")
        if len(enabled) != 6:
            fields["ctrl_mode"] = 0
            return fields, self._lock(f"expected six driver enable flags, got {len(enabled)}")
        for index, is_enabled in enumerate(enabled, start=1):
            if not bool(is_enabled):
                fields["ctrl_mode"] = 0
                return fields, self._lock(f"driver {index} disabled")
        return fields, None

    def shutdown(self) -> list[str]:
        if self._shutdown_complete:
            return []
        self._shutdown_complete = True
        errors: list[str] = []
        if self.controller is None:
            return errors
        raw = getattr(self.controller, "controller", None)
        if raw is None:
            return errors
        disable = getattr(raw, "DisableArm", None)
        if callable(disable):
            try:
                disable(7)
            except Exception as exc:  # noqa: BLE001 - cleanup failures must be reported.
                errors.append(f"{self.name}: {exc}")
        disconnect = getattr(raw, "DisconnectPort", None)
        if callable(disconnect):
            try:
                disconnect()
            except Exception as exc:  # noqa: BLE001 - cleanup failures must be reported.
                errors.append(f"{self.name}: {exc}")
        return errors

    def _require_controller(self) -> Any:
        if self.controller is None:
            raise RuntimeError(f"{self.name} direct SDK arm has not been set up")
        return self.controller

    def _lock(self, reason: str) -> str:
        if self.locked_reason is None:
            self.locked_reason = reason
        return self.locked_reason


class PiperDirectSdkAdapter(Node):
    """ROS 2 node exposing direct SDK feedback, command echo, and status topics."""

    def __init__(self, *, left_arm: DirectSdkArm, right_arm: DirectSdkArm, setup_arms: bool = True) -> None:
        super().__init__("piper_direct_sdk_adapter")
        self.left_arm = left_arm
        self.right_arm = right_arm
        self._arms = {"left": left_arm, "right": right_arm}
        self._shutting_down = False
        self._shutdown_complete = False
        self._timer = None
        try:
            if setup_arms:
                self.left_arm.setup()
                self.right_arm.setup()
            self._feedback_publishers = {
                "left": self.create_publisher(JointState, "/direct_sdk/joint_left", 10),
                "right": self.create_publisher(JointState, "/direct_sdk/joint_right", 10),
            }
            self._command_publishers = {
                "left": self.create_publisher(JointState, "/direct_sdk/joint_ctrl_left", 10),
                "right": self.create_publisher(JointState, "/direct_sdk/joint_ctrl_right", 10),
            }
            self._status_publishers = {
                "left": self.create_publisher(PiperStatusMsg, "/direct_sdk/arm_status_left", 10),
                "right": self.create_publisher(PiperStatusMsg, "/direct_sdk/arm_status_right", 10),
            }
            self._subscriptions = [
                self.create_subscription(JointState, "/direct_sdk/joint_cmd_left", self._make_action_callback("left"), 10),
                self.create_subscription(JointState, "/direct_sdk/joint_cmd_right", self._make_action_callback("right"), 10),
            ]
            self._last_command_echo: dict[str, list[float] | None] = {"left": None, "right": None}
            self._timer = self.create_timer(0.05, self._on_timer)
        except Exception:
            self.shutdown()
            raise

    @property
    def controllers(self) -> list[Any]:
        return [self.left_arm.controller, self.right_arm.controller]

    def _on_timer(self) -> None:
        if self._shutting_down:
            return
        for side, arm in self._arms.items():
            try:
                feedback = arm.read_state()
                feedback_msg = self._make_joint_state(feedback)
                self._feedback_publishers[side].publish(feedback_msg)
                if self._last_command_echo[side] is None:
                    self._last_command_echo[side] = list(feedback)
                    self._command_publishers[side].publish(self._make_joint_state(feedback))
                self._publish_status(side, arm)
            except Exception as exc:  # noqa: BLE001 - ROS timer must survive and latch unsafe actions.
                arm._lock(str(exc))
                self.get_logger().error(f"{side} direct SDK timer failure: {exc}")

    def _make_action_callback(self, side: str):
        def callback(message: JointState) -> None:
            if self._shutting_down:
                raise RuntimeError("direct SDK adapter is shutting down")
            action = self._extract_action(message)
            accepted = self._arms[side].apply_action(action)
            self._last_command_echo[side] = list(accepted)
            self._command_publishers[side].publish(self._make_joint_state(accepted))

        return callback

    def _extract_action(self, message: JointState) -> list[float]:
        if len(message.position) != 7:
            raise ValueError("direct SDK JointState command must contain seven positions")
        return _validate_action(message.position)

    def _make_joint_state(self, values: Sequence[float]) -> JointState:
        message = JointState()
        stamp = self.get_clock().now().to_msg()
        message.header.stamp.sec = int(getattr(stamp, "sec"))
        message.header.stamp.nanosec = int(getattr(stamp, "nanosec"))
        message.name = list(JOINT_NAMES)
        message.position = list(values)
        return message

    def _publish_status(self, side: str, arm: DirectSdkArm) -> None:
        fields, reason = arm.read_status()
        message = PiperStatusMsg()
        message.ctrl_mode = fields["ctrl_mode"]
        message.arm_status = fields["arm_status"]
        message.mode_feedback = fields["mode_feed"]
        message.motion_status = fields["motion_status"]
        message.err_code = fields["err_code"]
        self._status_publishers[side].publish(message)
        if reason is not None:
            self.get_logger().warning(f"{side} direct SDK actions locked: {reason}")

    def shutdown(self) -> list[str]:
        if getattr(self, "_shutdown_complete", False):
            return []
        self._shutdown_complete = True
        self._shutting_down = True
        timer = getattr(self, "_timer", None)
        if timer is not None and hasattr(timer, "cancel"):
            timer.cancel()
        errors: list[str] = []
        for arm in (self.left_arm, self.right_arm):
            errors.extend(arm.shutdown())
        return errors


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct Piper SDK ROS 2 adapter")
    parser.add_argument("--left-can", required=True)
    parser.add_argument("--right-can", required=True)
    parser.add_argument("--allow-enable", action="store_true", help="required before importing or enabling PiperController")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.allow_enable:
        print("--allow-enable is required before loading PiperController", file=sys.stderr)
        return 2
    controller_type = _load_piper_controller()
    left_arm = DirectSdkArm("left", args.left_can, controller_factory=controller_type)
    right_arm = DirectSdkArm("right", args.right_can, controller_factory=controller_type)
    node: PiperDirectSdkAdapter | None = None
    shutdown_errors: list[str] = []
    exit_code = 0
    rclpy_initialized = False
    old_sigint = None
    old_sigterm = None

    def _handle_signal(_signum, _frame) -> None:
        raise KeyboardInterrupt

    try:
        rclpy.init(args=None)
        rclpy_initialized = True
        node = PiperDirectSdkAdapter(left_arm=left_arm, right_arm=right_arm)
        old_sigint = signal.signal(signal.SIGINT, _handle_signal)
        old_sigterm = signal.signal(signal.SIGTERM, _handle_signal)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
    except Exception as exc:  # noqa: BLE001 - startup failures must clean up and return nonzero.
        print(exc, file=sys.stderr)
        exit_code = 1
    finally:
        if old_sigint is not None:
            signal.signal(signal.SIGINT, old_sigint)
        if old_sigterm is not None:
            signal.signal(signal.SIGTERM, old_sigterm)
        if node is not None:
            shutdown_errors = node.shutdown()
            node.destroy_node()
        else:
            shutdown_errors = left_arm.shutdown() + right_arm.shutdown()
        if rclpy_initialized:
            rclpy.shutdown()
    for error in shutdown_errors:
        print(error, file=sys.stderr)
    return 1 if shutdown_errors or exit_code else 0


if __name__ == "__main__":
    raise SystemExit(main())
