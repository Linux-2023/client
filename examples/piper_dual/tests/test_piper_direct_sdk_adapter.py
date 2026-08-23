"""Focused fake-controller tests for the direct Piper SDK ROS adapter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import piper_direct_sdk_adapter as adapter
from piper_direct_sdk_adapter import DirectSdkArm
from piper_direct_sdk_adapter import PiperDirectSdkAdapter


class FakeRawSdk:
    def __init__(self, owner: "FakeController") -> None:
        self._owner = owner
        self.status = SimpleNamespace(ctrl_mode=1, arm_status=0, mode_feed=1, motion_status=0, err_code=0)
        self.enabled = (True, True, True, True, True, True)
        self.disconnect_error: Exception | None = None

    def GetArmStatus(self):
        return SimpleNamespace(arm_status=self.status)

    def GetArmEnableStatus(self):
        return tuple(self.enabled)

    def DisableArm(self, axis: int) -> None:
        self._owner.disable_calls.append(axis)

    def DisconnectPort(self) -> None:
        if self.disconnect_error is not None:
            raise self.disconnect_error
        self._owner.disconnect_calls.append(True)


class FakeController:
    instances: list["FakeController"] = []
    setup_failures: dict[str, Exception] = {}

    def __init__(self, name: str) -> None:
        self.name = name
        self.can_port: str | None = None
        self.state = {"joint": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "gripper": 0.0}
        self.set_joint_calls: list[list[float]] = []
        self.set_gripper_calls: list[object] = []
        self.disable_calls: list[int] = []
        self.disconnect_calls: list[bool] = []
        self.controller = FakeRawSdk(self)
        FakeController.instances.append(self)

    def set_up(self, can_port: str) -> None:
        if self.name in self.setup_failures:
            raise self.setup_failures[self.name]
        self.can_port = can_port

    def get_state(self):
        return {"joint": list(self.state["joint"]), "gripper": self.state["gripper"]}

    def set_joint(self, joints) -> None:
        self.set_joint_calls.append(list(joints))

    def set_gripper(self, gripper) -> None:
        self.set_gripper_calls.append(gripper)


class FakePublisher:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self.messages: list[object] = []

    def publish(self, message: object) -> None:
        self.messages.append(message)


class FakeTimer:
    def __init__(self, period: float, callback) -> None:
        self.period = period
        self.callback = callback
        self.cancel_calls = 0

    def cancel(self) -> None:
        self.cancel_calls += 1


class FakeClock:
    def now(self):
        return self

    def to_msg(self):
        return SimpleNamespace(sec=12, nanosec=345)


def _joint_message(values):
    msg = adapter.JointState()
    msg.name = list(adapter.JOINT_NAMES)
    msg.position = list(values)
    return msg


@pytest.fixture(autouse=True)
def _reset_fake_instances():
    FakeController.instances = []
    FakeController.setup_failures = {}
    yield
    FakeController.instances = []
    FakeController.setup_failures = {}


def make_fake_adapter(monkeypatch):
    capture = SimpleNamespace(publishers={}, subscriptions={}, timers=[])

    monkeypatch.setattr(adapter.Node, "__init__", lambda self, *_args, **_kwargs: None)

    def create_publisher(self, msg_type, topic, qos_depth):
        publisher = FakePublisher(topic)
        capture.publishers[topic] = publisher
        return publisher

    def create_subscription(self, msg_type, topic, callback, qos_depth):
        capture.subscriptions[topic] = SimpleNamespace(msg_type=msg_type, callback=callback, qos_depth=qos_depth)
        return capture.subscriptions[topic]

    def create_timer(self, period, callback):
        timer = FakeTimer(period, callback)
        capture.timers.append(timer)
        return timer

    monkeypatch.setattr(PiperDirectSdkAdapter, "create_publisher", create_publisher)
    monkeypatch.setattr(PiperDirectSdkAdapter, "create_subscription", create_subscription)
    monkeypatch.setattr(PiperDirectSdkAdapter, "create_timer", create_timer)
    monkeypatch.setattr(PiperDirectSdkAdapter, "get_clock", lambda self: FakeClock())
    monkeypatch.setattr(PiperDirectSdkAdapter, "get_logger", lambda self: SimpleNamespace(error=lambda *_args, **_kwargs: None, warning=lambda *_args, **_kwargs: None))

    left = DirectSdkArm("left", "can_left", controller_factory=FakeController)
    right = DirectSdkArm("right", "can_right", controller_factory=FakeController)
    node = PiperDirectSdkAdapter(left_arm=left, right_arm=right)
    capture.controllers = [left.controller, right.controller]
    return node, capture


def test_adapter_refuses_to_load_controller_without_allow_enable(monkeypatch):
    loaded = []
    monkeypatch.setattr(adapter, "_load_piper_controller", lambda: loaded.append(True))

    assert adapter.main(["--left-can", "can_left", "--right-can", "can_right"]) == 2
    assert loaded == []


def test_apply_action_converts_meter_gripper_to_normalized_controller_input():
    arm = DirectSdkArm("left", "can_left", controller_factory=FakeController)
    arm.setup()

    arm.apply_action([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.035])

    assert arm.controller.set_joint_calls == [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
    assert arm.controller.set_gripper_calls == [pytest.approx(0.5)]


def test_read_state_converts_normalized_gripper_to_meters():
    arm = DirectSdkArm("left", "can_left", controller_factory=FakeController)
    arm.setup()
    arm.controller.state["joint"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    arm.controller.state["gripper"] = 0.5

    assert arm.read_state() == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.035])


@pytest.mark.parametrize("gripper", [-0.001, 0.071])
def test_apply_action_rejects_gripper_outside_meter_contract(gripper):
    arm = DirectSdkArm("left", "can_left", controller_factory=FakeController)
    arm.setup()

    with pytest.raises(ValueError, match="0.0.*0.07"):
        arm.apply_action([0.0] * 6 + [gripper])


@pytest.mark.parametrize("values", ([0.0] * 6, [0.0] * 8, [0.0] * 6 + [float("nan")]))
def test_apply_action_rejects_invalid_action_vectors(values):
    arm = DirectSdkArm("left", "can_left", controller_factory=FakeController)
    arm.setup()

    with pytest.raises(ValueError):
        arm.apply_action(values)


@pytest.mark.parametrize("bad_value", [True, "0.0", 1 + 0j])
def test_apply_action_rejects_bool_string_and_non_real_scalars_before_conversion(bad_value):
    arm = DirectSdkArm("left", "can_left", controller_factory=FakeController)
    arm.setup()

    with pytest.raises(ValueError, match="real numeric"):
        arm.apply_action([bad_value, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert arm.controller.set_joint_calls == []
    assert arm.controller.set_gripper_calls == []


def test_apply_action_accepts_numpy_real_scalars():
    arm = DirectSdkArm("left", "can_left", controller_factory=FakeController)
    arm.setup()

    arm.apply_action([np.float64(0.0), np.float32(0.1), np.int64(0), 0.0, 0.0, 0.0, np.float64(0.035)])

    assert arm.controller.set_gripper_calls == [pytest.approx(0.5)]


def test_adapter_publishes_actual_feedback_and_separate_command_echo(monkeypatch):
    node, capture = make_fake_adapter(monkeypatch)
    left = node.left_arm.controller
    left.state["joint"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    left.state["gripper"] = 0.25

    node._on_timer()

    feedback = capture.publishers["/direct_sdk/joint_left"].messages[-1]
    command_echo = capture.publishers["/direct_sdk/joint_ctrl_left"].messages[-1]
    assert list(feedback.name) == list(adapter.JOINT_NAMES)
    assert list(feedback.position) == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0175])
    assert list(command_echo.position) == pytest.approx(list(feedback.position))

    command = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.035]
    capture.subscriptions["/direct_sdk/joint_cmd_left"].callback(_joint_message(command))

    assert left.set_joint_calls == [[0.6, 0.5, 0.4, 0.3, 0.2, 0.1]]
    assert left.set_gripper_calls == [pytest.approx(0.5)]
    assert list(capture.publishers["/direct_sdk/joint_ctrl_left"].messages[-1].position) == pytest.approx(command)

    left.state["joint"] = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    left.state["gripper"] = 1.0
    node._on_timer()

    assert list(capture.publishers["/direct_sdk/joint_left"].messages[-1].position) == pytest.approx([6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.07])
    assert list(capture.publishers["/direct_sdk/joint_ctrl_left"].messages[-1].position) == pytest.approx(command)


def test_status_uses_sdk_fields_and_disabled_driver_locks_actions(monkeypatch):
    node, capture = make_fake_adapter(monkeypatch)
    left = node.left_arm.controller
    left.controller.status = SimpleNamespace(ctrl_mode=1, arm_status=3, mode_feed=2, motion_status=4, err_code=5)
    left.controller.enabled = (True, True, False, True, True, True)

    node._on_timer()

    status = capture.publishers["/direct_sdk/arm_status_left"].messages[-1]
    assert status.ctrl_mode == 0
    assert status.arm_status == 3
    assert status.mode_feedback == 2
    assert status.motion_status == 4
    assert status.err_code == 5
    assert "driver 3 disabled" in node.left_arm.locked_reason

    left.controller.enabled = (True, True, True, True, True, True)
    with pytest.raises(RuntimeError, match="driver 3 disabled"):
        node.left_arm.apply_action([0.0] * 7)


def test_missing_sdk_status_field_projects_not_ready_and_locks_actions(monkeypatch):
    node, capture = make_fake_adapter(monkeypatch)
    left = node.left_arm.controller
    left.controller.status = SimpleNamespace(ctrl_mode=1, arm_status=0, motion_status=0, err_code=0)

    node._on_timer()

    status = capture.publishers["/direct_sdk/arm_status_left"].messages[-1]
    assert status.ctrl_mode == 0
    assert status.arm_status == 0
    assert status.motion_status == 0
    assert status.err_code == 0
    assert "mode_feed" in node.left_arm.locked_reason
    with pytest.raises(RuntimeError, match="mode_feed"):
        capture.subscriptions["/direct_sdk/joint_cmd_left"].callback(_joint_message([0.0] * 7))


def test_missing_status_field_preserves_available_real_fault_fields(monkeypatch):
    node, capture = make_fake_adapter(monkeypatch)
    left = node.left_arm.controller
    left.controller.status = SimpleNamespace(ctrl_mode=1, arm_status=9, motion_status=4, err_code=64)

    node._on_timer()

    status = capture.publishers["/direct_sdk/arm_status_left"].messages[-1]
    assert status.ctrl_mode == 0
    assert status.arm_status == 9
    assert status.mode_feedback == 0
    assert status.motion_status == 4
    assert status.err_code == 64
    assert node.left_arm.locked_reason == "missing SDK status field mode_feed"


def test_shutdown_disables_and_disconnects_both_arms(monkeypatch):
    node, capture = make_fake_adapter(monkeypatch)

    errors = node.shutdown()

    assert errors == []
    assert all(controller.disable_calls == [7] for controller in capture.controllers)
    assert all(controller.disconnect_calls == [True] for controller in capture.controllers)
    assert capture.timers[0].cancel_calls == 1


def test_shutdown_is_idempotent(monkeypatch):
    node, capture = make_fake_adapter(monkeypatch)

    assert node.shutdown() == []
    assert node.shutdown() == []

    assert all(controller.disable_calls == [7] for controller in capture.controllers)
    assert all(controller.disconnect_calls == [True] for controller in capture.controllers)
    assert capture.timers[0].cancel_calls == 1


def test_shutdown_reports_cleanup_failure_after_attempting_both_arms(monkeypatch):
    node, capture = make_fake_adapter(monkeypatch)
    capture.controllers[1].controller.disconnect_error = RuntimeError("right disconnect failed")

    errors = node.shutdown()

    assert errors == ["right: right disconnect failed"]
    assert capture.controllers[0].disable_calls == [7]
    assert capture.controllers[1].disable_calls == [7]
    assert capture.controllers[0].disconnect_calls == [True]
    assert capture.controllers[1].disconnect_calls == []


def test_second_arm_setup_failure_cleans_first_setup_arm_and_shuts_down_rclpy(monkeypatch):
    FakeController.setup_failures = {"right": RuntimeError("right setup failed")}
    shutdown_calls = []
    monkeypatch.setattr(adapter, "_load_piper_controller", lambda: FakeController)
    monkeypatch.setattr(adapter.rclpy, "init", lambda args=None: None)
    monkeypatch.setattr(adapter.rclpy, "shutdown", lambda: shutdown_calls.append(True))
    monkeypatch.setattr(adapter.rclpy, "spin", lambda node: None)
    monkeypatch.setattr(adapter.Node, "__init__", lambda self, *_args, **_kwargs: None)

    assert adapter.main(["--left-can", "can_left", "--right-can", "can_right", "--allow-enable"]) == 1

    assert FakeController.instances[0].name == "left"
    assert FakeController.instances[0].disable_calls == [7]
    assert FakeController.instances[0].disconnect_calls == [True]
    assert shutdown_calls == [True]


def test_ros_entity_construction_failure_cleans_setup_arms_and_shuts_down_rclpy(monkeypatch):
    shutdown_calls = []
    monkeypatch.setattr(adapter, "_load_piper_controller", lambda: FakeController)
    monkeypatch.setattr(adapter.rclpy, "init", lambda args=None: None)
    monkeypatch.setattr(adapter.rclpy, "shutdown", lambda: shutdown_calls.append(True))
    monkeypatch.setattr(adapter.rclpy, "spin", lambda node: None)
    monkeypatch.setattr(adapter.Node, "__init__", lambda self, *_args, **_kwargs: None)
    monkeypatch.setattr(PiperDirectSdkAdapter, "create_publisher", lambda self, *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("publisher construction failed")))

    assert adapter.main(["--left-can", "can_left", "--right-can", "can_right", "--allow-enable"]) == 1

    assert [controller.name for controller in FakeController.instances] == ["left", "right"]
    assert all(controller.disable_calls == [7] for controller in FakeController.instances)
    assert all(controller.disconnect_calls == [True] for controller in FakeController.instances)
    assert shutdown_calls == [True]
