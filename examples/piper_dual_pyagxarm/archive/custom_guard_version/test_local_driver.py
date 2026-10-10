"""Exercise driver startup, ROS topic wiring, commands and cleanup without DDS/CAN."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arm import JOINT_NAMES, DualController
import local_driver
from test_arm import fake_arm


@pytest.mark.parametrize("enabled", [False, True])
def test_driver_topics_actions_and_cleanup(monkeypatch, enabled):
    arms = [fake_arm(), fake_arm()]
    monkeypatch.setattr(local_driver.Arm, "create", Mock(side_effect=arms))
    ticks = iter(range(10000))
    monkeypatch.setattr(local_driver.time, "monotonic", lambda: next(ticks) * 0.1)
    publishers, callbacks = {}, {}
    node = Mock()
    node.count_subscribers.side_effect = lambda topic: int(topic in callbacks)

    def publisher(kind, topic, qos):
        publishers[topic] = Mock()
        return publishers[topic]

    def subscribe(kind, topic, callback, qos):
        callbacks[topic] = callback

    node.create_publisher.side_effect = publisher
    node.create_subscription.side_effect = subscribe
    ros = Mock()

    def spin(node):
        tick = node.create_timer.call_args.args[1]
        tick()
        if enabled:
            command = NS(name=list(JOINT_NAMES), position=[0.1]*6+[0.04],
                         velocity=[0]*6+[100], effort=[0]*6+[1])
            callbacks["/joint_left_states"](command)
            callbacks["/joint_right_states"](command)
            tick()

    ros.spin.side_effect = spin
    monkeypatch.setitem(sys.modules, "rclpy", ros)
    monkeypatch.setitem(sys.modules, "rclpy.node", NS(Node=Mock(return_value=node)))
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", NS(
        JointState=lambda: NS(header=NS(stamp=None)),
    ))
    monkeypatch.setitem(sys.modules, "piper_msgs.msg", NS(PiperStatusMsg=NS))
    result = local_driver.main(["--left-firmware", "v183", "--right-firmware", "v188"]
                               + (["--enable"] if enabled else []))
    assert result == 0
    for side, arm in zip(("left", "right"), arms):
        assert f"/puppet/joint_{side}" in publishers
        assert f"/master/joint_{side}" in publishers
        assert f"/piper_{side}_ctrl_node/arm_status" in publishers
        assert publishers[f"/puppet/joint_{side}"].publish.call_args.args[0].position[-1] == 0.04
        arm.robot.disconnect.assert_called_once()
        if enabled:
            arm.robot.move_j.assert_called_once_with([0.1]*6)
            arm.robot.electronic_emergency_stop.assert_called_once()
        else:
            arm.robot.move_j.assert_not_called()
            arm.robot.enable.assert_not_called()
            arm.robot.electronic_emergency_stop.assert_not_called()


def test_firmware_is_explicit_and_same_bus_rejected():
    with pytest.raises(SystemExit):
        local_driver.parser().parse_args([])
    with pytest.raises(ValueError, match="ports must differ"):
        local_driver.main(["--left-firmware", "default", "--right-firmware", "default",
                           "--left-can-port", "can0", "--right-can-port", "can0"])


@pytest.mark.parametrize("requested", [False, True])
@pytest.mark.parametrize("motion_status", [0, 1])
@pytest.mark.parametrize("ctrl_mode", [0, 1])
def test_stop_recovery_is_explicit_and_sent_only_once(requested, motion_status, ctrl_mode):
    arms = {side: fake_arm() for side in ("left", "right")}
    controller = DualController(arms)
    for arm in arms.values():
        status = arm.robot.get_arm_status.return_value.msg
        status.arm_status, status.ctrl_mode = 1, ctrl_mode
        status.motion_status = motion_status
    sent = set()
    if requested:
        local_driver.reset_startup_stop(arms, controller, True, sent)
        local_driver.reset_startup_stop(arms, controller, True, sent)
        for arm in arms.values():
            arm.robot.reset.assert_called_once()
            arm.robot.move_j.assert_not_called()
            arm.robot.enable.assert_not_called()
        assert controller.enabled
    else:
        with pytest.raises(ValueError, match="--reset-stop"):
            local_driver.reset_startup_stop(arms, controller, False, sent)
        controller.stop("startup refused")
        for arm in arms.values():
            arm.robot.reset.assert_not_called()
            arm.robot.electronic_emergency_stop.assert_not_called()


def test_other_arm_fault_prevents_any_reset():
    arms = {side: fake_arm() for side in ("left", "right")}
    controller = DualController(arms)
    status = arms["left"].robot.get_arm_status.return_value.msg
    status.arm_status, status.ctrl_mode = 1, 0
    arms["right"].robot.get_arm_status.return_value.msg.err_code = 1
    with pytest.raises(ValueError, match="right: startup fault"):
        local_driver.reset_startup_stop(arms, controller, True, set())
    for arm in arms.values():
        arm.robot.reset.assert_not_called()


@pytest.mark.parametrize("mode,teach", [(2, 0), (1, 1)])
def test_reset_rejects_other_control_or_teaching_with_exact_status(mode, teach):
    arms = {side: fake_arm() for side in ("left", "right")}
    status = arms["right"].robot.get_arm_status.return_value.msg
    status.arm_status, status.ctrl_mode, status.teach_status = 1, mode, teach
    with pytest.raises(ValueError, match=f"ctrl_mode={mode}, teach_status={teach}"):
        local_driver.reset_startup_stop(arms, DualController(arms), True, set())
    for arm in arms.values():
        arm.robot.reset.assert_not_called()


@pytest.mark.parametrize("persistent", [False, True])
def test_startup_waits_through_delayed_recovery_without_commands(monkeypatch, persistent):
    clock = [0.0]
    monkeypatch.setattr(local_driver.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(local_driver.time, "sleep", lambda duration: clock.__setitem__(0, clock[0]+duration))
    arm = Mock()

    def read():
        if clock[0] >= 0.4 and (persistent or clock[0] < 0.7):
            raise RuntimeError("joint communication fault")

    arm.read.side_effect = read
    if persistent:
        with pytest.raises(RuntimeError, match="did not stabilize"):
            local_driver.wait_stable_feedback({"left": arm}, 3)
    else:
        local_driver.wait_stable_feedback({"left": arm}, 3)
        assert clock[0] >= 1.7
    arm.robot.enable.assert_not_called()
    arm.robot.reset.assert_not_called()
    arm.robot.move_j.assert_not_called()
