"""Exercise driver startup, ROS topic wiring, commands and cleanup without DDS/CAN."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arm import JOINT_NAMES, DualController
import local_driver
from unittest.mock import Mock

def fake_arm():
    arm = Mock()
    arm.read.return_value = ([0.0]*6+[0.04], dict(ctrl_mode=0, arm_status=0, mode_feedback=255, teach_status=0, motion_status=1, trajectory_num=0, err_code=0))
    arm.command = None
    arm.close.side_effect = arm.robot.disconnect
    arm.validate_command.side_effect = lambda p, v, e: (p, v, e)
    arm.send.side_effect = lambda command: arm.robot.move_j(command[0][:6])
    return arm


@pytest.mark.parametrize("enabled", [False, True])
def test_driver_topics_actions_and_cleanup(monkeypatch, enabled):
    arms = [fake_arm(), fake_arm()]
    monkeypatch.setattr(local_driver, "enable_arms", Mock())
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
            arm.robot.electronic_emergency_stop.assert_not_called()
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




@pytest.mark.parametrize('stopped', [False, True])
def test_startup_waits_for_reset_and_fresh_motor_feedback(monkeypatch, stopped):
    clock = [10.0]
    monkeypatch.setattr(local_driver.time, 'time', lambda: clock[0])
    monkeypatch.setattr(local_driver.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(local_driver.time, 'sleep', lambda dt: clock.__setitem__(0, clock[0]+dt))
    arm = Mock()
    enabled_at = [None]
    def status():
        return NS(timestamp=clock[0], msg=NS(arm_status=1 if stopped and clock[0]<10.5 else 0, err_code=0))
    def driver(i):
        is_enabled = not stopped or clock[0]<10.5
        if enabled_at[0] is not None and clock[0]>enabled_at[0]:
            is_enabled = True
        return NS(timestamp=clock[0], msg=NS(foc_status=NS(driver_enable_status=is_enabled)))
    def enable():
        if stopped:
            assert clock[0]>=10.5
        if enabled_at[0] is None:
            enabled_at[0] = clock[0]
        return True  # A cached True alone must not end the wait.
    arm.robot.get_arm_status.side_effect = status
    arm.robot.get_driver_states.side_effect = driver
    arm.robot.enable.side_effect = enable
    local_driver.enable_arms({'left': arm}, True, 2)
    assert arm.robot.reset.call_count == int(stopped)
    assert clock[0]>enabled_at[0]
    arm.robot.move_j.assert_not_called()


def test_eef_takes_over_after_external_publisher_disappears():
    node = Mock()
    publishers = {}
    node.count_publishers.return_value = 1
    local_driver.ensure_eef_publishers(node, {'left': object()}, publishers, object)
    assert not publishers
    node.create_publisher.assert_not_called()
    node.count_publishers.return_value = 0
    local_driver.ensure_eef_publishers(node, {'left': object()}, publishers, object)
    local_driver.ensure_eef_publishers(node, {'left': object()}, publishers, object)
    node.create_publisher.assert_called_once_with(object, '/puppet/end_pose_left', 1)
