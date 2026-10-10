from pathlib import Path
import math
import struct
import sys
import time
from types import SimpleNamespace as NS
from unittest.mock import Mock
import uuid

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arm import Arm, DualController, load_sdk, error_flags, fault_details


@pytest.mark.parametrize("code", [14, 0x0E00, 0x3F3F, 0])
def test_error_bits_match_actual_sdk(code):
    load_sdk()
    from pyAgxArm.protocols.can_protocol.msgs.piper.versions.v188 import ArmMsgFeedbackStatusV188
    status = ArmMsgFeedbackStatusV188()
    status.err_code = code
    for key, value in error_flags(code).items():
        assert value == getattr(status.err_status, key)
    if code == 14:
        assert "communication_fault_joints=[2, 3, 4]" in fault_details({"err_code": code})
        assert "angle_limit_joints=[]" in fault_details({"err_code": code})


def sample(msg):
    return NS(msg=msg, timestamp=time.time())


def fake_arm():
    robot = Mock()
    robot.has_comm_error.return_value = False
    robot._parser = NS(**{pair: sample(None) for pair in ("joint_12", "joint_34", "joint_56")})
    robot.get_joint_angles.return_value = sample([0.0] * 6)
    robot.get_arm_status.return_value = sample(NS(
        ctrl_mode=1, arm_status=0, mode_feedback=1, teach_status=0,
        motion_status=0, trajectory_num=0, err_code=0,
    ))
    robot.get_driver_states.return_value = sample(NS(foc_status=NS(driver_enable_status=True)))
    gripper = Mock()
    gripper.get_gripper_status.return_value = sample(NS(value=0.04, mode="width"))
    return Arm(robot, gripper, [(-3, 3)] * 6)


def test_si_units_and_local_speed_effort_contract():
    arm = fake_arm()
    values = [0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 0.04]
    arm.send(arm.validate_command(values, [0.0] * 6 + [100], [0.0] * 6 + [1]))
    arm.robot.move_j.assert_called_once_with(values[:6])
    arm.robot.move_js.assert_not_called()
    arm.robot.set_speed_percent.assert_called_once_with(100)
    arm.gripper.move_gripper_m.assert_called_once_with(value=0.04, force=1.0)
    assert arm.read(require_enabled=True)[0][-1] == 0.04


@pytest.mark.parametrize("values", [[0]*6, [math.nan]*7, [math.inf]*7, [True]*7])
def test_bad_actions_stop_both_without_motion(values):
    arms = {s: fake_arm() for s in ("left", "right")}
    controller = DualController(arms, enabled=True)
    with pytest.raises(ValueError):
        controller.apply("left", values)
    for arm in arms.values():
        arm.robot.move_j.assert_not_called()
        arm.robot.electronic_emergency_stop.assert_called_once()
    with pytest.raises(RuntimeError):
        controller.apply("right", [0]*7)


@pytest.mark.parametrize("failure", ["missing", "stale", "partial_stale", "disabled", "fault", "mode", "comm", "gripper_mode"])
def test_fault_in_other_arm_blocks_command(failure):
    arms = {s: fake_arm() for s in ("left", "right")}
    right = arms["right"]
    if failure == "missing":
        right.robot.get_joint_angles.return_value = None
    elif failure == "stale":
        right.robot.get_driver_states.return_value.timestamp -= 1
    elif failure == "partial_stale":
        right.robot._parser.joint_12.timestamp -= 1
    elif failure == "disabled":
        right.robot.get_driver_states.return_value.msg.foc_status.driver_enable_status = False
    elif failure == "fault":
        right.robot.get_arm_status.return_value.msg.err_code = 1
    elif failure == "mode":
        right.robot.get_arm_status.return_value.msg.mode_feedback = 4
    elif failure == "comm":
        right.robot.has_comm_error.return_value = True
    else:
        right.gripper.get_gripper_status.return_value.msg.mode = "angle"
    controller = DualController(arms, enabled=True)
    with pytest.raises(RuntimeError):
        controller.apply("left", [0]*7)
    arms["left"].robot.move_j.assert_not_called()
    assert controller.fault


def test_read_only_and_watchdog():
    arms = {s: fake_arm() for s in ("left", "right")}
    controller = DualController(arms)
    controller.apply("left", [0]*7)
    controller.stop("exit")
    for arm in arms.values():
        arm.robot.move_j.assert_not_called()
        arm.robot.electronic_emergency_stop.assert_not_called()
    controller = DualController(arms, enabled=True)
    controller.last_command["left"] = time.monotonic() - 10
    with pytest.raises(RuntimeError, match="watchdog"):
        controller.poll()


@pytest.mark.parametrize("after_action", [False, True])
@pytest.mark.parametrize("mode,motion", [(0, 0), (1, 1)])
def test_standby_diagnostic_preserves_stop_without_reclaiming_control(after_action, mode, motion):
    arms = {s: fake_arm() for s in ("left", "right")}
    controller = DualController(arms, enabled=True)
    if after_action:
        controller.apply("right", [0]*7)
    status = arms["right"].robot.get_arm_status.return_value.msg
    status.ctrl_mode = 0
    status.mode_feedback = mode
    status.motion_status = motion
    if not after_action:
        assert controller.poll()["right"][1]["ctrl_mode"] == 0
        for arm in arms.values():
            arm.robot.move_j.assert_not_called()
            arm.robot.electronic_emergency_stop.assert_not_called()
            arm.robot.set_motion_mode.assert_not_called()
        return
    with pytest.raises(RuntimeError) as error:
        controller.poll()
    text = str(error.value)
    assert "right: expected CAN control / MOVE J" in text
    assert f"commands_sent={int(after_action)}" in text
    assert ("action stream started" if after_action else "waiting for first action") in text
    for arm in arms.values():
        arm.robot.electronic_emergency_stop.assert_called_once()
        arm.robot.set_motion_mode.assert_not_called()
        arm.robot.enable.assert_not_called()
        arm.robot.reset.assert_not_called()
    assert arms["left"].robot.move_j.call_count == 0
    assert arms["right"].robot.move_j.call_count == int(after_action)


@pytest.mark.parametrize("response", ["confirmed", "no_ack", "old_ack"])
def test_first_action_waits_for_new_mode_feedback(monkeypatch, response):
    arms = {s: fake_arm() for s in ("left", "right")}
    for arm in arms.values():
        status = arm.robot.get_arm_status.return_value
        status.msg.ctrl_mode = status.msg.mode_feedback = 0

        def acknowledge(mode, arm=arm, status=status):
            for peer in arms.values():
                peer.robot.move_j.assert_not_called()
                peer.gripper.move_gripper_m.assert_not_called()
            if response != "no_ack":
                status.msg.ctrl_mode = status.msg.mode_feedback = 1
            if response == "confirmed":
                status.timestamp = time.time() + 0.001

        arm.robot.set_motion_mode.side_effect = acknowledge
    ticks = iter(i * 0.1 for i in range(100))
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    controller = DualController(arms, enabled=True)
    if response == "confirmed":
        controller.apply("left", [0]*7)
        arms["left"].robot.move_j.assert_called_once()
        arms["right"].robot.move_j.assert_not_called()
    else:
        with pytest.raises(RuntimeError, match="confirmation timed out"):
            controller.apply("left", [0]*7)
        for arm in arms.values():
            arm.robot.move_j.assert_not_called()
            arm.robot.electronic_emergency_stop.assert_called_once()
    for arm in arms.values():
        arm.robot.set_motion_mode.assert_called_once_with("j")
        arm.robot.enable.assert_not_called()
        arm.robot.reset.assert_not_called()


def test_invalid_first_action_cannot_wake_standby():
    arms = {s: fake_arm() for s in ("left", "right")}
    for arm in arms.values():
        arm.robot.get_arm_status.return_value.msg.ctrl_mode = 0
        arm.robot.get_arm_status.return_value.msg.mode_feedback = 0
    controller = DualController(arms, enabled=True)
    with pytest.raises(ValueError):
        controller.apply("left", [math.nan]*7)
    for arm in arms.values():
        arm.robot.set_motion_mode.assert_not_called()
        arm.robot.move_j.assert_not_called()


@pytest.mark.parametrize("firmware", ["default", "v183", "v188", "v189"])
@pytest.mark.parametrize("joints", [
    [0.1, 0.2, -0.3, 0.4, -0.5, 0.6],
    [-0.065, -0.047, 0.018, 0.139, 0.423, -0.018],
])
def test_real_sdk_virtual_can_units_and_mode(firmware, joints):
    load_sdk()
    import can
    channel = f"pyagx_test_{uuid.uuid4().hex}"
    peer = can.Bus(interface="virtual", channel=channel)
    arm = Arm.create(channel, firmware, interface="virtual", feedback_timeout=2)
    try:
        assert arm.robot.get_joint_limits_enabled() is False
        raw = [round(math.degrees(v)*1000) for v in joints]
        frames = [(0x2A1, bytes([1,0,1,0,0,0,0,0])),
                  (0x2A8, struct.pack(">ihBB", 40000, 1000, 0x40, 0))]
        frames += [(0x2A5+i, struct.pack(">ii", *raw[2*i:2*i+2])) for i in range(3)]
        frames += [(0x261+i, bytes([0,0,0,0,0,0x40,0,0])) for i in range(6)]
        for aid, data in frames:
            peer.send(can.Message(arbitration_id=aid, data=data, is_extended_id=False))
        deadline = time.monotonic() + 1
        while True:
            try:
                state, status = arm.read(require_enabled=True)
                break
            except RuntimeError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.005)
        assert state == pytest.approx(joints+[0.04], abs=2e-5)
        arm.send(arm.validate_command(joints+[0.04], [0]*6+[100], [0]*6+[1]))
        sent = {}
        deadline = time.monotonic()+1
        while time.monotonic() < deadline:
            msg = peer.recv(timeout=0.05)
            if msg is not None:
                sent[msg.arbitration_id] = bytes(msg.data)
            if all(aid in sent for aid in (0x151, 0x155, 0x156, 0x157, 0x159)):
                break
        assert sent[0x151][:4] == bytes([1,1,100,0])  # MOVE J, position-speed, not MIT.
        for i in range(3):
            assert struct.unpack(">ii", sent[0x155+i]) == tuple(raw[2*i:2*i+2])
        assert sent[0x159][:7] == struct.pack(">ihB", 40000, 1000, 1)
    finally:
        arm.close()
        peer.shutdown()


def test_constant_speed_is_not_retransmitted_for_every_target():
    arm = fake_arm()
    for _ in range(8):
        arm.send(arm.validate_command([0.0]*7))
    arm.robot.set_speed_percent.assert_called_once_with(30)
    assert arm.robot.move_j.call_count == 8
    arm.send(arm.validate_command([0.0]*7, [0]*6+[40]))
    assert arm.robot.set_speed_percent.call_count == 2


@pytest.mark.parametrize('age,accepted', [(None, False), (0.05, True), (0.11, False)])
def test_speed_only_mode_feedback_has_bounded_local_grace(monkeypatch, age, accepted):
    arm = fake_arm()
    monkeypatch.setattr(time, 'monotonic', lambda: 10.0)
    arm._speed_update_at = None if age is None else 10.0-age
    arm.robot.get_arm_status.return_value.msg.mode_feedback = 255
    if accepted:
        arm.read(require_enabled=True)
        arm.robot.get_arm_status.return_value.msg.err_code = 1
        with pytest.raises(RuntimeError, match='arm fault'):
            arm.read(require_enabled=True)
    else:
        with pytest.raises(RuntimeError, match='expected CAN control'):
            arm.read(require_enabled=True)


def test_completed_zero_stays_idle_and_next_task_rearms_watchdog():
    arms = {side: fake_arm() for side in ('left', 'right')}
    controller = DualController(arms, enabled=True)
    for side, arm in arms.items():
        arm.gripper.get_gripper_status.return_value.msg.value = 0.0
        controller.apply(side, [0.0]*7)
    controller.finish_zero()
    assert controller.last_command == {}
    controller.poll()
    for arm in arms.values():
        arm.robot.electronic_emergency_stop.assert_not_called()
    controller.apply('left', [0.0]*7)
    assert set(controller.last_command) == {'left', 'right'}
    controller.last_command['right'] -= 10
    with pytest.raises(RuntimeError, match='watchdog'):
        controller.poll()


@pytest.mark.parametrize('bad_target', [False, True])
def test_zero_completion_cannot_hide_unfinished_motion(bad_target):
    arms = {side: fake_arm() for side in ('left', 'right')}
    controller = DualController(arms, enabled=True)
    for side, arm in arms.items():
        arm.gripper.get_gripper_status.return_value.msg.value = 0.0
        controller.apply(side, [0.0]*7)
    if bad_target:
        arms['right'].command[0] = 0.1
    else:
        arms['right'].robot.get_joint_angles.return_value.msg[0] = 0.1
    with pytest.raises(RuntimeError):
        controller.finish_zero()
    assert controller.last_command


@pytest.mark.parametrize('width', [-0.001, 0.081])
def test_gripper_width_uses_official_sdk_unless_limit_requested(width):
    arm = fake_arm()
    arm.send(arm.validate_command([0]*6+[width]))
    arm.gripper.move_gripper_m.assert_called_once_with(value=width, force=1.0)
    arm.gripper_max = 0.08
    with pytest.raises(ValueError, match='explicitly configured'):
        arm.validate_command([0]*6+[width])
