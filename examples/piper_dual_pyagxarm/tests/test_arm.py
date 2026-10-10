from pathlib import Path
import math
import struct
import sys
import time
import uuid
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arm import Arm, DualController, load_sdk

@pytest.mark.parametrize("firmware", ["default", "v183", "v188", "v189"])
@pytest.mark.parametrize("joints", [
    [0.1, 0.2, -0.3, 0.4, -0.5, 0.6],
    [-0.065, -0.047, 0.018, 0.139, 0.423, -0.018],
])
@pytest.mark.parametrize("motion_mode", ["j", "js"])
def test_real_sdk_virtual_can_units_and_mode(firmware, joints, motion_mode):
    load_sdk()
    import can
    channel = f"pyagx_test_{uuid.uuid4().hex}"
    peer = can.Bus(interface="virtual", channel=channel)
    arm = Arm.create(channel, firmware, interface="virtual", feedback_timeout=2, motion_mode=motion_mode)
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
        assert sent[0x151][:4] == bytes([1,1,100,0xAD if motion_mode == "js" else 0])
        for i in range(3):
            assert struct.unpack(">ii", sent[0x155+i]) == tuple(raw[2*i:2*i+2])
        assert sent[0x159][:7] == struct.pack(">ihB", 40000, 1000, 1)
    finally:
        arm.close()
        peer.shutdown()


def test_dispatch_has_no_feedback_gate_or_automatic_stop():
    from unittest.mock import Mock
    arms = {side: Mock() for side in ('left', 'right')}
    for arm in arms.values():
        arm.read.side_effect = RuntimeError('no feedback')
    controller = DualController(arms, enabled=True)
    assert controller.poll() == {}
    controller.apply('left', [0.0]*7)
    arms['left'].send.assert_called_once()
    arms['right'].send.assert_not_called()
    controller.finish_zero()
    for arm in arms.values():
        arm.robot.electronic_emergency_stop.assert_not_called()


def test_adapter_preserves_gripper_speed_and_effort_for_sdk():
    from unittest.mock import Mock
    arm = Arm(Mock(), Mock(), [])
    command = arm.validate_command([0.0]*6+[-0.001], [0]*7, [0]*6+[0.1])
    arm.send(command)
    arm.robot.set_speed_percent.assert_called_once_with(0)
    arm.gripper.move_gripper_m.assert_called_once_with(value=-0.001, force=0.1)
