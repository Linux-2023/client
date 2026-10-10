from pathlib import Path
import sys
import time
import math
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arm import Arm
from eef_feedback import pose_message


def test_flange_feedback_units_quaternion_and_freshness():
    robot = Mock()
    arm = Arm(robot, Mock(), [])
    sample = NS(timestamp=time.time(), msg=[0.12, -0.03, 0.25, 0.0, 0.0, math.pi/2])
    robot._parser = NS(**{key: sample for key in ('end_pose_xy','end_pose_zrx','end_pose_ryrz')})
    robot.get_flange_pose.return_value = sample
    from builtin_interfaces.msg import Time
    msg = pose_message(arm, Time(sec=42))
    assert msg.pose.position.x == 0.12
    assert msg.pose.position.y == -0.03
    assert msg.pose.orientation.z == pytest.approx(math.sqrt(0.5))
    assert msg.pose.orientation.w == pytest.approx(math.sqrt(0.5))
    assert msg.header.stamp.sec == 42
    sample.timestamp -= 1
    with pytest.raises(RuntimeError, match='stale'):
        pose_message(arm, Time())
    robot.move_j.assert_not_called()
    robot.enable.assert_not_called()
