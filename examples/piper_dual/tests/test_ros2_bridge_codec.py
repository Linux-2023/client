"""Focused tests for bridge-side ROS message/request codecs."""

from pathlib import Path
import base64
import math
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ros2_bridge_process import _format_sensor_timestamp
from ros2_bridge_process import build_image_sensor_event
from ros2_bridge_process import build_joint_sensor_event
from ros2_bridge_process import validate_action_request


class Stamp:
    def __init__(self, sec, nanosec):
        self.sec = sec
        self.nanosec = nanosec


class Header:
    def __init__(self, sec, nanosec):
        self.stamp = Stamp(sec, nanosec)


class Message:
    def __init__(self, *, sec=0, nanosec=0, position=None):
        self.header = Header(sec, nanosec)
        self.position = position or []


def test_format_sensor_timestamp_preserves_ros_header_stamp_as_float_seconds():
    assert _format_sensor_timestamp(Stamp(123, 456_789_012)) == pytest.approx(123.456789012)


def test_build_image_sensor_event_base64_encodes_jpeg_bytes_at_bridge_boundary():
    jpeg_bytes = b"\xff\xd8binary-jpeg\xff\xd9"
    event = build_image_sensor_event("cam_left_wrist", Message(sec=7, nanosec=50), jpeg_bytes)

    assert event == {
        "type": "sensor",
        "sensor": "cam_left_wrist",
        "timestamp": 7.00000005,
        "jpeg_b64": base64.b64encode(jpeg_bytes).decode("ascii"),
    }


def test_build_joint_sensor_event_emits_float_values_without_extra_payload_fields():
    event = build_joint_sensor_event("puppet_left", Message(sec=1, nanosec=250_000_000, position=[1, 2.5, -3]))

    assert event == {
        "type": "sensor",
        "sensor": "puppet_left",
        "timestamp": 1.25,
        "values": [1.0, 2.5, -3.0],
    }
    assert "jpeg_b64" not in event


def test_validate_action_request_accepts_two_seven_value_finite_arm_vectors():
    request = {
        "type": "publish_action",
        "left": [0, 1, 2, 3, 4, 5, 6],
        "right": [6, 5, 4, 3, 2, 1, 0],
    }

    assert validate_action_request(request) == {
        "left": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "right": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
    }


@pytest.mark.parametrize(
    "request_data",
    [
        {"type": "publish_action", "left": [0] * 6, "right": [0] * 7},
        {"type": "publish_action", "left": [0] * 7, "right": [0] * 8},
        {"type": "publish_action", "left": [0] * 7, "right": [0, 1, 2, math.inf, 4, 5, 6]},
        {"type": "publish_action", "left": [0] * 7, "right": [0, 1, 2, math.nan, 4, 5, 6]},
        {"type": "publish_action", "left": [0] * 7},
    ],
)
def test_validate_action_request_rejects_vectors_that_are_not_seven_finite_values_per_arm(request_data):
    with pytest.raises(ValueError):
        validate_action_request(request_data)
