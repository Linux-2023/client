"""Focused tests for bridge-side ROS message/request codecs."""

from pathlib import Path
import json
import base64
import math
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ros2_bridge_process import _load_config
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


def test_load_config_accepts_shipped_default_yaml_bridge_contract():
    config_path = Path(__file__).resolve().parents[1] / "ros2_piper_dual.yaml"

    config = _load_config(str(config_path))

    assert config["schema_version"] == "piper_dual_ros2_v1"
    assert config["max_sync_error_ms"] == 30
    assert config["jpeg_quality"] == 90
    assert config["model_image"] == {
        "width": 224,
        "height": 224,
        "channel_order": "RGB",
        "layout": "CHW",
        "dtype": "uint8",
    }
    assert [camera["name"] for camera in config["cameras"]] == [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]
    assert [camera["topic"] for camera in config["cameras"]] == [
        "/camera_f_l/color/image_raw",
        "/camera_l/color/image_raw",
        "/camera_r/color/image_raw",
    ]


def test_load_config_preserves_json_object_support(tmp_path):
    config_path = tmp_path / "bridge-config.json"
    config_path.write_text(json.dumps({"mode": "test", "qos_depth": 2}), encoding="utf-8")

    assert _load_config(str(config_path)) == {"mode": "test", "qos_depth": 2}


def test_load_config_rejects_non_mapping_yaml_root(tmp_path):
    config_path = tmp_path / "bridge-config.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        _load_config(str(config_path))


def test_load_config_uses_safe_yaml_loader(tmp_path):
    marker_path = tmp_path / "unsafe-loader-executed"
    config_path = tmp_path / "bridge-config.yaml"
    config_path.write_text(
        f"!!python/object/apply:os.system ['touch {marker_path}']\n",
        encoding="utf-8",
    )

    with pytest.raises(Exception):
        _load_config(str(config_path))
    assert not marker_path.exists()
