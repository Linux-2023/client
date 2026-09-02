"""Tests for capturing three live ROS 2 CameraInfo messages."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture_camera_info import CAMERA_TOPICS
from capture_camera_info import build_camera_snapshot
from capture_camera_info import camera_info_to_dict


def _message(frame_id: str = "camera_color_optical_frame") -> SimpleNamespace:
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=123, nanosec=456_000_000), frame_id=frame_id),
        height=720,
        width=1280,
        distortion_model="plumb_bob",
        d=[0.1, 0.2, 0.3, 0.4, 0.5],
        k=[600.0, 0.0, 640.0, 0.0, 601.0, 360.0, 0.0, 0.0, 1.0],
        r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        p=[600.0, 0.0, 640.0, 0.0, 0.0, 601.0, 360.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        binning_x=0,
        binning_y=0,
        roi=SimpleNamespace(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False),
    )


def test_camera_info_to_dict_preserves_calibration_fields() -> None:
    payload = camera_info_to_dict(_message())

    assert payload == {
        "timestamp": 123.456,
        "frame_id": "camera_color_optical_frame",
        "width": 1280,
        "height": 720,
        "distortion_model": "plumb_bob",
        "d": [0.1, 0.2, 0.3, 0.4, 0.5],
        "k": [600.0, 0.0, 640.0, 0.0, 601.0, 360.0, 0.0, 0.0, 1.0],
        "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "p": [600.0, 0.0, 640.0, 0.0, 0.0, 601.0, 360.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "binning_x": 0,
        "binning_y": 0,
        "roi": {"x_offset": 0, "y_offset": 0, "height": 0, "width": 0, "do_rectify": False},
    }


def test_build_camera_snapshot_requires_all_three_cameras() -> None:
    incomplete = {"cam_high": _message("high")}

    with pytest.raises(ValueError, match="cam_left_wrist.*cam_right_wrist"):
        build_camera_snapshot(incomplete, captured_at_utc="2026-08-22T00:00:00+00:00")


def test_build_camera_snapshot_marks_live_values_as_current_not_historical() -> None:
    messages = {camera: _message(camera) for camera in CAMERA_TOPICS}

    payload = build_camera_snapshot(messages, captured_at_utc="2026-08-22T00:00:00+00:00")

    assert payload["schema_version"] == "piper_dual_camera_parameters_v1"
    assert payload["captured_at_utc"] == "2026-08-22T00:00:00+00:00"
    assert payload["historical_episode_calibration"] is False
    assert "current online" in payload["source_note"]
    assert set(payload["cameras"]) == set(CAMERA_TOPICS)
    for camera, topic in CAMERA_TOPICS.items():
        assert payload["cameras"][camera]["topic"] == topic
        assert payload["cameras"][camera]["camera_info"]["frame_id"] == camera
