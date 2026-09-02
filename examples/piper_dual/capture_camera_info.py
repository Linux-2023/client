#!/usr/bin/env python3
"""Capture one live ROS 2 CameraInfo message for each Piper dual camera."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping


CAMERA_TOPICS = {
    "cam_high": "/camera_f_l/color/camera_info",
    "cam_left_wrist": "/camera_l/color/camera_info",
    "cam_right_wrist": "/camera_r/color/camera_info",
}
CAMERA_PARAMETERS_SCHEMA_VERSION = "piper_dual_camera_parameters_v1"


def camera_info_to_dict(message: Any) -> dict[str, Any]:
    """Convert sensor_msgs/CameraInfo-compatible data into plain JSON values."""

    stamp = message.header.stamp
    timestamp = float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0
    payload = {
        "timestamp": timestamp,
        "frame_id": str(message.header.frame_id),
        "width": int(message.width),
        "height": int(message.height),
        "distortion_model": str(message.distortion_model),
        "d": _finite_float_list(message.d, "d"),
        "k": _fixed_float_list(message.k, 9, "k"),
        "r": _fixed_float_list(message.r, 9, "r"),
        "p": _fixed_float_list(message.p, 12, "p"),
        "binning_x": int(message.binning_x),
        "binning_y": int(message.binning_y),
        "roi": {
            "x_offset": int(message.roi.x_offset),
            "y_offset": int(message.roi.y_offset),
            "height": int(message.roi.height),
            "width": int(message.roi.width),
            "do_rectify": bool(message.roi.do_rectify),
        },
    }
    if not math.isfinite(timestamp):
        raise ValueError("CameraInfo timestamp must be finite")
    if payload["width"] <= 0 or payload["height"] <= 0:
        raise ValueError("CameraInfo width and height must be positive")
    return payload


def build_camera_snapshot(
    messages: Mapping[str, Any],
    *,
    captured_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build a complete three-camera live calibration snapshot."""

    missing = [camera for camera in CAMERA_TOPICS if camera not in messages]
    if missing:
        raise ValueError(f"missing CameraInfo messages: {', '.join(missing)}")
    captured_at = captured_at_utc or datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "schema_version": CAMERA_PARAMETERS_SCHEMA_VERSION,
        "captured_at_utc": captured_at,
        "source": "live_ros2_camera_info",
        "source_note": (
            "Calibration was read from the current online camera nodes at export time; "
            "it was not stored in, and cannot prove the calibration used by, the historical HDF5 episode."
        ),
        "historical_episode_calibration": False,
        "matrix_conventions": {
            "k": "3x3 row-major intrinsic camera matrix",
            "r": "3x3 row-major rectification matrix",
            "p": "3x4 row-major projection matrix",
            "d": "distortion coefficients interpreted using distortion_model",
        },
        "cross_camera_extrinsics": None,
        "cross_camera_extrinsics_note": (
            "CameraInfo does not contain transforms between the three physical cameras. "
            "Cross-camera extrinsics require a separate multi-camera calibration or a verified shared TF tree."
        ),
        "cameras": {
            camera: {"topic": topic, "camera_info": camera_info_to_dict(messages[camera])}
            for camera, topic in CAMERA_TOPICS.items()
        },
    }


def capture_live_camera_info(timeout_seconds: float) -> dict[str, Any]:
    """Subscribe until one CameraInfo message has arrived for every configured camera."""

    if isinstance(timeout_seconds, bool) or not math.isfinite(float(timeout_seconds)) or float(timeout_seconds) <= 0:
        raise ValueError("timeout must be a positive finite number")
    try:
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo
    except ImportError as exc:
        raise RuntimeError("ROS 2 Python packages rclpy and sensor_msgs are required") from exc

    rclpy.init(args=None)
    node = rclpy.create_node("piper_dual_camera_info_snapshot")
    messages: dict[str, Any] = {}
    subscriptions = []
    try:
        for camera, topic in CAMERA_TOPICS.items():
            def callback(message: Any, *, camera_name: str = camera) -> None:
                messages.setdefault(camera_name, message)

            subscriptions.append(node.create_subscription(CameraInfo, topic, callback, qos_profile_sensor_data))
        deadline = time.monotonic() + float(timeout_seconds)
        while len(messages) < len(CAMERA_TOPICS):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                missing = [camera for camera in CAMERA_TOPICS if camera not in messages]
                raise TimeoutError(
                    f"timed out waiting for CameraInfo after {timeout_seconds:g}s: {', '.join(missing)}"
                )
            rclpy.spin_once(node, timeout_sec=min(0.2, remaining))
        return build_camera_snapshot(messages)
    finally:
        subscriptions.clear()
        node.destroy_node()
        rclpy.shutdown()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture live ROS 2 CameraInfo for three Piper dual cameras")
    parser.add_argument("--output", required=True, type=Path, help="Output camera_parameters.json path")
    parser.add_argument("--timeout", type=float, default=10.0, help="Seconds to wait for all CameraInfo topics (default: 10)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        payload = capture_live_camera_info(args.timeout)
        _write_json(args.output, payload)
    except (RuntimeError, TimeoutError, ValueError) as exc:
        print(str(exc), file=__import__("sys").stderr)
        return 1
    print(f"Captured live CameraInfo to {args.output}")
    return 0


def _fixed_float_list(values: Any, expected_length: int, field: str) -> list[float]:
    result = _finite_float_list(values, field)
    if len(result) != expected_length:
        raise ValueError(f"CameraInfo {field} must contain exactly {expected_length} values")
    return result


def _finite_float_list(values: Any, field: str) -> list[float]:
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"CameraInfo {field} values must be finite")
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
