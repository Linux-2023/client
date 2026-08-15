#!/usr/bin/env python3
"""ROS 2 sidecar process for the Piper dual bridge JSON-line protocol."""

from __future__ import annotations

import argparse
import base64
import json
import math
import pathlib
import queue
import sys
import threading
from typing import Any

import cv_bridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_msgs.msg import String

try:
    from examples.piper_dual.ros2_protocol import decode_message
    from examples.piper_dual.ros2_protocol import encode_message
except ModuleNotFoundError:
    from ros2_protocol import decode_message
    from ros2_protocol import encode_message


IMAGE_TOPICS: dict[str, str] = {
    "/camera_f_l/color/image_raw": "cam_high",
    "/camera_l/color/image_raw": "cam_left_wrist",
    "/camera_r/color/image_raw": "cam_right_wrist",
}
JOINT_TOPICS: dict[str, str] = {
    "/puppet/joint_left": "puppet_left",
    "/puppet/joint_right": "puppet_right",
    "/master/joint_left": "master_left",
    "/master/joint_right": "master_right",
}
LEFT_ACTION_TOPIC = "/joint_left_states"
RIGHT_ACTION_TOPIC = "/joint_right_states"
JOINT_NAMES = [f"joint{i}" for i in range(7)]

JsonObject = dict[str, Any]


def _format_sensor_timestamp(stamp: Any) -> float:
    """Return a ROS builtin_interfaces/Time stamp as seconds."""
    return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0


def build_image_sensor_event(sensor: str, message: Any, jpeg_bytes: bytes) -> JsonObject:
    """Build one image sensor event with bridge-side base64 encoding."""
    return {
        "type": "sensor",
        "sensor": sensor,
        "timestamp": _format_sensor_timestamp(message.header.stamp),
        "jpeg_b64": base64.b64encode(jpeg_bytes).decode("ascii"),
    }


def build_joint_sensor_event(sensor: str, message: Any) -> JsonObject:
    """Build one joint stream sensor event from a JointState position vector."""
    return {
        "type": "sensor",
        "sensor": sensor,
        "timestamp": _format_sensor_timestamp(message.header.stamp),
        "values": [float(value) for value in message.position],
    }


def validate_action_request(request: JsonObject) -> dict[str, list[float]]:
    """Validate and normalize a publish_action request for both arms."""
    vectors: dict[str, list[float]] = {}
    for arm in ("left", "right"):
        raw_values = request.get(arm)
        if not isinstance(raw_values, list) or len(raw_values) != 7:
            raise ValueError(f"publish_action.{arm} must contain exactly seven values")

        values: list[float] = []
        for raw_value in raw_values:
            if isinstance(raw_value, bool):
                raise ValueError(f"publish_action.{arm} values must be finite numbers")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"publish_action.{arm} values must be finite numbers") from exc
            if not math.isfinite(value):
                raise ValueError(f"publish_action.{arm} values must be finite numbers")
            values.append(value)
        vectors[arm] = values
    return vectors


def _load_config(path: str | None) -> JsonObject:
    if path is None:
        return {}
    with pathlib.Path(path).open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("Bridge config must be a JSON object")
    return data


class LockedJsonLineWriter:
    """Thread-safe protocol writer for stdout JSON lines."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._lock = threading.Lock()

    def write(self, message: JsonObject) -> None:
        line = encode_message(message)
        with self._lock:
            self._stream.write(line)
            self._stream.flush()

    def status(self, state: str, **metadata: Any) -> None:
        message: JsonObject = {"type": "status", "state": state}
        if metadata:
            message["metadata"] = metadata
        self.write(message)

    def error(self, text: str, **metadata: Any) -> None:
        message: JsonObject = {"type": "error", "message": text}
        if metadata:
            message["metadata"] = metadata
        self.write(message)


class PiperRos2Bridge(Node):
    """ROS 2 node that streams individual sensor events and applies action requests."""

    def __init__(self, *, writer: LockedJsonLineWriter, dry_run: bool, publish_actions: bool, config: JsonObject) -> None:
        super().__init__("piper_dual_ros2_bridge")
        self._writer = writer
        self._dry_run = dry_run
        self._publish_actions = publish_actions and not dry_run
        self._bridge = cv_bridge.CvBridge()
        self._requests: queue.Queue[JsonObject | None] = queue.Queue(maxsize=64)
        self._stop_event = threading.Event()
        self._mode = str(config.get("mode", "default"))

        image_topics = _merge_topics(IMAGE_TOPICS, config.get("image_topics"))
        joint_topics = _merge_topics(JOINT_TOPICS, config.get("joint_topics"))

        qos_depth = int(config.get("qos_depth", 1))
        for topic, sensor in image_topics.items():
            self.create_subscription(Image, topic, self._make_image_callback(sensor), qos_depth)
        for topic, sensor in joint_topics.items():
            self.create_subscription(JointState, topic, self._make_joint_callback(sensor), qos_depth)

        self._left_publisher = None
        self._right_publisher = None
        if self._publish_actions:
            self._left_publisher = self.create_publisher(JointState, LEFT_ACTION_TOPIC, qos_depth)
            self._right_publisher = self.create_publisher(JointState, RIGHT_ACTION_TOPIC, qos_depth)

        self._request_timer = self.create_timer(0.01, self._drain_requests)
        self._writer.status(
            "ready",
            dry_run=self._dry_run,
            publish_actions=self._publish_actions,
            mode=self._mode,
        )

    def enqueue_request(self, request: JsonObject | None) -> None:
        self._requests.put(request)

    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def _make_image_callback(self, sensor: str):
        def callback(message: Image) -> None:
            try:
                jpeg = self._bridge.cv2_to_compressed_imgmsg(
                    self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8"), dst_format="jpg"
                )
                self._writer.write(build_image_sensor_event(sensor, message, bytes(jpeg.data)))
            except Exception as exc:  # noqa: BLE001 - ROS callback must survive bad frames.
                self._writer.error(str(exc), sensor=sensor)

        return callback

    def _make_joint_callback(self, sensor: str):
        def callback(message: JointState) -> None:
            try:
                self._writer.write(build_joint_sensor_event(sensor, message))
            except Exception as exc:  # noqa: BLE001 - ROS callback must survive bad frames.
                self._writer.error(str(exc), sensor=sensor)

        return callback

    def _drain_requests(self) -> None:
        while True:
            try:
                request = self._requests.get_nowait()
            except queue.Empty:
                return
            if request is None:
                self._stop_event.set()
                return
            try:
                self._handle_request(request)
            except Exception as exc:  # noqa: BLE001 - report malformed requests without killing the bridge.
                self._writer.error(str(exc), request_type=request.get("type"))

    def _handle_request(self, request: JsonObject) -> None:
        request_type = request.get("type")
        if request_type == "publish_action":
            vectors = validate_action_request(request)
            if self._publish_actions:
                self._publish_action(vectors["left"], vectors["right"])
                self._writer.status("action_published")
            else:
                self._writer.status("action_ignored", dry_run=self._dry_run, publish_actions=self._publish_actions)
        elif request_type == "set_mode":
            mode = request.get("mode")
            if not isinstance(mode, str) or not mode:
                raise ValueError("set_mode.mode must be a non-empty string")
            self._mode = mode
            self._writer.status("mode_set", mode=self._mode)
        elif request_type == "ping":
            self._writer.status("pong")
        elif request_type == "stop":
            self._writer.write({"type": "stopped"})
            self._stop_event.set()
        else:
            raise ValueError(f"Unsupported request type: {request_type!r}")

    def _publish_action(self, left: list[float], right: list[float]) -> None:
        if self._left_publisher is None or self._right_publisher is None:
            raise RuntimeError("Action publishers are disabled")
        stamp = self.get_clock().now().to_msg()
        self._left_publisher.publish(_make_joint_state(left, stamp))
        self._right_publisher.publish(_make_joint_state(right, stamp))


def _merge_topics(defaults: dict[str, str], overrides: Any) -> dict[str, str]:
    topics = dict(defaults)
    if overrides is None:
        return topics
    if not isinstance(overrides, dict):
        raise ValueError("Configured topics must be a JSON object")
    for topic, sensor in overrides.items():
        if not isinstance(topic, str) or not isinstance(sensor, str):
            raise ValueError("Configured topic and sensor names must be strings")
        topics[topic] = sensor
    return topics


def _make_joint_state(values: list[float], stamp: Any) -> JointState:
    message = JointState()
    message.header.stamp = stamp
    message.name = list(JOINT_NAMES)
    message.position = list(values)
    return message


def _stdin_reader(node: PiperRos2Bridge, writer: LockedJsonLineWriter) -> None:
    for line in sys.stdin:
        try:
            node.enqueue_request(decode_message(line))
        except ValueError as exc:
            writer.error(str(exc))
    node.enqueue_request(None)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Piper dual ROS 2 bridge process")
    parser.add_argument("--config", help="Path to optional JSON bridge config")
    parser.add_argument("--dry-run", action="store_true", help="Never publish joint action messages")
    parser.add_argument("--publish-actions", action="store_true", help="Enable publishing validated joint actions")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    writer = LockedJsonLineWriter(sys.stdout)
    try:
        config = _load_config(args.config)
    except Exception as exc:  # noqa: BLE001 - config errors must be emitted as protocol errors.
        writer.error(str(exc))
        return 2

    rclpy.init(args=None)
    node = PiperRos2Bridge(
        writer=writer,
        dry_run=args.dry_run,
        publish_actions=args.publish_actions,
        config=config,
    )
    reader = threading.Thread(target=_stdin_reader, args=(node, writer), daemon=True)
    reader.start()

    try:
        while rclpy.ok() and not node.stop_requested():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        writer.write({"type": "stopped"})
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
