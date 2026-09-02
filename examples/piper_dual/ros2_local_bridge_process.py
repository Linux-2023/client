#!/usr/bin/python3
"""Local Piper ROS 2 bridge preserving the pre-three-stack control behavior."""

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
from geometry_msgs.msg import PoseStamped
from piper_msgs.msg import PosCmd
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
import yaml

try:
    from examples.piper_dual.ros2_protocol import decode_message, encode_message
except ModuleNotFoundError:
    from ros2_protocol import decode_message, encode_message

JsonObject = dict[str, Any]


def _stamp_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0


def _image_event(sensor: str, message: Any, jpeg: bytes) -> JsonObject:
    return {
        "type": "sensor",
        "sensor": sensor,
        "timestamp": _stamp_seconds(message.header.stamp),
        "jpeg_b64": base64.b64encode(jpeg).decode("ascii"),
    }


def _joint_event(sensor: str, message: Any) -> JsonObject:
    return {
        "type": "sensor",
        "sensor": sensor,
        "timestamp": _stamp_seconds(message.header.stamp),
        "values": [float(value) for value in message.position],
    }


def _quaternion_to_rpy(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    values = (float(x), float(y), float(z), float(w))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("EEF quaternion values must be finite")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("EEF quaternion must have non-zero finite norm")
    x, y, z, w = (value / norm for value in values)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _eef_event(sensor: str, message: Any) -> JsonObject:
    position = message.pose.position
    orientation = message.pose.orientation
    values = [
        float(position.x),
        float(position.y),
        float(position.z),
        *_quaternion_to_rpy(orientation.x, orientation.y, orientation.z, orientation.w),
    ]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("EEF pose values must be finite")
    return {"type": "sensor", "sensor": sensor, "timestamp": _stamp_seconds(message.header.stamp), "values": values}


def _validate_vectors(request: JsonObject, request_type: str) -> dict[str, list[float]]:
    vectors: dict[str, list[float]] = {}
    for arm in ("left", "right"):
        raw = request.get(arm)
        if not isinstance(raw, list) or len(raw) != 7:
            raise ValueError(f"{request_type}.{arm} must contain exactly seven values")
        values: list[float] = []
        for item in raw:
            if isinstance(item, bool):
                raise ValueError(f"{request_type}.{arm} values must be finite numbers")
            try:
                value = float(item)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{request_type}.{arm} values must be finite numbers") from exc
            if not math.isfinite(value):
                raise ValueError(f"{request_type}.{arm} values must be finite numbers")
            values.append(value)
        vectors[arm] = values
    return vectors


class Writer:
    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._lock = threading.Lock()

    def write(self, message: JsonObject) -> None:
        with self._lock:
            self._stream.write(encode_message(message))
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


class LocalPiperBridge(Node):
    def __init__(
        self,
        *,
        writer: Writer,
        config: JsonObject,
        dry_run: bool,
        publish_actions: bool,
        include_eef: bool,
        eef_control: bool,
        eef_left_topic: str | None,
        eef_right_topic: str | None,
        eef_left_action_topic: str,
        eef_right_action_topic: str,
    ) -> None:
        super().__init__("piper_dual_local_bridge")
        self._writer = writer
        self._dry_run = bool(dry_run)
        self._publish_actions = bool(publish_actions) and not self._dry_run
        self._eef_control = bool(eef_control)
        self._requests: queue.Queue[JsonObject | None] = queue.Queue(maxsize=64)
        self._stop = threading.Event()
        self._bridge = cv_bridge.CvBridge()
        self._action_publishers: dict[str, Any] = {}
        qos = int(config.get("qos_depth", 1))
        contract = config["bridge_contract"]
        cameras = {entry["name"]: entry["topic"] for entry in config["cameras"]}
        for sensor, topic in cameras.items():
            self.create_subscription(Image, topic, self._image_callback(sensor), qos)
        for topic, sensor in contract["joint_topics"].items():
            self.create_subscription(JointState, topic, self._joint_callback(sensor), qos)
        if include_eef or eef_control:
            topics = {
                "eef_puppet_left": eef_left_topic or contract["eef_topics"]["left"],
                "eef_puppet_right": eef_right_topic or contract["eef_topics"]["right"],
            }
            for sensor, topic in topics.items():
                self.create_subscription(PoseStamped, topic, self._eef_callback(sensor), qos)
        if self._publish_actions:
            if self._eef_control:
                topics = {
                    "left": eef_left_action_topic,
                    "right": eef_right_action_topic,
                }
                msg_type = PosCmd
            else:
                topics = dict(contract["joint_action_topics"])
                msg_type = JointState
            for arm, topic in topics.items():
                self._action_publishers[arm] = self.create_publisher(msg_type, topic, qos)
        self._timer = self.create_timer(0.01, self._drain)
        self._writer.status(
            "ready",
            dry_run=self._dry_run,
            publish_actions=self._publish_actions,
            eef_control=self._eef_control,
            mode="local-legacy",
        )

    def enqueue(self, request: JsonObject | None) -> None:
        self._requests.put(request)

    def stopped(self) -> bool:
        return self._stop.is_set()

    def _image_callback(self, sensor: str):
        def callback(message: Image) -> None:
            try:
                frame = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
                jpeg = self._bridge.cv2_to_compressed_imgmsg(frame, dst_format="jpg")
                self._writer.write(_image_event(sensor, message, bytes(jpeg.data)))
            except Exception as exc:
                self._writer.error(str(exc), sensor=sensor)
        return callback

    def _joint_callback(self, sensor: str):
        def callback(message: JointState) -> None:
            try:
                self._writer.write(_joint_event(sensor, message))
            except Exception as exc:
                self._writer.error(str(exc), sensor=sensor)
        return callback

    def _eef_callback(self, sensor: str):
        def callback(message: PoseStamped) -> None:
            try:
                self._writer.write(_eef_event(sensor, message))
            except Exception as exc:
                self._writer.error(str(exc), sensor=sensor)
        return callback

    def _drain(self) -> None:
        while True:
            try:
                request = self._requests.get_nowait()
            except queue.Empty:
                return
            if request is None:
                self._stop.set()
                return
            try:
                self._handle(request)
            except Exception as exc:
                self._writer.error(str(exc), request_type=request.get("type"))

    def _handle(self, request: JsonObject) -> None:
        request_type = request.get("type")
        if request_type == "publish_action":
            vectors = _validate_vectors(request, "publish_action")
            if self._publish_actions:
                self._publish_joint(vectors)
                self._writer.status("action_published")
            else:
                self._writer.status("action_ignored", dry_run=self._dry_run, publish_actions=False)
        elif request_type == "publish_eef_action":
            vectors = _validate_vectors(request, "publish_eef_action")
            if self._publish_actions:
                self._publish_eef(vectors)
                self._writer.status("eef_action_published")
            else:
                self._writer.status("eef_action_ignored", dry_run=self._dry_run, publish_actions=False)
        elif request_type == "ping":
            metadata = {"id": request["id"]} if "id" in request else {}
            self._writer.status("pong", **metadata)
        elif request_type == "stop":
            self._writer.write({"type": "stopped"})
            self._stop.set()
        else:
            raise ValueError(f"Unsupported request type: {request_type!r}")

    def _publish_joint(self, vectors: dict[str, list[float]]) -> None:
        stamp = self.get_clock().now().to_msg()
        for arm in ("left", "right"):
            message = JointState()
            message.header.stamp = stamp
            message.name = [f"joint{i}" for i in range(7)]
            message.position = vectors[arm]
            self._action_publishers[arm].publish(message)

    def _publish_eef(self, vectors: dict[str, list[float]]) -> None:
        for arm in ("left", "right"):
            values = vectors[arm]
            message = PosCmd()
            message.x, message.y, message.z = values[:3]
            message.roll, message.pitch, message.yaw = values[3:6]
            message.gripper = values[6]
            message.mode1 = 0
            message.mode2 = 0
            self._action_publishers[arm].publish(message)


def _stdin(node: LocalPiperBridge, writer: Writer) -> None:
    for line in sys.stdin:
        try:
            node.enqueue(decode_message(line))
        except ValueError as exc:
            writer.error(str(exc))
    node.enqueue(None)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Piper legacy ROS 2 bridge")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--publish-actions", action="store_true")
    parser.add_argument("--include-eef", action="store_true")
    parser.add_argument("--eef-control", action="store_true")
    parser.add_argument("--eef-left-topic")
    parser.add_argument("--eef-right-topic")
    parser.add_argument("--eef-left-action-topic", default="/pos_left_cmd")
    parser.add_argument("--eef-right-action-topic", default="/pos_right_cmd")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    writer = Writer(sys.stdout)
    try:
        data = yaml.safe_load(pathlib.Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("control_stack_id") != "local-ros":
            raise ValueError("Local bridge requires a local-ros config")
    except Exception as exc:
        writer.error(str(exc))
        return 2
    rclpy.init(args=None)
    node: LocalPiperBridge | None = None
    try:
        node = LocalPiperBridge(
            writer=writer,
            config=data,
            dry_run=args.dry_run,
            publish_actions=args.publish_actions,
            include_eef=args.include_eef,
            eef_control=args.eef_control,
            eef_left_topic=args.eef_left_topic,
            eef_right_topic=args.eef_right_topic,
            eef_left_action_topic=args.eef_left_action_topic,
            eef_right_action_topic=args.eef_right_action_topic,
        )
        threading.Thread(target=_stdin, args=(node, writer), daemon=True).start()
        while rclpy.ok() and not node.stopped():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        writer.write({"type": "stopped"})
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
