#!/usr/bin/python3
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
import time
from dataclasses import replace
from typing import Any

import yaml

import cv_bridge
import rclpy
from geometry_msgs.msg import PoseStamped
from piper_msgs.msg import PosCmd
from piper_msgs.msg import PiperStatusMsg
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState

try:
    from examples.piper_dual.control_stack import load_profile_config
    from examples.piper_dual.control_stack import profile_for
    from examples.piper_dual.ros2_contract import BridgeContract
    from examples.piper_dual.ros2_contract import EndpointPlan
    from examples.piper_dual.ros2_contract import validate_profile_graph
    from examples.piper_dual.ros2_protocol import decode_message
    from examples.piper_dual.ros2_protocol import encode_message
    from examples.piper_dual.ros2_safety import PiperStatusLatch
except ModuleNotFoundError:
    from control_stack import load_profile_config
    from control_stack import profile_for
    from ros2_contract import BridgeContract
    from ros2_contract import EndpointPlan
    from ros2_contract import validate_profile_graph
    from ros2_protocol import decode_message
    from ros2_protocol import encode_message
    from ros2_safety import PiperStatusLatch

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


def _quaternion_to_rpy(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    """Convert a finite quaternion to ROS/tf2 XYZ roll, pitch, yaw radians."""
    components = (float(x), float(y), float(z), float(w))
    if not all(math.isfinite(component) for component in components):
        raise ValueError("EEF quaternion values must be finite")
    norm = math.sqrt(sum(component * component for component in components))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("EEF quaternion must have non-zero finite norm")
    x, y, z, w = (component / norm for component in components)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def build_eef_sensor_event(sensor: str, message: Any) -> JsonObject:
    """Build one EEF PoseStamped event as finite XYZ+RPY values."""
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
    return {
        "type": "sensor",
        "sensor": sensor,
        "timestamp": _format_sensor_timestamp(message.header.stamp),
        "values": values,
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


def validate_eef_action_request(request: JsonObject) -> dict[str, list[float]]:
    """Validate two Piper PosCmd vectors: xyz, RPY radians, and gripper meters."""
    vectors: dict[str, list[float]] = {}
    for arm in ("left", "right"):
        raw_values = request.get(arm)
        if not isinstance(raw_values, list) or len(raw_values) != 7:
            raise ValueError(f"publish_eef_action.{arm} must contain exactly seven values")
        values: list[float] = []
        for raw_value in raw_values:
            if isinstance(raw_value, bool):
                raise ValueError(f"publish_eef_action.{arm} values must be finite numbers")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"publish_eef_action.{arm} values must be finite numbers") from exc
            if not math.isfinite(value):
                raise ValueError(f"publish_eef_action.{arm} values must be finite numbers")
            values.append(value)
        vectors[arm] = values
    return vectors


def _load_config(path: str | None) -> JsonObject:
    if path is None:
        return {}

    config_path = pathlib.Path(path)
    suffix = config_path.suffix.lower()
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            if suffix in {".yaml", ".yml"}:
                data = yaml.safe_load(stream)
            elif suffix == ".json":
                data = json.load(stream)
            else:
                raise ValueError(f"Unsupported bridge config extension {suffix!r}; expected .json, .yaml, or .yml")
    except FileNotFoundError as exc:
        raise ValueError(f"Bridge config file not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON bridge config {config_path}: {exc.msg}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed YAML bridge config {config_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Bridge config must be a mapping object")
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

    def __init__(
        self,
        *,
        writer: LockedJsonLineWriter,
        dry_run: bool,
        publish_actions: bool,
        contract: BridgeContract,
        config: JsonObject,
        eef_left_topic: str | None = None,
        eef_right_topic: str | None = None,
        eef_left_action_topic: str | None = None,
        eef_right_action_topic: str | None = None,
        eef_control: bool = False,
        control_stack: str = "official-ros",
    ) -> None:
        super().__init__("piper_dual_ros2_bridge")
        if (eef_left_topic is None) != (eef_right_topic is None):
            raise ValueError("EEF left and right topics must be provided together")
        if (eef_left_action_topic is None) != (eef_right_action_topic is None):
            raise ValueError("EEF left and right action topics must be provided together")
        self._profile = profile_for(control_stack)
        config_identity = config.get("control_stack_id")
        if config_identity != self._profile.stack_id:
            raise ValueError(f"control_stack_id {config_identity!r} does not match expected {self._profile.stack_id!r}")


        self._writer = writer
        self._dry_run = dry_run
        self._publish_actions = publish_actions and not dry_run
        self._eef_control = bool(eef_control)
        self._bridge = cv_bridge.CvBridge()
        self._requests: queue.Queue[JsonObject | None] = queue.Queue(maxsize=64)
        self._stop_event = threading.Event()
        self._mode = str(config.get("mode", "default"))
        self._contract = _apply_eef_overrides(
            contract,
            eef_left_topic=eef_left_topic,
            eef_right_topic=eef_right_topic,
            eef_left_action_topic=eef_left_action_topic,
            eef_right_action_topic=eef_right_action_topic,
        )
        qos_depth = int(config.get("qos_depth", 1))
        self._endpoint_plan = EndpointPlan.for_mode(
            self._contract,
            dry_run=dry_run,
            publish_actions=publish_actions,
            eef_control=self._eef_control,
            include_eef=bool(self._contract.eef_topics),
        )
        self._qos_depth = qos_depth
        self._action_publishers: dict[str, Any] = {}
        self._action_publishing_disabled_reason: str | None = None
        self._status_latch = PiperStatusLatch(status_watchdog_s=self._contract.control.status_watchdog_s)
        self._hardware_fault_reported = False

        for sensor, topic in self._endpoint_plan.image_subscriptions:
            self.create_subscription(Image, topic, self._make_image_callback(sensor), qos_depth)
        for topic, sensor in self._endpoint_plan.joint_subscriptions:
            self.create_subscription(JointState, topic, self._make_joint_callback(sensor), qos_depth)
        for arm, topic in self._endpoint_plan.status_subscriptions:
            self.create_subscription(PiperStatusMsg, topic, self._make_status_callback(arm), qos_depth)
        for arm, topic in self._endpoint_plan.eef_subscriptions:
            self.create_subscription(PoseStamped, topic, self._make_eef_callback(_eef_sensor_alias(arm)), qos_depth)


        self._request_timer = self.create_timer(0.01, self._drain_requests)
        self._writer.status(
            "ready",
            dry_run=self._dry_run,
            publish_actions=self._publish_actions,
            eef_control=self._eef_control,
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

    def _make_status_callback(self, side: str):
        def callback(message: PiperStatusMsg) -> None:
            try:
                self._status_latch.update(side, message, received_at=time.monotonic())
                self._report_hardware_fault_if_latched()
            except Exception as exc:  # noqa: BLE001 - ROS callback must survive bad status frames.
                self._writer.error(str(exc), side=side)

        callback._sensor = side  # type: ignore[attr-defined]
        callback._side = side  # type: ignore[attr-defined]

        return callback

    def _make_eef_callback(self, sensor: str):
        def callback(message: PoseStamped) -> None:
            try:
                self._writer.write(build_eef_sensor_event(sensor, message))
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


    def _ensure_joint_publishers_ready(self) -> None:
        self._ensure_publishers_ready(validate_graph=True)

    def _ensure_eef_publishers_ready(self) -> None:
        self._ensure_publishers_ready()

    def _ensure_publishers_ready(self, *, validate_graph: bool = True) -> None:
        if self._action_publishing_disabled_reason is not None:
            raise RuntimeError(f"Action publishing permanently disabled: {self._action_publishing_disabled_reason}")
        try:
            if validate_graph:
                validate_profile_graph(self, self._profile, self._contract)
            self._assert_live_joint_ready()
            if self._action_publishers:
                return
            publisher_type = PosCmd if self._eef_control else JointState
            for arm, topic in self._endpoint_plan.action_publishers:
                self._action_publishers[arm] = self.create_publisher(publisher_type, topic, self._qos_depth)
        except Exception as exc:
            self._action_publishing_disabled_reason = str(exc)
            raise RuntimeError(f"Action publishing permanently disabled: {exc}") from exc

    def _assert_live_joint_ready(self) -> None:
        try:
            self._status_latch.assert_joint_ready(now=time.monotonic())
        finally:
            self._report_hardware_fault_if_latched()

    def _report_hardware_fault_if_latched(self) -> None:
        if self._hardware_fault_reported or self._status_latch.fault_reason is None:
            return
        self._hardware_fault_reported = True
        metadata: dict[str, Any] = {"reason": self._status_latch.fault_reason}
        if self._status_latch.fault_side is not None:
            metadata["side"] = self._status_latch.fault_side
        self._writer.status("hardware_fault", **metadata)

    def _handle_request(self, request: JsonObject) -> None:
        request_type = request.get("type")
        if request_type == "publish_action":
            if self._eef_control:
                raise ValueError("publish_action is unavailable in EEF control mode")
            vectors = validate_action_request(request)
            if self._publish_actions:
                self._ensure_joint_publishers_ready()
                self._publish_action(vectors["left"], vectors["right"])
                self._writer.status("action_published")
            else:
                self._writer.status("action_ignored", dry_run=self._dry_run, publish_actions=self._publish_actions)
        elif request_type == "publish_eef_action":
            if not self._eef_control:
                raise ValueError("publish_eef_action requires EEF control mode")
            vectors = validate_eef_action_request(request)
            if self._publish_actions:
                self._ensure_eef_publishers_ready()
                self._publish_eef_action(vectors["left"], vectors["right"])
                self._writer.status("eef_action_published")
            else:
                self._writer.status("eef_action_ignored", dry_run=self._dry_run, publish_actions=self._publish_actions)
        elif request_type == "set_mode":
            mode = request.get("mode")
            if not isinstance(mode, str) or not mode:
                raise ValueError("set_mode.mode must be a non-empty string")
            self._mode = mode
            self._writer.status("mode_set", mode=self._mode)
        elif request_type == "ping":
            metadata: dict[str, Any] = {}
            if "id" in request:
                metadata["id"] = request.get("id")
            self._writer.status("pong", **metadata)
        elif request_type == "stop":
            self._writer.write({"type": "stopped"})
            self._stop_event.set()
        else:
            raise ValueError(f"Unsupported request type: {request_type!r}")

    def _publish_action(self, left: list[float], right: list[float]) -> None:
        if "left" not in self._action_publishers or "right" not in self._action_publishers:
            raise RuntimeError("Action publishers are disabled")
        stamp = self.get_clock().now().to_msg()
        control = self._contract.control
        self._action_publishers["left"].publish(
            _make_joint_state(left, stamp, self._contract.joint_names, control.speed_percent, control.gripper_effort)
        )
        self._action_publishers["right"].publish(
            _make_joint_state(right, stamp, self._contract.joint_names, control.speed_percent, control.gripper_effort)
        )

    def _publish_eef_action(self, left: list[float], right: list[float]) -> None:
        if "left" not in self._action_publishers or "right" not in self._action_publishers:
            raise RuntimeError("EEF action publishers are disabled")
        stamp = self.get_clock().now().to_msg()
        self._action_publishers["left"].publish(_make_pos_cmd(left, stamp))
        self._action_publishers["right"].publish(_make_pos_cmd(right, stamp))


def _eef_sensor_alias(arm: str) -> str:
    if arm == "left":
        return "eef_puppet_left"
    if arm == "right":
        return "eef_puppet_right"
    raise ValueError(f"Unknown EEF arm: {arm!r}")


def _apply_eef_overrides(
    contract: BridgeContract,
    *,
    eef_left_topic: str | None,
    eef_right_topic: str | None,
    eef_left_action_topic: str | None,
    eef_right_action_topic: str | None,
) -> BridgeContract:
    if (eef_left_topic is None) != (eef_right_topic is None):
        raise ValueError("EEF left and right topics must be provided together")
    if (eef_left_action_topic is None) != (eef_right_action_topic is None):
        raise ValueError("EEF left and right action topics must be provided together")

    if eef_left_topic is not None:
        contract = replace(contract, eef_topics={"left": eef_left_topic, "right": eef_right_topic})
    if eef_left_action_topic is not None:
        contract = replace(
            contract,
            eef_action_topics={"left": eef_left_action_topic, "right": eef_right_action_topic},
        )
    return contract


def _make_joint_state(
    values: list[float],
    stamp: Any,
    joint_names: tuple[str, ...],
    speed_percent: int,
    gripper_effort: float,
) -> JointState:
    if len(values) != len(joint_names):
        raise ValueError("JointState values must contain exactly seven values")
    message = JointState()
    message.header.stamp.sec = int(getattr(stamp, "sec"))
    message.header.stamp.nanosec = int(getattr(stamp, "nanosec"))
    message.name = list(joint_names)
    message.position = list(values)
    message.velocity = [0.0] * (len(joint_names) - 1) + [float(speed_percent)]
    message.effort = [0.0] * (len(joint_names) - 1) + [float(gripper_effort)]
    return message


def _default_config_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().with_name("ros2_piper_dual.yaml")


def _make_pos_cmd(values: list[float], stamp: Any) -> PosCmd:
    if len(values) != 7:
        raise ValueError("PosCmd values must contain exactly seven values")
    message = PosCmd()
    message.x, message.y, message.z = values[0], values[1], values[2]
    message.roll, message.pitch, message.yaw = values[3], values[4], values[5]
    message.gripper = values[6]
    message.mode1 = 0
    message.mode2 = 0
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
    parser.add_argument("--config", help="Path to optional YAML or JSON bridge config")
    parser.add_argument("--dry-run", action="store_true", help="Never publish joint action messages")
    parser.add_argument("--publish-actions", action="store_true", help="Enable publishing validated joint actions")
    parser.add_argument(
        "--control-stack",
        choices=("local-ros", "official-ros", "direct-sdk"),
        default="official-ros",
        help="Control-stack profile whose config identity and ROS graph must match before publishing",
    )
    parser.add_argument("--eef-left-topic", help="Optional puppet-left PoseStamped topic")
    parser.add_argument("--eef-right-topic", help="Optional puppet-right PoseStamped topic")
    parser.add_argument("--eef-control", action="store_true", help="Publish Piper PosCmd EEF actions")
    parser.add_argument("--eef-left-action-topic", help="Optional left PosCmd action topic override")
    parser.add_argument("--eef-right-action-topic", help="Optional right PosCmd action topic override")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    writer = LockedJsonLineWriter(sys.stdout)
    try:
        config_path = pathlib.Path(args.config) if args.config is not None else profile_for(args.control_stack).config_path
        _profile, config = load_profile_config(args.control_stack, config_path)
        contract = BridgeContract.from_mapping(config)
    except Exception as exc:  # noqa: BLE001 - config errors must be emitted as protocol errors.
        writer.error(str(exc))
        return 2

    rclpy.init(args=None)
    node: PiperRos2Bridge | None = None
    try:
        node = PiperRos2Bridge(
            writer=writer,
            dry_run=args.dry_run,
            publish_actions=args.publish_actions,
            contract=contract,
            config=config,
            eef_left_topic=args.eef_left_topic,
            eef_right_topic=args.eef_right_topic,
            eef_left_action_topic=args.eef_left_action_topic,
            eef_right_action_topic=args.eef_right_action_topic,
            eef_control=args.eef_control,
            control_stack=args.control_stack,
        )
        reader = threading.Thread(target=_stdin_reader, args=(node, writer), daemon=True)
        reader.start()

        while rclpy.ok() and not node.stop_requested():
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
