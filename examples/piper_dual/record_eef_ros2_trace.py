#!/usr/bin/env python3
"""Read-only, two-arm ROS 2 EEF trace collector; no hardware driver imports.

Run with the system ROS Python after sourcing ROS and the piper_msgs overlay:
    /usr/bin/python3 examples/piper_dual/record_eef_ros2_trace.py --output-dir PATH --duration 60

Receipt timestamps use host wall/monotonic clocks, not ROS simulation time.
Received counts cannot establish loss-free DDS delivery or physical execution.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
QOS_DEPTH = 1000
FLUSH_INTERVAL_SECONDS = 0.5
POSCMD_TYPE = "piper_msgs/msg/PosCmd"
DIAGNOSTIC_TYPE = "std_msgs/msg/String"
JOINT_TYPE = "sensor_msgs/msg/JointState"
POSE_TYPE = "geometry_msgs/msg/PoseStamped"

_TOPIC_ARGUMENTS = (
    ("eef-left-action-topic", "/pos_left_cmd", POSCMD_TYPE),
    ("eef-right-action-topic", "/pos_right_cmd", POSCMD_TYPE),
    ("driver-left-trace-topic", "/piper_left_ctrl_node/eef_command_trace", DIAGNOSTIC_TYPE),
    ("driver-right-trace-topic", "/piper_right_ctrl_node/eef_command_trace", DIAGNOSTIC_TYPE),
    ("joint-left-topic", "/puppet/joint_left", JOINT_TYPE),
    ("joint-right-topic", "/puppet/joint_right", JOINT_TYPE),
    ("eef-left-topic", "/puppet/end_pose_left", POSE_TYPE),
    ("eef-right-topic", "/puppet/end_pose_right", POSE_TYPE),
)
_POSITION_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
_POSITION_UNITS = ("m", "m", "m", "rad", "rad", "rad", "m")
_QOS = {"reliability": "best_effort", "durability": "volatile", "history": "keep_last", "depth": QOS_DEPTH}


def _event(event: str, *, timestamp_ns: int | None = None, monotonic_ns: int | None = None) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "event": event,
        "timestamp_ns": time.time_ns() if timestamp_ns is None else timestamp_ns,
        "monotonic_ns": time.monotonic_ns() if monotonic_ns is None else monotonic_ns,
    }


def _header_data(header: Any) -> dict[str, Any]:
    return {
        "stamp": {"sec": int(header.stamp.sec), "nanosec": int(header.stamp.nanosec)},
        "frame_id": str(header.frame_id),
    }


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _finite_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite JSON number: {value}")
    return number


def _driver_payload_error(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return "driver diagnostic must be a JSON object"
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("event") != "driver_eef_command":
        return "expected schema_version=1 and event=driver_eef_command"
    if payload.get("arm") not in ("left", "right") or payload.get("quantization") not in ("sdk", "legacy_mm"):
        return "driver diagnostic has invalid arm or quantization"
    for field in ("sequence", "timestamp_ns", "monotonic_ns"):
        if type(payload.get(field)) is not int:
            return f"driver diagnostic {field} must be an integer"
    if type(payload.get("dispatched")) is not bool:
        return "driver diagnostic dispatched must be a boolean"
    for field, length, integer in (
        ("input_xyzrpy_gripper", 7, False),
        ("sdk_pose", 6, True),
        ("motion_ctrl_2", 4, True),
    ):
        values = payload.get(field)
        if not isinstance(values, list) or len(values) != length:
            return f"driver diagnostic {field} must have {length} values"
        for value in values:
            if integer and type(value) is not int:
                return f"driver diagnostic {field} must contain integers"
            if not integer and (type(value) not in (int, float) or not math.isfinite(value)):
                return f"driver diagnostic {field} must contain finite numbers"
    return None


def _diagnostic_data(message: Any) -> dict[str, Any]:
    raw = str(message.data)
    payload = None
    try:
        payload = json.loads(raw, parse_constant=_reject_json_constant, parse_float=_finite_json_float)
        parse_error = _driver_payload_error(payload)
    except (ValueError, OverflowError) as exc:
        parse_error = str(exc)
    return {
        "raw": raw,
        "payload": payload,
        "parse_error": parse_error,
        "units": {
            "input_xyzrpy_gripper": list(_POSITION_UNITS),
            "sdk_pose": ["0.001 mm"] * 3 + ["0.001 degree"] * 3,
            "timestamp_ns": "ns (host wall clock)",
            "monotonic_ns": "ns (host monotonic clock)",
            "sequence": "count (driver process local)",
            "motion_ctrl_2": "SDK protocol values [ctrl_mode, move_mode, speed_percent, mit_mode]",
        },
        "dispatched_semantics": "SDK called; not CAN success or physical execution",
    }


def serialize_ros_message(
    topic: str,
    message_type: str,
    message: Any,
    *,
    timestamp_ns: int | None = None,
    monotonic_ns: int | None = None,
) -> dict[str, Any]:
    """Preserve received message values, with separate receipt and source stamps.

    Joint units describe the active local Piper driver's six joints plus gripper
    layout. The driver's fixed zero velocity/effort placeholders are NOT measured.
    Pose quaternions are retained verbatim, without normalization or RPY rounding.
    """
    event = _event("ros_message", timestamp_ns=timestamp_ns, monotonic_ns=monotonic_ns)
    event.update(topic=topic, message_type=message_type)
    if message_type == POSCMD_TYPE:
        data = {field: float(getattr(message, field)) for field in _POSITION_FIELDS}
        data.update(mode1=int(message.mode1), mode2=int(message.mode2))
        data["units"] = dict(zip(_POSITION_FIELDS, _POSITION_UNITS))
        data["units"].update(mode1="dimensionless", mode2="dimensionless")
    elif message_type == DIAGNOSTIC_TYPE:
        data = _diagnostic_data(message)
    elif message_type == JOINT_TYPE:
        data = {
            "name": [str(value) for value in message.name],
            "position": [float(value) for value in message.position],
            "velocity": [float(value) for value in message.velocity],
            "effort": [float(value) for value in message.effort],
        }
        data["units"] = {
            "position": [
                "rad" if index < 6 else "m" if index == 6 else "unknown" for index in range(len(data["position"]))
            ],
            "velocity": [
                "rad/s" if index < 6 else "unmeasured placeholder" if index == 6 else "unknown"
                for index in range(len(data["velocity"]))
            ],
            "effort": [
                "unmeasured placeholder" if index < 6 else "N m" if index == 6 else "unknown"
                for index in range(len(data["effort"]))
            ],
        }
        data["unit_provenance"] = (
            "Local Piper driver index order: six revolute joints, gripper. SDK motor_speed/1000 is rad/s; "
            "grippers_effort/1000 is torque N m per SDK English definition (Chinese text says N/m). "
            "Gripper velocity and first six efforts are fixed zeros, not measurements. "
            "Topic overrides must retain this layout; no conversion is applied."
        )
    elif message_type == POSE_TYPE:
        data = {
            "position": {axis: float(getattr(message.pose.position, axis)) for axis in ("x", "y", "z")},
            "orientation": {axis: float(getattr(message.pose.orientation, axis)) for axis in ("x", "y", "z", "w")},
            "units": {"position": "m", "orientation": "dimensionless (quaternion xyzw)"},
        }
    else:
        raise ValueError(f"unsupported message type: {message_type}")
    if hasattr(message, "header"):
        data["header"] = _header_data(message.header)
        stamp = data["header"]["stamp"]
        event["source_timestamp_ns"] = stamp["sec"] * 1_000_000_000 + stamp["nanosec"]
    event["data"] = data
    return event


class RosTraceWriter:
    """Exclusive, buffered JSONL sink; callers periodically flush and close it."""

    def __init__(self, output_dir: Path | str, topic_types: Mapping[str, str], *, diagnostic_topics: Sequence[str]):
        self.path = Path(output_dir) / "ros.jsonl"
        self.topic_types = dict(topic_types)
        self.diagnostic_topics = tuple(diagnostic_topics)
        if any(topic not in self.topic_types for topic in self.diagnostic_topics):
            raise ValueError("diagnostic topics must be included in topic_types")
        self.counts = dict.fromkeys(self.topic_types, 0)
        self.valid_diagnostic_counts = dict.fromkeys(self.diagnostic_topics, 0)
        self.invalid_diagnostic_counts = dict.fromkeys(self.diagnostic_topics, 0)
        self.end_summary: dict[str, Any] | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8", buffering=64 * 1024)
        try:
            start = self._summary("ros_trace_start", "started")
            start.update(
                topics=self.topic_types,
                qos=_QOS,
                flush_interval_seconds=FLUSH_INTERVAL_SECONDS,
                receipt_clock="timestamp_ns=time.time_ns(); monotonic_ns=time.monotonic_ns(); same host as client/driver",
                source_clock="original message header stamp, if present; producer clock may differ from receipt clock",
                unstamped_messages="PosCmd and String have no source stamp; no synthetic message IDs or matching claims",
                observation_semantics="receipt only, not control acceptance or physical execution",
                dds_loss_note="Best-effort subscriber accepts reliable and best-effort publishers; DDS loss is not fully observable.",
            )
            self.write_event(start)
            self.flush()
        except BaseException:
            self._stream.close()
            raise

    def _summary(self, event: str, reason: str) -> dict[str, Any]:
        summary = _event(event)
        missing_diagnostics = [topic for topic, count in self.valid_diagnostic_counts.items() if count == 0]
        summary.update(
            reason=reason,
            counts=dict(self.counts),
            missing_streams=[topic for topic, count in self.counts.items() if count == 0],
            valid_diagnostic_counts=dict(self.valid_diagnostic_counts),
            invalid_diagnostic_counts=dict(self.invalid_diagnostic_counts),
            missing_diagnostic_streams=missing_diagnostics,
            diagnostic_streams_observed=not missing_diagnostics,
            dds_loss_fully_observable=False,
        )
        return summary

    def write_event(self, event: Mapping[str, Any]) -> None:
        self._stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")

    def record_message(self, topic: str, message_type: str, message: Any) -> dict[str, Any]:
        if self.topic_types.get(topic) != message_type:
            raise ValueError(f"message type does not match configured topic {topic}")
        event = serialize_ros_message(topic, message_type, message)
        self.write_event(event)
        self.counts[topic] += 1
        if topic in self.valid_diagnostic_counts:
            counts = (
                self.valid_diagnostic_counts if event["data"]["parse_error"] is None else self.invalid_diagnostic_counts
            )
            counts[topic] += 1
        return event

    def flush(self) -> None:
        self._stream.flush()

    def close(self, reason: str = "completed") -> None:
        if self._stream.closed:
            return
        try:
            self.end_summary = self._summary("ros_trace_end", reason)
            self.write_event(self.end_summary)
            self.flush()
        finally:
            self._stream.close()

    def __enter__(self) -> RosTraceWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        reason = (
            "completed"
            if exc_type is None
            else "interrupted"
            if issubclass(exc_type, KeyboardInterrupt)
            else "exception"
        )
        self.close(reason)


def _positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("duration must be a positive finite number") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("duration must be a positive finite number")
    return seconds


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Observe two-arm EEF commands, SDK diagnostics, joints and EEF feedback without robot control.",
        epilog="Source ROS and the piper_msgs overlay first. Missing diagnostics prevent end-to-end capture; received counts do not prove zero DDS loss.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="directory for exclusive ros.jsonl (may share client trace directory)",
    )
    parser.add_argument(
        "--duration", type=_positive_seconds, help="positive finite seconds after READY; otherwise stop with Ctrl-C"
    )
    for option, default, message_type in _TOPIC_ARGUMENTS:
        parser.add_argument(f"--{option}", default=default, help=f"{message_type} topic (default: {default})")
    return parser


def topics_from_args(args: argparse.Namespace) -> dict[str, str]:
    topics: dict[str, str] = {}
    for option, _, message_type in _TOPIC_ARGUMENTS:
        topic = getattr(args, option.replace("-", "_"))
        if not topic or topic in topics:
            raise ValueError(f"topic overrides must be nonempty and distinct: {topic!r}")
        topics[topic] = message_type
    return topics


def record_ros_trace(args: argparse.Namespace) -> dict[str, Any]:
    """Run subscriptions only, flush on duration/SIGINT/exception, return summary.

    ROS imports are lazy so argument parsing and serialization work without ROS.
    No application publisher, service, client or hardware interface is created;
    standard ROS node metadata and DDS discovery traffic are still present.
    """
    topic_types = topics_from_args(args)
    diagnostic_topics = (args.driver_left_trace_topic, args.driver_right_trace_topic)
    writer = RosTraceWriter(args.output_dir, topic_types, diagnostic_topics=diagnostic_topics)
    try:
        with writer:
            try:
                import rclpy
                from geometry_msgs.msg import PoseStamped
                from piper_msgs.msg import PosCmd
                from rclpy.context import Context
                from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
                from rclpy.node import Node
                from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
                from sensor_msgs.msg import JointState
                from std_msgs.msg import String
            except ImportError as exc:
                raise RuntimeError(
                    "Use system ROS Python and source ROS plus piper_msgs overlay; required packages: rclpy, geometry_msgs, piper_msgs, sensor_msgs, std_msgs"
                ) from exc

            context = Context()
            node = None
            executor = None
            subscriptions = []
            try:
                rclpy.init(args=[], context=context)
                node = Node(
                    "piper_dual_eef_trace_recorder",
                    context=context,
                    enable_rosout=False,
                    start_parameter_services=False,
                )
                executor = SingleThreadedExecutor(context=context)
                executor.add_node(node)
                qos = QoSProfile(
                    history=HistoryPolicy.KEEP_LAST,
                    depth=QOS_DEPTH,
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    durability=DurabilityPolicy.VOLATILE,
                )
                message_classes = {
                    POSCMD_TYPE: PosCmd,
                    DIAGNOSTIC_TYPE: String,
                    JOINT_TYPE: JointState,
                    POSE_TYPE: PoseStamped,
                }
                for topic, message_type in topic_types.items():

                    def callback(message: Any, *, topic_name: str = topic, type_name: str = message_type) -> None:
                        event = writer.record_message(topic_name, type_name, message)
                        if (
                            type_name == DIAGNOSTIC_TYPE
                            and event["data"]["parse_error"] is not None
                            and writer.invalid_diagnostic_counts[topic_name] == 1
                        ):
                            print(
                                f"WARNING: invalid driver diagnostic on {topic_name}: {event['data']['parse_error']}; raw text retained",
                                file=sys.stderr,
                                flush=True,
                            )

                    subscriptions.append(node.create_subscription(message_classes[message_type], topic, callback, qos))
                writer.flush()
                print(
                    f"READY: read-only EEF trace collector; {len(subscriptions)} subscriptions; output={writer.path}; QoS=best_effort depth={QOS_DEPTH}",
                    flush=True,
                )
                print(
                    "Received counters only: DDS loss is not fully observable; diagnostics report SDK calls, NOT CAN success or physical execution.",
                    flush=True,
                )
                started = time.monotonic()
                deadline = None if args.duration is None else started + args.duration
                next_flush = started + FLUSH_INTERVAL_SECONDS
                next_graph_check = started
                previous_missing = None
                reason = "ros_shutdown"
                while context.ok():
                    now = time.monotonic()
                    if deadline is not None and now >= deadline:
                        reason = "duration"
                        break
                    if now >= next_graph_check:
                        missing = [topic for topic in diagnostic_topics if node.count_publishers(topic) == 0]
                        if missing != previous_missing:
                            graph_event = _event("ros_diagnostic_publishers")
                            graph_event["missing_diagnostic_publishers"] = missing
                            writer.write_event(graph_event)
                            if missing:
                                print(
                                    "WARNING: driver diagnostic publishers not discovered: "
                                    + ", ".join(missing)
                                    + "; discovery may still be pending, driver may be old, or eef_command_trace may be off. Capture is incomplete without valid diagnostics.",
                                    file=sys.stderr,
                                    flush=True,
                                )
                            previous_missing = missing
                        next_graph_check = now + 1.0
                    timeout = min(0.1, max(0.0, next_flush - now))
                    if deadline is not None:
                        timeout = min(timeout, max(0.0, deadline - now))
                    executor.spin_once(timeout_sec=timeout)
                    if time.monotonic() >= next_flush:
                        writer.flush()
                        next_flush = time.monotonic() + FLUSH_INTERVAL_SECONDS
                writer.close(reason)
            except (KeyboardInterrupt, ExternalShutdownException):
                writer.close("interrupted")
            finally:
                try:
                    if executor is not None:
                        executor.shutdown()
                finally:
                    try:
                        subscriptions.clear()
                        if node is not None:
                            node.destroy_node()
                    finally:
                        if context.ok():
                            context.shutdown()
    finally:
        if writer.end_summary is not None:
            print("TRACE SUMMARY: " + json.dumps(writer.end_summary, sort_keys=True), flush=True)
            if writer.end_summary["missing_diagnostic_streams"]:
                print(
                    "WARNING: incomplete driver capture; no valid diagnostic messages from: "
                    + ", ".join(writer.end_summary["missing_diagnostic_streams"]),
                    file=sys.stderr,
                    flush=True,
                )
    return writer.end_summary


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        record_ros_trace(args)
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
