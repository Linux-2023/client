from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def collector():
    return importlib.import_module("record_eef_ros2_trace")


def poscmd():
    return SimpleNamespace(
        x=0.123456789012345,
        y=-0.00000049,
        z=0.243000001,
        roll=0.1,
        pitch=-0.2,
        yaw=0.3,
        gripper=0.004321,
        mode1=1,
        mode2=0,
    )


def header():
    return SimpleNamespace(stamp=SimpleNamespace(sec=1_750_000_000, nanosec=123_456_789), frame_id="piper_base")


def driver_payload():
    return {
        "schema_version": 1,
        "event": "driver_eef_command",
        "arm": "left",
        "sequence": 42,
        "timestamp_ns": 1_750_000_000_123_456_700,
        "monotonic_ns": 123456789,
        "input_xyzrpy_gripper": [0.123456789012345, -0.00000049, 0.243000001, 0.1, -0.2, 0.3, 0.004321],
        "sdk_pose": [123457, 0, 243000, 5730, -11459, 17189],
        "quantization": "sdk",
        "dispatched": False,
        "motion_ctrl_2": [1, 0, 50, 0],
        "future_driver_field": {"values": [17, False, None]},
    }


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_poscmd_preserves_precision_modes_units_and_unstamped_identity(collector):
    event = collector.serialize_ros_message(
        "/pos_left_cmd",
        "piper_msgs/msg/PosCmd",
        poscmd(),
        timestamp_ns=111,
        monotonic_ns=222,
    )
    assert event["schema_version"] == 1
    assert event["event"] == "ros_message"
    assert event["timestamp_ns"] == 111
    assert event["monotonic_ns"] == 222
    assert event["topic"] == "/pos_left_cmd"
    assert event["message_type"] == "piper_msgs/msg/PosCmd"
    assert "source_timestamp_ns" not in event
    assert "sequence" not in event
    assert event["data"]["x"] == poscmd().x
    assert event["data"]["y"] == poscmd().y
    assert event["data"]["gripper"] == poscmd().gripper
    assert event["data"]["mode1"] == 1
    assert event["data"]["mode2"] == 0
    assert event["data"]["units"] == {
        "x": "m",
        "y": "m",
        "z": "m",
        "roll": "rad",
        "pitch": "rad",
        "yaw": "rad",
        "gripper": "m",
        "mode1": "dimensionless",
        "mode2": "dimensionless",
    }
    assert json.loads(json.dumps(event))["data"]["x"] == poscmd().x


def test_pose_preserves_raw_quaternion_position_and_integer_header_stamp(collector):
    message = SimpleNamespace(
        header=header(),
        pose=SimpleNamespace(
            position=SimpleNamespace(x=0.123456789012345, y=-0.00123, z=0.25),
            orientation=SimpleNamespace(x=0.1, y=0.2, z=0.3, w=0.9),
        ),
    )
    event = collector.serialize_ros_message("/eef", "geometry_msgs/msg/PoseStamped", message)
    assert event["source_timestamp_ns"] == 1_750_000_000_123_456_789
    assert event["data"]["header"] == {
        "stamp": {"sec": 1_750_000_000, "nanosec": 123_456_789},
        "frame_id": "piper_base",
    }
    assert event["data"]["position"] == {"x": 0.123456789012345, "y": -0.00123, "z": 0.25}
    assert event["data"]["orientation"] == {"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.9}
    assert event["data"]["units"] == {"position": "m", "orientation": "dimensionless (quaternion xyzw)"}
    assert isinstance(event["timestamp_ns"], int)
    assert isinstance(event["monotonic_ns"], int)


def test_joint_feedback_keeps_all_fields_separate_from_eef_coordinates(collector):
    message = SimpleNamespace(
        header=header(),
        name=[f"joint{i}" for i in range(7)],
        position=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.007],
        velocity=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0],
        effort=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
    )
    event = collector.serialize_ros_message("/joints", "sensor_msgs/msg/JointState", message)
    assert event["source_timestamp_ns"] == 1_750_000_000_123_456_789
    for field in ("name", "position", "velocity", "effort"):
        assert event["data"][field] == getattr(message, field)
    assert event["data"]["units"]["position"] == ["rad"] * 6 + ["m"]
    assert len(event["data"]["units"]["velocity"]) == len(message.velocity)
    assert len(event["data"]["units"]["effort"]) == len(message.effort)
    assert "tracking_error" not in event["data"]


def test_diagnostic_preserves_entire_payload_and_original_text(collector):
    payload = driver_payload()
    text = json.dumps(payload, indent=2)
    event = collector.serialize_ros_message("/trace", "std_msgs/msg/String", SimpleNamespace(data=text))
    assert "source_timestamp_ns" not in event
    assert event["data"]["raw"] == text
    assert event["data"]["payload"] == payload
    assert event["data"]["parse_error"] is None
    assert event["data"]["units"]["sdk_pose"] == ["0.001 mm"] * 3 + ["0.001 degree"] * 3
    assert event["data"]["units"]["input_xyzrpy_gripper"] == ["m", "m", "m", "rad", "rad", "rad", "m"]


@pytest.mark.parametrize("text", ["{broken", "[]", "null", '{"event":"unrelated"}'])
def test_malformed_diagnostic_is_recorded_without_stopping_capture(collector, text):
    event = collector.serialize_ros_message("/trace", "std_msgs/msg/String", SimpleNamespace(data=text))
    assert event["data"]["raw"] == text
    assert event["data"]["parse_error"]
    if text != "{broken":
        assert event["data"]["payload"] == json.loads(text)


def test_overflowing_diagnostic_number_retains_raw_and_flushes_summary(collector, tmp_path):
    text = '{"input_xyzrpy_gripper":[1e999]}'
    with collector.RosTraceWriter(tmp_path, {"/trace": "std_msgs/msg/String"}, diagnostic_topics=("/trace",)) as writer:
        writer.record_message("/trace", "std_msgs/msg/String", SimpleNamespace(data=text))
    written = records(tmp_path / "ros.jsonl")
    assert written[1]["data"]["raw"] == text
    assert written[1]["data"]["parse_error"]
    assert written[-1]["invalid_diagnostic_counts"] == {"/trace": 1}
    assert written[-1]["reason"] == "completed"


def test_output_exclusivity_preserves_existing_ros_and_client_files(collector, tmp_path):
    ros_path = tmp_path / "ros.jsonl"
    ros_path.write_text("existing trace\n", encoding="utf-8")
    metadata = tmp_path / "client_metadata.json"
    metadata.write_text('{"existing":true}', encoding="utf-8")
    with pytest.raises(FileExistsError):
        collector.RosTraceWriter(tmp_path, {"/cmd": "piper_msgs/msg/PosCmd"}, diagnostic_topics=())
    assert ros_path.read_text(encoding="utf-8") == "existing trace\n"
    assert metadata.read_text(encoding="utf-8") == '{"existing":true}'


def test_periodic_flush_makes_received_records_visible(collector, tmp_path):
    with collector.RosTraceWriter(tmp_path, {"/cmd": "piper_msgs/msg/PosCmd"}, diagnostic_topics=()) as writer:
        writer.record_message("/cmd", "piper_msgs/msg/PosCmd", poscmd())
        writer.flush()
        written = records(tmp_path / "ros.jsonl")
        assert written[0]["event"] == "ros_trace_start"
        assert written[-1]["event"] == "ros_message"
        assert written[-1]["data"]["x"] == poscmd().x


@pytest.mark.parametrize("failure", [None, KeyboardInterrupt, RuntimeError])
def test_shutdown_flushes_buffered_events_and_reports_missing_streams(collector, tmp_path, failure):
    topic_types = {"/cmd": "piper_msgs/msg/PosCmd", "/trace": "std_msgs/msg/String"}
    try:
        with collector.RosTraceWriter(tmp_path, topic_types, diagnostic_topics=("/trace",)) as writer:
            writer.record_message("/cmd", "piper_msgs/msg/PosCmd", poscmd())
            if failure is not None:
                raise failure("shutdown")
    except (KeyboardInterrupt, RuntimeError):
        pass
    written = records(tmp_path / "ros.jsonl")
    assert [event["event"] for event in written] == ["ros_trace_start", "ros_message", "ros_trace_end"]
    assert written[0]["counts"] == {"/cmd": 0, "/trace": 0}
    summary = written[-1]
    assert summary["counts"] == {"/cmd": 1, "/trace": 0}
    assert summary["missing_streams"] == ["/trace"]
    assert summary["missing_diagnostic_streams"] == ["/trace"]
    assert summary["diagnostic_streams_observed"] is False
    assert summary["dds_loss_fully_observable"] is False
    assert summary["reason"] == (
        "completed" if failure is None else "interrupted" if failure is KeyboardInterrupt else "exception"
    )


def test_malformed_diagnostics_do_not_count_as_observed_driver_payloads(collector, tmp_path):
    topic_types = {"/trace": "std_msgs/msg/String"}
    with collector.RosTraceWriter(tmp_path, topic_types, diagnostic_topics=("/trace",)) as writer:
        writer.record_message("/trace", "std_msgs/msg/String", SimpleNamespace(data="bad json"))
    summary = records(tmp_path / "ros.jsonl")[-1]
    assert summary["counts"] == {"/trace": 1}
    assert summary["missing_streams"] == []
    assert summary["missing_diagnostic_streams"] == ["/trace"]
    assert summary["invalid_diagnostic_counts"] == {"/trace": 1}
    assert summary["diagnostic_streams_observed"] is False


def test_parser_defaults_and_named_overrides_without_ros(collector, tmp_path):
    parser = collector.build_arg_parser()
    args = parser.parse_args(["--output-dir", str(tmp_path), "--duration", "0.25"])
    assert args.duration == 0.25
    assert args.eef_left_action_topic == "/pos_left_cmd"
    assert args.eef_right_action_topic == "/pos_right_cmd"
    assert args.eef_left_topic == "/puppet/end_pose_left"
    assert args.eef_right_topic == "/puppet/end_pose_right"
    assert args.joint_left_topic == "/puppet/joint_left"
    assert args.joint_right_topic == "/puppet/joint_right"
    assert args.driver_left_trace_topic == "/piper_left_ctrl_node/eef_command_trace"
    assert args.driver_right_trace_topic == "/piper_right_ctrl_node/eef_command_trace"
    args = parser.parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--eef-left-action-topic",
            "/custom_cmd",
            "--eef-right-topic",
            "/custom_eef",
            "--driver-left-trace-topic",
            "/custom_trace",
        ]
    )
    assert args.eef_left_action_topic == "/custom_cmd"
    assert args.eef_right_topic == "/custom_eef"
    assert args.driver_left_trace_topic == "/custom_trace"


@pytest.mark.parametrize("duration", ["0", "-1", "nan", "inf"])
def test_parser_rejects_invalid_duration(collector, tmp_path, duration):
    with pytest.raises(SystemExit):
        collector.build_arg_parser().parse_args(["--output-dir", str(tmp_path), "--duration", duration])
