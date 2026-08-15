"""Focused tests for the ROS 2 bridge backend client."""

from pathlib import Path
import json
import stat
import sys
import textwrap
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ros2_backend
from ros2_backend import Ros2BackendClient


SENSOR_ORDER = (
    "cam_high",
    "cam_left_wrist",
    "cam_right_wrist",
    "puppet_left",
    "puppet_right",
    "master_left",
    "master_right",
)


def write_fake_bridge(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "fake_bridge.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import base64\n"
        "import json\n"
        "import pathlib\n"
        "import sys\n"
        "import time\n"
        + textwrap.dedent(body),
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def write_config(tmp_path: Path) -> Path:
    config = tmp_path / "bridge-config.json"
    config.write_text('{"mode":"test"}', encoding="utf-8")
    return config


def sensor_event(sensor: str, timestamp: float) -> dict[str, object]:
    if sensor.startswith("cam_"):
        return {
            "type": "sensor",
            "sensor": sensor,
            "timestamp": timestamp,
            "jpeg_b64": __import__("base64").b64encode(f"{sensor}-jpeg".encode()).decode("ascii"),
        }
    start = {
        "puppet_left": 0,
        "puppet_right": 10,
        "master_left": 20,
        "master_right": 30,
    }[sensor]
    return {"type": "sensor", "sensor": sensor, "timestamp": timestamp, "values": [float(start + i) for i in range(7)]}


def complete_events(timestamp: float = 1.0) -> list[dict[str, object]]:
    return [sensor_event(sensor, timestamp) for sensor in SENSOR_ORDER]


def test_start_uses_specified_interpreter_config_and_safety_flags(tmp_path, monkeypatch):
    records = tmp_path / "records.jsonl"
    bridge = write_fake_bridge(
        tmp_path,
        f"""
        pathlib.Path({str(records)!r}).write_text(json.dumps({{"argv": sys.argv, "executable": sys.executable}}) + "\\n", encoding="utf-8")
        print(json.dumps({{"type": "status", "state": "ready"}}), flush=True)
        for line in sys.stdin:
            request = json.loads(line)
            pathlib.Path({str(records)!r}).open("a", encoding="utf-8").write(json.dumps({{"stdin": request}}) + "\\n")
            if request.get("type") == "stop":
                print(json.dumps({{"type": "stopped"}}), flush=True)
                break
        """,
    )
    monkeypatch.setattr(ros2_backend, "BRIDGE_SCRIPT", bridge)
    config = write_config(tmp_path)
    client = Ros2BackendClient(bridge_python=Path(sys.executable), config=config, dry_run=True)

    client.start()
    client.close()

    lines = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["executable"] == sys.executable
    assert lines[0]["argv"] == [str(bridge), "--config", str(config), "--dry-run"]
    assert {"stdin": {"type": "stop"}} in lines


def test_start_rejects_unsafe_dry_run_publish_actions_combination(tmp_path):
    client = Ros2BackendClient(
        bridge_python=Path(sys.executable),
        config=write_config(tmp_path),
        publish_actions=True,
        dry_run=True,
    )

    with pytest.raises(ValueError, match="dry_run=True.*publish_actions=True"):
        client.start()


def test_next_frame_returns_only_a_complete_synchronized_frame(tmp_path, monkeypatch):
    bridge = write_fake_bridge(
        tmp_path,
        """
        print(json.dumps({"type": "status", "state": "ready"}), flush=True)
        events = __EVENTS__
        for event in events[:-1]:
            print(json.dumps(event), flush=True)
        time.sleep(0.2)
        print(json.dumps(events[-1]), flush=True)
        for line in sys.stdin:
            if json.loads(line).get("type") == "stop":
                break
        """.replace("__EVENTS__", repr(complete_events(2.0))),
    )
    monkeypatch.setattr(ros2_backend, "BRIDGE_SCRIPT", bridge)
    client = Ros2BackendClient(bridge_python=Path(sys.executable), config=write_config(tmp_path))

    client.start()
    with pytest.raises(TimeoutError):
        client.next_frame(timeout=0.05)
    frame = client.next_frame(timeout=1.0)
    client.close()

    assert frame.timestamp == pytest.approx(2.0)
    assert sorted(frame.images) == ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    assert frame.images["cam_high"] == b"cam_high-jpeg"
    np.testing.assert_array_equal(frame.state, np.array(list(range(7)) + list(range(10, 17)), dtype=np.float32))


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, "6.0"], "real JSON numbers"),
        ([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, True], "real JSON numbers"),
        ([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "exactly seven"),
        ([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, float("inf")], "finite"),
        ({"joint0": 0.0}, "list or tuple"),
    ],
)
def test_raw_joint_sensor_values_are_validated_before_numpy_coercion(tmp_path, values, match):
    client = Ros2BackendClient(bridge_python=Path(sys.executable), config=write_config(tmp_path))

    with pytest.raises(RuntimeError, match=match):
        client._handle_message({"type": "sensor", "sensor": "puppet_left", "timestamp": 1.0, "values": values})


def test_clear_buffers_drops_queued_frames_and_synchronizer_state(tmp_path, monkeypatch):
    events = complete_events(3.0) + complete_events(4.0)
    bridge = write_fake_bridge(
        tmp_path,
        """
        print(json.dumps({"type": "status", "state": "ready"}), flush=True)
        for event in __EVENTS__:
            print(json.dumps(event), flush=True)
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("type") == "ping":
                print(json.dumps({"type": "status", "state": "pong", "metadata": {"id": request.get("id")}}), flush=True)
            if request.get("type") == "stop":
                break
        """.replace("__EVENTS__", repr(events)),
    )
    monkeypatch.setattr(ros2_backend, "BRIDGE_SCRIPT", bridge)
    client = Ros2BackendClient(bridge_python=Path(sys.executable), config=write_config(tmp_path))

    client.start()
    first = client.next_frame(timeout=1.0)
    client.clear_buffers()
    with pytest.raises(TimeoutError):
        client.next_frame(timeout=0.1)
    client.close()

    assert first.timestamp == pytest.approx(3.0)


def test_clear_buffers_uses_ping_barrier_to_exclude_pre_barrier_stdout(tmp_path, monkeypatch):
    bridge = write_fake_bridge(
        tmp_path,
        """
        print(json.dumps({"type": "status", "state": "ready"}), flush=True)
        for event in __PRE_EVENTS__:
            print(json.dumps(event), flush=True)
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("type") == "ping":
                print(json.dumps({"type": "status", "state": "pong", "metadata": {"id": request.get("id")}}), flush=True)
                for event in __POST_EVENTS__:
                    print(json.dumps(event), flush=True)
            if request.get("type") == "stop":
                break
        """.replace("__PRE_EVENTS__", repr(complete_events(7.0))).replace("__POST_EVENTS__", repr(complete_events(8.0))),
    )
    monkeypatch.setattr(ros2_backend, "BRIDGE_SCRIPT", bridge)
    client = Ros2BackendClient(bridge_python=Path(sys.executable), config=write_config(tmp_path))

    client.start()
    client.clear_buffers()
    frame = client.next_frame(timeout=1.0)
    client.close()

    assert frame.timestamp == pytest.approx(8.0)


def test_publish_action_sends_validated_fourteen_dimension_request_and_rejects_malformed_actions(tmp_path, monkeypatch):
    records = tmp_path / "stdin.jsonl"
    bridge = write_fake_bridge(
        tmp_path,
        f"""
        print(json.dumps({{"type": "status", "state": "ready"}}), flush=True)
        with pathlib.Path({str(records)!r}).open("w", encoding="utf-8") as sink:
            for line in sys.stdin:
                request = json.loads(line)
                sink.write(json.dumps(request) + "\\n")
                sink.flush()
                if request.get("type") == "stop":
                    break
        """,
    )
    monkeypatch.setattr(ros2_backend, "BRIDGE_SCRIPT", bridge)
    client = Ros2BackendClient(bridge_python=Path(sys.executable), config=write_config(tmp_path), publish_actions=True, dry_run=False)

    client.start()
    client.publish_action(np.arange(14, dtype=np.float64))
    with pytest.raises(ValueError):
        client.publish_action(np.arange(13, dtype=np.float32))
    client.close()

    requests = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()]
    assert requests[0] == {
        "type": "publish_action",
        "left": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "right": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
    }
    assert requests[-1] == {"type": "stop"}


def test_close_sends_stop_and_terminates_fake_bridge(tmp_path, monkeypatch):
    marker = tmp_path / "stopped.txt"
    bridge = write_fake_bridge(
        tmp_path,
        f"""
        print(json.dumps({{"type": "status", "state": "ready"}}), flush=True)
        for line in sys.stdin:
            if json.loads(line).get("type") == "stop":
                pathlib.Path({str(marker)!r}).write_text("stopped", encoding="utf-8")
                print(json.dumps({{"type": "stopped"}}), flush=True)
                break
        """,
    )
    monkeypatch.setattr(ros2_backend, "BRIDGE_SCRIPT", bridge)
    client = Ros2BackendClient(bridge_python=Path(sys.executable), config=write_config(tmp_path))

    client.start()
    process = client.process
    client.close()
    client.close()

    assert marker.read_text(encoding="utf-8") == "stopped"
    assert process is not None
    assert process.poll() == 0


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ('print("not-json", flush=True)', "Protocol message is not valid JSON"),
        ('print(json.dumps({"type": "error", "message": "bridge exploded"}), flush=True)', "bridge exploded"),
        ("sys.exit(0)", "stdout closed"),
        ('print(json.dumps({"type": "status", "state": "ready"}), flush=True)\nsys.stderr.write("noisy child\\n")\nsys.stderr.flush()\nsys.exit(7)', "exited with status 7.*noisy child"),
    ],
)
def test_protocol_errors_bridge_errors_eof_stderr_and_non_zero_exit_surface_without_deadlock(tmp_path, monkeypatch, body, match):
    bridge = write_fake_bridge(tmp_path, body)
    monkeypatch.setattr(ros2_backend, "BRIDGE_SCRIPT", bridge)
    client = Ros2BackendClient(bridge_python=Path(sys.executable), config=write_config(tmp_path))

    client.start()
    with pytest.raises(RuntimeError, match=match):
        client.next_frame(timeout=1.0)
    client.close()


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (
            """
            print(json.dumps({"type": "status", "state": "ready"}), flush=True)
            time.sleep(0.2)
            print(json.dumps({"type": "error", "message": "async bridge error"}), flush=True)
            for line in sys.stdin:
                if json.loads(line).get("type") == "stop":
                    break
            """,
            "async bridge error",
        ),
        (
            """
            print(json.dumps({"type": "status", "state": "ready"}), flush=True)
            time.sleep(0.2)
            sys.exit(6)
            """,
            "bridge exited with status 6",
        ),
    ],
)
def test_next_frame_wakes_promptly_for_asynchronous_backend_errors(tmp_path, monkeypatch, body, match):
    bridge = write_fake_bridge(tmp_path, body)
    monkeypatch.setattr(ros2_backend, "BRIDGE_SCRIPT", bridge)
    client = Ros2BackendClient(bridge_python=Path(sys.executable), config=write_config(tmp_path))

    client.start()
    started = time.monotonic()
    with pytest.raises(RuntimeError, match=match):
        client.next_frame(timeout=2.0)
    elapsed = time.monotonic() - started
    client.close()

    assert elapsed < 0.8



def test_stdout_eof_from_live_bridge_surfaces_bridge_failure_not_generic_timeout(tmp_path, monkeypatch):
    bridge = write_fake_bridge(
        tmp_path,
        """
        import os
        print(json.dumps({"type": "status", "state": "ready"}), flush=True)
        os.close(sys.stdout.fileno())
        time.sleep(2.0)
        """
    )
    monkeypatch.setattr(ros2_backend, "BRIDGE_SCRIPT", bridge)
    client = Ros2BackendClient(bridge_python=Path(sys.executable), config=write_config(tmp_path))

    client.start()
    with pytest.raises(RuntimeError, match="stdout closed before bridge exited"):
        client.next_frame(timeout=1.0)
    client.close()

def test_timeout_raises_timeouterror_without_killing_live_bridge(tmp_path, monkeypatch):
    bridge = write_fake_bridge(
        tmp_path,
        """
        print(json.dumps({"type": "status", "state": "ready"}), flush=True)
        for line in sys.stdin:
            if json.loads(line).get("type") == "stop":
                break
        """,
    )
    monkeypatch.setattr(ros2_backend, "BRIDGE_SCRIPT", bridge)
    client = Ros2BackendClient(bridge_python=Path(sys.executable), config=write_config(tmp_path))

    client.start()
    with pytest.raises(TimeoutError):
        client.next_frame(timeout=0.05)
    assert client.process is not None and client.process.poll() is None
    client.close()


def test_bounded_complete_frame_queue_drops_oldest_and_records_drop(tmp_path, monkeypatch):
    events = []
    for timestamp in range(1, 10):
        events.extend(complete_events(float(timestamp)))
    bridge = write_fake_bridge(
        tmp_path,
        """
        print(json.dumps({"type": "status", "state": "ready"}), flush=True)
        for event in __EVENTS__:
            print(json.dumps(event), flush=True)
        for line in sys.stdin:
            if json.loads(line).get("type") == "stop":
                break
        """.replace("__EVENTS__", repr(events)),
    )
    monkeypatch.setattr(ros2_backend, "BRIDGE_SCRIPT", bridge)
    client = Ros2BackendClient(bridge_python=Path(sys.executable), config=write_config(tmp_path))

    client.start()
    time.sleep(0.5)
    first = client.next_frame(timeout=1.0)
    client.close()

    assert first.timestamp == pytest.approx(2.0)
    assert client.dropped_complete_frames == 1
