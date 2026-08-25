"""Pure tests for ROS 2 official Piper status safety latching."""

import base64
import json
from pathlib import Path
import sys

from types import SimpleNamespace

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from ros2_safety import PiperStatusLatch


def status(
    *,
    ctrl_mode=1,
    arm_status=0,
    mode_feedback=1,
    motion_status=1,
    err_code=0,
):
    return SimpleNamespace(
        ctrl_mode=ctrl_mode,
        arm_status=arm_status,
        mode_feedback=mode_feedback,
        motion_status=motion_status,
        err_code=err_code,
    )


def assert_fault(latch: PiperStatusLatch, match: str, *, now: float = 10.1) -> None:
    with pytest.raises(RuntimeError, match=match):
        latch.assert_joint_ready(now=now)


def test_joint_ready_requires_recent_healthy_status_from_both_sides_and_allows_motion_in_progress():
    latch = PiperStatusLatch(status_watchdog_s=0.5)
    healthy = status(ctrl_mode=1, arm_status=0, mode_feedback=1, motion_status=1, err_code=0)

    latch.update("left", healthy, received_at=10.0)
    latch.update("right", healthy, received_at=10.0)

    latch.assert_joint_ready(now=10.1)
    assert latch.fault_reason is None


def test_eef_ready_accepts_recent_healthy_mode_zero_status_from_both_sides():
    latch = PiperStatusLatch(status_watchdog_s=0.5, expected_mode_feedback=0)
    healthy = status(ctrl_mode=1, arm_status=0, mode_feedback=0, motion_status=1, err_code=0)

    latch.update("left", healthy, received_at=10.0)
    latch.update("right", healthy, received_at=10.0)

    latch.assert_joint_ready(now=10.1)
    assert latch.fault_reason is None


def test_eef_ready_permanently_rejects_mode_one_feedback():
    latch = PiperStatusLatch(status_watchdog_s=0.5, expected_mode_feedback=0)

    latch.update("left", status(mode_feedback=1), received_at=10.0)
    latch.update("right", status(mode_feedback=0), received_at=10.0)
    assert_fault(latch, "left mode_feedback=1")

    latch.update("left", status(mode_feedback=0), received_at=10.1)
    latch.update("right", status(mode_feedback=0), received_at=10.1)

    assert_fault(latch, "left mode_feedback=1", now=10.2)
    assert latch.fault_reason == "left mode_feedback=1"


@pytest.mark.parametrize("expected_mode_feedback", [True, 1.0, "1", -1, 2])
def test_latch_rejects_invalid_expected_mode_feedback(expected_mode_feedback):
    with pytest.raises(ValueError, match="expected_mode_feedback must be integer 0 or 1"):
        PiperStatusLatch(status_watchdog_s=0.5, expected_mode_feedback=expected_mode_feedback)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("arm_status", 1, "left arm_status=1"),
        ("err_code", 64, "left err_code=64"),
        ("ctrl_mode", 0, "left ctrl_mode=0"),
        ("mode_feedback", 0, "left mode_feedback=0"),
    ],
)
def test_joint_ready_rejects_each_status_fault_independently(field: str, value: int, match: str):
    latch = PiperStatusLatch(status_watchdog_s=0.5)
    faulty = status()
    setattr(faulty, field, value)

    latch.update("left", faulty, received_at=10.0)
    latch.update("right", status(), received_at=10.0)

    assert_fault(latch, match)
    assert latch.fault_reason == match


def test_joint_ready_rejects_missing_side_status():
    latch = PiperStatusLatch(status_watchdog_s=0.5)

    latch.update("left", status(), received_at=10.0)

    assert_fault(latch, "right status missing")
    assert latch.fault_reason == "right status missing"


def test_joint_ready_rejects_stale_status():
    latch = PiperStatusLatch(status_watchdog_s=0.5)

    latch.update("left", status(), received_at=10.0)
    latch.update("right", status(), received_at=10.0)

    assert_fault(latch, "left status stale", now=10.6)
    assert latch.fault_reason == "left status stale"


def test_latched_fault_is_permanent_even_after_later_healthy_status():
    latch = PiperStatusLatch(status_watchdog_s=0.5)

    latch.update("left", status(arm_status=2), received_at=10.0)
    latch.update("right", status(), received_at=10.0)
    assert_fault(latch, "left arm_status=2")

    latch.update("left", status(), received_at=10.1)
    latch.update("right", status(), received_at=10.1)

    assert_fault(latch, "left arm_status=2", now=10.2)
    assert latch.fault_reason == "left arm_status=2"


class _FakeClock:
    def now(self):
        return self

    def to_msg(self):
        return SimpleNamespace(sec=0, nanosec=0)


class _CaptureStream:
    def __init__(self):
        self.messages = []

    def write(self, line):
        self.messages.append(json.loads(line))

    def flush(self):
        pass


class _FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _FakeProcess:
    returncode = None

    def poll(self):
        return None


def _bridge_config():
    from ros2_bridge_process import _load_config
    from ros2_contract import BridgeContract

    config_path = Path(__file__).resolve().parents[1] / "ros2_piper_dual.yaml"
    config = _load_config(str(config_path))
    return config, BridgeContract.from_mapping(config)


def _make_bridge(monkeypatch, *, dry_run: bool, publish_actions: bool = True, eef_control: bool = False):
    from ros2_bridge_process import LockedJsonLineWriter
    from ros2_bridge_process import PiperRos2Bridge

    config, contract = _bridge_config()
    capture = SimpleNamespace(stream=_CaptureStream(), subscriptions=[], publishers=[], contract=contract)

    def create_subscription(self, msg_type, topic, callback, qos_depth):
        capture.subscriptions.append(SimpleNamespace(msg_type=msg_type.__name__, topic=topic, callback=callback, qos_depth=qos_depth))
        return object()

    def create_publisher(self, msg_type, topic, qos_depth):
        publisher = _FakePublisher()
        capture.publishers.append(SimpleNamespace(msg_type=msg_type.__name__, topic=topic, qos_depth=qos_depth, publisher=publisher))
        return publisher

    monkeypatch.setattr("ros2_bridge_process.PiperRos2Bridge.create_subscription", create_subscription)
    monkeypatch.setattr("ros2_bridge_process.PiperRos2Bridge.create_publisher", create_publisher)
    monkeypatch.setattr("ros2_bridge_process.PiperRos2Bridge.create_timer", lambda self, period, callback: object())
    monkeypatch.setattr("ros2_bridge_process.PiperRos2Bridge.get_clock", lambda self: _FakeClock())
    monkeypatch.setattr("ros2_bridge_process.Node.__init__", lambda self, *_args, **_kwargs: None)
    monkeypatch.setattr("ros2_bridge_process.time.monotonic", lambda: 20.0)
    monkeypatch.setattr("ros2_bridge_process.validate_profile_graph", lambda *_args, **_kwargs: None)

    node = PiperRos2Bridge(
        writer=LockedJsonLineWriter(capture.stream),
        dry_run=dry_run,
        publish_actions=publish_actions,
        contract=contract,
        config=config,
        eef_control=eef_control,
    )
    return node, capture


def _subscription(capture, topic: str):
    for subscription in capture.subscriptions:
        if subscription.topic == topic:
            return subscription
    raise AssertionError(f"missing subscription for {topic}")


def _status_msg(**overrides):
    from piper_msgs.msg import PiperStatusMsg

    values = {"ctrl_mode": 1, "arm_status": 0, "mode_feedback": 1, "motion_status": 1, "err_code": 0}
    values.update(overrides)
    return PiperStatusMsg(**values)


def _joint_message(timestamp: float = 21.0):
    seconds = int(timestamp)
    nanoseconds = int(round((timestamp - seconds) * 1_000_000_000))
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=seconds, nanosec=nanoseconds)),
        position=[float(index) for index in range(7)],
    )


def _write_backend_config(tmp_path):
    config = tmp_path / "bridge-config.json"
    config.write_text('{"mode":"test"}', encoding="utf-8")
    return config


def _complete_sensor_events(timestamp: float):
    from frame_synchronizer import IMAGE_SENSORS
    from frame_synchronizer import JOINT_SENSORS

    jpeg = base64.b64encode(b"jpeg").decode("ascii")
    events = [
        {"type": "sensor", "sensor": sensor, "timestamp": timestamp, "jpeg_b64": jpeg}
        for sensor in IMAGE_SENSORS
    ]
    events.extend(
        {"type": "sensor", "sensor": sensor, "timestamp": timestamp, "values": [float(offset + index) for index in range(7)]}
        for offset, sensor in enumerate(JOINT_SENSORS)
    )
    return events


def test_bridge_eef_control_uses_mode_zero_status_latch_and_publishes_after_ready(monkeypatch):
    node, capture = _make_bridge(monkeypatch, dry_run=False, publish_actions=True, eef_control=True)

    _subscription(capture, capture.contract.status_topics["left"]).callback(_status_msg(mode_feedback=0))
    _subscription(capture, capture.contract.status_topics["right"]).callback(_status_msg(mode_feedback=0))

    node._handle_request({"type": "publish_eef_action", "left": [0.0] * 7, "right": [0.0] * 7})

    assert node._status_latch.fault_reason is None
    assert [publisher.msg_type for publisher in capture.publishers] == ["PosCmd", "PosCmd"]
    assert [len(publisher.publisher.messages) for publisher in capture.publishers] == [1, 1]


def test_bridge_joint_control_keeps_mode_one_status_latch_and_rejects_mode_zero(monkeypatch):
    node, capture = _make_bridge(monkeypatch, dry_run=False, publish_actions=True, eef_control=False)

    _subscription(capture, capture.contract.status_topics["left"]).callback(_status_msg(mode_feedback=0))
    _subscription(capture, capture.contract.status_topics["right"]).callback(_status_msg(mode_feedback=1))

    with pytest.raises(RuntimeError, match="left mode_feedback=0"):
        node._handle_request({"type": "publish_action", "left": [0.0] * 7, "right": [0.0] * 7})

    assert [message for publisher in capture.publishers for message in publisher.publisher.messages] == []
    assert node._status_latch.fault_reason == "left mode_feedback=0"


def test_bridge_bad_piper_status_callback_emits_exactly_one_hardware_fault(monkeypatch):
    node, capture = _make_bridge(monkeypatch, dry_run=False, publish_actions=True)
    left_status = _subscription(capture, capture.contract.status_topics["left"])
    bad_status = _status_msg(arm_status=1)

    assert left_status.msg_type == "PiperStatusMsg"
    left_status.callback(bad_status)
    left_status.callback(bad_status)

    assert [message for message in capture.stream.messages if message.get("state") == "hardware_fault"] == [
        {"type": "status", "state": "hardware_fault", "metadata": {"reason": "left arm_status=1", "side": "left"}}
    ]
    assert node._status_latch.fault_reason == "left arm_status=1"


def test_bridge_live_joint_and_eef_actions_reject_before_publishing_after_fault(monkeypatch):
    joint_node, joint_capture = _make_bridge(monkeypatch, dry_run=False, publish_actions=True, eef_control=False)
    _subscription(joint_capture, joint_capture.contract.status_topics["left"]).callback(_status_msg(err_code=7))

    with pytest.raises(RuntimeError, match="left err_code=7"):
        joint_node._handle_request({"type": "publish_action", "left": [0.0] * 7, "right": [0.0] * 7})
    assert [message for publisher in joint_capture.publishers for message in publisher.publisher.messages] == []

    eef_node, eef_capture = _make_bridge(monkeypatch, dry_run=False, publish_actions=True, eef_control=True)
    _subscription(eef_capture, eef_capture.contract.status_topics["right"]).callback(_status_msg(ctrl_mode=0))

    with pytest.raises(RuntimeError, match="right ctrl_mode=0"):
        eef_node._handle_request({"type": "publish_eef_action", "left": [0.0] * 7, "right": [0.0] * 7})
    assert [message for publisher in eef_capture.publishers for message in publisher.publisher.messages] == []


def test_bridge_dry_run_has_zero_publishers_and_sensor_callbacks_continue_after_fault(monkeypatch):
    node, capture = _make_bridge(monkeypatch, dry_run=True, publish_actions=True)
    status_callback = _subscription(capture, capture.contract.status_topics["left"]).callback
    joint_topic = next(iter(capture.contract.joint_topics.keys()))
    joint_sensor = capture.contract.joint_topics[joint_topic]

    assert capture.publishers == []
    status_callback(_status_msg(mode_feedback=0))
    status_callback(_status_msg(mode_feedback=0))
    _subscription(capture, joint_topic).callback(_joint_message())

    assert [message for message in capture.stream.messages if message.get("state") == "hardware_fault"] == [
        {"type": "status", "state": "hardware_fault", "metadata": {"reason": "left mode_feedback=0", "side": "left"}}
    ]
    sensor_events = [message for message in capture.stream.messages if message.get("type") == "sensor"]
    assert sensor_events[-1]["sensor"] == joint_sensor
    assert sensor_events[-1]["values"] == pytest.approx([float(index) for index in range(7)])
    assert node._dry_run is True
    node._handle_request(
        {"type": "publish_action", "left": [0.0] * 7, "right": [0.0] * 7}
    )
    assert capture.publishers == []
    assert [
        message for message in capture.stream.messages
        if message.get("state") == "action_ignored"
    ] == [
        {
            "type": "status",
            "state": "action_ignored",
            "metadata": {"dry_run": True, "publish_actions": False},
        }
    ]


def test_backend_dry_run_records_hardware_fault_metadata_and_still_accepts_frames(tmp_path):
    from ros2_backend import Ros2BackendClient

    client = Ros2BackendClient(bridge_python=Path(sys.executable), config=_write_backend_config(tmp_path), dry_run=True)
    client.process = _FakeProcess()
    client._started = True

    client._handle_message({"type": "status", "state": "hardware_fault", "metadata": {"side": "left", "reason": "left arm_status=1"}})
    for event in _complete_sensor_events(30.0):
        client._handle_message(event)

    frame = client.next_frame(timeout=0.1)
    assert client.hardware_fault == {"side": "left", "reason": "left arm_status=1"}
    assert frame.timestamp == pytest.approx(30.0)


def test_backend_live_hardware_fault_is_fatal_and_publish_action_sends_no_request(tmp_path):
    from ros2_backend import Ros2BackendClient

    client = Ros2BackendClient(
        bridge_python=Path(sys.executable),
        config=_write_backend_config(tmp_path),
        publish_actions=True,
        dry_run=False,
    )
    client.process = _FakeProcess()
    client._started = True
    sent_requests = []
    client._send_request = sent_requests.append

    client._handle_message({"type": "status", "state": "hardware_fault", "metadata": {"side": "right", "reason": "right err_code=9"}})

    assert client.hardware_fault == {"side": "right", "reason": "right err_code=9"}
    with pytest.raises(RuntimeError, match="hardware fault: right err_code=9"):
        client.publish_action(np.zeros(14, dtype=np.float32))
    assert sent_requests == []
