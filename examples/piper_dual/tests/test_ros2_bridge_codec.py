"""Focused tests for bridge-side ROS message/request codecs."""

from pathlib import Path
import base64
import json
import math
import sys
import time

import pytest
import yaml
from builtin_interfaces.msg import Time as Stamp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ros2_bridge_process import _format_sensor_timestamp
from ros2_bridge_process import _load_config
from ros2_bridge_process import _make_joint_state
from ros2_bridge_process import build_eef_sensor_event
from ros2_bridge_process import build_image_sensor_event
from ros2_bridge_process import build_joint_sensor_event
from ros2_bridge_process import validate_action_request
from ros2_bridge_process import validate_eef_action_request

from ros2_contract import BridgeContract
from control_stack import load_profile_config
from control_stack import profile_for


class StatusMessage:
    ctrl_mode = 1
    arm_status = 0
    mode_feedback = 1
    motion_status = 0
    err_code = 0


def _complete_topic_types(contract: BridgeContract) -> dict[str, list[str]]:
    topics: dict[str, list[str]] = {}
    for topic in contract.image_topics.values():
        topics[topic] = ["sensor_msgs/msg/Image"]
    for topic in contract.joint_topics.keys():
        topics[topic] = ["sensor_msgs/msg/JointState"]
    for topic in contract.joint_action_topics.values():
        topics[topic] = ["sensor_msgs/msg/JointState"]
    for topic in contract.status_topics.values():
        topics[topic] = ["piper_msgs/msg/PiperStatusMsg"]
    for topic in contract.eef_action_topics.values():
        topics[topic] = ["piper_msgs/msg/PosCmd"]
    return topics


def _construct_test_bridge(
    monkeypatch,
    *,
    dry_run: bool,
    publish_actions: bool,
    control_stack: str = "official-ros",
    eef_control: bool = False,
):
    from ros2_bridge_process import LockedJsonLineWriter
    from ros2_bridge_process import PiperRos2Bridge

    profile, config = load_profile_config(control_stack, profile_for(control_stack).config_path)
    contract = BridgeContract.from_mapping(config)
    calls = {"subscriptions": [], "publishers": [], "published": [], "status": []}

    class DummyStream:
        def write(self, line):
            calls["status"].append(json.loads(line))

        def flush(self):
            pass

    class FakeTimer:
        pass

    class FakeClock:
        def now(self):
            return self

        def to_msg(self):
            return Stamp(sec=0, nanosec=0)

    class FakePublisher:
        def __init__(self, topic):
            self.topic = topic

        def publish(self, message):
            calls["published"].append((self.topic, message))

    def create_subscription(self, msg_type, topic, callback, qos_depth):
        calls["subscriptions"].append((msg_type.__name__, topic, qos_depth))
        return object()

    def create_publisher(self, msg_type, topic, qos_depth):
        calls["publishers"].append((msg_type.__name__, topic, qos_depth))
        return FakePublisher(topic)

    monkeypatch.setattr("ros2_bridge_process.PiperRos2Bridge.create_subscription", create_subscription)
    monkeypatch.setattr("ros2_bridge_process.PiperRos2Bridge.create_publisher", create_publisher)
    monkeypatch.setattr("ros2_bridge_process.PiperRos2Bridge.create_timer", lambda self, period, callback: FakeTimer())
    monkeypatch.setattr("ros2_bridge_process.PiperRos2Bridge.get_clock", lambda self: FakeClock())
    monkeypatch.setattr("ros2_bridge_process.Node.__init__", lambda self, *_args, **_kwargs: None)

    node = PiperRos2Bridge(
        writer=LockedJsonLineWriter(DummyStream()),
        dry_run=dry_run,
        publish_actions=publish_actions,
        contract=contract,
        config=config,
        control_stack=control_stack,
        eef_control=eef_control,
    )
    node.get_topic_names_and_types = lambda: [(topic, list(types)) for topic, types in _complete_topic_types(contract).items()]
    selected_action_topics = contract.eef_action_topics.values() if eef_control else contract.joint_action_topics.values()
    node.count_subscribers = lambda topic: 1 if topic in selected_action_topics else 0
    node.get_node_names = lambda: []
    return node, contract, calls


def _mark_status_ready(node) -> None:
    now = time.monotonic()
    expected_mode = node._status_latch.expected_mode_feedback
    ready_status = StatusMessage()
    ready_status.mode_feedback = expected_mode
    node._status_latch.update("left", ready_status, received_at=now)
    node._status_latch.update("right", ready_status, received_at=now)


class Header:
    def __init__(self, sec, nanosec):
        self.stamp = Stamp(sec=sec, nanosec=nanosec)


class Message:
    def __init__(self, *, sec=0, nanosec=0, position=None):
        self.header = Header(sec, nanosec)
        self.position = position or []


class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Quaternion:
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w


class Pose:
    def __init__(self, position, orientation):
        self.position = Vector3(*position)
        self.orientation = Quaternion(*orientation)


class PoseMessage:
    def __init__(self, *, sec=0, nanosec=0, position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0)):
        self.header = Header(sec, nanosec)
        self.pose = Pose(position, orientation)


def test_make_joint_state_uses_official_names_speed_and_effort():
    msg = _make_joint_state(
        [0.1] * 6 + [0.02],
        Stamp(sec=1, nanosec=2),
        joint_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"),
        speed_percent=30,
        gripper_effort=0.5,
    )

    assert msg.name == ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
    assert msg.position == pytest.approx([0.1] * 6 + [0.02])
    assert msg.velocity == pytest.approx([0.0] * 6 + [30.0])
    assert msg.effort == pytest.approx([0.0] * 6 + [0.5])


def test_bridge_joint_only_default_ignores_yaml_eef_mappings_and_include_eef_adds_pose_streams(monkeypatch):
    config_path = Path(__file__).resolve().parents[1] / "ros2_piper_dual.yaml"
    config = _load_config(str(config_path))
    contract = BridgeContract.from_mapping(config)
    calls = {"subscriptions": [], "publishers": []}
    image_callback_sensors: list[str] = []
    joint_callback_sensors: list[str] = []
    status_callback_sensors: list[str] = []
    eef_callback_sensors: list[str] = []

    class FakeTimer:
        pass

    class FakeClock:
        def now(self):
            return self

        def to_msg(self):
            return Stamp(sec=0, nanosec=0)

    def make_callback(sensor: str):
        def callback(_message):
            return None

        callback._sensor = sensor  # type: ignore[attr-defined]
        return callback

    def create_subscription(self, msg_type, topic, callback, qos_depth):
        calls["subscriptions"].append((msg_type.__name__, topic, qos_depth))
        sensor = getattr(callback, "_sensor", None)
        if topic in contract.image_topics.values():
            image_callback_sensors.append(sensor)
        elif topic in contract.joint_topics.keys():
            joint_callback_sensors.append(sensor)
        elif topic in contract.status_topics.values():
            status_callback_sensors.append(sensor)
        elif msg_type.__name__ == "PoseStamped":
            eef_callback_sensors.append(sensor)
        return object()

    monkeypatch.setattr("ros2_bridge_process.PiperRos2Bridge.create_subscription", create_subscription)
    monkeypatch.setattr(
        "ros2_bridge_process.PiperRos2Bridge.create_publisher",
        lambda self, msg_type, topic, qos_depth: calls["publishers"].append((msg_type.__name__, topic, qos_depth)) or object(),
    )
    monkeypatch.setattr("ros2_bridge_process.PiperRos2Bridge.create_timer", lambda self, period, callback: FakeTimer())
    monkeypatch.setattr("ros2_bridge_process.PiperRos2Bridge.get_clock", lambda self: FakeClock())
    monkeypatch.setattr("ros2_bridge_process.Node.__init__", lambda self, *_args, **_kwargs: None)
    monkeypatch.setattr("ros2_bridge_process.PiperRos2Bridge._make_image_callback", lambda self, sensor: make_callback(sensor))
    monkeypatch.setattr("ros2_bridge_process.PiperRos2Bridge._make_joint_callback", lambda self, sensor: make_callback(sensor))
    monkeypatch.setattr("ros2_bridge_process.PiperRos2Bridge._make_eef_callback", lambda self, sensor: make_callback(sensor))

    from ros2_bridge_process import LockedJsonLineWriter
    from ros2_bridge_process import PiperRos2Bridge

    class DummyStream:
        def write(self, _line):
            pass

        def flush(self):
            pass

    def construct_bridge(**kwargs):
        return PiperRos2Bridge(
            writer=LockedJsonLineWriter(DummyStream()),
            dry_run=True,
            publish_actions=True,
            contract=contract,
            config=config,
            eef_control=False,
            **kwargs,
        )

    construct_bridge()

    assert calls["publishers"] == []
    assert {topic for _, topic, _ in calls["subscriptions"]} == (
        set(contract.image_topics.values()) | set(contract.joint_topics.keys()) | set(contract.status_topics.values())
    )
    assert image_callback_sensors == list(contract.image_topics.keys())
    assert joint_callback_sensors == list(contract.joint_topics.values())
    assert status_callback_sensors == list(contract.status_topics.keys())
    assert eef_callback_sensors == []
    assert len(image_callback_sensors) + len(joint_callback_sensors) + len(eef_callback_sensors) == 7
    assert all(msg_type != "PoseStamped" for msg_type, _topic, _qos in calls["subscriptions"])

    calls["subscriptions"].clear()
    image_callback_sensors.clear()
    joint_callback_sensors.clear()
    status_callback_sensors.clear()
    eef_callback_sensors.clear()

    construct_bridge(include_eef=True)

    pose_subscriptions = [(topic, qos_depth) for msg_type, topic, qos_depth in calls["subscriptions"] if msg_type == "PoseStamped"]
    assert len(pose_subscriptions) == 2
    assert {topic for topic, _qos_depth in pose_subscriptions} == set(contract.eef_topics.values())
    assert eef_callback_sensors == ["eef_puppet_left", "eef_puppet_right"]
    assert len(image_callback_sensors) + len(joint_callback_sensors) + len(eef_callback_sensors) == 9

    calls["subscriptions"].clear()
    image_callback_sensors.clear()
    joint_callback_sensors.clear()
    status_callback_sensors.clear()
    eef_callback_sensors.clear()

    construct_bridge(include_eef=True, eef_left_topic="/override/eef_left", eef_right_topic="/override/eef_right")

    override_pose_topics = {topic for msg_type, topic, _qos_depth in calls["subscriptions"] if msg_type == "PoseStamped"}
    assert override_pose_topics == {"/override/eef_left", "/override/eef_right"}


def test_bridge_live_publishers_are_deferred_until_first_valid_action_graph_and_status_ready(monkeypatch):
    node, contract, calls = _construct_test_bridge(monkeypatch, dry_run=False, publish_actions=True)
    assert calls["publishers"] == []

    now = time.monotonic()
    node._status_latch.update("left", StatusMessage(), received_at=now)
    node._status_latch.update("right", StatusMessage(), received_at=now)
    node._handle_request({"type": "publish_action", "left": [0.0] * 7, "right": [0.0] * 7})

    assert calls["publishers"] == [
        ("JointState", contract.joint_action_topics["left"], 1),
        ("JointState", contract.joint_action_topics["right"], 1),
    ]
    assert [topic for topic, _message in calls["published"]] == [
        contract.joint_action_topics["left"],
        contract.joint_action_topics["right"],
    ]


def test_bridge_live_readiness_failure_latches_without_creating_publishers(monkeypatch):
    node, contract, calls = _construct_test_bridge(monkeypatch, dry_run=False, publish_actions=True)
    topics = _complete_topic_types(contract)
    topics[contract.joint_action_topics["left"]] = ["std_msgs/msg/Bool"]
    node.get_topic_names_and_types = lambda: [(topic, list(types)) for topic, types in topics.items()]
    now = time.monotonic()
    node._status_latch.update("left", StatusMessage(), received_at=now)
    node._status_latch.update("right", StatusMessage(), received_at=now)

    with pytest.raises(RuntimeError, match="permanently disabled"):
        node._handle_request({"type": "publish_action", "left": [0.0] * 7, "right": [0.0] * 7})
    assert calls["publishers"] == []

    node.get_topic_names_and_types = lambda: [(topic, list(types)) for topic, types in _complete_topic_types(contract).items()]
    with pytest.raises(RuntimeError, match="permanently disabled"):
        node._handle_request({"type": "publish_action", "left": [0.0] * 7, "right": [0.0] * 7})
    assert calls["publishers"] == []



def test_bridge_rechecks_status_before_every_live_joint_publish(monkeypatch):
    node, _contract, calls = _construct_test_bridge(monkeypatch, dry_run=False, publish_actions=True)
    _mark_status_ready(node)
    node._handle_request({"type": "publish_action", "left": [0.0] * 7, "right": [0.0] * 7})

    class FaultStatus(StatusMessage):
        arm_status = 1

    node._status_latch.update("left", FaultStatus(), received_at=time.monotonic())

    with pytest.raises(RuntimeError, match="permanently disabled"):
        node._handle_request({"type": "publish_action", "left": [0.1] * 7, "right": [0.1] * 7})
    assert len(calls["published"]) == 2

    node._status_latch.update("left", StatusMessage(), received_at=time.monotonic())
    with pytest.raises(RuntimeError, match="permanently disabled"):
        node._handle_request({"type": "publish_action", "left": [0.2] * 7, "right": [0.2] * 7})
    assert len(calls["published"]) == 2


def test_bridge_eef_live_publishers_require_profile_graph_and_status_ready(monkeypatch):
    node, _contract, calls = _construct_test_bridge(
        monkeypatch,
        dry_run=False,
        publish_actions=True,
        eef_control=True,
    )
    node.count_subscribers = lambda _topic: 0
    _mark_status_ready(node)

    with pytest.raises(RuntimeError, match="exactly one external subscriber"):
        node._handle_request({"type": "publish_eef_action", "left": [0.0] * 7, "right": [0.0] * 7})
    assert calls["publishers"] == []
    assert calls["published"] == []


def test_bridge_eef_graph_uses_eef_action_subscribers_not_joint_subscribers(monkeypatch):
    node, contract, calls = _construct_test_bridge(
        monkeypatch,
        dry_run=False,
        publish_actions=True,
        eef_control=True,
    )
    node.count_subscribers = lambda topic: 1 if topic in contract.joint_action_topics.values() else 0
    _mark_status_ready(node)

    with pytest.raises(RuntimeError, match="exactly one external subscriber"):
        node._handle_request({"type": "publish_eef_action", "left": [0.0] * 7, "right": [0.0] * 7})
    assert calls["publishers"] == []
    assert calls["published"] == []


def test_bridge_eef_graph_rejects_selected_action_topic_wrong_type_before_publishers(monkeypatch):
    node, contract, calls = _construct_test_bridge(
        monkeypatch,
        dry_run=False,
        publish_actions=True,
        eef_control=True,
    )
    topics = _complete_topic_types(contract)
    topics[contract.eef_action_topics["left"]] = ["sensor_msgs/msg/JointState"]
    topics[contract.eef_action_topics["right"]] = ["sensor_msgs/msg/JointState"]
    node.get_topic_names_and_types = lambda: [(topic, list(types)) for topic, types in topics.items()]
    _mark_status_ready(node)

    with pytest.raises(RuntimeError, match="piper_msgs/msg/PosCmd"):
        node._handle_request({"type": "publish_eef_action", "left": [0.0] * 7, "right": [0.0] * 7})
    assert calls["publishers"] == []
    assert calls["published"] == []


def test_bridge_eef_live_action_publishes_after_eef_action_subscribers_pass(monkeypatch):
    node, contract, calls = _construct_test_bridge(
        monkeypatch,
        dry_run=False,
        publish_actions=True,
        eef_control=True,
    )
    _mark_status_ready(node)

    node._handle_request({"type": "publish_eef_action", "left": [0.0] * 7, "right": [0.0] * 7})

    assert calls["publishers"] == [
        ("PosCmd", contract.eef_action_topics["left"], 1),
        ("PosCmd", contract.eef_action_topics["right"], 1),
    ]
    assert [topic for topic, _message in calls["published"]] == [
        contract.eef_action_topics["left"],
        contract.eef_action_topics["right"],
    ]

def test_bridge_constructor_rejects_config_identity_mismatch(monkeypatch):
    from ros2_bridge_process import LockedJsonLineWriter
    from ros2_bridge_process import PiperRos2Bridge

    _profile, config = load_profile_config("official-ros", profile_for("official-ros").config_path)
    contract = BridgeContract.from_mapping(config)

    class DummyStream:
        def write(self, _line):
            pass

        def flush(self):
            pass

    monkeypatch.setattr("ros2_bridge_process.Node.__init__", lambda self, *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="control_stack_id"):
        PiperRos2Bridge(
            writer=LockedJsonLineWriter(DummyStream()),
            dry_run=True,
            publish_actions=False,
            contract=contract,
            config=config,
            control_stack="local-ros",
        )


def test_parse_args_accepts_control_stack_choice_and_include_eef_flag():
    from ros2_bridge_process import _parse_args

    args = _parse_args(["--control-stack", "direct-sdk", "--include-eef"])

    assert args.control_stack == "direct-sdk"
    assert args.include_eef is True

def test_format_sensor_timestamp_preserves_ros_header_stamp_as_float_seconds():
    assert _format_sensor_timestamp(Stamp(sec=123, nanosec=456_789_012)) == pytest.approx(123.456789012)


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


def test_build_eef_sensor_event_converts_pose_stamped_quaternion_to_xyz_rpy():
    half_yaw = math.pi / 4.0
    message = PoseMessage(
        sec=4,
        nanosec=500_000_000,
        position=(0.1, -0.2, 0.3),
        orientation=(0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)),
    )

    event = build_eef_sensor_event("eef_puppet_left", message)

    assert event["type"] == "sensor"
    assert event["sensor"] == "eef_puppet_left"
    assert event["timestamp"] == pytest.approx(4.5)
    assert event["values"] == pytest.approx([0.1, -0.2, 0.3, 0.0, 0.0, math.pi / 2.0])


@pytest.mark.parametrize(
    "message",
    [
        PoseMessage(position=(math.nan, 0.0, 0.0)),
        PoseMessage(orientation=(0.0, 0.0, 0.0, 0.0)),
        PoseMessage(orientation=(0.0, math.inf, 0.0, 1.0)),
    ],
)
def test_build_eef_sensor_event_rejects_non_finite_values_and_zero_quaternion(message):
    with pytest.raises(ValueError):
        build_eef_sensor_event("eef_puppet_left", message)


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


def test_validate_eef_action_request_accepts_two_seven_value_pose_vectors():
    request = {
        "type": "publish_eef_action",
        "left": [0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.04],
        "right": [0.4, 0.5, 0.6, -0.1, 0.0, 0.3, 0.05],
    }

    assert validate_eef_action_request(request) == {"left": request["left"], "right": request["right"]}


@pytest.mark.parametrize(
    "request_data",
    [
        {"type": "publish_eef_action", "left": [0.0] * 6, "right": [0.0] * 7},
        {"type": "publish_eef_action", "left": [0.0] * 7, "right": [0.0] * 8},
        {"type": "publish_eef_action", "left": [0.0] * 7, "right": [0.0, 0.0, 0.0, 0.0, 0.0, float("nan"), 0.0]},
    ],
)
def test_validate_eef_action_request_rejects_invalid_pose_vectors(request_data):
    with pytest.raises(ValueError):
        validate_eef_action_request(request_data)


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
    assert [camera["name"] for camera in config["cameras"]] == ["cam_high", "cam_left_wrist", "cam_right_wrist"]
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
    config_path.write_text(f"!!python/object/apply:os.system ['touch {marker_path}']\n", encoding="utf-8")

    with pytest.raises(Exception):
        _load_config(str(config_path))
    assert not marker_path.exists()


def test_main_initializes_rclpy_before_constructing_node_and_shuts_down(monkeypatch):
    events = []
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "ros2_piper_dual.yaml").read_text(encoding="utf-8"))

    class FakeNode:
        def __init__(self, **_kwargs):
            events.append("node")

        def stop_requested(self):
            return True

        def destroy_node(self):
            events.append("destroy")

    class FakeReader:
        def start(self):
            events.append("reader")

    monkeypatch.setattr("ros2_bridge_process._load_config", lambda _path: config)
    monkeypatch.setattr("ros2_bridge_process.rclpy.init", lambda **_kwargs: events.append("init"))
    monkeypatch.setattr("ros2_bridge_process.rclpy.ok", lambda: True)
    monkeypatch.setattr("ros2_bridge_process.rclpy.shutdown", lambda: events.append("shutdown"))
    monkeypatch.setattr("ros2_bridge_process.PiperRos2Bridge", FakeNode)
    monkeypatch.setattr("ros2_bridge_process.threading.Thread", lambda **_kwargs: FakeReader())

    from ros2_bridge_process import main

    assert main(["--dry-run"]) == 0
    assert events == ["init", "node", "reader", "destroy", "shutdown"]
