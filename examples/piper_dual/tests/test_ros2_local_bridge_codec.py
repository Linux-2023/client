"""Focused tests for the local-stack bridge ROS action message shape and its wiring."""

from pathlib import Path
import sys

from builtin_interfaces.msg import Time as Stamp
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ros2_contract import BridgeContract
from ros2_local_bridge_process import LocalPiperBridge
from ros2_local_bridge_process import Writer
from ros2_local_bridge_process import _make_joint_state
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "ros2_piper_dual_local.yaml"


class FakeClock:
    def now(self):
        return self

    def to_msg(self):
        return Stamp(sec=0, nanosec=0)


class DummyStream:
    def write(self, _line):
        pass

    def flush(self):
        pass


def _load_config() -> dict:
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _construct_local_bridge(monkeypatch, config: dict):
    published: list[tuple[str, object]] = []

    class FakePublisher:
        def __init__(self, topic: str) -> None:
            self.topic = topic

        def publish(self, message) -> None:
            published.append((self.topic, message))

    monkeypatch.setattr(
        "ros2_local_bridge_process.LocalPiperBridge.create_subscription",
        lambda self, msg_type, topic, callback, qos_depth: object(),
    )
    monkeypatch.setattr(
        "ros2_local_bridge_process.LocalPiperBridge.create_publisher",
        lambda self, msg_type, topic, qos_depth: FakePublisher(topic),
    )
    monkeypatch.setattr(
        "ros2_local_bridge_process.LocalPiperBridge.create_timer", lambda self, period, callback: object()
    )
    monkeypatch.setattr("ros2_local_bridge_process.LocalPiperBridge.get_clock", lambda self: FakeClock())
    monkeypatch.setattr("ros2_local_bridge_process.Node.__init__", lambda self, *_args, **_kwargs: None)

    node = LocalPiperBridge(
        writer=Writer(DummyStream()),
        config=config,
        dry_run=False,
        publish_actions=True,
        include_eef=False,
        eef_control=False,
        eef_left_topic=None,
        eef_right_topic=None,
        eef_left_action_topic="/pos_left_cmd",
        eef_right_action_topic="/pos_right_cmd",
    )
    return node, published


def test_make_joint_state_carries_speed_and_gripper_effort():
    msg = _make_joint_state(
        [0.1] * 6 + [0.02],
        Stamp(sec=1, nanosec=2),
        joint_names=("joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
        speed_percent=100,
        gripper_effort=0.5,
    )

    assert msg.header.stamp.sec == 1
    assert msg.header.stamp.nanosec == 2
    assert msg.name == ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    assert msg.position == pytest.approx([0.1] * 6 + [0.02])
    # Non-empty trailing velocity/effort is what stops piper_ctrl_single_node from
    # silently falling back to its own 30% / 1.0 N*m defaults.
    assert msg.velocity == pytest.approx([0.0] * 6 + [100.0])
    assert msg.effort == pytest.approx([0.0] * 6 + [0.5])


def test_local_bridge_publishes_contract_speed_and_gripper_effort(monkeypatch):
    config = _load_config()
    config["bridge_contract"]["control"]["speed_percent"] = 45
    config["bridge_contract"]["control"]["gripper_effort"] = 2.25
    contract = BridgeContract.from_mapping(config)

    node, published = _construct_local_bridge(monkeypatch, config)
    node._publish_joint({"left": [0.1] * 6 + [0.02], "right": [0.2] * 6 + [0.01]})

    assert [topic for topic, _message in published] == [
        contract.joint_action_topics["left"],
        contract.joint_action_topics["right"],
    ]
    expected_positions = {
        contract.joint_action_topics["left"]: [0.1] * 6 + [0.02],
        contract.joint_action_topics["right"]: [0.2] * 6 + [0.01],
    }
    for topic, message in published:
        assert message.position == pytest.approx(expected_positions[topic])
        assert message.velocity == pytest.approx([0.0] * 6 + [45.0])
        assert message.effort == pytest.approx([0.0] * 6 + [2.25])
