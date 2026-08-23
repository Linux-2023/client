"""Focused integration tests for the three Piper control-stack profiles."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control_stack import load_profile_config
from control_stack import profile_for
from ros2_contract import BridgeContract
from ros2_contract import EndpointPlan
from ros2_contract import validate_profile_graph
from test_ros2_bridge_codec import _construct_test_bridge


STACK_IDS = ("local-ros", "official-ros", "direct-sdk")
EXPECTED_SPEED = {"local-ros": 30, "official-ros": 30, "direct-sdk": 100}
EXPECTED_JOINT_NAMES = {
    "local-ros": ("joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
    "official-ros": ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"),
    "direct-sdk": ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"),
}
EXPECTED_ACTION_TOPICS = {
    "local-ros": ("/joint_left_states", "/joint_right_states"),
    "official-ros": ("/joint_ctrl_cmd_left", "/joint_ctrl_cmd_right"),
    "direct-sdk": ("/direct_sdk/joint_cmd_left", "/direct_sdk/joint_cmd_right"),
}
EXPECTED_JOINT_PREFIXES = {
    "local-ros": ("/puppet/joint_left", "/puppet/joint_right", "/master/joint_left", "/master/joint_right"),
    "official-ros": ("/joint_left", "/joint_right", "/joint_states_ctrl_left", "/joint_states_ctrl_right"),
    "direct-sdk": (
        "/direct_sdk/joint_left",
        "/direct_sdk/joint_right",
        "/direct_sdk/joint_ctrl_left",
        "/direct_sdk/joint_ctrl_right",
    ),
}
EXPECTED_EEF_TOPICS = {
    "local-ros": ("/puppet/end_pose_left", "/puppet/end_pose_right"),
    "official-ros": ("/end_pose_stamped_left", "/end_pose_stamped_right"),
    "direct-sdk": (),
}


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


class _FakeGraphNode:
    def __init__(self, *, topics: dict[str, list[str]], subscriber_counts: dict[str, int] | None = None, node_names: list[str] | None = None) -> None:
        self._topics = topics
        self._subscriber_counts = subscriber_counts or {}
        self._node_names = node_names or []

    def get_topic_names_and_types(self):
        return [(topic, list(types)) for topic, types in self._topics.items()]

    def count_subscribers(self, topic: str) -> int:
        return self._subscriber_counts.get(topic, 0)

    def get_node_names(self):
        return list(self._node_names)




def test_profiles_have_exact_three_stack_contracts_speed_and_action_topics() -> None:
    for stack_id in STACK_IDS:
        profile, mapping = load_profile_config(stack_id, profile_for(stack_id).config_path)
        contract = BridgeContract.from_mapping(mapping)

        assert profile.stack_id == stack_id
        assert contract.control.speed_percent == EXPECTED_SPEED[stack_id]
        assert contract.joint_names == EXPECTED_JOINT_NAMES[stack_id]
        assert profile.action_topics == EXPECTED_ACTION_TOPICS[stack_id]
        assert contract.image_topics == {
            "cam_high": "/camera_f_l/color/image_raw",
            "cam_left_wrist": "/camera_l/color/image_raw",
            "cam_right_wrist": "/camera_r/color/image_raw",
        }
        assert tuple(contract.joint_topics.keys()) == EXPECTED_JOINT_PREFIXES[stack_id]
        assert tuple(contract.joint_topics.values()) == ("puppet_left", "puppet_right", "master_left", "master_right")
        assert tuple(contract.eef_topics.keys()) == (("left", "right") if EXPECTED_EEF_TOPICS[stack_id] else ())
        assert tuple(contract.eef_topics.values()) == EXPECTED_EEF_TOPICS[stack_id]

        dry_run_plan = EndpointPlan.for_mode(
            contract,
            dry_run=True,
            publish_actions=False,
            eef_control=False,
            include_eef=False,
        )
        assert dry_run_plan.image_subscriptions == tuple(contract.image_topics.items())
        assert len(dry_run_plan.image_subscriptions) == 3
        assert len(dry_run_plan.joint_subscriptions) == 4
        assert len(dry_run_plan.status_subscriptions) == 2
        assert dry_run_plan.action_publishers == ()


@pytest.mark.parametrize("stack_id", STACK_IDS)
def test_dry_run_bridge_constructs_real_node_zero_publishers_and_ignores_valid_action(monkeypatch, stack_id: str) -> None:
    profile = profile_for(stack_id)
    node, contract, calls = _construct_test_bridge(
        monkeypatch,
        dry_run=True,
        publish_actions=True,
        control_stack=stack_id,
    )

    assert calls["publishers"] == []

    node._handle_request({"type": "publish_action", "left": [0.0] * 7, "right": [0.0] * 7})

    assert calls["publishers"] == []
    assert calls["published"] == []
    assert [message for message in calls["status"] if message.get("state") == "action_ignored"] == [
        {
            "type": "status",
            "state": "action_ignored",
            "metadata": {"dry_run": True, "publish_actions": False},
        }
    ]
    assert contract.joint_action_topics == {
        "left": profile.action_topics[0],
        "right": profile.action_topics[1],
    }


@pytest.mark.parametrize("stack_id", STACK_IDS)
def test_profile_graph_guards_required_forbidden_action_types_subscribers_and_direct_adapter(stack_id: str) -> None:
    profile, mapping = load_profile_config(stack_id, profile_for(stack_id).config_path)
    contract = BridgeContract.from_mapping(mapping)
    topics = _complete_topic_types(contract)
    node_names = [profile.direct_adapter_node] if profile.direct_adapter_node is not None else []
    joint_action_counts = {topic: 1 for topic in contract.joint_action_topics.values()}
    eef_action_counts = {topic: 1 for topic in contract.eef_action_topics.values()}

    validate_profile_graph(
        _FakeGraphNode(topics=topics, subscriber_counts=joint_action_counts, node_names=node_names),
        profile,
        contract,
        selected_action_types={topic: "sensor_msgs/msg/JointState" for topic in contract.joint_action_topics.values()},
    )

    missing_required = dict(topics)
    missing_required.pop(profile.required_topics[0])
    with pytest.raises(ValueError, match="missing required topic"):
        validate_profile_graph(
            _FakeGraphNode(topics=missing_required, subscriber_counts=joint_action_counts, node_names=node_names),
            profile,
            contract,
        )

    forbidden_present = dict(topics)
    forbidden_present[profile.forbidden_topics[0]] = ["sensor_msgs/msg/JointState"]
    with pytest.raises(ValueError, match="found forbidden topic"):
        validate_profile_graph(
            _FakeGraphNode(topics=forbidden_present, subscriber_counts=joint_action_counts, node_names=node_names),
            profile,
            contract,
        )

    bad_joint_type = dict(topics)
    bad_joint_type[contract.joint_action_topics["left"]] = ["std_msgs/msg/Bool"]
    with pytest.raises(ValueError, match="joint action.*sensor_msgs/msg/JointState"):
        validate_profile_graph(
            _FakeGraphNode(topics=bad_joint_type, subscriber_counts=joint_action_counts, node_names=node_names),
            profile,
            contract,
        )

    bad_joint_subscribers = dict(joint_action_counts)
    bad_joint_subscribers[contract.joint_action_topics["left"]] = 2
    with pytest.raises(ValueError, match="exactly one external subscriber; found 2"):
        validate_profile_graph(
            _FakeGraphNode(topics=topics, subscriber_counts=bad_joint_subscribers, node_names=node_names),
            profile,
            contract,
        )

    validate_profile_graph(
        _FakeGraphNode(topics=topics, subscriber_counts=eef_action_counts, node_names=node_names),
        profile,
        contract,
        selected_action_types={topic: "piper_msgs/msg/PosCmd" for topic in contract.eef_action_topics.values()},
    )

    first_eef_action_topic = next(iter(contract.eef_action_topics.values()))
    bad_eef_type = dict(topics)
    bad_eef_type[first_eef_action_topic] = ["sensor_msgs/msg/JointState"]
    with pytest.raises(ValueError, match="action topic.*piper_msgs/msg/PosCmd"):
        validate_profile_graph(
            _FakeGraphNode(topics=bad_eef_type, subscriber_counts=eef_action_counts, node_names=node_names),
            profile,
            contract,
            selected_action_types={topic: "piper_msgs/msg/PosCmd" for topic in contract.eef_action_topics.values()},
        )

    bad_eef_subscribers = dict(eef_action_counts)
    bad_eef_subscribers[first_eef_action_topic] = 0
    with pytest.raises(ValueError, match="exactly one external subscriber; found 0"):
        validate_profile_graph(
            _FakeGraphNode(topics=topics, subscriber_counts=bad_eef_subscribers, node_names=node_names),
            profile,
            contract,
            selected_action_types={topic: "piper_msgs/msg/PosCmd" for topic in contract.eef_action_topics.values()},
        )

    if profile.direct_adapter_node is not None:
        with pytest.raises(ValueError, match="missing required node"):
            validate_profile_graph(
                _FakeGraphNode(topics=topics, subscriber_counts=joint_action_counts, node_names=[]),
                profile,
                contract,
            )
