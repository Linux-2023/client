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

STACK_IDS = ("local-ros", "official-ros", "direct-sdk")
EXPECTED_SPEED = {"local-ros": 30, "official-ros": 30, "direct-sdk": 100}
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


class _FakeBridgeNode:
    def __init__(self, contract: BridgeContract, *, dry_run: bool, publish_actions: bool) -> None:
        self.contract = contract
        self.publishers: list[tuple[str, str]] = []
        self.published: list[dict[str, object]] = []
        self.status_messages: list[dict[str, object]] = []
        self._ready = False
        self._publish_actions = publish_actions and not dry_run
        self._topics = _complete_topic_types(contract)

    def mark_ready(self) -> None:
        self._ready = True

    def get_topic_names_and_types(self):
        return [(topic, list(types)) for topic, types in self._topics.items()]

    def count_subscribers(self, topic: str) -> int:
        return 1 if topic in self.contract.joint_action_topics.values() else 0

    def get_node_names(self):
        return []

    def _handle_request(self, request: dict[str, object]) -> None:
        if not self._ready:
            raise RuntimeError("status not ready")
        if request.get("type") != "publish_action":
            raise ValueError("unsupported request")
        if self._publish_actions:
            self.publishers = [
                ("JointState", self.contract.joint_action_topics["left"]),
                ("JointState", self.contract.joint_action_topics["right"]),
            ]
            self.published.append(request)
            self.status_messages.append({"type": "status", "state": "action_published"})
        else:
            self.status_messages.append(
                {
                    "type": "status",
                    "state": "action_ignored",
                    "metadata": {"dry_run": True, "publish_actions": False},
                }
            )


def test_profiles_have_exact_three_stack_contracts_speed_and_action_topics() -> None:
    for stack_id in STACK_IDS:
        profile, mapping = load_profile_config(stack_id, profile_for(stack_id).config_path)
        contract = BridgeContract.from_mapping(mapping)

        assert profile.stack_id == stack_id
        assert contract.control.speed_percent == EXPECTED_SPEED[stack_id]
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
def test_dry_run_bridge_constructs_zero_publishers_and_ignores_valid_action(stack_id: str) -> None:
    profile, mapping = load_profile_config(stack_id, profile_for(stack_id).config_path)
    contract = BridgeContract.from_mapping(mapping)
    node = _FakeBridgeNode(contract, dry_run=True, publish_actions=True)

    node.mark_ready()
    assert node.publishers == []

    node._handle_request({"type": "publish_action", "left": [0.0] * 7, "right": [0.0] * 7})

    assert node.publishers == []
    assert node.published == []
    assert node.status_messages == [
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
def test_profile_graph_guards_joint_and_eef_action_types_and_conflicts(stack_id: str) -> None:
    profile, mapping = load_profile_config(stack_id, profile_for(stack_id).config_path)
    contract = BridgeContract.from_mapping(mapping)
    topics = _complete_topic_types(contract)
    joint_action_counts = {topic: 1 for topic in contract.joint_action_topics.values()}
    eef_action_counts = {topic: 1 for topic in contract.eef_action_topics.values()}

    node = _FakeGraphNode(
        topics=topics,
        subscriber_counts=joint_action_counts,
        node_names=["piper_direct_sdk_adapter"] if stack_id == "direct-sdk" else [],
    )
    validate_profile_graph(node, profile, contract)

    joint_conflict = dict(topics)
    first_joint_action_topic = contract.joint_action_topics["left"]
    joint_conflict[first_joint_action_topic] = ["std_msgs/msg/Bool"]
    joint_node = _FakeGraphNode(
        topics=joint_conflict,
        subscriber_counts=joint_action_counts,
        node_names=["piper_direct_sdk_adapter"] if stack_id == "direct-sdk" else [],
    )
    with pytest.raises(ValueError, match="joint action.*sensor_msgs/msg/JointState"):
        validate_profile_graph(joint_node, profile, contract)

    if contract.eef_topics:
        eef_node = _FakeGraphNode(
            topics=topics,
            subscriber_counts=eef_action_counts,
            node_names=["piper_direct_sdk_adapter"] if stack_id == "direct-sdk" else [],
        )
        validate_profile_graph(
            eef_node,
            profile,
            contract,
            selected_action_types={topic: "piper_msgs/msg/PosCmd" for topic in contract.eef_action_topics.values()},
        )

        eef_conflict = dict(topics)
        eef_conflict[next(iter(contract.eef_action_topics.values()))] = ["sensor_msgs/msg/JointState"]
        bad_eef_node = _FakeGraphNode(
            topics=eef_conflict,
            subscriber_counts=eef_action_counts,
            node_names=["piper_direct_sdk_adapter"] if stack_id == "direct-sdk" else [],
        )
        with pytest.raises(ValueError, match="action topic.*piper_msgs/msg/PosCmd"):
            validate_profile_graph(
                bad_eef_node,
                profile,
                contract,
                selected_action_types={topic: "piper_msgs/msg/PosCmd" for topic in contract.eef_action_topics.values()},
            )
