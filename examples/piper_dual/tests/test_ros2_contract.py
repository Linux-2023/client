"""Focused contract tests for the official piper_dual ROS 2 endpoint contract."""

from pathlib import Path
import math
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ros2_contract import BridgeContract
from ros2_contract import ControlContract
from ros2_contract import EndpointPlan
from control_stack import load_profile_config
from control_stack import profile_for
from ros2_contract import validate_profile_graph


class FakeGraphNode:
    def __init__(
        self,
        *,
        topics: dict[str, list[str]],
        subscriber_counts: dict[str, int] | None = None,
        node_names: list[str] | None = None,
    ) -> None:
        self._topics = topics
        self._subscriber_counts = subscriber_counts or {}
        self._node_names = node_names or []

    def get_topic_names_and_types(self):
        return [(topic, list(types)) for topic, types in self._topics.items()]

    def count_subscribers(self, topic: str) -> int:
        return self._subscriber_counts.get(topic, 0)

    def get_node_names(self):
        return list(self._node_names)


def _profile_contract(stack_id: str) -> tuple[object, BridgeContract]:
    profile, mapping = load_profile_config(stack_id, profile_for(stack_id).config_path)
    return profile, BridgeContract.from_mapping(mapping)


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
    return topics


def _action_subscribers(contract: BridgeContract, count: int = 1) -> dict[str, int]:
    return {topic: count for topic in contract.joint_action_topics.values()}



def _load_shipped_mapping() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "ros2_piper_dual.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _load_complete_mapping() -> dict:
    return _load_shipped_mapping()


def test_bridge_contract_from_shipped_yaml_matches_official_contract() -> None:
    contract = BridgeContract.from_mapping(_load_complete_mapping())

    assert contract.joint_topics == {
        "/joint_left": "puppet_left",
        "/joint_right": "puppet_right",
        "/joint_states_ctrl_left": "master_left",
        "/joint_states_ctrl_right": "master_right",
    }
    assert contract.joint_action_topics == {
        "left": "/joint_ctrl_cmd_left",
        "right": "/joint_ctrl_cmd_right",
    }
    assert contract.eef_topics == {
        "left": "/end_pose_stamped_left",
        "right": "/end_pose_stamped_right",
    }
    assert contract.status_topics == {
        "left": "/arm_status_left",
        "right": "/arm_status_right",
    }
    assert contract.joint_names == ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper")
    assert contract.control == ControlContract(
        speed_percent=30,
        gripper_effort=0.5,
        max_action_delta=0.05,
        status_watchdog_s=0.5,
    )


def test_bridge_contract_rejects_missing_keys_duplicate_topics_and_invalid_control_values() -> None:
    mapping = _load_complete_mapping()
    bridge = dict(mapping["bridge_contract"])

    for missing_key, expected_message in (
        ("control", "bridge_contract missing required keys: control"),
        ("joint_names", "bridge_contract missing required keys: joint_names"),
    ):
        invalid_bridge = dict(bridge)
        invalid_bridge.pop(missing_key)
        with pytest.raises(ValueError, match=expected_message):
            BridgeContract.from_mapping({"cameras": mapping["cameras"], "bridge_contract": invalid_bridge})

    duplicate_topics = dict(bridge)
    duplicate_topics["joint_topics"] = {"/joint_left": "puppet_left", "/camera_l/color/image_raw": "puppet_right"}
    with pytest.raises(ValueError, match=r"duplicate topic name: /camera_l/color/image_raw"):
        BridgeContract.from_mapping({"cameras": mapping["cameras"], "bridge_contract": duplicate_topics})

    for control_patch, expected_message in (
        ({"speed_percent": 0, "gripper_effort": 0.5, "max_action_delta": 0.05, "status_watchdog_s": 0.5}, "speed_percent must be an integer within 1..100"),
        ({"speed_percent": 101, "gripper_effort": 0.5, "max_action_delta": 0.05, "status_watchdog_s": 0.5}, "speed_percent must be an integer within 1..100"),
        ({"speed_percent": 30.0, "gripper_effort": 0.5, "max_action_delta": 0.05, "status_watchdog_s": 0.5}, "speed_percent must be an integer within 1..100"),
        ({"speed_percent": "30", "gripper_effort": 0.5, "max_action_delta": 0.05, "status_watchdog_s": 0.5}, "speed_percent must be an integer within 1..100"),
        ({"speed_percent": True, "gripper_effort": 0.5, "max_action_delta": 0.05, "status_watchdog_s": 0.5}, "speed_percent must be an integer within 1..100"),
        ({"speed_percent": 30, "gripper_effort": 0.5, "max_action_delta": 0.05, "status_watchdog_s": 0.0}, "status_watchdog_s must be positive and finite"),
        ({"speed_percent": 30, "gripper_effort": float("nan"), "max_action_delta": 0.05, "status_watchdog_s": 0.5}, "gripper_effort must be finite"),
        ({"speed_percent": 30, "gripper_effort": 0.5, "max_action_delta": 0.0, "status_watchdog_s": 0.5}, "max_action_delta must be positive and finite"),
        ({"speed_percent": 30, "gripper_effort": 0.5, "max_action_delta": float("inf"), "status_watchdog_s": 0.5}, "max_action_delta must be positive and finite"),
    ):
        invalid_bridge = dict(bridge)
        invalid_bridge["control"] = control_patch
        with pytest.raises(ValueError, match=expected_message):
            BridgeContract.from_mapping({"cameras": mapping["cameras"], "bridge_contract": invalid_bridge})


def test_endpoint_plan_modes_match_required_publishers_and_subscriptions() -> None:
    contract = BridgeContract.from_mapping(_load_complete_mapping())

    dry_run_plan = EndpointPlan.for_mode(contract, dry_run=True, publish_actions=False, eef_control=False, include_eef=False)
    assert len(dry_run_plan.image_subscriptions) == 3
    assert len(dry_run_plan.joint_subscriptions) == 4
    assert len(dry_run_plan.status_subscriptions) == 2
    assert dry_run_plan.eef_subscriptions == ()
    assert dry_run_plan.action_publishers == ()

    dry_run_eef_plan = EndpointPlan.for_mode(contract, dry_run=True, publish_actions=False, eef_control=True, include_eef=True)
    assert len(dry_run_eef_plan.eef_subscriptions) == 2
    assert dry_run_eef_plan.action_publishers == ()

    live_joint_plan = EndpointPlan.for_mode(contract, dry_run=False, publish_actions=True, eef_control=False, include_eef=False)
    assert live_joint_plan.action_publishers == (
        ("left", "/joint_ctrl_cmd_left"),
        ("right", "/joint_ctrl_cmd_right"),
    )

    live_eef_plan = EndpointPlan.for_mode(contract, dry_run=False, publish_actions=True, eef_control=True, include_eef=True)
    assert live_eef_plan.action_publishers == (
        ("left", "/pos_left_cmd"),
        ("right", "/pos_right_cmd"),
    )
    assert len(live_eef_plan.eef_subscriptions) == 2


def test_profile_graph_accepts_complete_local_ros_contract() -> None:
    profile, contract = _profile_contract("local-ros")
    node = FakeGraphNode(topics=_complete_topic_types(contract), subscriber_counts=_action_subscribers(contract))

    validate_profile_graph(node, profile, contract)


def test_profile_graph_rejects_forbidden_stack_topics() -> None:
    profile, contract = _profile_contract("local-ros")
    topics = _complete_topic_types(contract)
    topics["/joint_left"] = ["sensor_msgs/msg/JointState"]
    node = FakeGraphNode(topics=topics, subscriber_counts=_action_subscribers(contract))

    with pytest.raises(ValueError, match="forbidden.*joint_left"):
        validate_profile_graph(node, profile, contract)


def test_profile_graph_rejects_wrong_message_type() -> None:
    profile, contract = _profile_contract("local-ros")
    topics = _complete_topic_types(contract)
    topics["/puppet/joint_left"] = ["std_msgs/msg/Bool"]
    node = FakeGraphNode(topics=topics, subscriber_counts=_action_subscribers(contract))

    with pytest.raises(ValueError, match="sensor_msgs/msg/JointState"):
        validate_profile_graph(node, profile, contract)


def test_profile_graph_rejects_missing_direct_sdk_adapter_node() -> None:
    profile, contract = _profile_contract("direct-sdk")
    node = FakeGraphNode(topics=_complete_topic_types(contract), subscriber_counts=_action_subscribers(contract))

    with pytest.raises(ValueError, match="piper_direct_sdk_adapter"):
        validate_profile_graph(node, profile, contract)


def test_profile_graph_rejects_action_topics_without_exactly_one_external_subscriber() -> None:
    profile, contract = _profile_contract("official-ros")
    subscriber_counts = _action_subscribers(contract)
    subscriber_counts["/joint_ctrl_cmd_left"] = 2
    node = FakeGraphNode(topics=_complete_topic_types(contract), subscriber_counts=subscriber_counts)

    with pytest.raises(ValueError, match="exactly one external subscriber"):
        validate_profile_graph(node, profile, contract)


def test_profile_graph_rejects_selected_contract_action_topic_without_subscriber() -> None:
    profile = profile_for("official-ros")
    mapping = _load_complete_mapping()
    bridge = dict(mapping["bridge_contract"])
    bridge["joint_action_topics"] = {
        "left": "/custom/joint_left_cmd",
        "right": "/custom/joint_right_cmd",
    }
    mapping = dict(mapping)
    mapping["bridge_contract"] = bridge
    contract = BridgeContract.from_mapping(mapping)
    topics = _complete_topic_types(contract)
    subscriber_counts = {topic: 1 for topic in profile.action_topics}
    node = FakeGraphNode(topics=topics, subscriber_counts=subscriber_counts)

    with pytest.raises(ValueError, match="/custom/joint_left_cmd"):
        validate_profile_graph(node, profile, contract)
