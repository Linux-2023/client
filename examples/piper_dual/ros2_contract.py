"""Immutable official ROS endpoint/control contract for piper_dual."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from types import MappingProxyType
from typing import Any
from typing import Mapping

import math
try:
    from examples.piper_dual.control_stack import ControlStackProfile
except ModuleNotFoundError:
    from control_stack import ControlStackProfile



_REQUIRED_BRIDGE_KEYS = {
    "joint_topics",
    "joint_action_topics",
    "eef_topics",
    "eef_action_topics",
    "status_topics",
    "joint_names",
    "control",
}
_REQUIRED_CONTROL_KEYS = {"speed_percent", "gripper_effort", "max_action_delta", "status_watchdog_s"}


@dataclass(frozen=True, slots=True)
class ControlContract:
    speed_percent: int
    gripper_effort: float
    max_action_delta: float
    status_watchdog_s: float

    def __post_init__(self) -> None:
        if type(self.speed_percent) is not int or isinstance(self.speed_percent, bool):
            raise ValueError("speed_percent must be an integer within 1..100")
        speed_percent = self.speed_percent
        if speed_percent < 1 or speed_percent > 100:
            raise ValueError("speed_percent must be an integer within 1..100")
        object.__setattr__(self, "speed_percent", speed_percent)
        object.__setattr__(self, "gripper_effort", _finite_float(self.gripper_effort, "gripper_effort"))
        object.__setattr__(self, "max_action_delta", _positive_finite_float(self.max_action_delta, "max_action_delta"))
        object.__setattr__(self, "status_watchdog_s", _positive_finite_float(self.status_watchdog_s, "status_watchdog_s"))


@dataclass(frozen=True, slots=True)
class BridgeContract:
    image_topics: Mapping[str, str] = field(repr=False)
    joint_topics: Mapping[str, str] = field(repr=False)
    joint_action_topics: Mapping[str, str] = field(repr=False)
    eef_topics: Mapping[str, str] = field(repr=False)
    eef_action_topics: Mapping[str, str] = field(repr=False)
    status_topics: Mapping[str, str] = field(repr=False)
    joint_names: tuple[str, ...]
    control: ControlContract

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_topics", _freeze_mapping(self.image_topics, "image_topics"))
        object.__setattr__(self, "joint_topics", _freeze_mapping(self.joint_topics, "joint_topics"))
        object.__setattr__(self, "joint_action_topics", _freeze_mapping(self.joint_action_topics, "joint_action_topics"))
        object.__setattr__(self, "eef_topics", _freeze_mapping(self.eef_topics, "eef_topics"))
        object.__setattr__(self, "eef_action_topics", _freeze_mapping(self.eef_action_topics, "eef_action_topics"))
        object.__setattr__(self, "status_topics", _freeze_mapping(self.status_topics, "status_topics"))
        object.__setattr__(self, "joint_names", tuple(str(name) for name in self.joint_names))
        _require_unique_topic_values(
            self.image_topics.values(),
            self.joint_topics.keys(),
            self.joint_action_topics.values(),
            self.eef_topics.values(),
            self.eef_action_topics.values(),
            self.status_topics.values(),
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> BridgeContract:
        if not isinstance(mapping, Mapping):
            raise ValueError("mapping must be a mapping")
        bridge_mapping = mapping.get("bridge_contract", mapping)
        if not isinstance(bridge_mapping, Mapping):
            raise ValueError("bridge_contract must be a mapping")
        missing = sorted(_REQUIRED_BRIDGE_KEYS.difference(bridge_mapping.keys()))
        if missing:
            raise ValueError(f"bridge_contract missing required keys: {', '.join(missing)}")

        cameras = mapping.get("cameras", ())
        image_topics = _parse_named_topic_list(cameras, "cameras")
        joint_topics = _parse_named_topic_mapping(bridge_mapping["joint_topics"], "joint_topics")
        joint_action_topics = _parse_named_topic_mapping(bridge_mapping["joint_action_topics"], "joint_action_topics")
        eef_topics = _parse_named_topic_mapping(bridge_mapping["eef_topics"], "eef_topics")
        eef_action_topics = _parse_named_topic_mapping(bridge_mapping["eef_action_topics"], "eef_action_topics")
        status_topics = _parse_named_topic_mapping(bridge_mapping["status_topics"], "status_topics")
        joint_names = _parse_joint_names(bridge_mapping["joint_names"])
        control = _parse_control(bridge_mapping["control"])
        return cls(
            image_topics=image_topics,
            joint_topics=joint_topics,
            joint_action_topics=joint_action_topics,
            eef_topics=eef_topics,
            eef_action_topics=eef_action_topics,
            status_topics=status_topics,
            joint_names=joint_names,
            control=control,
        )


@dataclass(frozen=True, slots=True)
class EndpointPlan:
    image_subscriptions: tuple[tuple[str, str], ...]
    joint_subscriptions: tuple[tuple[str, str], ...]
    status_subscriptions: tuple[tuple[str, str], ...]
    eef_subscriptions: tuple[tuple[str, str], ...]
    action_publishers: tuple[tuple[str, str], ...]

    @classmethod
    def for_mode(
        cls,
        contract: BridgeContract,
        dry_run: bool,
        publish_actions: bool,
        eef_control: bool,
        include_eef: bool,
    ) -> EndpointPlan:
        image_subscriptions = tuple(contract.image_topics.items())
        joint_subscriptions = tuple(contract.joint_topics.items())
        status_subscriptions = tuple(contract.status_topics.items())
        eef_subscriptions = tuple(contract.eef_topics.items()) if include_eef else ()
        if dry_run or not publish_actions:
            action_publishers: tuple[tuple[str, str], ...] = ()
        elif eef_control:
            action_publishers = tuple(contract.eef_action_topics.items())
        else:
            action_publishers = tuple(contract.joint_action_topics.items())
        return cls(
            image_subscriptions=image_subscriptions,
            joint_subscriptions=joint_subscriptions,
            status_subscriptions=status_subscriptions,
            eef_subscriptions=eef_subscriptions,
            action_publishers=action_publishers,
        )


def validate_profile_graph(
    node: Any,
    profile: ControlStackProfile,
    contract: BridgeContract,
    action_topics: tuple[str, ...] | None = None,
) -> None:
    """Validate that the live ROS graph matches the selected control stack contract."""
    actual = {name: tuple(types) for name, types in node.get_topic_names_and_types()}

    for topic in profile.required_topics:
        if topic not in actual:
            raise ValueError(f"control stack {profile.stack_id} missing required topic {topic}")
    for topic in profile.forbidden_topics:
        if topic in actual:
            raise ValueError(f"control stack {profile.stack_id} found forbidden topic {topic}")

    for topic in contract.image_topics.values():
        _require_topic_type(actual, topic, "sensor_msgs/msg/Image", "camera")
    for topic in contract.joint_topics.keys():
        _require_topic_type(actual, topic, "sensor_msgs/msg/JointState", "joint stream")
    for topic in contract.joint_action_topics.values():
        _require_topic_type(actual, topic, "sensor_msgs/msg/JointState", "joint action")
    for topic in contract.status_topics.values():
        _require_topic_type(actual, topic, "piper_msgs/msg/PiperStatusMsg", "status")

    if profile.direct_adapter_node is not None:
        node_names = tuple(str(name) for name in node.get_node_names())
        if not any(profile.direct_adapter_node in name for name in node_names):
            raise ValueError(f"control stack {profile.stack_id} missing required node {profile.direct_adapter_node}")

    selected_action_topics = action_topics if action_topics is not None else tuple(contract.joint_action_topics.values())
    for topic in selected_action_topics:
        subscriber_count = int(node.count_subscribers(topic))
        if subscriber_count != 1:
            raise ValueError(
                f"control stack {profile.stack_id} action topic {topic} must have exactly one external subscriber; "
                f"found {subscriber_count}"
            )



def _freeze_mapping(mapping: Mapping[str, Any], label: str) -> Mapping[str, str]:
    parsed = _parse_named_topic_mapping(mapping, label)
    return MappingProxyType(dict(parsed))


def _parse_named_topic_list(values: Any, label: str) -> Mapping[str, str]:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list")
    parsed: dict[str, str] = {}
    for index, item in enumerate(values):
        if not isinstance(item, Mapping):
            raise ValueError(f"{label}[{index}] must be a mapping")
        name = item.get("name")
        topic = item.get("topic")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label}[{index}].name must be a non-empty string")
        if not isinstance(topic, str) or not topic:
            raise ValueError(f"{label}[{index}].topic must be a non-empty string")
        if name in parsed:
            raise ValueError(f"duplicate {label} name: {name}")
        if topic in parsed.values():
            raise ValueError(f"duplicate topic name: {topic}")
        parsed[name] = topic
    return MappingProxyType(parsed)


def _parse_named_topic_mapping(values: Any, label: str) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{label} must be a mapping")
    parsed: dict[str, str] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} keys must be non-empty strings")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label}.{key} must be a non-empty string")
        if key in parsed:
            raise ValueError(f"duplicate {label} key: {key}")
        parsed[key] = value
    return MappingProxyType(parsed)


def _parse_joint_names(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise ValueError("joint_names must be a list")
    names = tuple(str(value) for value in values)
    if len(names) != 7:
        raise ValueError("joint_names must contain exactly seven entries")
    if any(not name for name in names):
        raise ValueError("joint_names must not contain empty names")
    return names


def _parse_control(values: Any) -> ControlContract:
    if not isinstance(values, Mapping):
        raise ValueError("control must be a mapping")
    missing = sorted(_REQUIRED_CONTROL_KEYS.difference(values.keys()))
    if missing:
        raise ValueError(f"control missing required keys: {', '.join(missing)}")
    return ControlContract(
        speed_percent=values["speed_percent"],
        gripper_effort=values["gripper_effort"],
        max_action_delta=values["max_action_delta"],
        status_watchdog_s=values["status_watchdog_s"],
    )


def _finite_float(value: Any, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


def _positive_finite_float(value: Any, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be positive and finite") from exc
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return numeric


def _require_topic_type(actual: Mapping[str, tuple[str, ...]], topic: str, expected_type: str, label: str) -> None:
    types = actual.get(topic)
    if types is None:
        raise ValueError(f"missing {label} topic {topic}")
    if expected_type not in types:
        raise ValueError(f"{label} topic {topic} must use {expected_type}; found {', '.join(types) or 'no types'}")


def _require_unique_topic_values(*topic_groups: Any) -> None:
    seen: set[str] = set()
    for topics in topic_groups:
        for topic in topics:
            if topic in seen:
                raise ValueError(f"duplicate topic name: {topic}")
            seen.add(topic)
