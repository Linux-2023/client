"""Immutable control-stack profiles for the Piper dual ROS 2 comparison configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

ControlStackId = Literal["local-ros", "official-ros", "direct-sdk"]

_CONFIG_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class ControlStackProfile:
    stack_id: ControlStackId
    config_path: Path
    required_topics: tuple[str, ...]
    forbidden_topics: tuple[str, ...]
    action_topics: tuple[str, ...]
    direct_adapter_node: str | None


_PROFILES: dict[ControlStackId, ControlStackProfile] = {
    "local-ros": ControlStackProfile(
        stack_id="local-ros",
        config_path=_CONFIG_DIR / "ros2_piper_dual_local.yaml",
        required_topics=(
            "/puppet/joint_left",
            "/puppet/joint_right",
            "/master/joint_left",
            "/master/joint_right",
            "/piper_left_ctrl_node/arm_status",
            "/piper_right_ctrl_node/arm_status",
        ),
        forbidden_topics=("/joint_left", "/joint_right", "/direct_sdk/joint_left", "/direct_sdk/joint_right"),
        action_topics=("/joint_left_states", "/joint_right_states"),
        direct_adapter_node=None,
    ),
    "official-ros": ControlStackProfile(
        stack_id="official-ros",
        config_path=_CONFIG_DIR / "ros2_piper_dual.yaml",
        required_topics=(
            "/joint_left",
            "/joint_right",
            "/joint_states_ctrl_left",
            "/joint_states_ctrl_right",
            "/arm_status_left",
            "/arm_status_right",
        ),
        forbidden_topics=("/puppet/joint_left", "/puppet/joint_right", "/direct_sdk/joint_left", "/direct_sdk/joint_right"),
        action_topics=("/joint_ctrl_cmd_left", "/joint_ctrl_cmd_right"),
        direct_adapter_node=None,
    ),
    "direct-sdk": ControlStackProfile(
        stack_id="direct-sdk",
        config_path=_CONFIG_DIR / "ros2_piper_dual_direct_sdk.yaml",
        required_topics=(
            "/direct_sdk/joint_left",
            "/direct_sdk/joint_right",
            "/direct_sdk/joint_ctrl_left",
            "/direct_sdk/joint_ctrl_right",
            "/direct_sdk/arm_status_left",
            "/direct_sdk/arm_status_right",
        ),
        forbidden_topics=("/puppet/joint_left", "/puppet/joint_right", "/joint_left", "/joint_right"),
        action_topics=("/direct_sdk/joint_cmd_left", "/direct_sdk/joint_cmd_right"),
        direct_adapter_node="piper_direct_sdk_adapter",
    ),
}


def profile_for(stack_id: str) -> ControlStackProfile:
    try:
        return _PROFILES[stack_id]  # type: ignore[index]
    except KeyError as exc:
        raise ValueError(f"unknown control stack: {stack_id}") from exc


def load_profile_config(stack_id: str, config_path: Path) -> tuple[ControlStackProfile, dict[str, Any]]:
    profile = profile_for(stack_id)
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{config_path} must contain a top-level mapping")
    identity = loaded.get("control_stack_id")
    if not isinstance(identity, str):
        raise ValueError(f"{config_path} missing string control_stack_id for {stack_id}")
    if identity != stack_id:
        raise ValueError(f"control_stack_id {identity!r} does not match expected {stack_id!r}")
    return profile, loaded
