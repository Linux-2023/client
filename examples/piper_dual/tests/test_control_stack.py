"""Focused tests for immutable control-stack profiles and shipped YAML identities."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control_stack import ControlStackProfile
from control_stack import load_profile_config
from control_stack import profile_for


STACK_IDS = ("local-ros", "official-ros", "direct-sdk")


def test_profiles_have_distinct_ids_and_identity_files() -> None:
    assert [profile_for(name).stack_id for name in STACK_IDS] == ["local-ros", "official-ros", "direct-sdk"]
    for name in STACK_IDS:
        profile, mapping = load_profile_config(name, profile_for(name).config_path)
        assert profile.stack_id == name
        assert mapping["control_stack_id"] == name


def test_profile_rejects_yaml_for_another_stack(tmp_path: Path) -> None:
    path = tmp_path / "wrong.yaml"
    path.write_text("control_stack_id: official-ros\n", encoding="utf-8")
    with pytest.raises(ValueError, match="control_stack_id.*local-ros"):
        load_profile_config("local-ros", path)


def test_profiles_have_non_overlapping_identity_topics() -> None:
    profiles = [profile_for(name) for name in STACK_IDS]
    assert all(set(profile.required_topics).isdisjoint(profile.forbidden_topics) for profile in profiles)


def test_profiles_expose_exact_immutable_endpoint_sets() -> None:
    local = profile_for("local-ros")
    assert local == ControlStackProfile(
        stack_id="local-ros",
        config_path=Path(__file__).resolve().parents[1] / "ros2_piper_dual_local.yaml",
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
    )

    official = profile_for("official-ros")
    assert official == ControlStackProfile(
        stack_id="official-ros",
        config_path=Path(__file__).resolve().parents[1] / "ros2_piper_dual.yaml",
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
    )

    direct = profile_for("direct-sdk")
    assert direct == ControlStackProfile(
        stack_id="direct-sdk",
        config_path=Path(__file__).resolve().parents[1] / "ros2_piper_dual_direct_sdk.yaml",
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
    )

    with pytest.raises(Exception):
        direct.required_topics += ("/mutated",)


def test_profile_yaml_contracts_have_required_speeds_and_direct_streams() -> None:
    expected_speed = {"local-ros": 30, "official-ros": 30, "direct-sdk": 100}
    for name in STACK_IDS:
        _, mapping = load_profile_config(name, profile_for(name).config_path)
        assert mapping["bridge_contract"]["control"]["speed_percent"] == expected_speed[name]

    _, direct_mapping = load_profile_config("direct-sdk", profile_for("direct-sdk").config_path)
    assert direct_mapping["bridge_contract"]["joint_topics"] == {
        "/direct_sdk/joint_left": "puppet_left",
        "/direct_sdk/joint_right": "puppet_right",
        "/direct_sdk/joint_ctrl_left": "master_left",
        "/direct_sdk/joint_ctrl_right": "master_right",
    }
    assert direct_mapping["bridge_contract"]["status_topics"] == {
        "left": "/direct_sdk/arm_status_left",
        "right": "/direct_sdk/arm_status_right",
    }


def test_load_profile_config_rejects_missing_or_non_string_identity(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    missing.write_text("bridge_contract: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="control_stack_id"):
        load_profile_config("local-ros", missing)

    non_string = tmp_path / "non-string.yaml"
    non_string.write_text("control_stack_id: 123\n", encoding="utf-8")
    with pytest.raises(ValueError, match="control_stack_id"):
        load_profile_config("local-ros", non_string)


def test_profile_for_rejects_unknown_id() -> None:
    with pytest.raises(ValueError, match="unknown control stack"):
        profile_for("missing")
