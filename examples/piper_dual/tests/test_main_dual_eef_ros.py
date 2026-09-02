"""Tests for the independent EEF ROS 2 deployment entrypoint."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main_dual_eef_ros


def test_parser_defaults_to_safe_eef_ros2_contract() -> None:
    args = main_dual_eef_ros.parse_args([])

    assert args.dry_run is True
    assert args.publish_actions is False
    assert args.eef_left_topic == "/puppet/end_pose_left"
    assert args.eef_right_topic == "/puppet/end_pose_right"
    assert args.eef_left_action_topic == "/pos_left_cmd"
    assert args.eef_right_action_topic == "/pos_right_cmd"
    assert args.bridge_python == Path("/usr/bin/python3")


def test_parser_exposes_eef_topics_and_ros2_safety_flags() -> None:
    help_text = main_dual_eef_ros.build_arg_parser().format_help()

    for flag in (
        "--eef-left-topic",
        "--eef-right-topic",
        "--eef-left-action-topic",
        "--eef-right-action-topic",
        "--dry-run",
        "--no-dry-run",
        "--publish-actions",
        "--max-action-delta",
    ):
        assert flag in help_text


def test_publish_actions_requires_no_dry_run() -> None:
    args = main_dual_eef_ros.Args(publish_actions=True, dry_run=True)

    with pytest.raises(ValueError, match="--publish-actions requires --no-dry-run"):
        main_dual_eef_ros._validate_eef_contract(args)


def test_build_environment_constructs_eef_backend_and_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeBackend:
        def __init__(self, **kwargs: object) -> None:
            captured["backend"] = kwargs

    class FakeEnvironment:
        def __init__(self, **kwargs: object) -> None:
            captured["environment"] = kwargs

    monkeypatch.setattr(main_dual_eef_ros, "Ros2BackendClient", FakeBackend)
    monkeypatch.setattr(main_dual_eef_ros, "Ros2DualEnvironment", FakeEnvironment)

    args = main_dual_eef_ros.Args(
        prompt="Stack the cups",
        eef_left_topic="/left_pose",
        eef_right_topic="/right_pose",
        eef_left_action_topic="/left_cmd",
        eef_right_action_topic="/right_cmd",
        max_action_delta=0.2,
    )
    environment = main_dual_eef_ros._build_environment(args)

    assert isinstance(environment, FakeEnvironment)
    backend_kwargs = captured["backend"]
    assert backend_kwargs["eef_control"] is True
    assert backend_kwargs["eef_left_topic"] == "/left_pose"
    assert backend_kwargs["eef_right_topic"] == "/right_pose"
    assert backend_kwargs["eef_left_action_topic"] == "/left_cmd"
    assert backend_kwargs["eef_right_action_topic"] == "/right_cmd"
    env_kwargs = captured["environment"]
    assert env_kwargs["prompt"] == "Stack the cups"
    assert env_kwargs["max_action_delta"] == 0.2
    assert env_kwargs["observation_adapter"].__class__.__name__ == "EefObservationAdapter"
    assert env_kwargs["action_adapter"].__class__.__name__ == "EefActionAdapter"


def test_main_rejects_unsafe_publish_before_environment_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fail_build(_: main_dual_eef_ros.Args) -> object:
        nonlocal called
        called = True
        raise AssertionError("environment must not be created")

    monkeypatch.setattr(main_dual_eef_ros, "_build_environment", fail_build)

    result = main_dual_eef_ros.main(main_dual_eef_ros.Args(publish_actions=True, dry_run=True))

    assert result == 2
    assert called is False
