"""Tests for the ROS2-only dual-arm Piper deployment entrypoint."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import main_dual_ros


def test_parser_has_ros2_options_but_no_sdk_options() -> None:
    help_text = main_dual_ros.build_arg_parser().format_help()

    for flag in (
        "--host",
        "--port",
        "--bridge-python",
        "--ros2-config",
        "--dry-run",
        "--no-dry-run",
        "--publish-actions",
        "--max-action-delta",
        "--control-stack",
    ):
        assert flag in help_text
    for flag in (
        "--backend",
        "--left-can-port",
        "--right-can-port",
        "--high-camera-id",
        "--left-wrist-camera-id",
        "--right-wrist-camera-id",
        "--tele-mode",
        "--gripper-norm",
    ):
        assert flag not in help_text


def test_publish_actions_requires_no_dry_run() -> None:
    args = main_dual_ros.Args(publish_actions=True, dry_run=True)

    with pytest.raises(ValueError, match="--publish-actions requires --no-dry-run"):
        main_dual_ros._validate_ros2_contract(args)


def test_negative_action_delta_is_rejected() -> None:
    args = main_dual_ros.Args(max_action_delta=-0.1)

    with pytest.raises(ValueError, match="--max-action-delta must be finite and non-negative"):
        main_dual_ros._validate_ros2_contract(args)


def test_parse_args_defaults_to_safe_ros2_deployment() -> None:
    args = main_dual_ros.parse_args([])

    assert args.dry_run is True
    assert args.publish_actions is False
    assert args.bridge_python == Path("/usr/bin/python3")
    assert args.control_stack == "official-ros"



@pytest.mark.parametrize("control_stack", ["local-ros", "official-ros", "direct-sdk"])
def test_parser_accepts_control_stack_choices(control_stack: str) -> None:
    args = main_dual_ros.parse_args(["--control-stack", control_stack])

    assert args.control_stack == control_stack

def test_build_environment_omits_explicit_max_action_delta_and_loads_contract_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeRos2Environment:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(main_dual_ros, "Ros2DualEnvironment", FakeRos2Environment)

    environment = main_dual_ros._build_environment(main_dual_ros.Args(prompt="Fold the towel", control_stack="local-ros"))

    assert isinstance(environment, FakeRos2Environment)
    assert captured["prompt"] == "Fold the towel"
    assert captured["dry_run"] is True
    assert captured["publish_actions"] is False
    assert "max_action_delta" not in captured
    assert captured["control_stack"] == "local-ros"


def test_build_environment_forwards_explicit_max_action_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeRos2Environment:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(main_dual_ros, "Ros2DualEnvironment", FakeRos2Environment)

    environment = main_dual_ros._build_environment(main_dual_ros.Args(prompt="Fold the towel", max_action_delta=0.25, control_stack="direct-sdk"))

    assert isinstance(environment, FakeRos2Environment)
    assert captured["prompt"] == "Fold the towel"
    assert captured["dry_run"] is True
    assert captured["publish_actions"] is False
    assert captured["max_action_delta"] == 0.25
    assert captured["control_stack"] == "direct-sdk"
    assert "left_can_port" not in captured
    assert "high_camera_id" not in captured


def test_contract_summary_reflects_safe_ros2_defaults() -> None:
    summary = main_dual_ros.contract_summary(main_dual_ros.Args())

    assert "backend=ros2" in summary
    assert "dry_run=True" in summary
    assert "publish_actions=False" in summary
    assert "control_stack=official-ros" in summary
    assert "publish-actions requires --no-dry-run" in summary


def test_contract_summary_is_ros2_only() -> None:
    summary = main_dual_ros.contract_summary(main_dual_ros.Args())

    assert "backend=sdk" not in summary
    assert "ROS 2 backend" in summary
    assert "dry_run=True" in summary
    assert "publish-actions requires --no-dry-run" in summary


def test_main_rejects_unsafe_publish_before_environment_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fail_build(_: main_dual_ros.Args) -> object:
        nonlocal called
        called = True
        raise AssertionError("environment must not be created")

    monkeypatch.setattr(main_dual_ros, "_build_environment", fail_build)

    result = main_dual_ros.main(main_dual_ros.Args(publish_actions=True, dry_run=True))

    assert result == 2
    assert called is False


@pytest.mark.parametrize("failure, expected", [(None, 0), (RuntimeError("boom"), 1), (KeyboardInterrupt(), 130)])
def test_main_closes_environment_and_runtime_on_all_exit_paths(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException | None, expected: int
) -> None:
    class FakeEnvironment:
        closed = 0

        def close(self) -> None:
            self.closed += 1

    class FakeRuntime:
        closed = 0

        def run(self) -> None:
            if failure is not None:
                raise failure

        def close(self) -> None:
            self.closed += 1

    environment = FakeEnvironment()
    runtime = FakeRuntime()
    monkeypatch.setattr(main_dual_ros, "_build_environment", lambda _: environment)
    monkeypatch.setattr(main_dual_ros, "_build_policy", lambda _: object())
    monkeypatch.setattr(main_dual_ros, "_build_runtime", lambda *_: runtime)

    result = main_dual_ros.main(main_dual_ros.Args())

    assert result == expected
    assert environment.closed == 1
    assert runtime.closed == 1




