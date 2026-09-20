"""Tests for the independent XYZ3D EEF ROS 2 deployment entrypoint."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main_dual_eef_xyz3d_ros


def test_parser_defaults_to_safe_xyz3d_contract() -> None:
    args = main_dual_eef_xyz3d_ros.parse_args([])

    assert args.action_horizon == main_dual_eef_xyz3d_ros.Args.action_horizon
    assert args.fps == 30
    assert args.dry_run is True
    assert args.publish_actions is False
    assert args.eef_left_topic == "/puppet/end_pose_left"
    assert args.eef_right_topic == "/puppet/end_pose_right"
    assert args.eef_left_action_topic == "/pos_left_cmd"
    assert args.eef_right_action_topic == "/pos_right_cmd"
    assert args.bridge_python == Path("/usr/bin/python3")
    assert args.control_stack == "official-ros"


def test_parser_exposes_xyz3d_topics_and_safety_flags() -> None:
    help_text = main_dual_eef_xyz3d_ros.build_arg_parser().format_help()

    for flag in (
        "--eef-left-topic",
        "--eef-right-topic",
        "--eef-left-action-topic",
        "--eef-right-action-topic",
        "--dry-run",
        "--no-dry-run",
        "--publish-actions",
        "--max-action-delta",
        "--max-position-delta",
        "--max-orientation-delta",
        "--max-gripper-delta",
        "--control-stack",
    ):
        assert flag in help_text


@pytest.mark.parametrize("control_stack", ["local-ros", "official-ros", "direct-sdk"])
def test_parser_accepts_control_stack_choices(control_stack: str) -> None:
    args = main_dual_eef_xyz3d_ros.parse_args(["--control-stack", control_stack])

    assert args.control_stack == control_stack


def test_publish_actions_requires_no_dry_run() -> None:
    args = main_dual_eef_xyz3d_ros.Args(publish_actions=True, dry_run=True)

    with pytest.raises(ValueError, match="--publish-actions requires --no-dry-run"):
        main_dual_eef_xyz3d_ros._validate_xyz3d_contract(args)


def test_contract_accepts_explicit_max_action_delta_above_selected_config_default() -> None:
    args = main_dual_eef_xyz3d_ros.Args(
        control_stack="local-ros",
        ros2_config=Path(__file__).resolve().parents[1] / "ros2_piper_dual_local.yaml",
        max_action_delta=20.0,
    )

    main_dual_eef_xyz3d_ros._validate_xyz3d_contract(args)
    assert main_dual_eef_xyz3d_ros._effective_max_action_delta(args) == pytest.approx(20.0)


def test_contract_accepts_max_action_delta_equal_to_selected_config_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = False

    class FakeEnvironment:
        def close(self) -> None:
            pass

    class FakeRuntime:
        def run(self) -> None:
            pass

        def close(self) -> None:
            pass

    def build_environment(args: main_dual_eef_xyz3d_ros.Args) -> FakeEnvironment:
        nonlocal built
        built = True
        assert args.max_action_delta == pytest.approx(0.05)
        return FakeEnvironment()
    monkeypatch.setattr(main_dual_eef_xyz3d_ros, "_build_environment", build_environment)
    monkeypatch.setattr(main_dual_eef_xyz3d_ros, "_build_policy", lambda args: object())
    monkeypatch.setattr(
        main_dual_eef_xyz3d_ros, "_build_runtime", lambda args, environment, policy: FakeRuntime()
    )

    result = main_dual_eef_xyz3d_ros.main(
        main_dual_eef_xyz3d_ros.Args(
            control_stack="local-ros",
            ros2_config=Path(__file__).resolve().parents[1] / "ros2_piper_dual_local.yaml",
            dry_run=False,
            publish_actions=True,
            max_action_delta=0.05,
        )
    )

    assert result == 0
    assert built is True


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (main_dual_eef_xyz3d_ros.Args(max_action_delta=float("nan")), "finite and non-negative"),
        (main_dual_eef_xyz3d_ros.Args(max_action_delta=-0.1), "finite and non-negative"),
        (main_dual_eef_xyz3d_ros.Args(eef_left_topic=""), "observation topics"),
        (main_dual_eef_xyz3d_ros.Args(eef_right_topic=""), "observation topics"),
        (main_dual_eef_xyz3d_ros.Args(eef_left_action_topic=""), "action topics"),
        (main_dual_eef_xyz3d_ros.Args(eef_right_action_topic=""), "action topics"),
    ],
)
def test_xyz3d_contract_rejects_invalid_safety_values(
    args: main_dual_eef_xyz3d_ros.Args,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        main_dual_eef_xyz3d_ros._validate_xyz3d_contract(args)


def test_contract_summary_names_14d_xyz3d_layout_and_safety_limits() -> None:
    summary = main_dual_eef_xyz3d_ros.contract_summary(main_dual_eef_xyz3d_ros.Args())

    assert "selected_stack=official-ros" in summary
    assert "control_mode=MOVE L/P selected by PosCmd; mode_feedback is not pre-gated" in summary
    assert "effective_max_action_delta=0.05" in summary
    assert "effective_position_delta_limit=0.05 m" in summary
    assert "effective_orientation_delta_limit=0.05 rad" in summary
    assert "14D" in summary
    assert "left_xyzrpy" in summary
    assert "right_xyzrpy" in summary


def test_build_environment_constructs_xyz3d_backend_and_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeBackend:
        def __init__(self, **kwargs: object) -> None:
            captured["backend"] = kwargs

    class FakeEnvironment:
        def __init__(self, **kwargs: object) -> None:
            captured["environment"] = kwargs

    monkeypatch.setattr(main_dual_eef_xyz3d_ros, "Ros2BackendClient", FakeBackend)
    monkeypatch.setattr(main_dual_eef_xyz3d_ros, "Ros2DualEnvironment", FakeEnvironment)

    args = main_dual_eef_xyz3d_ros.Args(
        prompt="Stack the cups",
        eef_left_topic="/left_pose",
        eef_right_topic="/right_pose",
        eef_left_action_topic="/left_cmd",
        eef_right_action_topic="/right_cmd",
        max_action_delta=20.0,
        max_position_delta=0.05,
        max_orientation_delta=0.10,
        max_gripper_delta=0.02,
        ros2_config=Path(__file__).resolve().parents[1] / "ros2_piper_dual_local.yaml",
        control_stack="local-ros",
    )
    environment = main_dual_eef_xyz3d_ros._build_environment(args)

    assert isinstance(environment, FakeEnvironment)
    backend_kwargs = captured["backend"]
    assert backend_kwargs["eef_control"] is True
    assert backend_kwargs["eef_left_topic"] == "/left_pose"
    assert backend_kwargs["eef_right_topic"] == "/right_pose"
    assert backend_kwargs["eef_left_action_topic"] == "/left_cmd"
    assert backend_kwargs["eef_right_action_topic"] == "/right_cmd"
    assert backend_kwargs["control_stack"] == "local-ros"
    env_kwargs = captured["environment"]
    assert env_kwargs["prompt"] == "Stack the cups"
    assert env_kwargs["max_action_delta"] == pytest.approx(20.0)
    assert env_kwargs["action_delta_limits"] == pytest.approx(
        (0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.02) * 2
    )
    assert env_kwargs["action_names"][5] == "left_yaw"
    assert env_kwargs["ros2_config"] == Path(__file__).resolve().parents[1] / "ros2_piper_dual_local.yaml"
    assert env_kwargs["control_stack"] == "local-ros"
    assert env_kwargs["observation_adapter"].__class__.__name__ == "EefXyz3dObservationAdapter"
    assert env_kwargs["action_adapter"].__class__.__name__ == "EefXyz3dActionAdapter"


def test_main_rejects_unsafe_publish_before_environment_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fail_build(_: main_dual_eef_xyz3d_ros.Args) -> object:
        nonlocal called
        called = True
        raise AssertionError("environment must not be created")

    monkeypatch.setattr(main_dual_eef_xyz3d_ros, "_build_environment", fail_build)

    result = main_dual_eef_xyz3d_ros.main(
        main_dual_eef_xyz3d_ros.Args(publish_actions=True, dry_run=True)
    )

    assert result == 2
    assert called is False


def test_parser_trace_is_opt_in(tmp_path: Path) -> None:
    assert main_dual_eef_xyz3d_ros.parse_args([]).trace_dir is None
    assert main_dual_eef_xyz3d_ros.parse_args(["--trace-dir", str(tmp_path)]).trace_dir == tmp_path


@pytest.mark.parametrize("existing", ["client.jsonl", "client_metadata.json"])
def test_main_trace_refuses_overwrite_before_environment_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: str
) -> None:
    (tmp_path / existing).write_text("preserved")
    constructed = []
    monkeypatch.setattr(main_dual_eef_xyz3d_ros, "_build_environment", lambda args: constructed.append(args))
    monkeypatch.setattr(main_dual_eef_xyz3d_ros.signal, "signal", lambda *args: None)
    assert main_dual_eef_xyz3d_ros.main(main_dual_eef_xyz3d_ros.Args(trace_dir=tmp_path)) == 1
    assert not constructed
    assert (tmp_path / existing).read_text() == "preserved"


@pytest.mark.parametrize("failure", [None, RuntimeError, KeyboardInterrupt])
def test_main_trace_metadata_precedes_environment_and_flushes_on_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type | None
) -> None:
    import json
    import numpy as np
    from eef_trace_recorder import EefTraceRecorder

    observed = {}
    original_close = EefTraceRecorder.close

    def close(recorder, *args, **kwargs):
        observed["recorder_closed"] = True
        return original_close(recorder, *args, **kwargs)

    class Environment:
        def close(self):
            observed["environment_closed"] = True

    def build_environment(args):
        metadata = json.loads((tmp_path / "client_metadata.json").read_text())
        assert metadata["args"]["trace_dir"] == str(tmp_path)
        assert metadata["args"]["action_horizon"] == args.action_horizon
        assert metadata["args"]["use_rtc"] == args.use_rtc
        assert metadata["args"]["host"] == args.host
        assert metadata["args"]["port"] == args.port
        assert metadata["args"]["prompt"] == args.prompt
        assert metadata["units"]["xyz"] == "m"
        assert metadata["units"]["rpy"] == "rad"
        assert "time.time_ns" in metadata["clocks"]["timestamp_ns"]
        assert "time.monotonic_ns" in metadata["clocks"]["monotonic_ns"]
        return Environment()

    def build_runtime(args, environment, policy, *, recorder=None):
        assert recorder is not None

        class Runtime:
            def run(self):
                recorder.on_episode_start()
                recorder.on_step({"state": np.zeros(14)}, {"actions": np.ones(14)})
                if failure is not None:
                    raise failure("test runtime stopped")
                recorder.on_episode_end()

        return Runtime()

    monkeypatch.setattr(EefTraceRecorder, "close", close)
    monkeypatch.setattr(main_dual_eef_xyz3d_ros, "_build_environment", build_environment)
    monkeypatch.setattr(main_dual_eef_xyz3d_ros, "_build_policy", lambda args: object())
    monkeypatch.setattr(main_dual_eef_xyz3d_ros, "_build_runtime", build_runtime)
    monkeypatch.setattr(main_dual_eef_xyz3d_ros.signal, "signal", lambda *args: None)
    result = main_dual_eef_xyz3d_ros.main(main_dual_eef_xyz3d_ros.Args(trace_dir=tmp_path))
    assert result == (0 if failure is None else 130 if failure is KeyboardInterrupt else 1)
    assert observed == {"environment_closed": True, "recorder_closed": True}
    records = [json.loads(line) for line in (tmp_path / "client.jsonl").read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "server_metadata", "episode_start", "action_submitted", "episode_end"
    ]
    assert records[0]["metadata"] == {}
    if failure is not None:
        assert records[-1]["reason"] == ("interrupted" if failure is KeyboardInterrupt else "error")


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_main_trace_closes_when_environment_construction_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type
) -> None:
    from eef_trace_recorder import EefTraceRecorder

    closed = []
    original_close = EefTraceRecorder.close

    def close(recorder, *args, **kwargs):
        closed.append(recorder)
        return original_close(recorder, *args, **kwargs)

    def build_environment(args):
        raise failure("construction stopped")

    monkeypatch.setattr(EefTraceRecorder, "close", close)
    monkeypatch.setattr(main_dual_eef_xyz3d_ros, "_build_environment", build_environment)
    monkeypatch.setattr(main_dual_eef_xyz3d_ros.signal, "signal", lambda *args: None)
    result = main_dual_eef_xyz3d_ros.main(main_dual_eef_xyz3d_ros.Args(trace_dir=tmp_path))
    assert result == (130 if failure is KeyboardInterrupt else 1)
    assert len(closed) == 1


def test_build_runtime_records_only_after_action_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    from types import SimpleNamespace
    import numpy as np
    from eef_trace_recorder import EefTraceRecorder

    class Environment:
        reject = False

        def get_observation(self):
            return {"state": np.zeros(14)}

        def apply_action(self, action):
            assert not (tmp_path / "client.jsonl").read_text().count("action_submitted")
            if self.reject:
                raise ValueError("submission rejected")

        def is_episode_complete(self):
            return True

    recorder = EefTraceRecorder(tmp_path, {}, flush_every=1)
    monkeypatch.setitem(sys.modules, "tshape_video_saver", SimpleNamespace(TShapeVideoSaver=lambda path, fps: object()))
    environment = Environment()
    policy = SimpleNamespace(infer=lambda obs: {"actions": np.ones(14), "_chunk_trace": {"chunk_id": 9}})
    runtime = main_dual_eef_xyz3d_ros._build_runtime(
        main_dual_eef_xyz3d_ros.Args(trace_dir=tmp_path), environment, policy, recorder=recorder
    )
    assert recorder in runtime._subscribers
    runtime._subscribers = [recorder]
    recorder.on_episode_start()
    environment.reject = True
    with pytest.raises(ValueError, match="submission rejected"):
        runtime._step()
    environment.reject = False
    runtime._step()
    recorder.close()
    records = [json.loads(line) for line in (tmp_path / "client.jsonl").read_text().splitlines()]
    submitted = [record for record in records if record["event"] == "action_submitted"]
    assert len(submitted) == 1
    assert submitted[0]["_chunk_trace"] == {"chunk_id": 9}


@pytest.mark.parametrize("use_async,use_rtc", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("traced", [False, True])
def test_build_policy_trace_wiring_preserves_all_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_async: bool, use_rtc: bool, traced: bool
) -> None:
    from types import SimpleNamespace
    import numpy as np
    import openpi_client

    policy = SimpleNamespace(infer=lambda obs: {"actions": np.zeros((50, 14))}, reset=lambda: None)
    transport = SimpleNamespace(WebsocketClientPolicy=lambda **kwargs: policy)
    monkeypatch.setitem(sys.modules, "openpi_client.websocket_client_policy", transport)
    monkeypatch.setattr(openpi_client, "websocket_client_policy", transport, raising=False)
    args = main_dual_eef_xyz3d_ros.Args(
        use_async=use_async, use_rtc=use_rtc, trace_dir=tmp_path if traced else None
    )
    broker = main_dual_eef_xyz3d_ros._build_policy(args)
    result = broker.infer({"state": np.zeros(14)})
    assert ("_chunk_trace" in result) is traced
    if traced:
        trace = result["_chunk_trace"]
        assert trace["use_rtc"] is (use_async and use_rtc)
        assert trace["configured_delay_steps"] == (args.actions_during_latency if use_async else 0)
        assert trace["action_horizon"] == args.action_horizon


@pytest.mark.parametrize("closing", ["runtime", "environment"])
def test_main_trace_flushes_even_when_other_cleanup_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, closing: str
) -> None:
    import json
    import numpy as np

    class Environment:
        def close(self):
            if closing == "environment":
                raise RuntimeError("environment close failed")

    def build_runtime(args, environment, policy, *, recorder=None):
        class Runtime:
            def run(self):
                recorder.on_episode_start()
                recorder.on_step({"state": np.zeros(14)}, {"actions": np.ones(14)})

            def close(self):
                if closing == "runtime":
                    raise RuntimeError("runtime close failed")

        return Runtime()

    monkeypatch.setattr(main_dual_eef_xyz3d_ros, "_build_environment", lambda args: Environment())
    monkeypatch.setattr(main_dual_eef_xyz3d_ros, "_build_policy", lambda args: object())
    monkeypatch.setattr(main_dual_eef_xyz3d_ros, "_build_runtime", build_runtime)
    monkeypatch.setattr(main_dual_eef_xyz3d_ros.signal, "signal", lambda *args: None)
    with pytest.raises(RuntimeError, match=f"{closing} close failed"):
        main_dual_eef_xyz3d_ros.main(main_dual_eef_xyz3d_ros.Args(trace_dir=tmp_path))
    records = [json.loads(line) for line in (tmp_path / "client.jsonl").read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "server_metadata", "episode_start", "action_submitted", "episode_end"
    ]


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("mode", ["legacy", "wrapped-rpy", "so3", None])
def test_main_records_server_provenance_once_before_episodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_async: bool, mode: str | None
) -> None:
    import json
    from types import SimpleNamespace
    import numpy as np
    import openpi_client

    trace_dir = tmp_path / "directory-label-not-authoritative"
    metadata = (
        {
            "rtc_orientation": {"mode": mode},
            "policy_config": "pi05_piper_dual_stack_cups_eef_xyz3d",
            "checkpoint_dir": "/checkpoints/xyz3d test",
            "existing_server_field": [1, "preserved"],
        }
        if mode is not None
        else {}
    )
    metadata_calls = []
    initial_metadata = []

    def get_server_metadata():
        metadata_calls.append(True)
        return metadata

    def connect(**kwargs):
        initial_metadata.append((trace_dir / "client_metadata.json").read_bytes())
        return SimpleNamespace(
            infer=lambda obs: {"actions": np.ones((50, 14))},
            reset=lambda: None,
            get_server_metadata=get_server_metadata,
        )

    transport = SimpleNamespace(WebsocketClientPolicy=connect)
    monkeypatch.setitem(sys.modules, "openpi_client.websocket_client_policy", transport)
    monkeypatch.setattr(openpi_client, "websocket_client_policy", transport, raising=False)

    def build_runtime(args, environment, policy, *, recorder=None):
        assert recorder is not None
        records = [json.loads(line) for line in (trace_dir / "client.jsonl").read_text().splitlines()]
        assert [record["event"] for record in records] == ["server_metadata"]
        assert records[0]["metadata"] == metadata
        assert (trace_dir / "client_metadata.json").read_bytes() == initial_metadata[0]

        def run():
            for _ in range(2):
                recorder.on_episode_start()
                observation = {"state": np.zeros(14)}
                recorder.on_step(observation, policy.infer(observation))
                recorder.on_episode_end()

        return SimpleNamespace(run=run)

    monkeypatch.setattr(main_dual_eef_xyz3d_ros, "_build_environment", lambda args: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(main_dual_eef_xyz3d_ros, "_build_runtime", build_runtime)
    monkeypatch.setattr(main_dual_eef_xyz3d_ros.signal, "signal", lambda *args: None)
    args = main_dual_eef_xyz3d_ros.Args(trace_dir=trace_dir, use_async=use_async)
    assert main_dual_eef_xyz3d_ros.main(args) == 0
    assert len(metadata_calls) == 1
    records = [json.loads(line) for line in (trace_dir / "client.jsonl").read_text().splitlines()]
    assert [record["event"] for record in records] == ["server_metadata"] + [
        "episode_start", "action_submitted", "episode_end"
    ] * 2
    assert records[0]["metadata"] == metadata
    assert (trace_dir / "client_metadata.json").read_bytes() == initial_metadata[0]


def test_main_without_trace_never_reads_server_metadata_or_constructs_recorder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace
    import eef_trace_recorder

    def unexpected(*args, **kwargs):
        raise AssertionError("tracing is disabled")

    ran = []
    policy = SimpleNamespace(get_server_metadata=unexpected)
    monkeypatch.setattr(eef_trace_recorder, "EefTraceRecorder", unexpected)
    monkeypatch.setattr(main_dual_eef_xyz3d_ros, "_build_environment", lambda args: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(main_dual_eef_xyz3d_ros, "_build_policy", lambda args: policy)
    monkeypatch.setattr(
        main_dual_eef_xyz3d_ros,
        "_build_runtime",
        lambda args, environment, policy: SimpleNamespace(run=lambda: ran.append(True)),
    )
    monkeypatch.setattr(main_dual_eef_xyz3d_ros.signal, "signal", lambda *args: None)
    assert main_dual_eef_xyz3d_ros.main(main_dual_eef_xyz3d_ros.Args(out_dir=tmp_path)) == 0
    assert ran == [True]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_main_preserves_server_metadata_when_runtime_construction_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type
) -> None:
    import json
    from types import SimpleNamespace

    metadata = {"rtc_orientation": {"mode": "so3"}, "checkpoint_dir": "/checkpoints/xyz3d"}
    closed = []

    def build_runtime(args, environment, policy, *, recorder=None):
        raise failure("runtime construction stopped")

    monkeypatch.setattr(
        main_dual_eef_xyz3d_ros, "_build_environment", lambda args: SimpleNamespace(close=lambda: closed.append(True))
    )
    monkeypatch.setattr(
        main_dual_eef_xyz3d_ros, "_build_policy", lambda args: SimpleNamespace(get_server_metadata=lambda: metadata)
    )
    monkeypatch.setattr(main_dual_eef_xyz3d_ros, "_build_runtime", build_runtime)
    monkeypatch.setattr(main_dual_eef_xyz3d_ros.signal, "signal", lambda *args: None)
    result = main_dual_eef_xyz3d_ros.main(main_dual_eef_xyz3d_ros.Args(trace_dir=tmp_path))
    assert result == (130 if failure is KeyboardInterrupt else 1)
    assert closed == [True]
    records = [json.loads(line) for line in (tmp_path / "client.jsonl").read_text().splitlines()]
    assert [record["event"] for record in records] == ["server_metadata"]
    assert records[0]["metadata"] == metadata
