#!/usr/bin/env python3
"""ROS 2-only dual-arm Piper XYZ+RPY EEF policy deployment entrypoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import logging
import math
from pathlib import Path
import signal
import sys
from typing import Any, Literal

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OPENPI_CLIENT_SRC = _PROJECT_ROOT / "packages/openpi-client/src"
if str(_OPENPI_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(_OPENPI_CLIENT_SRC))

from control_stack import load_profile_config
from eef_xyz3d_action_adapter import EefXyz3dActionAdapter
from eef_xyz3d_observation_adapter import EefXyz3dObservationAdapter
from ros2_backend import Ros2BackendClient
from ros2_contract import BridgeContract
from ros2_environment import DEFAULT_BRIDGE_PYTHON
from ros2_environment import DEFAULT_FRAME_TIMEOUT
from ros2_environment import DEFAULT_ROS2_CONFIG
from ros2_environment import DEFAULT_WATCHDOG_TIMEOUT
from ros2_environment import Ros2DualEnvironment

DEFAULT_EEF_LEFT_TOPIC = "/puppet/end_pose_left"
DEFAULT_EEF_RIGHT_TOPIC = "/puppet/end_pose_right"
DEFAULT_EEF_LEFT_ACTION_TOPIC = "/pos_left_cmd"
DEFAULT_EEF_RIGHT_ACTION_TOPIC = "/pos_right_cmd"
EEF_ACTION_NAMES = tuple(
    f"{arm}_{axis}"
    for arm in ("left", "right")
    for axis in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
)


@dataclass
class Args:
    out_dir: Path = Path("data/piper_dual_eef_xyz3d/videos")
    trace_dir: Path | None = None
    # TODO
    # action_horizon: int = 40
    # action_horizon: int = 30
    action_horizon: int = 20
    fps: int = 30
    actions_during_latency: int = 12
    num_steps: int = 8000
    num_episodes: int = 1
    run_tag: str = ""
    bridge_python: Path = DEFAULT_BRIDGE_PYTHON
    ros2_config: Path = DEFAULT_ROS2_CONFIG
    dry_run: bool = True
    publish_actions: bool = False
    max_action_delta: float | None = None
    max_position_delta: float | None = None
    max_orientation_delta: float | None = None
    max_gripper_delta: float | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    # TODO
    # prompt: str = "Stack_the_paper_cups_together."
    prompt: str = "Fold_the_towel."
    # prompt: str = "Beat_the_drum_three_times."
    # prompt: str = "Weigh_the_apple."
    # prompt: str = "Put_the_block_in_the_drawer."
    # prompt: str = "Clean_the_table"
    use_async: bool = True
    use_rtc: bool = True
    eef_left_topic: str = DEFAULT_EEF_LEFT_TOPIC
    eef_right_topic: str = DEFAULT_EEF_RIGHT_TOPIC
    eef_left_action_topic: str = DEFAULT_EEF_LEFT_ACTION_TOPIC
    eef_right_action_topic: str = DEFAULT_EEF_RIGHT_ACTION_TOPIC
    control_stack: Literal["local-ros", "official-ros", "direct-sdk"] = "official-ros"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deploy a dual-arm Piper XYZ+RPY EEF policy through ROS 2 PosCmd control.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir", type=Path, default=Args.out_dir)
    parser.add_argument("--trace-dir", type=Path, default=Args.trace_dir, help="Record EEF submissions and chunk metadata without overwriting existing client trace files")
    parser.add_argument(
        "--action-horizon",
        "--action_horizon",
        dest="action_horizon",
        type=int,
        default=Args.action_horizon,
    )
    parser.add_argument("--fps", type=int, default=Args.fps)
    parser.add_argument(
        "--actions-during-latency",
        "--actions_during_latency",
        dest="actions_during_latency",
        type=int,
        default=Args.actions_during_latency,
    )
    parser.add_argument("--num-steps", type=int, default=Args.num_steps)
    parser.add_argument("--num-episodes", type=int, default=Args.num_episodes)
    parser.add_argument("--run-tag", type=str, default=Args.run_tag)
    parser.add_argument("--bridge-python", type=Path, default=Args.bridge_python)
    parser.add_argument("--ros2-config", type=Path, default=Args.ros2_config)
    parser.add_argument(
        "--control-stack",
        choices=("local-ros", "official-ros", "direct-sdk"),
        default=Args.control_stack,
    )
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=Args.dry_run)
    parser.add_argument("--publish-actions", action="store_true", default=Args.publish_actions)
    parser.add_argument("--max-action-delta", type=float, default=Args.max_action_delta)
    parser.add_argument("--max-position-delta", type=float, default=Args.max_position_delta)
    parser.add_argument("--max-orientation-delta", type=float, default=Args.max_orientation_delta)
    parser.add_argument("--max-gripper-delta", type=float, default=Args.max_gripper_delta)
    parser.add_argument("--host", type=str, default=Args.host)
    parser.add_argument("--port", type=int, default=Args.port)
    parser.add_argument("--prompt", type=str, default=Args.prompt)
    parser.add_argument("--use-async", action=argparse.BooleanOptionalAction, default=Args.use_async)
    parser.add_argument("--use-rtc", action=argparse.BooleanOptionalAction, default=Args.use_rtc)
    parser.add_argument("--eef-left-topic", default=Args.eef_left_topic)
    parser.add_argument("--eef-right-topic", default=Args.eef_right_topic)
    parser.add_argument("--eef-left-action-topic", default=Args.eef_left_action_topic)
    parser.add_argument("--eef-right-action-topic", default=Args.eef_right_action_topic)
    return parser


def parse_args(argv: list[str] | None = None) -> Args:
    return Args(**vars(build_arg_parser().parse_args(argv)))


def _selected_bridge_contract(args: Args) -> BridgeContract:
    _profile, config = load_profile_config(args.control_stack, args.ros2_config)
    return BridgeContract.from_mapping(config)


def _effective_max_action_delta(args: Args, contract: BridgeContract | None = None) -> float:
    selected_contract = contract if contract is not None else _selected_bridge_contract(args)
    if args.max_action_delta is None:
        return selected_contract.control.max_action_delta
    requested = float(args.max_action_delta)
    if not math.isfinite(requested) or requested < 0:
        raise ValueError("--max-action-delta must be finite and non-negative")
    return requested


def _validate_xyz3d_contract(args: Args) -> None:
    if args.publish_actions and args.dry_run:
        raise ValueError("--publish-actions requires --no-dry-run")
    _effective_action_delta_limits(args)
    if not args.eef_left_topic or not args.eef_right_topic:
        raise ValueError("EEF observation topics must both be non-empty")
    if not args.eef_left_action_topic or not args.eef_right_action_topic:
        raise ValueError("EEF action topics must both be non-empty")


def _effective_action_delta_limits(args: Args) -> tuple[float, ...]:
    fallback = _effective_max_action_delta(args)
    requested = {
        "position": fallback if args.max_position_delta is None else float(args.max_position_delta),
        "orientation": fallback if args.max_orientation_delta is None else float(args.max_orientation_delta),
        "gripper": fallback if args.max_gripper_delta is None else float(args.max_gripper_delta),
    }
    for name, value in requested.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--max-{name}-delta must be finite and non-negative")
    per_arm = (
        requested["position"], requested["position"], requested["position"],
        requested["orientation"], requested["orientation"], requested["orientation"],
        requested["gripper"],
    )
    return per_arm + per_arm


def contract_summary(args: Args) -> str:
    contract = _selected_bridge_contract(args)
    effective_max_delta = _effective_max_action_delta(args, contract)
    delta_limits = _effective_action_delta_limits(args)
    return "\n".join(
        [
            "ROS 2 XYZ3D EEF deployment contract:",
            f"- selected_stack={args.control_stack}",
            "- expected_control_mode=MOVE P (mode_feedback=0)",
            "- policy state/action = 14D [left_xyzrpy,left_gripper,right_xyzrpy,right_gripper]",
            f"- action_horizon={args.action_horizon}",
            f"- dry_run={args.dry_run}",
            f"- publish_actions={args.publish_actions}",
            "- control_mode=MOVE L/P selected by PosCmd; mode_feedback is not pre-gated",
            f"- effective_max_action_delta={effective_max_delta:g}",
            f"- effective_position_delta_limit={delta_limits[0]:g} m",
            f"- effective_orientation_delta_limit={delta_limits[3]:g} rad",
            f"- effective_gripper_delta_limit={delta_limits[6]:g} m",
            f"- selected_config_default_max_action_delta={contract.control.max_action_delta:g}",
            f"- eef_observation_topics={args.eef_left_topic}, {args.eef_right_topic}",
            f"- eef_action_topics={args.eef_left_action_topic}, {args.eef_right_action_topic}",
            f"- bridge_python={args.bridge_python}",
            f"- ros2_config={args.ros2_config}",
            f"- policy_server={args.host}:{args.port}",
        ]
    )


def _build_environment(args: Args) -> Ros2DualEnvironment:
    effective_max_delta = _effective_max_action_delta(args)
    delta_limits = _effective_action_delta_limits(args)
    action_adapter = EefXyz3dActionAdapter()
    backend = Ros2BackendClient(
        bridge_python=args.bridge_python,
        config=args.ros2_config,
        dry_run=args.dry_run,
        publish_actions=args.publish_actions,
        control_stack=args.control_stack,
        action_adapter=action_adapter,
        eef_control=True,
        eef_left_topic=args.eef_left_topic,
        eef_right_topic=args.eef_right_topic,
        eef_left_action_topic=args.eef_left_action_topic,
        eef_right_action_topic=args.eef_right_action_topic,
    )
    return Ros2DualEnvironment(
        backend=backend,
        ros2_config=args.ros2_config,
        dry_run=args.dry_run,
        publish_actions=args.publish_actions,
        control_stack=args.control_stack,
        observation_adapter=EefXyz3dObservationAdapter(task=args.prompt),
        action_adapter=action_adapter,
        prompt=args.prompt,
        max_episode_steps=args.num_steps,
        watchdog_timeout=DEFAULT_WATCHDOG_TIMEOUT,
        frame_timeout=DEFAULT_FRAME_TIMEOUT,
        max_action_delta=effective_max_delta,
        action_delta_limits=delta_limits,
        action_names=EEF_ACTION_NAMES,
    )


def _build_policy(args: Args) -> Any:
    from openpi_client import action_chunk_broker
    from openpi_client import websocket_client_policy

    base_policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    if args.use_async:
        return action_chunk_broker.ActionChunkBroker_RTC(
            policy=base_policy,
            action_horizon=args.action_horizon,
            fps=args.fps,
            actions_during_latency=args.actions_during_latency,
            use_rtc=args.use_rtc,
            trace_enabled=args.trace_dir is not None,
            handoff_blend_steps=args.actions_during_latency,
        )
    return action_chunk_broker.ActionChunkBroker(
        policy=base_policy,
        action_horizon=args.action_horizon,
        fps=args.fps,
        trace_enabled=args.trace_dir is not None,
    )


def _build_runtime(args: Args, environment: Ros2DualEnvironment, policy: Any, *, recorder: Any | None = None) -> Any:
    from openpi_client.runtime import runtime
    from openpi_client.runtime.agents import policy_agent
    from tshape_video_saver import TShapeVideoSaver

    video_recorder = TShapeVideoSaver(args.out_dir, fps=args.fps)
    subscribers = [video_recorder]
    if recorder is not None:
        # Record accepted submissions before video I/O can fail. Both subscribers
        # retain the existing runtime on_step(observation, action) contract.
        subscribers.insert(0, recorder)

    instance = runtime.Runtime(
        environment=environment,
        agent=policy_agent.PolicyAgent(policy=policy),
        subscribers=subscribers,
        max_hz=args.fps,
        num_episodes=args.num_episodes,
    )
    instance.video_recorder = video_recorder
    return instance


def _trace_metadata(args: Args) -> dict[str, Any]:
    return {
        "args": {name: str(value) if isinstance(value, Path) else value for name, value in asdict(args).items()},
        "layout": [f"{arm}_{axis}" for arm in ("left", "right") for axis in ("x", "y", "z", "roll", "pitch", "yaw", "gripper")],
        "units": {"xyz": "m", "rpy": "rad", "gripper": "m"},
        "clocks": {
            "timestamp_ns": "time.time_ns(): host wall clock, nanoseconds since Unix epoch; may be adjusted",
            "monotonic_ns": "time.monotonic_ns(): host monotonic clock, nanoseconds; same-host alignment only",
            "selected_timestamp_ns": "Broker selection uses time.time_ns() before returning the action",
            "selected_monotonic_ns": "Broker selection uses time.monotonic_ns() before returning the action",
        },
        "semantics": {
            "action_submitted": "Recorded after environment.apply_action returns; submission accepted, NOT physical execution or confirmed CAN dispatch. Dry-run can submit without publishing.",
            "action": "14D policy EEF action returned by the broker and submitted to apply_action, before backend conversion",
            "observation_state": "14D EEF state in the observation obtained before action selection; not joint feedback or a simultaneous tracking-error measurement",
            "episode": "Zero-based runtime episode index",
            "step": "Zero-based action submission index, increasing across episodes; boundary records use the next action index",
            "chunk_id": "Zero-based adopted chunk index, reset with broker.reset() for each episode",
            "chunk_step": "Index in the original full returned chunk, including any skipped prefix",
            "chunk_boundary": "First action selected from the adopted chunk, including when a prefix was skipped",
            "skipped_steps": "Prefix length skipped when this chunk was adopted; constant for that chunk",
            "inference_ms": "Observed client policy-call duration in milliseconds for this chunk, not server-only inference time",
        },
    }


def main(args: Args | argparse.Namespace | None = None) -> int:
    if args is None:
        args = parse_args()
    elif not isinstance(args, Args):
        args = Args(**vars(args))
    try:
        _validate_xyz3d_contract(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(contract_summary(args))
    environment: Ros2DualEnvironment | None = None
    runtime_instance: Any | None = None
    recorder: Any | None = None
    exit_reason = "error"

    def signal_handler(sig: int, frame: Any) -> None:
        del sig, frame
        print("\nStopping XYZ3D EEF ROS 2 deployment...")
        # All resources, including buffered trace data, close in the common finally.
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        if args.trace_dir is not None:
            from eef_trace_recorder import EefTraceRecorder

            recorder = EefTraceRecorder(args.trace_dir, _trace_metadata(args))
        print("\nInitializing XYZ3D EEF ROS 2 environment...")
        environment = _build_environment(args)
        print("XYZ3D EEF ROS 2 environment initialized")
        print("\nConnecting to policy server...")
        policy = _build_policy(args)
        print("Policy server connected")
        if recorder is not None:
            get_metadata = getattr(policy, "get_server_metadata", None)
            recorder.record_server_metadata(get_metadata() if callable(get_metadata) else {})
        runtime_instance = (
            _build_runtime(args, environment, policy, recorder=recorder)
            if recorder is not None
            else _build_runtime(args, environment, policy)
        )
        print(f"\nTask: {args.prompt}\n\nRunning XYZ3D EEF policy...")
        runtime_instance.run()
        exit_reason = "completed"
        return 0
    except KeyboardInterrupt:
        exit_reason = "interrupted"
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"\nRuntime error: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
    finally:
        try:
            if runtime_instance is not None and hasattr(runtime_instance, "close"):
                runtime_instance.close()
        finally:
            try:
                if environment is not None:
                    environment.close()
            finally:
                try:
                    video_recorder = getattr(runtime_instance, "video_recorder", None)
                    if video_recorder is not None:
                        video_recorder.close()
                finally:
                    if recorder is not None:
                        recorder.close(reason=exit_reason)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    raise SystemExit(main())
