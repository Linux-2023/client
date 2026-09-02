#!/usr/bin/env python3
"""ROS 2-only dual-arm Piper XYZ+RPY EEF policy deployment entrypoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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


@dataclass
class Args:
    out_dir: Path = Path("data/piper_dual_eef_xyz3d/videos")
    action_horizon: int = 16
    fps: int = 30
    actions_during_latency: int = 8
    num_steps: int = 8000
    num_episodes: int = 1
    run_tag: str = ""
    bridge_python: Path = DEFAULT_BRIDGE_PYTHON
    ros2_config: Path = DEFAULT_ROS2_CONFIG
    dry_run: bool = True
    publish_actions: bool = False
    max_action_delta: float | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    prompt: str = "Stack_the_paper_cups_together."
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
    _effective_max_action_delta(args)
    if not args.eef_left_topic or not args.eef_right_topic:
        raise ValueError("EEF observation topics must both be non-empty")
    if not args.eef_left_action_topic or not args.eef_right_action_topic:
        raise ValueError("EEF action topics must both be non-empty")


def contract_summary(args: Args) -> str:
    contract = _selected_bridge_contract(args)
    effective_max_delta = _effective_max_action_delta(args, contract)
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
        )
    return action_chunk_broker.ActionChunkBroker(
        policy=base_policy,
        action_horizon=args.action_horizon,
        fps=args.fps,
    )


def _build_runtime(args: Args, environment: Ros2DualEnvironment, policy: Any) -> Any:
    from openpi_client.runtime import runtime
    from openpi_client.runtime.agents import policy_agent
    import saver

    return runtime.Runtime(
        environment=environment,
        agent=policy_agent.PolicyAgent(policy=policy),
        subscribers=[saver.VideoSaver(args.out_dir)],
        max_hz=args.fps,
        num_episodes=args.num_episodes,
    )


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

    def signal_handler(sig: int, frame: Any) -> None:
        del sig, frame
        print("\nStopping XYZ3D EEF ROS 2 deployment...")
        if environment is not None:
            environment.close()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        print("\nInitializing XYZ3D EEF ROS 2 environment...")
        environment = _build_environment(args)
        print("XYZ3D EEF ROS 2 environment initialized")
        print("\nConnecting to policy server...")
        policy = _build_policy(args)
        print("Policy server connected")
        runtime_instance = _build_runtime(args, environment, policy)
        print(f"\nTask: {args.prompt}\n\nRunning XYZ3D EEF policy...")
        runtime_instance.run()
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"\nRuntime error: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
    finally:
        if runtime_instance is not None and hasattr(runtime_instance, "close"):
            runtime_instance.close()
        if environment is not None:
            environment.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    raise SystemExit(main())
