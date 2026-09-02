#!/usr/bin/env python3
"""ROS 2-only dual-arm Piper PI05 deployment entrypoint."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import signal
import sys
from typing import Any
from typing import Literal

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OPENPI_CLIENT_SRC = _PROJECT_ROOT / "packages/openpi-client/src"
if str(_OPENPI_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(_OPENPI_CLIENT_SRC))

from ros2_environment import DEFAULT_ROS2_CONFIG
from ros2_environment import Ros2DualEnvironment

DEFAULT_BRIDGE_PYTHON = Path("/usr/bin/python3")
DEFAULT_WATCHDOG_TIMEOUT = 5.0


@dataclass
class Args:
    """Arguments for the ROS 2-only dual-arm deployment."""

    out_dir: Path = Path("data/piper_dual/videos")
    action_horizon: int = 18
    fps: int = 30
    actions_during_latency: int = 8
    num_steps: int = 1600
    num_episodes: int = 1
    run_tag: str = ""
    bridge_python: Path = DEFAULT_BRIDGE_PYTHON
    ros2_config: Path = DEFAULT_ROS2_CONFIG
    dry_run: bool = True
    publish_actions: bool = False
    control_stack: Literal["local-ros", "official-ros", "direct-sdk"] = "official-ros"
    max_action_delta: float | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    prompt: str = "Stack_the_paper_cups_together."
    use_async: bool = True
    use_rtc: bool = True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deploy the dual-arm Piper PI05 policy through the ROS 2 backend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir", type=Path, default=Args.out_dir, help="Directory for saved outputs")
    parser.add_argument(
        "--action-horizon",
        "--action_horizon",
        dest="action_horizon",
        type=int,
        default=Args.action_horizon,
        help="Number of actions retained from each remote action chunk",
    )
    parser.add_argument("--fps", type=int, default=Args.fps, help="Runtime frame rate")
    parser.add_argument(
        "--actions-during-latency",
        "--actions_during_latency",
        dest="actions_during_latency",
        type=int,
        default=Args.actions_during_latency,
        help="Actions queued while waiting for the next policy result",
    )
    parser.add_argument("--num-steps", type=int, default=Args.num_steps, help="Maximum steps per episode")
    parser.add_argument("--num-episodes", type=int, default=Args.num_episodes, help="Number of episodes")
    parser.add_argument("--run-tag", type=str, default=Args.run_tag, help="Optional output label")
    parser.add_argument(
        "--bridge-python",
        type=Path,
        default=Args.bridge_python,
        help="Python interpreter used to launch the ROS 2 bridge",
    )
    parser.add_argument(
        "--ros2-config",
        type=Path,
        default=Args.ros2_config,
        help="ROS 2 bridge configuration file",
    )
    parser.add_argument(
        "--control-stack",
        choices=("local-ros", "official-ros", "direct-sdk"),
        default=Args.control_stack,
        help="Select the ROS 2 control stack profile",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=Args.dry_run,
        help="Keep ROS 2 action publishing disabled",
    )
    parser.add_argument(
        "--publish-actions",
        action="store_true",
        default=Args.publish_actions,
        help="Allow validated actions to be published to ROS 2",
    )
    parser.add_argument(
        "--max-action-delta",
        type=float,
        default=Args.max_action_delta,
        help="Optional maximum per-step action change",
    )
    parser.add_argument("--host", type=str, default=Args.host, help="PI05 WebSocket server host")
    parser.add_argument("--port", type=int, default=Args.port, help="PI05 WebSocket server port")
    parser.add_argument("--prompt", type=str, default=Args.prompt, help="Task prompt")
    parser.add_argument(
        "--use-async",
        action=argparse.BooleanOptionalAction,
        default=Args.use_async,
        help="Use the asynchronous action-chunk broker",
    )
    parser.add_argument(
        "--use-rtc",
        action=argparse.BooleanOptionalAction,
        default=Args.use_rtc,
        help="Enable RTC mode in the asynchronous broker",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> Args:
    return Args(**vars(build_arg_parser().parse_args(argv)))


def _validate_ros2_contract(args: Args) -> None:
    if args.publish_actions and args.dry_run:
        raise ValueError("--publish-actions requires --no-dry-run")
    if args.max_action_delta is not None and (
        not math.isfinite(args.max_action_delta) or args.max_action_delta < 0
    ):
        raise ValueError("--max-action-delta must be finite and non-negative")


def contract_summary(args: Args) -> str:
    lines = [
        "ROS 2 backend deployment contract:",
        "- backend=ros2",
        f"- dry_run={args.dry_run}",
        f"- publish_actions={args.publish_actions}",
        f"- control_stack={args.control_stack}",
        "- publish-actions requires --no-dry-run",
        f"- bridge_python={args.bridge_python}",
        f"- ros2_config={args.ros2_config}",
        f"- policy_server={args.host}:{args.port}",
    ]
    if args.max_action_delta is not None:
        lines.append(f"- max_action_delta={args.max_action_delta}")
    return "\n".join(lines)


def _build_environment(args: Args) -> Ros2DualEnvironment:
    kwargs: dict[str, Any] = {
        "bridge_python": args.bridge_python,
        "ros2_config": args.ros2_config,
        "dry_run": args.dry_run,
        "publish_actions": args.publish_actions,
        "control_stack": args.control_stack,
        "prompt": args.prompt,
        "max_episode_steps": args.num_steps,
        "watchdog_timeout": DEFAULT_WATCHDOG_TIMEOUT,
    }
    if args.max_action_delta is not None:
        kwargs["max_action_delta"] = args.max_action_delta
    return Ros2DualEnvironment(**kwargs)


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
    from joint_trace_recorder import JointTraceRecorder
    import saver

    return runtime.Runtime(
        environment=environment,
        agent=policy_agent.PolicyAgent(policy=policy),
        subscribers=[
            saver.VideoSaver(args.out_dir, fps=args.fps),
            JointTraceRecorder(args.out_dir / "joint_traces"),
        ],
        max_hz=args.fps,
        num_episodes=args.num_episodes,
    )


def main(args: Args | argparse.Namespace | None = None) -> int:
    if args is None:
        args = parse_args()
    elif not isinstance(args, Args):
        args = Args(**vars(args))

    try:
        _validate_ros2_contract(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(contract_summary(args))
    print()

    environment: Ros2DualEnvironment | None = None
    runtime: Any | None = None

    def signal_handler(sig: int, frame: Any) -> None:
        del sig, frame
        print("\nStopping ROS 2 deployment...")
        if environment is not None:
            try:
                environment.close()
            except Exception as exc:  # noqa: BLE001
                print(f"Environment cleanup failed: {exc}", file=sys.stderr)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        print("=" * 60)
        print("Dual-arm Piper PI05 - ROS 2 deployment")
        print("=" * 60)
        print("\nInitializing ROS 2 environment...")
        environment = _build_environment(args)
        print("ROS 2 environment initialized")
        print(f"  bridge python: {args.bridge_python}")
        print(f"  ROS 2 config: {args.ros2_config}")
        print(f"  dry_run: {args.dry_run}")
        print(f"  publish_actions: {args.publish_actions}")

        print("\nConnecting to PI05 policy server...")
        policy = _build_policy(args)
        print("Policy server connected")
        print(f"  server: {args.host}:{args.port}")

        print("\nInitializing runtime...")
        runtime = _build_runtime(args, environment, policy)
        print("Runtime initialized")
        print(f"\nTask: {args.prompt}")
        print("\nRunning policy...")
        runtime.run()
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"\nRuntime error: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
    finally:
        if runtime is not None and hasattr(runtime, "close"):
            try:
                runtime.close()
            except Exception as exc:  # noqa: BLE001
                print(f"Runtime cleanup failed: {exc}", file=sys.stderr)
        if environment is not None:
            try:
                environment.close()
            except Exception as exc:  # noqa: BLE001
                print(f"Environment cleanup failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    raise SystemExit(main())
