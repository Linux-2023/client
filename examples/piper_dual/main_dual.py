#!/usr/bin/env python3
"""Dual-arm Piper deployment entrypoint with safe ROS 2 backend selection."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import signal
import sys
from typing import Any, Literal

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OPENPI_CLIENT_SRC = _PROJECT_ROOT / "packages/openpi-client/src"
if str(_OPENPI_CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(_OPENPI_CLIENT_SRC))


DEFAULT_BRIDGE_PYTHON = Path("/usr/bin/python3")
DEFAULT_ROS2_CONFIG = Path(__file__).resolve().with_name("ros2_piper_dual.yaml")
DEFAULT_WATCHDOG_TIMEOUT = 5.0


@dataclass
class Args:
    """Arguments for dual-arm deployment with ROS 2 safety gates."""

    out_dir: Path = Path("data/piper_dual/videos")
    seed: int = 0
    max_action_horizon: int = 50
    action_horizon: int = 10
    fps: int = 30
    actions_during_latency: int = 5
    num_steps: int = 800
    num_episodes: int = 1
    run_tag: str = ""
    mode: str = "remote"
    backend: Literal["sdk", "ros2"] = "ros2"
    bridge_python: Path = DEFAULT_BRIDGE_PYTHON
    ros2_config: Path = DEFAULT_ROS2_CONFIG
    dry_run: bool = True
    publish_actions: bool = False
    control_stack: Literal["local-ros", "official-ros", "direct-sdk"] | None = None
    max_action_delta: float | None = None
    host: str = "127.0.0.1"
    port: int = 8001
    display: bool = False
    high_camera_id: str = "148522073709"
    left_wrist_camera_id: int = 0
    right_wrist_camera_id: int = 8
    left_can_port: str = "can_left"
    right_can_port: str = "can_right"
    prompt: str = "Place the red and blue blocks on the wooden board"
    use_async: bool = False
    use_rtc: bool = False
    gripper_norm: bool = True
    tele_mode: bool = False
    record_mode: bool = False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deploy the dual-arm Piper policy with the SDK or safe ROS 2 backend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir", type=Path, default=Args.out_dir, help="Directory for saved outputs")
    parser.add_argument("--seed", type=int, default=Args.seed, help="Random seed for the environment")
    parser.add_argument(
        "--max-action-horizon",
        type=int,
        default=Args.max_action_horizon,
        help="Maximum chunk horizon kept for compatibility with the remote broker",
    )
    parser.add_argument(
        "--action-horizon",
        "--action_horizon",
        dest="action_horizon",
        type=int,
        default=Args.action_horizon,
        help="Remote action horizon",
    )
    parser.add_argument("--fps", type=int, default=Args.fps, help="Runtime frame rate")
    parser.add_argument(
        "--actions-during-latency",
        "--actions_during_latency",
        dest="actions_during_latency",
        type=int,
        default=Args.actions_during_latency,
        help="Extra actions queued during policy latency",
    )
    parser.add_argument("--num-steps", type=int, default=Args.num_steps, help="Maximum episode steps")
    parser.add_argument("--num-episodes", type=int, default=Args.num_episodes, help="Episode count to run")
    parser.add_argument("--run-tag", type=str, default=Args.run_tag, help="Optional run label for plots")
    parser.add_argument("--mode", choices=("remote", "local"), default=Args.mode, help="Policy mode")
    parser.add_argument("--backend", choices=("sdk", "ros2"), default=Args.backend, help="Environment backend")
    parser.add_argument(
        "--bridge-python",
        type=Path,
        default=DEFAULT_BRIDGE_PYTHON,
        help="Python interpreter used to launch the ROS 2 bridge",
    )
    parser.add_argument(
        "--ros2-config",
        type=Path,
        default=DEFAULT_ROS2_CONFIG,
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
        help="Keep the ROS 2 backend in no-action-publishing mode",
    )
    parser.add_argument(
        "--publish-actions",
        action="store_true",
        default=Args.publish_actions,
        help="Allow the ROS 2 backend to publish validated actions",
    )
    parser.add_argument(
        "--max-action-delta",
        type=float,
        default=Args.max_action_delta,
        help="Optional per-step action delta limit for the ROS 2 environment",
    )
    parser.add_argument("--host", type=str, default=Args.host, help="Remote websocket host")
    parser.add_argument("--port", type=int, default=Args.port, help="Remote websocket port")
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=Args.display,
        help="Show the USB camera preview window",
    )
    parser.add_argument(
        "--high-camera-id",
        "--high_camera_id",
        dest="high_camera_id",
        type=str,
        default=Args.high_camera_id,
        help="RealSense serial number",
    )
    parser.add_argument(
        "--left-wrist-camera-id",
        "--left_wrist_camera_id",
        dest="left_wrist_camera_id",
        type=int,
        default=Args.left_wrist_camera_id,
        help="Left wrist camera device ID",
    )
    parser.add_argument(
        "--right-wrist-camera-id",
        "--right_wrist_camera_id",
        dest="right_wrist_camera_id",
        type=int,
        default=Args.right_wrist_camera_id,
        help="Right wrist camera device ID",
    )
    parser.add_argument(
        "--left-can-port",
        "--left_can_port",
        dest="left_can_port",
        type=str,
        default=Args.left_can_port,
        help="Left arm CAN port",
    )
    parser.add_argument(
        "--right-can-port",
        "--right_can_port",
        dest="right_can_port",
        type=str,
        default=Args.right_can_port,
        help="Right arm CAN port",
    )
    parser.add_argument("--prompt", type=str, default=Args.prompt, help="Task prompt")
    parser.add_argument(
        "--use-async",
        action=argparse.BooleanOptionalAction,
        default=Args.use_async,
        help="Use the chunked remote policy wrapper",
    )
    parser.add_argument(
        "--use-rtc",
        "--use_rtc",
        dest="use_rtc",
        action=argparse.BooleanOptionalAction,
        default=Args.use_rtc,
        help="Enable RTC mode in the remote broker",
    )
    parser.add_argument(
        "--gripper-norm",
        action=argparse.BooleanOptionalAction,
        default=Args.gripper_norm,
        help="Keep grippers in normalized command space",
    )
    parser.add_argument(
        "--tele-mode",
        "--tele_mode",
        dest="tele_mode",
        action=argparse.BooleanOptionalAction,
        default=Args.tele_mode,
        help="Disable robot motion commands for safe testing",
    )
    parser.add_argument(
        "--record-mode",
        "--record_mode",
        dest="record_mode",
        action=argparse.BooleanOptionalAction,
        default=Args.record_mode,
        help="Record trajectory data during episodes",
    )
    return parser


def _namespace_to_args(namespace: argparse.Namespace) -> Args:
    return Args(**vars(namespace))


def parse_args(argv: list[str] | None = None) -> Args:
    return _namespace_to_args(build_arg_parser().parse_args(argv))


def _resolve_control_stack(args: Args) -> str | None:
    if args.backend == "sdk":
        return None
    return args.control_stack or "official-ros"


def _validate_backend_contract(args: Args) -> None:
    if args.control_stack is not None and args.backend != "ros2":
        raise ValueError("--control-stack requires --backend ros2")
    if args.publish_actions and args.backend != "ros2":
        raise ValueError("--publish-actions requires --backend ros2")
    if args.publish_actions and args.dry_run:
        raise ValueError("--publish-actions requires --no-dry-run")
    _comparison_out_dir(args)


def _comparison_out_dir(args: Args) -> Path:
    if not args.run_tag:
        return args.out_dir
    if args.run_tag in {".", ".."} or "/" in args.run_tag or "\\" in args.run_tag:
        raise ValueError("--run-tag must be a single safe path component")
    return args.out_dir / args.run_tag


def contract_summary(args: Args) -> str:
    control_stack = _resolve_control_stack(args)
    lines = [
        "Contract summary:",
        f"- backend={args.backend}",
        f"- selected control stack={control_stack}",
        f"- mode={args.mode}",
        f"- ROS 2 defaults to dry-run: {args.dry_run if args.backend == 'ros2' else True}",
        "- publish-actions requires --backend ros2 and --no-dry-run",
        f"- ROS 2 bridge python: {args.bridge_python}",
        f"- ROS 2 config: {args.ros2_config}",
    ]
    if args.max_action_delta is not None:
        lines.append(f"- max_action_delta={args.max_action_delta}")
    return "\n".join(lines)


def _environment_kwargs(args: Args) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "left_can_port": args.left_can_port,
        "right_can_port": args.right_can_port,
        "camera_fps": args.fps,
        "high_camera_id": args.high_camera_id,
        "left_wrist_camera_id": args.left_wrist_camera_id,
        "right_wrist_camera_id": args.right_wrist_camera_id,
        "max_episode_steps": args.num_steps,
        "seed": args.seed,
        "tele_mode": args.tele_mode,
        "prompt": args.prompt,
        "watchdog_timeout": DEFAULT_WATCHDOG_TIMEOUT,
        "show_usb_camera": args.display,
        "gripper_norm": args.gripper_norm,
        "action_horizon": args.action_horizon,
        "record_mode": args.record_mode,
    }
    if args.backend == "ros2":
        kwargs.update(
            bridge_python=args.bridge_python,
            ros2_config=args.ros2_config,
            dry_run=args.dry_run,
            publish_actions=args.publish_actions,
            control_stack=_resolve_control_stack(args),
        )
        if args.max_action_delta is not None:
            kwargs["max_action_delta"] = args.max_action_delta
    return kwargs


def _build_environment(args: Args) -> Any:
    from env_dual import create_dual_environment

    return create_dual_environment(backend=args.backend, **_environment_kwargs(args))


def _build_policy(args: Args) -> Any:
    if args.mode != "remote":
        raise ValueError(f"Unsupported mode: {args.mode}")

    from openpi_client import action_chunk_broker
    from openpi_client import websocket_client_policy as _websocket_client_policy

    base_policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
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


def _build_runtime(args: Args, environment: Any, policy: Any) -> Any:
    from openpi_client.runtime import runtime as _runtime
    from openpi_client.runtime.agents import policy_agent as _policy_agent
    import saver as _saver
    from plot_dynamics import RobotStatePlotter

    broker_for_plot = policy if args.use_async else None
    output_dir = _comparison_out_dir(args)
    return _runtime.Runtime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(policy=policy),
        subscribers=[
            _saver.VideoSaver(output_dir, fps=args.fps),
            RobotStatePlotter(output_dir, broker=broker_for_plot, run_tag=args.run_tag),
        ],
        max_hz=args.fps,
        num_episodes=args.num_episodes,
    )


def _run_metadata_payload(args: Args, out_dir: Path) -> dict[str, Any]:
    control_stack = _resolve_control_stack(args)
    effective_speed_percent = 100 if args.backend == "sdk" or control_stack == "direct-sdk" else 30
    return {
        "selected_stack": control_stack,
        "control_stack": control_stack,
        "run_tag": args.run_tag,
        "prompt": args.prompt,
        "selected_output_dir": str(out_dir),
        "policy": {
            "mode": args.mode,
            "host": args.host,
            "port": args.port,
        },
        "runtime": {
            "backend": args.backend,
            "fps": args.fps,
            "num_steps": args.num_steps,
            "num_episodes": args.num_episodes,
            "action_horizon": args.action_horizon,
            "max_action_horizon": args.max_action_horizon,
            "actions_during_latency": args.actions_during_latency,
            "display": args.display,
            "seed": args.seed,
            "use_async": args.use_async,
            "use_rtc": args.use_rtc,
            "gripper_norm": args.gripper_norm,
            "tele_mode": args.tele_mode,
            "record_mode": args.record_mode,
        },
        "config": {
            "bridge_python": str(args.bridge_python),
            "ros2_config": str(args.ros2_config),
            "dry_run": args.dry_run,
            "publish_actions": args.publish_actions,
            "max_action_delta": args.max_action_delta,
            "left_can_port": args.left_can_port,
            "right_can_port": args.right_can_port,
            "high_camera_id": args.high_camera_id,
            "left_wrist_camera_id": args.left_wrist_camera_id,
            "right_wrist_camera_id": args.right_wrist_camera_id,
        },
        "video_fps": args.fps,
        "effective_speed_percent": effective_speed_percent,
    }


def main(args: Args | argparse.Namespace | None = None) -> int:
    if args is None:
        args = parse_args()
    elif not isinstance(args, Args):
        args = _namespace_to_args(args)

    try:
        _validate_backend_contract(args)
    except ValueError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2

    comparison_out_dir = _comparison_out_dir(args)
    from run_metadata import RunMetadataRecorder

    recorder = RunMetadataRecorder(comparison_out_dir / "run_metadata.json", _run_metadata_payload(args, comparison_out_dir))
    environment: Any | None = None
    runtime: Any | None = None
    exit_code = 0
    exit_reason = "completed"

    def signal_handler(sig: int, frame: Any) -> None:
        del sig, frame
        print("\n🛑 检测到 Ctrl+C，正在安全关闭...")
        if environment is not None:
            try:
                environment.close()
            except Exception as exc:  # noqa: BLE001 - surface cleanup failures verbosely.
                print(f"❌ 关闭环境时出错: {exc}", file=sys.stderr)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    recorder.start()
    try:
        print(contract_summary(args))
        print()

        print("=" * 60)
        print("🤖 双臂 Piper 机器人 - ZR-0/PI05 部署")
        print("=" * 60)

        print("\n🚀 正在初始化双臂 Piper 环境...")
        environment = _build_environment(args)
        print("✅ 双臂 Piper 环境初始化成功")
        print(f"   - backend: {args.backend}")
        print(f"   - selected control stack: {_resolve_control_stack(args)}")
        if args.backend == "ros2":
            print(f"   - bridge python: {args.bridge_python}")
            print(f"   - ros2 config: {args.ros2_config}")
            print(f"   - dry_run: {args.dry_run}")
            print(f"   - publish_actions: {args.publish_actions}")
        print(f"   - 左臂 CAN: {args.left_can_port}")
        print(f"   - 右臂 CAN: {args.right_can_port}")
        print(f"   - 全局相机: {args.high_camera_id}")
        print(f"   - 左腕相机: {args.left_wrist_camera_id}")
        print(f"   - 右腕相机: {args.right_wrist_camera_id}")

        print(f"\n🚀 正在初始化策略 (模式: {args.mode})...")
        policy = _build_policy(args)
        print("✅ 远程策略服务器连接成功")
        print(f"   - 服务器地址: {args.host}:{args.port}")

        print("\n🚀 正在初始化运行时...")
        runtime = _build_runtime(args, environment, policy)
        print("✅ 运行时初始化成功")

        print("\n" + "=" * 60)
        print(f"🎯 任务: {args.prompt}")
        print("=" * 60)

        print("\n🏃 开始运行策略...")
        runtime.run()
        if args.backend == "sdk" and args.record_mode and hasattr(environment, "_save_episode_to_hdf5"):
            print("\n💾 正在保存录制的 episode 数据...")
            environment._save_episode_to_hdf5()
            print("✅ 数据保存完成")
        return 0
    except KeyboardInterrupt:
        exit_code = 130
        exit_reason = "keyboard_interrupt"
        return exit_code
    except Exception as exc:  # noqa: BLE001 - deployment entrypoint should surface the root failure.
        exit_code = 1
        exit_reason = f"{type(exc).__name__}: {exc}"
        print(f"\n❌ 运行时错误: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return exit_code
    finally:
        print("\n🔄 正在清理资源...")
        if runtime is not None and hasattr(runtime, "close"):
            try:
                runtime.close()
            except Exception as exc:  # noqa: BLE001 - cleanup should not mask the primary result.
                print(f"❌ 关闭运行时出错: {exc}", file=sys.stderr)
        if environment is not None:
            try:
                environment.close()
            except Exception as exc:  # noqa: BLE001 - cleanup should not mask the primary result.
                print(f"❌ 关闭环境时出错: {exc}", file=sys.stderr)
        try:
            recorder.finish(exit_code, exit_reason)
        except Exception as exc:  # noqa: BLE001 - metadata cleanup should not mask the primary result.
            print(f"❌ 写入运行 metadata 时出错: {exc}", file=sys.stderr)
        print("✅ 程序已完成")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    raise SystemExit(main())
