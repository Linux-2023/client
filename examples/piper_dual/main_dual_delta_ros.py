#!/usr/bin/env python3
"""Delta-joint variant of ``main_dual_ros.py``.

``main_dual_ros.py`` treats every server response as **absolute** joint targets. The
``delta_joint_current_v1`` checkpoints are trained on deltas, and a server may hand them back
either way:

* ``server_output == "absolute_joint"`` (what ``deploy_49999_joint/serve_delta_joint.py``
  emits, anchor ``request_state``): the server already decoded the deltas -- the targets pass
  through unchanged. Adding the state again would double-count the motion.
* ``server_output == "delta_joint"``: the targets are deltas anchored at the measured state of
  the same request -- adding that state yields absolute targets.

Everything else (arguments, ROS 2 environment, runtime, safety guards, recording) is inherited
verbatim from ``main_dual_ros.py``; this module only swaps in the policy wrapper, so the two
entrypoints stay in lockstep.

    python examples/piper_dual/main_dual_delta_ros.py --host 127.0.0.1 --port 8000 \
        --no-dry-run --publish-actions            # defaults to --control-stack local-ros

Use ``--server-output absolute|delta`` to override the metadata autodetection.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np

import main_dual_ros as original
from main_dual_ros import Args as _BaseArgs

JOINT_DIM = 14


@dataclass
class Args(_BaseArgs):
    """``main_dual_ros.Args`` plus this cell's defaults and the delta-server switch.

    ``action_horizon``/``fps`` are pinned to the values the delta-joint deployment was verified
    with, so edits to ``main_dual_ros.Args`` defaults cannot silently change the playback rate.
    """

    action_horizon: int = 22
    fps: int = 15
    control_stack: Literal["local-ros", "official-ros", "direct-sdk"] = "local-ros"
    ros2_config: Path = Path(__file__).with_name("ros2_piper_dual_local.yaml")
    server_output: Literal["auto", "absolute", "delta"] = "auto"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = original.build_arg_parser()
    parser.description = (
        "Deploy a dual-arm Piper PI05 delta-joint policy through the ROS 2 backend "
        "(delta-aware wrapper around main_dual_ros.py)."
    )
    parser.set_defaults(control_stack=Args.control_stack, ros2_config=Args.ros2_config,
                        action_horizon=Args.action_horizon, fps=Args.fps)
    parser.add_argument(
        "--server-output",
        dest="server_output",
        choices=("auto", "absolute", "delta"),
        default=Args.server_output,
        help="Interpret the server response as absolute targets or deltas (auto = use metadata)",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> Args:
    return Args(**vars(build_arg_parser().parse_args(argv)))


class DeltaJointPolicy:
    """Validate the joint contract and decode deltas when the server returns them."""

    def __init__(self, policy: Any, mode: str = "auto") -> None:
        self.policy = policy
        metadata: dict[str, Any] = {}
        if hasattr(policy, "get_server_metadata"):
            metadata = policy.get_server_metadata() or {}
        self.metadata = metadata
        declared = str(metadata.get("server_output", "")).lower()
        if mode == "auto":
            self.decode_delta = declared.startswith("delta")
        else:
            self.decode_delta = mode == "delta"
        anchor = metadata.get("anchor")
        print(
            f"Delta-joint policy: server_output={declared or 'unknown'}"
            + (f" anchor={anchor}" if anchor else "")
            + " -> "
            + ("adding the measured state to delta targets" if self.decode_delta
               else "passing targets through as absolute")
        )

    def get_server_metadata(self) -> dict[str, Any]:
        return self.metadata

    def reset(self) -> Any:
        return self.policy.reset()

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        result = self.policy.infer(obs)
        actions = np.asarray(result.get("actions"), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != JOINT_DIM or not np.isfinite(actions).all():
            raise ValueError(
                f"joint server must return (N, {JOINT_DIM}) finite targets, got {actions.shape}"
            )
        if not self.decode_delta:
            return result
        state = np.asarray(obs.get("state"), dtype=np.float32).reshape(-1)
        if state.shape != (JOINT_DIM,) or not np.isfinite(state).all():
            raise ValueError(f"delta decoding needs the measured {JOINT_DIM}-dim joint state")
        decoded = dict(result)
        decoded["actions"] = (state[None, :] + actions).astype(np.float32)
        return decoded


def _build_policy(args: Args) -> Any:
    from openpi_client import action_chunk_broker
    from openpi_client import websocket_client_policy
    original._validate_ros2_contract(args)

    base_policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    delta_policy = DeltaJointPolicy(base_policy, mode=args.server_output)
    if args.training_rtc and delta_policy.decode_delta:
        raise ValueError('Training-RTC requires absolute server targets, not client-side delta decoding')
    if args.use_async:
        return action_chunk_broker.ActionChunkBroker_RTC(
            policy=delta_policy,
            action_horizon=args.action_horizon,
            fps=args.fps,
            actions_during_latency=args.actions_during_latency,
            use_rtc=args.use_rtc,
            training_rtc=args.training_rtc,
            handoff_blend_steps=args.handoff_blend_steps,
            handoff_angle_indices=(),
        )
    return action_chunk_broker.ActionChunkBroker(
        policy=delta_policy,
        action_horizon=args.action_horizon,
        fps=args.fps,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    # main_dual_ros.main() builds its policy through this module-level hook.
    original._build_policy = _build_policy  # noqa: SLF001 - intentional minimal override
    return original.main(args)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, force=True)
    raise SystemExit(main())
