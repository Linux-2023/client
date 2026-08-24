#!/usr/bin/env python3
"""Preview-first ROS 2 collector for Piper dual-arm EEF episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

from collect_data_ros2 import DEFAULT_BRIDGE_PYTHON
from collect_data_ros2 import DEFAULT_FRAME_TIMEOUT_SECONDS
from collect_data_ros2 import CollectorController
from collect_data_ros2 import CollectorState
from collect_data_ros2 import PreviewWindow
from collect_data_ros2 import _print_validation_report
from collect_data_ros2 import _render_episode
from collect_data_ros2 import run_collection_loop
from observation_adapter import ObservationAdapter
from ros2_backend import DEFAULT_SYNC_ERROR
from ros2_backend import Ros2BackendClient
from streaming_hdf5 import StreamingEpisodeWriter
from streaming_hdf5 import validate_episode

DEFAULT_EEF_LEFT_TOPIC = "/puppet/end_pose_left"
DEFAULT_EEF_RIGHT_TOPIC = "/puppet/end_pose_right"
EEF_ORDER = ["x", "y", "z", "roll", "pitch", "yaw"]
EEF_UNITS = ["m", "m", "m", "rad", "rad", "rad"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview synchronized Piper dual ROS 2 frames and save joint plus puppet EEF HDF5 episodes."
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for episode_XXXXXX.hdf5 outputs")
    parser.add_argument("--prompt", required=True, help="Task prompt stored in each episode metadata")
    parser.add_argument("--config", required=True, type=Path, help="ROS 2 bridge config YAML/JSON path")
    parser.add_argument(
        "--bridge-python",
        type=Path,
        default=DEFAULT_BRIDGE_PYTHON,
        help="Python interpreter used to launch the ROS 2 bridge (default: /usr/bin/python3)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality metadata recorded by the streaming writer (1-100, default: 90)",
    )
    parser.add_argument(
        "--max-sync-error-ms",
        type=float,
        default=DEFAULT_SYNC_ERROR * 1000.0,
        help="Maximum camera, joint, and EEF timestamp mismatch in milliseconds (default: 30)",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep ROS 2 bridge in no-action-publishing mode (default: true)",
    )
    parser.add_argument(
        "--publish-actions",
        action="store_true",
        default=False,
        help="Enable validated action publishing; unsafe with --dry-run and not used by collection state machine",
    )
    parser.add_argument(
        "--eef-left-topic",
        default=DEFAULT_EEF_LEFT_TOPIC,
        help=f"Puppet-left PoseStamped topic (default: {DEFAULT_EEF_LEFT_TOPIC})",
    )
    parser.add_argument(
        "--eef-right-topic",
        default=DEFAULT_EEF_RIGHT_TOPIC,
        help=f"Puppet-right PoseStamped topic (default: {DEFAULT_EEF_RIGHT_TOPIC})",
    )
    parser.add_argument(
        "--render-after-save",
        action="store_true",
        default=False,
        help="Render a preview MP4 next to each finalized episode",
    )
    return parser


def create_controller(args: argparse.Namespace) -> CollectorController:
    if args.dry_run and args.publish_actions:
        raise ValueError("Unsafe configuration: --dry-run cannot be combined with --publish-actions")
    backend = Ros2BackendClient(
        bridge_python=args.bridge_python,
        config=args.config,
        publish_actions=args.publish_actions,
        dry_run=args.dry_run,
        eef_left_topic=args.eef_left_topic,
        eef_right_topic=args.eef_right_topic,
    )
    try:
        backend.max_sync_error = float(args.max_sync_error_ms) / 1000.0
        backend.synchronizer.max_error = backend.max_sync_error
        backend.start()
        adapter = ObservationAdapter(task=args.prompt)
        preview = PreviewWindow(adapter=adapter)
    except Exception:
        backend.close()
        raise
    return CollectorController(
        backend=backend,
        output_dir=args.output_dir,
        prompt=args.prompt,
        jpeg_quality=args.jpeg_quality,
        preview=preview,
        render_after_save=args.render_after_save,
        collector_name="collect_data_eef_ros2.py",
        metadata_extra={
            "eef": {
                "enabled": True,
                "topics": {
                    "puppet_left": args.eef_left_topic,
                    "puppet_right": args.eef_right_topic,
                },
                "order": list(EEF_ORDER),
                "units": list(EEF_UNITS),
                "representation": "xyz_rpy",
            }
        },
        writer_options={"include_eef": True},
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    controller = create_controller(args)
    print("Previewing synchronized ROS 2 frames with puppet EEF. Press s to record; while recording, s or e finalize; q to quit.")
    try:
        run_collection_loop(controller, frame_timeout=DEFAULT_FRAME_TIMEOUT_SECONDS)
    except KeyboardInterrupt:
        controller.close()
        print("Interrupted; closed EEF collector cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
