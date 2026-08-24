"""Tests for the opt-in Piper dual ROS 2 EEF collector."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import collect_data_eef_ros2
from collect_data_eef_ros2 import CollectorController
from collect_data_eef_ros2 import build_arg_parser
from collect_data_eef_ros2 import create_controller
from frame_synchronizer import IMAGE_SENSORS
from frame_synchronizer import REQUIRED_SENSORS
from frame_synchronizer import SynchronizedFrame


class FakeBackend:
    def __init__(self, *, eef_left_topic, eef_right_topic, **kwargs):
        self.eef_left_topic = eef_left_topic
        self.eef_right_topic = eef_right_topic
        self.kwargs = kwargs
        self.max_sync_error = 0.03
        self.synchronizer = type("Sync", (), {"max_error": 0.03})()
        self.closed = False
        self.started = False

    def start(self):
        self.started = True

    def clear_buffers(self):
        pass

    def close(self):
        self.closed = True


class FakePreview:
    def close(self):
        pass


class FakeWriter:
    def __init__(self, final_path, metadata, jpeg_quality, include_eef=False, **kwargs):
        self.final_path = Path(final_path)
        self.partial_path = self.final_path.with_suffix(f"{self.final_path.suffix}.partial")
        self.metadata = metadata
        self.jpeg_quality = jpeg_quality
        self.include_eef = include_eef
        self.kwargs = kwargs
        self.frames = []
        self.finalized = False
        self.aborted = False

    def append(self, frame):
        self.frames.append(frame)

    def finalize(self):
        self.finalized = True
        return self.final_path

    def abort(self, remove_partial=True):
        self.aborted = True


class FakeWriterFactory:
    def __init__(self):
        self.writers = []

    def __call__(self, *args, **kwargs):
        writer = FakeWriter(*args, **kwargs)
        self.writers.append(writer)
        return writer


@pytest.mark.parametrize("control_stack", ["local-ros", "official-ros", "direct-sdk"])
def test_eef_cli_uses_safe_defaults_and_accepts_control_stack(tmp_path: Path, control_stack: str):
    parser = build_arg_parser()
    args = parser.parse_args([
        "--output-dir", str(tmp_path),
        "--prompt", "Fold the towel",
        "--config", "examples/piper_dual/ros2_piper_dual.yaml",
        "--control-stack", control_stack,
    ])

    assert args.control_stack == control_stack
    assert args.dry_run is True
    assert args.publish_actions is False
    assert args.eef_left_topic == "/puppet/end_pose_left"
    assert args.eef_right_topic == "/puppet/end_pose_right"


def test_eef_cli_defaults_to_official_stack(tmp_path: Path):
    parser = build_arg_parser()
    args = parser.parse_args([
        "--output-dir", str(tmp_path),
        "--prompt", "Fold the towel",
        "--config", "examples/piper_dual/ros2_piper_dual.yaml",
    ])

    assert args.control_stack == "official-ros"
    assert args.dry_run is True
    assert args.publish_actions is False
    assert args.eef_left_topic == "/puppet/end_pose_left"
    assert args.eef_right_topic == "/puppet/end_pose_right"


def test_eef_create_controller_passes_topics_and_sync_limit(tmp_path: Path, monkeypatch):
    created = []

    class Backend(FakeBackend):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            created.append(self)

    monkeypatch.setattr(collect_data_eef_ros2, "Ros2BackendClient", Backend)
    monkeypatch.setattr(collect_data_eef_ros2, "ObservationAdapter", lambda task: object())
    monkeypatch.setattr(collect_data_eef_ros2, "PreviewWindow", lambda adapter: FakePreview())
    args = build_arg_parser().parse_args([
        "--output-dir", str(tmp_path),
        "--prompt", "Fold the towel",
        "--config", "config.yaml",
        "--control-stack", "local-ros",
        "--eef-left-topic", "/custom/eef_left",
        "--eef-right-topic", "/custom/eef_right",
        "--max-sync-error-ms", "12.5",
    ])

    controller = create_controller(args)

    assert controller.backend is created[0]
    assert created[0].kwargs["control_stack"] == "local-ros"
    assert created[0].eef_left_topic == "/custom/eef_left"
    assert created[0].eef_right_topic == "/custom/eef_right"
    assert created[0].kwargs["publish_actions"] is False
    assert created[0].kwargs["dry_run"] is True
    assert created[0].max_sync_error == pytest.approx(0.0125)
    assert created[0].synchronizer.max_error == pytest.approx(0.0125)


def test_eef_controller_writes_eef_metadata_and_enables_writer(tmp_path: Path):
    backend = FakeBackend(eef_left_topic="/left", eef_right_topic="/right")
    factory = FakeWriterFactory()
    controller = CollectorController(
        backend=backend,
        output_dir=tmp_path,
        prompt="Fold the towel",
        writer_factory=factory,
        preview=None,
        collector_name="collect_data_eef_ros2.py",
        metadata_extra={
            "eef": {
                "enabled": True,
                "topics": {"puppet_left": "/left", "puppet_right": "/right"},
                "order": ["x", "y", "z", "roll", "pitch", "yaw"],
                "units": ["m", "m", "m", "rad", "rad", "rad"],
            }
        },
        writer_options={"include_eef": True},
    )

    controller.start_episode()
    writer = factory.writers[0]

    assert writer.include_eef is True
    assert writer.metadata["collector"] == "collect_data_eef_ros2.py"
    assert writer.metadata["eef"]["enabled"] is True
    assert writer.metadata["eef"]["topics"]["puppet_left"] == "/left"

    controller.close()


def test_eef_cli_rejects_dry_run_action_publishing(tmp_path: Path):
    args = build_arg_parser().parse_args([
        "--output-dir", str(tmp_path),
        "--prompt", "Fold the towel",
        "--config", "config.yaml",
        "--no-dry-run",
        "--publish-actions",
    ])
    assert args.dry_run is False
    assert args.publish_actions is True
