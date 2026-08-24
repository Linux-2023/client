"""State-machine tests for the preview-first ROS 2 collector."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import collect_data_ros2
from collect_data_ros2 import CAMERA_ORDER
from collect_data_ros2 import CollectorController
from collect_data_ros2 import CollectorState
from collect_data_ros2 import PreviewWindow
from collect_data_ros2 import build_arg_parser
from collect_data_ros2 import create_controller
from collect_data_ros2 import run_collection_loop
from frame_synchronizer import REQUIRED_SENSORS
from frame_synchronizer import SynchronizedFrame
from streaming_hdf5 import StreamingEpisodeWriter
from streaming_hdf5 import validate_episode


class FakeBackend:
    def __init__(self, frames=(), failure: Exception | None = None, events: list[str] | None = None) -> None:
        self.frames = deque(frames)
        self.failure = failure
        self.events = events if events is not None else []
        self.clear_calls = 0
        self.close_calls = 0
        self.publish_action_calls = []
        self.dry_run = True
        self.publish_actions_enabled = False

    def next_frame(self, timeout: float) -> SynchronizedFrame:
        if self.frames:
            return self.frames.popleft()
        if self.failure is not None:
            raise self.failure
        raise TimeoutError("no frame")

    def clear_buffers(self) -> None:
        self.clear_calls += 1
        self.events.append("clear_buffers")

    def close(self) -> None:
        self.close_calls += 1
        self.events.append("close_backend")

    def publish_action(self, action) -> None:
        self.publish_action_calls.append(action)
        self.events.append("publish_action")


class FakeWriter:
    def __init__(self, final_path: Path, metadata: dict, jpeg_quality: int, events: list[str]) -> None:
        self.final_path = Path(final_path)
        self.partial_path = self.final_path.with_suffix(f"{self.final_path.suffix}.partial")
        self.metadata = metadata
        self.jpeg_quality = jpeg_quality
        self.events = events
        self.appended: list[SynchronizedFrame] = []
        self.finalized = False
        self.aborted = False

    def append(self, frame: SynchronizedFrame) -> None:
        self.appended.append(frame)
        self.events.append(f"append:{frame.timestamp}")

    def finalize(self) -> Path:
        self.finalized = True
        self.events.append(f"finalize:{self.final_path.name}")
        return self.final_path

    def abort(self, remove_partial: bool = True) -> None:
        self.aborted = True
        self.events.append(f"abort:{self.partial_path.name}:{remove_partial}")


class RecordingWriterFactory:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events if events is not None else []
        self.writers: list[FakeWriter] = []

    def __call__(self, path: Path, *, metadata: dict, jpeg_quality: int, **kwargs) -> FakeWriter:
        writer = FakeWriter(Path(path), metadata, jpeg_quality, self.events)
        self.writers.append(writer)
        self.events.append(f"open:{writer.partial_path.name}")
        return writer


class FakePreview:
    def __init__(self, keys=()) -> None:
        self.keys = deque(keys)
        self.rendered = []
        self.closed = False

    def render(self, frame: SynchronizedFrame, state: CollectorState, prompt: str) -> str | None:
        self.rendered.append((frame, state, prompt))
        if self.keys:
            return self.keys.popleft()
        return None

    def close(self) -> None:
        self.closed = True


def make_frame(index: int) -> SynchronizedFrame:
    timestamp = float(index)
    return SynchronizedFrame(
        timestamp=timestamp,
        images={camera: f"{camera}-{index}".encode("ascii") for camera in CAMERA_ORDER},
        state=np.full((14,), index, dtype=np.float32),
        action=np.full((14,), index + 0.5, dtype=np.float32),
        sensor_timestamps={sensor: timestamp for sensor in REQUIRED_SENSORS},
        sync_error={sensor: 0.0 for sensor in REQUIRED_SENSORS},
    )


def make_controller(
    tmp_path: Path,
    *,
    backend: FakeBackend | None = None,
    writer_factory: RecordingWriterFactory | None = None,
    preview: FakePreview | None = None,
    renderer=None,
    reports: list[dict] | None = None,
    validator=None,
    render_after_save: bool = False,
) -> CollectorController:
    writer_factory = writer_factory or RecordingWriterFactory()
    reports = reports if reports is not None else []
    return CollectorController(
        backend=backend or FakeBackend(),
        output_dir=tmp_path,
        prompt="Fold the towel",
        jpeg_quality=77,
        writer_factory=writer_factory,
        validator=validator or (lambda path: {"frame_count": 0, "errors": [], "path": str(path)}),
        report_sink=reports.append,
        preview=preview or FakePreview(),
        render_after_save=render_after_save or renderer is not None,
        renderer=renderer,
    )


def valid_jpeg(seed: int) -> bytes:
    image = np.zeros((16, 18, 3), dtype=np.uint8)
    image[:] = [seed % 255, (seed + 40) % 255, (seed + 80) % 255]
    ok, encoded = collect_data_ros2.cv2.imencode(".jpg", image)
    assert ok
    return bytes(encoded)


def render_frame(index: int) -> SynchronizedFrame:
    timestamp = float(index)
    return SynchronizedFrame(
        timestamp=timestamp,
        images={camera: valid_jpeg(index * 10 + offset) for offset, camera in enumerate(CAMERA_ORDER)},
        state=np.full((14,), index, dtype=np.float32),
        action=np.full((14,), index + 0.5, dtype=np.float32),
        sensor_timestamps={sensor: timestamp for sensor in REQUIRED_SENSORS},
        sync_error={sensor: 0.0 for sensor in REQUIRED_SENSORS},
    )


def test_preview_startup_e_is_noop_and_frames_do_not_open_or_append_before_s(tmp_path: Path) -> None:
    writer_factory = RecordingWriterFactory()
    backend = FakeBackend()
    preview = FakePreview()
    controller = make_controller(tmp_path, backend=backend, writer_factory=writer_factory, preview=preview)

    assert controller.state is CollectorState.PREVIEW
    assert controller.handle_key("e") is CollectorState.PREVIEW

    assert controller.process_frame(make_frame(0)) is None

    assert writer_factory.writers == []
    assert preview.rendered[0][1] is CollectorState.PREVIEW
    assert backend.clear_calls == 0
    assert backend.publish_action_calls == []


def test_s_clears_buffers_before_opening_partial_and_appends_only_while_recording(tmp_path: Path) -> None:
    events: list[str] = []
    backend = FakeBackend(events=events)
    writer_factory = RecordingWriterFactory(events)
    reports: list[dict] = []
    controller = make_controller(tmp_path, backend=backend, writer_factory=writer_factory, reports=reports)

    controller.process_frame(make_frame(0))
    assert writer_factory.writers == []

    assert controller.handle_key("s") is CollectorState.RECORDING
    assert events[:2] == ["clear_buffers", "open:episode_000000.hdf5.partial"]
    writer = writer_factory.writers[0]
    assert writer.partial_path == tmp_path / "episode_000000.hdf5.partial"
    assert writer.metadata["prompt"] == "Fold the towel"
    assert writer.jpeg_quality == 77

    controller.process_frame(make_frame(1))
    controller.process_frame(make_frame(2))
    assert [frame.timestamp for frame in writer.appended] == [1.0, 2.0]

    assert controller.handle_key("s") is CollectorState.PREVIEW
    assert writer.finalized is True
    assert reports == [{"frame_count": 0, "errors": [], "path": str(writer.final_path)}]

    controller.process_frame(make_frame(3))
    assert [frame.timestamp for frame in writer.appended] == [1.0, 2.0]
    assert backend.publish_action_calls == []


def test_s_toggles_recording_and_finalizes_episode(tmp_path: Path) -> None:
    writer_factory = RecordingWriterFactory()
    reports: list[dict] = []
    controller = make_controller(tmp_path, writer_factory=writer_factory, reports=reports)

    assert controller.handle_key("s") is CollectorState.RECORDING
    controller.process_frame(make_frame(1))
    assert controller.handle_key("s") is CollectorState.PREVIEW

    writer = writer_factory.writers[0]
    assert writer.finalized is True
    assert reports == [{"frame_count": 0, "errors": [], "path": str(writer.final_path)}]

def test_e_finalizes_recording_episode_with_real_writer_and_preserves_preview_noop(tmp_path: Path, capsys) -> None:
    backend = FakeBackend()
    reports: list[dict] = []
    controller = make_controller(
        tmp_path,
        backend=backend,
        writer_factory=StreamingEpisodeWriter.open,
        reports=reports,
        validator=validate_episode,
    )

    assert controller.handle_key("e") is CollectorState.PREVIEW
    assert controller.writer_open is False

    assert controller.handle_key("s") is CollectorState.RECORDING
    controller.process_frame(render_frame(21))
    assert controller.handle_key("e") is CollectorState.PREVIEW

    final_path = tmp_path / "episode_000000.hdf5"
    assert final_path.exists()
    assert validate_episode(final_path)["errors"] == []
    assert controller.state is CollectorState.PREVIEW
    assert controller.writer_open is False
    assert reports[-1]["frame_count"] == 1
    assert backend.close_calls == 0
    output = capsys.readouterr().out
    assert "Finalized" in output


def test_repeated_episodes_skip_existing_outputs_and_use_distinct_files(tmp_path: Path) -> None:
    (tmp_path / "episode_000000.hdf5").write_text("existing", encoding="utf-8")
    (tmp_path / "episode_000001.hdf5.partial").write_text("existing partial", encoding="utf-8")
    writer_factory = RecordingWriterFactory()
    controller = make_controller(tmp_path, writer_factory=writer_factory)

    assert controller.handle_key("s") is CollectorState.RECORDING
    controller.process_frame(make_frame(4))
    assert controller.handle_key("s") is CollectorState.PREVIEW

    assert controller.handle_key("s") is CollectorState.RECORDING
    controller.process_frame(make_frame(5))
    assert controller.handle_key("s") is CollectorState.PREVIEW

    assert [writer.final_path.name for writer in writer_factory.writers] == [
        "episode_000002.hdf5",
        "episode_000003.hdf5",
    ]
    assert len({writer.partial_path for writer in writer_factory.writers}) == 2



def test_empty_real_writer_episode_restores_preview_keeps_backend_and_next_valid_episode_succeeds(
    tmp_path: Path, capsys
) -> None:
    backend = FakeBackend()
    reports: list[dict] = []
    controller = make_controller(
        tmp_path,
        backend=backend,
        writer_factory=StreamingEpisodeWriter.open,
        reports=reports,
        validator=validate_episode,
    )

    assert controller.handle_key("s") is CollectorState.RECORDING
    assert controller.handle_key("s") is CollectorState.PREVIEW

    final_path = tmp_path / "episode_000000.hdf5"
    partial_path = tmp_path / "episode_000000.hdf5.partial"
    assert controller.state is CollectorState.PREVIEW
    assert controller.writer_open is False
    assert backend.close_calls == 0
    assert not final_path.exists()
    assert not partial_path.exists()
    assert reports == []
    output = capsys.readouterr().out
    assert "Failed to finalize" in output
    assert "zero frames" in output

    assert controller.handle_key("s") is CollectorState.RECORDING
    controller.process_frame(render_frame(10))
    assert controller.handle_key("s") is CollectorState.PREVIEW

    next_final_path = tmp_path / "episode_000001.hdf5"
    assert next_final_path.exists()
    assert validate_episode(next_final_path)["errors"] == []
    assert reports[-1]["frame_count"] == 1
    assert backend.close_calls == 0


def test_real_writer_validation_failure_restores_preview_removes_partial_and_allows_next_valid_episode(
    tmp_path: Path, capsys
) -> None:
    backend = FakeBackend()
    reports: list[dict] = []
    controller = make_controller(
        tmp_path,
        backend=backend,
        writer_factory=StreamingEpisodeWriter.open,
        reports=reports,
        validator=validate_episode,
    )

    assert controller.handle_key("s") is CollectorState.RECORDING
    controller.process_frame(make_frame(11))
    assert controller.handle_key("s") is CollectorState.PREVIEW

    failed_final_path = tmp_path / "episode_000000.hdf5"
    failed_partial_path = tmp_path / "episode_000000.hdf5.partial"
    assert controller.state is CollectorState.PREVIEW
    assert controller.writer_open is False
    assert backend.close_calls == 0
    assert not failed_final_path.exists()
    assert not failed_partial_path.exists()
    assert reports == []
    output = capsys.readouterr().out
    assert "Failed to finalize" in output
    assert "not decodable JPEG bytes" in output

    assert controller.handle_key("s") is CollectorState.RECORDING
    controller.process_frame(render_frame(12))
    assert controller.handle_key("s") is CollectorState.PREVIEW

    next_final_path = tmp_path / "episode_000001.hdf5"
    assert next_final_path.exists()
    assert validate_episode(next_final_path)["errors"] == []
    assert reports[-1]["frame_count"] == 1
    assert backend.close_calls == 0

def test_q_aborts_open_partial_and_closes_backend_without_publishing_actions(tmp_path: Path) -> None:
    backend = FakeBackend()
    writer_factory = RecordingWriterFactory()
    preview = FakePreview()
    controller = make_controller(tmp_path, backend=backend, writer_factory=writer_factory, preview=preview)

    assert controller.handle_key("s") is CollectorState.RECORDING
    controller.process_frame(make_frame(6))
    assert controller.handle_key("q") is CollectorState.EXIT

    writer = writer_factory.writers[0]
    assert writer.aborted is True
    assert writer.finalized is False
    assert backend.close_calls == 1
    assert preview.closed is True
    assert backend.publish_action_calls == []


def test_run_loop_aborts_writer_and_closes_backend_when_backend_raises(tmp_path: Path) -> None:
    backend = FakeBackend(frames=[make_frame(7)], failure=RuntimeError("bridge boom"))
    writer_factory = RecordingWriterFactory()
    preview = FakePreview(keys=["s"])
    controller = make_controller(tmp_path, backend=backend, writer_factory=writer_factory, preview=preview)

    with pytest.raises(RuntimeError, match="bridge boom"):
        run_collection_loop(controller, frame_timeout=0.001)

    writer = writer_factory.writers[0]
    assert writer.aborted is True
    assert writer.finalized is False
    assert backend.close_calls == 1
    assert preview.closed is True
    assert backend.publish_action_calls == []


class FakeAdapter:
    def __init__(self) -> None:
        self.decoded: list[bytes] = []

    def decode_image(self, jpeg: bytes) -> np.ndarray:
        self.decoded.append(jpeg)
        value = len(self.decoded) * 40
        return np.full((2, 3, 3), value, dtype=np.uint8)


class FakeCv2:
    COLOR_RGB2BGR = 1
    FONT_HERSHEY_SIMPLEX = 0

    def __init__(self) -> None:
        self.shown = []
        self.destroyed = []

    def cvtColor(self, image: np.ndarray, code: int) -> np.ndarray:
        assert code == self.COLOR_RGB2BGR
        return image[..., ::-1].copy()

    def putText(self, image, text, org, font, font_scale, color, thickness):
        return image

    def imshow(self, window_name: str, image: np.ndarray) -> None:
        self.shown.append((window_name, image.copy()))

    def waitKey(self, delay: int) -> int:
        return ord("s")

    def destroyWindow(self, window_name: str) -> None:
        self.destroyed.append(window_name)


def test_preview_window_decodes_and_displays_three_cameras_without_gui_hardware() -> None:
    adapter = FakeAdapter()
    cv2_module = FakeCv2()
    preview = PreviewWindow(adapter=adapter, cv2_module=cv2_module, window_name="test-preview")

    key = preview.render(make_frame(8), CollectorState.RECORDING, "Fold the towel")

    assert key == "s"
    assert adapter.decoded == [
        b"cam_high-8",
        b"cam_left_wrist-8",
        b"cam_right_wrist-8",
    ]
    assert len(cv2_module.shown) == 1
    window_name, image = cv2_module.shown[0]
    assert window_name == "test-preview"
    assert image.shape[1] == 9
    assert image.shape[0] > 2

    preview.close()
    assert cv2_module.destroyed == ["test-preview"]

def test_arg_parser_exposes_required_flags_safe_defaults_and_control_stack_choices(tmp_path: Path) -> None:
    parser = build_arg_parser()
    help_text = parser.format_help()

    for flag in (
        "--output-dir",
        "--prompt",
        "--config",
        "--bridge-python",
        "--jpeg-quality",
        "--max-sync-error-ms",
        "--dry-run",
        "--publish-actions",
        "--render-after-save",
        "--control-stack",
    ):
        assert flag in help_text

    args = parser.parse_args([
        "--output-dir",
        str(tmp_path),
        "--prompt",
        "Fold the towel",
        "--config",
        "examples/piper_dual/ros2_piper_dual.yaml",
    ])

    assert args.bridge_python == Path("/usr/bin/python3")
    assert args.dry_run is True
    assert args.publish_actions is False
    assert args.control_stack == "official-ros"

    for control_stack in ("local-ros", "official-ros", "direct-sdk"):
        parsed = parser.parse_args([
            "--output-dir",
            str(tmp_path),
            "--prompt",
            "Fold the towel",
            "--config",
            "examples/piper_dual/ros2_piper_dual.yaml",
            "--control-stack",
            control_stack,
        ])
        assert parsed.control_stack == control_stack


def test_create_controller_forwards_selected_control_stack_to_backend_without_enabling_publish(tmp_path: Path, monkeypatch) -> None:
    created = []

    class FakeSynchronizer:
        def __init__(self) -> None:
            self.max_error = None

    class FakeRos2Backend(FakeBackend):
        def __init__(
            self,
            *,
            bridge_python: Path,
            config: Path,
            publish_actions: bool,
            dry_run: bool,
            control_stack: str,
        ) -> None:
            super().__init__()
            self.bridge_python = bridge_python
            self.config = config
            self.publish_actions_enabled = publish_actions
            self.dry_run = dry_run
            self.control_stack = control_stack
            self.max_sync_error = None
            self.synchronizer = FakeSynchronizer()
            self.start_calls = 0
            created.append(self)

        def start(self) -> None:
            self.start_calls += 1

    monkeypatch.setattr(collect_data_ros2, "Ros2BackendClient", FakeRos2Backend)
    monkeypatch.setattr(collect_data_ros2, "ObservationAdapter", lambda task: object())
    monkeypatch.setattr(collect_data_ros2, "PreviewWindow", lambda adapter: FakePreview())

    args = build_arg_parser().parse_args([
        "--output-dir",
        str(tmp_path),
        "--prompt",
        "Fold the towel",
        "--config",
        "examples/piper_dual/ros2_piper_dual.yaml",
        "--control-stack",
        "local-ros",
    ])

    controller = create_controller(args)

    assert controller.backend is created[0]
    assert created[0].start_calls == 1
    assert created[0].control_stack == "local-ros"
    assert created[0].dry_run is True
    assert created[0].publish_actions_enabled is False
    assert created[0].max_sync_error == pytest.approx(0.03)
    assert created[0].synchronizer.max_error == pytest.approx(0.03)


def test_create_controller_applies_max_sync_error_to_backend_synchronizer(tmp_path: Path, monkeypatch) -> None:
    created = []

    class FakeSynchronizer:
        def __init__(self) -> None:
            self.max_error = None

    class FakeRos2Backend(FakeBackend):
        def __init__(
            self,
            *,
            bridge_python: Path,
            config: Path,
            publish_actions: bool,
            dry_run: bool,
            control_stack: str,
        ) -> None:
            super().__init__()
            self.bridge_python = bridge_python
            self.config = config
            self.publish_actions_enabled = publish_actions
            self.dry_run = dry_run
            self.control_stack = control_stack
            self.max_sync_error = None
            self.synchronizer = FakeSynchronizer()
            self.start_calls = 0
            created.append(self)

        def start(self) -> None:
            self.start_calls += 1

    monkeypatch.setattr(collect_data_ros2, "Ros2BackendClient", FakeRos2Backend)
    monkeypatch.setattr(collect_data_ros2, "ObservationAdapter", lambda task: object())
    monkeypatch.setattr(collect_data_ros2, "PreviewWindow", lambda adapter: FakePreview())

    args = build_arg_parser().parse_args([
        "--output-dir",
        str(tmp_path),
        "--prompt",
        "Fold the towel",
        "--config",
        "examples/piper_dual/ros2_piper_dual.yaml",
        "--max-sync-error-ms",
        "12.5",
    ])

    controller = create_controller(args)

    assert controller.backend is created[0]
    assert created[0].start_calls == 1
    assert created[0].max_sync_error == pytest.approx(0.0125)
    assert created[0].synchronizer.max_error == pytest.approx(0.0125)


def test_render_after_save_uses_self_contained_hdf5_schema_and_returns_to_preview(tmp_path: Path, monkeypatch) -> None:
    opened: list[object] = []

    class FakeRenderBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.action_adapter = object()
            self.synchronizer = type("Sync", (), {"include_eef": False})()

    monkeypatch.setattr(collect_data_ros2, "render_episode", lambda path, output_dir: opened.append((path, output_dir)))
    monkeypatch.setattr(collect_data_ros2, "validate_episode", lambda path: {"frame_count": 1, "errors": [], "path": str(path)})

    backend = FakeRenderBackend()
    preview = FakePreview()
    controller = CollectorController(
        backend=backend,
        output_dir=tmp_path,
        prompt="Fold the towel",
        jpeg_quality=90,
        preview=preview,
        render_after_save=True,
    )
    controller._writer_factory = RecordingWriterFactory()

    controller.start_episode()
    controller.process_frame(make_frame(1))
    controller.stop_episode()

    assert opened
    assert controller.state is collect_data_ros2.CollectorState.PREVIEW
    assert backend.close_calls == 0


def test_render_after_save_uses_self_contained_hdf5_schema_and_returns_to_preview(tmp_path: Path, monkeypatch) -> None:
    opened: list[object] = []

    class FakeVideoWriter:
        def __init__(self, path, fourcc, fps, size) -> None:
            self.path = Path(path)
            self.fourcc = fourcc
            self.fps = fps
            self.size = size
            self.frames = []
            self.closed = False
            opened.append(self)

        def isOpened(self) -> bool:
            return True

        def write(self, frame: np.ndarray) -> None:
            self.frames.append(frame.copy())

        def release(self) -> None:
            self.closed = True

    monkeypatch.setattr(collect_data_ros2.cv2, "VideoWriter", FakeVideoWriter)

    backend = FakeBackend()
    controller = make_controller(
        tmp_path,
        backend=backend,
        writer_factory=StreamingEpisodeWriter.open,
        preview=FakePreview(),
        validator=validate_episode,
        render_after_save=True,
    )

    assert controller.handle_key("s") is CollectorState.RECORDING
    controller.process_frame(render_frame(9))

    assert controller.handle_key("s") is CollectorState.PREVIEW
    assert controller.state is CollectorState.PREVIEW
    assert len(opened) == 1
    assert opened[0].path == tmp_path / "episode_000000.preview.mp4"
    assert opened[0].closed is True
    assert len(opened[0].frames) == 1
    assert opened[0].size[0] > 0 and opened[0].size[1] > 0
    assert backend.close_calls == 0
