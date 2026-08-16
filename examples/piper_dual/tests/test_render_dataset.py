"""Tests for self-contained Piper dual HDF5 rendering."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frame_synchronizer import FRAME_DIMENSION
from frame_synchronizer import IMAGE_SENSORS
from frame_synchronizer import REQUIRED_SENSORS
from frame_synchronizer import SynchronizedFrame
from streaming_hdf5 import SCHEMA_VERSION
from streaming_hdf5 import StreamingEpisodeWriter

from render_dataset import render_episode
import visualize_hdf5

CAMERA_COLORS: dict[str, tuple[int, int, int]] = {
    "cam_high": (255, 0, 0),
    "cam_left_wrist": (0, 255, 0),
    "cam_right_wrist": (0, 0, 255),
}


def _jpeg_bytes(color: tuple[int, int, int], size: tuple[int, int] = (48, 32)) -> bytes:
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    image[:] = np.asarray(color, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    return bytes(encoded)


def _frame(index: int) -> SynchronizedFrame:
    timestamp = 100.0 + index * 0.25
    state = (np.arange(FRAME_DIMENSION, dtype=np.float32) + index).astype(np.float32)
    action = (np.arange(FRAME_DIMENSION, dtype=np.float32) + 100.0 + index).astype(np.float32)
    sensor_timestamps = {sensor: timestamp + sensor_index * 0.01 for sensor_index, sensor in enumerate(REQUIRED_SENSORS)}
    sync_error = {sensor: sensor_index * 0.001 for sensor_index, sensor in enumerate(REQUIRED_SENSORS)}
    images = {camera: _jpeg_bytes(color) for camera, color in CAMERA_COLORS.items()}
    return SynchronizedFrame(
        timestamp=timestamp,
        images=images,
        state=state,
        action=action,
        sensor_timestamps=sensor_timestamps,
        sync_error=sync_error,
    )


def _write_new_schema_episode(path: Path, *, frame_count: int = 5, prompt: str = "Fold the towel") -> Path:
    writer = StreamingEpisodeWriter.open(
        path,
        metadata={"prompt": prompt, "episode_id": "episode-000000", "operator": "tester"},
        jpeg_quality=92,
        flush_every=2,
    )
    for index in range(frame_count):
        writer.append(_frame(index))
    return writer.finalize()


def _write_legacy_episode(path: Path, *, frame_count: int = 5) -> Path:
    with h5py.File(path, "w") as episode:
        observations = episode.create_group("observations")
        observations.create_dataset(
            "qpos",
            data=np.stack([np.arange(FRAME_DIMENSION, dtype=np.float32) + index for index in range(frame_count)]),
            maxshape=(None, FRAME_DIMENSION),
            chunks=(1, FRAME_DIMENSION),
        )
        images = observations.create_group("images")
        for camera, color in CAMERA_COLORS.items():
            image_stack = np.stack([np.full((32, 48, 3), color, dtype=np.uint8) for _ in range(frame_count)])
            images.create_dataset(camera, data=image_stack, maxshape=(None, 32, 48, 3), chunks=(1, 32, 48, 3))
        episode.create_dataset("task", data=np.asarray([b"legacy schema"] * frame_count, dtype="S13"))
    return path


def _write_official_legacy_episode(path: Path, *, frame_count: int = 3) -> Path:
    with h5py.File(path, "w") as episode:
        camera = episode.create_group("camera")
        color = camera.create_group("color")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        for camera_name in ("left", "front_l", "right"):
            values = np.asarray(
                [f"camera/color/{camera_name}/{1000.0 + index:.6f}.jpg" for index in range(frame_count)],
                dtype=object,
            )
            color.create_dataset(camera_name, data=values, dtype=string_dtype)

        arm = episode.create_group("arm")
        position = arm.create_group("jointStatePosition")
        for arm_name in ("masterLeft", "masterRight", "puppetLeft", "puppetRight"):
            data = np.stack([np.arange(7, dtype=np.float64) + index for index in range(frame_count)])
            position.create_dataset(arm_name, data=data)

        full = episode.create_group("instructions").create_group("full_instructions")
        full.create_dataset("text", data=np.asarray([b"legacy"], dtype="S6"))
        episode.create_dataset("timestamp", data=np.arange(frame_count, dtype=np.float64))
        episode.create_dataset("size", data=np.asarray(frame_count, dtype=np.int64))
    return path


def _write_malformed_new_episode_missing_camera(path: Path) -> Path:
    episode_path = _write_new_schema_episode(path, frame_count=1)
    with h5py.File(episode_path, "a") as episode:
        del episode["/observations/images/cam_right_wrist"]
    return episode_path


def _write_ambiguous_schema_episode(path: Path) -> Path:
    episode_path = _write_official_legacy_episode(path, frame_count=1)
    with h5py.File(episode_path, "a") as episode:
        episode.attrs["schema_version"] = SCHEMA_VERSION
        episode.attrs["metadata_json"] = "{}"
        episode.attrs["camera_mapping_json"] = "{}"
        episode.attrs["units_json"] = "{}"
        episode.attrs["image_preprocessing_json"] = "{}"
        episode.attrs["timing_counters_json"] = "{}"
        episode.attrs["frame_count"] = 1
        episode.attrs["jpeg_quality"] = 90
        observations = episode.create_group("observations")
        observations.create_dataset("state", data=np.zeros((1, FRAME_DIMENSION), dtype=np.float32), maxshape=(None, FRAME_DIMENSION))
        observations.create_dataset("timestamp", data=np.zeros((1,), dtype=np.float32), maxshape=(None,))
        episode.create_dataset("action", data=np.zeros((1, FRAME_DIMENSION), dtype=np.float32), maxshape=(None, FRAME_DIMENSION))
        vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
        images = observations.create_group("images")
        sensor_timestamps = observations.create_group("sensor_timestamps")
        sync_error = observations.create_group("sync_error")
        for camera in IMAGE_SENSORS:
            images.create_dataset(camera, shape=(1,), dtype=vlen_uint8, maxshape=(None,))
        for sensor in REQUIRED_SENSORS:
            sensor_timestamps.create_dataset(sensor, data=np.zeros((1,), dtype=np.float32), maxshape=(None,))
            sync_error.create_dataset(sensor, data=np.zeros((1,), dtype=np.float32), maxshape=(None,))
    return episode_path


def _write_unrecognized_hdf5(path: Path) -> Path:
    with h5py.File(path, "w") as episode:
        episode.create_dataset("not_a_schema", data=np.arange(3))
    return path


def _write_invalid_visualizer_dispatch_case(tmp_path: Path, case: str) -> Path:
    if case == "malformed_new_missing_camera":
        return _write_malformed_new_episode_missing_camera(tmp_path / "malformed_new_missing_camera.hdf5")
    if case == "ambiguous_mixed":
        return _write_ambiguous_schema_episode(tmp_path / "ambiguous_mixed.hdf5")
    if case == "unrecognized":
        return _write_unrecognized_hdf5(tmp_path / "unrecognized.hdf5")
    raise AssertionError(case)


def _read_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened(), path
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            assert frame is not None
            frames.append(frame)
    finally:
        capture.release()
    return frames


def _assert_color_close(actual: np.ndarray, expected: tuple[int, int, int], tolerance: int = 40) -> None:
    expected_array = np.asarray(expected, dtype=np.int16)
    actual_array = np.asarray(actual, dtype=np.int16)
    assert np.max(np.abs(actual_array - expected_array)) <= tolerance, (actual_array, expected_array)


def test_render_episode_writes_three_camera_videos_views_and_quality_report_when_plots_disabled(tmp_path: Path) -> None:
    episode_path = _write_new_schema_episode(tmp_path / "episode.hdf5")
    output_dir = tmp_path / "rendered"

    report = render_episode(episode_path, output_dir, fps=12, make_plots=False)

    expected_videos = {
        "cam_high.mp4",
        "cam_left_wrist.mp4",
        "cam_right_wrist.mp4",
        "views_3x1.mp4",
    }
    for name in expected_videos:
        assert (output_dir / name).is_file()
    assert not (output_dir / "state_action.png").exists()
    assert (output_dir / "quality.json").is_file()

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["frame_count"] == 5
    assert report["image_decode_counts"] == {camera: 5 for camera in IMAGE_SENSORS}
    assert report["state_action_plot"]["status"] == "skipped"
    assert "no-plots" in report["state_action_plot"]["reason"].lower()

    quality = json.loads((output_dir / "quality.json").read_text())
    assert quality == report

    for camera, expected_color in CAMERA_COLORS.items():
        frames = _read_video(output_dir / f"{camera}.mp4")
        assert len(frames) == 5
        center_pixel = frames[0][frames[0].shape[0] // 2, frames[0].shape[1] // 2]
        _assert_color_close(center_pixel, expected_color)

    views = _read_video(output_dir / "views_3x1.mp4")
    assert len(views) == 5
    first = views[0]
    width_third = first.shape[1] // 3
    sample_y = max(0, first.shape[0] // 4)
    sampled_colors = [
        first[sample_y, width_third // 2],
        first[sample_y, width_third + width_third // 2],
        first[sample_y, 2 * width_third + width_third // 2],
    ]
    for actual, expected in zip(sampled_colors, CAMERA_COLORS.values(), strict=True):
        _assert_color_close(actual, expected)


def test_render_episode_can_generate_state_action_plot(tmp_path: Path) -> None:
    episode_path = _write_new_schema_episode(tmp_path / "episode.hdf5")
    output_dir = tmp_path / "rendered"

    report = render_episode(episode_path, output_dir, fps=30, make_plots=True)

    plot_report = report["state_action_plot"]
    if plot_report["status"] == "created":
        plot_path = output_dir / "state_action.png"
        assert plot_path.is_file()
        assert plot_path.stat().st_size > 0
    else:
        assert plot_report["status"] == "skipped"
        assert "matplotlib" in plot_report["reason"].lower()
        assert not (output_dir / "state_action.png").exists()


def test_render_episode_refuses_legacy_schema_with_clear_message(tmp_path: Path) -> None:
    legacy_path = _write_legacy_episode(tmp_path / "legacy.hdf5")

    with pytest.raises(ValueError, match="legacy schema; use existing visualizer"):
        render_episode(legacy_path, tmp_path / "output")



def test_visualize_hdf5_schema_gate_distinguishes_valid_new_from_official_legacy(tmp_path: Path) -> None:
    new_path = _write_new_schema_episode(tmp_path / "new.hdf5")
    legacy_path = _write_official_legacy_episode(tmp_path / "legacy_official.hdf5")

    assert visualize_hdf5._is_new_schema_episode(new_path) is True
    assert visualize_hdf5._is_new_schema_episode(legacy_path) is False


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("malformed_new_missing_camera", "malformed piper_dual_ros2_v1.*observations/images/cam_right_wrist"),
        ("ambiguous_mixed", "ambiguous HDF5 schema.*piper_dual_ros2_v1.*legacy_official_hdf5"),
        ("unrecognized", "unrecognized HDF5 schema.*missing"),
    ],
)
def test_visualize_hdf5_main_rejects_invalid_schema_without_constructing_legacy_visualizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, match: str
) -> None:
    episode_path = _write_invalid_visualizer_dispatch_case(tmp_path, case)
    calls: list[tuple[str, object]] = []

    def fake_render_episode(path: Path, output_dir: Path, fps: int = 30, make_plots: bool = True) -> dict:
        calls.append(("render", (Path(path), Path(output_dir), fps, make_plots)))
        raise AssertionError("render_episode should not be called for invalid schemas")

    class FakeVisualizer:
        def __init__(self, hdf5_path: Path) -> None:
            calls.append(("legacy_init", Path(hdf5_path)))
            raise AssertionError("HDF5Visualizer should not be constructed for invalid schemas")

    monkeypatch.setattr(visualize_hdf5, "render_episode", fake_render_episode, raising=False)
    monkeypatch.setattr(visualize_hdf5, "HDF5Visualizer", FakeVisualizer)
    monkeypatch.setattr(sys, "argv", ["visualize_hdf5.py", "--hdf5_path", str(episode_path), "--make_video"])

    with pytest.raises(ValueError, match=match):
        visualize_hdf5.main()
    assert calls == []


def test_visualize_hdf5_cli_reports_invalid_schema_and_exits_nonzero(tmp_path: Path) -> None:
    episode_path = _write_malformed_new_episode_missing_camera(tmp_path / "malformed_cli.hdf5")
    script_path = Path(visualize_hdf5.__file__).resolve()

    completed = subprocess.run(
        [sys.executable, str(script_path), "--hdf5_path", str(episode_path), "--make_video"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "malformed piper_dual_ros2_v1" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert "✅ 可视化完成" not in completed.stdout


def test_visualize_hdf5_dispatches_new_schema_to_renderer_and_preserves_legacy_behavior(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    new_path = _write_new_schema_episode(tmp_path / "new.hdf5")
    legacy_path = _write_official_legacy_episode(tmp_path / "legacy_official.hdf5")

    calls: list[tuple[str, object]] = []

    def fake_render_episode(path: Path, output_dir: Path, fps: int = 30, make_plots: bool = True) -> dict:
        calls.append(("render", Path(path)))
        calls.append(("output_dir", Path(output_dir)))
        calls.append(("fps", fps))
        calls.append(("make_plots", make_plots))
        return {
            "schema_version": SCHEMA_VERSION,
            "frame_count": 5,
            "image_decode_counts": {camera: 5 for camera in IMAGE_SENSORS},
            "state_action_plot": {"status": "skipped", "reason": "patched"},
        }

    class FakeVisualizer:
        def __init__(self, hdf5_path: Path) -> None:
            calls.append(("legacy_init", Path(hdf5_path)))

        def plot_joint_curves(self, save_path: str | None = None) -> None:
            calls.append(("legacy_plot", save_path))

        def make_video_from_images(self, cam_name: str = "cam_high", save_path: str | None = None, preview: bool = True) -> None:
            calls.append(("legacy_video", (cam_name, save_path, preview)))

        def close(self) -> None:
            calls.append(("legacy_close", None))

    monkeypatch.setattr(visualize_hdf5, "render_episode", fake_render_episode, raising=False)
    monkeypatch.setattr(visualize_hdf5, "HDF5Visualizer", FakeVisualizer)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_hdf5.py",
            "--hdf5_path",
            str(new_path),
            "--output_dir",
            str(tmp_path / "rendered"),
            "--fps",
            "24",
            "--no_plots",
        ],
    )
    assert visualize_hdf5.main() is None
    assert calls[:4] == [
        ("render", new_path),
        ("output_dir", tmp_path / "rendered"),
        ("fps", 24),
        ("make_plots", False),
    ]

    calls.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize_hdf5.py",
            "--hdf5_path",
            str(legacy_path),
            "--joint_plot",
            "--make_video",
            "--cam_name",
            "cam_left_wrist",
            "--video_save_path",
            str(tmp_path / "legacy.mp4"),
            "--no_preview",
        ],
    )
    assert visualize_hdf5.main() is None
    assert calls[0][0] == "legacy_init"
    assert any(call[0] == "legacy_plot" and call[1] == "output/joint_curves.png" for call in calls)
    assert calls[-1][0] == "legacy_close"
