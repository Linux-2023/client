"""Compatibility tests for legacy official and ROS 2 Piper dual HDF5 schemas."""

from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frame_synchronizer import IMAGE_SENSORS
from frame_synchronizer import REQUIRED_SENSORS
from frame_synchronizer import SynchronizedFrame
from render_dataset import detect_schema
from streaming_hdf5 import SCHEMA_VERSION
from streaming_hdf5 import StreamingEpisodeWriter

LEGACY_SCHEMA = "legacy_official_hdf5"
OFFICIAL_LEGACY_EPISODE = Path("/home/agilex/lgd_data/episode0/episode0.hdf5")
LEGACY_CAMERA_NAMES = ("left", "front_l", "right")
LEGACY_ARM_NAMES = ("masterLeft", "masterRight", "puppetLeft", "puppetRight")


def _jpeg_bytes(seed: int) -> bytes:
    import cv2

    image = np.zeros((10, 12, 3), dtype=np.uint8)
    image[:] = (seed % 255, (seed + 60) % 255, (seed + 120) % 255)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return bytes(encoded)


def _frame(index: int) -> SynchronizedFrame:
    timestamp = 123.0 + index * 0.05
    return SynchronizedFrame(
        timestamp=timestamp,
        images={camera: _jpeg_bytes(index * 30 + offset) for offset, camera in enumerate(IMAGE_SENSORS)},
        state=(np.arange(14, dtype=np.float32) + index).astype(np.float32),
        action=(np.arange(14, dtype=np.float32) + 100.0 + index).astype(np.float32),
        sensor_timestamps={sensor: timestamp for sensor in REQUIRED_SENSORS},
        sync_error={sensor: 0.0 for sensor in REQUIRED_SENSORS},
    )


def _write_ros2_episode(path: Path, *, frame_count: int = 2) -> Path:
    writer = StreamingEpisodeWriter.open(
        path,
        metadata={"purpose": "compatibility", "prompt": "Fold the towel"},
        jpeg_quality=86,
        flush_every=1,
    )
    for index in range(frame_count):
        writer.append(_frame(index))
    return writer.finalize()


def _write_structural_legacy_episode(path: Path, *, frame_count: int = 3) -> Path:
    with h5py.File(path, "w") as episode:
        camera = episode.create_group("camera")
        color = camera.create_group("color")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        for camera_name in LEGACY_CAMERA_NAMES:
            values = np.asarray(
                [f"camera/color/{camera_name}/{1000.0 + index:.6f}.jpg" for index in range(frame_count)],
                dtype=object,
            )
            color.create_dataset(camera_name, data=values, dtype=string_dtype)

        arm = episode.create_group("arm")
        position = arm.create_group("jointStatePosition")
        velocity = arm.create_group("jointStateVelocity")
        effort = arm.create_group("jointStateEffort")
        for arm_name in LEGACY_ARM_NAMES:
            data = np.stack([np.arange(7, dtype=np.float64) + index for index in range(frame_count)])
            position.create_dataset(arm_name, data=data)
            velocity.create_dataset(arm_name, data=data * 0.1)
            effort.create_dataset(arm_name, data=data * 0.01)

        localization = episode.create_group("localization")
        pose = localization.create_group("pose")
        pose.create_dataset("puppetLeft", data=np.zeros((frame_count, 6), dtype=np.float64))
        pose.create_dataset("puppetRight", data=np.zeros((frame_count, 6), dtype=np.float64))
        instructions = episode.create_group("instructions")
        full = instructions.create_group("full_instructions")
        full.create_dataset("text", data=np.asarray([b"legacy"], dtype="S6"))
        episode.create_dataset("timestamp", data=np.arange(frame_count, dtype=np.float64))
        episode.create_dataset("size", data=np.asarray(frame_count, dtype=np.int64))
    return path


def test_detect_schema_identifies_source_verified_official_legacy_path_index_episode() -> None:
    assert OFFICIAL_LEGACY_EPISODE.is_file(), f"missing source-verified legacy fixture: {OFFICIAL_LEGACY_EPISODE}"

    assert detect_schema(OFFICIAL_LEGACY_EPISODE) == LEGACY_SCHEMA


@pytest.mark.parametrize("filename", ["legacy_official_hdf5.hdf5", "renamed_without_schema_hint.hdf5"])
def test_detect_schema_identifies_new_ros2_episode_from_metadata_and_required_paths_not_filename(
    tmp_path: Path, filename: str
) -> None:
    episode_path = _write_ros2_episode(tmp_path / filename)

    assert detect_schema(episode_path) == SCHEMA_VERSION


@pytest.mark.parametrize("filename", ["piper_dual_ros2_v1.hdf5", "renamed_legacy_episode.hdf5"])
def test_detect_schema_identifies_legacy_episode_from_structure_not_filename(tmp_path: Path, filename: str) -> None:
    legacy_path = _write_structural_legacy_episode(tmp_path / filename)

    assert detect_schema(legacy_path) == LEGACY_SCHEMA


def test_detect_schema_rejects_new_schema_metadata_without_required_datasets_descriptively(tmp_path: Path) -> None:
    malformed = tmp_path / "looks_new.hdf5"
    with h5py.File(malformed, "w") as episode:
        episode.attrs["schema_version"] = SCHEMA_VERSION
        episode.attrs["metadata_json"] = "{}"
        episode.create_group("observations")

    with pytest.raises(ValueError, match="malformed piper_dual_ros2_v1.*missing.*observations/state"):
        detect_schema(malformed)


def test_detect_schema_rejects_ambiguous_file_with_new_metadata_and_legacy_paths(tmp_path: Path) -> None:
    ambiguous = _write_structural_legacy_episode(tmp_path / "ambiguous.hdf5")
    with h5py.File(ambiguous, "a") as episode:
        episode.attrs["schema_version"] = SCHEMA_VERSION
        episode.attrs["metadata_json"] = "{}"
        episode.attrs["camera_mapping_json"] = "{}"
        episode.attrs["units_json"] = "{}"
        episode.attrs["image_preprocessing_json"] = "{}"
        episode.attrs["timing_counters_json"] = "{}"
        episode.attrs["frame_count"] = 1
        episode.attrs["jpeg_quality"] = 90
        observations = episode.require_group("observations")
        observations.create_dataset("state", data=np.zeros((1, 14), dtype=np.float32), maxshape=(None, 14))
        observations.create_dataset("timestamp", data=np.zeros((1,), dtype=np.float32), maxshape=(None,))
        episode.create_dataset("action", data=np.zeros((1, 14), dtype=np.float32), maxshape=(None, 14))
        vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
        images = observations.create_group("images")
        sensor_timestamps = observations.create_group("sensor_timestamps")
        sync_error = observations.create_group("sync_error")
        for camera in IMAGE_SENSORS:
            images.create_dataset(camera, shape=(1,), dtype=vlen_uint8, maxshape=(None,))
        for sensor in REQUIRED_SENSORS:
            sensor_timestamps.create_dataset(sensor, data=np.zeros((1,), dtype=np.float32), maxshape=(None,))
            sync_error.create_dataset(sensor, data=np.zeros((1,), dtype=np.float32), maxshape=(None,))

    with pytest.raises(ValueError, match="ambiguous HDF5 schema.*piper_dual_ros2_v1.*legacy_official_hdf5"):
        detect_schema(ambiguous)


def test_detect_schema_rejects_unrecognized_hdf5_without_trusting_extension(tmp_path: Path) -> None:
    unknown = tmp_path / "episode0.hdf5"
    with h5py.File(unknown, "w") as episode:
        episode.create_dataset("not_a_schema", data=np.arange(3))

    with pytest.raises(ValueError, match="unrecognized HDF5 schema.*missing"):
        detect_schema(unknown)
