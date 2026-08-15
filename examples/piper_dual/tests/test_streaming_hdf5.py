"""Tests for crash-safe streaming HDF5 episode writing."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frame_synchronizer import SynchronizedFrame
from streaming_hdf5 import SCHEMA_VERSION
from streaming_hdf5 import StreamingEpisodeWriter
from streaming_hdf5 import validate_episode

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
SENSORS = CAMERAS + ("puppet_left", "puppet_right", "master_left", "master_right")


def jpeg_bytes(seed: int) -> bytes:
    image = np.zeros((8, 9, 3), dtype=np.uint8)
    image[:, :, 0] = seed
    image[:, :, 1] = np.arange(9, dtype=np.uint8)
    image[:, :, 2] = np.arange(8, dtype=np.uint8)[:, None]
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    assert ok
    return bytes(encoded)


def frame(index: int, *, timestamp: float | None = None) -> SynchronizedFrame:
    base_time = 10.0 + index * 0.05 if timestamp is None else timestamp
    return SynchronizedFrame(
        timestamp=base_time,
        images={camera: jpeg_bytes(index * 20 + offset) for offset, camera in enumerate(CAMERAS)},
        state=(np.arange(14, dtype=np.float32) + index).astype(np.float32),
        action=(np.arange(14, dtype=np.float32) + 100 + index).astype(np.float32),
        sensor_timestamps={sensor: base_time + sensor_index * 0.001 for sensor_index, sensor in enumerate(SENSORS)},
        sync_error={sensor: sensor_index * 0.001 for sensor_index, sensor in enumerate(SENSORS)},
    )


def open_writer(path: Path, *, metadata: dict | None = None, flush_every: int = 30) -> StreamingEpisodeWriter:
    return StreamingEpisodeWriter.open(
        path,
        metadata={"episode_id": "episode-7", "operator": "test", **(metadata or {})},
        jpeg_quality=82,
        flush_every=flush_every,
    )


def test_finalize_rejects_zero_frame_episode_and_leaves_diagnostic_partial(tmp_path: Path):
    final_path = tmp_path / "zero.hdf5"
    writer = open_writer(final_path)

    with pytest.raises(ValueError, match="zero frames"):
        writer.finalize()

    assert not final_path.exists()
    assert final_path.with_suffix(".hdf5.partial").exists()
    report = validate_episode(final_path.with_suffix(".hdf5.partial"))
    assert report["frame_count"] == 0
    assert any("zero frames" in error for error in report["errors"])


def test_single_frame_append_round_trips_vlen_jpeg_bytes(tmp_path: Path):
    final_path = tmp_path / "single.hdf5"
    expected = frame(0)
    writer = open_writer(final_path)

    writer.append(expected)
    writer.finalize()

    with h5py.File(final_path, "r") as episode:
        assert episode.attrs["frame_count"] == 1
        np.testing.assert_array_equal(episode["/observations/state"][0], expected.state)
        np.testing.assert_array_equal(episode["/action"][0], expected.action)
        for camera in CAMERAS:
            stored = episode[f"/observations/images/{camera}"][0]
            assert h5py.check_dtype(vlen=episode[f"/observations/images/{camera}"].dtype) == np.dtype("uint8")
            assert bytes(np.asarray(stored, dtype=np.uint8)) == expected.images[camera]
            decoded = cv2.imdecode(np.asarray(stored, dtype=np.uint8), cv2.IMREAD_COLOR)
            assert decoded is not None


def test_multi_frame_append_is_self_contained_and_finalizes_atomically(tmp_path: Path):
    final_path = tmp_path / "episode.hdf5"
    source_buffers = [frame(0), frame(1), frame(2)]
    writer = open_writer(final_path, metadata={"task": "stack blocks", "nested": {"side": "left"}}, flush_every=2)

    for item in source_buffers:
        writer.append(item)

    partial_path = final_path.with_suffix(".hdf5.partial")
    assert partial_path.exists()
    assert not final_path.exists()

    result = writer.finalize()

    assert result == final_path
    assert final_path.exists()
    assert not partial_path.exists()

    del source_buffers
    with h5py.File(final_path, "r") as episode:
        assert episode.attrs["schema_version"] == SCHEMA_VERSION
        assert episode.attrs["frame_count"] == 3
        assert episode.attrs["jpeg_quality"] == 82
        metadata = json.loads(episode.attrs["metadata_json"])
        assert metadata["task"] == "stack blocks"
        assert metadata["nested"] == {"side": "left"}
        assert json.loads(episode.attrs["camera_mapping_json"]) == {camera: f"/observations/images/{camera}" for camera in CAMERAS}
        assert json.loads(episode.attrs["units_json"])["timestamp"] == "seconds"
        assert json.loads(episode.attrs["image_preprocessing_json"])["encoding"] == "jpeg_bytes_passthrough"
        assert json.loads(episode.attrs["timing_counters_json"])["frames_written"] == 3

        np.testing.assert_array_equal(episode["/observations/state"][:], np.stack([frame(i).state for i in range(3)]))
        np.testing.assert_array_equal(episode["/action"][:], np.stack([frame(i).action for i in range(3)]))
        np.testing.assert_allclose(episode["/observations/timestamp"][:], [10.0, 10.05, 10.1])
        assert episode["/observations/state"].dtype == np.dtype("float32")
        assert episode["/action"].dtype == np.dtype("float32")
        assert episode["/observations/timestamp"].dtype == np.dtype("float32")
        assert episode["/observations/state"].maxshape == (None, 14)
        assert episode["/action"].maxshape == (None, 14)
        assert episode["/observations/timestamp"].maxshape == (None,)

        for camera in CAMERAS:
            dataset = episode[f"/observations/images/{camera}"]
            assert dataset.shape == (3,)
            assert dataset.maxshape == (None,)
            assert h5py.check_dtype(vlen=dataset.dtype) == np.dtype("uint8")
            for index, encoded in enumerate(dataset):
                decoded = cv2.imdecode(np.asarray(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                assert decoded is not None, f"{camera} frame {index} did not decode"

        for sensor in SENSORS:
            assert episode[f"/observations/sensor_timestamps/{sensor}"].shape == (3,)
            assert episode[f"/observations/sensor_timestamps/{sensor}"].dtype == np.dtype("float32")
            assert episode[f"/observations/sync_error/{sensor}"].shape == (3,)
            assert episode[f"/observations/sync_error/{sensor}"].dtype == np.dtype("float32")

    validation = validate_episode(final_path)
    assert validation["schema_version"] == SCHEMA_VERSION
    assert validation["frame_count"] == 3
    assert validation["timestamp_monotonic"] is True
    assert validation["image_decode_counts"] == {camera: 3 for camera in CAMERAS}
    assert validation["errors"] == []


def test_abort_removes_or_keeps_partial_without_publishing_final_file(tmp_path: Path):
    removed_path = tmp_path / "removed.hdf5"
    removed_writer = open_writer(removed_path)
    removed_writer.append(frame(0))
    removed_writer.abort()

    assert not removed_path.exists()
    assert not removed_path.with_suffix(".hdf5.partial").exists()

    kept_path = tmp_path / "kept.hdf5"
    kept_writer = open_writer(kept_path)
    kept_writer.append(frame(0))
    kept_writer.abort(remove_partial=False)

    assert not kept_path.exists()
    assert kept_path.with_suffix(".hdf5.partial").exists()


def test_append_validates_required_frame_shape_and_camera_bytes(tmp_path: Path):
    writer = open_writer(tmp_path / "bad-state.hdf5")
    malformed = frame(0)
    malformed = SynchronizedFrame(
        timestamp=malformed.timestamp,
        images=malformed.images,
        state=np.zeros(13, dtype=np.float32),
        action=malformed.action,
        sensor_timestamps=malformed.sensor_timestamps,
        sync_error=malformed.sync_error,
    )
    with pytest.raises(ValueError, match="state.*14"):
        writer.append(malformed)
    writer.abort()

    writer = open_writer(tmp_path / "bad-image.hdf5")
    malformed = frame(0)
    malformed = SynchronizedFrame(
        timestamp=malformed.timestamp,
        images={**malformed.images, "cam_high": "not bytes"},
        state=malformed.state,
        action=malformed.action,
        sensor_timestamps=malformed.sensor_timestamps,
        sync_error=malformed.sync_error,
    )
    with pytest.raises(ValueError, match="JPEG bytes"):
        writer.append(malformed)
    writer.abort()
