"""Schema validation tests for Piper dual streaming HDF5 episodes."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streaming_hdf5 import SCHEMA_VERSION
from streaming_hdf5 import StreamingEpisodeWriter
from streaming_hdf5 import validate_episode
from test_streaming_hdf5 import CAMERAS
from test_streaming_hdf5 import SENSORS
from test_streaming_hdf5 import frame


def write_valid_episode(path: Path, count: int = 2) -> Path:
    writer = StreamingEpisodeWriter.open(path, metadata={"purpose": "schema-test"}, jpeg_quality=75, flush_every=1)
    for index in range(count):
        writer.append(frame(index))
    return writer.finalize()


def test_schema_contains_required_extendible_datasets_and_versioned_metadata(tmp_path: Path):
    final_path = write_valid_episode(tmp_path / "schema.hdf5", count=2)

    with h5py.File(final_path, "r") as episode:
        assert episode.attrs["schema_version"] == SCHEMA_VERSION
        assert episode.attrs["schema_version"] == "piper_dual_ros2_v1"
        assert json.loads(episode.attrs["metadata_json"]) == {"purpose": "schema-test"}
        assert episode["/observations/state"].shape == (2, 14)
        assert episode["/observations/state"].maxshape == (None, 14)
        assert episode["/action"].shape == (2, 14)
        assert episode["/action"].maxshape == (None, 14)
        assert episode["/observations/timestamp"].shape == (2,)
        assert episode["/observations/timestamp"].maxshape == (None,)
        assert episode["/observations/timestamp"].dtype == np.dtype("float32")

        for camera in CAMERAS:
            dataset = episode[f"/observations/images/{camera}"]
            assert dataset.shape == (2,)
            assert dataset.maxshape == (None,)
            assert dataset.chunks == (1,)
            assert h5py.check_dtype(vlen=dataset.dtype) == np.dtype("uint8")

        for sensor in SENSORS:
            timestamps = episode[f"/observations/sensor_timestamps/{sensor}"]
            errors = episode[f"/observations/sync_error/{sensor}"]
            assert timestamps.shape == (2,)
            assert timestamps.maxshape == (None,)
            assert timestamps.dtype == np.dtype("float32")
            assert errors.shape == (2,)
            assert errors.maxshape == (None,)
            assert errors.dtype == np.dtype("float32")


def test_validate_episode_reports_missing_mismatched_bad_jpeg_and_nonmonotonic_timestamp_errors(tmp_path: Path):
    final_path = write_valid_episode(tmp_path / "corrupt-source.hdf5", count=2)
    corrupt_path = tmp_path / "corrupt.hdf5"
    final_path.replace(corrupt_path)

    with h5py.File(corrupt_path, "a") as episode:
        episode.attrs["schema_version"] = "wrong"
        episode["/observations/timestamp"][1] = episode["/observations/timestamp"][0] - np.float32(1.0)
        episode["/observations/images/cam_high"][1] = np.frombuffer(b"not a jpeg", dtype=np.uint8)
        del episode["/observations/sensor_timestamps/master_right"]
        episode["/observations/sync_error/master_left"].resize((1,))

    report = validate_episode(corrupt_path)

    assert report["frame_count"] == 2
    assert report["schema_version"] == "wrong"
    assert report["timestamp_monotonic"] is False
    assert report["image_decode_counts"]["cam_high"] == 1
    assert any("schema version" in error for error in report["errors"])
    assert any("timestamps" in error and "nondecreasing" in error for error in report["errors"])
    assert any("cam_high" in error and "JPEG" in error for error in report["errors"])
    assert any("sensor_timestamps/master_right" in error for error in report["errors"])
    assert any("sync_error/master_left" in error and "length" in error for error in report["errors"])


def test_finalize_leaves_partial_when_validation_fails_and_does_not_replace_existing_final(tmp_path: Path):
    final_path = write_valid_episode(tmp_path / "published.hdf5", count=1)
    with h5py.File(final_path, "r") as episode:
        original_state = episode["/observations/state"][:]

    writer = StreamingEpisodeWriter.open(final_path, metadata={"purpose": "bad-finalize"})
    bad = frame(0)
    writer.append(bad)
    writer.flush()

    partial_path = final_path.with_suffix(".hdf5.partial")
    with h5py.File(partial_path, "r+") as episode:
        episode["/observations/images/cam_left_wrist"][0] = np.frombuffer(b"bad jpeg bytes", dtype=np.uint8)
        episode.flush()

    try:
        try:
            writer.finalize()
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError("finalize should reject invalid partial")

        assert "validation failed" in message
        assert final_path.exists()
        assert final_path.with_suffix(".hdf5.partial").exists()
        with h5py.File(final_path, "r") as episode:
            np.testing.assert_array_equal(episode["/observations/state"][:], original_state)
    finally:
        if final_path.with_suffix(".hdf5.partial").exists():
            final_path.with_suffix(".hdf5.partial").unlink()
