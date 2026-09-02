"""Tests for exporting synchronized Piper HDF5 data as JSON."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_episode_json import export_puppet_state


def _write_episode(path: Path) -> Path:
    with h5py.File(path, "w") as episode:
        episode.attrs["schema_version"] = "piper_dual_ros2_v1"
        episode.attrs["frame_count"] = 2
        episode.attrs["metadata_json"] = json.dumps({"collector": "test"})
        episode.attrs["units_json"] = json.dumps({"state": "mixed_rad_m", "timestamp": "seconds", "sync_error": "seconds"})
        state = np.asarray(
            [
                [1, 2, 3, 4, 5, 6, 0.07, 11, 12, 13, 14, 15, 16, 0.08],
                [21, 22, 23, 24, 25, 26, 0.06, 31, 32, 33, 34, 35, 36, 0.09],
            ],
            dtype=np.float32,
        )
        episode.create_dataset("observations/state", data=state)
        episode.create_dataset("observations/timestamp", data=np.asarray([100.0, 100.1], dtype=np.float32))
        timestamps = episode.create_group("observations/sensor_timestamps")
        errors = episode.create_group("observations/sync_error")
        for sensor, values in {
            "puppet_left": [100.01, 100.11],
            "puppet_right": [99.99, 100.09],
        }.items():
            timestamps.create_dataset(sensor, data=np.asarray(values, dtype=np.float32))
            errors.create_dataset(sensor, data=np.asarray([0.01, -0.01], dtype=np.float32))
    return path


def _add_eef_data(path: Path) -> Path:
    with h5py.File(path, "a") as episode:
        units = json.loads(episode.attrs["units_json"])
        units["eef"] = "xyz_m_rpy_rad"
        episode.attrs["units_json"] = json.dumps(units)
        eef_group = episode.create_group("observations/eef")
        eef_group.create_dataset(
            "puppet_left",
            data=np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]], dtype=np.float32),
        )
        eef_group.create_dataset(
            "puppet_right",
            data=np.asarray([[0.7, 0.8, 0.9, 1.0, 1.1, 1.2], [1.7, 1.8, 1.9, 2.0, 2.1, 2.2]], dtype=np.float32),
        )
        timestamps = episode["observations/sensor_timestamps"]
        errors = episode["observations/sync_error"]
        timestamps.create_dataset("eef_puppet_left", data=np.asarray([100.02, 100.12], dtype=np.float32))
        timestamps.create_dataset("eef_puppet_right", data=np.asarray([99.98, 100.08], dtype=np.float32))
        errors.create_dataset("eef_puppet_left", data=np.asarray([0.02, 0.02], dtype=np.float32))
        errors.create_dataset("eef_puppet_right", data=np.asarray([-0.02, -0.02], dtype=np.float32))
    return path


def test_export_puppet_state_includes_timestamped_eef_for_both_arms(tmp_path: Path) -> None:
    episode_path = _add_eef_data(_write_episode(tmp_path / "episode_eef.hdf5"))
    output_path = tmp_path / "puppet_state.json"

    report = export_puppet_state(episode_path, output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["frame_count"] == 2
    assert payload["eef_units"] == ["m", "m", "m", "rad", "rad", "rad"]
    assert payload["eef_names"] == ["x", "y", "z", "roll", "pitch", "yaw"]
    assert payload["frames"][0]["left"]["eef"] == {
        "timestamp": pytest.approx(100.02),
        "pose": pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        "sync_error": pytest.approx(0.02),
    }
    assert payload["frames"][1]["right"]["eef"] == {
        "timestamp": pytest.approx(100.08),
        "pose": pytest.approx([1.7, 1.8, 1.9, 2.0, 2.1, 2.2]),
        "sync_error": pytest.approx(-0.02),
    }


def test_export_puppet_state_rejects_partial_eef_data(tmp_path: Path) -> None:
    episode_path = _add_eef_data(_write_episode(tmp_path / "episode_eef.hdf5"))
    with h5py.File(episode_path, "a") as episode:
        del episode["observations/eef/puppet_right"]

    with pytest.raises(ValueError, match="EEF.*puppet_right"):
        export_puppet_state(episode_path, tmp_path / "puppet_state.json")


def test_export_puppet_state_writes_timestamped_left_and_right_arms(tmp_path: Path) -> None:
    output_path = tmp_path / "puppet_state.json"
    report = export_puppet_state(_write_episode(tmp_path / "episode.hdf5"), output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["frame_count"] == 2
    assert payload["schema_version"] == "piper_dual_puppet_state_v1"
    assert payload["joint_names"] == ["J1", "J2", "J3", "J4", "J5", "J6", "gripper"]
    assert payload["joint_units"] == ["rad", "rad", "rad", "rad", "rad", "rad", "m"]
    frames = payload["frames"]
    assert [frame["index"] for frame in frames] == [0, 1]
    assert [frame["reference_timestamp"] for frame in frames] == pytest.approx([100.0, 100.1])
    assert frames[0]["left"]["timestamp"] == pytest.approx(100.01)
    assert frames[0]["left"]["position"] == pytest.approx([1, 2, 3, 4, 5, 6, 0.07])
    assert frames[0]["left"]["sync_error"] == pytest.approx(0.01)
    assert frames[0]["right"]["timestamp"] == pytest.approx(99.99)
    assert frames[0]["right"]["position"] == pytest.approx([11, 12, 13, 14, 15, 16, 0.08])
    assert frames[0]["right"]["sync_error"] == pytest.approx(0.01)
    assert frames[1]["left"]["timestamp"] == pytest.approx(100.11)
    assert frames[1]["left"]["position"] == pytest.approx([21, 22, 23, 24, 25, 26, 0.06])
    assert frames[1]["left"]["sync_error"] == pytest.approx(-0.01)
    assert frames[1]["right"]["timestamp"] == pytest.approx(100.09)
    assert frames[1]["right"]["position"] == pytest.approx([31, 32, 33, 34, 35, 36, 0.09])
    assert frames[1]["right"]["sync_error"] == pytest.approx(-0.01)


def test_export_puppet_state_rejects_missing_sensor_timestamps(tmp_path: Path) -> None:
    episode_path = _write_episode(tmp_path / "episode.hdf5")
    with h5py.File(episode_path, "a") as episode:
        del episode["observations/sensor_timestamps/puppet_right"]

    try:
        export_puppet_state(episode_path, tmp_path / "puppet_state.json")
    except ValueError as exc:
        assert "puppet_right" in str(exc)
    else:
        raise AssertionError("expected missing puppet timestamp to fail")


@pytest.mark.parametrize(
    ("attribute", "value", "match"),
    [
        ("schema_version", "wrong_schema", "schema_version"),
        ("units_json", json.dumps({"state": "degrees"}), "units_json"),
    ],
)
def test_export_puppet_state_rejects_wrong_schema_or_units(
    tmp_path: Path, attribute: str, value: str, match: str
) -> None:
    episode_path = _write_episode(tmp_path / "episode.hdf5")
    with h5py.File(episode_path, "a") as episode:
        episode.attrs[attribute] = value

    with pytest.raises(ValueError, match=match):
        export_puppet_state(episode_path, tmp_path / "puppet_state.json")
