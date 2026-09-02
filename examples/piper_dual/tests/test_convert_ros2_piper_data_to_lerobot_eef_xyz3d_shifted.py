"""Tests for raw XYZ+RPY shifted Piper dual ROS2 EEF conversion."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "utils" / "convert_ros2_piper_data_to_lerobot_eef_xyz3d_shifted.py"
spec = importlib.util.spec_from_file_location("convert_ros2_piper_data_to_lerobot_eef_xyz3d_shifted", SCRIPT)
assert spec is not None and spec.loader is not None
converter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = converter
spec.loader.exec_module(converter)


def _jpeg(value: int) -> bytes:
    image = np.full((4, 6, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return bytes(encoded)


def _write_episode(
    path: Path,
    *,
    frame_count: int = 3,
    eef_enabled: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = np.zeros((frame_count, 6), dtype=np.float32)
    right = np.zeros((frame_count, 6), dtype=np.float32)
    master = np.zeros((frame_count, 14), dtype=np.float32)
    for index in range(frame_count):
        left[index] = (index + 1, index + 2, index + 3, 0.1 + index, 0.2 + index, 0.3 + index)
        right[index] = (index + 11, index + 12, index + 13, -0.1 - index, -0.2 - index, -0.3 - index)
        master[index, 6] = 100 + index
        master[index, 13] = 200 + index

    with h5py.File(path, "w") as episode:
        episode.attrs["schema_version"] = "piper_dual_ros2_v1"
        episode.attrs["frame_count"] = frame_count
        metadata = {"prompt": "Stack the cups"}
        if eef_enabled:
            metadata["eef"] = {"enabled": True}
        episode.attrs["metadata_json"] = json.dumps(metadata)
        episode.create_dataset("observations/state", data=np.zeros((frame_count, 14), dtype=np.float32))
        episode.create_dataset("action", data=master)
        episode.create_dataset("observations/timestamp", data=np.arange(frame_count, dtype=np.float32) / 30.0)
        eef_group = episode.create_group("observations/eef")
        eef_group.create_dataset("puppet_left", data=left)
        eef_group.create_dataset("puppet_right", data=right)
        images = episode.create_group("observations/images")
        for camera_offset, camera in enumerate(converter.IMAGE_SENSORS):
            dataset = images.create_dataset(
                camera,
                shape=(frame_count,),
                dtype=h5py.vlen_dtype(np.dtype("uint8")),
            )
            for index in range(frame_count):
                dataset[index] = np.frombuffer(_jpeg(index * 20 + camera_offset), dtype=np.uint8)
    return left, right, master


class _FakeDataset:
    fps = 30

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.saved_episodes = 0

    def add_frame(self, frame: dict) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        self.saved_episodes += 1


def test_build_shifted_eef_state_action_preserves_raw_xyzrpy_and_shifts_pose() -> None:
    left = np.array(
        [[1, 2, 3, 0.1, 0.2, 0.3], [4, 5, 6, 0.4, 0.5, 0.6], [7, 8, 9, 0.7, 0.8, 0.9]],
        dtype=np.float32,
    )
    right = np.array(
        [[10, 11, 12, -0.1, -0.2, -0.3], [13, 14, 15, -0.4, -0.5, -0.6], [16, 17, 18, -0.7, -0.8, -0.9]],
        dtype=np.float32,
    )
    master = np.zeros((3, 14), dtype=np.float32)
    master[:, 6] = [60, 160, 260]
    master[:, 13] = [70, 170, 270]

    state, action = converter.build_shifted_eef_state_action(left, right, master)

    assert state.shape == action.shape == (2, 14)
    assert state.dtype == action.dtype == np.float32
    np.testing.assert_array_equal(state[0], np.r_[left[0], 60, right[0], 70])
    np.testing.assert_array_equal(action[0], np.r_[left[1], 60, right[1], 70])
    np.testing.assert_array_equal(action[1], np.r_[left[2], 160, right[2], 170])


@pytest.mark.parametrize(
    ("left", "right", "master", "message"),
    [
        (np.zeros((1, 6), dtype=np.float32), np.zeros((1, 6), dtype=np.float32), np.zeros((1, 14), dtype=np.float32), "at least two"),
        (np.zeros((2, 5), dtype=np.float32), np.zeros((2, 6), dtype=np.float32), np.zeros((2, 14), dtype=np.float32), "shape"),
        (np.zeros((2, 6), dtype=np.float32), np.zeros((3, 6), dtype=np.float32), np.zeros((2, 14), dtype=np.float32), "same number"),
        (np.zeros((2, 6), dtype=np.float32), np.zeros((2, 6), dtype=np.float32), np.zeros((2, 13), dtype=np.float32), "shape"),
        (np.array([[np.nan] * 6, [0.0] * 6], dtype=np.float32), np.zeros((2, 6), dtype=np.float32), np.zeros((2, 14), dtype=np.float32), "finite"),
        (np.zeros((2, 6), dtype=np.float32), np.zeros((2, 6), dtype=np.float32), np.array([[np.inf] * 14, [0.0] * 14], dtype=np.float32), "finite"),
    ],
)
def test_build_shifted_eef_state_action_rejects_invalid_inputs(
    left: np.ndarray, right: np.ndarray, master: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        converter.build_shifted_eef_state_action(left, right, master)


def test_populate_dataset_writes_14d_raw_eef_state_action_and_current_images(tmp_path: Path) -> None:
    path = tmp_path / "episode_000000.hdf5"
    left, right, master = _write_episode(path)
    dataset = _FakeDataset()

    converter.populate_dataset(dataset, [path])

    assert len(dataset.frames) == 2
    assert dataset.saved_episodes == 1
    expected_state, expected_action = converter.build_shifted_eef_state_action(left, right, master)
    for index, frame in enumerate(dataset.frames):
        np.testing.assert_array_equal(frame["observation.state"], expected_state[index])
        np.testing.assert_array_equal(frame["action"], expected_action[index])
        assert frame["task"] == "Stack the cups"
        assert "timestamp" not in frame
    assert dataset.frames[0]["observation.images.cam_high"].shape == (224, 224, 3)
    assert int(dataset.frames[1]["observation.images.cam_high"][112, 112, 0]) == 20


def test_populate_dataset_rejects_non_eef_episode(tmp_path: Path) -> None:
    path = tmp_path / "episode_000000.hdf5"
    _write_episode(path, eef_enabled=False)

    with pytest.raises(ValueError, match="eef.enabled"):
        converter.populate_dataset(_FakeDataset(), [path])


def test_create_empty_dataset_declares_14d_xyzrpy_gripper_features(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    class FakeLeRobotDataset:
        @classmethod
        def create(cls, **kwargs):
            captured.update(kwargs)
            return cls()

    monkeypatch.setattr(converter, "LeRobotDataset", FakeLeRobotDataset)

    converter.create_empty_dataset("local/eef_xyz3d", tmp_path / "eef_xyz3d", mode="image")

    assert captured["features"]["observation.state"]["shape"] == (14,)
    assert captured["features"]["action"]["shape"] == (14,)
    assert captured["features"]["observation.state"]["names"] == [converter.EEF_FEATURE_NAMES]
    assert captured["features"]["action"]["names"] == [converter.EEF_FEATURE_NAMES]
    assert captured["robot_type"] == "piper_dual_ros2_eef"
    assert converter.EEF_FEATURE_NAMES == [
        "eef_puppet_left_x",
        "eef_puppet_left_y",
        "eef_puppet_left_z",
        "eef_puppet_left_roll",
        "eef_puppet_left_pitch",
        "eef_puppet_left_yaw",
        "left_gripper",
        "eef_puppet_right_x",
        "eef_puppet_right_y",
        "eef_puppet_right_z",
        "eef_puppet_right_roll",
        "eef_puppet_right_pitch",
        "eef_puppet_right_yaw",
        "right_gripper",
    ]


def test_populate_dataset_rejects_one_frame_eef_episode(tmp_path: Path) -> None:
    path = tmp_path / "episode_000000.hdf5"
    _write_episode(path, frame_count=1)

    with pytest.raises(ValueError, match="at least two"):
        converter.populate_dataset(_FakeDataset(), [path])


def test_parse_args_supports_existing_converter_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_ros2_piper_data_to_lerobot_eef_xyz3d_shifted.py",
            "--raw-dir",
            "/tmp/raw",
            "--repo-id",
            "local/eef_xyz3d",
            "--mode",
            "video",
            "--episodes",
            "0,2",
            "--overwrite",
        ],
    )

    args = converter._parse_args()

    assert args.raw_dir == Path("/tmp/raw")
    assert args.repo_id == "local/eef_xyz3d"
    assert args.mode == "video"
    assert args.episodes == "0,2"
    assert args.overwrite is True
