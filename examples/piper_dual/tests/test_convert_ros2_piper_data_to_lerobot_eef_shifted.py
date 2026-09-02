"""Tests for shifted Piper dual ROS2 EEF to LeRobot conversion."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "utils" / "convert_ros2_piper_data_to_lerobot_eef_shifted.py"
spec = importlib.util.spec_from_file_location("convert_ros2_piper_data_to_lerobot_eef_shifted", SCRIPT)
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
        left[index, :3] = (index + 1, index + 2, index + 3)
        right[index, :3] = (index + 11, index + 12, index + 13)
        left[index, 3:] = (0.1 * index, 0.0, 0.0)
        right[index, 3:] = (0.0, 0.0, np.pi / 2.0 * index)
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
                encoded = _jpeg(index * 20 + camera_offset)
                dataset[index] = np.frombuffer(encoded, dtype=np.uint8)
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


def test_rpy_to_rot6d_uses_ros_xyz_order_and_matrix_columns() -> None:
    result = converter.rpy_to_rot6d(np.array([0.0, 0.0, np.pi / 2.0], dtype=np.float32))

    np.testing.assert_allclose(result, [0.0, 1.0, 0.0, -1.0, 0.0, 0.0], atol=1e-6)
    assert result.dtype == np.float32


def test_build_shifted_eef_state_action_uses_current_gripper_with_current_and_next_pose() -> None:
    left = np.array(
        [[1, 2, 3, 0, 0, 0], [4, 5, 6, 0, 0, 0], [7, 8, 9, 0, 0, 0]],
        dtype=np.float32,
    )
    right = np.array(
        [[10, 11, 12, 0, 0, 0], [13, 14, 15, 0, 0, np.pi / 2], [16, 17, 18, 0, 0, np.pi]],
        dtype=np.float32,
    )
    master = np.zeros((3, 14), dtype=np.float32)
    master[:, 6] = [60, 160, 260]
    master[:, 13] = [70, 170, 270]

    state, action = converter.build_shifted_eef_state_action(left, right, master)

    assert state.shape == action.shape == (2, 20)
    assert state.dtype == action.dtype == np.float32
    np.testing.assert_array_equal(state[0, :9], converter.eef_rpy_to_rot9(left[0]))
    assert state[0, 9] == 60
    np.testing.assert_array_equal(state[0, 10:19], converter.eef_rpy_to_rot9(right[0]))
    assert state[0, 19] == 70
    np.testing.assert_array_equal(action[0, :9], converter.eef_rpy_to_rot9(left[1]))
    assert action[0, 9] == 60
    np.testing.assert_array_equal(action[0, 10:19], converter.eef_rpy_to_rot9(right[1]))
    assert action[0, 19] == 70
    np.testing.assert_array_equal(
        action[1],
        np.concatenate(
            (
                converter.eef_rpy_to_rot9(left[2]),
                np.array([160], dtype=np.float32),
                converter.eef_rpy_to_rot9(right[2]),
                np.array([170], dtype=np.float32),
            )
        ),
    )


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


def test_populate_dataset_writes_20d_eef_state_action_and_current_images(tmp_path: Path) -> None:
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


def test_create_empty_dataset_declares_20d_eef_gripper_features(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    class FakeLeRobotDataset:
        @classmethod
        def create(cls, **kwargs):
            captured.update(kwargs)
            return cls()

    monkeypatch.setattr(converter, "LeRobotDataset", FakeLeRobotDataset)

    converter.create_empty_dataset("local/eef", tmp_path / "eef", mode="image")

    assert captured["features"]["observation.state"]["shape"] == (20,)
    assert captured["features"]["action"]["shape"] == (20,)
    assert captured["features"]["observation.state"]["names"] == [converter.EEF_FEATURE_NAMES]
    assert captured["features"]["action"]["names"] == [converter.EEF_FEATURE_NAMES]


def test_populate_dataset_rejects_one_frame_eef_episode(tmp_path: Path) -> None:
    path = tmp_path / "episode_000000.hdf5"
    _write_episode(path, frame_count=1)

    with pytest.raises(ValueError, match="at least two"):
        converter.populate_dataset(_FakeDataset(), [path])
