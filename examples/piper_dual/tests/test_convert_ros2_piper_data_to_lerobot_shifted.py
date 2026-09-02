"""Tests for the shifted Piper dual ROS2 to LeRobot mapping."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import json

import cv2
import h5py

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "utils" / "convert_ros2_piper_data_to_lerobot_shifted.py"
spec = importlib.util.spec_from_file_location("convert_ros2_piper_data_to_lerobot_shifted", SCRIPT)
assert spec is not None and spec.loader is not None
converter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = converter
spec.loader.exec_module(converter)


def _jpeg(value: int) -> bytes:
    image = np.full((4, 6, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return bytes(encoded)


def _write_episode(path: Path, *, frame_count: int = 3) -> tuple[np.ndarray, np.ndarray]:
    puppet = np.arange(frame_count * 14, dtype=np.float32).reshape(frame_count, 14)
    master = puppet + np.float32(1_000.0)
    with h5py.File(path, "w") as episode:
        episode.attrs["schema_version"] = "piper_dual_ros2_v1"
        episode.attrs["frame_count"] = frame_count
        episode.attrs["metadata_json"] = json.dumps({"prompt": "Stack the cups"})
        episode.create_dataset("observations/state", data=puppet)
        episode.create_dataset("action", data=master)
        episode.create_dataset("observations/timestamp", data=np.arange(frame_count, dtype=np.float32) / 30.0)
        images = episode.create_group("observations/images")
        for camera_offset, camera in enumerate(converter.IMAGE_SENSORS):
            dataset = images.create_dataset(camera, shape=(frame_count,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
            for index in range(frame_count):
                encoded = _jpeg(index * 20 + camera_offset)
                dataset[index] = np.frombuffer(encoded, dtype=np.uint8)
    return puppet, master


class _FakeDataset:
    fps = 30

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.saved_episodes = 0

    def add_frame(self, frame: dict) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        self.saved_episodes += 1


def test_build_shifted_state_action_uses_next_puppet_joints_and_current_master_grippers() -> None:
    puppet = np.array(
        [
            [0, 1, 2, 3, 4, 5, 60, 10, 11, 12, 13, 14, 15, 70],
            [100, 101, 102, 103, 104, 105, 160, 110, 111, 112, 113, 114, 115, 170],
            [200, 201, 202, 203, 204, 205, 260, 210, 211, 212, 213, 214, 215, 270],
        ],
        dtype=np.float32,
    )
    master = np.array(
        [
            [300, 301, 302, 303, 304, 305, 360, 310, 311, 312, 313, 314, 315, 370],
            [400, 401, 402, 403, 404, 405, 460, 410, 411, 412, 413, 414, 415, 470],
            [500, 501, 502, 503, 504, 505, 560, 510, 511, 512, 513, 514, 515, 570],
        ],
        dtype=np.float32,
    )

    state, action = converter.build_shifted_state_action(puppet, master)

    np.testing.assert_array_equal(
        state,
        np.array(
            [
                [0, 1, 2, 3, 4, 5, 360, 10, 11, 12, 13, 14, 15, 370],
                [100, 101, 102, 103, 104, 105, 460, 110, 111, 112, 113, 114, 115, 470],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        action,
        np.array(
            [
                [100, 101, 102, 103, 104, 105, 360, 110, 111, 112, 113, 114, 115, 370],
                [200, 201, 202, 203, 204, 205, 460, 210, 211, 212, 213, 214, 215, 470],
            ],
            dtype=np.float32,
        ),
    )
    assert state.shape == action.shape == (2, 14)
    assert state.dtype == action.dtype == np.float32


@pytest.mark.parametrize(
    ("puppet", "master", "message"),
    [
        (np.zeros((1, 14), dtype=np.float32), np.zeros((1, 14), dtype=np.float32), "at least two"),
        (np.zeros((2, 13), dtype=np.float32), np.zeros((2, 14), dtype=np.float32), "shape"),
        (np.zeros((2, 14), dtype=np.float32), np.zeros((3, 14), dtype=np.float32), "same number"),
        (
            np.array([[np.nan] * 14, [0.0] * 14], dtype=np.float32),
            np.zeros((2, 14), dtype=np.float32),
            "finite",
        ),
        (
            np.zeros((2, 14), dtype=np.float32),
            np.array([[0.0] * 14, [np.inf] * 14], dtype=np.float32),
            "finite",
        ),
    ],
)
def test_build_shifted_state_action_rejects_invalid_inputs(
    puppet: np.ndarray, master: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        converter.build_shifted_state_action(puppet, master)


def test_populate_dataset_drops_last_source_frame_and_keeps_current_images(tmp_path: Path) -> None:
    path = tmp_path / "episode_000000.hdf5"
    puppet, master = _write_episode(path)
    dataset = _FakeDataset()

    converter.populate_dataset(dataset, [path])

    assert len(dataset.frames) == 2
    assert dataset.saved_episodes == 1
    first, second = dataset.frames
    expected_state, expected_action = converter.build_shifted_state_action(puppet, master)
    np.testing.assert_array_equal(first["observation.state"], expected_state[0])
    np.testing.assert_array_equal(first["action"], expected_action[0])
    np.testing.assert_array_equal(second["observation.state"], expected_state[1])
    np.testing.assert_array_equal(second["action"], expected_action[1])
    assert first["task"] == second["task"] == "Stack the cups"
    assert "timestamp" not in first
    assert int(first["observation.images.cam_high"][112, 112, 0]) == 0
    assert int(second["observation.images.cam_high"][112, 112, 0]) == 20


def test_populate_dataset_rejects_one_frame_episode(tmp_path: Path) -> None:
    path = tmp_path / "episode_000000.hdf5"
    _write_episode(path, frame_count=1)

    with pytest.raises(ValueError, match="at least two"):
        converter.populate_dataset(_FakeDataset(), [path])
