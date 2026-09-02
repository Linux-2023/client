"""Regression tests for the ROS2 piper_v1 HDF5 to LeRobot converter."""

from __future__ import annotations

import importlib.util
import sys
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "utils" / "convert_ros2_piper_data_to_lerobot.py"
spec = importlib.util.spec_from_file_location("convert_ros2_piper_data_to_lerobot", SCRIPT)
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
    schema: str = "piper_dual_ros2_v1",
    malformed_image: bool = False,
    image_dtype: np.dtype = np.dtype("uint8"),
    timestamps: tuple[float, float] = (10.0, 10.033),
) -> None:
    with h5py.File(path, "w") as episode:
        episode.attrs["schema_version"] = schema
        episode.attrs["frame_count"] = 2
        episode.attrs["metadata_json"] = json.dumps({"prompt": "Stack the cups"})
        episode.create_dataset("observations/state", data=np.zeros((2, 14), dtype=np.float32))
        episode.create_dataset("action", data=np.ones((2, 14), dtype=np.float32))
        episode.create_dataset("observations/timestamp", data=np.array(timestamps, dtype=np.float32))
        images = episode.create_group("observations/images")
        for camera in converter.IMAGE_SENSORS:
            dataset = images.create_dataset(camera, shape=(2,), dtype=h5py.vlen_dtype(image_dtype))
            values = [_jpeg(20), _jpeg(40)]
            if malformed_image and camera == "cam_left_wrist":
                values[1] = b"not-jpeg"
            for index, value in enumerate(values):
                dataset[index] = np.frombuffer(value, dtype=np.uint8)


def test_load_ros2_episode_reads_state_action_prompt_and_decodes_three_images(tmp_path: Path) -> None:
    path = tmp_path / "episode_000000.hdf5"
    _write_episode(path)

    episode = converter.load_ros2_episode(path)

    assert episode.prompt == "Stack the cups"
    assert episode.state.shape == (2, 14)
    assert episode.action.shape == (2, 14)
    assert set(episode.images) == set(converter.IMAGE_SENSORS)
    assert episode.images["cam_high"].shape == (2, 224, 224, 3)
    assert episode.images["cam_high"].dtype == np.uint8


def test_validate_ros2_episode_rejects_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "episode_000000.hdf5"
    _write_episode(path, schema="legacy_official_hdf5")

    with pytest.raises(ValueError, match="schema_version"):
        converter.validate_ros2_episode(path)


def test_load_ros2_episode_rejects_malformed_jpeg(tmp_path: Path) -> None:
    path = tmp_path / "episode_000000.hdf5"
    _write_episode(path, malformed_image=True)

    with pytest.raises(ValueError, match="cam_left_wrist.*JPEG"):
        converter.load_ros2_episode(path)


def test_validate_ros2_episode_rejects_nonmonotonic_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "episode_000000.hdf5"
    _write_episode(path, timestamps=(10.033, 10.0))

    with pytest.raises(ValueError, match="timestamps.*nondecreasing"):
        converter.validate_ros2_episode(path)


def test_validate_ros2_episode_rejects_non_uint8_image_storage(tmp_path: Path) -> None:
    path = tmp_path / "episode_000000.hdf5"
    _write_episode(path, image_dtype=np.dtype("uint16"))

    with pytest.raises(ValueError, match="vlen uint8"):
        converter.validate_ros2_episode(path)


def test_parse_episode_selection() -> None:
    assert converter.parse_episode_selection(None) is None
    assert converter.parse_episode_selection("0, 2,5") == [0, 2, 5]

    with pytest.raises(ValueError, match="episode index"):
        converter.parse_episode_selection("0,-1")


def test_existing_output_requires_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "repo"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        converter.ensure_output_available(output, overwrite=False)


def test_populate_dataset_rejects_negative_episode_indices(tmp_path: Path) -> None:
    path = tmp_path / "episode_000000.hdf5"
    _write_episode(path)

    with pytest.raises(IndexError, match="episode index"):
        converter.populate_dataset(_FakeDataset(), [path], episodes=[-1])


class _FakeDataset:
    fps = 30

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.saved_episodes = 0

    def add_frame(self, frame: dict) -> None:
        assert "timestamp" not in frame
        captured = dict(frame)
        captured["generated_timestamp"] = len(self.frames) / self.fps
        self.frames.append(captured)

    def save_episode(self) -> None:
        self.saved_episodes += 1


def test_populate_dataset_lets_lerobot_generate_episode_relative_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "episode_000000.hdf5"
    _write_episode(path)
    dataset = _FakeDataset()

    converter.populate_dataset(dataset, [path])

    assert [frame["generated_timestamp"] for frame in dataset.frames] == pytest.approx([0.0, 1.0 / 30.0])
    assert dataset.saved_episodes == 1
