#!/usr/bin/env python3
"""Convert Piper dual ROS 2 v1 HDF5 episodes to LeRobot format.

The default ``rebuild`` timestamp mode uses LeRobot's exact ``frame_index / fps``
clock. ROS 2 collector timestamps are absolute float32 values and may not satisfy
LeRobot's fixed-rate timestamp contract after serialization. ``source`` mode is
available only when stored timestamps are monotonic and match the fixed-rate
tolerance.

This converter targets the self-contained schema emitted by ``collect_data_ros2.py``.
The legacy converter remains separate because its HDF5 schema uses
``/observations/qpos`` and a different task/image representation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
from typing import Literal

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np


if os.getenv("HF_LEROBOT_HOME") is None:
    os.environ["HF_LEROBOT_HOME"] = str(Path.home() / ".cache" / "huggingface" / "lerobot")
LEROBOT_HOME = Path(os.environ["HF_LEROBOT_HOME"])

SCHEMA_VERSION = "piper_dual_ros2_v1"
FRAME_DIMENSION = 14
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_SENSORS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
STATE_PATH = "/observations/state"
ACTION_PATH = "/action"
TIMESTAMP_PATH = "/observations/timestamp"
IMAGE_PATH_TEMPLATE = "/observations/images/{camera}"
EPISODE_PATTERN = re.compile(r"^episode_(\d+)\.hdf5$")
TIMESTAMP_TOLERANCE_S = 0.005
TimestampMode = Literal["rebuild", "source"]
MOTOR_NAMES = (
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
)


@dataclass(frozen=True, slots=True)
class Ros2Episode:
    """Decoded data from one validated ROS 2 HDF5 episode."""

    path: Path
    prompt: str
    state: np.ndarray
    action: np.ndarray
    timestamps: np.ndarray
    images: dict[str, np.ndarray]


def _decode_scalar(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_scalar(value.item())
    return value


def _metadata_prompt(episode: h5py.File, path: Path) -> str:
    raw_metadata = _decode_scalar(episode.attrs.get("metadata_json"))
    if not isinstance(raw_metadata, str):
        raise ValueError(f"{path}: metadata_json attribute is missing")
    try:
        metadata = json.loads(raw_metadata)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: metadata_json is not valid JSON") from exc
    prompt = metadata.get("prompt") if isinstance(metadata, dict) else None
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{path}: metadata_json.prompt must be a non-empty string")
    return prompt


def _jpeg_bytes(value: object, path: Path, camera: str, index: int) -> bytes:
    try:
        encoded = np.asarray(value, dtype=np.uint8)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {camera}[{index}] is not a uint8 JPEG payload") from exc
    if encoded.ndim != 1 or encoded.size == 0:
        raise ValueError(f"{path}: {camera}[{index}] is an empty or non-1D JPEG payload")
    return encoded.tobytes()


def _decode_jpeg(encoded: bytes, path: Path, camera: str, index: int) -> np.ndarray:
    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError(f"{path}: {camera}[{index}] is not decodable JPEG bytes")
    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    source_height, source_width = rgb.shape[:2]
    scale = min(IMAGE_WIDTH / source_width, IMAGE_HEIGHT / source_height)
    resized_width = max(1, min(IMAGE_WIDTH, int(source_width * scale + 0.5)))
    resized_height = max(1, min(IMAGE_HEIGHT, int(source_height * scale + 0.5)))
    interpolation = cv2.INTER_AREA if resized_width < source_width or resized_height < source_height else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=interpolation)
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    top = (IMAGE_HEIGHT - resized_height) // 2
    left = (IMAGE_WIDTH - resized_width) // 2
    image[top : top + resized_height, left : left + resized_width] = resized
    return image


def _require_dataset(episode: h5py.File, dataset_path: str, path: Path) -> h5py.Dataset:
    if dataset_path not in episode or not isinstance(episode[dataset_path], h5py.Dataset):
        raise ValueError(f"{path}: missing required dataset {dataset_path}")
    return episode[dataset_path]


def validate_ros2_episode(path: Path, *, decode_images: bool = True) -> None:
    """Validate the ROS 2 v1 contract, optionally decoding every JPEG."""
    path = Path(path)
    try:
        with h5py.File(path, "r") as episode:
            schema = _decode_scalar(episode.attrs.get("schema_version"))
            if schema != SCHEMA_VERSION:
                raise ValueError(f"{path}: schema_version mismatch; expected {SCHEMA_VERSION!r}, found {schema!r}")

            state = _require_dataset(episode, STATE_PATH, path)
            action = _require_dataset(episode, ACTION_PATH, path)
            timestamps = _require_dataset(episode, TIMESTAMP_PATH, path)
            if state.shape != (state.shape[0], FRAME_DIMENSION):
                raise ValueError(f"{path}: {STATE_PATH} must have shape (frames, {FRAME_DIMENSION}), found {state.shape}")
            frame_count = state.shape[0]
            expected_vector = (frame_count, FRAME_DIMENSION)
            if action.shape != expected_vector:
                raise ValueError(f"{path}: {ACTION_PATH} must have shape {expected_vector}, found {action.shape}")
            for dataset_path, dataset in ((STATE_PATH, state), (ACTION_PATH, action), (TIMESTAMP_PATH, timestamps)):
                if dataset.dtype != np.dtype("float32"):
                    raise ValueError(f"{path}: {dataset_path} must have dtype float32, found {dataset.dtype}")
                values = dataset[:]
                if not np.isfinite(values).all():
                    raise ValueError(f"{path}: {dataset_path} contains non-finite values")
                if dataset_path == TIMESTAMP_PATH and (frame_count > 1 and not np.all(np.diff(values) >= 0.0)):
                    raise ValueError(f"{path}: timestamps must be nondecreasing")
            if frame_count == 0:
                raise ValueError(f"{path}: episode contains zero frames")
            declared_count = _decode_scalar(episode.attrs.get("frame_count"))
            if declared_count is not None and int(declared_count) != frame_count:
                raise ValueError(f"{path}: frame_count attribute {declared_count} does not match datasets ({frame_count})")

            _metadata_prompt(episode, path)
            for camera in IMAGE_SENSORS:
                dataset = _require_dataset(episode, IMAGE_PATH_TEMPLATE.format(camera=camera), path)
                if h5py.check_dtype(vlen=dataset.dtype) != np.dtype("uint8"):
                    raise ValueError(f"{path}: {camera} image dataset must use vlen uint8 storage")
                if dataset.shape != (frame_count,):
                    raise ValueError(f"{path}: {camera} image count does not match frame_count ({dataset.shape})")
                if decode_images:
                    for index, value in enumerate(dataset):
                        _decode_jpeg(_jpeg_bytes(value, path, camera, index), path, camera, index)
    except OSError as exc:
        raise ValueError(f"{path}: could not open HDF5 file: {exc}") from exc


def load_ros2_episode(path: Path) -> Ros2Episode:
    """Validate and fully decode one ROS 2 HDF5 episode."""
    path = Path(path)
    validate_ros2_episode(path, decode_images=False)
    with h5py.File(path, "r") as episode:
        prompt = _metadata_prompt(episode, path)
        state = np.asarray(episode[STATE_PATH][:], dtype=np.float32)
        action = np.asarray(episode[ACTION_PATH][:], dtype=np.float32)
        timestamps = np.asarray(episode[TIMESTAMP_PATH][:], dtype=np.float32)
        images = {
            camera: np.stack(
                [_decode_jpeg(_jpeg_bytes(value, path, camera, index), path, camera, index) for index, value in enumerate(episode[IMAGE_PATH_TEMPLATE.format(camera=camera)])]
            )
            for camera in IMAGE_SENSORS
        }
    return Ros2Episode(path=path, prompt=prompt, state=state, action=action, timestamps=timestamps, images=images)


def parse_episode_selection(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    try:
        selected = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError("episode selection must be comma-separated non-negative integers") from exc
    if any(index < 0 for index in selected) or len(selected) != len(set(selected)):
        raise ValueError("episode index values must be unique and non-negative")
    return selected


def ensure_output_available(output: Path, *, overwrite: bool) -> None:
    output = Path(output)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"LeRobot output already exists: {output}; pass --overwrite to replace it")
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()


def discover_hdf5_files(raw_dir: Path) -> list[Path]:
    files = sorted(Path(raw_dir).glob("episode_*.hdf5"))
    files = [path for path in files if EPISODE_PATTERN.match(path.name)]
    if not files:
        raise ValueError(f"No ROS2 episode_*.hdf5 files found in {raw_dir}")
    return files


def create_empty_dataset(
    repo_id: str,
    output_dir: Path,
    *,
    mode: Literal["image", "video"] = "image",
    fps: int = 30,
) -> LeRobotDataset:
    if mode not in ("image", "video"):
        raise ValueError(f"mode must be 'image' or 'video', found {mode!r}")
    features = {
        "observation.state": {"dtype": "float32", "shape": (FRAME_DIMENSION,), "names": [list(MOTOR_NAMES)]},
        "action": {"dtype": "float32", "shape": (FRAME_DIMENSION,), "names": [list(MOTOR_NAMES)]},
    }
    for camera in IMAGE_SENSORS:
        features[f"observation.images.{camera}"] = {
            "dtype": mode,
            "shape": (3, IMAGE_HEIGHT, IMAGE_WIDTH),
            "names": ["channels", "height", "width"],
        }
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        fps=fps,
        robot_type="piper_dual_ros2",
        features=features,
        use_videos=mode == "video",
        tolerance_s=0.005,
    )


def _source_timestamps(episode: Ros2Episode, fps: int) -> np.ndarray:
    relative = episode.timestamps - episode.timestamps[0]
    if relative.size > 1:
        deltas = np.diff(relative)
        expected = 1.0 / fps
        if not np.all(np.abs(deltas - expected) <= TIMESTAMP_TOLERANCE_S):
            raise ValueError(
                f"{episode.path}: source timestamps do not match {fps} Hz within "
                f"{TIMESTAMP_TOLERANCE_S}s; use timestamp_mode='rebuild'"
            )
    return relative.astype(np.float32, copy=False)


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    episodes: list[int] | None = None,
    *,
    timestamp_mode: TimestampMode = "rebuild",
) -> LeRobotDataset:
    if timestamp_mode not in ("rebuild", "source"):
        raise ValueError(f"timestamp_mode must be 'rebuild' or 'source', found {timestamp_mode!r}")
    selected = list(range(len(hdf5_files))) if episodes is None else list(episodes)
    invalid = [index for index in selected if index < 0 or index >= len(hdf5_files)]
    if invalid:
        raise IndexError(f"episode index out of range: {invalid[0]} (found {len(hdf5_files)} files)")
    for source_index in selected:
        episode = load_ros2_episode(hdf5_files[source_index])
        source_timestamps = _source_timestamps(episode, dataset.fps) if timestamp_mode == "source" else None
        for frame_index in range(episode.state.shape[0]):
            frame = {
                "observation.state": episode.state[frame_index],
                "action": episode.action[frame_index],
                "task": episode.prompt,
            }
            if source_timestamps is not None:
                frame["timestamp"] = np.asarray([source_timestamps[frame_index]], dtype=np.float32)
            for camera in IMAGE_SENSORS:
                frame[f"observation.images.{camera}"] = episode.images[camera][frame_index]
            dataset.add_frame(frame)
        dataset.save_episode()
    return dataset

def convert_ros2_dataset(
    raw_dir: Path,
    repo_id: str,
    *,
    mode: Literal["image", "video"] = "image",
    episodes: list[int] | None = None,
    overwrite: bool = False,
    timestamp_mode: TimestampMode = "rebuild",
) -> Path:
    files = discover_hdf5_files(raw_dir)
    selected = list(range(len(files))) if episodes is None else list(episodes)
    invalid = [index for index in selected if index < 0 or index >= len(files)]
    if invalid:
        raise IndexError(f"episode index out of range: {invalid[0]} (found {len(files)} files)")
    for index in selected:
        validate_ros2_episode(files[index], decode_images=False)
    output_dir = LEROBOT_HOME / repo_id
    ensure_output_available(output_dir, overwrite=overwrite)
    try:
        dataset = create_empty_dataset(repo_id, output_dir, mode=mode)
        populate_dataset(dataset, files, episodes=selected, timestamp_mode=timestamp_mode)
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return output_dir

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory containing ROS2 episode_*.hdf5 files")
    parser.add_argument("--repo-id", required=True, help="LeRobot repository identifier")
    parser.add_argument("--mode", choices=("image", "video"), default="image")
    parser.add_argument("--episodes", help="Comma-separated source episode indices; default: all")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing local LeRobot output")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = convert_ros2_dataset(
        args.raw_dir,
        args.repo_id,
        mode=args.mode,
        episodes=parse_episode_selection(args.episodes),
        overwrite=args.overwrite,
    )
    print(f"Converted ROS2 dataset to {output}")


if __name__ == "__main__":
    main()
