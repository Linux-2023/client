#!/usr/bin/env python3
"""Convert shifted Piper dual ROS2 EEF episodes to LeRobot.

For source frame ``t``, observation state contains both current puppet EEF
poses and current master grippers. Action contains both puppet EEF poses from
frame ``t + 1`` and the same current master grippers. Each raw pose is converted
from ``[x, y, z, roll, pitch, yaw]`` to
``[x, y, z, r11, r21, r31, r12, r22, r32]``. The final source frame is dropped
because it has no next-frame EEF target.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Literal

import h5py
import numpy as np


_BASE_PATH = Path(__file__).with_name("convert_ros2_piper_data_to_lerobot.py")
_BASE_MODULE_NAME = "_piper_dual_ros2_lerobot_base"
_base_spec = importlib.util.spec_from_file_location(_BASE_MODULE_NAME, _BASE_PATH)
if _base_spec is None or _base_spec.loader is None:
    raise ImportError(f"Could not load base ROS2 converter from {_BASE_PATH}")
_base = importlib.util.module_from_spec(_base_spec)
sys.modules[_BASE_MODULE_NAME] = _base
_base_spec.loader.exec_module(_base)

LeRobotDataset = _base.LeRobotDataset
IMAGE_SENSORS = _base.IMAGE_SENSORS
IMAGE_HEIGHT = _base.IMAGE_HEIGHT
IMAGE_WIDTH = _base.IMAGE_WIDTH
LEROBOT_HOME = _base.LEROBOT_HOME
STATE_PATH = _base.STATE_PATH
EEF_DIMENSION = 6
EEF_ROT6D_DIMENSION = 9
FEATURE_DIMENSION = (EEF_ROT6D_DIMENSION + 1) * 2
EEF_LEFT_PATH = "/observations/eef/puppet_left"
EEF_RIGHT_PATH = "/observations/eef/puppet_right"
EEF_FEATURE_NAMES = [
    "eef_puppet_left_x",
    "eef_puppet_left_y",
    "eef_puppet_left_z",
    "eef_puppet_left_rot6d_0",
    "eef_puppet_left_rot6d_1",
    "eef_puppet_left_rot6d_2",
    "eef_puppet_left_rot6d_3",
    "eef_puppet_left_rot6d_4",
    "eef_puppet_left_rot6d_5",
    "left_gripper",
    "eef_puppet_right_x",
    "eef_puppet_right_y",
    "eef_puppet_right_z",
    "eef_puppet_right_rot6d_0",
    "eef_puppet_right_rot6d_1",
    "eef_puppet_right_rot6d_2",
    "eef_puppet_right_rot6d_3",
    "eef_puppet_right_rot6d_4",
    "eef_puppet_right_rot6d_5",
    "right_gripper",
]


@dataclass(frozen=True, slots=True)
class EefEpisode:
    """Decoded source episode including its two raw EEF pose streams."""

    base: object
    left: np.ndarray
    right: np.ndarray



def _coerce_eef_matrix(value: object, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric frame matrix") from exc
    if matrix.ndim != 2 or matrix.shape[1] != EEF_DIMENSION:
        raise ValueError(f"{name} must have shape (frames, {EEF_DIMENSION}), found {matrix.shape}")
    if matrix.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain numeric values")
    matrix = matrix.astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _coerce_master_action(value: object) -> np.ndarray:
    try:
        matrix = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("master_action must be a numeric frame matrix") from exc
    frame_dimension = _base.FRAME_DIMENSION
    if matrix.ndim != 2 or matrix.shape[1] != frame_dimension:
        raise ValueError(f"master_action must have shape (frames, {frame_dimension}), found {matrix.shape}")
    if matrix.dtype.kind not in "fiu":
        raise ValueError("master_action must contain numeric values")
    matrix = matrix.astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("master_action must contain only finite values")
    return matrix



def rpy_to_rot6d(rpy: object) -> np.ndarray:
    """Convert ROS XYZ RPY to the first two columns of ``Rz(yaw)Ry(pitch)Rx(roll)``."""
    values = np.asarray(rpy)
    if values.shape != (3,):
        raise ValueError(f"rpy must have shape (3,), found {values.shape}")
    if values.dtype.kind not in "fiu":
        raise ValueError("rpy must contain numeric values")
    roll, pitch, yaw = (float(value) for value in values)
    if not all(math.isfinite(value) for value in (roll, pitch, yaw)):
        raise ValueError("rpy must contain only finite values")

    sin_roll, cos_roll = math.sin(roll), math.cos(roll)
    sin_pitch, cos_pitch = math.sin(pitch), math.cos(pitch)
    sin_yaw, cos_yaw = math.sin(yaw), math.cos(yaw)

    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll); flatten columns, not rows.
    rotation = np.array(
        [
            cos_yaw * cos_pitch,
            sin_yaw * cos_pitch,
            -sin_pitch,
            cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll,
            sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll,
            cos_pitch * sin_roll,
        ],
        dtype=np.float32,
    )
    return rotation



def eef_rpy_to_rot9(pose: object) -> np.ndarray:
    """Convert one ``[x,y,z,roll,pitch,yaw]`` pose to a 9D position+rot6d pose."""
    values = np.asarray(pose)
    if values.shape != (EEF_DIMENSION,):
        raise ValueError(f"EEF pose must have shape ({EEF_DIMENSION},), found {values.shape}")
    if values.dtype.kind not in "fiu":
        raise ValueError("EEF pose must contain numeric values")
    values = values.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("EEF pose must contain only finite values")
    return np.concatenate((values[:3], rpy_to_rot6d(values[3:])), dtype=np.float32)



def _eef_matrix_to_rot9(matrix: np.ndarray, name: str) -> np.ndarray:
    values = _coerce_eef_matrix(matrix, name)
    result = np.empty((values.shape[0], EEF_ROT6D_DIMENSION), dtype=np.float32)
    result[:, :3] = values[:, :3]
    roll = values[:, 3]
    pitch = values[:, 4]
    yaw = values[:, 5]
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    result[:, 3] = cy * cp
    result[:, 4] = sy * cp
    result[:, 5] = -sp
    result[:, 6] = cy * sp * sr - sy * cr
    result[:, 7] = sy * sp * sr + cy * cr
    result[:, 8] = cp * sr
    return result



def build_shifted_eef_state_action(
    puppet_left_eef: object,
    puppet_right_eef: object,
    master_action: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``N - 1`` 20D EEF-plus-gripper vectors with one-step targets."""
    left = _coerce_eef_matrix(puppet_left_eef, "puppet_left_eef")
    right = _coerce_eef_matrix(puppet_right_eef, "puppet_right_eef")
    master = _coerce_master_action(master_action)
    if left.shape[0] != right.shape[0] or left.shape[0] != master.shape[0]:
        raise ValueError("puppet EEF streams and master_action must have the same number of frames")
    if left.shape[0] < 2:
        raise ValueError("shifted EEF conversion requires at least two source frames per episode")

    left_rot9 = _eef_matrix_to_rot9(left, "puppet_left_eef")
    right_rot9 = _eef_matrix_to_rot9(right, "puppet_right_eef")
    output_frames = left.shape[0] - 1
    state = np.empty((output_frames, FEATURE_DIMENSION), dtype=np.float32)
    action = np.empty((output_frames, FEATURE_DIMENSION), dtype=np.float32)

    state[:, :9] = left_rot9[:-1]
    state[:, 9] = master[:-1, 6]
    state[:, 10:19] = right_rot9[:-1]
    state[:, 19] = master[:-1, 13]

    action[:, :9] = left_rot9[1:]
    action[:, 9] = master[:-1, 6]
    action[:, 10:19] = right_rot9[1:]
    action[:, 19] = master[:-1, 13]
    return state, action



def _metadata_eef_enabled(episode: h5py.File, path: Path) -> None:
    raw_metadata = episode.attrs.get("metadata_json")
    if isinstance(raw_metadata, bytes):
        raw_metadata = raw_metadata.decode("utf-8")
    if isinstance(raw_metadata, np.bytes_):
        raw_metadata = raw_metadata.tobytes().decode("utf-8")
    try:
        metadata = json.loads(raw_metadata)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: metadata_json is not valid JSON") from exc
    eef = metadata.get("eef") if isinstance(metadata, dict) else None
    if not isinstance(eef, dict) or eef.get("enabled") is not True:
        raise ValueError(f"{path}: metadata_json.eef.enabled must be true for EEF conversion")



def load_eef_episode(path: Path) -> EefEpisode:
    """Validate and load one EEF-enabled raw ROS2 episode."""
    path = Path(path)
    _base.validate_ros2_episode(path, decode_images=False)
    with h5py.File(path, "r") as episode:
        _metadata_eef_enabled(episode, path)
        frame_count = int(episode[STATE_PATH].shape[0])
        arrays = {}
        for name, dataset_path in (("left", EEF_LEFT_PATH), ("right", EEF_RIGHT_PATH)):
            if dataset_path not in episode or not isinstance(episode[dataset_path], h5py.Dataset):
                raise ValueError(f"{path}: missing required dataset {dataset_path}")
            arrays[name] = _coerce_eef_matrix(episode[dataset_path][:], dataset_path)
            if arrays[name].shape != (frame_count, EEF_DIMENSION):
                raise ValueError(
                    f"{path}: {dataset_path} must have shape ({frame_count}, {EEF_DIMENSION}), found {arrays[name].shape}"
                )
    return EefEpisode(base=_base.load_ros2_episode(path), left=arrays["left"], right=arrays["right"])



def _selected_indices(file_count: int, episodes: list[int] | None) -> list[int]:
    selected = list(range(file_count)) if episodes is None else list(episodes)
    invalid = [index for index in selected if index < 0 or index >= file_count]
    if invalid:
        raise IndexError(f"episode index out of range: {invalid[0]} (found {file_count} files)")
    return selected



def populate_dataset(
    dataset: object,
    hdf5_files: list[Path],
    episodes: list[int] | None = None,
) -> object:
    """Populate LeRobot with shifted 20D EEF-plus-current-gripper episodes."""
    selected = _selected_indices(len(hdf5_files), episodes)
    for source_index in selected:
        episode = load_eef_episode(hdf5_files[source_index])
        states, actions = build_shifted_eef_state_action(episode.left, episode.right, episode.base.action)
        for frame_index in range(states.shape[0]):
            frame = {
                "observation.state": states[frame_index],
                "action": actions[frame_index],
                "task": episode.base.prompt,
            }
            for camera in IMAGE_SENSORS:
                frame[f"observation.images.{camera}"] = episode.base.images[camera][frame_index]
            dataset.add_frame(frame)
        dataset.save_episode()
    return dataset



def _validate_shiftable_eef_episode(path: Path) -> None:
    episode = load_eef_episode(path)
    if episode.left.shape[0] < 2:
        raise ValueError(f"{path}: shifted EEF conversion requires at least two source frames per episode")



def create_empty_dataset(
    repo_id: str,
    output_dir: Path,
    *,
    mode: Literal["image", "video"] = "image",
    fps: int = 30,
) -> object:
    """Create a LeRobot dataset with dual-arm 9D EEF plus gripper vectors."""
    if mode not in ("image", "video"):
        raise ValueError(f"mode must be 'image' or 'video', found {mode!r}")
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (FEATURE_DIMENSION,),
            "names": [EEF_FEATURE_NAMES],
        },
        "action": {
            "dtype": "float32",
            "shape": (FEATURE_DIMENSION,),
            "names": [EEF_FEATURE_NAMES],
        },
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
        robot_type="piper_dual_ros2_eef",
        features=features,
        use_videos=mode == "video",
        tolerance_s=0.005,
    )



def convert_ros2_dataset(
    raw_dir: Path,
    repo_id: str,
    *,
    mode: Literal["image", "video"] = "image",
    episodes: list[int] | None = None,
    overwrite: bool = False,
) -> Path:
    """Validate selected EEF sources and convert them into local LeRobot data."""
    files = _base.discover_hdf5_files(raw_dir)
    selected = _selected_indices(len(files), episodes)
    for index in selected:
        _validate_shiftable_eef_episode(files[index])

    output_dir = LEROBOT_HOME / repo_id
    _base.ensure_output_available(output_dir, overwrite=overwrite)
    try:
        dataset = create_empty_dataset(repo_id, output_dir, mode=mode)
        populate_dataset(dataset, files, episodes=selected)
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return output_dir



def parse_episode_selection(value: str | None) -> list[int] | None:
    return _base.parse_episode_selection(value)



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory containing EEF ROS2 episode_*.hdf5 files")
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
    print(f"Converted shifted EEF ROS2 dataset to {output}")


if __name__ == "__main__":
    main()
