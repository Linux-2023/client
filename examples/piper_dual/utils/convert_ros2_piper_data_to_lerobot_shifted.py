#!/usr/bin/env python3
"""Convert Piper dual ROS 2 episodes with one-step puppet-joint actions.

For source frame ``t``, observation state uses the current puppet joint angles,
action uses puppet joint angles from frame ``t + 1``, and both vectors use the
current master grippers. The final source frame is dropped because it has no
next-frame joint target.
"""

from __future__ import annotations

import argparse
import importlib.util
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

FRAME_DIMENSION = _base.FRAME_DIMENSION
IMAGE_SENSORS = _base.IMAGE_SENSORS
LEROBOT_HOME = _base.LEROBOT_HOME
STATE_PATH = _base.STATE_PATH
MOTOR_NAMES = _base.MOTOR_NAMES


def _coerce_frame_matrix(value: np.ndarray, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric frame matrix") from exc
    if matrix.ndim != 2 or matrix.shape[1] != FRAME_DIMENSION:
        raise ValueError(f"{name} must have shape (frames, {FRAME_DIMENSION}), found {matrix.shape}")
    if matrix.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain numeric values")
    matrix = matrix.astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def build_shifted_state_action(
    puppet_state: np.ndarray,
    master_action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``N - 1`` state/action vectors under the shifted-joint contract."""
    puppet = _coerce_frame_matrix(puppet_state, "puppet_state")
    master = _coerce_frame_matrix(master_action, "master_action")
    if puppet.shape[0] != master.shape[0]:
        raise ValueError("puppet_state and master_action must have the same number of frames")
    if puppet.shape[0] < 2:
        raise ValueError("shifted conversion requires at least two source frames per episode")

    output_frames = puppet.shape[0] - 1
    state = np.empty((output_frames, FRAME_DIMENSION), dtype=np.float32)
    action = np.empty((output_frames, FRAME_DIMENSION), dtype=np.float32)

    state[:, :6] = puppet[:-1, :6]
    state[:, 6] = master[:-1, 6]
    state[:, 7:13] = puppet[:-1, 7:13]
    state[:, 13] = master[:-1, 13]

    action[:, :6] = puppet[1:, :6]
    action[:, 6] = master[:-1, 6]
    action[:, 7:13] = puppet[1:, 7:13]
    action[:, 13] = master[:-1, 13]
    return state, action


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
    """Populate a LeRobot dataset with shifted state/action episodes."""
    selected = _selected_indices(len(hdf5_files), episodes)
    for source_index in selected:
        episode = _base.load_ros2_episode(hdf5_files[source_index])
        states, actions = build_shifted_state_action(episode.state, episode.action)
        for frame_index in range(states.shape[0]):
            frame = {
                "observation.state": states[frame_index],
                "action": actions[frame_index],
                "task": episode.prompt,
            }
            for camera in IMAGE_SENSORS:
                frame[f"observation.images.{camera}"] = episode.images[camera][frame_index]
            dataset.add_frame(frame)
        dataset.save_episode()
    return dataset


def _validate_shiftable_episode(path: Path) -> None:
    _base.validate_ros2_episode(path, decode_images=False)
    with h5py.File(path, "r") as episode:
        frame_count = int(episode[STATE_PATH].shape[0])
    if frame_count < 2:
        raise ValueError(f"{path}: shifted conversion requires at least two source frames per episode")


def convert_ros2_dataset(
    raw_dir: Path,
    repo_id: str,
    *,
    mode: Literal["image", "video"] = "image",
    episodes: list[int] | None = None,
    overwrite: bool = False,
) -> Path:
    """Validate selected sources and convert them into a local LeRobot dataset."""
    files = _base.discover_hdf5_files(raw_dir)
    selected = _selected_indices(len(files), episodes)
    for index in selected:
        _validate_shiftable_episode(files[index])

    output_dir = LEROBOT_HOME / repo_id
    _base.ensure_output_available(output_dir, overwrite=overwrite)
    try:
        dataset = _base.create_empty_dataset(repo_id, output_dir, mode=mode)
        populate_dataset(dataset, files, episodes=selected)
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return output_dir


def parse_episode_selection(value: str | None) -> list[int] | None:
    return _base.parse_episode_selection(value)


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
    print(f"Converted shifted ROS2 dataset to {output}")


if __name__ == "__main__":
    main()
