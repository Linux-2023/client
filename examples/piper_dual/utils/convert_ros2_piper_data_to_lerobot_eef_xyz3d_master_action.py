#!/usr/bin/env python3
"""Convert Piper dual ROS2 EEF episodes to LeRobot with master-joint actions.

For source frame ``t``, observation state contains both current puppet EEF
poses (raw ``[x, y, z, roll, pitch, yaw]``) and the current master grippers.
Unlike ``convert_ros2_piper_data_to_lerobot_eef_xyz3d_shifted.py``, the action is
the **current master joint vector** (both arms' joint angles plus the master's
own grippers, i.e. the raw ``/action`` dataset of frame ``t``), not the puppet
EEF pose of frame ``t + 1``.

Consequences:
  * state and action are all taken from the same source frame ``t``;
  * no frame is dropped, so an episode with ``N`` source frames yields ``N``
    frames (the shifted EEF converter yields ``N - 1``).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Literal

import h5py
import numpy as np


_XYZ3D_PATH = Path(__file__).with_name("convert_ros2_piper_data_to_lerobot_eef_xyz3d_shifted.py")
_XYZ3D_MODULE_NAME = "_piper_dual_ros2_lerobot_eef_xyz3d"
_xyz3d_spec = importlib.util.spec_from_file_location(_XYZ3D_MODULE_NAME, _XYZ3D_PATH)
if _xyz3d_spec is None or _xyz3d_spec.loader is None:
    raise ImportError(f"Could not load XYZ+RPY EEF converter from {_XYZ3D_PATH}")
_xyz3d = importlib.util.module_from_spec(_xyz3d_spec)
sys.modules[_XYZ3D_MODULE_NAME] = _xyz3d
_xyz3d_spec.loader.exec_module(_xyz3d)

_base = _xyz3d._base
LeRobotDataset = _base.LeRobotDataset
IMAGE_SENSORS = _base.IMAGE_SENSORS
IMAGE_HEIGHT = _base.IMAGE_HEIGHT
IMAGE_WIDTH = _base.IMAGE_WIDTH
LEROBOT_HOME = _base.LEROBOT_HOME
MOTOR_NAMES = _base.MOTOR_NAMES
FRAME_DIMENSION = _base.FRAME_DIMENSION
load_eef_episode = _xyz3d.load_eef_episode
EefEpisode = _xyz3d.EefEpisode
EEF_POSE_DIMENSION = _xyz3d.EEF_POSE_DIMENSION
FEATURE_DIMENSION = _xyz3d.FEATURE_DIMENSION
EEF_FEATURE_NAMES = _xyz3d.EEF_FEATURE_NAMES
EEF_LEFT_PATH = _xyz3d.EEF_LEFT_PATH
EEF_RIGHT_PATH = _xyz3d.EEF_RIGHT_PATH
SCHEMA_VERSION = _base.SCHEMA_VERSION
ROBOT_TYPE = "piper_dual_ros2_eef_master_action"


@dataclass(frozen=True, slots=True)
class FrameArrays:
    """Aligned per-frame state/action matrices for one episode."""

    state: np.ndarray
    action: np.ndarray


def build_eef_state_master_action(
    puppet_left_eef: object,
    puppet_right_eef: object,
    master_action: object,
) -> FrameArrays:
    """Build ``N`` frames of EEF state plus master-joint action.

    ``state``  is ``[puppet_left_eef(t), left_gripper(t), puppet_right_eef(t), right_gripper(t)]``
    ``action`` is the master joint vector of frame ``t`` (both grippers included).
    """
    left = _xyz3d._coerce_eef_matrix(puppet_left_eef, "puppet_left_eef")
    right = _xyz3d._coerce_eef_matrix(puppet_right_eef, "puppet_right_eef")
    master = _xyz3d._coerce_master_action(master_action)
    if left.shape[0] != right.shape[0] or left.shape[0] != master.shape[0]:
        raise ValueError("puppet EEF streams and master_action must have the same number of frames")
    if left.shape[0] < 1:
        raise ValueError("master-action EEF conversion requires at least one source frame per episode")

    output_frames = left.shape[0]
    state = np.empty((output_frames, FEATURE_DIMENSION), dtype=np.float32)
    action = np.empty((output_frames, FRAME_DIMENSION), dtype=np.float32)

    state[:, :EEF_POSE_DIMENSION] = left
    state[:, EEF_POSE_DIMENSION] = master[:, 6]
    state[:, EEF_POSE_DIMENSION + 1 : 2 * EEF_POSE_DIMENSION + 1] = right
    state[:, -1] = master[:, 13]

    action[:, :] = master
    return FrameArrays(state=state, action=action)


def _decode_attr(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_attr(value.item())
    return value


def validate_eef_episode_metadata(path: Path) -> int:
    """Cheap structural pre-check: reads only attributes and dataset shapes.

    No image is decoded, so this is milliseconds per episode regardless of
    episode length. Returns the source frame count.
    """
    with h5py.File(path, "r") as episode:
        schema = _decode_attr(episode.attrs.get("schema_version"))
        if schema != SCHEMA_VERSION:
            raise ValueError(f"{path}: schema_version must be {SCHEMA_VERSION!r}, found {schema!r}")

        raw_metadata = _decode_attr(episode.attrs.get("metadata_json"))
        if not isinstance(raw_metadata, str):
            raise ValueError(f"{path}: metadata_json attribute is missing")
        try:
            metadata = json.loads(raw_metadata)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: metadata_json is not valid JSON") from exc
        prompt = metadata.get("prompt") if isinstance(metadata, dict) else None
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{path}: metadata_json.prompt must be a non-empty string")
        eef = metadata.get("eef") if isinstance(metadata, dict) else None
        if not isinstance(eef, dict) or eef.get("enabled") is not True:
            raise ValueError(f"{path}: metadata_json.eef.enabled must be true for EEF conversion")

        frame_count = int(episode.attrs.get("frame_count", -1))
        if frame_count < 1:
            raise ValueError(f"{path}: frame_count must be positive, found {frame_count}")

        for dataset_path in (_base.STATE_PATH, _base.ACTION_PATH):
            if dataset_path not in episode:
                raise ValueError(f"{path}: missing required dataset {dataset_path}")
            shape = episode[dataset_path].shape
            if shape != (frame_count, FRAME_DIMENSION):
                raise ValueError(
                    f"{path}: {dataset_path} must have shape ({frame_count}, {FRAME_DIMENSION}), found {shape}"
                )
        for dataset_path in (EEF_LEFT_PATH, EEF_RIGHT_PATH):
            if dataset_path not in episode:
                raise ValueError(f"{path}: missing required dataset {dataset_path}")
            shape = episode[dataset_path].shape
            if shape != (frame_count, EEF_POSE_DIMENSION):
                raise ValueError(
                    f"{path}: {dataset_path} must have shape ({frame_count}, {EEF_POSE_DIMENSION}), found {shape}"
                )
        for camera in IMAGE_SENSORS:
            dataset_path = _base.IMAGE_PATH_TEMPLATE.format(camera=camera)
            if dataset_path not in episode:
                raise ValueError(f"{path}: missing required dataset {dataset_path}")
            if len(episode[dataset_path]) != frame_count:
                raise ValueError(
                    f"{path}: {dataset_path} must contain {frame_count} frames, found {len(episode[dataset_path])}"
                )
    return frame_count


def populate_dataset(
    dataset: object,
    hdf5_files: list[Path],
    episodes: list[int] | None = None,
) -> object:
    """Populate LeRobot with EEF states and same-frame master-joint actions."""
    selected = _xyz3d._selected_indices(len(hdf5_files), episodes)
    for position, source_index in enumerate(selected, start=1):
        path = hdf5_files[source_index]
        episode = load_eef_episode(path)
        arrays = build_eef_state_master_action(episode.left, episode.right, episode.base.action)
        print(f"[{position}/{len(selected)}] {path.name}: {arrays.state.shape[0]} frames", flush=True)
        for frame_index in range(arrays.state.shape[0]):
            frame = {
                "observation.state": arrays.state[frame_index],
                "action": arrays.action[frame_index],
                "task": episode.base.prompt,
            }
            for camera in IMAGE_SENSORS:
                frame[f"observation.images.{camera}"] = episode.base.images[camera][frame_index]
            dataset.add_frame(frame)
        dataset.save_episode()
    return dataset


def create_empty_dataset(
    repo_id: str,
    output_dir: Path,
    *,
    mode: Literal["image", "video"] = "image",
    fps: int = 30,
) -> object:
    """Create a LeRobot dataset with EEF state and master-joint action features."""
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
            "shape": (FRAME_DIMENSION,),
            "names": [list(MOTOR_NAMES)],
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
        robot_type=ROBOT_TYPE,
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
    deep_validate: bool = False,
) -> Path:
    """Validate selected EEF sources and convert them into local LeRobot data.

    The pre-check reads only attributes and dataset shapes (milliseconds per
    episode). Pass ``deep_validate=True`` to instead decode every episode fully
    before writing anything (slow: it is a complete extra pass over all images).
    """
    files = _base.discover_hdf5_files(raw_dir)
    selected = _xyz3d._selected_indices(len(files), episodes)
    frame_total = 0
    for index in selected:
        path = files[index]
        if deep_validate:
            frame_total += load_eef_episode(path).left.shape[0]
        else:
            frame_total += validate_eef_episode_metadata(path)
    print(
        f"pre-check ok: {len(selected)} episodes / {frame_total} frames "
        f"(mode={'deep-decode' if deep_validate else 'metadata-only'})",
        flush=True,
    )

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
    parser.add_argument(
        "--deep-validate",
        action="store_true",
        help="Pre-check each episode by fully decoding its images (slow); default is a metadata-only structural check",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = convert_ros2_dataset(
        args.raw_dir,
        args.repo_id,
        mode=args.mode,
        episodes=parse_episode_selection(args.episodes),
        overwrite=args.overwrite,
        deep_validate=args.deep_validate,
    )
    print(f"Converted EEF-state / master-joint-action ROS2 dataset to {output}")


if __name__ == "__main__":
    main()
