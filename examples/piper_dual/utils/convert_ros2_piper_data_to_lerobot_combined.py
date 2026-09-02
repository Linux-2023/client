#!/usr/bin/env python3
"""Combine ordinary and EEF Piper dual ROS 2 episodes into one LeRobot dataset.

Ordinary episodes are emitted first, followed by EEF-marked episodes. EEF pose
fields are ignored. Frame state/action values use the one-step shifted puppet
joint mapping implemented by ``convert_ros2_piper_data_to_lerobot_shifted.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil
import sys
from typing import Literal


_SHIFTED_PATH = Path(__file__).with_name("convert_ros2_piper_data_to_lerobot_shifted.py")
_SHIFTED_MODULE_NAME = "_piper_dual_ros2_lerobot_shifted"
_shifted_spec = importlib.util.spec_from_file_location(_SHIFTED_MODULE_NAME, _SHIFTED_PATH)
if _shifted_spec is None or _shifted_spec.loader is None:
    raise ImportError(f"Could not load shifted ROS2 converter from {_SHIFTED_PATH}")
_shifted = importlib.util.module_from_spec(_shifted_spec)
sys.modules[_SHIFTED_MODULE_NAME] = _shifted
_shifted_spec.loader.exec_module(_shifted)

LEROBOT_HOME = _shifted.LEROBOT_HOME


def _discover_directory(raw_dir: Path, label: str) -> list[Path]:
    directory = Path(raw_dir)
    if not directory.is_dir():
        raise ValueError(f"{label} directory does not exist: {directory}")
    files = _shifted._base.discover_hdf5_files(directory)
    if not files:
        raise ValueError(f"No episode_*.hdf5 files found in {label} directory {directory}")
    return files


def discover_combined_hdf5_files(raw_dir: Path, raw_dir_eef: Path) -> list[Path]:
    """Return sorted ordinary files followed by sorted EEF files."""
    ordinary_files = _discover_directory(raw_dir, "ordinary source")
    eef_files = _discover_directory(raw_dir_eef, "EEF source")
    return ordinary_files + eef_files


def select_combined_files(files: list[Path], episodes: list[int] | None) -> list[Path]:
    """Select files using indices in the combined ordinary-then-EEF list."""
    if episodes is None:
        return list(files)
    selected = list(episodes)
    if len(selected) != len(set(selected)):
        raise ValueError("episode index values must be unique")
    invalid = [index for index in selected if index < 0 or index >= len(files)]
    if invalid:
        raise IndexError(f"episode index out of range: {invalid[0]} (found {len(files)} combined files)")
    return [files[index] for index in selected]


def populate_dataset(
    dataset: object,
    hdf5_files: list[Path],
    episodes: list[int] | None = None,
) -> object:
    """Populate using combined indices and the existing shifted frame mapping."""
    selected_files = select_combined_files(hdf5_files, episodes)
    return _shifted.populate_dataset(dataset, selected_files)


def convert_ros2_dataset(
    raw_dir: Path,
    raw_dir_eef: Path,
    repo_id: str,
    *,
    mode: Literal["image", "video"] = "image",
    episodes: list[int] | None = None,
    overwrite: bool = False,
) -> Path:
    """Validate and convert the combined ordinary-then-EEF source list."""
    files = discover_combined_hdf5_files(raw_dir, raw_dir_eef)
    selected_files = select_combined_files(files, episodes)
    for path in selected_files:
        _shifted._validate_shiftable_episode(path)

    output_dir = LEROBOT_HOME / repo_id
    _shifted._base.ensure_output_available(output_dir, overwrite=overwrite)
    try:
        dataset = _shifted._base.create_empty_dataset(repo_id, output_dir, mode=mode)
        _shifted.populate_dataset(dataset, selected_files)
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return output_dir


def parse_episode_selection(value: str | None) -> list[int] | None:
    return _shifted.parse_episode_selection(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True, help="Ordinary ROS2 episode directory; emitted first")
    parser.add_argument("--raw-dir-eef", type=Path, required=True, help="EEF ROS2 episode directory; EEF fields are ignored")
    parser.add_argument("--repo-id", required=True, help="LeRobot repository identifier")
    parser.add_argument("--mode", choices=("image", "video"), default="image")
    parser.add_argument("--episodes", help="Comma-separated indices in the combined ordinary-then-EEF source list")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing local LeRobot output")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = convert_ros2_dataset(
        args.raw_dir,
        args.raw_dir_eef,
        args.repo_id,
        mode=args.mode,
        episodes=parse_episode_selection(args.episodes),
        overwrite=args.overwrite,
    )
    print(f"Converted combined ROS2 dataset to {output}")


if __name__ == "__main__":
    main()
