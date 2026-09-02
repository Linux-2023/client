"""Tests for combining ordinary and EEF Piper dual ROS2 episodes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "utils" / "convert_ros2_piper_data_to_lerobot_combined.py"
spec = importlib.util.spec_from_file_location("convert_ros2_piper_data_to_lerobot_combined", SCRIPT)
assert spec is not None and spec.loader is not None
converter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = converter
spec.loader.exec_module(converter)


def _touch_episode(directory: Path, number: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"episode_{number:06d}.hdf5"
    path.touch()
    return path


def test_discover_combined_files_orders_ordinary_before_eef(tmp_path: Path) -> None:
    ordinary = tmp_path / "ordinary"
    eef = tmp_path / "eef"
    ordinary_files = [_touch_episode(ordinary, number) for number in (2, 0, 1)]
    eef_files = [_touch_episode(eef, number) for number in (1, 0, 2)]

    discovered = converter.discover_combined_hdf5_files(ordinary, eef)

    assert discovered == sorted(ordinary_files) + sorted(eef_files)


def test_discover_combined_files_rejects_missing_or_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        converter.discover_combined_hdf5_files(tmp_path / "missing", tmp_path / "eef")

    ordinary = tmp_path / "ordinary"
    eef = tmp_path / "eef"
    ordinary.mkdir()
    eef.mkdir()
    with pytest.raises(ValueError, match="No .*episode_.*hdf5"):
        converter.discover_combined_hdf5_files(ordinary, eef)


def test_select_combined_files_uses_combined_indices_and_preserves_requested_order(tmp_path: Path) -> None:
    ordinary = [_touch_episode(tmp_path / "ordinary", number) for number in (0, 1)]
    eef = [_touch_episode(tmp_path / "eef", number) for number in (0, 1)]
    files = ordinary + eef

    assert converter.select_combined_files(files, None) == files
    assert converter.select_combined_files(files, [3, 0, 2]) == [files[3], files[0], files[2]]

    with pytest.raises(IndexError, match="episode index"):
        converter.select_combined_files(files, [-1])
    with pytest.raises(IndexError, match="episode index"):
        converter.select_combined_files(files, [4])


def test_select_combined_files_rejects_duplicate_indices(tmp_path: Path) -> None:
    files = [_touch_episode(tmp_path / "ordinary", number) for number in (0, 1)]

    with pytest.raises(ValueError, match="unique"):
        converter.select_combined_files(files, [0, 0])
