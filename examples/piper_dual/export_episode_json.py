#!/usr/bin/env python3
"""Export synchronized Piper dual HDF5 metadata and robot state to JSON."""

from __future__ import annotations

import argparse
import sys
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np


PUPPET_STATE_SCHEMA_VERSION = "piper_dual_puppet_state_v1"
JOINT_NAMES = ("J1", "J2", "J3", "J4", "J5", "J6", "gripper")
JOINT_UNITS = ("rad", "rad", "rad", "rad", "rad", "rad", "m")
EXPECTED_HDF5_SCHEMA_VERSION = "piper_dual_ros2_v1"
EXPECTED_STATE_UNITS = "mixed_rad_m"
EXPECTED_TIMESTAMP_UNITS = "seconds"
EXPECTED_SYNC_ERROR_UNITS = "seconds"
EEF_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")
EEF_UNITS = ("m", "m", "m", "rad", "rad", "rad")


def export_puppet_state(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Export Puppet state and optional Puppet EEF with synchronized timestamps."""

    episode_path = Path(input_path)
    destination = Path(output_path)
    try:
        with h5py.File(episode_path, "r") as episode:
            _validate_hdf5_schema(episode, episode_path)
            state = _read_numeric_dataset(episode, "/observations/state", expected_columns=14)
            reference_timestamps = _read_numeric_dataset(episode, "/observations/timestamp")
            left_timestamps = _read_numeric_dataset(episode, "/observations/sensor_timestamps/puppet_left")
            right_timestamps = _read_numeric_dataset(episode, "/observations/sensor_timestamps/puppet_right")
            left_sync_error = _read_numeric_dataset(episode, "/observations/sync_error/puppet_left")
            right_sync_error = _read_numeric_dataset(episode, "/observations/sync_error/puppet_right")
            frame_count = int(episode.attrs.get("frame_count", len(state)))
            eef = _read_eef_data(episode, frame_count)
    except OSError as exc:
        raise ValueError(f"could not open HDF5 file {episode_path}: {exc}") from exc

    arrays = {
        "observations/state": state,
        "observations/timestamp": reference_timestamps,
        "observations/sensor_timestamps/puppet_left": left_timestamps,
        "observations/sensor_timestamps/puppet_right": right_timestamps,
        "observations/sync_error/puppet_left": left_sync_error,
        "observations/sync_error/puppet_right": right_sync_error,
    }
    if eef is not None:
        arrays.update(
            {
                "observations/eef/puppet_left": eef["left"]["pose"],
                "observations/eef/puppet_right": eef["right"]["pose"],
                "observations/sensor_timestamps/eef_puppet_left": eef["left"]["timestamp"],
                "observations/sensor_timestamps/eef_puppet_right": eef["right"]["timestamp"],
                "observations/sync_error/eef_puppet_left": eef["left"]["sync_error"],
                "observations/sync_error/eef_puppet_right": eef["right"]["sync_error"],
            }
        )
    for name, values in arrays.items():
        if len(values) != frame_count:
            raise ValueError(f"{name} length {len(values)} does not match frame_count {frame_count}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")

    frames: list[dict[str, Any]] = []
    for index in range(frame_count):
        left = {
            "timestamp": float(left_timestamps[index]),
            "position": [float(value) for value in state[index, :7]],
            "sync_error": float(left_sync_error[index]),
        }
        right = {
            "timestamp": float(right_timestamps[index]),
            "position": [float(value) for value in state[index, 7:]],
            "sync_error": float(right_sync_error[index]),
        }
        if eef is not None:
            left["eef"] = {
                "timestamp": float(eef["left"]["timestamp"][index]),
                "pose": [float(value) for value in eef["left"]["pose"][index]],
                "sync_error": float(eef["left"]["sync_error"][index]),
            }
            right["eef"] = {
                "timestamp": float(eef["right"]["timestamp"][index]),
                "pose": [float(value) for value in eef["right"]["pose"][index]],
                "sync_error": float(eef["right"]["sync_error"][index]),
            }
        frames.append(
            {
                "index": index,
                "reference_timestamp": float(reference_timestamps[index]),
                "left": left,
                "right": right,
            }
        )

    payload = {
        "schema_version": PUPPET_STATE_SCHEMA_VERSION,
        "source_hdf5": str(episode_path),
        "frame_count": frame_count,
        "state_source": "/observations/state",
        "timestamp_sources": {
            "reference": "/observations/timestamp",
            "left": "/observations/sensor_timestamps/puppet_left",
            "right": "/observations/sensor_timestamps/puppet_right",
        },
        "joint_names": list(JOINT_NAMES),
        "joint_units": list(JOINT_UNITS),
        "frames": frames,
    }
    if eef is not None:
        payload["eef_enabled"] = True
        payload["eef_names"] = list(EEF_NAMES)
        payload["eef_units"] = list(EEF_UNITS)
        payload["eef_sources"] = {
            "left": "/observations/eef/puppet_left",
            "right": "/observations/eef/puppet_right",
            "left_timestamp": "/observations/sensor_timestamps/eef_puppet_left",
            "right_timestamp": "/observations/sensor_timestamps/eef_puppet_right",
        }
    _write_json(destination, payload)
    return {"path": str(destination), "frame_count": frame_count, "eef_enabled": eef is not None}


def _read_numeric_dataset(
    episode: h5py.File,
    path: str,
    *,
    expected_columns: int | None = None,
) -> np.ndarray:
    dataset = episode.get(path)
    if not isinstance(dataset, h5py.Dataset):
        raise ValueError(f"HDF5 dataset is missing: {path}")
    values = np.asarray(dataset)
    expected_dimensions = 2 if expected_columns is not None else 1
    if values.ndim != expected_dimensions:
        raise ValueError(f"{path} must be {expected_dimensions}D")
    if expected_columns is not None and values.shape[1] != expected_columns:
        raise ValueError(f"{path} must contain exactly {expected_columns} columns")
    if values.dtype.kind not in "fiu":
        raise ValueError(f"{path} must contain numeric values")
    return values


def _read_eef_data(episode: h5py.File, frame_count: int) -> dict[str, dict[str, np.ndarray]] | None:
    dataset_paths = (
        "/observations/eef/puppet_left",
        "/observations/eef/puppet_right",
        "/observations/sensor_timestamps/eef_puppet_left",
        "/observations/sensor_timestamps/eef_puppet_right",
        "/observations/sync_error/eef_puppet_left",
        "/observations/sync_error/eef_puppet_right",
    )
    present = [path for path in dataset_paths if path in episode]
    if not present:
        return None
    missing = [path for path in dataset_paths if path not in episode]
    if missing:
        raise ValueError(f"EEF data is incomplete; missing {', '.join(missing)}")

    raw_units = episode.attrs.get("units_json")
    if isinstance(raw_units, bytes):
        raw_units = raw_units.decode("utf-8", errors="replace")
    try:
        units = json.loads(raw_units) if isinstance(raw_units, str) else {}
    except json.JSONDecodeError as exc:
        raise ValueError("units_json is invalid while reading EEF data") from exc
    if not isinstance(units, dict) or units.get("eef") != "xyz_m_rpy_rad":
        raise ValueError("units_json does not describe the expected EEF xyz_m_rpy_rad data")

    left_pose = _read_numeric_dataset(episode, "/observations/eef/puppet_left", expected_columns=6)
    right_pose = _read_numeric_dataset(episode, "/observations/eef/puppet_right", expected_columns=6)
    left_timestamp = _read_numeric_dataset(episode, "/observations/sensor_timestamps/eef_puppet_left")
    right_timestamp = _read_numeric_dataset(episode, "/observations/sensor_timestamps/eef_puppet_right")
    left_sync_error = _read_numeric_dataset(episode, "/observations/sync_error/eef_puppet_left")
    right_sync_error = _read_numeric_dataset(episode, "/observations/sync_error/eef_puppet_right")
    return {
        "left": {"pose": left_pose, "timestamp": left_timestamp, "sync_error": left_sync_error},
        "right": {"pose": right_pose, "timestamp": right_timestamp, "sync_error": right_sync_error},
    }



def _validate_hdf5_schema(episode: h5py.File, episode_path: Path) -> None:
    schema_version = episode.attrs.get("schema_version")
    if isinstance(schema_version, bytes):
        schema_version = schema_version.decode("utf-8", errors="replace")
    if schema_version != EXPECTED_HDF5_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version for {episode_path} must be {EXPECTED_HDF5_SCHEMA_VERSION!r}, got {schema_version!r}"
        )
    raw_units = episode.attrs.get("units_json")
    if isinstance(raw_units, bytes):
        raw_units = raw_units.decode("utf-8", errors="replace")
    if not isinstance(raw_units, str):
        raise ValueError(f"units_json missing from {episode_path}")
    try:
        units = json.loads(raw_units)
    except json.JSONDecodeError as exc:
        raise ValueError(f"units_json for {episode_path} is invalid JSON") from exc
    expected = {
        "state": EXPECTED_STATE_UNITS,
        "timestamp": EXPECTED_TIMESTAMP_UNITS,
        "sync_error": EXPECTED_SYNC_ERROR_UNITS,
    }
    if not isinstance(units, dict) or any(units.get(key) != value for key, value in expected.items()):
        raise ValueError(f"units_json for {episode_path} does not describe the Piper dual state schema")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Puppet state from one Piper dual HDF5 episode")
    parser.add_argument("--input", required=True, type=Path, help="Input episode_*.hdf5 file")
    parser.add_argument("--puppet-output", required=True, type=Path, help="Output puppet_state.json path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = export_puppet_state(args.input, args.puppet_output)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
