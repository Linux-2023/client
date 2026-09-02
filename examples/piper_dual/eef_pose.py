"""Conversions between Piper XYZ-RPY poses and position-plus-rot6d poses."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_EPSILON = 1e-7


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), found {array.shape}")
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain numeric values")
    result = array.astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def rpy_to_rot6d(rpy: Any) -> np.ndarray:
    """Convert XYZ roll/pitch/yaw to column-major first-two-column rot6d."""
    roll, pitch, yaw = (float(value) for value in _vector(rpy, 3, "rpy"))
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return np.asarray(
        [
            cy * cp,
            sy * cp,
            -sp,
            cy * sp * sr - sy * cr,
            sy * sp * sr + cy * cr,
            cp * sr,
        ],
        dtype=np.float32,
    )


def eef_rpy_to_rot9(pose: Any) -> np.ndarray:
    """Convert ``[x,y,z,roll,pitch,yaw]`` to ``[xyz,rot6d]``."""
    values = _vector(pose, 6, "pose")
    return np.concatenate((values[:3], rpy_to_rot6d(values[3:])), dtype=np.float32)


def rot6d_to_rpy(rot6d: Any) -> np.ndarray:
    """Decode column-major rot6d with Gram-Schmidt orthogonalization."""
    values = _vector(rot6d, 6, "rot6d").astype(np.float64)
    first = values[:3]
    second = values[3:]
    first_norm = float(np.linalg.norm(first))
    if first_norm < _EPSILON:
        raise ValueError("rot6d first column is degenerate")
    column1 = first / first_norm
    second_orthogonal = second - float(np.dot(column1, second)) * column1
    second_norm = float(np.linalg.norm(second_orthogonal))
    if second_norm < _EPSILON:
        raise ValueError("rot6d second column is degenerate")
    column2 = second_orthogonal / second_norm
    column3 = np.cross(column1, column2)
    rotation = np.column_stack((column1, column2, column3))
    sin_pitch = float(-rotation[2, 0])
    pitch = math.asin(float(np.clip(sin_pitch, -1.0, 1.0)))
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    result = np.asarray([roll, pitch, yaw], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("decoded RPY values must be finite")
    return result


def eef_rot9_to_poscmd_values(pose9: Any) -> np.ndarray:
    """Convert ``[xyz,rot6d]`` to Piper PosCmd's six pose values."""
    values = _vector(pose9, 9, "pose9")
    return np.concatenate((values[:3], rot6d_to_rpy(values[3:])), dtype=np.float32)
