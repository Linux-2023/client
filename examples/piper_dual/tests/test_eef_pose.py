"""Tests for EEF RPY and rot6d conversions."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eef_pose import eef_rot9_to_poscmd_values
from eef_pose import eef_rpy_to_rot9
from eef_pose import rot6d_to_rpy
from eef_pose import rpy_to_rot6d


def test_rpy_to_rot6d_identity() -> None:
    result = rpy_to_rot6d(np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(result, [1, 0, 0, 0, 1, 0], atol=1e-6)
    assert result.dtype == np.float32


def test_rpy_to_rot6d_positive_yaw_90_degrees() -> None:
    result = rpy_to_rot6d(np.array([0, 0, np.pi / 2], dtype=np.float32))
    np.testing.assert_allclose(result, [0, 1, 0, -1, 0, 0], atol=1e-6)


def test_rpy_rot6d_round_trip() -> None:
    rpy = np.array([0.31, -0.42, 1.17], dtype=np.float32)
    decoded = rot6d_to_rpy(rpy_to_rot6d(rpy))
    np.testing.assert_allclose(decoded, rpy, atol=1e-5)


def test_eef_pose_round_trip_preserves_position() -> None:
    pose = np.array([0.2, -0.1, 0.35, 0.1, -0.2, 0.3], dtype=np.float32)
    pose9 = eef_rpy_to_rot9(pose)
    decoded = eef_rot9_to_poscmd_values(pose9)
    np.testing.assert_allclose(decoded, pose, atol=1e-5)
    assert pose9.shape == (9,)
    assert pose9.dtype == np.float32


@pytest.mark.parametrize(
    "value",
    [
        np.zeros(5, dtype=np.float32),
        np.array([0, 0, np.nan], dtype=np.float32),
        np.array(["0", "0", "0"], dtype=object),
    ],
)
def test_rpy_to_rot6d_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        rpy_to_rot6d(value)


@pytest.mark.parametrize(
    "rot6d",
    [
        np.zeros(6, dtype=np.float32),
        np.array([np.nan, 0, 0, 0, 1, 0], dtype=np.float32),
        np.ones(5, dtype=np.float32),
    ],
)
def test_rot6d_to_rpy_rejects_degenerate_or_invalid_values(rot6d: object) -> None:
    with pytest.raises(ValueError):
        rot6d_to_rpy(rot6d)


def test_eef_rot9_to_poscmd_rejects_nonfinite_pose() -> None:
    with pytest.raises(ValueError):
        eef_rot9_to_poscmd_values(np.array([0, 0, 0, 1, 0, 0, 0, np.inf, 0], dtype=np.float32))
