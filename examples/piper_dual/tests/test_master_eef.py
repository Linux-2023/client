"""Tests for forward-kinematic master EEF poses."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from master_eef import DH_IS_OFFSET
from master_eef import MasterEefKinematics

# Live Piper pairs: joint feedback of one frame and the /puppet/end_pose_<side> value the
# firmware reported for the same arm. Both sides were stationary, so the recorded joint
# vector and pose describe one configuration.
LIVE_SAMPLES = (
    # left arm: 0.6 mm agreement limited by the sampling skew of the capture
    (
        [-9.994, 0.111, 0.0, -4.232, -26.973, 2.406, 0.0606],
        [0.042808, -0.004451, 0.253326],
        [-4.117, 58.129, -12.582],
    ),
    # right arm: matches the firmware end pose to the 0.001 mm feedback resolution
    (
        [1.273, 0.22, -0.162, -4.386, 14.023, 3.994, 0.0536],
        [0.055545, -0.000451, 0.191872],
        [-177.296, 80.949, -177.063],
    ),
)


@pytest.mark.parametrize(("degrees", "xyz_meters", "rpy_degrees"), LIVE_SAMPLES)
def test_master_eef_reproduces_the_firmware_end_pose_feedback(degrees, xyz_meters, rpy_degrees):
    kinematics = MasterEefKinematics()
    joints = np.deg2rad(np.asarray(degrees, dtype=np.float32))

    pose = kinematics.eef(joints)

    assert pose.dtype == np.dtype("float32")
    assert pose.shape == (6,)
    # The float32 output quantizes positions to ~0.002 mm, below the 0.001 mm feedback resolution scale.
    np.testing.assert_allclose(pose[:3], xyz_meters, atol=1e-5)
    np.testing.assert_allclose(np.rad2deg(pose[3:]), rpy_degrees, atol=1e-2)


def test_master_eef_ignores_the_gripper_value_of_the_joint_vector():
    kinematics = MasterEefKinematics()
    joints = np.deg2rad(np.asarray([-9.994, 0.111, 0.0, -4.232, -26.973, 2.406], dtype=np.float32))

    closed = kinematics.eef(np.append(joints, np.float32(0.0)))
    open_wide = kinematics.eef(np.append(joints, np.float32(0.07)))

    np.testing.assert_array_equal(closed, open_wide)


def test_master_eef_requires_the_vendor_dh_offset_used_by_the_firmware():
    assert DH_IS_OFFSET == 0x01
    joints = np.deg2rad(np.asarray([-9.994, 0.111, 0.0, -4.232, -26.973, 2.406, 0.0606], dtype=np.float32))

    without_offset = MasterEefKinematics(dh_is_offset=0x00).eef(joints)

    # The 0x00 table drops the 2 degree joint1/joint2 offset and misses the firmware pose by ~10 mm.
    assert abs(without_offset[2] - 0.253326) > 0.005


def test_master_eef_rejects_missing_offset_and_unusable_joint_vectors(tmp_path: Path):
    with pytest.raises(ValueError, match="dh_is_offset"):
        MasterEefKinematics(dh_is_offset=2)
    with pytest.raises(RuntimeError, match="kinematics not found"):
        MasterEefKinematics(kinematics_path=tmp_path / "missing.py")

    kinematics = MasterEefKinematics()
    with pytest.raises(ValueError, match="numpy"):
        kinematics.eef([0.0] * 7)
    with pytest.raises(ValueError, match="seven"):
        kinematics.eef(np.zeros(6, dtype=np.float32))
    with pytest.raises(ValueError, match="numeric"):
        kinematics.eef(np.array(["j"] * 7))
    with pytest.raises(ValueError, match="finite"):
        kinematics.eef(np.array([0.0] * 6 + [math.inf], dtype=np.float32))
