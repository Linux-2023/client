"""Forward-kinematic master EEF poses derived from the Piper dual master joint stream."""

from __future__ import annotations

from collections.abc import Sequence
import importlib.util
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

EEF_DIMENSION = 6
JOINT_DIMENSION = 7
DH_IS_OFFSET = 0x01
VENDOR_KINEMATICS_PATH = (
    Path(__file__).resolve().parents[2] / "third_party" / "piper_sdk" / "piper_sdk" / "kinematics" / "piper_fk.py"
)


def _load_vendor_forward_kinematics(path: Path) -> Any:
    """Load the vendored ``C_PiperForwardKinematics`` class without importing the CAN SDK."""
    if not path.is_file():
        raise RuntimeError(f"vendored Piper kinematics not found: {path}")
    spec = importlib.util.spec_from_file_location("piper_vendor_forward_kinematics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load vendored Piper kinematics: {path}")
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.C_PiperForwardKinematics


class MasterEefKinematics:
    """Convert a master joint vector into the vendor link6 EEF pose in meters and radians.

    The Piper driver publishes end-pose feedback for the measured arm only, so the master
    (command-side) pose is computed from the synchronized master joint values with the same
    Denavit-Hartenberg model the firmware uses. ``dh_is_offset=0x01`` reproduces the vendor
    end-pose feedback to the feedback quantization floor.
    """

    def __init__(self, *, dh_is_offset: int = DH_IS_OFFSET, kinematics_path: Path = VENDOR_KINEMATICS_PATH) -> None:
        if dh_is_offset not in (0x00, 0x01):
            raise ValueError("dh_is_offset must be 0x00 or 0x01")
        self.dh_is_offset = int(dh_is_offset)
        self._forward_kinematics = _load_vendor_forward_kinematics(Path(kinematics_path))(self.dh_is_offset)

    def eef(self, joints: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return ``[x, y, z, roll, pitch, yaw]`` in meters and radians for one master arm.

        ``joints`` is the seven-value master vector ``[J1..J6, gripper]`` in radians and meters.
        The FK chain ends at link6, matching the puppet EEF feedback frame, so the gripper value
        is accepted for shape compatibility and does not influence the pose.
        """
        vector = _coerce_master_joints(joints)
        pose = self._forward_kinematics.CalFK([float(value) for value in vector[:6]])[-1]
        values = np.asarray(pose, dtype=np.float64)
        if values.shape != (EEF_DIMENSION,) or not np.isfinite(values).all():
            raise ValueError("forward kinematics returned an invalid pose")
        return np.asarray(
            [
                values[0] / 1000.0,
                values[1] / 1000.0,
                values[2] / 1000.0,
                math.radians(values[3]),
                math.radians(values[4]),
                math.radians(values[5]),
            ],
            dtype=np.float32,
        )


def _coerce_master_joints(joints: object) -> np.ndarray:
    if not isinstance(joints, np.ndarray):
        raise ValueError("master joints must be a numpy array")
    if joints.ndim != 1 or joints.shape[0] != JOINT_DIMENSION:
        raise ValueError("master joints must contain exactly seven values")
    if joints.dtype.kind not in "fiu":
        raise ValueError("master joint values must be numeric")
    values = joints.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("master joint values must be finite")
    return values
