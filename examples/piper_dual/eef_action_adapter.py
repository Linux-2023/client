"""EEF policy action adapter for the Piper dual ROS 2 contract."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from eef_pose import eef_rot9_to_poscmd_values

ACTION_DIMENSION = 20
EEF_POSE_DIMENSION = 9
ARM_ACTION_DIMENSION = 10


class EefActionAdapter:
    """Validate 20D EEF actions and encode Piper PosCmd requests."""

    def validate(self, action: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(action)
        if values.ndim != 1 or values.shape != (ACTION_DIMENSION,):
            raise ValueError("EEF action must contain exactly twenty values")
        if values.dtype.kind not in "fiu":
            raise ValueError("EEF action values must be numeric")
        values = values.astype(np.float32, copy=False)
        if not np.isfinite(values).all():
            raise ValueError("EEF action values must be finite")
        # Decode before accepting the action so degenerate rotation columns fail closed.
        eef_rot9_to_poscmd_values(values[:EEF_POSE_DIMENSION])
        eef_rot9_to_poscmd_values(values[10:19])
        return values

    def to_bridge_request(self, action: Sequence[float] | np.ndarray) -> dict[str, list[float] | str]:
        values = self.validate(action)
        left_pose = eef_rot9_to_poscmd_values(values[:EEF_POSE_DIMENSION])
        right_pose = eef_rot9_to_poscmd_values(values[10:19])
        return {
            "type": "publish_eef_action",
            "left": np.concatenate((left_pose, values[9:10])).astype(np.float32).tolist(),
            "right": np.concatenate((right_pose, values[19:20])).astype(np.float32).tolist(),
        }
