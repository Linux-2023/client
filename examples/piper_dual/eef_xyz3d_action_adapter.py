"""XYZ+RPY EEF action adapter for the Piper dual ROS 2 contract."""

from __future__ import annotations

from typing import Sequence

import numpy as np

ACTION_DIMENSION = 14
ARM_ACTION_DIMENSION = 7


class EefXyz3dActionAdapter:
    """Validate 14D XYZ+RPY EEF actions and encode Piper PosCmd requests."""

    def validate(self, action: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(action)
        if values.ndim != 1 or values.shape != (ACTION_DIMENSION,):
            raise ValueError("XYZ3D EEF action must contain exactly fourteen values")
        if values.dtype.kind not in "fiu":
            raise ValueError("XYZ3D EEF action values must be numeric")
        values = values.astype(np.float32, copy=False)
        if not np.isfinite(values).all():
            raise ValueError("XYZ3D EEF action values must be finite")
        return values

    def to_bridge_request(self, action: Sequence[float] | np.ndarray) -> dict[str, object]:
        values = self.validate(action)
        return {
            "type": "publish_eef_action",
            "left": values[:ARM_ACTION_DIMENSION].tolist(),
            "right": values[ARM_ACTION_DIMENSION:].tolist(),
        }
