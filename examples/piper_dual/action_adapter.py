"""Shared action adapter for the Piper dual ROS 2 contract."""

from __future__ import annotations

from typing import Sequence

import numpy as np

ACTION_DIMENSION = 14
ARM_DIMENSION = 7


class ActionAdapter:
    """Validate and split 14-DoF dual-arm action vectors."""

    def validate(self, action: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(action)
        if values.ndim != 1 or values.shape[0] != ACTION_DIMENSION:
            raise ValueError("Action must contain exactly fourteen values")
        if values.dtype.kind not in "fiu":
            raise ValueError("Action values must be numeric")
        values = values.astype(np.float32, copy=False)
        if not np.isfinite(values).all():
            raise ValueError("Action values must be finite")
        return values

    def split(self, action: Sequence[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        validated = self.validate(action)
        return validated[:ARM_DIMENSION], validated[ARM_DIMENSION:]

    def to_bridge_request(self, action: Sequence[float] | np.ndarray) -> dict[str, list[float] | str]:
        left, right = self.split(action)
        return {
            "type": "publish_action",
            "left": left.tolist(),
            "right": right.tolist(),
        }
