"""XYZ+RPY EEF observation adapter for the Piper dual ROS 2 contract."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from observation_adapter import CameraSpec
from observation_adapter import DEFAULT_CAMERA_SPECS
from observation_adapter import ObservationAdapter

OBSERVATION_DIMENSION = 14
EEF_POSE_DIMENSION = 6


def _validate_pose(name: str, pose: object) -> np.ndarray:
    values = np.asarray(pose)
    if values.ndim != 1 or values.shape != (EEF_POSE_DIMENSION,) or values.dtype.kind not in "fiu":
        raise ValueError(f"Observation {name} must contain exactly six numeric values")
    values = values.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"Observation {name} values must be finite")
    return values


class EefXyz3dObservationAdapter:
    """Adapt synchronized images, puppet XYZ+RPY poses, and master grippers."""

    def __init__(self, cameras: Sequence[CameraSpec] = DEFAULT_CAMERA_SPECS, task: str = "") -> None:
        self._image_adapter = ObservationAdapter(cameras=cameras, task=task)

    @property
    def camera_specs(self) -> tuple[CameraSpec, ...]:
        return self._image_adapter.camera_specs

    def adapt(self, message: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(message, dict):
            raise ValueError("Observation message must be a JSON object")
        eef = message.get("eef")
        if not isinstance(eef, dict):
            raise ValueError("Observation message must include eef")
        if "puppet_left" not in eef or "puppet_right" not in eef:
            raise ValueError("Observation eef must include puppet_left and puppet_right")
        left = _validate_pose("puppet_left", eef["puppet_left"])
        right = _validate_pose("puppet_right", eef["puppet_right"])

        master = np.asarray(message.get("action"))
        if master.ndim != 1 or master.shape != (14,) or master.dtype.kind not in "fiu":
            raise ValueError("Observation action must contain exactly fourteen numeric values")
        master = master.astype(np.float32, copy=False)
        if not np.isfinite(master).all():
            raise ValueError("Observation action values must be finite")

        state = np.empty((OBSERVATION_DIMENSION,), dtype=np.float32)
        state[:6] = left
        state[6] = master[6]
        state[7:13] = right
        state[13] = master[13]

        image_message = dict(message)
        image_message["state"] = np.zeros(14, dtype=np.float32)
        adapted = self._image_adapter.adapt(image_message)
        adapted["observation.state"] = state
        return adapted
