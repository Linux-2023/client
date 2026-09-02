"""EEF policy observation adapter for the Piper dual ROS 2 contract."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from eef_pose import eef_rpy_to_rot9
from observation_adapter import CameraSpec
from observation_adapter import DEFAULT_CAMERA_SPECS
from observation_adapter import ObservationAdapter

OBSERVATION_DIMENSION = 20


class EefObservationAdapter:
    """Adapt synchronized images, puppet EEF poses, and master grippers."""

    def __init__(self, cameras: Sequence[CameraSpec] = DEFAULT_CAMERA_SPECS, task: str = "") -> None:
        self._image_adapter = ObservationAdapter(cameras=cameras, task=task)
        self._task = task

    @property
    def camera_specs(self) -> tuple[CameraSpec, ...]:
        return self._image_adapter.camera_specs

    def adapt(self, message: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(message, dict):
            raise ValueError("Observation message must be a JSON object")
        eef = message.get("eef")
        if not isinstance(eef, dict):
            raise ValueError("Observation message must include eef")
        left = eef.get("puppet_left")
        right = eef.get("puppet_right")
        if left is None or right is None:
            raise ValueError("Observation eef must include puppet_left and puppet_right")
        master_action = message.get("action")
        master = np.asarray(master_action)
        if master.ndim != 1 or master.shape != (14,) or master.dtype.kind not in "fiu":
            raise ValueError("Observation action must contain exactly fourteen numeric values")
        master = master.astype(np.float32, copy=False)
        if not np.isfinite(master).all():
            raise ValueError("Observation action values must be finite")
        left_rot9 = eef_rpy_to_rot9(left)
        right_rot9 = eef_rpy_to_rot9(right)
        state = np.empty((OBSERVATION_DIMENSION,), dtype=np.float32)
        state[:9] = left_rot9
        state[9] = master[6]
        state[10:19] = right_rot9
        state[19] = master[13]
        # Reuse the established camera/timestamp/sync/task validation without accepting
        # the 14D joint state as the model state.
        image_message = dict(message)
        image_message["state"] = np.zeros(14, dtype=np.float32)
        adapted = self._image_adapter.adapt(image_message)
        adapted["observation.state"] = state
        return adapted
