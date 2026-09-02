from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


class JointTraceRecorder(_subscriber.Subscriber):
    """Append server actions and measured joints to one JSONL file per episode."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._path: Path | None = None
        self._step = 0

    @override
    def on_episode_start(self) -> None:
        indices = [int(path.stem.split("_")[1]) for path in self._output_dir.glob("episode_[0-9]*.jsonl")]
        self._path = self._output_dir / f"episode_{max(indices, default=-1) + 1:06d}.jsonl"
        self._step = 0

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        if self._path is None:
            raise RuntimeError("joint trace episode has not started")
        actual = self._vector(observation.get("state"), "actual_joint")
        target = self._vector(action.get("actions"), "server_action")
        record = {
            "timestamp": time.time(),
            "step": self._step,
            "server_action": target.tolist(),
            "actual_joint": actual.tolist(),
            "tracking_error": (actual - target).tolist(),
        }
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._step += 1

    @override
    def on_episode_end(self) -> None:
        self._path = None

    @staticmethod
    def _vector(value: object, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (14,) or not np.isfinite(vector).all():
            raise ValueError(f"{name} must contain exactly 14 finite values")
        return vector
