"""Buffered, opt-in recording of EEF submissions through runtime subscribers."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, TextIO

import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


class EefTraceRecorder(_subscriber.Subscriber):
    """Record accepted client submissions, never inferred physical execution.

    Only client.jsonl and client_metadata.json are owned here; an independent
    read-only ROS collector may use the same directory. Neither file is replaced.
    """

    def __init__(self, output_dir: Path, metadata: dict[str, Any], *, flush_every: int = 30) -> None:
        if flush_every < 1:
            raise ValueError("flush_every must be positive")
        self._stream: TextIO | None = None
        self._episode = -1
        self._step = 0
        self._in_episode = False
        self._flush_every = flush_every
        self._pending = 0
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = output_dir / "client_metadata.json"
        trace_path = output_dir / "client.jsonl"
        for path in (metadata_path, trace_path):
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"Refusing to overwrite trace file: {path}")
        encoded_metadata = json.dumps(
            {
                **metadata,
                "schema_version": 1,
                "event": "client_metadata",
                "timestamp_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                "episode": None,
                "step": 0,
            },
            indent=2,
            allow_nan=False,
        )
        # Exclusive opens also protect against a second writer racing preflight.
        # Context management and BaseException cleanup include Ctrl-C mid-init.
        try:
            with metadata_path.open("x", encoding="utf-8") as stream:
                stream.write(encoded_metadata + "\n")
            self._stream = trace_path.open("x", encoding="utf-8", buffering=65536)
        except BaseException:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
            raise

    @override
    def on_episode_start(self) -> None:
        self._ensure_open()
        if self._in_episode:
            raise RuntimeError("EEF trace episode has already started")
        self._episode += 1
        self._in_episode = True
        self._record("episode_start")

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        self._ensure_open()
        if not self._in_episode:
            raise RuntimeError("EEF trace episode has not started")
        # Runtime calls this only after environment.apply_action returns. These
        # are policy EEF values, not measured joints and not SDK dispatch claims.
        self._record(
            "action_submitted",
            action=self._vector(action.get("actions"), "action"),
            observation_state=self._vector(observation.get("state"), "observation_state"),
            _chunk_trace=action.get("_chunk_trace"),
        )
        self._step += 1

    @override
    def on_episode_end(self) -> None:
        self._ensure_open()
        if self._in_episode:
            self._record("episode_end", reason="completed")
            self._in_episode = False
            self._stream.flush()
            self._pending = 0

    def close(self, *, reason: str = "closed") -> None:
        """Flush all records; mark an unfinished episode with its exit reason."""
        if self._stream is None:
            return
        try:
            if self._in_episode:
                self._record("episode_end", reason=reason)
                self._in_episode = False
            self._stream.flush()
        finally:
            try:
                self._stream.close()
            finally:
                self._stream = None

    def _ensure_open(self) -> None:
        if self._stream is None:
            raise RuntimeError("EEF trace recorder is closed")

    def _record(self, event: str, **fields: Any) -> None:
        self._ensure_open()
        record = {
            "schema_version": 1,
            "event": event,
            "timestamp_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            "episode": self._episode,
            "step": self._step,
            **fields,
        }
        self._stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        self._pending += 1
        if self._pending >= self._flush_every:
            self._stream.flush()
            self._pending = 0

    @staticmethod
    def _vector(value: object, name: str) -> list[float]:
        # float64 conversion preserves float32 inputs without introducing a new
        # float32 rounding boundary for policies that return float64 values.
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (14,) or not np.isfinite(vector).all():
            raise ValueError(f"{name} must contain exactly 14 finite EEF values")
        return vector.tolist()
