from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json


class RunMetadataRecorder:
    def __init__(self, path: Path, metadata: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self._metadata = dict(metadata)
        self._finished = False
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        payload = dict(self._metadata)
        payload["started_at"] = "start"
        self._write_atomic(payload)

    def finish(self, exit_code: int, exit_reason: str) -> None:
        if self._finished:
            return
        self._finished = True
        payload = dict(self._metadata)
        payload["started_at"] = payload.get("started_at", "start")
        payload["ended_at"] = "end"
        payload["exit_code"] = exit_code
        payload["exit_reason"] = exit_reason
        self._write_atomic(payload)

    def _write_atomic(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        temporary.replace(self.path)
