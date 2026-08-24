from __future__ import annotations
from datetime import datetime, timezone

from pathlib import Path
from typing import Any, Mapping
import json


class RunMetadataRecorder:
    def __init__(self, path: Path, metadata: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self._metadata = dict(metadata)
        self._started_at: str | None = None
        self._finished = False
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        if self._started_at is None:
            self._started_at = self._now_utc_iso()
        payload = dict(self._metadata)
        payload["started_at"] = self._started_at
        self._write_atomic(payload)

    def finish(self, exit_code: int, exit_reason: str) -> None:
        if self._finished:
            return
        if self._started_at is None:
            self._started_at = self._now_utc_iso()
        self._finished = True
        payload = dict(self._metadata)
        payload["started_at"] = self._started_at
        payload["ended_at"] = self._now_utc_iso()
        payload["exit_code"] = exit_code
        payload["exit_reason"] = exit_reason
        self._write_atomic(payload)

    def _now_utc_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _write_atomic(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        temporary.replace(self.path)
