from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from joint_trace_recorder import JointTraceRecorder


def test_records_server_action_actual_joint_and_error(tmp_path: Path) -> None:
    recorder = JointTraceRecorder(tmp_path)
    recorder.on_episode_start()
    recorder.on_step(
        {"state": np.arange(14, dtype=np.float32)},
        {"actions": np.arange(14, dtype=np.float32) + 0.5},
    )

    path = tmp_path / "episode_000000.jsonl"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["step"] == 0
    assert record["actual_joint"] == np.arange(14).tolist()
    assert record["server_action"] == (np.arange(14) + 0.5).tolist()
    assert record["tracking_error"] == [-0.5] * 14


def test_each_episode_uses_a_new_file(tmp_path: Path) -> None:
    recorder = JointTraceRecorder(tmp_path)
    for _ in range(2):
        recorder.on_episode_start()
        recorder.on_step({"state": np.zeros(14)}, {"actions": np.zeros(14)})
        recorder.on_episode_end()

    assert sorted(path.name for path in tmp_path.glob("*.jsonl")) == [
        "episode_000000.jsonl",
        "episode_000001.jsonl",
    ]
