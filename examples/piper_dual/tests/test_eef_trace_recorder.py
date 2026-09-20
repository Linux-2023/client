"""EEF trace recorder contracts; no ROS or hardware access."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_eef_trace_records_submissions_and_episode_boundaries(tmp_path: Path) -> None:
    from eef_trace_recorder import EefTraceRecorder

    recorder = EefTraceRecorder(tmp_path, {"args": {"use_rtc": True}}, flush_every=1)
    trace = {"chunk_id": 0, "chunk_step": 3, "chunk_boundary": True, "skipped_steps": 3}
    for _ in range(2):
        recorder.on_episode_start()
        recorder.on_step(
            {"state": np.arange(14, dtype=np.float64)}, {"actions": np.arange(14) + 0.123456789, "_chunk_trace": trace}
        )
        recorder.on_episode_end()
    records = [json.loads(line) for line in (tmp_path / "client.jsonl").read_text().splitlines()]
    assert [record["event"] for record in records] == ["episode_start", "action_submitted", "episode_end"] * 2
    submitted = [record for record in records if record["event"] == "action_submitted"]
    assert [record["step"] for record in submitted] == [0, 1]
    assert [record["episode"] for record in submitted] == [0, 1]
    assert submitted[0]["action"] == (np.arange(14) + 0.123456789).tolist()
    assert submitted[0]["observation_state"] == np.arange(14).tolist()
    assert submitted[0]["_chunk_trace"] == trace
    assert "tracking_error" not in submitted[0]
    assert all(record["schema_version"] == 1 for record in records)
    assert all(
        isinstance(record["timestamp_ns"], int) and isinstance(record["monotonic_ns"], int) for record in records
    )
    recorder.close()
    recorder.close()
    assert json.loads((tmp_path / "client_metadata.json").read_text())["args"]["use_rtc"] is True


@pytest.mark.parametrize("mode", ["legacy", "wrapped-rpy", "so3", None])
def test_server_metadata_roundtrips_and_flushes_before_first_episode(tmp_path: Path, mode: str | None) -> None:
    from eef_trace_recorder import EefTraceRecorder

    metadata = (
        {
            "rtc_orientation": {"mode": mode, "rpy_indices": [[3, 4, 5], [10, 11, 12]]},
            "policy_config": "pi05_piper_dual_stack_cups_eef_xyz3d",
            "checkpoint_dir": "/checkpoints/xyz3d test",
            "existing_server_field": {"nested": [1, True, None]},
        }
        if mode is not None
        else {}
    )
    recorder = EefTraceRecorder(tmp_path, {"args": {"use_rtc": True}}, flush_every=100)
    initial_metadata = (tmp_path / "client_metadata.json").read_bytes()
    try:
        recorder.record_server_metadata(metadata)
        records = [json.loads(line) for line in (tmp_path / "client.jsonl").read_text().splitlines()]
        assert len(records) == 1
        assert records[0]["event"] == "server_metadata"
        assert records[0]["metadata"] == metadata
        assert records[0]["schema_version"] == 1
        assert records[0]["episode"] == -1
        assert records[0]["step"] == 0
        assert isinstance(records[0]["timestamp_ns"], int)
        assert isinstance(records[0]["monotonic_ns"], int)
        assert (tmp_path / "client_metadata.json").read_bytes() == initial_metadata
        recorder.on_episode_start()
        recorder.on_step({"state": np.zeros(14)}, {"actions": np.ones(14)})
        recorder.on_episode_end()
    finally:
        recorder.close()
    records = [json.loads(line) for line in (tmp_path / "client.jsonl").read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "server_metadata", "episode_start", "action_submitted", "episode_end"
    ]
    assert (tmp_path / "client_metadata.json").read_bytes() == initial_metadata
    with pytest.raises(RuntimeError, match="closed"):
        recorder.record_server_metadata(metadata)


@pytest.mark.parametrize("existing", ["client.jsonl", "client_metadata.json"])
def test_eef_trace_refuses_overwrite_without_touching_existing_files(tmp_path: Path, existing: str) -> None:
    from eef_trace_recorder import EefTraceRecorder

    (tmp_path / existing).write_text("keep me")
    with pytest.raises(FileExistsError):
        EefTraceRecorder(tmp_path, {})
    assert (tmp_path / existing).read_text() == "keep me"
    assert {path.name for path in tmp_path.iterdir()} == {existing}


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_eef_trace_constructor_closes_partial_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type
) -> None:
    from eef_trace_recorder import EefTraceRecorder

    original_open = Path.open
    opened = []

    def open_path(path: Path, *args: object, **kwargs: object):
        if len(opened) == 1:
            raise error_type("second open failed")
        stream = original_open(path, *args, **kwargs)
        opened.append(stream)
        return stream

    monkeypatch.setattr(Path, "open", open_path)
    with pytest.raises(error_type):
        EefTraceRecorder(tmp_path, {})
    assert len(opened) == 1
    assert opened[0].closed


def test_eef_trace_close_flushes_and_marks_interrupted_episode(tmp_path: Path) -> None:
    from eef_trace_recorder import EefTraceRecorder

    recorder = EefTraceRecorder(tmp_path, {}, flush_every=100)
    recorder.on_episode_start()
    recorder.on_step({"state": np.zeros(14)}, {"actions": np.ones(14)})
    recorder.close(reason="interrupted")
    records = [json.loads(line) for line in (tmp_path / "client.jsonl").read_text().splitlines()]
    assert records[-1]["event"] == "episode_end"
    assert records[-1]["reason"] == "interrupted"
    assert records[1]["action"] == [1.0] * 14
    with pytest.raises(RuntimeError, match="closed"):
        recorder.on_episode_start()
