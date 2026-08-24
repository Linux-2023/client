from __future__ import annotations

import json
from pathlib import Path

from run_metadata import RunMetadataRecorder


def test_run_metadata_recorder_writes_atomically_and_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / 'run_metadata.json'
    recorder = RunMetadataRecorder(path, {'control_stack': 'direct-sdk', 'video_fps': 30, 'effective_speed_percent': 100})

    recorder.start()
    first = json.loads(path.read_text(encoding='utf-8'))
    assert first['control_stack'] == 'direct-sdk'
    assert first['video_fps'] == 30
    assert first['effective_speed_percent'] == 100
    assert first['started_at'] == 'start'
    assert 'ended_at' not in first

    recorder.finish(130, 'keyboard_interrupt')
    second = json.loads(path.read_text(encoding='utf-8'))
    assert second['control_stack'] == 'direct-sdk'
    assert second['video_fps'] == 30
    assert second['effective_speed_percent'] == 100
    assert second['started_at'] == 'start'
    assert second['ended_at'] == 'end'
    assert second['exit_code'] == 130
    assert second['exit_reason'] == 'keyboard_interrupt'

    recorder.finish(1, 'ignored')
    third = json.loads(path.read_text(encoding='utf-8'))
    assert third == second
