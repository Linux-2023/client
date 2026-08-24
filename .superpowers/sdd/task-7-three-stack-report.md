# Task 7 three-stack report

## Status
DONE

## Scope
Focused Task 7 only:
- `examples/piper_dual/main_dual.py`
- `examples/piper_dual/saver.py`
- `examples/piper_dual/run_metadata.py`
- `examples/piper_dual/README.md`
- `examples/piper_dual/tests/test_main_dual.py`
- `examples/piper_dual/tests/test_run_metadata.py`

## What changed
- Added `_comparison_out_dir()` to keep each run tag isolated under `out_dir/<run-tag>` and reject unsafe tags.
- Wired `main_dual.py` to pass the stack-specific output directory to `VideoSaver` and `RobotStatePlotter`.
- Added `RunMetadataRecorder` with atomic sibling-temp + `replace()` writes and idempotent finalization.
- Extended metadata to record stack, prompt, policy/runtime/config values, `video_fps`, `effective_speed_percent`, timestamps, exit code, and exit reason.
- Updated `VideoSaver` to validate positive finite runtime fps and emit playback fps as `fps / subsample`.
- Added focused tests for output isolation, metadata lifecycle, and recorder idempotence.
- Appended a three-stack manual runbook with exact local, official, and direct commands, explicit status checks/enables, stop order, and the 30% vs 100% disclosure.

## Verification
- RED: focused Task 7 tests failed before the final fixes when output isolation and metadata lifecycle were incomplete.
- GREEN: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q examples/piper_dual/tests/test_main_dual.py examples/piper_dual/tests/test_run_metadata.py`
  - Result: `22 passed in 0.14s`
- GREEN syntax check: `python -m py_compile examples/piper_dual/main_dual.py examples/piper_dual/saver.py examples/piper_dual/run_metadata.py examples/piper_dual/tests/test_main_dual.py examples/piper_dual/tests/test_run_metadata.py`
  - Result: no output

## Attribution / staging
- The six pre-existing `main_dual.py` user-default hunks remain unstaged: `action_horizon=30`, `num_steps=8000`, `port=8001`, prompt text, `use_async=False`, and `record_mode=False`.
- Task 7 hunks are intended to be committed separately from those unstaged defaults.

## Notes
- No hardware actions were run.
- No broad suites or formatters were used.
