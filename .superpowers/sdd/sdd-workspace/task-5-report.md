# Task 5 Report

## Status
Completed.

## Commit
`bdeb359` — `feat: add preview-first ROS 2 dual-arm collector`

## What changed
- Added `examples/piper_dual/collect_data_ros2.py`.
- Added focused state-machine tests in `examples/piper_dual/tests/test_collector_state_machine.py`.
- Updated `examples/piper_dual/README.md` with the ROS 2 launch flow and collector usage.

## Implementation summary
- Built a preview-first collector state machine with `PREVIEW`, `RECORDING`, `FINALIZING`, and `EXIT` states.
- Kept preview, controller, and renderer separated for testability.
- Used `Ros2BackendClient` for synchronized frames.
- Used `StreamingEpisodeWriter` for self-contained HDF5 episodes.
- Kept `--dry-run` as the safe default and rejected `--dry-run` with `--publish-actions`.
- `s` starts recording only after `clear_buffers()`.
- `e` finalizes the current episode, validates it, optionally renders a preview MP4, and returns to preview.
- `q` aborts the current partial file and closes backend/preview cleanly.
- Added unique episode numbering that skips existing final and partial files.
- Added a self-contained renderer for the new HDF5 schema so `--render-after-save` works with Task 5 output.

## Verification
### Focused tests
Command:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_collector_state_machine.py -q
```
Result: `9 passed in 0.16s`

### CLI help
Command:
```bash
python examples/piper_dual/collect_data_ros2.py --help
```
Result: help output includes:
- `--output-dir`
- `--prompt`
- `--config`
- `--bridge-python`
- `--jpeg-quality`
- `--max-sync-error-ms`
- `--dry-run` / `--no-dry-run`
- `--publish-actions`
- `--render-after-save`

## Review follow-up
A read-only review flagged one important issue: the original `--render-after-save` path called `visualize_hdf5.py`, which expects the old `observations/qpos` schema. That was fixed by replacing the render path with a self-contained MP4 preview renderer that consumes the new streaming HDF5 schema.

## Concerns
- `uv` is not available in this environment, so the help check was verified with `python examples/piper_dual/collect_data_ros2.py --help` instead of `uv run ...`.
- The repository still contains an unrelated pre-existing modification in `.superpowers/sdd/sdd-workspace/task-1-report.md`.

## 2026-08-16 Review fixes

### RED
Command:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_collector_state_machine.py -q
```
Output excerpt:
```text
FAILED examples/piper_dual/tests/test_collector_state_machine.py::test_empty_real_writer_episode_restores_preview_keeps_backend_and_next_valid_episode_succeeds - ValueError: episode validation failed: ['episode contains zero frames', 'timestamps must be strictly nondecreasing']
FAILED examples/piper_dual/tests/test_collector_state_machine.py::test_real_writer_validation_failure_restores_preview_removes_partial_and_allows_next_valid_episode - ValueError: episode validation failed: ['/observations/images/cam_high[0] is not decodable JPEG bytes', '/observations/images/cam_left_wrist[0] is not decodable JPEG bytes', '/observations/images/cam_right_wrist[0] is not decodable JPEG bytes']
2 failed, 9 passed in 0.27s
```

### GREEN
Command:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_collector_state_machine.py -q
/usr/bin/python3 -m py_compile examples/piper_dual/collect_data_ros2.py
```
Output:
```text
11 passed in 0.20s
```

### Verification note
`/usr/bin/python3 -m pytest examples/piper_dual/tests/test_collector_state_machine.py -q` still fails during collection in this environment because that interpreter cannot import `h5py`.
