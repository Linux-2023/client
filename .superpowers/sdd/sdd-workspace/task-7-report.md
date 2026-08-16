# Task 7 report

## Status
Implemented safe ROS 2 backend integration for the dual-arm Piper environment in the isolated worktree.

## What changed
- Added `examples/piper_dual/ros2_environment.py` as a safe ROS 2 environment wrapper.
- Updated `examples/piper_dual/env_dual.py` to select `sdk` or `ros2` without importing SDK hardware modules for the ROS 2 path.
- Rewrote `examples/piper_dual/main_dual.py` to expose `--backend`, `--bridge-python`, `--ros2-config`, `--dry-run`, `--publish-actions`, and `--max-action-delta`.
- Updated `examples/piper_dual/README.md` with ROS 2 dry-run and hardware safety instructions.
- Added focused tests in `examples/piper_dual/tests/test_ros2_environment.py` and extended `examples/piper_dual/tests/test_main_dual.py`.

## Safety behavior
- SDK remains the default backend.
- ROS 2 defaults to dry-run.
- `--publish-actions` is rejected unless `--backend ros2` and `--no-dry-run` are both explicit.
- ROS 2 path does not instantiate SDK hardware modules.
- Invalid actions, stale observations, bridge exit conditions, missing cameras, per-step delta violations, and backend publish failures all call the common fatal-fault path, which permanently latches action publication off on the current `Ros2DualEnvironment` instance.
- Ordinary max-step completion still resets normally; restoring action publication after a fatal fault requires a new environment/backend instance.

## Verification
- RED: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_ros2_environment.py::test_fatal_fault_latches_action_publishing_until_new_environment -q` failed as expected before the production edit because `reset()` re-enabled publishing on the same instance.
- GREEN: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_ros2_environment.py::test_fatal_fault_latches_action_publishing_until_new_environment examples/piper_dual/tests/test_ros2_environment.py::test_max_step_completion_remains_resettable -q` -> `7 passed in 0.10s`.
- Focused Task 7 suites plus compile: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_ros2_environment.py examples/piper_dual/tests/test_main_dual.py -q && python -m py_compile examples/piper_dual/main_dual.py examples/piper_dual/env_dual.py examples/piper_dual/ros2_environment.py examples/piper_dual/tests/test_main_dual.py examples/piper_dual/tests/test_ros2_environment.py` -> `32 passed in 0.16s`; `py_compile` produced no output and exited successfully.

## Notes
- No physical action publishing was executed.
- The ROS 2 bridge remains dry-run by default unless explicitly enabled with the required flags.

## Task 7 fatal-fault latch regression
- Design decision: keep the safety latch on the `Ros2DualEnvironment` instance. `_fail_episode()` sets the lock, `reset()` preserves it, and a new environment/backend instance is required to publish again after a fatal fault.
- RED:
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_ros2_environment.py::test_fatal_fault_latches_action_publishing_until_new_environment -q
FAILED examples/piper_dual/tests/test_ros2_environment.py::test_fatal_fault_latches_action_publishing_until_new_environment[stale-observation] - assert 1 == 0
FAILED examples/piper_dual/tests/test_ros2_environment.py::test_fatal_fault_latches_action_publishing_until_new_environment[bridge-failure] - assert 1 == 0
FAILED examples/piper_dual/tests/test_ros2_environment.py::test_fatal_fault_latches_action_publishing_until_new_environment[missing-camera] - assert 1 == 0
FAILED examples/piper_dual/tests/test_ros2_environment.py::test_fatal_fault_latches_action_publishing_until_new_environment[invalid-action] - assert 1 == 0
FAILED examples/piper_dual/tests/test_ros2_environment.py::test_fatal_fault_latches_action_publishing_until_new_environment[action-delta-violation] - assert 2 == 1
FAILED examples/piper_dual/tests/test_ros2_environment.py::test_fatal_fault_latches_action_publishing_until_new_environment[backend-publish-failure] - assert 1 == 0
pytest: 6 failed in 0.16s
```
- GREEN:
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_ros2_environment.py::test_fatal_fault_latches_action_publishing_until_new_environment examples/piper_dual/tests/test_ros2_environment.py::test_max_step_completion_remains_resettable -q
.......                                                                  [100%]
7 passed in 0.10s
```