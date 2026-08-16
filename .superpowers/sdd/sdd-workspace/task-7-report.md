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

## Task 7 remaining safety hardening
- Design decision: validate `max_action_delta` immediately after conversion with `np.isfinite()` and the existing non-negative check, storing only `None` or a finite non-negative scalar. This rejects NaN, +inf, and -inf before any environment reset or action publication can occur while preserving `0.0` as a valid strict no-change limit.
- Design decision: keep `_ensure_open()` outside the reset bridge-fault handler so a deliberate reset on a closed environment remains the existing closed-environment error. Only backend `start()` and `clear_buffers()` are wrapped; failures use `_fail_episode()` to permanently latch publication off on the current environment instance, then re-raise the original backend exception.
- README update: clarified that reset-stage `start()` / `clear_buffers()` failures are fatal lockout events for the current environment instance.
- RED:
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_ros2_environment.py::test_constructor_rejects_nonfinite_max_action_delta examples/piper_dual/tests/test_ros2_environment.py::test_reset_stage_bridge_failure_latches_action_publishing_until_new_environment -q
FAILED examples/piper_dual/tests/test_ros2_environment.py::test_constructor_rejects_nonfinite_max_action_delta[nan] - Failed: DID NOT RAISE ValueError
FAILED examples/piper_dual/tests/test_ros2_environment.py::test_constructor_rejects_nonfinite_max_action_delta[inf] - Failed: DID NOT RAISE ValueError
FAILED examples/piper_dual/tests/test_ros2_environment.py::test_constructor_rejects_nonfinite_max_action_delta[-inf] - AssertionError: Regex pattern did not match.
  Expected regex: 'finite non-negative'
  Actual message: 'max_action_delta must be non-negative'
FAILED examples/piper_dual/tests/test_ros2_environment.py::test_reset_stage_bridge_failure_latches_action_publishing_until_new_environment[start] - assert [array([0., 0...type=float32)] == []
FAILED examples/piper_dual/tests/test_ros2_environment.py::test_reset_stage_bridge_failure_latches_action_publishing_until_new_environment[clear-buffers] - assert [array([0., 0...type=float32)] == []
pytest: 5 failed in 0.14s
```
- GREEN targeted regressions:
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_ros2_environment.py::test_constructor_rejects_nonfinite_max_action_delta examples/piper_dual/tests/test_ros2_environment.py::test_constructor_accepts_zero_max_action_delta examples/piper_dual/tests/test_ros2_environment.py::test_reset_stage_bridge_failure_latches_action_publishing_until_new_environment -q
......                                                                   [100%]
6 passed in 0.09s
```
- Focused Task 7 suites:
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_ros2_environment.py examples/piper_dual/tests/test_main_dual.py -q
......................................                                   [100%]
38 passed in 0.18s
```
- Compile check:
```text
python -m py_compile examples/piper_dual/main_dual.py examples/piper_dual/env_dual.py examples/piper_dual/ros2_environment.py examples/piper_dual/tests/test_main_dual.py examples/piper_dual/tests/test_ros2_environment.py
(no output)
```
- No hardware or physical publishing commands were run.

## Task 7 CLI compatibility
- Decision: keep dashed spellings canonical and add explicit underscore aliases only for the prior PI05 options already documented or confirmed by focused search: `left_can_port`, `right_can_port`, `high_camera_id`, `left_wrist_camera_id`, `right_wrist_camera_id`, plus `action_horizon`, `actions_during_latency`, `use_rtc`, `tele_mode`, and `record_mode`.
- No README change was needed because the documented underscore command now parses unchanged.
- RED:
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_main_dual.py::test_parse_args_accepts_documented_underscore_pi05_options -q
FAILED examples/piper_dual/tests/test_main_dual.py::test_parse_args_accepts_documented_underscore_pi05_options - SystemExit: 2
__main__.py: error: unrecognized arguments: --left_can_port can_left_doc --right_can_port can_right_doc --high_camera_id 148522073709 --left_wrist_camera_id 6 --right_wrist_camera_id 8
pytest: 1 failed in 0.21s
```
- GREEN:
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_main_dual.py::test_parse_args_accepts_documented_underscore_pi05_options examples/piper_dual/tests/test_main_dual.py::test_parse_args_preserves_dashed_pi05_options -q
2 passed in 0.10s
```
- Full Task 7 focused suites:
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_ros2_environment.py examples/piper_dual/tests/test_main_dual.py -q
40 passed in 0.20s
```
- Help check: `python examples/piper_dual/main_dual.py --help` remained readable and showed both dashed canonical flags and the preserved underscore aliases.
- Compile check: `python -m py_compile examples/piper_dual/main_dual.py examples/piper_dual/env_dual.py examples/piper_dual/ros2_environment.py examples/piper_dual/tests/test_main_dual.py examples/piper_dual/tests/test_ros2_environment.py` produced no output.