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

## Task 7 bridge YAML config compatibility
- Root cause: the shipped `ros2_piper_dual.yaml` is YAML, while `_load_config()` previously called `json.load()` for every extension.
- Decision: parse `.yaml`/`.yml` with `yaml.safe_load()` and `.json` with `json.load()`. Unsupported extensions, malformed documents, missing files, and non-mapping roots fail descriptively; unsafe YAML constructors are not enabled.
- RED:
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest examples/piper_dual/tests/test_ros2_bridge_codec.py::test_load_config_accepts_shipped_default_yaml_bridge_contract -q
FAILED: _load_config() attempted JSON decoding of the shipped YAML and raised JSONDecodeError.
```
- GREEN config regressions: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest examples/piper_dual/tests/test_ros2_bridge_codec.py -q` -> `13 passed in 0.28s`.
- Bridge protocol/process verification: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest examples/piper_dual/tests/test_ros2_protocol.py examples/piper_dual/tests/test_ros2_bridge_codec.py examples/piper_dual/tests/test_ros2_backend.py -q` -> `46 passed in 3.99s`.
- Task 7 focused verification: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_ros2_environment.py examples/piper_dual/tests/test_main_dual.py -q` -> `40 passed in 0.16s`.
- Compile verification: `/usr/bin/python3 -m py_compile examples/piper_dual/ros2_bridge_process.py examples/piper_dual/tests/test_ros2_bridge_codec.py` and the Task 7 modules completed with no output.
- No-hardware config smoke: `/usr/bin/python3` loaded `examples/piper_dual/ros2_piper_dual.yaml` through `_load_config()` and printed `piper_dual_ros2_v1 3` (schema version and camera count); no ROS node was instantiated.
- No hardware or physical action publishing was executed.

## Task 7 action aliasing snapshot regression
- Root cause: `ActionAdapter.validate()` intentionally preserves zero-copy behavior for already-float32 ndarrays, but `Ros2DualEnvironment.apply_action()` was forwarding that alias directly into delta comparison, backend publication, and `_previous_action` storage.
- Decision: snapshot the validated action immediately inside `apply_action()` with `np.array(validated, dtype=np.float32, copy=True)` before any delta check, publication, or state storage. `ActionAdapter` remains unchanged.
- Ownership invariant: after `apply_action()` returns, caller-owned mutation cannot affect the environment's `_previous_action` or any backend-held published action snapshot, even when the backend retains references instead of copying inputs.
- RED:
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_ros2_environment.py::test_apply_action_snapshots_float32_caller_arrays_before_publish_and_delta_check -q
FAILED examples/piper_dual/tests/test_ros2_environment.py::test_apply_action_snapshots_float32_caller_arrays_before_publish_and_delta_check - Failed: DID NOT RAISE ValueError
pytest: 1 failed in 0.14s
```
- Additional RED evidence before reordering the assertions showed the backend-retained publish argument was the caller alias: the same regression failed with `Arrays are not equal`, `Mismatched elements: 14 / 14 (100%)`, `ACTUAL: array([0.5, ...], dtype=float32)`, and `DESIRED: array([0., ...], dtype=float32)`.
- GREEN regression: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_ros2_environment.py::test_apply_action_snapshots_float32_caller_arrays_before_publish_and_delta_check -q` -> `1 passed in 0.09s`.
- GREEN Task 7 focused suites: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_ros2_environment.py examples/piper_dual/tests/test_main_dual.py -q` -> `41 passed in 0.17s`.
- GREEN bridge suites: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest examples/piper_dual/tests/test_ros2_protocol.py examples/piper_dual/tests/test_ros2_bridge_codec.py examples/piper_dual/tests/test_ros2_backend.py -q` -> `46 passed in 4.01s`.
- GREEN compile check: `python -m py_compile examples/piper_dual/main_dual.py examples/piper_dual/env_dual.py examples/piper_dual/ros2_environment.py examples/piper_dual/tests/test_main_dual.py examples/piper_dual/tests/test_ros2_environment.py && /usr/bin/python3 -m py_compile examples/piper_dual/ros2_bridge_process.py examples/piper_dual/tests/test_ros2_protocol.py examples/piper_dual/tests/test_ros2_bridge_codec.py examples/piper_dual/tests/test_ros2_backend.py` -> no output.
- No hardware or physical action publishing was executed.

## Task 7 backend publish snapshot ownership regression
- Root cause: the first action-aliasing fix copied the caller-owned action into a local `validated` snapshot, but still passed that same ndarray to `backend.publish_action()` and stored the same object as `_previous_action`. A backend that retains and mutates its publish argument can therefore corrupt the environment-owned previous-action baseline and make later delta checks compare against backend-mutated state.
- Decision: keep the caller boundary snapshot before delta checking, then hand `backend.publish_action()` its own `validated.copy()` and store a separate `validated.copy()` in `_previous_action` only after successful publication. Dry-run follows the same state-storage path, so dry-run `_previous_action` is also independently owned.
- Ownership invariant: caller input, backend publish argument, and `_previous_action` are three distinct float32 snapshots for a successful publishing action. Backend in-place mutation after `publish_action()` cannot affect `_previous_action`; the next oversized action is rejected and no second publish occurs.
- RED:
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_ros2_environment.py::test_backend_publish_mutation_cannot_corrupt_previous_action_delta_baseline -q
F                                                                        [100%]
=================================== FAILURES ===================================
_ test_backend_publish_mutation_cannot_corrupt_previous_action_delta_baseline __

    def test_backend_publish_mutation_cannot_corrupt_previous_action_delta_baseline() -> None:
        backend = MutatingPublishBackend()
        env = Ros2DualEnvironment(backend=backend, dry_run=False, publish_actions=True, max_action_delta=0.4)
        first_action = np.zeros(14, dtype=np.float32)
    
        env.reset()
        env.apply_action({"actions": first_action})
    
        assert backend.publish_action_calls[0] is not first_action
        np.testing.assert_array_equal(backend.publish_action_calls[0], np.full(14, 0.5, dtype=np.float32))
    
>       with pytest.raises(ValueError, match="max_action_delta"):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE ValueError

examples/piper_dual/tests/test_ros2_environment.py:414: Failed
=========================== short test summary info ============================
FAILED examples/piper_dual/tests/test_ros2_environment.py::test_backend_publish_mutation_cannot_corrupt_previous_action_delta_baseline - Failed: DID NOT RAISE ValueError
1 failed in 0.12s
```
- GREEN regression:
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_ros2_environment.py::test_backend_publish_mutation_cannot_corrupt_previous_action_delta_baseline -q
.                                                                        [100%]
1 passed in 0.10s
```
- GREEN Task 7 environment/main suites:
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_ros2_environment.py examples/piper_dual/tests/test_main_dual.py -q
..........................................                               [100%]
42 passed in 0.17s
```
- GREEN bridge protocol/codec/backend suites including YAML config:
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest examples/piper_dual/tests/test_ros2_protocol.py examples/piper_dual/tests/test_ros2_bridge_codec.py examples/piper_dual/tests/test_ros2_backend.py -q
..............................................                           [100%]
46 passed in 4.02s
```
- GREEN compile check:
```text
python -m py_compile examples/piper_dual/main_dual.py examples/piper_dual/env_dual.py examples/piper_dual/ros2_environment.py examples/piper_dual/tests/test_main_dual.py examples/piper_dual/tests/test_ros2_environment.py && /usr/bin/python3 -m py_compile examples/piper_dual/ros2_bridge_process.py examples/piper_dual/tests/test_ros2_protocol.py examples/piper_dual/tests/test_ros2_bridge_codec.py examples/piper_dual/tests/test_ros2_backend.py
(no output)
```
- No hardware or physical action publishing was executed.

## Task 7 post-fatal action rejection regression
- Root cause: `_fail_episode()` set `_publish_actions_locked` and `_done`, but `apply_action()` did not check either flag before validating the next action, updating `_previous_action`, incrementing `_step_count`, and only suppressing publication.
- Decision: `apply_action()` now calls `_ensure_open()` first, then rejects a latched fatal fault with a descriptive `RuntimeError` telling the caller to create a new environment before any validation, publication, or action-state mutation. Non-fatal completed episodes are also rejected until `reset()` so ordinary max-step completion remains resettable without silently processing actions while `_done` is true.
- Post-fault invariant: after any fatal fault, subsequent `apply_action()` calls on the same `Ros2DualEnvironment` instance raise before action validation/publication/storage and leave `_previous_action`, `_step_count`, and backend `publish_action_calls` unchanged, including after `reset()`; physical action recovery requires a new environment/backend instance.
- RED:
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_ros2_environment.py::test_fatal_fault_rejects_future_actions_and_survives_reset examples/piper_dual/tests/test_ros2_environment.py::test_invalid_action_marks_episode_complete_and_rejects_future_actions examples/piper_dual/tests/test_ros2_environment.py::test_max_step_completion_remains_resettable -q
FAILED examples/piper_dual/tests/test_ros2_environment.py::test_fatal_fault_rejects_future_actions_and_survives_reset[stale-observation] - Failed: DID NOT RAISE RuntimeError
FAILED examples/piper_dual/tests/test_ros2_environment.py::test_fatal_fault_rejects_future_actions_and_survives_reset[bridge-failure] - Failed: DID NOT RAISE RuntimeError
FAILED examples/piper_dual/tests/test_ros2_environment.py::test_fatal_fault_rejects_future_actions_and_survives_reset[missing-camera] - Failed: DID NOT RAISE RuntimeError
FAILED examples/piper_dual/tests/test_ros2_environment.py::test_fatal_fault_rejects_future_actions_and_survives_reset[invalid-action] - Failed: DID NOT RAISE RuntimeError
FAILED examples/piper_dual/tests/test_ros2_environment.py::test_fatal_fault_rejects_future_actions_and_survives_reset[action-delta-violation] - Failed: DID NOT RAISE RuntimeError
FAILED examples/piper_dual/tests/test_ros2_environment.py::test_fatal_fault_rejects_future_actions_and_survives_reset[backend-publish-failure] - Failed: DID NOT RAISE RuntimeError
FAILED examples/piper_dual/tests/test_ros2_environment.py::test_invalid_action_marks_episode_complete_and_rejects_future_actions - Failed: DID NOT RAISE RuntimeError
FAILED examples/piper_dual/tests/test_ros2_environment.py::test_max_step_completion_remains_resettable - Failed: DID NOT RAISE RuntimeError
pytest: 8 failed in 0.20s
```
- GREEN targeted regressions:
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_ros2_environment.py::test_fatal_fault_rejects_future_actions_and_survives_reset examples/piper_dual/tests/test_ros2_environment.py::test_invalid_action_marks_episode_complete_and_rejects_future_actions examples/piper_dual/tests/test_ros2_environment.py::test_max_step_completion_remains_resettable -q
........                                                                 [100%]
8 passed in 0.11s
```
- GREEN Task 7 environment/main suites:
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest examples/piper_dual/tests/test_ros2_environment.py examples/piper_dual/tests/test_main_dual.py -q
..........................................                               [100%]
42 passed in 0.21s
```
- GREEN bridge protocol/codec/backend suites including YAML config:
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest examples/piper_dual/tests/test_ros2_protocol.py examples/piper_dual/tests/test_ros2_bridge_codec.py examples/piper_dual/tests/test_ros2_backend.py -q
..............................................                           [100%]
46 passed in 3.99s
```
- GREEN compile check:
```text
python -m py_compile examples/piper_dual/main_dual.py examples/piper_dual/env_dual.py examples/piper_dual/ros2_environment.py examples/piper_dual/tests/test_main_dual.py examples/piper_dual/tests/test_ros2_environment.py && /usr/bin/python3 -m py_compile examples/piper_dual/ros2_bridge_process.py examples/piper_dual/tests/test_ros2_protocol.py examples/piper_dual/tests/test_ros2_bridge_codec.py examples/piper_dual/tests/test_ros2_backend.py
(no output)
```
- No hardware or physical action publishing was executed.