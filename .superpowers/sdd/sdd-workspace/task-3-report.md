# Task 3 Report

Status: completed

## Implemented
- Added `FrameSynchronizer` with bounded per-sensor deques, nearest-neighbor matching, complete-frame-only emission, monotonic reference timestamps, episode clearing, and readable rejection counters.
- Added `Ros2BackendClient` with subprocess lifecycle management, stdout/stderr reader threads, bounded complete-frame queueing, validated action publishing, clear/drain behavior, and idempotent shutdown.
- Added focused tests for synchronization, backend process handling, queue dropping, and error surfacing.

## RED
Initial test run after creating the tests:

```text
E   ModuleNotFoundError: No module named 'frame_synchronizer'
=========================== short test summary info ============================
ERROR  - ModuleNotFoundError: No module named 'frame_synchronizer'
pytest: 1 error in 0.12s
Command exited with code 2
```

Later interface mismatch while tightening the backend contract:

```text
TypeError: Ros2BackendClient.__init__() got an unexpected keyword argument 'complete_frame_queue_size'
```

## GREEN
Focused verification after the implementation settled:

```text
..........................                                               [100%]
26 passed in 5.11s
```

Bytecode verification:

```text
(no output)
```

## Self-review
- `FrameSynchronizer` emits only complete frames, uses nearest-neighbor selection within the configured error bound, rejects unknown sensors and invalid values, evicts old samples, and resets episode boundaries on `clear()`.
- `Ros2BackendClient` launches the bridge subprocess with `stdin=PIPE`, `stdout=PIPE`, `stderr=PIPE`, drains both pipes on dedicated threads, surfaces malformed JSON / bridge errors / EOF / non-zero exits / timeouts, and keeps the complete-frame queue bounded by dropping the oldest frame.
- The client-side Python modules do not import ROS packages.

## Concerns
- `.superpowers/sdd/sdd-workspace/task-1-report.md` was already modified before this task; it was not changed here.

## Review fixes — 2026-08-15

Fixed the three Important Task 3 review findings:
- Kept a reference pending when the nearest sample for a required stream is outside `max_error` but that stream has not advanced beyond `reference_timestamp + max_error`, allowing a later in-window sample to complete the frame.
- Added monotonically nondecreasing `latest_observed_timestamp` state, reset it in `clear()`, and evict all sensor buffers from `latest_observed_timestamp - max_buffer_seconds`, including after old out-of-order pushes.
- Caught `subprocess.TimeoutExpired` when stdout reaches EOF while the bridge remains alive and surfaced `stdout closed before bridge exited` through `next_frame()`.

Regression RED:

```text
$ /usr/bin/python3 -m pytest examples/piper_dual/tests/test_frame_synchronizer.py::test_reference_remains_pending_until_late_in_window_sample_arrives examples/piper_dual/tests/test_frame_synchronizer.py::test_eviction_uses_latest_observed_timestamp_for_out_of_order_samples examples/piper_dual/tests/test_ros2_backend.py::test_stdout_eof_from_live_bridge_surfaces_bridge_failure_not_generic_timeout -q
______ test_reference_remains_pending_until_late_in_window_sample_arrives ______
E       assert 1 == 0
E        +  where 1 = <frame_synchronizer.FrameSynchronizer object at 0x7464d6089840>.rejected_stale
____ test_eviction_uses_latest_observed_timestamp_for_out_of_order_samples _____
E       assert 1 == 0
_ test_stdout_eof_from_live_bridge_surfaces_bridge_failure_not_generic_timeout _
E           TimeoutError: Timed out waiting for a synchronized frame
pytest: 3 failed in 1.10s
Command exited with code 1
```

Regression GREEN:

```text
$ /usr/bin/python3 -m pytest examples/piper_dual/tests/test_frame_synchronizer.py::test_reference_remains_pending_until_late_in_window_sample_arrives examples/piper_dual/tests/test_frame_synchronizer.py::test_eviction_uses_latest_observed_timestamp_for_out_of_order_samples examples/piper_dual/tests/test_ros2_backend.py::test_stdout_eof_from_live_bridge_surfaces_bridge_failure_not_generic_timeout -q
...                                                                      [100%]
3 passed in 2.12s
```

Focused verification:

```text
$ /usr/bin/python3 -m pytest examples/piper_dual/tests/test_frame_synchronizer.py examples/piper_dual/tests/test_ros2_backend.py -q
.............................                                            [100%]
29 passed in 7.23s
```

Bytecode verification:

```text
$ /usr/bin/python3 -m py_compile examples/piper_dual/frame_synchronizer.py examples/piper_dual/ros2_backend.py
(no output)
```

## Review fixes 2 — 2026-08-16

Fixed the three remaining Important Task 3 review findings:
- Added strict raw backend joint sensor decoding before NumPy conversion: `values` must be a list/tuple of exactly seven real JSON numbers, booleans and numeric strings are rejected, and every value must be finite.
- Added a strong `clear_buffers()` episode barrier using the existing bridge `ping` request with a correlated `pong` response. The backend clears local state, sends ping, waits until the reader reaches the matching pong after processing earlier stdout, clears local state again, then releases the reader so only post-barrier samples can form the next frame.
- Added immediate backend error wakeup by enqueueing an internal sentinel when `_set_error()` stores an error, so `next_frame()` callers blocked with long timeouts raise promptly on bridge error, EOF, or nonzero exit.

Regression RED:

```text
$ /usr/bin/python3 -m pytest examples/piper_dual/tests/test_ros2_backend.py::test_raw_joint_sensor_values_are_validated_before_numpy_coercion examples/piper_dual/tests/test_ros2_backend.py::test_clear_buffers_uses_ping_barrier_to_exclude_pre_barrier_stdout examples/piper_dual/tests/test_ros2_backend.py::test_next_frame_wakes_promptly_for_asynchronous_backend_errors -q
FAILED examples/piper_dual/tests/test_ros2_backend.py::test_raw_joint_sensor_values_are_validated_before_numpy_coercion[values0-real JSON numbers]
FAILED examples/piper_dual/tests/test_ros2_backend.py::test_raw_joint_sensor_values_are_validated_before_numpy_coercion[values1-real JSON numbers]
FAILED examples/piper_dual/tests/test_ros2_backend.py::test_raw_joint_sensor_values_are_validated_before_numpy_coercion[values2-exactly seven]
FAILED examples/piper_dual/tests/test_ros2_backend.py::test_raw_joint_sensor_values_are_validated_before_numpy_coercion[values3-finite]
FAILED examples/piper_dual/tests/test_ros2_backend.py::test_raw_joint_sensor_values_are_validated_before_numpy_coercion[values4-list or tuple]
FAILED examples/piper_dual/tests/test_ros2_backend.py::test_clear_buffers_uses_ping_barrier_to_exclude_pre_barrier_stdout
FAILED examples/piper_dual/tests/test_ros2_backend.py::test_next_frame_wakes_promptly_for_asynchronous_backend_errors[async bridge error]
FAILED examples/piper_dual/tests/test_ros2_backend.py::test_next_frame_wakes_promptly_for_asynchronous_backend_errors[bridge exited with status 6]
pytest: 8 failed in 4.15s
Command exited with code 1
```

Regression GREEN:

```text
$ /usr/bin/python3 -m pytest examples/piper_dual/tests/test_ros2_backend.py::test_raw_joint_sensor_values_are_validated_before_numpy_coercion examples/piper_dual/tests/test_ros2_backend.py::test_clear_buffers_uses_ping_barrier_to_exclude_pre_barrier_stdout examples/piper_dual/tests/test_ros2_backend.py::test_next_frame_wakes_promptly_for_asynchronous_backend_errors -q
........                                                                 [100%]
8 passed in 0.53s
```

Focused verification:

```text
$ /usr/bin/python3 -m pytest examples/piper_dual/tests/test_frame_synchronizer.py examples/piper_dual/tests/test_ros2_backend.py -q
.....................................                                    [100%]
37 passed in 3.79s
```

Bytecode verification:

```text
$ /usr/bin/python3 -m py_compile examples/piper_dual/frame_synchronizer.py examples/piper_dual/ros2_backend.py /home/agilex/client/.worktrees/piper-dual-ros2-migration/examples/piper_dual/ros2_bridge_process.py
(no output)
```
