# Task 8 three-stack safety ordering report

## Status
Minimal live-action readiness ordering fix completed for `ros2_bridge_process.py`.

## Prior evidence
- Prior client result supplied by parent: `112 passed in 4.36s`.
- Prior system RED supplied in `artifact://1786`: `1 failed, 82 passed in 0.57s`.

## Root cause
`PiperRos2Bridge._ensure_publishers_ready()` validated the selected-profile ROS graph before checking the status latch. In the pure safety regression, the fake node intentionally has no real `rclpy` handle, so graph introspection raised `AttributeError: 'PiperRos2Bridge' object has no attribute '_Node__node'` before the already-latched hardware fault (`left err_code=7`) could block publishing. That masked the safety fault and latched the wrong permanent disable reason.

## Change
Changed only `examples/piper_dual/ros2_bridge_process.py` so live publisher readiness checks call `_assert_live_joint_ready()` before graph validation. The existing post-graph `_assert_live_joint_ready()` remains in place, so healthy live paths still run the selected-profile graph/type/subscriber/direct-node guard before creating any publisher, then recheck status immediately before publisher creation. Existing per-action publisher readiness remains the entry point for both live joint and EEF action requests.

## Changed paths
- `examples/piper_dual/ros2_bridge_process.py`
- `.superpowers/sdd/task-8-three-stack-report.md`

## RED reproduction
Initial default interpreter attempt was blocked by the known ROS/pytest plugin environment (`ModuleNotFoundError: No module named 'lark'`) and was not counted as task evidence. The correct system-Python focused RED reproduced the artifact failure:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest examples/piper_dual/tests/test_ros2_safety.py::test_bridge_live_joint_and_eef_actions_reject_before_publishing_after_fault -q
FAILED examples/piper_dual/tests/test_ros2_safety.py::test_bridge_live_joint_and_eef_actions_reject_before_publishing_after_fault
AssertionError: Regex pattern 'left err_code=7' does not match "Action publishing permanently disabled: 'PiperRos2Bridge' object has no attribute '_Node__node'".
pytest: 1 failed in 0.34s
```

## Focused GREEN
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest examples/piper_dual/tests/test_ros2_safety.py::test_bridge_live_joint_and_eef_actions_reject_before_publishing_after_fault -q
.                                                                        [100%]
1 passed in 0.30s
```

## Full ROS-message Task 2/8 GREEN
```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest -q examples/piper_dual/tests/test_ros2_protocol.py examples/piper_dual/tests/test_ros2_backend.py examples/piper_dual/tests/test_ros2_contract.py examples/piper_dual/tests/test_ros2_bridge_codec.py examples/piper_dual/tests/test_ros2_safety.py
........................................................................ [ 77%]
.....................                                                    [100%]
93 passed in 4.28s
```

## Self-review
- Diff is the minimal production ordering change: one pre-graph `_assert_live_joint_ready()` call.
- The graph/type/subscriber guard is not weakened or mocked; it still runs before any publisher is created for otherwise-ready live hardware.
- The existing post-graph status recheck remains, preserving a final readiness check after graph validation and before publisher creation.
- No tests were changed because the existing failing regression fully proved the ordering bug.
- No ROS process, CAN, hardware, formatter, docs, or broad suite was run.

## Concerns
- No physical hardware or live ROS graph was exercised by design; verification is limited to system-Python pure ROS-message tests.
- The worktree has unrelated pre-existing unstaged/untracked changes outside this task; only the changed source file and this report should be staged/committed.

## Commit
Pending.
