# Task 5 Three-Stack Report

## Status
DONE

## Scope completed
- Replaced the provisional `_FakeBridgeNode` behavior copy in `examples/piper_dual/tests/test_control_stack_integration.py` with the real `PiperRos2Bridge` constructed through the existing monkeypatched `_construct_test_bridge` fixture from `test_ros2_bridge_codec.py`.
- Added profile contract assertions for all three stacks covering exact control-stack ids, joint names, action topics, speed percent, three image aliases, four joint aliases, EEF aliases, and dry-run `EndpointPlan` behavior.
- Extended graph integration coverage for required topics, forbidden topic identities, selected `sensor_msgs/msg/JointState` and `piper_msgs/msg/PosCmd` action types, exact subscriber counts, and the `piper_direct_sdk_adapter` requirement for the direct SDK profile.
- Extended `examples/piper_dual/tests/test_ros2_end_to_end_mock.py` with ten-frame, per-profile `InProcessBridgeBackend` normalization coverage: ten accepted frames, stacked state/action `(10, 14)`, the same seven sensor aliases, and sync error within 30 ms.
- No production files, docs, `README.md`, or `main_dual.py` were modified.

## RED / GREEN evidence
### Original RED
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest examples/piper_dual/tests/test_control_stack_integration.py -q`
- Prior agent saw collection fail before the parent fixed the local import path:
  - `ModuleNotFoundError: No module named 'rclpy._rclpy_pybind11'`
  - `cv_bridge` import failure with `_ARRAY_API not found`

### Parent-corrected intermediate GREEN
- `/usr/bin/python3 -m pytest -q examples/piper_dual/tests/test_control_stack_integration.py`
- Result before replacing `_FakeBridgeNode`: `7 passed in 0.05s`.

### Final focused GREEN
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest -q examples/piper_dual/tests/test_ros2_contract.py examples/piper_dual/tests/test_ros2_bridge_codec.py examples/piper_dual/tests/test_control_stack_integration.py`
- Result: `49 passed in 0.46s`.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q examples/piper_dual/tests/test_ros2_end_to_end_mock.py`
- Result: `9 passed in 0.27s`.
- Combined rerun result: `49 passed in 0.46s` and `9 passed in 0.27s`.

## Changed paths
- `examples/piper_dual/tests/test_control_stack_integration.py`
- `examples/piper_dual/tests/test_ros2_end_to_end_mock.py`
- `.superpowers/sdd/task-5-three-stack-report.md`

## Self-review
- The dry-run publishing test now crosses the production `PiperRos2Bridge` boundary and asserts a valid `publish_action` request emits exactly one `action_ignored` status while creating zero real publishers and publishing no messages.
- The graph test still uses a small graph-node test double because `validate_profile_graph` accepts a node-like graph API; it no longer copies bridge request/publish behavior.
- The backend normalization test uses the existing `InProcessBridgeBackend`, which exercises `Ros2BackendClient._handle_message`, decoding, `FrameSynchronizer`, and frame queueing without starting ROS/CAN/hardware.
- Focused tests were run with the compatible interpreters only; no broad suites, hardware, ROS processes, CAN, formatters, docs, `main_dual.py`, or README were touched.

## Commit
- Pending at report-write time; final commit SHA is recorded by the parent task result after commit.
