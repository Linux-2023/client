# Task 1 Report: Python 3.10 ROS 2 Bridge Protocol

## Status
Implemented Task 1 in `/home/agilex/client/.worktrees/piper-dual-ros2-migration`.

## Commit(s)
- Implementation commit subject: `feat: add Python ROS 2 bridge protocol`
- Final hash is reported in the terminal completion status after this report is committed.

## Changed Files
- `examples/piper_dual/ros2_protocol.py`
  - Added ROS-independent JSON-line protocol helpers.
  - `encode_message(message: dict) -> str` emits compact JSON with `separators=(",", ":")`, `ensure_ascii=False`, and newline framing.
  - `decode_message(line: str) -> dict` rejects blank, malformed, and non-object JSON with `ValueError`.
- `examples/piper_dual/ros2_bridge_process.py`
  - Added `/usr/bin/python3` ROS 2 bridge process entrypoint.
  - Imports `rclpy`, `cv_bridge`, `sensor_msgs`, and `std_msgs` only in the ROS bridge module.
  - Subscribes to the required image and joint topics and emits bounded individual sensor events without synchronization.
  - Preserves ROS header timestamps as float seconds.
  - Encodes image bytes at the bridge boundary as base64 ASCII in `jpeg_b64`.
  - Reads stdin requests in a dedicated thread.
  - Serializes JSON-line status/error/sensor output behind a writer lock.
  - Supports `--config`, `--dry-run`, and `--publish-actions`.
  - Validates `publish_action` requests as exactly seven finite values per arm before publishing `JointState` messages to `/joint_left_states` and `/joint_right_states` with `joint0` through `joint6` names and current ROS clock stamps.
- `examples/piper_dual/tests/test_ros2_protocol.py`
  - Added behavior-focused protocol tests for Unicode metadata, compact framing, malformed input rejection, base64 payload preservation, action request round-tripping, and stop serialization.
- `examples/piper_dual/tests/test_ros2_bridge_codec.py`
  - Added bridge-side codec tests for timestamp preservation, bridge-boundary base64 encoding, joint value events, and action vector validation.

## TDD RED Evidence
- First RED attempt: `python -m pytest examples/piper_dual/tests/test_ros2_protocol.py examples/piper_dual/tests/test_ros2_bridge_codec.py -q`
  - Result: RED due to unavailable default-environment pytest: `/home/agilex/miniforge3/bin/python: No module named pytest`.
- Focused RED with system Python after tests existed and before implementation: `/usr/bin/python3 -m pytest examples/piper_dual/tests/test_ros2_protocol.py examples/piper_dual/tests/test_ros2_bridge_codec.py -q`
  - Result: RED during collection because `examples.piper_dual.ros2_protocol` did not exist yet: `ModuleNotFoundError: No module named 'examples.piper_dual.ros2_protocol'`.

## GREEN / Verification Evidence
- Focused tests: `/usr/bin/python3 -m pytest examples/piper_dual/tests/test_ros2_protocol.py examples/piper_dual/tests/test_ros2_bridge_codec.py -q`
  - Result: `15 passed in 0.27s`.
- ROS import smoke test: `/usr/bin/python3 - <<'PY' ... import rclpy, cv_bridge, sensor_msgs, std_msgs, examples.piper_dual.ros2_bridge_process ... PY`
  - Result: `ok rclpy cv_bridge sensor_msgs std_msgs examples.piper_dual.ros2_bridge_process`.
- Protocol ROS-independence review: parsed imports show `examples/piper_dual/ros2_protocol.py` imports only `json` and `typing`.
- System Python compile check: `/usr/bin/python3 -m py_compile examples/piper_dual/ros2_protocol.py examples/piper_dual/ros2_bridge_process.py`
  - Result: exit 0, no output.
- Client-side import compatibility check in the available non-ROS Python runtime: `python - <<'PY' import examples.piper_dual.ros2_protocol as protocol; print(protocol.encode_message({'type': 'ping'}), end='') PY`
  - Result: `{"type":"ping"}`.

## Self-Review
- Confirmed `ros2_protocol.py` contains no ROS imports and is usable independently of the ROS process.
- Confirmed bridge process keeps synchronization out of the bridge and emits individual `sensor` events only.
- Confirmed request handling supports `publish_action`, `set_mode`, `stop`, and `ping`.
- Confirmed action publishing is disabled in `--dry-run` and also unavailable unless `--publish-actions` is explicitly set.
- Confirmed stdout protocol writes are guarded by a lock.
- Confirmed no dependency file was changed, so `rclpy` was not added to Python 3.11 client dependencies.
- Skipped formatters, linters, and project-wide tests per task constraints.

## Concerns
- The available default `python` is a conda Python without pytest and reports Python 3.12. The required ROS import smoke test and focused tests were therefore run with `/usr/bin/python3` (Python 3.10), as required by the brief.
- The ROS bridge implementation imports the requested ROS packages successfully, but hardware/runtime ROS topic behavior was not exercised because the task acceptance only requested focused tests and import smoke testing.

## Review fixes
- Command: `/usr/bin/python3 -m pytest examples/piper_dual/tests/test_ros2_protocol.py examples/piper_dual/tests/test_ros2_bridge_codec.py -q`
  - Observed output: `21 passed in 0.27s`
- Command: `/usr/bin/python3 -m py_compile examples/piper_dual/ros2_protocol.py examples/piper_dual/ros2_bridge_process.py`
  - Observed output: exit 0, no output
