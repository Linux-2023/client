# Piper Dual EEF Inference Client Implementation Plan

> **For agentic workers:** Implement task-by-task with TDD. Existing joint inference behavior must remain unchanged.

**Goal:** Deploy a 20D dual-arm EEF policy through Piper ROS 2 `PosCmd` control topics.

**Architecture:** Reuse the existing policy runtime, synchronized camera/joint/EEF bridge stream, and ROS2 environment lifecycle. Add focused EEF pose/action/observation adapters and an explicit bridge EEF-control mode; keep all existing 14D joint defaults unchanged.

**Tech Stack:** Python 3.11 client, system Python 3.10 ROS 2 Humble bridge, NumPy, rclpy, geometry/sensor messages, `piper_msgs/msg/PosCmd`, pytest.

## Global Constraints

- EEF vectors are 20D `[left_xyz,left_rot6d,left_gripper,right_xyz,right_rot6d,right_gripper]`.
- Raw EEF ROS values are `[x,y,z,roll,pitch,yaw]`, meters/radians.
- rot6d is the first two rotation-matrix columns in column-major order.
- `PosCmd` receives meters, radians, and gripper meters on `/pos_left_cmd` and `/pos_right_cmd`.
- Publishing requires `--no-dry-run --publish-actions`; default is dry-run.
- Existing joint client, converter, bridge default mode, and tests must remain compatible.

---

### Task 1: Add pose conversion primitives

**Files:**
- Create: `examples/piper_dual/eef_pose.py`
- Test: `examples/piper_dual/tests/test_eef_pose.py`

**Interfaces:**
- Produce `rpy_to_rot6d(rpy) -> np.ndarray`, `eef_rpy_to_rot9(pose) -> np.ndarray`, `rot6d_to_rpy(rot6d) -> np.ndarray`, and `eef_rot9_to_poscmd_values(pose9) -> np.ndarray`.
- Inputs use finite numeric NumPy-compatible vectors; outputs are float32 except RPY decode may be float64 internally but returns float32.

- [ ] Write failing tests for identity, +90° yaw, round-trip pose, and degenerate/NaN rot6d rejection.
- [ ] Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest examples/piper_dual/tests/test_eef_pose.py -q`; expect import failure because the module does not exist.
- [ ] Implement explicit `Rz(yaw) @ Ry(pitch) @ Rx(roll)` formulas and Gram-Schmidt rot6d decoding.
- [ ] Run the focused test and require all tests to pass.

### Task 2: Add EEF action and observation adapters

**Files:**
- Create: `examples/piper_dual/eef_action_adapter.py`
- Create: `examples/piper_dual/eef_observation_adapter.py`
- Modify: `examples/piper_dual/ros2_environment.py`
- Test: `examples/piper_dual/tests/test_eef_adapters.py`

**Interfaces:**
- `EefActionAdapter.validate(action) -> np.ndarray` validates 20D finite vectors.
- `EefActionAdapter.to_bridge_request(action) -> dict` returns `{"type":"publish_eef_action","left":[7 values],"right":[7 values]}`.
- `EefObservationAdapter.adapt(message) -> dict` consumes `message["eef"]` with left/right 6D poses and `message["action"]` 14D master joints, returning normal image fields plus 20D `observation.state`.

- [ ] Write failing tests for current EEF+master-gripper observation mapping, action encoding, and invalid dimensions/degenerate rotations.
- [ ] Run the focused adapter tests and verify the expected missing-module failure.
- [ ] Implement adapters and add `eef`/`action` payloads to the existing environment's adapter message without changing default adapters.
- [ ] Run focused adapter tests and existing `test_ros2_environment.py`.

### Task 3: Add EEF PosCmd bridge mode and backend injection

**Files:**
- Modify: `examples/piper_dual/ros2_bridge_process.py`
- Modify: `examples/piper_dual/ros2_backend.py`
- Test: `examples/piper_dual/tests/test_ros2_bridge_codec.py`
- Test: `examples/piper_dual/tests/test_ros2_backend.py`

**Interfaces:**
- Bridge CLI accepts `--eef-control`.
- Add `validate_eef_action_request(request) -> dict[str,list[float]]` for two seven-value `[xyz,rpy,gripper]` vectors.
- `Ros2BackendClient(..., action_adapter=None, eef_control=False)` remains joint-compatible by default; EEF mode appends `--eef-control` and uses the injected adapter.

- [ ] Write failing tests for EEF request validation, PosCmd-like message field construction, and backend argv/action adapter selection.
- [ ] Run bridge/backend focused tests and verify failures are due to absent EEF behavior.
- [ ] Implement piper_msgs import, `/pos_left_cmd` and `/pos_right_cmd` publishers, fixed mode fields, request dispatch, CLI flag, and injectable backend adapter.
- [ ] Run bridge/backend tests with system Python for ROS bridge codec and venv Python for backend tests.

### Task 4: Add independent EEF deployment entrypoint

**Files:**
- Create: `examples/piper_dual/main_dual_eef_ros.py`
- Test: `examples/piper_dual/tests/test_main_dual_eef_ros.py`

**Interfaces:**
- CLI mirrors `main_dual_ros.py` policy/runtime options and adds `--eef-left-topic`, `--eef-right-topic`, `--eef-left-action-topic`, `--eef-right-action-topic`.
- `parse_args()` defaults EEF observation topics to `/puppet/end_pose_left/right`, action topics to `/pos_left_cmd/right_cmd`, `dry_run=True`, `publish_actions=False`.
- `_build_environment()` creates `Ros2BackendClient` in EEF mode and `Ros2DualEnvironment` with EEF adapters.

- [ ] Write failing tests for parser defaults, safety contract, EEF backend construction, and runtime cleanup.
- [ ] Run tests to confirm the new entrypoint is absent.
- [ ] Implement runtime setup by adapting the existing main client without changing it.
- [ ] Run focused CLI tests and `--help` smoke test.

### Task 5: Document and verify

**Files:**
- Modify: `examples/piper_dual/README.md`
- Modify: `docs/superpowers/specs/2026-08-21-piper-dual-eef-inference-design.md`

- [ ] Add source/launch commands, 20D feature contract, dry-run-first example, and explicit ROS topic/type prerequisites.
- [ ] Run `python -m py_compile` on all new/modified Python files.
- [ ] Run focused EEF tests, existing ROS2 tests, and system-Python bridge codec tests.
- [ ] Run `.venv/bin/python examples/piper_dual/main_dual_eef_ros.py --help`.
- [ ] Do not claim hardware validation without connected robot/cameras.
