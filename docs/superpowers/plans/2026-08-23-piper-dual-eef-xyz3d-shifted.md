# Piper Dual EEF XYZ+RPY Shifted Converter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone ROS 2 EEF converter that preserves raw `[x,y,z,roll,pitch,yaw]` poses as 14D dual-arm LeRobot state/action vectors while retaining the existing shifted converter's behavior.

**Architecture:** Create one new converter beside `convert_ros2_piper_data_to_lerobot_eef_shifted.py`; dynamically reuse the existing non-EEF ROS 2 converter for shared constants, HDF5 validation, image decoding, file discovery, output handling, and episode loading. The new converter validates the same raw six-value EEF matrices but copies them directly into a 14D `[left_xyzrpy,left_gripper,right_xyzrpy,right_gripper]` layout. Add a focused test module that loads the new script and exercises conversion, validation, dataset population, feature declarations, and CLI parsing without touching the existing rot6d converter.

**Tech Stack:** Python 3.11, NumPy, h5py, pytest, LeRobotDataset, existing Piper dual ROS 2 converter helpers.

## Global Constraints

- Raw EEF values are `[x, y, z, roll, pitch, yaw]`, with meters and radians.
- “xyz+3d” means raw XYZ plus 3D RPY; do not convert to rot6d, normalize angles, or change units.
- `observation.state` and `action` are float32 shape `(14,)` with `[left_xyzrpy,left_gripper,right_xyzrpy,right_gripper]` names.
- For source frame `t`, state uses EEF frame `t`, action uses EEF frame `t+1`, and both use master action grippers at indices 6 and 13 from frame `t`.
- Source frame `N-1` is omitted; episodes require at least two source frames.
- Require `metadata_json.eef.enabled is true` and both raw EEF datasets with exact finite numeric `(N,6)` shape.
- Preserve image/task alignment, HDF5 validation, CLI flags, image/video modes, episode selection, overwrite behavior, cleanup behavior, and all existing scripts.
- Do not run formatters, linters, or project-wide test suites during implementation; run only the focused converter tests and smoke command.
- Existing user modifications are present in the worktree; do not rewrite or revert them.

---

### Task 1: Add failing XYZ+RPY converter tests

**Files:**
- Create: `examples/piper_dual/tests/test_convert_ros2_piper_data_to_lerobot_eef_xyz3d_shifted.py`
- Reference: `examples/piper_dual/tests/test_convert_ros2_piper_data_to_lerobot_eef_shifted.py`

**Interfaces:**
- The test imports `convert_ros2_piper_data_to_lerobot_eef_xyz3d_shifted.py` as `converter`.
- The implementation must provide `EEF_FEATURE_NAMES`, `IMAGE_SENSORS`, `build_shifted_eef_state_action`, `populate_dataset`, `create_empty_dataset`, and `_parse_args`.

- [ ] **Step 1: Write the failing test module.**

Use the existing shifted-converter test fixture shape, but load the new script and assert the raw layout:

```python
SCRIPT = Path(__file__).resolve().parents[1] / "utils" / "convert_ros2_piper_data_to_lerobot_eef_xyz3d_shifted.py"
spec = importlib.util.spec_from_file_location("convert_ros2_piper_data_to_lerobot_eef_xyz3d_shifted", SCRIPT)
assert spec is not None and spec.loader is not None
converter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = converter
spec.loader.exec_module(converter)
```

Create a three-frame HDF5 fixture containing `observations/state` `(N,14)`, `action` `(N,14)`, `observations/timestamp`, both `(N,6)` EEF datasets, three encoded image datasets, and `metadata_json = {"prompt":"Stack the cups", "eef":{"enabled":true}}`. Use distinct nonzero RPY values in each row so a rot6d conversion cannot accidentally satisfy the assertions.

Add these behavior assertions:

```python
def test_build_shifted_eef_state_action_preserves_raw_xyzrpy_and_shifts_pose():
    left = np.array([[1,2,3,0.1,0.2,0.3],[4,5,6,0.4,0.5,0.6],[7,8,9,0.7,0.8,0.9]], dtype=np.float32)
    right = np.array([[10,11,12,-0.1,-0.2,-0.3],[13,14,15,-0.4,-0.5,-0.6],[16,17,18,-0.7,-0.8,-0.9]], dtype=np.float32)
    master = np.zeros((3,14), dtype=np.float32)
    master[:,6] = [60,160,260]
    master[:,13] = [70,170,270]

    state, action = converter.build_shifted_eef_state_action(left, right, master)

    assert state.shape == action.shape == (2, 14)
    assert state.dtype == action.dtype == np.float32
    np.testing.assert_array_equal(state[0], np.r_[left[0], 60, right[0], 70])
    np.testing.assert_array_equal(action[0], np.r_[left[1], 60, right[1], 70])
    np.testing.assert_array_equal(action[1], np.r_[left[2], 160, right[2], 170])
```

Add parameterized rejection coverage for fewer than two frames, EEF width not six, mismatched stream lengths, master action width not the base frame dimension, and non-finite EEF/master values. Add a population test asserting two frames, one saved episode, raw state/action values, current-frame task/images, and no timestamp in the LeRobot frame. Add a metadata-disabled population rejection. Add feature declaration assertions for shape `(14,)`, exact ordered names, and `robot_type == "piper_dual_ros2_eef"`. Add a parser test by monkeypatching `sys.argv` with required `--raw-dir` and `--repo-id`, asserting defaults and `--mode video`, `--episodes 0,2`, and `--overwrite`.

- [ ] **Step 2: Run only the new focused test module.**

Run:

```bash
.venv/bin/python -m pytest examples/piper_dual/tests/test_convert_ros2_piper_data_to_lerobot_eef_xyz3d_shifted.py -q
```

Expected: collection fails because the new converter file does not exist. This confirms the test is red for the intended missing-production-module reason.

---

### Task 2: Implement standalone raw XYZ+RPY converter

**Files:**
- Create: `examples/piper_dual/utils/convert_ros2_piper_data_to_lerobot_eef_xyz3d_shifted.py`
- Reference only: `examples/piper_dual/utils/convert_ros2_piper_data_to_lerobot_eef_shifted.py`

**Interfaces:**
- `build_shifted_eef_state_action(puppet_left_eef: object, puppet_right_eef: object, master_action: object) -> tuple[np.ndarray, np.ndarray]` returns two float32 arrays of shape `(N-1, 14)`.
- `populate_dataset(dataset: object, hdf5_files: list[Path], episodes: list[int] | None = None) -> object` writes selected episodes and returns `dataset`.
- `create_empty_dataset(repo_id: str, output_dir: Path, *, mode: Literal["image","video"] = "image", fps: int = 30) -> object` declares 14D features.
- `convert_ros2_dataset(raw_dir: Path, repo_id: str, *, mode: Literal["image","video"] = "image", episodes: list[int] | None = None, overwrite: bool = False) -> Path` performs validation and conversion.
- `parse_episode_selection(value: str | None) -> list[int] | None` delegates to the base converter.

- [ ] **Step 1: Implement shared-base loading and constants.**

Load `convert_ros2_piper_data_to_lerobot.py` with the same `importlib.util.spec_from_file_location` pattern as the existing EEF converter. Re-export `LeRobotDataset`, `IMAGE_SENSORS`, `IMAGE_HEIGHT`, `IMAGE_WIDTH`, `LEROBOT_HOME`, and `STATE_PATH`. Define:

```python
EEF_DIMENSION = 6
EEF_POSE_DIMENSION = 6
FEATURE_DIMENSION = (EEF_POSE_DIMENSION + 1) * 2
EEF_LEFT_PATH = "/observations/eef/puppet_left"
EEF_RIGHT_PATH = "/observations/eef/puppet_right"
EEF_FEATURE_NAMES = [
    "eef_puppet_left_x", "eef_puppet_left_y", "eef_puppet_left_z",
    "eef_puppet_left_roll", "eef_puppet_left_pitch", "eef_puppet_left_yaw", "left_gripper",
    "eef_puppet_right_x", "eef_puppet_right_y", "eef_puppet_right_z",
    "eef_puppet_right_roll", "eef_puppet_right_pitch", "eef_puppet_right_yaw", "right_gripper",
]
```

- [ ] **Step 2: Implement matrix validation and raw shifted layout.**

Implement `_coerce_eef_matrix` and `_coerce_master_action` with the existing script's numeric, shape, float32, and finite checks. Implement `build_shifted_eef_state_action` without any rotation helper:

```python
left = _coerce_eef_matrix(puppet_left_eef, "puppet_left_eef")
right = _coerce_eef_matrix(puppet_right_eef, "puppet_right_eef")
master = _coerce_master_action(master_action)
if left.shape[0] != right.shape[0] or left.shape[0] != master.shape[0]:
    raise ValueError("puppet EEF streams and master_action must have the same number of frames")
if left.shape[0] < 2:
    raise ValueError("shifted EEF conversion requires at least two source frames per episode")
output_frames = left.shape[0] - 1
state = np.empty((output_frames, FEATURE_DIMENSION), dtype=np.float32)
action = np.empty((output_frames, FEATURE_DIMENSION), dtype=np.float32)
state[:, :6] = left[:-1]
state[:, 6] = master[:-1, 6]
state[:, 7:13] = right[:-1]
state[:, 13] = master[:-1, 13]
action[:, :6] = left[1:]
action[:, 6] = master[:-1, 6]
action[:, 7:13] = right[1:]
action[:, 13] = master[:-1, 13]
return state, action
```

This directly enforces preservation of source XYZ+RPY values and current-gripper semantics.

- [ ] **Step 3: Implement HDF5 loading, population, dataset creation, and CLI by preserving existing behavior.**

Copy the existing EEF converter's `_metadata_eef_enabled`, `load_eef_episode`, `_selected_indices`, `populate_dataset`, `_validate_shiftable_eef_episode`, `convert_ros2_dataset`, `parse_episode_selection`, `_parse_args`, and `main` structure, changing only names/docstrings/output descriptions and the feature dimension/names. Keep `load_eef_episode` requiring both raw paths to match base `/observations/state` frame count. Keep `populate_dataset` assigning `episode.base.images[camera][frame_index]` and `episode.base.prompt` to each frame. Keep conversion validation before creating output and remove the partial output directory in the exception path. The CLI flags remain `--raw-dir`, `--repo-id`, `--mode`, `--episodes`, and `--overwrite`; the final message identifies raw XYZ+RPY conversion.

`create_empty_dataset` must use `FEATURE_DIMENSION == 14`, `[EEF_FEATURE_NAMES]` for both state/action names, and the same camera features, FPS, `robot_type`, `use_videos`, and tolerance as the existing converter.

- [ ] **Step 4: Run the focused tests to verify green.**

Run:

```bash
.venv/bin/python -m pytest examples/piper_dual/tests/test_convert_ros2_piper_data_to_lerobot_eef_xyz3d_shifted.py -q
```

Expected: all new tests pass, including exact raw RPY preservation, 14D alignment, HDF5 validation, dataset feature declarations, and parser behavior.

---

### Task 3: Verify the real CLI smoke surface and repository regression boundary

**Files:**
- Verify: `examples/piper_dual/utils/convert_ros2_piper_data_to_lerobot_eef_xyz3d_shifted.py`
- Verify: `examples/piper_dual/tests/test_convert_ros2_piper_data_to_lerobot_eef_shifted.py`

**Interfaces:**
- The new executable must import without side effects beyond loading the existing base module.
- `--help` must return successfully and describe raw XYZ+RPY conversion.

- [ ] **Step 1: Run CLI help smoke test.**

```bash
.venv/bin/python examples/piper_dual/utils/convert_ros2_piper_data_to_lerobot_eef_xyz3d_shifted.py --help
```

Expected: exit code 0 and help output containing `--raw-dir`, `--repo-id`, `--mode`, `--episodes`, and `--overwrite`.

- [ ] **Step 2: Re-run the existing rot6d converter focused tests.**

```bash
.venv/bin/python -m pytest examples/piper_dual/tests/test_convert_ros2_piper_data_to_lerobot_eef_shifted.py -q
```

Expected: existing tests pass unchanged, demonstrating that the new standalone converter did not alter rot6d behavior.

- [ ] **Step 3: Inspect only the intended new files and diff whitespace.**

```bash
git diff --check -- examples/piper_dual/utils/convert_ros2_piper_data_to_lerobot_eef_xyz3d_shifted.py examples/piper_dual/tests/test_convert_ros2_piper_data_to_lerobot_eef_xyz3d_shifted.py
git status --short -- examples/piper_dual/utils/convert_ros2_piper_data_to_lerobot_eef_xyz3d_shifted.py examples/piper_dual/tests/test_convert_ros2_piper_data_to_lerobot_eef_xyz3d_shifted.py
```

Expected: no whitespace errors and only the two intended new implementation/test files listed (the separately committed design document is already present).

---

## Self-Review

- **Spec coverage:** The new script, exact raw 14D layout, shifted frame alignment, gripper indices, metadata/HDF5 validation, image/task preservation, CLI, output cleanup, and focused verification are covered by Tasks 1–3.
- **Placeholder scan:** No TBD, TODO, deferred implementation, or unspecified “appropriate handling” steps are used.
- **Type consistency:** The test and implementation use the same `FEATURE_DIMENSION`, `EEF_FEATURE_NAMES`, and `build_shifted_eef_state_action` signatures; the population and dataset creation contracts use the same 14D layout.
- **Scope boundary:** No existing converter, collector, runtime, or unrelated user-modified file is changed.
