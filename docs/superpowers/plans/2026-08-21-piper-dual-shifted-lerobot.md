# Piper Dual Shifted LeRobot Converter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans (recommended). Steps use checkbox syntax for tracking.

**Goal:** Add an independent converter that maps puppet joint state/action with a one-frame joint-only shift and current-frame master grippers.

**Architecture:** Keep `convert_ros2_piper_data_to_lerobot.py` unchanged. Add a pure NumPy mapping function in `convert_ros2_piper_data_to_lerobot_shifted.py`, reuse the validated ROS2 episode loader and LeRobot dataset setup, and populate only `N - 1` frames per source episode. The output frame at source index `t` uses source images/task/timestamp `t`.

**Tech Stack:** Python 3.11, NumPy, h5py, OpenCV, LeRobotDataset, pytest.

## Global Constraints

- Input is `piper_dual_ros2_v1` with 14-value left-then-right vectors, six joints plus one gripper per arm.
- State joints come from `/observations/state[t]`; action joints come from `/observations/state[t+1]`.
- Both output grippers come from `/action[t]` indices 6 and 13; grippers are not shifted.
- Drop each episode's final source frame; reject episodes shorter than two frames.
- Preserve image/task from source frame `t`, let LeRobot rebuild exact `frame_index / fps` timestamps, and preserve the existing 14D LeRobot feature names.
- Do not modify the existing converter or the ROS2 raw-data schema.
- Skip formatters, linters, and project-wide test suites until the final focused verification.

---

### Task 1: Specify and prove temporal mapping

**Files:**
- Create: `examples/piper_dual/tests/test_convert_ros2_piper_data_to_lerobot_shifted.py`

**Interfaces:**
- Consumes `converter.build_shifted_state_action(state, master_action)`.
- Produces `(state_out, action_out)` with shape `(N - 1, 14)`.

- [ ] Write tests using three deterministic frames. Assert state joints are current puppet joints, action joints are next-frame puppet joints, and each side's state/action gripper equals the current master action gripper.
- [ ] Assert one-frame input raises `ValueError` and mismatched/non-finite arrays are rejected.
- [ ] Run `.venv/bin/pytest examples/piper_dual/tests/test_convert_ros2_piper_data_to_lerobot_shifted.py -q`; expected failure because the new module/function does not exist.

### Task 2: Implement the independent converter

**Files:**
- Create: `examples/piper_dual/utils/convert_ros2_piper_data_to_lerobot_shifted.py`

**Interfaces:**
- `build_shifted_state_action(state: np.ndarray, master_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
- `populate_dataset(dataset, hdf5_files, episodes=None)`
- `convert_ros2_dataset(raw_dir, repo_id, mode="image", episodes=None, overwrite=False) -> Path`
- CLI flags: `--raw-dir`, `--repo-id`, `--mode`, `--episodes`, `--overwrite`.

- [ ] Import shared constants/loaders from the existing converter without changing that file.
- [ ] Validate the two input arrays as finite numeric `(N, 14)` arrays, require `N >= 2`, compute output arrays exactly as:
  `state_out = [puppet[:, :6], master[:, 6], puppet[:, 7:13], master[:, 13]]` for indices `0..N-2`; `action_out` uses puppet indices `1..N-1` with the same current master grippers.
- [ ] Load each validated source episode, build the transformed arrays, and add frames for indices `0..N-2`; use source image/task/timestamp at each index and let LeRobot rebuild relative timestamps.
- [ ] Create the same 14D LeRobot features and robot type as the existing converter, validate selected episode indices, remove partial output on failure, and reject any source episode with fewer than two frames.
- [ ] Keep module import usable in the repository's direct-script and pytest invocation styles.

### Task 3: Verify focused behavior and a real episode

**Files:**
- Modify only if needed: `examples/piper_dual/tests/test_convert_ros2_piper_data_to_lerobot_shifted.py`

- [ ] Run the focused mapping/converter tests and the existing converter regression module.
- [ ] Run the new script against `episode_000000.hdf5` via `--episodes 0`, with a temporary `HF_LEROBOT_HOME`, `--mode image`, and `--overwrite`.
- [ ] Inspect the produced LeRobot metadata and one episode's tensors: output frame count is 799, state/action are float32 shape `(799, 14)`, first output state has source puppet joints at frame 0 plus source master grippers at frame 0, first output action has source puppet joints at frame 1 plus the same frame-0 master grippers, and no frame uses source index 799.
- [ ] Run the existing focused converter test module to prove the old converter remains compatible.
