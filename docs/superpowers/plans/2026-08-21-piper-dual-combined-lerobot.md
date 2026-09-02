# Piper Dual Combined LeRobot Converter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans (recommended). Steps use checkbox syntax for tracking.

**Goal:** Add an independent converter that combines 50 ordinary Piper dual ROS2 episodes and 50 EEF-marked episodes into one 100-episode LeRobot dataset while ignoring EEF data.

**Architecture:** Create `convert_ros2_piper_data_to_lerobot_combined.py` beside the existing shifted converter. Load the shifted converter as a module and reuse its exact state/action transformation and base HDF5/LeRobot helpers. The new script only owns source-directory discovery, ordinary-then-EEF concatenation, combined episode-index selection, and CLI/output cleanup. No existing converter or raw HDF5 file is modified.

**Tech Stack:** Python 3.11, NumPy, h5py, OpenCV, LeRobotDataset, pytest.

## Global Constraints

- Ordinary source directory: `/home/agilex/piper_dual_dataset/stack_cups`.
- EEF source directory: `/home/agilex/piper_dual_dataset/stack_cups_eef`.
- Each directory contributes its sorted `episode_*.hdf5` files; ordinary files precede EEF files.
- The combined list is renumbered by LeRobot as output episodes 0–99; `--episodes` addresses these combined indices.
- Ignore `/observations/eef` and EEF timing metadata; use only the shared state, action, prompt, timestamps, and three image streams.
- Preserve the existing shifted mapping: state uses current puppet joints, action uses next-frame puppet joints, and both grippers use current master action grippers.
- Each source episode emits N−1 frames and rejects fewer than two source frames.
- Preserve existing 14D float32 LeRobot features and image/video modes.
- On conversion failure, delete the partial local output directory; do not alter either input directory.
- Skip formatters, linters, and project-wide test suites; run only focused tests and the real conversion verification.

---

### Task 1: Define combined source ordering and selection

**Files:**
- Create: `examples/piper_dual/tests/test_convert_ros2_piper_data_to_lerobot_combined.py`

**Interfaces:**
- Consumes `converter.discover_combined_hdf5_files(raw_dir, raw_dir_eef)`.
- Consumes `converter.select_combined_files(files, episodes)`.
- Produces a list with ordinary files first, EEF files second, and validated combined-index selection.

- [ ] Write tests with shuffled temporary filenames proving ordinary files precede EEF files and each group is sorted by episode filename.
- [ ] Write tests proving `None` selects every combined file, valid indices preserve requested order, and negative/out-of-range indices raise `IndexError`.
- [ ] Write tests proving a missing or empty source directory raises a clear `ValueError`.
- [ ] Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest examples/piper_dual/tests/test_convert_ros2_piper_data_to_lerobot_combined.py -q`; expected collection failure because the new module does not exist.

### Task 2: Implement the independent combined converter

**Files:**
- Create: `examples/piper_dual/utils/convert_ros2_piper_data_to_lerobot_combined.py`

**Interfaces:**
- `discover_combined_hdf5_files(raw_dir: Path, raw_dir_eef: Path) -> list[Path]`
- `select_combined_files(files: list[Path], episodes: list[int] | None) -> list[Path]`
- `populate_dataset(dataset, hdf5_files: list[Path], episodes: list[int] | None = None) -> object`
- `convert_ros2_dataset(raw_dir, raw_dir_eef, repo_id, mode="image", episodes=None, overwrite=False) -> Path`
- CLI flags: `--raw-dir`, `--raw-dir-eef`, `--repo-id`, `--mode`, `--episodes`, `--overwrite`.

- [ ] Load the existing shifted converter via a path-based import so direct script execution works from any working directory.
- [ ] Implement directory discovery with sorted `episode_*.hdf5` files, reject missing/empty directories, and concatenate ordinary then EEF lists.
- [ ] Implement combined-index selection without renumbering or sorting the caller's explicit selection; validate every index before conversion.
- [ ] Validate every selected source with the shifted converter's existing schema/image checks and N≥2 check. The EEF loader naturally ignores unrequested EEF datasets.
- [ ] Reuse shifted `populate_dataset` and base `create_empty_dataset`, `ensure_output_available`, and `parse_episode_selection`; this preserves the exact state/action mapping and LeRobot episode numbering.
- [ ] Remove the output directory on any exception and expose the requested CLI.

### Task 3: Verify the complete 100-episode conversion

**Files:**
- Modify only if a focused regression assertion needs correction: `examples/piper_dual/tests/test_convert_ros2_piper_data_to_lerobot_combined.py`

- [ ] Run the new focused tests plus the existing shifted converter tests.
- [ ] Run the new script with the two real directories, `--repo-id local/piper_dual_stack_cups_combined`, `--mode image`, and a temporary `HF_LEROBOT_HOME`.
- [ ] Inspect `meta/info.json` and `meta/episodes.jsonl`: total episodes must be 100; episode indices must span 0–99; each length must equal its source frame count minus one.
- [ ] Independently calculate total expected frames from both HDF5 directories and compare with LeRobot `total_frames` and the sum of episode lengths.
- [ ] Confirm the output contains ordinary episodes 0–49 followed by EEF episodes 50–99 and that no EEF feature is present in LeRobot metadata.
