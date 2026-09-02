# Piper Dual Combined LeRobot Converter Design

> **Status:** Approved interactively.

## Goal

Add an independent converter that combines the 50 episodes in `stack_cups` and the 50 episodes in `stack_cups_eef` into one 100-episode LeRobot dataset. The EEF directory's EEF datasets are intentionally ignored.

## Input and ordering

The first source directory is `/home/agilex/piper_dual_dataset/stack_cups`; its sorted `episode_*.hdf5` files become output episodes 0–49. The second source directory is `/home/agilex/piper_dual_dataset/stack_cups_eef`; its sorted files become output episodes 50–99. The implementation validates both directories and preserves each directory's filename order before concatenation.

Both directories use the existing `piper_dual_ros2_v1` HDF5 contract. EEF episodes may contain `/observations/eef` and EEF timing metadata, but the converter reads only the shared state, action, timestamp, prompt, and three image streams through the existing loader.

## Frame mapping

For each source episode with N frames, output N−1 frames using the existing shifted contract:

```text
state[t]  = [puppet[t, 0:6],  master[t, 6],  puppet[t, 7:13],  master[t, 13]]
action[t] = [puppet[t+1, 0:6], master[t, 6], puppet[t+1, 7:13], master[t, 13]]
```

The final source frame is not emitted as an observation because no next-frame puppet joint vector exists. Its puppet joints are still used by the preceding output action. Images and task prompt remain aligned to source frame t. LeRobot rebuilds timestamps from output frame indices.

## Interface and errors

Create `examples/piper_dual/utils/convert_ros2_piper_data_to_lerobot_combined.py`. The CLI accepts `--raw-dir`, `--raw-dir-eef`, `--repo-id`, `--mode`, `--episodes`, and `--overwrite`. `--episodes` addresses the combined output index 0–99. Reuse the existing shifted mapping and base HDF5/LeRobot helpers without modifying either existing converter. Reject missing/empty directories, invalid combined indices, malformed source episodes, and episodes with fewer than two frames; remove partial output on failure.

## Verification

Test source list ordering, combined index selection, EEF-field ignoring, exact frame mapping, N−1 episode lengths, and one-frame rejection. Run focused tests and a real 100-source-episode conversion, then inspect the output episode count and total frame count against the two source directories.
