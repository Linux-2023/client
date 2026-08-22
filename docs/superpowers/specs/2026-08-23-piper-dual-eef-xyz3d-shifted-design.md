# Piper Dual EEF XYZ+RPY Shifted Converter Design

> **Status:** Approved interactively; implementation starts after written-spec review.

## Goal

Add a separate ROS 2 EEF-to-LeRobot converter that preserves each raw six-dimensional EEF pose as `[x, y, z, roll, pitch, yaw]` instead of converting orientation to rot6d, while retaining all other behavior of `convert_ros2_piper_data_to_lerobot_eef_shifted.py`.

## Decisions

- Create `examples/piper_dual/utils/convert_ros2_piper_data_to_lerobot_eef_xyz3d_shifted.py`.
- Do not modify the existing rot6d converter or its output contract.
- Interpret “xyz+3d” as the source collector's original XYZ RPY representation: `[x, y, z, roll, pitch, yaw]` in meters and radians.
- Emit 14-dimensional `observation.state` and `action` vectors:

```text
[left_x, left_y, left_z, left_roll, left_pitch, left_yaw, left_gripper,
 right_x, right_y, right_z, right_roll, right_pitch, right_yaw, right_gripper]
```

- Keep the existing shifted alignment. For source frame `t`, state uses left/right EEF poses at `t`; action uses left/right EEF poses at `t+1`; both use master grippers from source action frame `t` at indices 6 and 13.
- Keep current images and task at frame `t`. Drop source frame `N-1`, so an `N`-frame episode emits `N-1` LeRobot frames.
- Keep HDF5 discovery, ROS 2 schema validation, `metadata_json.eef.enabled` validation, episode selection, image/video modes, overwrite handling, output cleanup, FPS, image features, and CLI behavior unchanged.
- Use robot type `piper_dual_ros2_eef` to preserve compatibility with the existing EEF dataset family; the feature names and 14D shape distinguish this representation.

## Architecture and Data Flow

The new script follows the existing converter's module structure and loads `convert_ros2_piper_data_to_lerobot.py` as its shared ROS 2 base. It validates each EEF stream as a finite numeric `(frames, 6)` float32-compatible matrix but performs no rotation conversion. A focused `build_shifted_eef_state_action` function allocates two `(N-1, 14)` float32 matrices and copies raw EEF rows plus the current left/right master grippers into the fixed layout.

Dataset population loads each selected episode once, builds shifted state/action arrays, associates each output row with source image and task frame `t`, adds frames, and saves one LeRobot episode per selected HDF5 source episode. Dataset creation publishes exact feature names for XYZ, roll, pitch, yaw, and gripper on both arms.

## Error Handling

The converter rejects nonnumeric, non-finite, or incorrectly shaped EEF and master-action matrices; unequal stream lengths; episodes shorter than two frames; absent EEF datasets; malformed or false EEF metadata; invalid episode indices; and unsupported output modes. If conversion fails after output creation begins, it removes the partial output directory exactly as the existing converter does.

No fallback to rot6d, angle normalization, implicit unit conversion, frame padding, or reuse of the last pose is allowed.

## Verification

Add `examples/piper_dual/tests/test_convert_ros2_piper_data_to_lerobot_eef_xyz3d_shifted.py` before implementation and verify its initial failure is caused by the missing converter. Focused tests cover:

1. Exact 14D state/action layout and float32 dtype.
2. Current pose in state, next pose in action, current master grippers in both, and `N-1` output rows.
3. Preservation of raw RPY values without rot6d conversion or angle normalization.
4. Current-frame images/task and one saved episode per source episode.
5. Feature shape and ordered feature names for both arms.
6. Existing metadata, dimensions, finite-value, stream-length, and minimum-frame rejection.
7. CLI import/help smoke behavior and the focused converter test module.

The existing rot6d converter and its tests remain unchanged.
