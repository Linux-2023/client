# Piper Dual Shifted LeRobot Converter Design

> **Status:** Approved by continuation of the requested implementation.

## Goal

Add an independent converter for the existing `piper_dual_ros2_v1` HDF5 episodes. The converted LeRobot dataset uses puppet-arm joint angles for both observation state and action, shifts only the twelve joint-angle values by one frame for the action, and uses the current-frame master-arm grippers for both state and action.

## Input contract

Each selected episode contains `/observations/state` and `/action` with shape `(N, 14)`. The vector order is left arm seven values followed by right arm seven values; each arm is six joint angles followed by one gripper value. Images remain under `/observations/images/{cam_high,cam_left_wrist,cam_right_wrist}`. Metadata prompt is read from `metadata_json.prompt`.

## Output mapping

For each source frame `t` in `0 <= t < N - 1`, with `p = /observations/state` and `m = /action`:

```text
state[t]  = [p[t, 0:6],  m[t, 6],  p[t, 7:13], m[t, 13]]
action[t] = [p[t+1, 0:6], m[t, 6],  p[t+1, 7:13], m[t, 13]]
```

The output keeps source frame `t`'s images and task prompt. As in the existing converter's default mode, LeRobot rebuilds exact episode-relative timestamps as `frame_index / fps`. The source episode's final frame is discarded as an observation because it has no next-frame joint target; its puppet joints are used as the final output action target. Episodes with fewer than two frames are rejected before output creation.

The LeRobot feature remains a fourteen-dimensional float32 vector with the existing motor names. This keeps left/right ordering and gripper units unchanged while making the temporal contract explicit in the new script. The existing converter is not modified.

## Architecture

Create `examples/piper_dual/utils/convert_ros2_piper_data_to_lerobot_shifted.py`. Reuse the existing converter's validated HDF5 loading, JPEG decoding, output handling, LeRobot dataset creation, episode selection, and CLI conventions through explicit imports where safe or a focused independent implementation where the new mapping requires different episode lengths. Keep the transformation itself as a pure function so its boundary behavior can be tested without LeRobot or image encoding.

## Error handling

Reject malformed schema, non-finite state/action values, missing metadata/images, invalid episode indices, and episodes shorter than two frames. On any conversion failure, remove the partially created local output directory. Do not silently pad or duplicate the final action frame.

## Verification

Add focused tests for the exact mapping, current-frame master gripper preservation, dropping the final frame, and rejecting one-frame episodes. Run the focused test module, then convert one real source episode with the target virtual environment and inspect the generated LeRobot data for output frame count and mapped first/last values. Run the existing converter regression tests to ensure the old script remains unchanged.
