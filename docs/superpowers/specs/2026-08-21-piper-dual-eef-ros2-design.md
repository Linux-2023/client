# Piper Dual ROS 2 EEF Collector Design

> **Status:** Approved interactively and self-reviewed; implementation starts after written-spec review.

## Goal

Add `examples/piper_dual/collect_data_eef_ros2.py` to collect puppet-left and puppet-right end-effector poses in every synchronized frame, in addition to the existing dual-arm joint state/action data collected by `collect_data_ros2.py`.

## Decisions

- Collect puppet EEF only; do not collect master EEF.
- Subscribe by default to `/puppet/end_pose_left` and `/puppet/end_pose_right`.
- Use `geometry_msgs/msg/PoseStamped`.
- Convert each pose to `[x, y, z, roll, pitch, yaw]`.
- Position units are meters; orientation units are radians; RPY follows the official collector's `tf2::Matrix3x3::getRPY` convention.
- EEF streams are required synchronization inputs. A frame is written only when both EEF streams, all three cameras, and all four joint streams are within `--max-sync-error-ms` of the camera reference timestamp.
- Preserve the existing 14-dimensional `observations/state` and `action` datasets and the behavior of `collect_data_ros2.py`.
- Keep the original and joint-shifted LeRobot converters unchanged. A separate EEF-shifted converter consumes the raw six-dimensional EEF datasets and emits 20-dimensional EEF-plus-gripper state/action vectors.
- Default dry-run behavior remains enabled; the EEF collector does not publish actions.

## Architecture

The existing ROS 2 bridge, backend, synchronizer, frame object, and streaming writer are extended once and reused by both collectors. The legacy collector keeps its current sensor set and output behavior through default arguments. The new collector supplies EEF topics as explicit backend/bridge process arguments and uses the same preview/state-machine lifecycle, episode numbering, partial-file recovery, validation, and clean shutdown behavior.

### ROS 2 bridge

`ros2_bridge_process.py` gains optional PoseStamped topic configuration. When EEF topics are configured, it subscribes to them and emits sensor events containing a sensor name, the message header timestamp, and a six-element finite float vector. The default bridge configuration remains camera plus joint streams, so the existing collector does not subscribe to EEF topics.

The bridge event builder is ROS-independent at the conversion boundary: it accepts a PoseStamped-like message and computes translation plus quaternion-to-RPY conversion. Invalid or non-finite values are reported through the existing bridge error protocol and are not silently serialized.

### Backend and synchronization

`Ros2BackendClient` accepts optional `eef_left_topic` and `eef_right_topic` arguments and passes them to the bridge as process arguments; neither the supplied YAML file nor a generated configuration file is modified. EEF sensor names are `eef_puppet_left` and `eef_puppet_right`.

`FrameSynchronizer` accepts an `include_eef` mode while retaining its existing required camera/joint set by default. In EEF mode, the required set is the union of the existing seven sensors and the two EEF sensors. `SynchronizedFrame.eef` is `None` for ordinary frames and a mapping containing exactly `puppet_left` and `puppet_right` six-element vectors for EEF frames. EEF-mode writers require both vectors.

The synchronizer continues to use `cam_high` as reference, nearest timestamp selection, bounded buffers, stale-sample rejection, and the configured maximum absolute timestamp error. Missing EEF data waits for a matching sample; an EEF stream that advances beyond the matching window causes the reference frame to be rejected as stale.

### HDF5 writer

The existing schema version remains `piper_dual_ros2_v1` for backward compatibility. EEF episodes are distinguished by `metadata_json.eef.enabled = true` and add these extendible float32 datasets:

- `/observations/eef/puppet_left`: shape `(N, 6)`
- `/observations/eef/puppet_right`: shape `(N, 6)`

The EEF values are frame-aligned with `/observations/state`, `/action`, timestamps, images, and synchronization metadata. EEF sensor timestamps and sync errors are also stored in the existing sensor metadata groups under `eef_puppet_left` and `eef_puppet_right`. Metadata records `collector = "collect_data_eef_ros2.py"`, `eef.enabled = true`, the topic mapping, vector order, and units. The normal collector does not create EEF datasets or EEF sensor metadata.

`StreamingEpisodeWriter.open(..., include_eef=False)` preserves the ordinary schema by default; the new collector passes `include_eef=True`. Validation uses `metadata_json.eef.enabled` as the contract marker: marked episodes must contain both EEF datasets with exact `(frame_count, 6)` shape, float32 dtype, finite values, an extendible first dimension, and matching EEF timestamp/error datasets. Unmarked existing episodes remain valid without EEF datasets.

### Shifted LeRobot EEF conversion

`convert_ros2_piper_data_to_lerobot_eef_shifted.py` is a separate downstream converter. It never mutates raw HDF5 files and accepts only episodes marked by `metadata_json.eef.enabled = true`. Each raw `[x, y, z, roll, pitch, yaw]` pose is converted using the collector's ROS/tf2 convention `R = Rz(yaw) Ry(pitch) Rx(roll)`. The orientation representation is the first two rotation-matrix columns flattened as `[r11, r21, r31, r12, r22, r32]`, producing a nine-dimensional pose.

LeRobot `observation.state` and `action` each use the following 20-dimensional layout:

```text
[left_xyz, left_rot6d, left_gripper, right_xyz, right_rot6d, right_gripper]
```

For source frame `t`, state uses both EEF poses at `t`; action uses both EEF poses at `t+1`. Both vectors use the two master grippers from source action frame `t`, matching the existing joint-shifted mapping. Images and task use frame `t`; source frame `N-1` is omitted, so an `N`-frame source episode emits `N-1` LeRobot frames. Episodes with fewer than two frames, missing/malformed EEF data, a false EEF metadata marker, invalid dimensions, or non-finite values are rejected.

## New CLI

`collect_data_eef_ros2.py` mirrors the current collector's required and optional arguments:

- `--output-dir`
- `--prompt`
- `--config`
- `--bridge-python`
- `--jpeg-quality`
- `--max-sync-error-ms`
- `--dry-run` / `--no-dry-run`
- `--publish-actions`, with the same existing safety rule that it cannot be combined with `--dry-run`; the collection state machine never calls `publish_action`, so data collection itself emits no action messages.
- `--eef-left-topic`, default `/puppet/end_pose_left`
- `--eef-right-topic`, default `/puppet/end_pose_right`
- `--render-after-save`

The preview displays the same three camera panels and recording controls as the existing collector. `s` starts/finalizes an episode and `q` aborts and exits.

## Error handling

- Malformed bridge configuration, unsupported PoseStamped payloads, invalid dimensions, non-finite pose values, and protocol failures surface as collector errors.
- Missing or late EEF samples do not produce a partial logical frame. They wait while the frame can still be matched, or reject the stale reference when the stream has advanced beyond the synchronization window.
- A zero-frame or invalid episode is removed as a partial file and does not become a final HDF5 output.
- Interrupts, preview quit, backend failures, and bridge failures close the writer and backend exactly once.
- No NaN fill values, nearest-value fallback outside the synchronization window, or independent EEF time series are used.

## Verification

Add focused tests for:

1. PoseStamped quaternion-to-RPY event conversion, finite-value validation, topic/sensor mapping, and protocol payload shape.
2. Synchronizer operation with and without EEF streams, including exact boundary acceptance, missing EEF waiting, stale EEF rejection, and frame alignment.
3. EEF-enabled HDF5 writing and validation, including the metadata marker, shapes, dtype, finite values, timestamp/error datasets, and rejection of missing or mismatched EEF data.
4. New collector CLI defaults, topic overrides, safe dry-run behavior, state-machine reuse, clean abort, and multi-episode numbering.
5. Existing collector regression: default sensor set and existing HDF5 schema remain unchanged.
6. EEF-shifted LeRobot conversion: ROS XYZ RPY to column-major rot6d, 20-dimensional dual-arm EEF-plus-gripper layout, current/next EEF alignment, current master grippers, current images/task, `N-1` output count, metadata/dimension/finite-value rejection, and unchanged legacy converters.

Run the focused Piper dual test modules and a real dry-run smoke path that exercises the collector's parser/configuration without robot action publishing. Do not claim hardware validation unless ROS 2 and the physical cameras/arms are actually available.
