# Piper Dual EEF XYZ3D ROS 2 Client Design

## Goal

Add an independent ROS 2 deployment client for the checkpoint `/home/agilex/checkpoints/pi05_piper_dual_stack_cups_eef_xyz3d_jax_step10000_pytorch`. The client must use the checkpoint's 14-dimensional XYZ+RPY EEF state/action contract and publish validated dual-arm `PosCmd` requests without changing the existing 20D rot6d client.

## Confirmed checkpoint contract

The checkpoint contains model metadata `action_dim=32` and `action_horizon=50`. Its normalization assets are stored at `assets/HITdongdong/piper_dual_stack_cups_eef_xyz3d/norm_stats.json`; both `state` and `actions` statistics contain 14 values. The existing policy server configuration is `pi05_piper_dual_stack_cups_eef_xyz3d` and its output transform crops model output to 14 dimensions.

The external state/action layout is:

```text
[left_x, left_y, left_z,
 left_roll, left_pitch, left_yaw,
 left_gripper,
 right_x, right_y, right_z,
 right_roll, right_pitch, right_yaw,
 right_gripper]
```

Positions and grippers are in meters. Roll, pitch, and yaw are in radians and follow the existing ROS/tf2 XYZ RPY convention.

## Architecture

Create three independent files under `examples/piper_dual`:

1. `eef_xyz3d_action_adapter.py` validates a one-dimensional, finite, numeric 14D policy action and returns the established bridge request:

   ```python
   {
       "type": "publish_eef_action",
       "left": [x, y, z, roll, pitch, yaw, gripper],
       "right": [x, y, z, roll, pitch, yaw, gripper],
   }
   ```

   No rot6d decoding or rotation conversion is performed.

2. `eef_xyz3d_observation_adapter.py` consumes synchronized left/right puppet EEF poses as six-value `[x,y,z,roll,pitch,yaw]` vectors. It takes current master grippers from joint-action indices 6 and 13 and builds the 14D `observation.state`. Camera decoding, timestamp validation, synchronization metadata, and task handling reuse the existing `ObservationAdapter`.

3. `main_dual_eef_xyz3d_ros.py` mirrors the lifecycle of `main_dual_eef_ros.py` while injecting only the XYZ3D observation/action adapters. It reuses `Ros2BackendClient`, `Ros2DualEnvironment`, the WebSocket policy client, action chunk brokers, runtime, video saver, bridge process, EEF topics, and `PosCmd` publishers.

The existing 20D rot6d adapters and `main_dual_eef_ros.py` remain unchanged. The new client does not load the checkpoint directly; `scripts/serve_policy.py` owns model loading and normalization.

## Data flow

For each synchronized frame:

1. The bridge supplies three camera images, left/right puppet EEF XYZ+RPY poses, and the current 14D master joint action.
2. `EefXyz3dObservationAdapter` emits a 14D state containing both puppet poses and current master grippers.
3. The WebSocket server normalizes the 14D state, pads it to the model's internal 32D width, performs inference, unnormalizes the output, and returns the first 14 action dimensions.
4. `EefXyz3dActionAdapter` validates and splits the action into two seven-value requests.
5. The existing ROS 2 bridge publishes `piper_msgs/msg/PosCmd` on `/pos_left_cmd` and `/pos_right_cmd` when publishing is explicitly enabled.

The new client defaults to `action_horizon=50`, matching the checkpoint metadata, and retains 30 Hz as the requested runtime rate.

## CLI and safety

The new CLI retains the existing EEF topic, ROS 2 config, bridge Python, policy host/port, prompt, episode, RTC, and maximum-action-delta options. Its contract summary explicitly identifies the 14D XYZ+RPY representation.

Safe defaults are mandatory:

- `dry_run=True`.
- `publish_actions=False`.
- Real command publication requires both `--no-dry-run` and `--publish-actions`.
- Empty observation or action topic names are rejected before the environment is created.
- `max_action_delta` must be finite and non-negative when supplied.
- Existing watchdog, frame timeout, synchronization, and publish-lock behavior remain active.

## Failure handling

Both adapters fail closed:

- Reject inputs with the wrong rank or width.
- Reject non-numeric values.
- Reject NaN and infinity.
- Reject missing left/right EEF observations.
- Reject malformed 14D master actions before using gripper indices.
- Reject malformed server actions before creating any bridge request.

The existing environment failure path locks further action publication for stale observations, synchronization failures, and action-delta violations. Runtime initialization and cleanup retain the existing signal and exception behavior.

## Verification

Use test-driven development. Add focused tests before production code for:

1. Exact 14D observation layout, master-gripper selection, dtype, and malformed/non-finite input rejection.
2. Exact left/right action splitting and malformed/non-finite action rejection.
3. CLI defaults: horizon 50, dry-run enabled, publishing disabled, expected EEF topics, and exposed safety flags.
4. Validation that publishing cannot be enabled while dry-run remains active.
5. Environment construction with `eef_control=True` and both XYZ3D adapters injected.
6. Contract summary identifying 14D XYZ+RPY state/action.

Run focused pytest with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, compile the new Python modules, run the new entrypoint with `--help`, and exercise a dry-run smoke path that cannot publish robot actions. Existing rot6d adapter and entrypoint tests must remain green. Hardware movement is outside this verification unless the actual ROS 2 control stack and robot are explicitly exercised.

## Deployment commands

The server uses the existing registered configuration:

```bash
python scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_piper_dual_stack_cups_eef_xyz3d \
  --policy.dir=/home/agilex/checkpoints/pi05_piper_dual_stack_cups_eef_xyz3d_jax_step10000_pytorch
```

The client command will be:

```bash
python examples/piper_dual/main_dual_eef_xyz3d_ros.py \
  --host 127.0.0.1 \
  --port 8000 \
  --bridge-python /usr/bin/python3 \
  --ros2-config examples/piper_dual/ros2_piper_dual.yaml \
  --prompt "Stack_the_paper_cups_together." \
  --action-horizon 50 \
  --dry-run
```

Real action publication additionally requires `--no-dry-run --publish-actions`.

## Non-goals

- Do not modify the checkpoint, normalization assets, or server-side XYZ3D policy transforms.
- Do not modify or parameterize the existing 20D rot6d client.
- Do not convert XYZ+RPY actions to rot6d.
- Do not introduce a shared base-entrypoint refactor.
- Do not change bridge message types, topic defaults, camera preprocessing, control-loop logging, or robot-controller behavior.
