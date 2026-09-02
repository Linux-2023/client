# Piper Dual EEF Inference Client Design

> **Status:** Approved interactively.

## Goal

Add an independent ROS 2 deployment client that runs a policy trained on the dual-arm EEF LeRobot representation instead of the existing 14D joint representation.

## Contract

The EEF policy observation and action are 20D float32 vectors with this layout:

```text
[left_x, left_y, left_z, left_rot6d_0..5, left_gripper,
 right_x, right_y, right_z, right_rot6d_0..5, right_gripper]
```

The raw ROS 2 bridge provides puppet EEF poses as six values `[x,y,z,roll,pitch,yaw]` and master joint streams. The inference observation uses the current puppet EEF pose converted to position plus column-major rot6d and the current master grippers at action indices 6 and 13. Images, task, and synchronization metadata follow the existing PI05 observation contract.

The policy action uses the same 20D layout. The client validates finite values, converts each rot6d to a right-handed rotation matrix with Gram-Schmidt orthogonalization, extracts XYZ RPY radians, and sends `[x,y,z,roll,pitch,yaw,gripper]` to Piper's `piper_msgs/msg/PosCmd` bridge action. Piper's existing node consumes meters, radians, and gripper meters; mode fields remain fixed at zero because the node's callback selects its own position-control mode.

## Components

- `eef_pose.py`: ROS XYZ RPY ↔ position/rot6d conversion, including robust rot6d decoding.
- `eef_action_adapter.py`: validates 20D policy actions and encodes bridge `publish_eef_action` requests.
- `eef_observation_adapter.py`: reuses image adaptation while constructing 20D EEF observations from synchronized EEF and master-gripper data.
- `ros2_bridge_process.py`: optional EEF-control mode, `PosCmd` publishers, and validated EEF action requests; default joint behavior remains unchanged.
- `ros2_backend.py`: injectable action adapter plus optional EEF-control bridge flag; default behavior remains joint control.
- `ros2_environment.py`: pass EEF/action payloads to custom adapters; existing 14D environment behavior remains unchanged.
- `main_dual_eef_ros.py`: deployment entrypoint mirroring `main_dual_ros.py` policy/runtime setup with EEF topics and safe publishing defaults.

## Safety and Failure Handling

- Dry-run is enabled by default; publishing requires both `--no-dry-run` and `--publish-actions`.
- Invalid dimensions, non-finite values, degenerate rot6d vectors, bridge errors, stale observations, and action-delta violations fail the episode and lock further publishing.
- Existing joint client and bridge protocol remain backward compatible unless EEF-control mode is explicitly enabled.
- No hardware validation is claimed by unit or synthetic tests.

## Verification

Focused tests cover rot6d identity/yaw and degenerate inputs, 20D observation/action mapping, EEF request validation, PosCmd message encoding, bridge/backend mode selection, CLI safety defaults, and cleanup. Run focused EEF tests, existing ROS2 regression tests, syntax checks, and a dry-run CLI help smoke test.
