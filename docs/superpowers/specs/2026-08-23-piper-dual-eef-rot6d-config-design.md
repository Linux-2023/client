# Piper Dual-Stack EEF Rot6D Policy Configuration

## Goal

Register a dedicated OpenPI configuration for the checkpoint `pi05_piper_dual_stack_cups_eef_rot6d_jax_step25000_pytorch` so inference uses the checkpoint's 20-dimensional EEF normalization statistics and preserves the 20D EEF action contract.

## Checkpoint contract

The checkpoint contains:

- PyTorch weights with model metadata `action_dim=32` and `action_horizon=50`.
- Normalization statistics at `assets/HITdongdong/piper_dual_stack_cups_eef_rot6d_shifted/norm_stats.json`.
- 20-dimensional `state` and `actions` statistics.

The EEF dataset contract is:

```text
[left_xyz, left_rot6d, left_gripper, right_xyz, right_rot6d, right_gripper]
```

The model still requires its configured 32-dimensional tensors. The normal model transform path will zero-pad the 20D normalized state/actions to 32D before model inference. The EEF output transform must crop the padded model output back to 20D before unnormalization is considered complete for the client contract; the unnormalizer pads the 20D statistics with identity values for the extra model dimensions.

## Design

Add a separate config named `pi05_piper_dual_stack_cups_eef_rot6d` in `src/openpi/training/config.py` with:

- `Pi0Config(pi05=True, action_dim=32, action_horizon=50, discrete_state_input=False)`.
- `LeRobotPiperDataConfig` using `repo_id="HITdongdong/piper_dual_stack_cups_eef_rot6d_shifted"`.
- `AssetsConfig(asset_id="HITdongdong/piper_dual_stack_cups_eef_rot6d_shifted")` so checkpoint asset lookup matches the actual directory.
- The existing three-camera repack structure and cup-stacking prompt.
- `use_delta_joint_actions=False`; EEF actions are already next-pose targets, not joint deltas.
- A 20D Piper output transform that returns `actions[:, :20]`; no 14D joint conversion is applied.

Add a configurable action width to the existing `PiperOutputs` transform, defaulting to 14 so ordinary joint policies remain unchanged. The EEF configuration passes `action_dim=20`. This is the smallest shared transform change that preserves the current 14D behavior while making the output width explicit and testable.

The original `pi05_piper_dual_stack_cups` and `pi05_piper_dual_stack_cups_shifted` configurations remain unchanged.

## Data flow

`serve_policy.py` resolves `pi05_piper_dual_stack_cups_eef_rot6d`; `DataConfig.create()` loads the explicit shifted asset ID; `Normalize` consumes 20D checkpoint statistics; `ModelTransformFactory` pads state/actions to the model's 32D width; `Unnormalize` restores the first 20 dimensions and leaves padded dimensions unchanged; `PiperOutputs(action_dim=20)` returns the 20D EEF policy action.

## Verification

Add focused tests for:

1. Config registration, repo ID, explicit asset ID, model dimensions, disabled delta conversion, and 20D output transform.
2. `PiperOutputs(action_dim=20)` returning exactly the first 20 action dimensions while the default remains 14.
3. Loading the real checkpoint norm stats and constructing the policy far enough to prove the transform chain is dimension-compatible.

Run the focused tests with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to avoid the system ROS pytest plugin importing an unavailable `lark` dependency. Start the actual policy server with the new config and checkpoint, and require the shifted norm_stats log plus `Creating server` before stopping it.

## Non-goals

- Do not change model weights or checkpoint contents.
- Do not add global normalization fallback behavior.
- Do not change ordinary 14D Piper policy behavior.
- Do not change ROS 2 bridge behavior or claim hardware validation.
