# Shifted Piper Dual-Stack Inference Configuration

## Goal

Allow the shifted PyTorch checkpoint to load its existing normalization statistics without changing the original `pi05_piper_dual_stack_cups` training contract or creating a compatibility symlink.

## Design

Add a separate training configuration named `pi05_piper_dual_stack_cups_shifted` in `src/openpi/training/config.py`.

The new configuration reuses the existing dual-arm Pi05 model and data transformation contract:

- `repo_id`: `HITdongdong/piper_dual_stack_cups_ros2`
- `asset_id`: `HITdongdong/piper_dual_stack_cups_ros2_shifted`
- Pi05 model with `action_horizon=50` and `discrete_state_input=False`
- Existing dual-camera repack transform
- Existing prompt and delta-joint-action behavior
- Existing batch and checkpoint training settings

Only the configuration name and explicit asset ID differ. The original configuration remains unchanged.

## Data flow

`serve_policy.py` resolves the new config, `DataConfig.create()` preserves the dataset repo ID while using the explicit shifted asset ID, and `create_trained_policy()` loads:

`<checkpoint>/assets/HITdongdong/piper_dual_stack_cups_ros2_shifted/norm_stats.json`

The supplied checkpoint already contains this file.

## Verification

Extend the existing configuration contract test to verify the new config's model contract, repo ID, explicit shifted asset ID, and loaded normalization statistics. Run the focused configuration test and a policy-construction smoke check against the supplied checkpoint. Do not alter checkpoint contents or add fallback path logic.

## Non-goals

- Do not rename or modify the original configuration.
- Do not change normalization loading semantics globally.
- Do not copy or symlink checkpoint assets.
- Do not change model weights, transforms, or runtime server behavior.
