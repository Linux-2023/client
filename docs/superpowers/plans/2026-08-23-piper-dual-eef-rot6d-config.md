# Piper Dual-Stack EEF Rot6D Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register and serve the 20D EEF Rot6D policy checkpoint with its matching shifted normalization assets while preserving ordinary 14D Piper policies.

**Architecture:** Add an explicit `action_dim` selector to `PiperOutputs`, keeping its default 14D joint behavior. Register a dedicated EEF `TrainConfig` with model width 32, dataset/asset ID `HITdongdong/piper_dual_stack_cups_eef_rot6d_shifted`, disabled joint-delta processing, and output width 20. Reuse existing Pi05 model transforms for 20D-to-32D padding and existing normalization loading.

**Tech Stack:** Python 3.11, OpenPI dataclass configs and transforms, PyTorch safetensors checkpoint, pytest.

## Global Constraints

- EEF vectors are 20D `[left_xyz,left_rot6d,left_gripper,right_xyz,right_rot6d,right_gripper]`.
- Checkpoint model metadata remains `action_dim=32` and `action_horizon=50`.
- Checkpoint norm stats live under `assets/HITdongdong/piper_dual_stack_cups_eef_rot6d_shifted/norm_stats.json`.
- New config name is `pi05_piper_dual_stack_cups_eef_rot6d`.
- Dataset repo ID and explicit asset ID are both `HITdongdong/piper_dual_stack_cups_eef_rot6d_shifted`.
- `use_delta_joint_actions=False` for the EEF data config.
- Existing `PiperOutputs()` behavior remains 14D and unchanged.
- Do not change checkpoint contents, global normalization fallback behavior, ROS 2 bridge behavior, or existing ordinary Piper configs.
- Run pytest with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` because the system ROS pytest plugin imports unavailable `lark`.

---

### Task 1: Add failing EEF transform and config contract tests

**Files:**
- Modify: `src/openpi/policies/piper_policy_contract_test.py:1-40`
- Modify: `src/openpi/training/config_test.py:1-100`

**Interfaces:**
- Consumes: `piper_policy.PiperOutputs`, `_config.get_config`, `TrainConfig.data.create`, and the existing EEF checkpoint path.
- Produces: tests defining `PiperOutputs(action_dim=20)` and `get_config("pi05_piper_dual_stack_cups_eef_rot6d")` behavior.

- [ ] **Step 1: Add the failing output-width test**

Append to `src/openpi/policies/piper_policy_contract_test.py`:

```python
def test_piper_outputs_supports_explicit_eef_action_width():
    actions = np.arange(64, dtype=np.float32).reshape(2, 32)

    result = piper_policy.PiperOutputs(action_dim=20)(dict(actions=actions))

    np.testing.assert_array_equal(result["actions"], actions[:, :20])
    assert result["actions"].shape == (2, 20)
```

- [ ] **Step 2: Add the failing config contract test**

Append to `src/openpi/training/config_test.py`:

```python
def test_pi05_piper_dual_stack_cups_eef_rot6d_config_matches_checkpoint_contract():
    config = _config.get_config("pi05_piper_dual_stack_cups_eef_rot6d")
    data_config = config.data.create(config.assets_dirs, config.model)

    assert config.model.pi05 is True
    assert config.model.action_dim == 32
    assert config.model.action_horizon == 50
    assert config.model.discrete_state_input is False
    assert config.data.repo_id == "HITdongdong/piper_dual_stack_cups_eef_rot6d_shifted"
    assert config.data.assets.asset_id == "HITdongdong/piper_dual_stack_cups_eef_rot6d_shifted"
    assert config.data.use_delta_joint_actions is False
    assert data_config.asset_id == "HITdongdong/piper_dual_stack_cups_eef_rot6d_shifted"
    assert data_config.norm_stats is not None
    assert config.data.default_prompt == "Stack the paper cups together."
```

- [ ] **Step 3: Run the new tests before production changes**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  src/openpi/policies/piper_policy_contract_test.py::test_piper_outputs_supports_explicit_eef_action_width \
  src/openpi/training/config_test.py::test_pi05_piper_dual_stack_cups_eef_rot6d_config_matches_checkpoint_contract -v
```

Expected: FAIL because `PiperOutputs` has no `action_dim` parameter and the EEF config is not registered.

---

### Task 2: Implement EEF output width and configuration registration

**Files:**
- Modify: `src/openpi/policies/piper_policy.py:96-107`
- Modify: `src/openpi/training/config.py:377-428 and _CONFIGS near the existing dual-stack configs`

**Interfaces:**
- Consumes: `PiperOutputs(adapt_to_pi: bool = False, action_dim: int = 14)` and existing `LeRobotPiperDataConfig`.
- Produces: `PiperOutputs(action_dim=20)` returning 20D actions; `get_config("pi05_piper_dual_stack_cups_eef_rot6d")` resolving to a 32D Pi05 config whose data pipeline loads 20D stats.

- [ ] **Step 1: Add the minimal configurable output width**

Change `PiperOutputs` to:

```python
@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    """Outputs for the Piper policy."""

    adapt_to_pi: bool = False
    action_dim: int = 14

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, : self.action_dim])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}
```

The default remains 14, so existing callers and tests retain their behavior.

- [ ] **Step 2: Add the EEF configuration**

Insert immediately after the existing `pi05_piper_dual_stack_cups_shifted` entry:

```python
    TrainConfig(
        name="pi05_piper_dual_stack_cups_eef_rot6d",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=50,
            discrete_state_input=False,
        ),
        data=LeRobotPiperDataConfig(
            repo_id="HITdongdong/piper_dual_stack_cups_eef_rot6d_shifted",
            assets=AssetsConfig(asset_id="HITdongdong/piper_dual_stack_cups_eef_rot6d_shifted"),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_joint_actions=False,
            default_prompt="Stack the paper cups together.",
            adapt_to_pi=False,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                            "episode_index": "episode_index",
                            "frame_index": "frame_index",
                        }
                    )
                ]
            ),
        ),
        batch_size=64,
        num_train_steps=30_000,
        save_interval=5_000,
        keep_period=5_000,
    ),
```

- [ ] **Step 3: Wire the 20D output transform**

Update `LeRobotPiperDataConfig.create()` so it passes the model-aware width to `PiperOutputs` without changing default joint behavior:

```python
data_transforms = _transforms.Group(
    inputs=[piper_policy.PiperInputs(adapt_to_pi=self.adapt_to_pi)],
    outputs=[piper_policy.PiperOutputs(adapt_to_pi=self.adapt_to_pi, action_dim=self.output_action_dim)],
)
```

Add the config field:

```python
output_action_dim: int | None = None
```

and use `self.output_action_dim or model_config.action_dim` only if the field is explicitly set by the EEF config. The EEF entry must set `output_action_dim=20`; ordinary Piper entries leave it unset and continue returning 14D. The preferred implementation is to name the field `output_action_dim` rather than infer a crop from `model_config.action_dim`, because the EEF model width is intentionally 32 while its external action contract is 20.

- [ ] **Step 4: Run focused tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  src/openpi/policies/piper_policy_contract_test.py \
  src/openpi/training/config_test.py -v
```

Expected: all focused tests PASS, including the existing default 14D Piper output test and the new EEF config test.

---

### Task 3: Verify real checkpoint normalization and policy server

**Files:**
- No source changes expected.

**Interfaces:**
- Consumes: `pi05_piper_dual_stack_cups_eef_rot6d` and checkpoint `/home/agilex/checkpoints/pi05_piper_dual_stack_cups_eef_rot6d_jax_step25000_pytorch`.
- Produces: evidence that config resolution, norm stats, model padding, EEF output cropping, and server startup work together.

- [ ] **Step 1: Verify config and real stats dimensions**

```bash
PYTHONPATH=src .venv/bin/python -c 'from pathlib import Path; from openpi.training import checkpoints, config; c=config.get_config("pi05_piper_dual_stack_cups_eef_rot6d"); d=c.data.create(c.assets_dirs,c.model); stats=checkpoints.load_norm_stats(Path("/home/agilex/checkpoints/pi05_piper_dual_stack_cups_eef_rot6d_jax_step25000_pytorch/assets"), d.asset_id); print(c.name, c.model.action_dim, d.asset_id, {k: v.mean.shape[-1] for k, v in stats.items()})'
```

Expected: config name is the new EEF name, model width is 32, asset ID is shifted EEF ID, and both stats dimensions are 20.

- [ ] **Step 2: Start the actual server**

```bash
PYTHONPATH=src .venv/bin/python scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_piper_dual_stack_cups_eef_rot6d \
  --policy.dir=/home/agilex/checkpoints/pi05_piper_dual_stack_cups_eef_rot6d_jax_step25000_pytorch
```

Expected logs include:

```text
Loaded norm stats from /home/agilex/checkpoints/pi05_piper_dual_stack_cups_eef_rot6d_jax_step25000_pytorch/assets/HITdongdong/piper_dual_stack_cups_eef_rot6d_shifted
Creating server
```

Stop after readiness. No `FileNotFoundError`, action-dimension mismatch, or 14D output should occur.

- [ ] **Step 3: Commit only focused files**

```bash
git add src/openpi/policies/piper_policy.py src/openpi/policies/piper_policy_contract_test.py src/openpi/training/config.py src/openpi/training/config_test.py
 git commit -m "feat: register Piper EEF rot6d policy config"
```
