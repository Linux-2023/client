# Shifted Piper Dual-Stack Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated `pi05_piper_dual_stack_cups_shifted` configuration that loads normalization statistics from the supplied checkpoint's existing shifted asset directory.

**Architecture:** Register one additional `TrainConfig` beside the existing Piper dual-stack configuration. Preserve the original dataset and model contract while explicitly setting the shifted configuration's `AssetsConfig.asset_id`; keep checkpoint loading and server behavior unchanged.

**Tech Stack:** Python 3.11, dataclasses-based OpenPI training configuration, pytest, Tyro CLI.

## Global Constraints

- Keep `pi05_piper_dual_stack_cups` unchanged.
- New config name: `pi05_piper_dual_stack_cups_shifted`.
- Dataset repo ID remains `HITdongdong/piper_dual_stack_cups_ros2`.
- Explicit asset ID is `HITdongdong/piper_dual_stack_cups_ros2_shifted`.
- Do not add global normalization fallback logic.
- Do not copy, rename, or symlink checkpoint assets.
- Do not change model weights, transforms, prompt, training settings, or server behavior.

---

### Task 1: Register and verify shifted inference configuration

**Files:**
- Modify: `src/openpi/training/config.py:738-769`
- Modify: `src/openpi/training/config_test.py:1-26`

**Interfaces:**
- Consumes: `_config.get_config(name: str) -> TrainConfig`, `AssetsConfig(asset_id: str | None = None)`, and the existing `LeRobotPiperDataConfig` contract.
- Produces: `_config.get_config("pi05_piper_dual_stack_cups_shifted") -> TrainConfig` with `data.assets.asset_id == "HITdongdong/piper_dual_stack_cups_ros2_shifted"`.

- [ ] **Step 1: Write the failing shifted-config contract test**

Append this focused test to `src/openpi/training/config_test.py`:

```python
def test_pi05_piper_dual_stack_cups_shifted_config_uses_shifted_assets():
    config = _config.get_config("pi05_piper_dual_stack_cups_shifted")

    assert config.model.pi05 is True
    assert config.model.action_horizon == 50
    assert config.model.discrete_state_input is False
    assert config.data.repo_id == "HITdongdong/piper_dual_stack_cups_ros2"
    assert config.data.assets.asset_id == "HITdongdong/piper_dual_stack_cups_ros2_shifted"
    assert config.data.default_prompt == "Stack the paper cups together."
    assert config.data.adapt_to_pi is False

    repack = config.data.repack_transforms.inputs[0]
    assert repack.structure == {
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
```

- [ ] **Step 2: Run the focused test and verify the missing config fails**

Run from the repository root:

```bash
.venv/bin/python -m pytest src/openpi/training/config_test.py::test_pi05_piper_dual_stack_cups_shifted_config_uses_shifted_assets -v
```

Expected: FAIL because `get_config` cannot find `pi05_piper_dual_stack_cups_shifted`.

- [ ] **Step 3: Add the minimal independent shifted configuration**

Insert this `TrainConfig` immediately after the existing `pi05_piper_dual_stack_cups` entry in `src/openpi/training/config.py`:

```python
    TrainConfig(
        name="pi05_piper_dual_stack_cups_shifted",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50, discrete_state_input=False),
        data=LeRobotPiperDataConfig(
            repo_id="HITdongdong/piper_dual_stack_cups_ros2",
            assets=AssetsConfig(asset_id="HITdongdong/piper_dual_stack_cups_ros2_shifted"),
            base_config=DataConfig(prompt_from_task=True),
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

This duplicates the established adjacent configuration intentionally: the original configuration remains textually and behaviorally unchanged, while the new entry differs only by name and explicit asset ID.

- [ ] **Step 4: Run both configuration contract tests**

```bash
.venv/bin/python -m pytest src/openpi/training/config_test.py -v
```

Expected: both original and shifted configuration tests PASS.

- [ ] **Step 5: Verify the shifted config resolves the supplied norm-stats path**

```bash
.venv/bin/python -c 'from pathlib import Path; from openpi.training import checkpoints, config; c = config.get_config("pi05_piper_dual_stack_cups_shifted"); asset_id = c.data.create(c.assets_dirs, c.model).asset_id; stats = checkpoints.load_norm_stats(Path("/home/agilex/checkpoints/pi05_piper_dual_stack_cups_shifted_jax_step20000_pytorch/assets"), asset_id); print(asset_id, sorted(stats))'
```

Expected: output contains `HITdongdong/piper_dual_stack_cups_ros2_shifted` and the normalization-stat keys, with no `FileNotFoundError`.

- [ ] **Step 6: Smoke-test actual policy construction and server startup**

Launch using the new configuration:

```bash
.venv/bin/python scripts/serve_policy.py --port=8000 policy:checkpoint --policy.config=pi05_piper_dual_stack_cups_shifted --policy.dir=/home/agilex/checkpoints/pi05_piper_dual_stack_cups_shifted_jax_step20000_pytorch
```

Expected logs:

```text
Loaded norm stats from /home/agilex/checkpoints/pi05_piper_dual_stack_cups_shifted_jax_step20000_pytorch/assets/HITdongdong/piper_dual_stack_cups_ros2_shifted
Creating server
```

Stop the server after readiness is observed. The prior `Norm stats file not found` traceback must not recur.

- [ ] **Step 7: Commit only the focused configuration change**

```bash
git add src/openpi/training/config.py src/openpi/training/config_test.py
git commit -m "feat: add shifted Piper dual-stack config"
```
