import dataclasses

import numpy as np
import pytest

from openpi import transforms
from openpi.training import config

CONFIG_NAME = "pi05_dobot_yx_digital_scale_right"


def _raw_sample():
    return {
        "observation.state": np.arange(14, dtype=np.float32),
        "action": np.arange(50 * 14, dtype=np.float32).reshape(50, 14),
        "observation.images.top": np.full((3, 16, 20), 0.5, dtype=np.float32),
        "observation.images.right_wrist": np.ones((3, 16, 20), dtype=np.float32),
        "prompt": "Weigh the mango.",
    }


def test_config_uses_independent_right_arm_contract(tmp_path):
    cfg = config.get_config(CONFIG_NAME)
    data = cfg.data.create(tmp_path, cfg.model)
    assert cfg.model.pi05
    assert cfg.model.discrete_state_input
    assert (cfg.model.action_dim, cfg.model.action_horizon) == (32, 50)
    assert (cfg.batch_size, cfg.fsdp_devices, cfg.num_train_steps) == (128, 4, 30_000)
    assert data.repo_id == "wwccww/yx_digital_scale"
    assert data.action_sequence_keys == ("action",)
    assert data.image_keys == ("observation.images.top", "observation.images.right_wrist")
    assert data.prompt_from_task
    assert data.use_quantile_norm
    assert cfg.assets_dirs.name == CONFIG_NAME
    assert dataclasses.replace(cfg, exp_name="check").checkpoint_dir.parent.name == CONFIG_NAME
    sample = transforms.compose(data.repack_transforms.inputs)(_raw_sample())
    np.testing.assert_array_equal(sample["state"], np.arange(7, 14))
    np.testing.assert_array_equal(sample["actions"], _raw_sample()["action"][:, 7:14])
    assert set(sample["images"]) == {"top", "right_wrist"}


def test_button_config_preserves_right_arm_training_contract(tmp_path):
    cfg = config.get_config("pi05_dobot_yx_button_right")
    mango = config.get_config(CONFIG_NAME)
    data = cfg.data.create(tmp_path, cfg.model)
    assert data.repo_id == "wwccww/yx_button_lerobot_video"
    assert data.asset_id == "wwccww/yx_button_lerobot_video"
    assert cfg.assets_dirs != mango.assets_dirs
    assert cfg.model == mango.model
    assert cfg.weight_loader == mango.weight_loader
    assert cfg.lr_schedule == mango.lr_schedule
    assert cfg.optimizer == mango.optimizer
    assert (cfg.batch_size, cfg.num_train_steps, cfg.save_interval, cfg.keep_period) == (128, 30000, 10000, 10000)
    assert data.image_keys == ("observation.images.top", "observation.images.right_wrist")
    assert data.prompt_from_task and data.use_quantile_norm
    raw = _raw_sample()
    raw["prompt"] = "yx button"
    sample = transforms.compose(data.repack_transforms.inputs)(raw)
    assert sample["prompt"] == "yx button"
    np.testing.assert_array_equal(sample["state"], raw["observation.state"][7:14])
    np.testing.assert_array_equal(sample["actions"], raw["action"][:, 7:14])


def test_repack_ignores_left_values_and_left_camera():
    from openpi.policies.dobot_right_policy import DobotRightRepack

    raw = _raw_sample()
    raw["observation.state"][:7] = np.nan
    raw["action"][:, :7] = np.nan
    # Left-wrist bytes must never be inspected by the adapter.
    raw["observation.images.left_wrist"] = object()
    result = DobotRightRepack()(raw)
    assert result["state"].shape == (7,)
    assert np.isfinite(result["state"]).all()
    assert result["actions"].shape == (50, 7)
    assert np.isfinite(result["actions"]).all()
    assert result["prompt"] == raw["prompt"]
    np.testing.assert_array_equal(result["state"], raw["observation.state"][7:14])
    np.testing.assert_array_equal(result["actions"], raw["action"][:, 7:14])
    assert np.isnan(raw["observation.state"][:7]).all()


def test_right_arm_inputs_mask_missing_camera_and_preserve_actions():
    from openpi.policies.dobot_right_policy import DobotRightInputs
    from openpi.policies.dobot_right_policy import DobotRightRepack

    data = DobotRightRepack()(_raw_sample())
    result = DobotRightInputs()(data)
    assert result["state"].shape == (7,)
    assert result["image_mask"] == {
        "base_0_rgb": True,
        "left_wrist_0_rgb": False,
        "right_wrist_0_rgb": True,
    }
    assert result["image"]["base_0_rgb"].shape == (16, 20, 3)
    assert result["image"]["base_0_rgb"].dtype == np.uint8
    assert not result["image"]["left_wrist_0_rgb"].any()
    np.testing.assert_array_equal(result["image"]["right_wrist_0_rgb"], 255)
    np.testing.assert_array_equal(result["actions"], data["actions"])
    np.testing.assert_array_equal(result["state"], data["state"])


def test_inference_needs_only_right_state_and_two_hwc_images():
    from openpi.policies.dobot_right_policy import DobotRightInputs

    top = np.full((16, 20, 3), 23, dtype=np.uint8)
    wrist = np.full((16, 20, 3), 241, dtype=np.uint8)
    state = np.arange(7, dtype=np.float32)
    result = DobotRightInputs()({"state": state, "images": {"top": top, "right_wrist": wrist}})
    np.testing.assert_array_equal(result["state"], state)
    np.testing.assert_array_equal(result["image"]["base_0_rgb"], top)
    np.testing.assert_array_equal(result["image"]["right_wrist_0_rgb"], wrist)
    assert "actions" not in result


@pytest.mark.parametrize("shape", [(50, 32), (2, 50, 32)])
def test_output_uses_first_seven_model_dimensions_after_unnormalization(shape):
    from openpi.policies.dobot_right_policy import DobotRightOutputs

    actions = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    stats = {"actions": transforms.NormStats(mean=np.arange(7), std=np.ones(7), q01=np.arange(7), q99=np.arange(7) + 4)}
    stats["state"] = stats["actions"]
    unnormalized = transforms.Unnormalize(stats, use_quantiles=True)(
        {"state": np.zeros((*shape[:-2], 32)), "actions": actions.copy()}
    )
    result = DobotRightOutputs()(unnormalized)
    assert result["actions"].shape == (*shape[:-1], 7)
    np.testing.assert_array_equal(result["actions"], unnormalized["actions"][..., :7])
    np.testing.assert_array_equal(actions, np.arange(np.prod(shape), dtype=np.float32).reshape(shape))


@pytest.mark.parametrize("width", [6, 7, 13, 20, 32])
def test_training_repack_rejects_non_dual_arm_source(width):
    from openpi.policies.dobot_right_policy import DobotRightRepack

    raw = _raw_sample()
    raw["observation.state"] = np.zeros(width, dtype=np.float32)
    with pytest.raises(ValueError, match="14"):
        DobotRightRepack()(raw)


def test_inference_rejects_full_dual_arm_state():
    from openpi.policies.dobot_right_policy import DobotRightInputs

    with pytest.raises(ValueError, match="7"):
        DobotRightInputs()({"state": np.zeros(14), "images": {}})


@pytest.mark.parametrize("width", [7, 13, 20])
def test_training_repack_rejects_wrong_action_width(width):
    from openpi.policies.dobot_right_policy import DobotRightRepack

    raw = _raw_sample()
    raw["action"] = np.zeros((50, width))
    with pytest.raises(ValueError, match="14"):
        DobotRightRepack()(raw)


def test_inference_requires_right_wrist_camera():
    from openpi.policies.dobot_right_policy import DobotRightInputs

    with pytest.raises(KeyError, match="right_wrist"):
        DobotRightInputs()({"state": np.zeros(7), "images": {"top": np.zeros((3, 16, 20))}})
