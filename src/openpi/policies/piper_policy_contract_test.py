"""Contract tests for the dual-arm Piper PI05 policy configuration."""

import numpy as np
from openpi.policies import piper_policy
from openpi.training import config as training_config


MAGIC_ASSETS = "/home/agilex/lgd/lgd_checkpoints/pi05_magic_cobot_pick_jax_step2000/assets"
MAGIC_ASSET_ID = "HITdongdong/magic_cobot_pick_lerobot"


def test_make_piper_example_uses_dual_arm_state() -> None:
    example = piper_policy.make_piper_example()

    assert example["state"].shape == (14,)


def test_piper_outputs_returns_full_dual_arm_action_from_padded_model_output() -> None:
    actions = np.arange(64, dtype=np.float32).reshape(2, 32)

    result = piper_policy.PiperOutputs()(dict(actions=actions))

    np.testing.assert_array_equal(result["actions"], actions[:, :14])


def test_magic_cobot_config_uses_local_normalization_assets() -> None:
    config = training_config.get_config("pi05_magic_cobot_pick")
    data_config = config.data.create(config.assets_dirs, config.model)

    assert config.model.action_dim == 32
    assert config.data.assets.asset_id == MAGIC_ASSET_ID
    assert config.data.assets.assets_dir == MAGIC_ASSETS
    assert data_config.asset_id == MAGIC_ASSET_ID
    assert data_config.norm_stats is not None
    assert tuple(data_config.data_transforms.inputs[1].mask) == (
        True, True, True, True, True, True, False,
        True, True, True, True, True, True, False,
    )


def test_piper_outputs_supports_explicit_eef_action_width():
    actions = np.arange(64, dtype=np.float32).reshape(2, 32)

    result = piper_policy.PiperOutputs(action_dim=20)(dict(actions=actions))

    np.testing.assert_array_equal(result["actions"], actions[:, :20])
    assert result["actions"].shape == (2, 20)
