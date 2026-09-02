import numpy as np
import pytest

import openpi.training.config as config
from openpi.policies import piper_eef_policy


def _images():
    return {
        "cam_high": np.zeros((3, 224, 224), dtype=np.uint8),
        "cam_left_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
        "cam_right_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
    }


def test_eef_inputs_require_20d_state_and_actions():
    transform = piper_eef_policy.PiperEefRot6dInputs()

    with pytest.raises(ValueError, match="20D EEF state"):
        transform({"state": np.zeros(14), "images": _images()})

    with pytest.raises(ValueError, match="20D EEF actions"):
        transform({"state": np.zeros(20), "actions": np.zeros((50, 14)), "images": _images()})


def test_eef_inputs_and_outputs_keep_20d_data_contract():
    inputs = piper_eef_policy.PiperEefRot6dInputs()
    outputs = piper_eef_policy.PiperEefRot6dOutputs()
    transformed = inputs(
        {
            "state": np.zeros(20),
            "actions": np.zeros((50, 20)),
            "images": _images(),
        }
    )

    assert transformed["state"].shape == (20,)
    assert transformed["actions"].shape == (50, 20)
    assert outputs({"actions": np.zeros((50, 32))})["actions"].shape == (50, 20)


def test_eef_config_is_separate_from_joint_config():
    cfg = config.get_config("pi05_piper_dual_stack_cups_eef_rot6d")
    data = cfg.data.create(cfg.assets_dirs, cfg.model)

    assert cfg.model.pi05 is True
    assert cfg.model.action_dim == 32
    assert cfg.model.action_horizon == 50
    assert cfg.batch_size == 64
    assert cfg.data.__class__.__name__ == "LeRobotPiperEefRot6dDataConfig"
    assert data.action_sequence_keys == ("action",)
    assert data.episodes is None


def test_eef_20ep_config_selects_first_twenty_episodes():
    cfg = config.get_config("pi05_piper_dual_stack_cups_eef_rot6d_20ep")
    data = cfg.data.create(cfg.assets_dirs, cfg.model)

    assert cfg.model.action_dim == 32
    assert cfg.num_train_steps == 20_000
    assert cfg.fsdp_devices == 4
    assert data.episodes == tuple(range(20))
    assert data.asset_id == "HITdongdong/piper_dual_stack_cups_eef_rot6d_shifted_20ep"
