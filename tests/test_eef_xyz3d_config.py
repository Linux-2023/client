import numpy as np
import pytest

import openpi.training.config as config


def _images():
    return {
        "cam_high": np.zeros((3, 224, 224), dtype=np.uint8),
        "cam_left_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
        "cam_right_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
    }


def test_eef_xyz3d_config_matches_training_contract():
    cfg = config.get_config("pi05_piper_dual_stack_cups_eef_xyz3d")
    data = cfg.data.create(cfg.assets_dirs, cfg.model)

    assert cfg.model.pi05 is True
    assert cfg.model.action_dim == 32
    assert cfg.model.action_horizon == 50
    assert cfg.model.discrete_state_input is False
    assert cfg.data.__class__.__name__ == "LeRobotPiperEefXyz3dDataConfig"
    assert cfg.data.repo_id == "HITdongdong/piper_dual_stack_cups_eef_xyz3d"
    assert cfg.batch_size == 64
    assert cfg.num_workers == 0
    assert cfg.num_train_steps == 30_000
    assert cfg.save_interval == 5_000
    assert cfg.keep_period == 5_000
    assert cfg.fsdp_devices == 4
    assert data.action_sequence_keys == ("action",)


def test_eef_xyz3d_transforms_preserve_14d_contract():
    cfg = config.get_config("pi05_piper_dual_stack_cups_eef_xyz3d")
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    inputs = data.data_transforms.inputs[0]
    outputs = data.data_transforms.outputs[0]

    transformed = inputs(
        {
            "state": np.zeros(14),
            "actions": np.zeros((50, 14)),
            "images": _images(),
        }
    )

    assert transformed["state"].shape == (14,)
    assert transformed["actions"].shape == (50, 14)
    assert outputs({"actions": np.zeros((50, 32))})["actions"].shape == (50, 14)

    with pytest.raises(ValueError, match="14D XYZ3D state"):
        inputs({"state": np.zeros(20), "images": _images()})
