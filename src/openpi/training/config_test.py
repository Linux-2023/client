from openpi.training import config as _config


def test_pi05_piper_dual_stack_cups_config_matches_training_contract():
    config = _config.get_config("pi05_piper_dual_stack_cups")

    assert config.model.pi05 is True
    assert config.model.action_horizon == 50
    assert config.model.discrete_state_input is False
    assert config.data.repo_id == "HITdongdong/piper_dual_stack_cups_ros2"
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
    assert config.data.output_action_dim == 20
    assert data_config.asset_id == "HITdongdong/piper_dual_stack_cups_eef_rot6d_shifted"
    assert config.data.default_prompt == "Stack the paper cups together."

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

    assert data_config.data_transforms.outputs[0].action_dim == 20
