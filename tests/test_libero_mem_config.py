import numpy as np

import openpi.training.config as config


def _apply(transforms, value):
    for transform in transforms:
        value = transform(value)
    return value


def test_libero_mem_config_has_requested_training_contract():
    cfg = config.get_config("pi05_libero_mem_bowl_two_tasks")

    assert cfg.data.repo_id == "HITdongdong/libero_mem_bowl_two_tasks"
    assert cfg.data.base_config.prompt_from_task is True
    assert cfg.data.extra_delta_transform is False
    assert cfg.model.pi05 is True
    assert cfg.model.action_horizon == 10
    assert cfg.model.discrete_state_input is False
    assert cfg.batch_size == 64
    assert cfg.num_train_steps == 20_000
    assert cfg.save_interval == 5_000
    assert cfg.keep_period == 5_000
    assert cfg.fsdp_devices == 4


def test_libero_mem_config_reuses_8d_state_7d_action_policy_contract(tmp_path):
    cfg = config.get_config("pi05_libero_mem_bowl_two_tasks")
    data = cfg.data.create(tmp_path, cfg.model)
    prompt = "lift the bowl and place it back on the plate 3 times"
    source = {
        "image": np.zeros((3, 256, 256), dtype=np.float32),
        "wrist_image": np.zeros((3, 256, 256), dtype=np.float32),
        "state": np.arange(8, dtype=np.float32),
        "actions": np.arange(70, dtype=np.float32).reshape(10, 7),
        "prompt": prompt,
    }

    repacked = _apply(data.repack_transforms.inputs, source)
    transformed = _apply(data.data_transforms.inputs, repacked)

    np.testing.assert_array_equal(transformed["state"], source["state"])
    np.testing.assert_array_equal(transformed["actions"], source["actions"])
    assert transformed["state"].shape == (8,)
    assert transformed["actions"].shape == (10, 7)
    assert transformed["prompt"] == prompt
    assert transformed["image"]["base_0_rgb"].shape == (256, 256, 3)
    assert transformed["image"]["left_wrist_0_rgb"].shape == (256, 256, 3)

    output = _apply(data.data_transforms.outputs, {"actions": np.zeros((10, 32), dtype=np.float32)})
    assert output["actions"].shape == (10, 7)
