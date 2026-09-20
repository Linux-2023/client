import jax
import numpy as np
import pytest
import torch

from openpi import transforms
from openpi.models import model
from openpi.policies.piper_eef_xyz3d_policy import PiperEefXyz3dInputs
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


@pytest.mark.parametrize(
    ("name", "repo_id", "prompt"),
    [
        (
            "pi05_piper_dual_stack_cups_eef_xyz3d_100",
            "HITdongdong/piper_dual_stack_cups_eef_xyz3d_100",
            "Stack the paper cups together.",
        ),
        (
            "pi05_piper_dual_fold_towel_eef_xyz3d_100",
            "HITdongdong/piper_dual_fold_towel_eef_xyz3d_100",
            "Fold the towel.",
        ),
        (
            "pi05_piper_dual_beat_drum_eef_xyz3d_100",
            "HITdongdong/piper_dual_beat_drum_eef_xyz3d_100",
            "Beat the drum three times.",
        ),
        (
            "pi05_piper_dual_block_drawer_eef_xyz3d_100",
            "HITdongdong/piper_dual_block_drawer_eef_xyz3d_100",
            "Put the block in the drawer.",
        ),
        (
            "pi05_piper_dual_weigh_apple_eef_xyz3d_100",
            "HITdongdong/piper_dual_weigh_apple_eef_xyz3d_100",
            "Weigh the apple.",
        ),
    ],
)
def test_eef_xyz3d_100_configs_match_training_contract(name, repo_id, prompt):
    cfg = config.get_config(name)
    data = cfg.data.create(cfg.assets_dirs, cfg.model)

    assert cfg.model.pi05 is True
    assert cfg.model.action_dim == 32
    assert cfg.model.action_horizon == 50
    assert cfg.model.discrete_state_input is False
    assert cfg.data.__class__.__name__ == "LeRobotPiperEefXyz3dDataConfig"
    assert cfg.data.repo_id == repo_id
    assert cfg.data.default_prompt == prompt
    assert cfg.batch_size == 128
    assert cfg.num_workers == 0
    assert cfg.num_train_steps == 50_000
    assert cfg.save_interval == 10_000
    assert cfg.keep_period == 10_000
    assert cfg.fsdp_devices == 4
    assert data.action_sequence_keys == ("action",)


def test_fold_towel_first50_preserves_previous_training_contract():
    import dataclasses

    previous = config.get_config("pi05_piper_dual_fold_towel_eef_xyz3d_100")
    subset = config.get_config("pi05_piper_dual_fold_towel_eef_xyz3d_50")
    assert subset.num_train_steps == 30_000
    assert subset.data.base_config.episodes == tuple(range(50))
    assert previous.data.base_config.episodes is None
    assert (
        dataclasses.replace(
            subset,
            name=previous.name,
            num_train_steps=previous.num_train_steps,
            data=previous.data,
        )
        == previous
    )
    assert dataclasses.replace(subset.data, base_config=previous.data.base_config) == previous.data
    assert dataclasses.replace(subset.data.base_config, episodes=None) == previous.data.base_config
    data = subset.data.create(subset.assets_dirs, subset.model)
    assert data.episodes == tuple(range(50))
    assert data.action_sequence_keys == ("action",)
    assert subset.assets_dirs != previous.assets_dirs
    assert dataclasses.replace(subset, exp_name="first50").checkpoint_dir != (
        dataclasses.replace(previous, exp_name="first50").checkpoint_dir
    )


@pytest.mark.parametrize("episode_count", [10, 20])
def test_fold_towel_small_subsets_preserve_fold50_contract(episode_count):
    import dataclasses

    previous = config.get_config("pi05_piper_dual_fold_towel_eef_xyz3d_50")
    subset = config.get_config(f"pi05_piper_dual_fold_towel_eef_xyz3d_{episode_count}")
    assert subset.num_train_steps == 30_000
    assert subset.data.base_config.episodes == tuple(range(episode_count))
    assert dataclasses.replace(subset, name=previous.name, data=previous.data) == previous
    assert dataclasses.replace(subset.data, base_config=previous.data.base_config) == previous.data
    assert dataclasses.replace(
        subset.data.base_config, episodes=previous.data.base_config.episodes
    ) == previous.data.base_config
    data = subset.data.create(subset.assets_dirs, subset.model)
    assert data.episodes == tuple(range(episode_count))
    assert data.action_sequence_keys == ("action",)
    assert subset.assets_dirs != previous.assets_dirs
    assert dataclasses.replace(subset, exp_name="subset").checkpoint_dir != (
        dataclasses.replace(previous, exp_name="subset").checkpoint_dir
    )


@pytest.mark.parametrize("episode_count", [50, 20, 10])
def test_stack_cups_subsets_preserve_training_contract(episode_count):
    import dataclasses

    previous = config.get_config("pi05_piper_dual_stack_cups_eef_xyz3d_100")
    subset = config.get_config(f"pi05_piper_dual_stack_cups_eef_xyz3d_{episode_count}")
    assert subset.num_train_steps == 30_000
    assert subset.data.base_config.episodes == tuple(range(episode_count))
    assert dataclasses.replace(
        subset, name=previous.name, data=previous.data,
        num_train_steps=previous.num_train_steps,
    ) == previous
    assert dataclasses.replace(subset.data, base_config=previous.data.base_config) == previous.data
    assert dataclasses.replace(subset.data.base_config, episodes=None) == previous.data.base_config
    data = subset.data.create(subset.assets_dirs, subset.model)
    assert data.episodes == tuple(range(episode_count))
    assert data.action_sequence_keys == ("action",)
    assert subset.assets_dirs != previous.assets_dirs
    assert dataclasses.replace(subset, exp_name="subset").checkpoint_dir != (
        dataclasses.replace(previous, exp_name="subset").checkpoint_dir
    )


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


def test_eef_xyz3d_inputs_preserve_rtc_without_rebinding_caller_fields():
    previous_action = np.arange(50 * 14, dtype=np.float32).reshape(50, 14) / 1000
    previous_state = np.linspace(-0.4, 0.4, 14, dtype=np.float32)
    rtc = {"prev_action": previous_action, "prev_state": previous_state, "s": 20, "d": 8}
    result = PiperEefXyz3dInputs()({"state": np.ones(14), "images": _images(), "rtc_obs": rtc})

    assert "rtc_obs" in result
    assert result["rtc_obs"] is not rtc
    np.testing.assert_array_equal(result["rtc_obs"]["prev_action"], previous_action)
    np.testing.assert_array_equal(result["rtc_obs"]["prev_state"], previous_state)
    assert result["rtc_obs"]["s"] == 20
    assert result["rtc_obs"]["d"] == 8

    # The shared padding transform rebinds fields: it must not change the caller's payload.
    transforms.PadStatesAndActions_Prev(32)(result)
    assert result["rtc_obs"]["prev_action"].shape == (50, 32)
    assert rtc["prev_action"].shape == (50, 14)
    assert rtc["prev_state"].shape == (14,)


@pytest.mark.parametrize(
    "config_name",
    ["pi05_piper_dual_stack_cups_eef_xyz3d", "pi05_piper_dual_stack_cups_eef_xyz3d_100"],
)
@pytest.mark.parametrize("with_rtc", [False, True])
def test_eef_xyz3d_serving_transforms_deliver_absolute_rtc_to_model(config_name, with_rtc):
    cfg = config.get_config(config_name)
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    state = np.linspace(-0.7, 0.6, 14, dtype=np.float32)
    previous_state = state + 0.2
    previous_action = np.linspace(-0.4, 0.8, 50 * 14, dtype=np.float32).reshape(50, 14)
    stats = {
        "state": transforms.NormStats(mean=np.zeros(14), std=np.ones(14), q01=np.full(14, -2.0), q99=np.full(14, 4.0)),
        "actions": transforms.NormStats(
            mean=np.zeros(14), std=np.ones(14), q01=np.full(14, -1.0), q99=np.full(14, 3.0)
        ),
    }
    inputs = {"state": state.copy(), "images": _images(), "prompt": "Stack the paper cups together."}
    if with_rtc:
        inputs["rtc_obs"] = {
            "prev_action": previous_action.copy(),
            "prev_state": previous_state.copy(),
            "s": 20,
            "d": 8,
        }
    # Same order as create_trained_policy: data adapter, normalization, model transforms.
    transform = transforms.compose(
        [
            *data.data_transforms.inputs,
            transforms.Normalize(stats, use_quantiles=data.use_quantile_norm),
            *data.model_transforms.inputs,
        ]
    )
    converted = transform(inputs)
    batched = jax.tree.map(lambda value: torch.from_numpy(np.array(value))[None], converted)
    observation = model.Observation.from_dict(batched)

    assert observation.state.shape == (1, 32)
    if not with_rtc:
        assert observation.prev_action is None
        assert observation.prev_state is None
        assert observation.action_horizon is None
        assert observation.actions_during_latency is None
        return

    assert observation.prev_action is not None
    assert observation.prev_action.shape == (1, 50, 32)
    assert observation.prev_state.shape == (1, 32)
    # EEF actions remain absolute: normalize with action stats, without subtracting either state.
    expected_action = (previous_action + 1.0) / (4.0 + 1e-6) * 2.0 - 1.0
    expected_state = (previous_state + 2.0) / (6.0 + 1e-6) * 2.0 - 1.0
    np.testing.assert_allclose(observation.prev_action[0, :, :14].numpy(), expected_action, atol=1e-7)
    np.testing.assert_allclose(observation.prev_state[0, :14].numpy(), expected_state, atol=1e-7)
    assert torch.count_nonzero(observation.prev_action[..., 14:]).item() == 0
    assert torch.count_nonzero(observation.prev_state[..., 14:]).item() == 0
    assert observation.action_horizon.item() == 20
    assert observation.actions_during_latency.item() == 8
    np.testing.assert_array_equal(inputs["rtc_obs"]["prev_action"], previous_action)
    np.testing.assert_array_equal(inputs["rtc_obs"]["prev_state"], previous_state)
