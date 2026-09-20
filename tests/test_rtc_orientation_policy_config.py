import dataclasses
import json
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from openpi import transforms
from openpi.policies import policy_config
from openpi.shared import normalize
from openpi.training import config


RPY_INDICES = np.array([[3, 4, 5], [10, 11, 12]])
XYZ_CONFIG = "pi05_piper_dual_stack_cups_eef_xyz3d"


def _stats(width=14):
    values = np.arange(width, dtype=np.float32)
    return normalize.NormStats(
        mean=values * 0.17 - 0.8,
        std=values * 0.13 + 0.4,
        q01=values * 0.2 - 3.0,
        q99=values * 0.9 + 0.7,
    )


@pytest.fixture
def serving_setup(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").touch()
    config_file = checkpoint / "config.json"
    config_file.write_text('{"action_dim":32,"action_horizon":50}')
    train_config = dataclasses.replace(config.get_config(XYZ_CONFIG), policy_metadata={"existing_key": {"keep": True}})
    action_stats = _stats()
    checkpoint_stats = {"actions": action_stats, "state": _stats()}
    asset_id = train_config.data.assets.asset_id or train_config.data.repo_id
    normalize.save(checkpoint / "assets" / asset_id, checkpoint_stats)
    # The config's assets deliberately differ: orientation must use checkpoint stats.
    config_stats = {"actions": _stats(20), "state": _stats(20)}
    monkeypatch.setattr(config.DataConfigFactory, "_load_norm_stats", lambda *args: config_stats)
    # Tokenizer setup can download assets; it is unrelated to affine normalization.
    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda *args: transforms.Group())
    model = SimpleNamespace(paligemma_with_expert=mock.Mock())
    load_pytorch = mock.Mock(return_value=model)
    load_jax = mock.Mock(return_value=model)
    monkeypatch.setattr(type(train_config.model), "load_pytorch", load_pytorch)
    monkeypatch.setattr(type(train_config.model), "load", load_jax)
    monkeypatch.setattr(policy_config._model, "restore_params", mock.Mock(return_value={}))
    download = mock.Mock(return_value=checkpoint)
    monkeypatch.setattr(policy_config.download, "maybe_download", download)
    policy_factory = mock.Mock(side_effect=lambda model, **kwargs: SimpleNamespace(model=model, **kwargs))
    monkeypatch.setattr(policy_config._policy, "Policy", policy_factory)
    return SimpleNamespace(
        checkpoint=checkpoint,
        config_file=config_file,
        train_config=train_config,
        checkpoint_stats=checkpoint_stats,
        asset_id=asset_id,
        load_pytorch=load_pytorch,
        load_jax=load_jax,
        download=download,
        policy_factory=policy_factory,
    )


@pytest.mark.parametrize("mode", ["wrapped-rpy", "so3"])
@pytest.mark.parametrize("quantiles", [True, False])
def test_checkpoint_affine_matches_real_normalization_and_exact_rpy_dims(serving_setup, mode, quantiles):
    setup = serving_setup
    train_config = dataclasses.replace(
        setup.train_config,
        model=dataclasses.replace(setup.train_config.model, pi05=quantiles),
    )
    caller_kwargs = {"num_steps": 4, "rtc_enabled": True}
    policy = policy_config.create_trained_policy(
        train_config,
        setup.checkpoint,
        pytorch_device="cpu",
        rtc_orientation_mode=mode,
        sample_kwargs=caller_kwargs,
    )
    helper = policy.sample_kwargs["rtc_orientation_guidance"]
    assert isinstance(helper, torch.nn.Module)
    assert helper.mode == mode
    assert helper.rpy_scale.device == torch.device("cpu")
    assert helper.rpy_offset.device == torch.device("cpu")
    assert helper.rpy_scale.dtype == torch.float32
    assert helper.rpy_scale.shape == helper.rpy_offset.shape == (2, 3)
    normalizer = next(t for t in policy.transforms if isinstance(t, transforms.Normalize))
    # JSON loading can change array dtype: the actual normalizer's stats are authoritative.
    stats = normalizer.norm_stats["actions"]
    np.testing.assert_array_equal(stats.q01, setup.checkpoint_stats["actions"].q01)
    np.testing.assert_array_equal(stats.q99, setup.checkpoint_stats["actions"].q99)
    if quantiles:
        scale = (stats.q99 - stats.q01 + 1e-6) / 2.0
        offset = stats.q01 + scale
    else:
        scale = stats.std + 1e-6
        offset = stats.mean
    np.testing.assert_array_equal(helper.rpy_scale.cpu().numpy(), scale[RPY_INDICES].astype(np.float32))
    np.testing.assert_array_equal(helper.rpy_offset.cpu().numpy(), offset[RPY_INDICES].astype(np.float32))
    physical = np.linspace(-2.0, 2.0, 28, dtype=np.float32).reshape(2, 14)
    assert normalizer.use_quantiles is quantiles
    normalized = normalizer({"actions": physical})["actions"]
    restored_rpy = normalized[:, RPY_INDICES] * helper.rpy_scale.numpy() + helper.rpy_offset.numpy()
    np.testing.assert_allclose(restored_rpy, physical[:, RPY_INDICES], atol=1e-6)
    assert caller_kwargs == {"num_steps": 4, "rtc_enabled": True}
    assert policy.sample_kwargs is not caller_kwargs
    assert policy.sample_kwargs["num_steps"] == 4
    assert policy.sample_kwargs["rtc_enabled"] is True
    assert policy.pytorch_device == "cpu"
    metadata = policy.metadata
    assert metadata["existing_key"] == {"keep": True}
    assert metadata["policy_config"] == train_config.name
    assert metadata["checkpoint_dir"] == str(setup.checkpoint)
    assert metadata["rtc_orientation"]["mode"] == mode
    assert metadata["rtc_orientation"]["rpy_indices"] == RPY_INDICES.tolist()
    np.testing.assert_allclose(metadata["rtc_orientation"]["rpy_scale"], scale[RPY_INDICES])
    np.testing.assert_allclose(metadata["rtc_orientation"]["rpy_offset"], offset[RPY_INDICES])
    expected_semantics = {
        "wrapped-rpy": "wrapped-physical-residual-divided-by-scale",
        "so3": "negative-physical-rpy-gradient-divided-by-scale",
    }
    assert metadata["rtc_orientation"]["gradient_semantics"] == expected_semantics[mode]
    json.dumps(metadata, allow_nan=False)
    assert setup.train_config.policy_metadata == {"existing_key": {"keep": True}}
    assert setup.config_file.read_text() == '{"action_dim":32,"action_horizon":50}'


def test_helper_is_explicitly_placed_on_requested_device(serving_setup, monkeypatch):
    from openpi.models_pytorch.rtc_orientation import RtcOrientationGuidance

    placed_on = []

    def capture_to(self, device):
        placed_on.append(device)
        return self

    monkeypatch.setattr(RtcOrientationGuidance, "to", capture_to)
    policy = policy_config.create_trained_policy(
        serving_setup.train_config,
        serving_setup.checkpoint,
        pytorch_device="cuda:7",
        rtc_orientation_mode="so3",
    )
    assert placed_on == ["cuda:7"]
    assert policy.pytorch_device == "cuda:7"


def test_explicit_norm_stats_drive_both_transform_and_orientation(serving_setup):
    override = _stats()
    override.q99 = override.q99 + 5.0
    override_stats = {"actions": override, "state": _stats()}
    policy = policy_config.create_trained_policy(
        serving_setup.train_config,
        serving_setup.checkpoint,
        norm_stats=override_stats,
        pytorch_device="cpu",
        rtc_orientation_mode="wrapped-rpy",
    )
    normalizer = next(t for t in policy.transforms if isinstance(t, transforms.Normalize))
    assert normalizer.norm_stats is override_stats
    np.testing.assert_allclose(
        policy.sample_kwargs["rtc_orientation_guidance"].rpy_scale.numpy(),
        ((override.q99 - override.q01 + 1e-6) / 2.0)[RPY_INDICES],
    )


@pytest.mark.parametrize("pytorch", [True, False])
def test_legacy_leaves_sampling_kwargs_and_non_xyz_models_unchanged(serving_setup, pytorch):
    setup = serving_setup
    if not pytorch:
        (setup.checkpoint / "model.safetensors").unlink()
    train_config = dataclasses.replace(
        setup.train_config, data=config.LeRobotPiperEefRot6dDataConfig(repo_id=setup.asset_id)
    )
    caller_kwargs = {"num_steps": 5}
    policy = policy_config.create_trained_policy(
        train_config, setup.checkpoint, sample_kwargs=caller_kwargs, pytorch_device="cpu"
    )
    assert policy.sample_kwargs is caller_kwargs
    assert "rtc_orientation_guidance" not in policy.sample_kwargs
    assert policy.is_pytorch is pytorch
    assert policy.pytorch_device == ("cpu" if pytorch else None)
    assert policy.metadata["rtc_orientation"] == {"mode": "legacy", "gradient_semantics": "normalized-euclidean"}
    assert policy.metadata["policy_config"] == train_config.name
    assert policy.metadata["checkpoint_dir"] == str(setup.checkpoint)
    assert policy.metadata["existing_key"] == {"keep": True}


def test_invalid_mode_is_rejected_before_checkpoint_resolution(serving_setup):
    setup = serving_setup
    with pytest.raises(ValueError, match="rtc_orientation_mode"):
        policy_config.create_trained_policy(setup.train_config, setup.checkpoint, rtc_orientation_mode="quat")
    setup.download.assert_not_called()
    setup.load_pytorch.assert_not_called()
    setup.load_jax.assert_not_called()


@pytest.mark.parametrize("mode", ["wrapped-rpy", "so3"])
def test_non_xyz_config_is_rejected_before_checkpoint_resolution(serving_setup, mode):
    setup = serving_setup
    train_config = dataclasses.replace(
        setup.train_config, data=config.LeRobotPiperEefRot6dDataConfig(repo_id=setup.asset_id)
    )
    with pytest.raises(ValueError, match="XYZ|Xyz3d"):
        policy_config.create_trained_policy(train_config, setup.checkpoint, rtc_orientation_mode=mode)
    setup.download.assert_not_called()
    setup.load_pytorch.assert_not_called()
    setup.load_jax.assert_not_called()


@pytest.mark.parametrize("mode", ["wrapped-rpy", "so3"])
def test_jax_checkpoint_is_rejected_before_loading_weights(serving_setup, mode):
    setup = serving_setup
    (setup.checkpoint / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="PyTorch"):
        policy_config.create_trained_policy(setup.train_config, setup.checkpoint, rtc_orientation_mode=mode)
    setup.load_pytorch.assert_not_called()
    setup.load_jax.assert_not_called()


@pytest.mark.parametrize("mode", ["legacy", "wrapped-rpy", "so3"])
def test_reserved_guidance_kwarg_is_rejected_without_clobbering_caller(serving_setup, mode):
    setup = serving_setup
    sentinel = object()
    kwargs = {"rtc_orientation_guidance": sentinel, "num_steps": 7}
    with pytest.raises(ValueError, match="rtc_orientation_guidance"):
        policy_config.create_trained_policy(
            setup.train_config, setup.checkpoint, rtc_orientation_mode=mode, sample_kwargs=kwargs
        )
    assert kwargs == {"rtc_orientation_guidance": sentinel, "num_steps": 7}
    setup.load_pytorch.assert_not_called()
    setup.load_jax.assert_not_called()


@pytest.mark.parametrize(
    ("quantiles", "field", "bad_value"),
    [
        (True, "q01", None),
        (True, "q99", None),
        (True, "q01", np.zeros(13)),
        (True, "q99", np.ones(20)),
        (True, "q01", np.zeros((1, 14))),
        (True, "q99", "nan"),
        (True, "q01", "inf"),
        (True, "q99", "zero_range"),
        (True, "q99", "negative_range"),
        (False, "mean", "inf"),
        (False, "std", "nan"),
        (False, "mean", np.zeros(32)),
        (False, "std", "zero"),
        (False, "std", "negative"),
    ],
)
def test_invalid_rotational_stats_fail_clearly(serving_setup, quantiles, field, bad_value):
    setup = serving_setup
    train_config = dataclasses.replace(
        setup.train_config, model=dataclasses.replace(setup.train_config.model, pi05=quantiles)
    )
    stats = _stats()
    if isinstance(bad_value, str):
        values = getattr(stats, field).copy()
        replacement = {
            "nan": np.nan,
            "inf": np.inf,
            "zero": 0.0,
            "negative": -1.0,
            "zero_range": stats.q01[10],
            "negative_range": stats.q01[10] - 1.0,
        }[bad_value]
        values[10] = replacement
    else:
        values = bad_value
    setattr(stats, field, values)
    with pytest.raises(ValueError, match="RTC orientation.*(stats|normalization)"):
        policy_config.create_trained_policy(
            train_config,
            setup.checkpoint,
            rtc_orientation_mode="so3",
            norm_stats={"actions": stats, "state": _stats()},
            pytorch_device="cpu",
        )
    setup.policy_factory.assert_not_called()


def test_missing_action_stats_are_rejected(serving_setup):
    with pytest.raises(ValueError, match="RTC orientation.*actions"):
        policy_config.create_trained_policy(
            serving_setup.train_config,
            serving_setup.checkpoint,
            rtc_orientation_mode="wrapped-rpy",
            norm_stats={"state": _stats()},
            pytorch_device="cpu",
        )
