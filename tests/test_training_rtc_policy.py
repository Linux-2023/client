"""Training-RTC request validation, checkpoint capability and action-space regressions."""

import copy
import dataclasses
import json
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from openpi import transforms
from openpi.policies import piper_policy
from openpi.policies import policy as policy_lib
from openpi.policies import policy_config
from openpi.shared import normalize
from openpi.training import config


PRESET = "pi05_piper_dual_tidy_desk_ros2_shifted_pytorch"
CAPABILITY = {"supported": True, "max_prefix_steps": 15, "action_horizon": 50, "action_dim": 14}
CONTRACT = {
    "train_time_rtc": True, "train_time_rtc_max_prefix_frac": 0.3,
    "train_time_rtc_action_dim": 14, "action_dim": 32, "action_horizon": 50,
    "discrete_state_input": False, "use_gp_prior": False, "use_gpr": False,
}
MASK = transforms.make_bool_mask(6, -1, 6, -1)


def _stats():
    values = np.arange(14, dtype=np.float64)
    return normalize.NormStats(mean=values * .1, std=values * .2 + .3, q01=values * .1 - 2, q99=values * .4 + 3)


def _request(prefix_steps=3):
    return {
        "state": np.linspace(-.8, .9, 14, dtype=np.float32),
        "images": {"cam_high": np.zeros((3, 4, 4), dtype=np.uint8)},
        "prompt": "tidy the desk",
        "training_rtc": {"action_prefix": np.linspace(-1.3, 1.4, prefix_steps * 14, dtype=np.float32).reshape(prefix_steps, 14)},
    }


@pytest.mark.parametrize("prefix_steps", [0, 1, 15])
@pytest.mark.parametrize("quantiles", [False, True])
def test_prefix_uses_current_joint_anchor_absolute_grippers_and_action_stats(prefix_steps, quantiles):
    request = _request(prefix_steps)
    original = copy.deepcopy(request)
    stats = {"state": _stats(), "actions": _stats()}
    result = piper_policy.PiperInputs()(copy.deepcopy(request))
    result = transforms.DeltaActions(MASK)(result)
    expected_delta = original["training_rtc"]["action_prefix"].copy()
    expected_delta[:, np.asarray(MASK)] -= original["state"][np.asarray(MASK)]
    np.testing.assert_array_equal(result["training_rtc"]["action_prefix"], expected_delta)
    np.testing.assert_array_equal(result["training_rtc"]["action_prefix"][:, [6, 13]], original["training_rtc"]["action_prefix"][:, [6, 13]])
    normalizer = transforms.Normalize(stats, use_quantiles=quantiles)
    result = transforms.PadStatesAndActions(32)(normalizer(result))
    prefix = result["training_rtc"]["action_prefix"]
    expected = normalizer({"actions": expected_delta})["actions"]
    np.testing.assert_array_equal(prefix[:, :14], expected)
    assert prefix.shape == (prefix_steps, 32)
    assert not prefix[:, 14:].any()
    np.testing.assert_array_equal(request["training_rtc"]["action_prefix"], original["training_rtc"]["action_prefix"])
    # Reanchoring the same absolute actions uses the new request state, never history.
    newer = copy.deepcopy(original)
    newer["state"] += 0.5
    newer = transforms.DeltaActions(MASK)(piper_policy.PiperInputs()(newer))
    np.testing.assert_allclose(newer["training_rtc"]["action_prefix"][:, np.asarray(MASK)], expected_delta[:, np.asarray(MASK)] - .5, atol=2e-7)


@pytest.mark.parametrize("payload", [
    None, {}, {"action_prefix": np.zeros((1, 14), dtype=np.float64)},
    {"action_prefix": np.zeros((1, 14), dtype=np.int32)},
    {"action_prefix": np.zeros((14,), dtype=np.float32)},
    {"action_prefix": np.zeros((1, 0, 14), dtype=np.float32)},
    {"action_prefix": np.zeros((1, 32), dtype=np.float32)},
    {"action_prefix": np.zeros((16, 14), dtype=np.float32)},
    {"action_prefix": np.full((1, 14), np.nan, dtype=np.float32)},
    {"action_prefix": np.full((1, 14), np.inf, dtype=np.float32)},
    {"action_prefix": np.zeros((0, 14), dtype=np.float32), "prefix_length": 0},
])
def test_policy_rejects_bad_payload_before_transforms_or_sampling(payload):
    policy = object.__new__(policy_lib.Policy)
    policy._metadata = {"training_rtc": CAPABILITY}
    policy._is_pytorch_model = True
    policy._sample_kwargs = {}
    policy._input_transform = mock.Mock(side_effect=AssertionError("must reject before transforms"))
    with pytest.raises(ValueError, match="training_rtc"):
        policy.infer({"training_rtc": payload})
    policy._input_transform.assert_not_called()


@pytest.mark.parametrize("field", ["rtc_obs", "prev_action", "gpr_prev_action"])
def test_conflicting_payload_is_rejected_even_when_legacy_value_is_none(field):
    request = _request(0)
    request[field] = None
    with pytest.raises(ValueError, match="combined"):
        transforms.validate_training_rtc_payload(request)


@pytest.mark.parametrize("pytorch,metadata", [(False, {"training_rtc": CAPABILITY}), (True, {}), (True, {"training_rtc": {"supported": False}})])
def test_unsupported_policy_rejects_training_rtc(pytorch, metadata):
    policy = object.__new__(policy_lib.Policy)
    policy._metadata = metadata
    policy._is_pytorch_model = pytorch
    policy._sample_kwargs = {}
    with pytest.raises(ValueError, match="unsupported"):
        policy.infer(_request(0))




class _Tokenizer:
    def __init__(self, *args, **kwargs):
        pass

    def tokenize(self, prompt, state=None, advantage=None):
        assert state is None, "Training-RTC must not enable state tokens"
        return np.array([1, 2], dtype=np.int32), np.array([True, True])


class _CaptureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = SimpleNamespace(to_bfloat16_for_selected_params=lambda precision: None)
        self.observations = []

    def sample_actions(self, device, observation, noise=None):
        self.observations.append(observation)
        result = torch.zeros(1, 50, 32)
        prefix = observation.training_rtc_action_prefix
        if prefix is not None:
            result[:, :prefix.shape[1]] = prefix
        return result


@pytest.fixture
def checkpoint_setup(monkeypatch, tmp_path):
    train_config = config.get_config(PRESET)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").touch()
    (checkpoint / "config.json").write_text(json.dumps(CONTRACT))
    stats = {"actions": _stats(), "state": _stats()}
    normalize.save(checkpoint / "assets" / train_config.data.repo_id, stats)
    # Do not download tokenizer assets or instantiate multi-billion parameter networks.
    monkeypatch.setattr(config._tokenizer, "PaligemmaTokenizer", _Tokenizer)
    monkeypatch.setattr(config.DataConfigFactory, "_load_norm_stats", lambda *args: None)
    model = _CaptureModel()
    monkeypatch.setattr(type(train_config.model), "load_pytorch", lambda *args: model)
    return SimpleNamespace(config=train_config, checkpoint=checkpoint, model=model, stats=stats)


@pytest.mark.parametrize("steps", [0, 1, 15])
def test_policy_pipeline_advertises_checkpoint_support_and_roundtrips_committed_actions(checkpoint_setup, steps):
    setup = checkpoint_setup
    policy = policy_config.create_trained_policy(setup.config, setup.checkpoint, pytorch_device="cpu")
    assert policy.metadata["training_rtc"] == CAPABILITY
    assert setup.model.training_rtc_max_prefix_steps == 15
    request = _request(steps)
    original = copy.deepcopy(request)
    output = policy.infer(request)
    assert output["actions"].shape == (50, 14)
    np.testing.assert_allclose(output["actions"][:steps], original["training_rtc"]["action_prefix"], atol=5e-7)
    np.testing.assert_array_equal(request["training_rtc"]["action_prefix"], original["training_rtc"]["action_prefix"])
    obs = setup.model.observations[-1]
    expected = original["training_rtc"]["action_prefix"].copy()
    expected[:, np.asarray(MASK)] -= original["state"][np.asarray(MASK)]
    expected = transforms.Normalize(setup.stats, use_quantiles=True)({"actions": expected})["actions"]
    np.testing.assert_allclose(obs.training_rtc_action_prefix[0, :, :14].numpy(), expected, rtol=0, atol=1e-12)
    assert not obs.training_rtc_action_prefix[..., 14:].any()
    assert obs.prev_action is None


@pytest.mark.parametrize("declaration", [None, {}, {**CONTRACT, "train_time_rtc": False}])
def test_untrained_or_missing_checkpoint_contract_never_advertises_support(checkpoint_setup, declaration):
    setup = checkpoint_setup
    path = setup.checkpoint / "config.json"
    if declaration is None:
        path.unlink()
    else:
        path.write_text(json.dumps(declaration))
    # A stale static capability must not override the checkpoint's declaration.
    preset = dataclasses.replace(setup.config, policy_metadata={**setup.config.policy_metadata, "training_rtc": CAPABILITY})
    policy = policy_config.create_trained_policy(preset, setup.checkpoint, pytorch_device="cpu")
    assert "training_rtc" not in policy.metadata
    assert setup.model.training_rtc_max_prefix_steps is None
    with pytest.raises(ValueError, match="unsupported"):
        policy.infer(_request(0))
    assert not setup.model.observations
    ordinary = _request()
    ordinary.pop("training_rtc")
    assert policy.infer(ordinary)["actions"].shape == (50, 14)


@pytest.mark.parametrize("field,value", [
    ("train_time_rtc_action_dim", 32), ("action_horizon", 100), ("action_dim", 14),
    ("train_time_rtc_max_prefix_frac", 0.2), ("discrete_state_input", True),
    ("use_gp_prior", True), ("use_gpr", True),
])
def test_incompatible_trained_checkpoint_contract_is_rejected(checkpoint_setup, field, value):
    setup = checkpoint_setup
    (setup.checkpoint / "config.json").write_text(json.dumps({**CONTRACT, field: value}))
    with pytest.raises(ValueError, match="incompatible"):
        policy_config.create_trained_policy(setup.config, setup.checkpoint, pytorch_device="cpu")
    assert not setup.model.observations


def test_unsupported_preset_and_backend_do_not_claim_checkpoint_capability(checkpoint_setup):
    setup = checkpoint_setup
    assert policy_config._training_rtc_capability(setup.config, setup.checkpoint, is_pytorch=False) is None
    assert policy_config._training_rtc_capability(dataclasses.replace(setup.config, name="other"), setup.checkpoint, is_pytorch=True) is None
