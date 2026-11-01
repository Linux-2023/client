"""Numerical Training-RTC sampler regressions without model weights or robot hardware."""

from types import SimpleNamespace

import pytest
import torch
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding


class _Sampler(torch.nn.Module):
    sample_actions = PI0Pytorch.sample_actions
    _prepare_attention_masks_4d = PI0Pytorch._prepare_attention_masks_4d
    embed_suffix = PI0Pytorch.embed_suffix
    _apply_checkpoint = PI0Pytorch._apply_checkpoint

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(action_horizon=50, action_dim=32)
        self.pi05 = True
        self.training_rtc_max_prefix_steps = 15
        self.gradient_checkpointing_enabled = False
        self.action_in_proj = torch.nn.Linear(32, 8)
        self.time_mlp_in = torch.nn.Linear(8, 8)
        self.time_mlp_out = torch.nn.Linear(8, 8)
        self.calls = []
        self.legacy_calls = 0
        self.paligemma_with_expert = SimpleNamespace(
            paligemma=SimpleNamespace(language_model=SimpleNamespace(config=SimpleNamespace())),
            forward=lambda **kwargs: (None, {}),
        )

    def _preprocess_observation(self, observation, *, train=False):
        return (
            [], [], None, None, observation.state, None, observation.prev_action,
            observation.action_horizon, observation.actions_during_latency,
        )

    def embed_prefix(self, *args):
        return torch.zeros(1, 1, 8), torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.bool)

    def denoise_step(self, state, masks, cache, x_t, timestep):
        self.calls.append((x_t.clone(), timestep.clone(), torch.is_grad_enabled()))
        time = timestep[:, None, None] if timestep.ndim == 1 else timestep[..., None]
        # Coupled velocity makes clean conditioning influence suffix predictions.
        return 0.2 * x_t + time + 0.1 * x_t[:, :1].mean(dim=-1, keepdim=True)

    def denoise_step_rtc(self, **kwargs):
        self.legacy_calls += 1
        return torch.zeros_like(kwargs["x_t"])


def _observation(prefix=None):
    return SimpleNamespace(
        state=torch.zeros(1, 32),
        training_rtc_action_prefix=prefix,
        prev_action=None,
        action_horizon=torch.tensor([20]),
        actions_during_latency=torch.tensor([2]),
    )


@pytest.mark.parametrize("steps", [1, 4, 10])
def test_empty_prefix_is_bitwise_ordinary_sampling(steps):
    sampler = _Sampler()
    noise = torch.linspace(-1, 1, 50 * 32).reshape(1, 50, 32)
    before = noise.clone()
    ordinary = sampler.sample_actions("cpu", _observation(), noise=noise, num_steps=steps)
    ordinary_calls = list(sampler.calls)
    sampler.calls.clear()
    empty = sampler.sample_actions("cpu", _observation(torch.empty(1, 0, 32)), noise=noise, num_steps=steps)
    assert torch.equal(empty, ordinary)
    assert torch.equal(noise, before)
    assert len(sampler.calls) == steps
    for (x, time, grad), (old_x, old_time, old_grad) in zip(sampler.calls, ordinary_calls, strict=True):
        assert torch.equal(x, old_x)
        assert torch.equal(time, old_time)
        assert time.shape == (1,)
        assert grad is old_grad is False
    assert sampler.legacy_calls == 0


@pytest.mark.parametrize("prefix_steps", [1, 7, 15])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_clean_prefix_is_pinned_and_zero_time_at_every_step(prefix_steps, dtype):
    sampler = _Sampler()
    noise = torch.linspace(-0.5, 0.5, 50 * 32).reshape(1, 50, 32)
    prefix = torch.linspace(0.1, 0.9, prefix_steps * 32, dtype=dtype).reshape(1, prefix_steps, 32)
    before = prefix.clone()
    output = sampler.sample_actions("cpu", _observation(prefix), noise=noise, num_steps=4)
    assert output.dtype == torch.float32
    assert torch.equal(output[:, :prefix_steps], prefix.float())
    assert torch.equal(prefix, before)
    assert sampler.legacy_calls == 0
    for index, (x, time, grad) in enumerate(sampler.calls):
        assert torch.equal(x[:, :prefix_steps], prefix.float())
        assert time.shape == (1, 50)
        assert torch.count_nonzero(time[:, :prefix_steps]) == 0
        torch.testing.assert_close(time[:, prefix_steps:], torch.full((1, 50 - prefix_steps), 1 - index / 4))
        assert grad is False
    ordinary = _Sampler().sample_actions("cpu", _observation(), noise=noise, num_steps=4)
    assert not torch.equal(output[:, prefix_steps:], ordinary[:, prefix_steps:])


@pytest.mark.parametrize("prefix", [
    torch.zeros(1, 16, 32), torch.zeros(1, 1, 14), torch.zeros(2, 1, 32),
    torch.zeros(1, 32), torch.ones(1, 1, 32, dtype=torch.int64),
    torch.full((1, 1, 32), float("nan")), torch.full((1, 1, 32), float("inf")),
])
def test_sampler_rejects_malformed_prefix_before_network(prefix):
    sampler = _Sampler()
    with pytest.raises(ValueError, match="Training-RTC"):
        sampler.sample_actions("cpu", _observation(prefix), noise=torch.zeros(1, 50, 32))
    assert not sampler.calls


@pytest.mark.parametrize("unsupported", ["pi0", "untrained", "legacy", "orientation"])
def test_sampler_rejects_unsupported_or_conflicting_modes(unsupported):
    sampler = _Sampler()
    obs = _observation(torch.zeros(1, 0, 32))
    kwargs = {}
    if unsupported == "pi0":
        sampler.pi05 = False
    elif unsupported == "untrained":
        sampler.training_rtc_max_prefix_steps = None
    elif unsupported == "legacy":
        obs.prev_action = torch.ones(1, 50, 32)
    else:
        kwargs["rtc_orientation_guidance"] = object()
    with pytest.raises(ValueError, match="Training-RTC"):
        sampler.sample_actions("cpu", obs, noise=torch.zeros(1, 50, 32), **kwargs)
    assert not sampler.calls
    assert sampler.legacy_calls == 0




def test_per_position_time_embedding_and_suffix_condition_match_scalar_positions():
    sampler = _Sampler()
    times = torch.tensor([[0.0, 0.25, 0.75], [0.5, 0.0, 1.0]])
    embedded = create_sinusoidal_pos_embedding(times, 8, 4e-3, 4, device=times.device)
    flattened = create_sinusoidal_pos_embedding(times.flatten(), 8, 4e-3, 4, device=times.device)
    assert torch.equal(embedded, flattened.reshape(2, 3, 8))
    action_times = torch.full((1, 50), 0.7)
    action_times[:, :5] = 0
    actions = torch.zeros(1, 50, 32)
    _, _, _, per_position = sampler.embed_suffix(torch.zeros(1, 32), actions, action_times)
    _, _, _, zero = sampler.embed_suffix(torch.zeros(1, 32), actions, torch.zeros(1))
    _, _, _, current = sampler.embed_suffix(torch.zeros(1, 32), actions, torch.tensor([0.7]))
    torch.testing.assert_close(per_position[:, :5], zero[:, None].expand(-1, 5, -1))
    torch.testing.assert_close(per_position[:, 5:], current[:, None].expand(-1, 45, -1))
    assert current.shape == (1, 8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_adaptive_rmsnorm_per_position_matches_independent_scalar_normalization(dtype):
    norm = GemmaRMSNorm(8, cond_dim=8)
    with torch.no_grad():
        norm.dense.weight.copy_(torch.linspace(-0.4, 0.5, 24 * 8).reshape(24, 8))
        norm.dense.bias.copy_(torch.linspace(-0.2, 0.3, 24))
    x = torch.linspace(-1, 1, 2 * 4 * 8).reshape(2, 4, 8).to(dtype)
    cond = torch.linspace(-0.8, 0.8, 2 * 4 * 8).reshape(2, 4, 8)
    actual, gates = norm(x, cond)
    expected, expected_gates = zip(*(norm(x[:, i:i + 1], cond[:, i]) for i in range(4)), strict=True)
    torch.testing.assert_close(actual, torch.cat(expected, dim=1))
    torch.testing.assert_close(gates, torch.cat(expected_gates, dim=1))
    shared = cond[:, 0]
    shared_actual, shared_gate = norm(x, shared)
    projected = norm.dense(shared).unsqueeze(1)
    scale, shift, expected_gate = projected.chunk(3, dim=-1)
    expected_shared = (norm._norm(x) * (1 + scale.float()) + shift.float()).to(dtype)
    assert torch.equal(shared_actual, expected_shared)
    assert torch.equal(shared_gate, expected_gate.to(dtype))
