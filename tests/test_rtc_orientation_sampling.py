"""Exercise RTC sampler dispatch without model weights or hardware."""

from types import SimpleNamespace

import pytest
import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


class _Sampler(torch.nn.Module):
    sample_actions = PI0Pytorch.sample_actions
    denoise_step_rtc = PI0Pytorch.denoise_step_rtc
    get_prefix_weights = PI0Pytorch.get_prefix_weights
    _linweights = PI0Pytorch._linweights
    _add_trailing_zeros = PI0Pytorch._add_trailing_zeros
    _add_leading_ones = PI0Pytorch._add_leading_ones
    _prepare_attention_masks_4d = PI0Pytorch._prepare_attention_masks_4d

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(action_horizon=50, action_dim=32)
        self.parameter = torch.nn.Parameter(torch.tensor(0.1))
        self.network_grad_modes = []
        self.paligemma_with_expert = SimpleNamespace(
            paligemma=SimpleNamespace(language_model=SimpleNamespace(config=SimpleNamespace())),
            forward=lambda **kwargs: (None, {}),
        )

    def _preprocess_observation(self, observation, *, train=False):
        return (
            [],
            [],
            None,
            None,
            observation.state,
            None,
            observation.prev_action,
            observation.action_horizon,
            observation.actions_during_latency,
        )

    def embed_prefix(self, *args):
        return torch.zeros(1, 1, 1), torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.bool)

    def denoise_step(self, state, masks, cache, x_t, timestep):
        self.network_grad_modes.append(torch.is_grad_enabled())
        return x_t * self.parameter


class _Correction:
    def __init__(self):
        self.calls = []

    def correction(self, predicted, previous, weights):
        assert not predicted.requires_grad
        self.calls.append((predicted.clone(), previous.clone(), weights.clone()))
        return (previous - predicted) * weights


def _observation(with_prefix):
    return SimpleNamespace(
        state=torch.zeros(1, 32),
        prev_action=torch.full((1, 50, 32), 0.3) if with_prefix else None,
        action_horizon=torch.tensor([20]),
        actions_during_latency=torch.tensor([12]),
    )


@pytest.mark.parametrize("with_prefix", [False, True])
def test_sampler_only_calls_orientation_guidance_for_rtc_prefix(with_prefix):
    sampler = _Sampler()
    correction = _Correction()
    result = sampler.sample_actions(
        "cpu",
        _observation(with_prefix),
        noise=torch.zeros(1, 50, 32),
        rtc_orientation_guidance=correction,
    )
    assert result.shape == (1, 50, 32)
    assert torch.isfinite(result).all()
    assert len(correction.calls) == (10 if with_prefix else 0)
    assert all(not enabled for enabled in sampler.network_grad_modes)
    assert sampler.parameter.grad is None
    if with_prefix:
        predicted, previous, weights = correction.calls[0]
        assert previous.shape == predicted.shape == (1, 50, 32)
        assert weights.shape == (1, 50, 1)
        torch.testing.assert_close(weights[:, :12], torch.ones(1, 12, 1))
        assert torch.count_nonzero(weights[:, 20:]) == 0


def test_orientation_branch_preserves_legacy_schedule_for_linear_correction():
    sampler = _Sampler()
    noise = torch.linspace(-0.1, 0.1, 50 * 32).reshape(1, 50, 32)
    legacy = sampler.sample_actions("cpu", _observation(True), noise=noise.clone())
    new = sampler.sample_actions("cpu", _observation(True), noise=noise.clone(), rtc_orientation_guidance=_Correction())
    torch.testing.assert_close(new, legacy, atol=1e-6, rtol=1e-6)


def test_orientation_branch_does_not_build_network_backward_graph():
    sampler = _Sampler()
    correction = _Correction()
    result = sampler.denoise_step_rtc(
        x_t=torch.zeros(1, 50, 32),
        prev_chunk_left_over=torch.ones(1, 50, 32),
        inference_delay=12,
        time=torch.tensor([0.5]),
        execution_horizon=20,
        num_steps=10,
        mask_schedule="exp",
        original_denoise_step_partial=lambda x: sampler.denoise_step(None, None, None, x, None),
        rtc_orientation_guidance=correction,
    )
    assert sampler.network_grad_modes == [False]
    assert not result.requires_grad
    assert len(correction.calls) == 1


@pytest.mark.parametrize("mode", ["wrapped-rpy", "so3"])
def test_float64_checkpoint_prefix_keeps_float32_sampling_state(mode):
    from openpi.models_pytorch.rtc_orientation import RtcOrientationGuidance

    sampler = _Sampler()
    observation = _observation(True)
    observation.prev_action = observation.prev_action.double()
    guidance = RtcOrientationGuidance(mode, torch.ones(2, 3), torch.zeros(2, 3))
    result = sampler.sample_actions("cpu", observation, noise=torch.zeros(1, 50, 32), rtc_orientation_guidance=guidance)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()


def test_orientation_branch_supports_unbatched_actions():
    sampler = _Sampler()
    result = sampler.denoise_step_rtc(
        x_t=torch.zeros(50, 32),
        prev_chunk_left_over=torch.ones(50, 32),
        inference_delay=12,
        time=torch.tensor([0.5]),
        execution_horizon=20,
        num_steps=10,
        mask_schedule="exp",
        original_denoise_step_partial=torch.zeros_like,
        rtc_orientation_guidance=_Correction(),
    )
    assert result.shape == (50, 32)
    assert torch.isfinite(result).all()
