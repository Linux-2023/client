import math

import pytest
import torch

from openpi.models_pytorch.rtc_orientation import RtcOrientationGuidance


_MODES = ("wrapped-rpy", "so3")
_DTYPES = (torch.float32, torch.float64)
_RPY_INDICES = (3, 4, 5, 10, 11, 12)


def _guidance(mode):
    return RtcOrientationGuidance(
        mode,
        rpy_scale=[[0.3, 1.7, 2.5], [3.1, 0.8, 1.2]],
        rpy_offset=[[0.7, -0.4, 1.1], [-0.8, 0.3, -1.4]],
    )


def _actions(physical_rpy, guidance, action_dim=32):
    actions = torch.zeros(*physical_rpy.shape[:2], action_dim, dtype=physical_rpy.dtype)
    normalized = (physical_rpy - guidance.rpy_offset) / guidance.rpy_scale
    actions[..., 3:6] = normalized[..., 0, :]
    actions[..., 10:13] = normalized[..., 1, :]
    return actions


def _rpy(actions):
    return torch.stack((actions[..., 3:6], actions[..., 10:13]), dim=-2)


def _tolerance(dtype):
    return 4e-6 if dtype == torch.float32 else 2e-10


def _rotation_matrix(rpy):
    roll, pitch, yaw = rpy
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=torch.float64,
    )


def _matrix_loss(predicted_rpy, previous_rpy):
    relative = _rotation_matrix(previous_rpy) @ _rotation_matrix(predicted_rpy).T
    angle = math.acos(float((relative.trace() - 1) / 2))
    return 0.5 * angle**2


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_physically_identical_minus179_plus181_prefixes_have_same_correction(mode, dtype):
    guidance = _guidance(mode)
    predicted_rpy = torch.zeros(2, 3, 2, 3, dtype=dtype)
    predicted_rpy[..., 2] = math.radians(-175)
    previous_rpy = predicted_rpy.clone()
    previous_rpy[..., 2] = math.radians(-179)
    equivalent_rpy = previous_rpy.clone()
    equivalent_rpy[..., 2] += 2 * math.pi
    weights = torch.tensor([1.0, 0.4, 0.0], dtype=dtype).reshape(1, 3, 1)
    predicted = _actions(predicted_rpy, guidance)

    correction = guidance.correction(predicted, _actions(previous_rpy, guidance), weights)
    equivalent = guidance.correction(predicted, _actions(equivalent_rpy, guidance), weights)

    torch.testing.assert_close(correction, equivalent, atol=_tolerance(dtype), rtol=_tolerance(dtype))
    expected_rpy = torch.zeros_like(previous_rpy)
    expected_rpy[..., 2] = math.radians(-4) / guidance.rpy_scale[:, 2].to(dtype)
    expected_rpy *= weights.unsqueeze(-1)
    torch.testing.assert_close(_rpy(correction), expected_rpy, atol=_tolerance(dtype), rtol=_tolerance(dtype))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_so3_equivalent_full_euler_triples_have_zero_error(dtype):
    guidance = _guidance("so3")
    predicted_rpy = torch.tensor(
        [[[[0.3, 0.7, -0.9], [-1.1, -0.4, 1.6]], [[-0.6, 1.2, 0.2], [0.9, -0.8, -1.3]]]], dtype=dtype
    )
    equivalent_rpy = predicted_rpy.clone()
    equivalent_rpy[..., 0] += math.pi
    equivalent_rpy[..., 1] = math.pi - equivalent_rpy[..., 1]
    equivalent_rpy[..., 2] += math.pi

    correction = guidance.correction(
        _actions(predicted_rpy, guidance), _actions(equivalent_rpy, guidance), torch.ones(1, 2, 1, dtype=dtype)
    )

    torch.testing.assert_close(correction, torch.zeros_like(correction), atol=_tolerance(dtype), rtol=0)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_so3_correction_is_invariant_to_equivalent_target_euler_triples(dtype):
    guidance = _guidance("so3")
    predicted_rpy = torch.tensor([[[[0.1, -0.4, 0.6], [-0.2, 0.5, -0.7]]]], dtype=dtype)
    previous_rpy = torch.tensor([[[[0.4, -0.2, 0.9], [0.1, 0.2, -0.9]]]], dtype=dtype)
    equivalent_rpy = previous_rpy.clone()
    equivalent_rpy[..., 0] += math.pi
    equivalent_rpy[..., 1] = math.pi - equivalent_rpy[..., 1]
    equivalent_rpy[..., 2] += math.pi
    predicted = _actions(predicted_rpy, guidance)
    weights = torch.ones(1, 1, 1, dtype=dtype)

    actual = guidance.correction(predicted, _actions(previous_rpy, guidance), weights)
    equivalent = guidance.correction(predicted, _actions(equivalent_rpy, guidance), weights)

    torch.testing.assert_close(actual, equivalent, atol=_tolerance(dtype), rtol=_tolerance(dtype))


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("angle", [1e-5, 0.025])
def test_single_axis_agreement_uses_physical_gradient_and_unequal_scales(axis, dtype, angle):
    wrapped = _guidance("wrapped-rpy")
    so3 = _guidance("so3")
    predicted_rpy = torch.zeros(2, 2, 2, 3, dtype=dtype)
    previous_rpy = predicted_rpy.clone()
    previous_rpy[..., axis] = torch.tensor([angle, -angle], dtype=dtype)
    predicted = _actions(predicted_rpy, wrapped)
    previous = _actions(previous_rpy, wrapped)
    weights = torch.tensor([[[1.0], [0.3]], [[0.7], [0.0]]], dtype=dtype)
    expected = (previous - predicted) * weights

    wrapped_correction = wrapped.correction(predicted, previous, weights)
    so3_correction = so3.correction(predicted, previous, weights)

    torch.testing.assert_close(wrapped_correction, expected, atol=_tolerance(dtype), rtol=_tolerance(dtype))
    torch.testing.assert_close(so3_correction, expected, atol=_tolerance(dtype), rtol=_tolerance(dtype))


def test_so3_matches_independent_matrix_loss_physical_coordinate_gradient():
    guidance = _guidance("so3")
    predicted_rpy = torch.tensor([[[[0.31, -0.46, 0.72], [-0.61, 0.23, -0.48]]]], dtype=torch.float64)
    previous_rpy = torch.tensor([[[[-0.13, 0.28, 1.06], [0.19, -0.34, 0.17]]]], dtype=torch.float64)
    weights = torch.tensor([[[0.37]]], dtype=torch.float64)
    expected = torch.zeros_like(predicted_rpy)
    epsilon = 1e-6
    for arm in range(2):
        for axis in range(3):
            plus = predicted_rpy[0, 0, arm].tolist()
            minus = plus.copy()
            plus[axis] += epsilon
            minus[axis] -= epsilon
            target = previous_rpy[0, 0, arm].tolist()
            gradient = (_matrix_loss(plus, target) - _matrix_loss(minus, target)) / (2 * epsilon)
            expected[0, 0, arm, axis] = -gradient * 0.37 / float(guidance.rpy_scale[arm, axis])

    actual = guidance.correction(_actions(predicted_rpy, guidance), _actions(previous_rpy, guidance), weights)

    torch.testing.assert_close(_rpy(actual), expected, atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("action_dim", [14, 32])
@pytest.mark.parametrize("shared_weights", [True, False])
def test_xyz_gripper_padding_and_prefix_weights_remain_exact(mode, dtype, action_dim, shared_weights):
    guidance = _guidance(mode)
    generator = torch.Generator().manual_seed(42)
    predicted = torch.randn(2, 4, action_dim, dtype=dtype, generator=generator)
    previous = torch.randn(2, 4, action_dim, dtype=dtype, generator=generator)
    weights = torch.tensor([[[1.0], [0.7], [0.2], [0.0]], [[0.3], [0.4], [0.9], [0.0]]], dtype=dtype)
    if shared_weights:
        weights = weights[:1]
    non_rpy_indices = [index for index in range(action_dim) if index not in _RPY_INDICES]
    expected = (previous - predicted) * weights

    actual = guidance.correction(predicted, previous, weights)
    unweighted = guidance.correction(predicted, previous, torch.ones_like(weights))

    assert actual.shape == predicted.shape
    assert actual.dtype == dtype
    assert actual.device == predicted.device
    assert torch.equal(actual[..., non_rpy_indices], expected[..., non_rpy_indices])
    assert torch.count_nonzero(actual[:, -1]) == 0
    torch.testing.assert_close(actual, unweighted * weights, atol=_tolerance(dtype), rtol=_tolerance(dtype))
    for batch in range(2):
        batch_weights = weights if shared_weights else weights[batch : batch + 1]
        individual = guidance.correction(predicted[batch : batch + 1], previous[batch : batch + 1], batch_weights)
        torch.testing.assert_close(actual[batch : batch + 1], individual, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("angle", [0.0, 1e-9, math.pi - 1e-4, math.pi, math.pi + 1e-4, -math.pi])
def test_so3_identity_and_pi_gradients_are_finite_and_pi_error_is_not_zeroed(dtype, angle):
    guidance = RtcOrientationGuidance("so3", torch.ones(2, 3), torch.zeros(2, 3))
    predicted_rpy = torch.zeros(1, 2, 2, 3, dtype=dtype)
    previous_rpy = predicted_rpy.clone()
    previous_rpy[..., 2] = angle
    predicted = _actions(predicted_rpy, guidance)
    previous = _actions(previous_rpy, guidance)
    weights = torch.tensor([[[1.0], [0.0]]], dtype=dtype)

    actual = guidance.correction(predicted, previous, weights)
    repeated = guidance.correction(predicted, previous, weights)

    assert torch.isfinite(actual).all()
    assert torch.equal(actual, repeated)
    assert torch.count_nonzero(actual[:, 1]) == 0
    expected_magnitude = abs(math.atan2(math.sin(angle), math.cos(angle)))
    torch.testing.assert_close(
        _rpy(actual)[0, 0, :, 2].abs(),
        torch.full((2,), expected_magnitude, dtype=dtype),
        atol=_tolerance(dtype),
        rtol=_tolerance(dtype),
    )
    if 0 < abs(angle) < math.pi or abs(angle) > math.pi:
        expected = math.atan2(math.sin(angle), math.cos(angle))
        torch.testing.assert_close(
            _rpy(actual)[0, 0, :, 2],
            torch.full((2,), expected, dtype=dtype),
            atol=_tolerance(dtype),
            rtol=_tolerance(dtype),
        )


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("outer_grad", [True, False])
def test_correction_does_not_mutate_inputs_or_connect_network_graph(mode, outer_grad):
    guidance = _guidance(mode)
    network = torch.nn.Linear(32, 32, dtype=torch.float64)
    predicted = network(torch.ones(2, 3, 32, dtype=torch.float64))
    previous = torch.full_like(predicted, 0.2, requires_grad=True)
    weights = torch.full((1, 3, 1), 0.4, dtype=torch.float64, requires_grad=True)
    snapshots = [tensor.detach().clone() for tensor in (predicted, previous, weights)]

    with torch.set_grad_enabled(outer_grad):
        actual = guidance.correction(predicted, previous, weights)

    assert not actual.requires_grad
    assert actual.grad_fn is None
    assert previous.grad is None
    assert weights.grad is None
    assert all(parameter.grad is None for parameter in network.parameters())
    for tensor, snapshot in zip((predicted, previous, weights), snapshots, strict=True):
        assert torch.equal(tensor, snapshot)
    predicted.sum().backward()
    assert all(parameter.grad is not None for parameter in network.parameters())


@pytest.mark.parametrize("mode", _MODES)
def test_denoising_correction_does_not_extract_tensor_values_on_host(mode):
    guidance = _guidance(mode)
    predicted = torch.zeros(2, 3, 32)
    previous = torch.ones_like(predicted)
    weights = torch.ones(1, 3, 1)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        guidance.correction(predicted, previous, weights)
        guidance.correction(predicted, previous, weights)

    operations = {event.key for event in profile.key_averages()}
    assert not operations.intersection({"aten::item", "aten::_local_scalar_dense", "aten::is_nonzero"})


@pytest.mark.parametrize("mode", _MODES)
def test_constructor_owns_float32_stats_buffers(mode):
    scale = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
    offset = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    guidance = RtcOrientationGuidance(mode, scale, offset)

    buffers = dict(guidance.named_buffers())
    assert buffers["rpy_scale"].dtype == torch.float32
    assert buffers["rpy_offset"].dtype == torch.float32
    assert buffers["rpy_scale"].shape == (2, 3)
    assert buffers["rpy_offset"].shape == (2, 3)
    assert not buffers["rpy_scale"].requires_grad
    assert not buffers["rpy_offset"].requires_grad
    assert not list(guidance.parameters())
    with torch.no_grad():
        scale.fill_(4)
        offset.fill_(5)
    assert torch.equal(guidance.rpy_scale, torch.ones(2, 3))
    assert torch.equal(guidance.rpy_offset, torch.zeros(2, 3))


@pytest.mark.parametrize("mode", ["legacy", "rpy", "", "SO3"])
def test_constructor_rejects_unsupported_mode(mode):
    with pytest.raises(ValueError, match="mode"):
        RtcOrientationGuidance(mode, torch.ones(2, 3), torch.zeros(2, 3))


@pytest.mark.parametrize("field", ["rpy_scale", "rpy_offset"])
@pytest.mark.parametrize("shape", [(3,), (6,), (1, 3), (2, 4)])
def test_constructor_rejects_wrong_stats_shape(field, shape):
    stats = {"rpy_scale": torch.ones(2, 3), "rpy_offset": torch.zeros(2, 3)}
    stats[field] = torch.ones(shape)
    with pytest.raises(ValueError, match=field):
        RtcOrientationGuidance("so3", **stats)


@pytest.mark.parametrize("field", ["rpy_scale", "rpy_offset"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_constructor_rejects_nonfinite_stats(field, value):
    stats = {"rpy_scale": torch.ones(2, 3), "rpy_offset": torch.zeros(2, 3)}
    stats[field][1, 2] = value
    with pytest.raises(ValueError, match=field):
        RtcOrientationGuidance("so3", **stats)


@pytest.mark.parametrize("value", [0.0, -0.1])
def test_constructor_rejects_nonpositive_scale(value):
    scale = torch.ones(2, 3)
    scale[0, 1] = value
    with pytest.raises(ValueError, match="rpy_scale"):
        RtcOrientationGuidance("so3", scale, torch.zeros(2, 3))
