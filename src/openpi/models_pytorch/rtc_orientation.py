"""Orientation-aware RTC residuals for normalized absolute dual-arm XYZRPY actions.

The action layout is left XYZ/RPY/gripper then right XYZ/RPY/gripper (14D),
optionally followed by model padding. Angles are radians with extrinsic xyz
Euler convention, i.e. R = Rz(yaw) Ry(pitch) Rx(roll).
"""

import torch


def _rpy_to_quaternion(rpy: torch.Tensor) -> torch.Tensor:
    """Convert extrinsic xyz Euler angles to scalar-first (w, x, y, z) quaternions."""
    half_angles = rpy * 0.5
    sr, sp, sy = half_angles.sin().unbind(-1)
    cr, cp, cy = half_angles.cos().unbind(-1)
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=-1,
    )


def _shortest_relative_quaternion(previous: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    """Return the canonical representative of previous * conjugate(predicted).

    Positive scalar selects the shortest arc. At an exactly zero scalar (pi),
    choose the sign making the largest-magnitude vector component positive;
    ties choose x, then y, then z. This is a deterministic one-sided branch,
    not a zero subgradient at the nondifferentiable SO(3) cut locus. Rounding
    near pi can select either side, so the branch is not continuous there.
    """
    previous_w, previous_v = previous[..., :1], previous[..., 1:]
    predicted_w, predicted_v = predicted[..., :1], predicted[..., 1:]
    scalar = (previous * predicted).sum(dim=-1, keepdim=True)
    vector = predicted_w * previous_v - previous_w * predicted_v - torch.linalg.cross(previous_v, predicted_v, dim=-1)
    largest_component = vector.gather(-1, vector.abs().argmax(dim=-1, keepdim=True))
    flip = (scalar < 0) | ((scalar == 0) & (largest_component < 0))
    return torch.cat((scalar, vector), dim=-1) * torch.where(flip, -1.0, 1.0)


def _half_squared_geodesic(relative: torch.Tensor) -> torch.Tensor:
    """Stable .5 * (2 atan2(||v||, w))**2, including its identity gradient."""
    scalar = relative[..., 0]
    squared_norm = relative[..., 1:].square().sum(dim=-1)
    small = squared_norm < 1e-8

    # Mask BEFORE sqrt: selecting a finite loss with where does not protect
    # backward from an unselected sqrt(0). Both candidate branches stay finite.
    norm = torch.sqrt(torch.where(small, 1.0, squared_norm))
    angle = 2 * torch.atan2(norm, scalar)
    safe_scalar = torch.where(small, scalar, 1.0)
    ratio_squared = squared_norm / safe_scalar.square()
    # 2 atan(sqrt(r))**2 = 2r - 4r**2/3 + O(r**3).
    # The omitted loss term is O(1e-24) at this branch's threshold.
    small_loss = 2 * ratio_squared * (1 - (2 / 3) * ratio_squared)
    return torch.where(small, small_loss, 0.5 * angle.square())


class RtcOrientationGuidance(torch.nn.Module):
    """Replace only the six RPY components of the normalized RTC correction.

    ``rpy_scale`` and ``rpy_offset`` are float32 (2, 3) buffers recovering
    physical radians as ``normalized * scale + offset``. Move this module to
    the inference device once before denoising. Constructor validation may
    synchronize; correction never extracts tensor values on the host.

    wrapped-rpy: wrap each physical angular residual to [-pi, pi] with atan2,
    divide by scale, then apply prefix weights. This handles independent 2pi
    aliases, but not coupled Euler equivalences or Euler singularities.

    so3: minimize the prefix-weighted half squared shortest quaternion angle.
    Differentiate ONLY with respect to a detached physical predicted-RPY leaf,
    then return ``-physical_gradient / scale``. This is a physical-coordinate
    gradient followed by unit conversion, equivalently diagonal preconditioning
    of the normalized-coordinate gradient, not full-model backpropagation.
    The rotation loss and target aliases are invariant to equivalent Euler
    triples. The returned gradient still uses the predicted Euler chart: its
    coordinates change with that chart and become rank-deficient at gimbal
    lock. The exact-pi deterministic one-sided branch is documented above.

    Every XYZ, gripper, and padding correction remains exactly the original
    ``(previous - predicted) * weights``. No caller tensor or network graph is
    modified; the returned correction is detached.
    """

    def __init__(self, mode: str, rpy_scale, rpy_offset):
        super().__init__()
        if mode not in ("wrapped-rpy", "so3"):
            raise ValueError(f"Unsupported RTC orientation mode: {mode!r}")
        self.mode = mode
        for name, value in (("rpy_scale", rpy_scale), ("rpy_offset", rpy_offset)):
            tensor = torch.as_tensor(value, dtype=torch.float32).detach().clone()
            if tensor.shape != (2, 3):
                raise ValueError(f"{name} must have shape (2, 3), got {tuple(tensor.shape)}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must contain only finite values")
            if name == "rpy_scale" and not (tensor > 0).all():
                raise ValueError("rpy_scale must contain only positive values")
            self.register_buffer(name, tensor)

    @torch.no_grad()
    def correction(self, predicted: torch.Tensor, previous: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Return (B, T, D>=14) correction; weights have shape (B or 1, T, 1)."""
        if predicted.ndim != 3 or predicted.shape[-1] < 14 or previous.shape != predicted.shape:
            raise ValueError("predicted and previous must have identical (B, T, D>=14) shapes")
        if (
            weights.ndim != 3
            or weights.shape[0] not in (1, predicted.shape[0])
            or weights.shape[1:] != (predicted.shape[1], 1)
        ):
            raise ValueError("weights must have shape (B or 1, T, 1)")
        predicted = predicted.detach()
        previous = previous.detach()
        weights = weights.detach()
        correction = (previous - predicted) * weights
        predicted_rpy = torch.stack((predicted[..., 3:6], predicted[..., 10:13]), dim=-2)
        previous_rpy = torch.stack((previous[..., 3:6], previous[..., 10:13]), dim=-2)

        if self.mode == "wrapped-rpy":
            delta = (previous_rpy - predicted_rpy) * self.rpy_scale
            rpy_correction = torch.atan2(delta.sin(), delta.cos()) / self.rpy_scale
            rpy_correction = rpy_correction * weights.unsqueeze(-1)
        else:
            # Checkpoint normalization may promote only the previous prefix to float64.
            # Match quaternion operands without reducing the prefix's precision.
            predicted_rpy = predicted_rpy.to(correction.dtype) * self.rpy_scale + self.rpy_offset
            previous_rpy = previous_rpy.to(correction.dtype) * self.rpy_scale + self.rpy_offset
            previous_quaternion = _rpy_to_quaternion(previous_rpy)
            with torch.enable_grad():
                predicted_rpy = predicted_rpy.detach().requires_grad_(True)
                predicted_quaternion = _rpy_to_quaternion(predicted_rpy)
                relative = _shortest_relative_quaternion(previous_quaternion, predicted_quaternion)
                loss = (_half_squared_geodesic(relative) * weights).sum()
                (physical_gradient,) = torch.autograd.grad(loss, predicted_rpy)
            rpy_correction = -physical_gradient / self.rpy_scale

        correction[..., 3:6] = rpy_correction[..., 0, :]
        correction[..., 10:13] = rpy_correction[..., 1, :]
        return correction
