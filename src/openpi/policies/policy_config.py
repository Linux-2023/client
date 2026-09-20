import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms

_RPY_INDICES = np.array([[3, 4, 5], [10, 11, 12]])


def _rtc_rpy_affine(
    norm_stats: dict[str, transforms.NormStats] | None, *, use_quantiles: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the action normalizer for the two absolute XYZ+RPY orientations."""
    if norm_stats is None or "actions" not in norm_stats:
        raise ValueError("RTC orientation normalization stats require an 'actions' entry.")
    stats = norm_stats["actions"]
    fields = ("q01", "q99") if use_quantiles else ("mean", "std")
    values = []
    for field in fields:
        value = getattr(stats, field, None)
        if value is None:
            raise ValueError(f"RTC orientation normalization stats require actions.{field}.")
        try:
            array = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"RTC orientation normalization stats actions.{field} must be numeric.") from exc
        if array.dtype.kind not in "fiu":
            raise ValueError(f"RTC orientation normalization stats actions.{field} must be numeric.")
        if array.shape != (14,) or not np.isfinite(array).all():
            raise ValueError(f"RTC orientation normalization stats actions.{field} must be finite with shape (14,).")
        values.append(array[_RPY_INDICES])
    # Match Normalize's arithmetic dtype; cast only the resulting helper buffers.

    if use_quantiles:
        q01, q99 = values
        span = q99 - q01
        if not np.isfinite(span).all() or np.any(span <= 0):
            raise ValueError("RTC orientation normalization stats require q99 > q01 for every RPY dimension.")
        scale = (span + 1e-6) / 2.0
        offset = q01 + scale
    else:
        offset, std = values
        if np.any(std <= 0):
            raise ValueError("RTC orientation normalization stats require positive std for every RPY dimension.")
        scale = std + 1e-6

    with np.errstate(over="ignore", invalid="ignore"):
        scale, offset = scale.astype(np.float32), offset.astype(np.float32)
    if not np.isfinite(scale).all() or not np.isfinite(offset).all() or np.any(scale <= 0):
        raise ValueError("RTC orientation normalization stats must produce finite float32 RPY scale and offset.")
    return scale, offset


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
    rtc_orientation_mode: str = "legacy",
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".
        rtc_orientation_mode: Server-side RTC guidance: "legacy" (unchanged), "wrapped-rpy", or "so3".
            New schemes require an absolute 14D XYZ+RPY Piper config and a PyTorch checkpoint.

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    if rtc_orientation_mode not in ("legacy", "wrapped-rpy", "so3"):
        raise ValueError(
            f"Invalid rtc_orientation_mode: {rtc_orientation_mode!r}; expected legacy, wrapped-rpy, or so3."
        )
    if sample_kwargs is not None and "rtc_orientation_guidance" in sample_kwargs:
        raise ValueError(
            "rtc_orientation_guidance is managed by rtc_orientation_mode; do not supply it in sample_kwargs."
        )
    if rtc_orientation_mode != "legacy" and not isinstance(train_config.data, _config.LeRobotPiperEefXyz3dDataConfig):
        raise ValueError(
            "RTC orientation guidance requires LeRobotPiperEefXyz3dDataConfig (absolute 14D XYZ+RPY actions)."
        )
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)
    if rtc_orientation_mode != "legacy" and not is_pytorch:
        raise ValueError("RTC orientation guidance requires a PyTorch checkpoint containing model.safetensors.")

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    orientation_metadata: dict[str, Any] = {"mode": rtc_orientation_mode, "gradient_semantics": "normalized-euclidean"}
    if rtc_orientation_mode != "legacy":
        from openpi.models_pytorch.rtc_orientation import RtcOrientationGuidance

        rpy_scale, rpy_offset = _rtc_rpy_affine(norm_stats, use_quantiles=data_config.use_quantile_norm)
        guidance = RtcOrientationGuidance(rtc_orientation_mode, rpy_scale, rpy_offset).to(pytorch_device)
        sample_kwargs = {**(sample_kwargs or {}), "rtc_orientation_guidance": guidance}
        orientation_metadata.update(
            rpy_indices=_RPY_INDICES.tolist(),
            rpy_scale=rpy_scale.tolist(),
            rpy_offset=rpy_offset.tolist(),
            gradient_semantics=(
                "wrapped-physical-residual-divided-by-scale"
                if rtc_orientation_mode == "wrapped-rpy"
                else "negative-physical-rpy-gradient-divided-by-scale"
            ),
        )
    metadata = {
        **(train_config.policy_metadata or {}),
        "policy_config": train_config.name,
        "checkpoint_dir": str(checkpoint_dir),
        "rtc_orientation": orientation_metadata,
    }

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
