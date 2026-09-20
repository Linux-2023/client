"""Right-arm joint-space interface for the dual-arm Dobot digital-scale data."""

import dataclasses

import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class DobotRightRepack(transforms.DataTransformFn):
    """Training-only 14D source selection; inference already supplies a 7D state."""

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation.state"])
        actions = np.asarray(data["action"])
        if state.shape != (14,):
            raise ValueError(f"Expected a 14D Dobot source state, got {state.shape}")
        if actions.ndim != 2 or actions.shape[-1] != 14:
            raise ValueError(f"Expected (horizon, 14) Dobot source actions, got {actions.shape}")
        result = {
            "state": state[7:14],
            "actions": actions[..., 7:14],
            "images": {
                "top": data["observation.images.top"],
                "right_wrist": data["observation.images.right_wrist"],
            },
        }
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected an RGB image, got {image.shape}")
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    return image


@dataclasses.dataclass(frozen=True)
class DobotRightInputs(transforms.DataTransformFn):
    """Seven right-arm joints/gripper values and two cameras; no unit remapping."""

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"])
        if state.shape != (7,):
            raise ValueError(f"Expected a 7D right-arm state, got {state.shape}")
        base = _parse_image(data["images"]["top"])
        wrist = _parse_image(data["images"]["right_wrist"])
        result = {
            "state": state,
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": np.zeros_like(base),
                "right_wrist_0_rgb": wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.ndim != 2 or actions.shape[-1] != 7:
                raise ValueError(f"Expected (horizon, 7) right-arm actions, got {actions.shape}")
            result["actions"] = actions
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class DobotRightOutputs(transforms.DataTransformFn):
    """Return only the learned right-arm slots after shared unnormalization."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim < 2 or actions.shape[-1] < 7:
            raise ValueError(f"Expected at least 7 action dimensions, got {actions.shape}")
        return {"actions": actions[..., :7]}
