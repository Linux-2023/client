"""Shared observation adapter for the Piper dual ROS 2 contract."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import cv2
import numpy as np

MODEL_IMAGE_WIDTH = 224
MODEL_IMAGE_HEIGHT = 224
_ALLOWED_ROTATIONS = {0, 90, 180, 270}


@dataclass(frozen=True, slots=True)
class CameraSpec:
    name: str
    topic: str
    width: int = MODEL_IMAGE_WIDTH
    height: int = MODEL_IMAGE_HEIGHT
    rotate_degrees: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("CameraSpec.name must be a non-empty string")
        if not isinstance(self.topic, str) or not self.topic:
            raise ValueError("CameraSpec.topic must be a non-empty string")
        if type(self.width) is not int or self.width <= 0:
            raise ValueError("CameraSpec.width must be a positive integer")
        if type(self.height) is not int or self.height <= 0:
            raise ValueError("CameraSpec.height must be a positive integer")
        if type(self.rotate_degrees) is not int or self.rotate_degrees not in _ALLOWED_ROTATIONS:
            raise ValueError("CameraSpec.rotate_degrees must be one of 0, 90, 180, or 270")


DEFAULT_CAMERA_SPECS = (
    CameraSpec("cam_high", "/camera_f_l/color/image_raw"),
    CameraSpec("cam_left_wrist", "/camera_l/color/image_raw"),
    CameraSpec("cam_right_wrist", "/camera_r/color/image_raw"),
)


class ObservationAdapter:
    """Adapt bridge-side raw observation payloads into model-facing tensors."""

    def __init__(self, cameras: Sequence[CameraSpec] = DEFAULT_CAMERA_SPECS, task: str = "") -> None:
        camera_specs = tuple(cameras)
        if not camera_specs:
            raise ValueError("ObservationAdapter requires at least one camera spec")
        if type(task) is not str:
            raise ValueError("ObservationAdapter.task must be a string")

        names = [spec.name for spec in camera_specs]
        if len(names) != len(set(names)):
            raise ValueError("Camera names must be unique")

        widths = {spec.width for spec in camera_specs}
        heights = {spec.height for spec in camera_specs}
        if len(widths) != 1 or len(heights) != 1:
            raise ValueError("All camera specs must share the same model image size")

        self._camera_specs = camera_specs
        self._camera_by_name = {spec.name: spec for spec in camera_specs}
        self._target_width = camera_specs[0].width
        self._target_height = camera_specs[0].height
        self._task = task

    @property
    def camera_specs(self) -> tuple[CameraSpec, ...]:
        return self._camera_specs

    def decode_image(self, jpeg: bytes) -> np.ndarray:
        """Decode a JPEG payload into RGB uint8 HWC with the model target size."""
        rgb_image = self._decode_jpeg_to_rgb(jpeg)
        return self._resize_with_padding(rgb_image)

    def encode_model_image(self, image: np.ndarray) -> np.ndarray:
        """Convert an RGB uint8 HWC image into an RGB uint8 CHW model tensor."""
        model_image = np.asarray(image)
        expected_shape = (self._target_height, self._target_width, 3)
        if model_image.shape != expected_shape:
            raise ValueError(f"Model image must have shape {expected_shape}")
        if model_image.dtype != np.uint8:
            raise ValueError("Model image must have dtype uint8")
        return np.transpose(model_image, (2, 0, 1)).copy()

    def adapt(self, message: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(message, dict):
            raise ValueError("Observation message must be a JSON object")

        state = self._parse_state(message)
        images = message.get("images")
        if not isinstance(images, dict):
            raise ValueError("Observation message must include an images object")
        timestamps = message.get("timestamps")
        if not isinstance(timestamps, dict):
            raise ValueError("Observation message must include a timestamps object")

        sync_error = message.get("sync_error")
        if sync_error is None:
            raise ValueError("Observation message must include sync_error")
        sync_error_value = self._coerce_finite_scalar(sync_error, "sync_error")

        task = message.get("task", self._task)
        if type(task) is not str:
            raise ValueError("task must be a string")

        adapted_images: dict[str, np.ndarray] = {}
        adapted_timestamps: dict[str, float] = {}
        for spec in self._camera_specs:
            if spec.name not in images:
                raise ValueError(f"Observation message is missing image data for camera '{spec.name}'")
            if spec.name not in timestamps:
                raise ValueError(f"Observation message is missing a timestamp for camera '{spec.name}'")
            raw_jpeg = images[spec.name]
            raw_timestamp = timestamps[spec.name]
            rgb_image = self._decode_jpeg_to_rgb(raw_jpeg)
            rotated = self._rotate_rgb_image(rgb_image, spec.rotate_degrees)
            adapted_images[spec.name] = self.encode_model_image(self._resize_with_padding(rotated))
            adapted_timestamps[spec.name] = self._coerce_finite_scalar(raw_timestamp, f"timestamps['{spec.name}']")

        return {
            "observation.state": state,
            "images": adapted_images,
            "timestamps": adapted_timestamps,
            "sync_error": sync_error_value,
            "task": task,
        }

    def _parse_state(self, message: dict[str, Any]) -> np.ndarray:
        raw_state = message.get("state", message.get("observation.state"))
        if raw_state is None:
            raise ValueError("Observation message must include state")
        state = np.asarray(raw_state)
        if state.ndim != 1 or state.shape[0] != 14:
            raise ValueError("Observation state must contain exactly fourteen values")
        if state.dtype.kind not in "fiu":
            raise ValueError("Observation state values must be numeric")
        state = state.astype(np.float32, copy=False)
        if not np.isfinite(state).all():
            raise ValueError("Observation state values must be finite")
        return state

    def _decode_jpeg_to_rgb(self, jpeg: bytes) -> np.ndarray:
        if not isinstance(jpeg, (bytes, bytearray, memoryview)):
            raise ValueError("JPEG payload must be bytes")
        jpeg_bytes = bytes(jpeg)
        if not jpeg_bytes:
            raise ValueError("JPEG payload is empty")
        encoded = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("JPEG payload is malformed")
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    def _rotate_rgb_image(self, image: np.ndarray, rotate_degrees: int) -> np.ndarray:
        if rotate_degrees == 0:
            return image
        turns = rotate_degrees // 90
        return np.rot90(image, turns).copy()

    def _resize_with_padding(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("RGB image must have shape HxWx3")

        source_height, source_width, _ = image.shape
        if source_height <= 0 or source_width <= 0:
            raise ValueError("RGB image must have positive dimensions")

        scale = min(self._target_width / source_width, self._target_height / source_height)
        resized_width = max(1, min(self._target_width, int(math.floor(source_width * scale + 0.5))))
        resized_height = max(1, min(self._target_height, int(math.floor(source_height * scale + 0.5))))
        interpolation = cv2.INTER_AREA if resized_width < source_width or resized_height < source_height else cv2.INTER_LINEAR
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)

        padded = np.zeros((self._target_height, self._target_width, 3), dtype=np.uint8)
        top = (self._target_height - resized_height) // 2
        left = (self._target_width - resized_width) // 2
        padded[top : top + resized_height, left : left + resized_width] = resized
        return padded

    def _coerce_finite_scalar(self, value: Any, field: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{field} must be a finite number")
        coerced = float(value)
        if not math.isfinite(coerced):
            raise ValueError(f"{field} must be finite")
        return coerced
