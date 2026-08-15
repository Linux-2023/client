"""Focused tests for the shared Piper dual observation contract."""

from pathlib import Path
import math
import sys

import cv2
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from observation_adapter import CameraSpec
from observation_adapter import DEFAULT_CAMERA_SPECS
from observation_adapter import ObservationAdapter


MODEL_IMAGE_HEIGHT = 224
MODEL_IMAGE_WIDTH = 224


def _encode_jpeg(bgr_image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        bgr_image,
        [cv2.IMWRITE_JPEG_QUALITY, 100],
    )
    assert ok
    return encoded.tobytes()


def test_default_camera_specs_match_the_fixed_aloha_topics_and_zero_rotation():
    assert [
        (spec.name, spec.topic, spec.width, spec.height, spec.rotate_degrees)
        for spec in DEFAULT_CAMERA_SPECS
    ] == [
        ("cam_high", "/camera_f_l/color/image_raw", 224, 224, 0),
        ("cam_left_wrist", "/camera_l/color/image_raw", 224, 224, 0),
        ("cam_right_wrist", "/camera_r/color/image_raw", 224, 224, 0),
    ]


def test_decode_image_converts_bgr_jpeg_to_rgb_with_asymmetric_channels():
    source = np.zeros((32, 32, 3), dtype=np.uint8)
    source[:] = [12, 34, 210]
    adapter = ObservationAdapter(cameras=DEFAULT_CAMERA_SPECS, task="pick")

    decoded = adapter.decode_image(_encode_jpeg(source))

    assert decoded.shape == (MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, 3)
    assert decoded.dtype == np.uint8
    np.testing.assert_allclose(decoded[112, 112], np.array([210, 34, 12]), atol=10)


def test_decode_image_center_pads_a_wider_frame_with_zero_rows():
    source = np.zeros((40, 80, 3), dtype=np.uint8)
    source[:] = [17, 99, 201]
    adapter = ObservationAdapter(cameras=DEFAULT_CAMERA_SPECS, task="pick")

    decoded = adapter.decode_image(_encode_jpeg(source))

    assert decoded.shape == (MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, 3)
    assert np.count_nonzero(decoded[:56]) == 0
    assert np.count_nonzero(decoded[168:]) == 0
    assert np.count_nonzero(decoded[56:168]) > 0


def test_encode_model_image_transposes_rgb_hwc_to_chw_uint8():
    adapter = ObservationAdapter(cameras=DEFAULT_CAMERA_SPECS, task="pick")
    image = np.zeros((MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, 3), dtype=np.uint8)
    image[..., 0] = 1
    image[..., 1] = 2
    image[..., 2] = 3

    model_image = adapter.encode_model_image(image)

    assert model_image.shape == (3, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH)
    assert model_image.dtype == np.uint8
    np.testing.assert_array_equal(model_image[0], np.full((MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH), 1, dtype=np.uint8))
    np.testing.assert_array_equal(model_image[1], np.full((MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH), 2, dtype=np.uint8))
    np.testing.assert_array_equal(model_image[2], np.full((MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH), 3, dtype=np.uint8))


def test_adapt_returns_observation_state_images_timestamps_sync_error_and_task():
    adapter = ObservationAdapter(cameras=DEFAULT_CAMERA_SPECS, task="fallback")
    source = np.zeros((32, 32, 3), dtype=np.uint8)
    source[:] = [10, 20, 30]
    jpeg = _encode_jpeg(source)
    state = np.arange(14, dtype=np.float32)
    timestamps = {spec.name: float(index) + 0.5 for index, spec in enumerate(DEFAULT_CAMERA_SPECS)}

    observation = adapter.adapt(
        {
            "state": state.tolist(),
            "images": {spec.name: jpeg for spec in DEFAULT_CAMERA_SPECS},
            "timestamps": timestamps,
            "sync_error": 12.5,
            "task": "stack",
        }
    )

    assert set(observation) == {"observation.state", "images", "timestamps", "sync_error", "task"}
    np.testing.assert_allclose(observation["observation.state"], state)
    assert observation["observation.state"].dtype == np.float32
    assert set(observation["images"]) == {spec.name for spec in DEFAULT_CAMERA_SPECS}
    for image in observation["images"].values():
        assert image.shape == (3, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH)
        assert image.dtype == np.uint8
    assert observation["timestamps"] == timestamps
    assert observation["sync_error"] == pytest.approx(12.5)
    assert observation["task"] == "stack"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0},
        {"height": 0},
        {"rotate_degrees": 45},
        {"rotate_degrees": -90},
    ],
)
def test_camera_spec_rejects_invalid_dimensions_and_rotation(kwargs):
    with pytest.raises(ValueError):
        CameraSpec("cam_high", "/camera", **kwargs)


@pytest.mark.parametrize("jpeg", [b"", b"not-a-jpeg"])
def test_decode_image_rejects_empty_or_malformed_jpeg(jpeg):
    adapter = ObservationAdapter(cameras=DEFAULT_CAMERA_SPECS, task="pick")

    with pytest.raises(ValueError):
        adapter.decode_image(jpeg)


def test_adapt_rejects_non_finite_state_values():
    adapter = ObservationAdapter(cameras=DEFAULT_CAMERA_SPECS, task="pick")
    jpeg = _encode_jpeg(np.zeros((32, 32, 3), dtype=np.uint8))

    with pytest.raises(ValueError):
        adapter.adapt(
            {
                "state": [0.0] * 13 + [math.nan],
                "images": {spec.name: jpeg for spec in DEFAULT_CAMERA_SPECS},
                "timestamps": {spec.name: float(index) for index, spec in enumerate(DEFAULT_CAMERA_SPECS)},
                "sync_error": 0.0,
                "task": "stack",
            }
        )
@pytest.mark.parametrize("invalid_value", ["12.5", True])
def test_adapt_rejects_non_numeric_sync_error(invalid_value):
    adapter = ObservationAdapter(cameras=DEFAULT_CAMERA_SPECS, task="pick")
    jpeg = _encode_jpeg(np.zeros((32, 32, 3), dtype=np.uint8))

    with pytest.raises(ValueError):
        adapter.adapt(
            {
                "state": [0.0] * 14,
                "images": {spec.name: jpeg for spec in DEFAULT_CAMERA_SPECS},
                "timestamps": {spec.name: float(index) for index, spec in enumerate(DEFAULT_CAMERA_SPECS)},
                "sync_error": invalid_value,
                "task": "stack",
            }
        )


@pytest.mark.parametrize("invalid_value", ["0.5", True])
def test_adapt_rejects_non_numeric_timestamp_values(invalid_value):
    adapter = ObservationAdapter(cameras=DEFAULT_CAMERA_SPECS, task="pick")
    jpeg = _encode_jpeg(np.zeros((32, 32, 3), dtype=np.uint8))
    timestamps = {spec.name: float(index) for index, spec in enumerate(DEFAULT_CAMERA_SPECS)}
    timestamps[DEFAULT_CAMERA_SPECS[0].name] = invalid_value

    with pytest.raises(ValueError):
        adapter.adapt(
            {
                "state": [0.0] * 14,
                "images": {spec.name: jpeg for spec in DEFAULT_CAMERA_SPECS},
                "timestamps": timestamps,
                "sync_error": 0.0,
                "task": "stack",
            }
        )


def test_ros2_piper_dual_yaml_matches_the_fixed_contract():
    config_path = Path(__file__).resolve().parents[1] / "ros2_piper_dual.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["schema_version"] == "piper_dual_ros2_v1"
    assert config["max_sync_error_ms"] == 30
    assert config["jpeg_quality"] == 90
    assert config["model_image"] == {
        "width": 224,
        "height": 224,
        "channel_order": "RGB",
        "layout": "CHW",
        "dtype": "uint8",
    }
    assert [
        (camera["name"], camera["topic"], camera["width"], camera["height"], camera["rotate_degrees"])
        for camera in config["cameras"]
    ] == [
        ("cam_high", "/camera_f_l/color/image_raw", 224, 224, 0),
        ("cam_left_wrist", "/camera_l/color/image_raw", 224, 224, 0),
        ("cam_right_wrist", "/camera_r/color/image_raw", 224, 224, 0),
    ]
    assert config["state"] == {
        "name": "observation.state",
        "dtype": "float32",
        "order": "left_then_right",
        "arms": ["puppet_left_7", "puppet_right_7"],
        "joint_names": ["J1", "J2", "J3", "J4", "J5", "J6", "gripper"],
        "joint_units": ["rad", "rad", "rad", "rad", "rad", "rad", "m"],
    }
    assert config["action"] == {
        "name": "action",
        "dtype": "float32",
        "order": "left_then_right",
        "arms": ["master_left_7", "master_right_7"],
        "joint_names": ["J1", "J2", "J3", "J4", "J5", "J6", "gripper"],
        "joint_units": ["rad", "rad", "rad", "rad", "rad", "rad", "m"],
    }
