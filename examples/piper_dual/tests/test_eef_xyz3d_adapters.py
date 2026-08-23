"""Tests for 14D XYZ+RPY EEF client adapters."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eef_xyz3d_action_adapter import EefXyz3dActionAdapter
from eef_xyz3d_observation_adapter import EefXyz3dObservationAdapter


def test_xyz3d_action_adapter_splits_dual_arm_poscmd_values() -> None:
    action = np.array(
        [
            0.1,
            -0.2,
            0.3,
            0.4,
            -0.5,
            0.6,
            0.07,
            -0.8,
            0.9,
            1.0,
            -1.1,
            1.2,
            -1.3,
            0.14,
        ],
        dtype=np.float32,
    )

    request = EefXyz3dActionAdapter().to_bridge_request(action)

    assert request["type"] == "publish_eef_action"
    np.testing.assert_allclose(request["left"], action[:7])
    np.testing.assert_allclose(request["right"], action[7:])


@pytest.mark.parametrize(
    ("action", "message"),
    [
        (np.zeros(13), "exactly fourteen"),
        (np.zeros((1, 14)), "exactly fourteen"),
        (["bad"] * 14, "numeric"),
        (np.array([0.0] * 13 + [np.inf]), "finite"),
    ],
)
def test_xyz3d_action_adapter_rejects_invalid_actions(action: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EefXyz3dActionAdapter().to_bridge_request(action)


def _jpeg_bytes(value: int) -> bytes:
    image = np.full((8, 12, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _message() -> dict[str, object]:
    master = np.arange(14, dtype=np.float32) / 100.0
    return {
        "eef": {
            "puppet_left": [0.1, 0.2, 0.3, -0.4, 0.5, -0.6],
            "puppet_right": [-0.7, 0.8, 0.9, 1.0, -1.1, 1.2],
        },
        "action": master,
        "images": {
            "cam_high": _jpeg_bytes(10),
            "cam_left_wrist": _jpeg_bytes(20),
            "cam_right_wrist": _jpeg_bytes(30),
        },
        "timestamps": {
            "cam_high": 1.0,
            "cam_left_wrist": 1.0,
            "cam_right_wrist": 1.0,
        },
        "sync_error": 0.0,
        "task": "Stack the cups",
    }


def test_xyz3d_observation_adapter_builds_exact_14d_state() -> None:
    adapted = EefXyz3dObservationAdapter(task="Stack the cups").adapt(_message())

    expected = np.array(
        [
            0.1,
            0.2,
            0.3,
            -0.4,
            0.5,
            -0.6,
            0.06,
            -0.7,
            0.8,
            0.9,
            1.0,
            -1.1,
            1.2,
            0.13,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(adapted["observation.state"], expected)
    assert adapted["observation.state"].shape == (14,)
    assert adapted["observation.state"].dtype == np.float32
    assert adapted["task"] == "Stack the cups"


Mutator = Callable[[dict[str, object]], None]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda item: item["eef"].update(puppet_left=[0.0] * 5), "six numeric values"),
        (lambda item: item["eef"].update(puppet_right=[0.0] * 5 + [np.nan]), "finite"),
        (lambda item: item.update(action=np.zeros(13)), "fourteen numeric values"),
        (lambda item: item.update(action=np.array([0.0] * 13 + [np.inf])), "finite"),
    ],
)
def test_xyz3d_observation_adapter_rejects_invalid_state_inputs(mutate: Mutator, message: str) -> None:
    item = _message()
    mutate(item)

    with pytest.raises(ValueError, match=message):
        EefXyz3dObservationAdapter().adapt(item)
