"""Tests for the dual-arm EEF policy adapters."""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eef_action_adapter import EefActionAdapter
from eef_observation_adapter import EefObservationAdapter


def _jpeg() -> bytes:
    image = np.full((32, 32, 3), 80, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _message() -> dict:
    return {
        "eef": {
            "puppet_left": np.array([0.2, -0.1, 0.3, 0.0, 0.0, 0.0], dtype=np.float32),
            "puppet_right": np.array([0.25, 0.1, 0.35, 0.0, 0.0, np.pi / 2], dtype=np.float32),
        },
        "action": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05], dtype=np.float32),
        "images": {"cam_high": _jpeg(), "cam_left_wrist": _jpeg(), "cam_right_wrist": _jpeg()},
        "timestamps": {"cam_high": 1.0, "cam_left_wrist": 1.0, "cam_right_wrist": 1.0},
        "sync_error": 0.01,
        "task": "stack cups",
    }


def test_eef_observation_adapter_builds_20d_state_with_master_grippers() -> None:
    observation = EefObservationAdapter(task="stack cups").adapt(_message())

    state = observation["observation.state"]
    assert state.shape == (20,)
    assert state.dtype == np.float32
    np.testing.assert_allclose(state[:3], [0.2, -0.1, 0.3])
    assert state[9] == pytest.approx(0.04)
    np.testing.assert_allclose(state[10:13], [0.25, 0.1, 0.35])
    assert state[19] == pytest.approx(0.05)
    assert observation["task"] == "stack cups"


def test_eef_action_adapter_encodes_poscmd_values() -> None:
    action = np.array(
        [0.2, -0.1, 0.3, 1, 0, 0, 0, 1, 0, 0.04, 0.25, 0.1, 0.35, 0, 1, 0, -1, 0, 0, 0.05],
        dtype=np.float32,
    )
    request = EefActionAdapter().to_bridge_request(action)

    assert request["type"] == "publish_eef_action"
    assert request["left"] == pytest.approx([0.2, -0.1, 0.3, 0.0, 0.0, 0.0, 0.04])
    assert request["right"] == pytest.approx([0.25, 0.1, 0.35, 0.0, 0.0, np.pi / 2, 0.05])


@pytest.mark.parametrize("value", [np.zeros(19), np.full(20, np.nan), np.full(20, np.inf)])
def test_eef_action_adapter_rejects_invalid_actions(value: np.ndarray) -> None:
    with pytest.raises(ValueError):
        EefActionAdapter().validate(value)


def test_eef_action_adapter_rejects_degenerate_rot6d() -> None:
    action = np.zeros(20, dtype=np.float32)
    with pytest.raises(ValueError, match="degenerate"):
        EefActionAdapter().to_bridge_request(action)


def test_eef_observation_adapter_rejects_missing_eef() -> None:
    message = _message()
    message.pop("eef")
    with pytest.raises(ValueError, match="eef"):
        EefObservationAdapter().adapt(message)
