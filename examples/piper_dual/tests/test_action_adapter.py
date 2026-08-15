"""Focused tests for the shared Piper dual action contract."""

from pathlib import Path
import math
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from action_adapter import ActionAdapter


def test_validate_returns_float32_vector_and_rejects_non_finite_or_wrong_length():
    adapter = ActionAdapter()
    action = np.arange(14, dtype=np.float64)

    validated = adapter.validate(action)

    assert validated.shape == (14,)
    assert validated.dtype == np.float32
    np.testing.assert_array_equal(validated, np.arange(14, dtype=np.float32))


@pytest.mark.parametrize(
    "action",
    [
        [0.0] * 13,
        [0.0] * 15,
        [0.0] * 13 + [math.nan],
        [0.0] * 13 + [math.inf],
        [0.0] * 13 + [-math.inf],
    ],
)
def test_validate_rejects_non_finite_or_wrong_length(action):
    adapter = ActionAdapter()

    with pytest.raises(ValueError):
        adapter.validate(action)


def test_split_returns_left_and_right_seven_element_vectors_in_order():
    adapter = ActionAdapter()
    action = np.arange(14, dtype=np.float32)

    left, right = adapter.split(action)

    assert left.shape == (7,)
    assert right.shape == (7,)
    assert left.dtype == np.float32
    assert right.dtype == np.float32
    np.testing.assert_array_equal(left, np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(right, np.arange(7, 14, dtype=np.float32))


def test_to_bridge_request_emits_task1_publish_action_shape():
    adapter = ActionAdapter()
    action = np.arange(14, dtype=np.float32)

    request = adapter.to_bridge_request(action)

    assert request == {
        "type": "publish_action",
        "left": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "right": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
    }
