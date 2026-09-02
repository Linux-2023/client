"""Focused tests for bounded timestamp synchronization of bridge sensor events."""

from pathlib import Path
import math
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import frame_synchronizer as frame_synchronizer_module
from frame_synchronizer import FrameSynchronizer


IMAGE_SENSORS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
JOINT_SENSORS = ("puppet_left", "puppet_right", "master_left", "master_right")
ALL_SENSORS = IMAGE_SENSORS + JOINT_SENSORS
EEF_SENSOR_NAMES = ("eef_puppet_left", "eef_puppet_right")
ALL_SENSORS_WITH_EEF = ALL_SENSORS + EEF_SENSOR_NAMES


def joint_values(start: float) -> np.ndarray:
    return np.arange(start, start + 7, dtype=np.float32)

def eef_values(start: float) -> np.ndarray:
    return np.arange(start, start + 6, dtype=np.float32)


def push_complete_frame(
    sync: FrameSynchronizer,
    timestamp: float,
    *,
    offsets: dict[str, float] | None = None,
    include_eef: bool = False,
) -> None:
    offsets = offsets or {}
    sync.push("cam_high", timestamp + offsets.get("cam_high", 0.0), b"high")
    sync.push("cam_left_wrist", timestamp + offsets.get("cam_left_wrist", 0.0), b"left")
    sync.push("cam_right_wrist", timestamp + offsets.get("cam_right_wrist", 0.0), b"right")
    sync.push("puppet_left", timestamp + offsets.get("puppet_left", 0.0), joint_values(0))
    sync.push("puppet_right", timestamp + offsets.get("puppet_right", 0.0), joint_values(10))
    sync.push("master_left", timestamp + offsets.get("master_left", 0.0), joint_values(20))
    sync.push("master_right", timestamp + offsets.get("master_right", 0.0), joint_values(30))
    if include_eef:
        sync.push("eef_puppet_left", timestamp + offsets.get("eef_puppet_left", 0.0), eef_values(40))
        sync.push("eef_puppet_right", timestamp + offsets.get("eef_puppet_right", 0.0), eef_values(50))


def test_nearest_neighbor_matching_within_thirty_milliseconds():
    sync = FrameSynchronizer(max_error=0.03)
    sync.push("cam_high", 10.0, b"reference")
    sync.push("cam_left_wrist", 9.960, b"too-early")
    sync.push("cam_left_wrist", 10.025, b"nearest-left")
    sync.push("cam_right_wrist", 10.029, b"nearest-right")
    sync.push("puppet_left", 9.974, joint_values(0))
    sync.push("puppet_right", 10.001, joint_values(10))
    sync.push("master_left", 10.018, joint_values(20))
    sync.push("master_right", 9.999, joint_values(30))

    frame = sync.try_sync()

    assert frame is not None
    assert frame.timestamp == pytest.approx(10.0)
    assert frame.images == {
        "cam_high": b"reference",
        "cam_left_wrist": b"nearest-left",
        "cam_right_wrist": b"nearest-right",
    }
    assert frame.sensor_timestamps == {
        "cam_high": pytest.approx(10.0),
        "cam_left_wrist": pytest.approx(10.025),
        "cam_right_wrist": pytest.approx(10.029),
        "puppet_left": pytest.approx(9.974),
        "puppet_right": pytest.approx(10.001),
        "master_left": pytest.approx(10.018),
        "master_right": pytest.approx(9.999),
    }
    assert frame.sync_error == {
        "cam_high": pytest.approx(0.0),
        "cam_left_wrist": pytest.approx(0.025),
        "cam_right_wrist": pytest.approx(0.029),
        "puppet_left": pytest.approx(-0.026),
        "puppet_right": pytest.approx(0.001),
        "master_left": pytest.approx(0.018),
        "master_right": pytest.approx(-0.001),
    }
    assert sync.accepted_frames == 1



def test_reference_remains_pending_until_late_in_window_sample_arrives():
    sync = FrameSynchronizer(max_error=0.03)
    sync.push("cam_high", 10.0, b"reference")
    sync.push("cam_left_wrist", 9.950, b"older-left")
    sync.push("cam_right_wrist", 10.0, b"right")
    sync.push("puppet_left", 10.0, joint_values(0))
    sync.push("puppet_right", 10.0, joint_values(10))
    sync.push("master_left", 10.0, joint_values(20))
    sync.push("master_right", 10.0, joint_values(30))

    assert sync.try_sync() is None
    assert sync.rejected_stale == 0

    sync.push("cam_left_wrist", 10.020, b"late-valid-left")
    frame = sync.try_sync()

    assert frame is not None
    assert frame.timestamp == pytest.approx(10.0)
    assert frame.images["cam_left_wrist"] == b"late-valid-left"
    assert frame.sync_error["cam_left_wrist"] == pytest.approx(0.020)
    assert sync.accepted_frames == 1

def test_rejects_reference_when_any_required_stream_is_missing_or_outside_window():
    missing = FrameSynchronizer(max_error=0.03)
    for sensor in ALL_SENSORS:
        if sensor != "cam_right_wrist":
            missing.push(sensor, 1.0, b"jpeg" if sensor.startswith("cam_") else joint_values(0))

    assert missing.try_sync() is None
    assert missing.rejected_missing == 1

    stale = FrameSynchronizer(max_error=0.03)
    push_complete_frame(stale, 2.0, offsets={"master_right": 0.031})

    assert stale.try_sync() is None
    assert stale.rejected_stale == 1
    assert stale.accepted_frames == 0


def test_does_not_emit_partial_frame_while_a_sensor_is_late():
    sync = FrameSynchronizer(max_error=0.03)
    for sensor in ALL_SENSORS:
        if sensor != "master_right":
            sync.push(sensor, 3.0, b"jpeg" if sensor.startswith("cam_") else joint_values(0))

    assert sync.try_sync() is None

    sync.push("master_right", 3.025, joint_values(30))
    frame = sync.try_sync()

    assert frame is not None
    assert frame.sync_error["master_right"] == pytest.approx(0.025)


def test_monotonic_output_and_no_duplicate_reference_emission():
    sync = FrameSynchronizer(max_error=0.03)
    push_complete_frame(sync, 5.0)
    first = sync.try_sync()

    assert first is not None
    assert sync.try_sync() is None

    push_complete_frame(sync, 4.99)
    assert sync.try_sync() is None
    assert sync.rejected_late == 1

    push_complete_frame(sync, 5.1)
    second = sync.try_sync()

    assert second is not None
    assert second.timestamp == pytest.approx(5.1)


def test_two_second_buffer_eviction_removes_old_samples():
    sync = FrameSynchronizer(max_error=0.03, max_buffer_seconds=2.0)
    sync.push("cam_high", 1.0, b"old-reference")
    sync.push("cam_left_wrist", 1.0, b"old-left")
    sync.push("cam_high", 3.1, b"new-reference")

    assert sync.buffered_counts["cam_high"] == 1
    assert sync.buffered_counts["cam_left_wrist"] == 0
    assert sync.try_sync() is None



def test_eviction_uses_latest_observed_timestamp_for_out_of_order_samples():
    sync = FrameSynchronizer(max_error=0.03, max_buffer_seconds=2.0)
    sync.push("cam_high", 10.0, b"new-reference")
    sync.push("cam_left_wrist", 7.9, b"out-of-order-old-left")

    assert sync.buffered_counts["cam_high"] == 1
    assert sync.buffered_counts["cam_left_wrist"] == 0

def test_clear_removes_samples_and_resets_episode_boundary_state():
    sync = FrameSynchronizer(max_error=0.03)
    push_complete_frame(sync, 10.0)
    assert sync.try_sync() is not None

    push_complete_frame(sync, 11.0)
    sync.clear()
    assert sync.try_sync() is None
    assert all(count == 0 for count in sync.buffered_counts.values())

    push_complete_frame(sync, 9.0)
    frame = sync.try_sync()

    assert frame is not None
    assert frame.timestamp == pytest.approx(9.0)


def test_state_action_concatenation_order_float32_shapes_and_finite_values():
    sync = FrameSynchronizer(max_error=0.03)
    push_complete_frame(sync, 7.0)

    frame = sync.try_sync()

    assert frame is not None
    assert frame.state.shape == (14,)
    assert frame.action.shape == (14,)
    assert frame.state.dtype == np.float32
    assert frame.action.dtype == np.float32
    np.testing.assert_array_equal(frame.state, np.concatenate([joint_values(0), joint_values(10)]).astype(np.float32))
    np.testing.assert_array_equal(frame.action, np.concatenate([joint_values(20), joint_values(30)]).astype(np.float32))
    assert np.isfinite(frame.state).all()
    assert np.isfinite(frame.action).all()


@pytest.mark.parametrize(
    ("sensor", "timestamp", "value", "match"),
    [
        ("unknown", 1.0, b"jpeg", "unknown sensor"),
        ("cam_high", math.nan, b"jpeg", "timestamp"),
        ("cam_high", math.inf, b"jpeg", "timestamp"),
        ("cam_high", 1.0, "not-bytes", "image"),
        ("puppet_left", 1.0, np.zeros(6, dtype=np.float32), "seven"),
        ("puppet_left", 1.0, np.array([0.0] * 6 + [math.nan], dtype=np.float32), "finite"),
        ("puppet_left", 1.0, [0.0] * 7, "joint sensor values must be numpy arrays"),
    ],
)
def test_unknown_sensors_and_invalid_timestamps_or_values_are_rejected(sensor, timestamp, value, match):
    sync = FrameSynchronizer(max_error=0.03)

    with pytest.raises(ValueError, match=match):
        sync.push(sensor, timestamp, value)

    assert sync.rejected_invalid == 1


def test_default_mode_preserves_required_sensors_and_rejects_eef_streams():
    assert frame_synchronizer_module.EEF_SENSORS == EEF_SENSOR_NAMES
    assert frame_synchronizer_module.EEF_DIMENSION == 6

    sync = FrameSynchronizer(max_error=0.03)

    assert sync.required_sensors == ALL_SENSORS
    assert set(sync.buffered_counts) == set(ALL_SENSORS)
    with pytest.raises(ValueError, match="unknown sensor"):
        sync.push("eef_puppet_left", 1.0, eef_values(0))
    assert sync.rejected_invalid == 1

    push_complete_frame(sync, 1.0)
    frame = sync.try_sync()

    assert frame is not None
    assert frame.eef is None
    assert frame.state.shape == (14,)
    assert frame.action.shape == (14,)


def test_eef_mode_aligns_pose_streams_and_emits_synchronized_metadata():
    sync = FrameSynchronizer(max_error=0.03, include_eef=True)

    assert sync.required_sensors == ALL_SENSORS_WITH_EEF
    push_complete_frame(sync, 10.0, offsets={"master_right": -0.005})
    sync.push("eef_puppet_left", 9.980, eef_values(40))
    sync.push("eef_puppet_left", 10.012, eef_values(60))
    sync.push("eef_puppet_right", 9.985, eef_values(80))

    frame = sync.try_sync()

    assert frame is not None
    assert frame.eef is not None
    assert set(frame.sensor_timestamps) == set(ALL_SENSORS_WITH_EEF)
    assert set(frame.sync_error) == set(ALL_SENSORS_WITH_EEF)
    np.testing.assert_array_equal(frame.eef["puppet_left"], eef_values(60))
    np.testing.assert_array_equal(frame.eef["puppet_right"], eef_values(80))
    assert frame.eef["puppet_left"].shape == (6,)
    assert frame.eef["puppet_right"].shape == (6,)
    assert np.isfinite(frame.eef["puppet_left"]).all()
    assert np.isfinite(frame.eef["puppet_right"]).all()
    assert frame.sensor_timestamps["eef_puppet_left"] == pytest.approx(10.012)
    assert frame.sensor_timestamps["eef_puppet_right"] == pytest.approx(9.985)
    assert frame.sync_error["eef_puppet_left"] == pytest.approx(0.012)
    assert frame.sync_error["eef_puppet_right"] == pytest.approx(-0.015)
    assert frame.state.shape == (14,)
    assert frame.action.shape == (14,)
    assert sync.accepted_frames == 1


def test_eef_mode_accepts_pose_samples_on_exact_error_boundary():
    sync = FrameSynchronizer(max_error=0.03, include_eef=True)
    push_complete_frame(
        sync,
        11.0,
        include_eef=True,
        offsets={"eef_puppet_left": -0.03, "eef_puppet_right": 0.03},
    )

    frame = sync.try_sync()

    assert frame is not None
    assert frame.eef is not None
    assert frame.sync_error["eef_puppet_left"] == pytest.approx(-0.03)
    assert frame.sync_error["eef_puppet_right"] == pytest.approx(0.03)


def test_eef_mode_waits_for_missing_then_emits_late_stream_and_rejects_stale_reference():
    late = FrameSynchronizer(max_error=0.03, include_eef=True)
    push_complete_frame(late, 12.0)
    late.push("eef_puppet_left", 12.0, eef_values(40))

    assert late.try_sync() is None
    assert late.rejected_missing == 1

    late.push("eef_puppet_right", 12.025, eef_values(50))
    frame = late.try_sync()

    assert frame is not None
    assert frame.eef is not None
    assert frame.sync_error["eef_puppet_right"] == pytest.approx(0.025)

    stale = FrameSynchronizer(max_error=0.03, include_eef=True)
    push_complete_frame(stale, 13.0)
    stale.push("eef_puppet_left", 13.0, eef_values(40))
    stale.push("eef_puppet_right", 13.031, eef_values(50))

    assert stale.try_sync() is None
    assert stale.rejected_stale == 1
    assert stale.accepted_frames == 0


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ([0.0] * 6, "numpy"),
        (np.zeros((1, 6), dtype=np.float32), "one-dimensional"),
        (np.zeros(5, dtype=np.float32), "six"),
        (np.array(["x"] * 6), "numeric"),
        (np.array([0.0] * 5 + [math.inf], dtype=np.float32), "finite"),
    ],
)
def test_invalid_eef_values_are_rejected(value, match):
    sync = FrameSynchronizer(max_error=0.03, include_eef=True)

    with pytest.raises(ValueError, match=match):
        sync.push("eef_puppet_left", 1.0, value)

    assert sync.rejected_invalid == 1
