"""Bounded timestamp synchronization for Piper dual ROS 2 bridge sensor events."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

IMAGE_SENSORS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
JOINT_SENSORS = ("puppet_left", "puppet_right", "master_left", "master_right")
EEF_SENSORS = ("eef_puppet_left", "eef_puppet_right")
REQUIRED_SENSORS = IMAGE_SENSORS + JOINT_SENSORS
EEF_DIMENSION = 6
JOINT_DIMENSION = 7
FRAME_DIMENSION = 14


@dataclass(frozen=True, slots=True)
class SynchronizedFrame:
    timestamp: float
    images: dict[str, bytes]
    state: np.ndarray
    action: np.ndarray
    sensor_timestamps: dict[str, float]
    sync_error: dict[str, float]
    eef: dict[str, np.ndarray] | None = None

@dataclass(frozen=True, slots=True)
class _Sample:
    timestamp: float
    value: object


class FrameSynchronizer:
    """Synchronize individual bridge sensor samples into complete observation frames."""

    def __init__(self, max_error: float, max_buffer_seconds: float = 2.0, *, include_eef: bool = False) -> None:
        self.max_error = self._validate_positive_float(max_error, "max_error")
        self.max_buffer_seconds = self._validate_positive_float(max_buffer_seconds, "max_buffer_seconds")
        self.include_eef = bool(include_eef)
        self.required_sensors = REQUIRED_SENSORS + EEF_SENSORS if self.include_eef else REQUIRED_SENSORS
        self._buffers: dict[str, deque[_Sample]] = {sensor: deque() for sensor in self.required_sensors}
        self.accepted_frames = 0
        self.rejected_missing = 0
        self.rejected_stale = 0
        self.rejected_late = 0
        self.rejected_invalid = 0
        self._last_emitted_reference: float | None = None
        self._latest_observed_timestamp: float | None = None

    @property
    def buffered_counts(self) -> dict[str, int]:
        return {sensor: len(samples) for sensor, samples in self._buffers.items()}

    def clear(self) -> None:
        for samples in self._buffers.values():
            samples.clear()
        self._last_emitted_reference = None
        self._latest_observed_timestamp = None

    def push(self, sensor: str, timestamp: float, value: object) -> None:
        try:
            self._validate_sensor(sensor)
            timestamp = self._validate_timestamp(timestamp)
            value = self._validate_value(sensor, value)
        except ValueError:
            self.rejected_invalid += 1
            raise

        samples = self._buffers[sensor]
        if samples and timestamp < samples[-1].timestamp:
            index = len(samples)
            for index, sample in enumerate(samples):
                if timestamp < sample.timestamp:
                    break
            samples.insert(index, _Sample(timestamp, value))
        else:
            samples.append(_Sample(timestamp, value))
        if self._latest_observed_timestamp is None or timestamp > self._latest_observed_timestamp:
            self._latest_observed_timestamp = timestamp
        self._evict_old_samples()

    def try_sync(self, reference_sensor: str = "cam_high") -> SynchronizedFrame | None:
        self._validate_sensor(reference_sensor)
        if reference_sensor not in IMAGE_SENSORS:
            raise ValueError("reference_sensor must be an image sensor")

        reference_buffer = self._buffers[reference_sensor]
        while reference_buffer and self._last_emitted_reference is not None and reference_buffer[0].timestamp <= self._last_emitted_reference:
            reference_buffer.popleft()
            self.rejected_late += 1

        for reference in tuple(reference_buffer):
            if self._last_emitted_reference is not None and reference.timestamp <= self._last_emitted_reference:
                continue

            selected: dict[str, _Sample] = {reference_sensor: reference}
            pending = False
            stale = False
            for sensor in self.required_sensors:
                if sensor == reference_sensor:
                    continue
                sample = self._nearest_sample(sensor, reference.timestamp)
                if sample is None:
                    pending = True
                    self.rejected_missing += 1
                    break
                if abs(sample.timestamp - reference.timestamp) > self.max_error:
                    if self._stream_advanced_beyond(sensor, reference.timestamp + self.max_error):
                        stale = True
                    else:
                        pending = True
                    break
                selected[sensor] = sample

            if stale:
                try:
                    reference_buffer.remove(reference)
                except ValueError:
                    pass
                self.rejected_stale += 1
                continue

            if pending:
                return None

            frame = self._build_frame(reference_sensor, reference.timestamp, selected)
            self._last_emitted_reference = reference.timestamp
            self.accepted_frames += 1
            self._discard_consumed_samples(selected)
            return frame

        return None

    def _build_frame(self, reference_sensor: str, reference_timestamp: float, selected: dict[str, _Sample]) -> SynchronizedFrame:
        images = {sensor: selected[sensor].value for sensor in IMAGE_SENSORS}
        puppet_left = selected["puppet_left"].value
        puppet_right = selected["puppet_right"].value
        master_left = selected["master_left"].value
        master_right = selected["master_right"].value
        state = np.concatenate([puppet_left, puppet_right]).astype(np.float32, copy=False)
        action = np.concatenate([master_left, master_right]).astype(np.float32, copy=False)
        if state.shape != (FRAME_DIMENSION,) or action.shape != (FRAME_DIMENSION,):
            self.rejected_invalid += 1
            raise ValueError("Synchronized state and action must contain exactly fourteen values")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            self.rejected_invalid += 1
            raise ValueError("Synchronized state and action values must be finite")

        sensor_timestamps = {sensor: selected[sensor].timestamp for sensor in self.required_sensors}
        sync_error = {sensor: selected[sensor].timestamp - reference_timestamp for sensor in self.required_sensors}
        sync_error[reference_sensor] = 0.0
        eef = None
        if self.include_eef:
            eef = {
                "puppet_left": np.asarray(selected["eef_puppet_left"].value, dtype=np.float32).copy(),
                "puppet_right": np.asarray(selected["eef_puppet_right"].value, dtype=np.float32).copy(),
            }
        return SynchronizedFrame(
            timestamp=reference_timestamp,
            images=images,
            state=state,
            action=action,
            sensor_timestamps=sensor_timestamps,
            sync_error=sync_error,
            eef=eef,
        )

    def _nearest_sample(self, sensor: str, timestamp: float) -> _Sample | None:
        samples = self._buffers[sensor]
        if not samples:
            return None
        return min(samples, key=lambda sample: (abs(sample.timestamp - timestamp), sample.timestamp))

    def _stream_advanced_beyond(self, sensor: str, timestamp: float) -> bool:
        samples = self._buffers[sensor]
        return bool(samples and samples[-1].timestamp > timestamp)

    def _discard_consumed_samples(self, selected: dict[str, _Sample]) -> None:
        for sensor, used in selected.items():
            samples = self._buffers[sensor]
            while samples and samples[0].timestamp <= used.timestamp:
                samples.popleft()

    def _evict_old_samples(self) -> None:
        if self._latest_observed_timestamp is None:
            return
        cutoff = self._latest_observed_timestamp - self.max_buffer_seconds
        for samples in self._buffers.values():
            while samples and samples[0].timestamp < cutoff:
                samples.popleft()

    def _validate_sensor(self, sensor: str) -> None:
        if sensor not in self._buffers:
            raise ValueError(f"unknown sensor: {sensor!r}")

    def _validate_timestamp(self, timestamp: float) -> float:
        if isinstance(timestamp, bool):
            raise ValueError("timestamp must be a finite number")
        try:
            value = float(timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError("timestamp must be a finite number") from exc
        if not math.isfinite(value):
            raise ValueError("timestamp must be finite")
        return value

    def _validate_value(self, sensor: str, value: object) -> object:
        if sensor in IMAGE_SENSORS:
            if not isinstance(value, bytes):
                raise ValueError("image sensor values must be JPEG bytes")
            return value

        if not isinstance(value, np.ndarray):
            if sensor in EEF_SENSORS:
                raise ValueError("EEF sensor values must be numpy arrays")
            raise ValueError("joint sensor values must be numpy arrays")
        if sensor in EEF_SENSORS:
            if value.ndim != 1:
                raise ValueError("EEF sensor values must be one-dimensional")
            if value.shape != (EEF_DIMENSION,):
                raise ValueError("EEF sensor values must contain exactly six values")
            if value.dtype.kind not in "fiu":
                raise ValueError("EEF sensor values must be numeric")
            values = value.astype(np.float32, copy=False)
            if not np.isfinite(values).all():
                raise ValueError("EEF sensor values must be finite")
            return values

        if value.ndim != 1 or value.shape[0] != JOINT_DIMENSION:
            raise ValueError("joint sensor values must contain exactly seven values")
        if value.dtype.kind not in "fiu":
            raise ValueError("joint sensor values must be numeric")
        values = value.astype(np.float32, copy=False)
        if not np.isfinite(values).all():
            raise ValueError("joint sensor values must be finite")
        return values

    def _validate_positive_float(self, value: float, field: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{field} must be a positive finite number")
        try:
            coerced = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a positive finite number") from exc
        if not math.isfinite(coerced) or coerced <= 0.0:
            raise ValueError(f"{field} must be a positive finite number")
        return coerced
