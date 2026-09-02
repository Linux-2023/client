"""Crash-safe streaming HDF5 writer for Piper dual synchronized episodes."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

from frame_synchronizer import EEF_DIMENSION
from frame_synchronizer import EEF_SENSORS
from frame_synchronizer import FRAME_DIMENSION
from frame_synchronizer import IMAGE_SENSORS
from frame_synchronizer import REQUIRED_SENSORS
from frame_synchronizer import SynchronizedFrame

SCHEMA_VERSION = "piper_dual_ros2_v1"
STATE_PATH = "/observations/state"
ACTION_PATH = "/action"
TIMESTAMP_PATH = "/observations/timestamp"
IMAGE_GROUP_PATH = "/observations/images"
EEF_GROUP_PATH = "/observations/eef"
SENSOR_TIMESTAMPS_GROUP_PATH = "/observations/sensor_timestamps"
SYNC_ERROR_GROUP_PATH = "/observations/sync_error"


@dataclass(slots=True)
class StreamingEpisodeWriter:
    """Append synchronized frames to a crash-safe self-contained HDF5 episode."""

    final_path: Path
    partial_path: Path
    _file: h5py.File
    _flush_every: int
    _include_eef: bool = False
    _required_sensors: tuple[str, ...] = REQUIRED_SENSORS
    _frame_count: int = 0
    _closed: bool = False

    @classmethod
    def open(
        cls,
        path: Path,
        metadata: dict,
        jpeg_quality: int = 90,
        flush_every: int = 30,
        *,
        include_eef: bool = False,
    ) -> "StreamingEpisodeWriter":
        """Create a writer that streams into ``<episode>.hdf5.partial`` until finalized."""
        final_path = Path(path)
        if final_path.suffix != ".hdf5":
            final_path = final_path.with_suffix(".hdf5")
        partial_path = final_path.with_suffix(f"{final_path.suffix}.partial")
        if isinstance(flush_every, bool) or int(flush_every) <= 0:
            raise ValueError("flush_every must be positive")
        if isinstance(jpeg_quality, bool) or not 1 <= int(jpeg_quality) <= 100:
            raise ValueError("jpeg_quality must be an integer from 1 through 100")
        _ensure_json_serializable(metadata, "metadata")

        eef_metadata = metadata.get("eef") if isinstance(metadata, dict) else None
        metadata_eef_enabled = isinstance(eef_metadata, dict) and eef_metadata.get("enabled") is True
        if bool(include_eef) != metadata_eef_enabled:
            raise ValueError("metadata eef.enabled must match include_eef")

        required_sensors = REQUIRED_SENSORS + EEF_SENSORS if include_eef else REQUIRED_SENSORS
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        h5_file = h5py.File(partial_path, "w")
        try:
            _initialize_file(
                h5_file,
                metadata,
                int(jpeg_quality),
                int(flush_every),
                include_eef=bool(include_eef),
                required_sensors=required_sensors,
            )
        except Exception:
            h5_file.close()
            partial_path.unlink(missing_ok=True)
            raise
        return cls(
            final_path=final_path,
            partial_path=partial_path,
            _file=h5_file,
            _flush_every=int(flush_every),
            _include_eef=bool(include_eef),
            _required_sensors=required_sensors,
        )

    def append(self, frame: SynchronizedFrame) -> None:
        """Append one synchronized frame without encoding images or retaining frame references."""
        self._ensure_open()
        timestamp = _coerce_finite_float(frame.timestamp, "timestamp")
        state = _coerce_vector(frame.state, "state")
        action = _coerce_vector(frame.action, "action")
        images = _coerce_images(frame.images)
        sensor_timestamps = _coerce_sensor_mapping(frame.sensor_timestamps, "sensor_timestamps", self._required_sensors)
        sync_error = _coerce_sensor_mapping(frame.sync_error, "sync_error", self._required_sensors)
        eef = _coerce_eef(frame.eef) if self._include_eef else None

        index = self._frame_count
        self._resize_for_next_frame(index + 1)
        self._file[STATE_PATH][index] = state
        self._file[ACTION_PATH][index] = action
        self._file[TIMESTAMP_PATH][index] = np.float32(timestamp)
        for camera, image_bytes in images.items():
            self._file[f"{IMAGE_GROUP_PATH}/{camera}"][index] = np.frombuffer(image_bytes, dtype=np.uint8).copy()
        if eef is not None:
            self._file[f"{EEF_GROUP_PATH}/puppet_left"][index] = eef["puppet_left"]
            self._file[f"{EEF_GROUP_PATH}/puppet_right"][index] = eef["puppet_right"]
        for sensor in self._required_sensors:
            self._file[f"{SENSOR_TIMESTAMPS_GROUP_PATH}/{sensor}"][index] = np.float32(sensor_timestamps[sensor])
            self._file[f"{SYNC_ERROR_GROUP_PATH}/{sensor}"][index] = np.float32(sync_error[sensor])

        self._frame_count = index + 1
        self._file.attrs["frame_count"] = self._frame_count
        self._write_timing_counters(finalized=False)
        if self._frame_count % self._flush_every == 0:
            self.flush()

    def flush(self) -> None:
        """Flush buffered HDF5 data to the partial file."""
        self._ensure_open()
        self._file.flush()

    def finalize(self) -> Path:
        """Close, validate, and atomically publish the partial HDF5 file."""
        self._ensure_open()
        self._file.attrs["frame_count"] = self._frame_count
        self._write_timing_counters(finalized=True)
        self._file.flush()
        self._file.close()
        self._closed = True

        report = validate_episode(self.partial_path)
        if report["errors"]:
            raise ValueError(f"episode validation failed: {report['errors']}")
        self.partial_path.replace(self.final_path)
        return self.final_path

    def abort(self, remove_partial: bool = True) -> None:
        """Close the partial file and optionally remove it without publishing a final file."""
        if not self._closed:
            self._file.close()
            self._closed = True
        if remove_partial:
            self.partial_path.unlink(missing_ok=True)

    def _resize_for_next_frame(self, size: int) -> None:
        self._file[STATE_PATH].resize((size, FRAME_DIMENSION))
        self._file[ACTION_PATH].resize((size, FRAME_DIMENSION))
        self._file[TIMESTAMP_PATH].resize((size,))
        for camera in IMAGE_SENSORS:
            self._file[f"{IMAGE_GROUP_PATH}/{camera}"].resize((size,))
        if self._include_eef:
            self._file[f"{EEF_GROUP_PATH}/puppet_left"].resize((size, EEF_DIMENSION))
            self._file[f"{EEF_GROUP_PATH}/puppet_right"].resize((size, EEF_DIMENSION))
        for sensor in self._required_sensors:
            self._file[f"{SENSOR_TIMESTAMPS_GROUP_PATH}/{sensor}"].resize((size,))
            self._file[f"{SYNC_ERROR_GROUP_PATH}/{sensor}"].resize((size,))

    def _write_timing_counters(self, *, finalized: bool) -> None:
        counters = {
            "frames_written": self._frame_count,
            "flush_every": self._flush_every,
            "finalized": finalized,
        }
        self._file.attrs["timing_counters_json"] = json.dumps(counters, sort_keys=True, separators=(",", ":"))

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("StreamingEpisodeWriter is closed")


def validate_episode(path: Path) -> dict:
    """Return frame counts, schema info, decode counts, monotonicity, and validation errors."""
    report = {
        "frame_count": 0,
        "schema_version": None,
        "timestamp_monotonic": False,
        "image_decode_counts": {camera: 0 for camera in IMAGE_SENSORS},
        "errors": [],
    }
    try:
        with h5py.File(path, "r") as episode:
            _validate_open_episode(episode, report)
    except OSError as exc:
        report["errors"].append(f"could not open HDF5 file: {exc}")
    return report


def _initialize_file(
    h5_file: h5py.File,
    metadata: dict,
    jpeg_quality: int,
    flush_every: int,
    *,
    include_eef: bool = False,
    required_sensors: tuple[str, ...] = REQUIRED_SENSORS,
) -> None:
    h5_file.attrs["schema_version"] = SCHEMA_VERSION
    h5_file.attrs["frame_count"] = 0
    h5_file.attrs["jpeg_quality"] = jpeg_quality
    h5_file.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    h5_file.attrs["camera_mapping_json"] = json.dumps(
        {camera: f"{IMAGE_GROUP_PATH}/{camera}" for camera in IMAGE_SENSORS}, sort_keys=True, separators=(",", ":")
    )
    units = {"state": "mixed_rad_m", "action": "mixed_rad_m", "timestamp": "seconds", "sync_error": "seconds"}
    if include_eef:
        units["eef"] = "xyz_m_rpy_rad"
    h5_file.attrs["units_json"] = json.dumps(units, sort_keys=True, separators=(",", ":"))
    h5_file.attrs["image_preprocessing_json"] = json.dumps(
        {"encoding": "jpeg_bytes_passthrough", "color_space": "source", "reencoded_by_writer": False},
        sort_keys=True,
        separators=(",", ":"),
    )
    h5_file.attrs["timing_counters_json"] = json.dumps(
        {"frames_written": 0, "flush_every": flush_every, "finalized": False},
        sort_keys=True,
        separators=(",", ":"),
    )

    observations = h5_file.create_group("observations")
    observations.create_dataset(
        "state",
        shape=(0, FRAME_DIMENSION),
        maxshape=(None, FRAME_DIMENSION),
        chunks=(1, FRAME_DIMENSION),
        dtype=np.float32,
    )
    observations.create_dataset("timestamp", shape=(0,), maxshape=(None,), chunks=(1,), dtype=np.float32)
    h5_file.create_dataset(
        "action",
        shape=(0, FRAME_DIMENSION),
        maxshape=(None, FRAME_DIMENSION),
        chunks=(1, FRAME_DIMENSION),
        dtype=np.float32,
    )

    vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
    image_group = observations.create_group("images")
    for camera in IMAGE_SENSORS:
        image_group.create_dataset(camera, shape=(0,), maxshape=(None,), chunks=(1,), dtype=vlen_uint8)
    if include_eef:
        eef_group = observations.create_group("eef")
        for arm in ("puppet_left", "puppet_right"):
            eef_group.create_dataset(
                arm,
                shape=(0, EEF_DIMENSION),
                maxshape=(None, EEF_DIMENSION),
                chunks=(1, EEF_DIMENSION),
                dtype=np.float32,
            )

    sensor_timestamps = observations.create_group("sensor_timestamps")
    sync_error = observations.create_group("sync_error")
    for sensor in required_sensors:
        sensor_timestamps.create_dataset(sensor, shape=(0,), maxshape=(None,), chunks=(1,), dtype=np.float32)
        sync_error.create_dataset(sensor, shape=(0,), maxshape=(None,), chunks=(1,), dtype=np.float32)


def _validate_open_episode(episode: h5py.File, report: dict) -> None:
    schema_version = episode.attrs.get("schema_version")
    report["schema_version"] = _decode_attr(schema_version)
    if report["schema_version"] != SCHEMA_VERSION:
        report["errors"].append(f"schema version mismatch: expected {SCHEMA_VERSION!r}, found {report['schema_version']!r}")

    frame_count = _dataset_length(episode, STATE_PATH, report)
    report["frame_count"] = max(frame_count, 0)
    if frame_count == 0:
        report["errors"].append("episode contains zero frames")

    expected_shapes = {
        STATE_PATH: (frame_count, FRAME_DIMENSION),
        ACTION_PATH: (frame_count, FRAME_DIMENSION),
        TIMESTAMP_PATH: (frame_count,),
    }
    for dataset_path, expected_shape in expected_shapes.items():
        dataset = _get_dataset(episode, dataset_path, report)
        if dataset is None:
            continue
        if dataset.shape != expected_shape:
            report["errors"].append(f"{dataset_path} shape {dataset.shape} does not match expected {expected_shape}")
        if dataset.dtype != np.dtype("float32"):
            report["errors"].append(f"{dataset_path} dtype {dataset.dtype} is not float32")
        if dataset.maxshape[0] is not None:
            report["errors"].append(f"{dataset_path} is not extendible")

    timestamps = _get_dataset(episode, TIMESTAMP_PATH, report)
    if timestamps is not None and timestamps.shape == (frame_count,):
        values = timestamps[:]
        monotonic = bool(frame_count > 0 and np.all(np.diff(values) >= 0.0))
        report["timestamp_monotonic"] = monotonic
        if not monotonic:
            report["errors"].append("timestamps must be strictly nondecreasing")

    for camera in IMAGE_SENSORS:
        dataset_path = f"{IMAGE_GROUP_PATH}/{camera}"
        dataset = _get_dataset(episode, dataset_path, report)
        if dataset is None:
            continue
        if dataset.shape != (frame_count,):
            report["errors"].append(f"{dataset_path} length {dataset.shape[0]} does not match frame count {frame_count}")
        if dataset.maxshape != (None,):
            report["errors"].append(f"{dataset_path} is not one-dimensional and extendible")
        if h5py.check_dtype(vlen=dataset.dtype) != np.dtype("uint8"):
            report["errors"].append(f"{dataset_path} must use vlen uint8 JPEG storage")
        decode_count = 0
        for index, encoded in enumerate(dataset):
            if _decode_jpeg_array(encoded):
                decode_count += 1
            else:
                report["errors"].append(f"{dataset_path}[{index}] is not decodable JPEG bytes")
        report["image_decode_counts"][camera] = decode_count

    include_eef = _metadata_eef_enabled(episode)
    required_sensors = REQUIRED_SENSORS + EEF_SENSORS if include_eef else REQUIRED_SENSORS
    for group_path in (SENSOR_TIMESTAMPS_GROUP_PATH, SYNC_ERROR_GROUP_PATH):
        for sensor in required_sensors:
            dataset_path = f"{group_path}/{sensor}"
            dataset = _get_dataset(episode, dataset_path, report)
            if dataset is None:
                continue
            if dataset.shape != (frame_count,):
                report["errors"].append(f"{dataset_path} length {dataset.shape[0]} does not match frame count {frame_count}")
            if dataset.dtype != np.dtype("float32"):
                report["errors"].append(f"{dataset_path} dtype {dataset.dtype} is not float32")
            if dataset.maxshape != (None,):
                report["errors"].append(f"{dataset_path} is not one-dimensional and extendible")
    if include_eef:
        _validate_eef_datasets(episode, frame_count, report)


def _metadata_eef_enabled(episode: h5py.File) -> bool:
    raw_metadata = _decode_attr(episode.attrs.get("metadata_json"))
    if not isinstance(raw_metadata, str):
        return False
    try:
        metadata = json.loads(raw_metadata)
    except json.JSONDecodeError:
        return False
    eef = metadata.get("eef") if isinstance(metadata, dict) else None
    return isinstance(eef, dict) and eef.get("enabled") is True


def _validate_eef_datasets(episode: h5py.File, frame_count: int, report: dict) -> None:
    for arm in ("puppet_left", "puppet_right"):
        dataset_path = f"{EEF_GROUP_PATH}/{arm}"
        dataset = _get_dataset(episode, dataset_path, report)
        if dataset is None:
            continue
        expected_shape = (frame_count, EEF_DIMENSION)
        if dataset.shape != expected_shape:
            report["errors"].append(f"{dataset_path} shape {dataset.shape} does not match expected {expected_shape}")
        if dataset.dtype != np.dtype("float32"):
            report["errors"].append(f"{dataset_path} dtype {dataset.dtype} is not float32")
        if dataset.maxshape != (None, EEF_DIMENSION):
            report["errors"].append(f"{dataset_path} is not two-dimensional and extendible")
        if dataset.shape == expected_shape and not np.isfinite(dataset[:]).all():
            report["errors"].append(f"{dataset_path} contains non-finite values")


def _dataset_length(episode: h5py.File, dataset_path: str, report: dict) -> int:
    dataset = _get_dataset(episode, dataset_path, report)
    if dataset is None:
        return -1
    return int(dataset.shape[0])


def _get_dataset(episode: h5py.File, dataset_path: str, report: dict) -> h5py.Dataset | None:
    if dataset_path not in episode:
        report["errors"].append(f"missing dataset {dataset_path}")
        return None
    dataset = episode[dataset_path]
    if not isinstance(dataset, h5py.Dataset):
        report["errors"].append(f"{dataset_path} is not a dataset")
        return None
    return dataset


def _coerce_vector(value: Any, field: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (FRAME_DIMENSION,):
        raise ValueError(f"{field} must contain exactly 14 values")
    if not np.isfinite(vector).all():
        raise ValueError(f"{field} values must be finite")
    return vector.copy()


def _coerce_eef(value: dict[str, Any] | None) -> dict[str, np.ndarray]:
    if not isinstance(value, dict) or set(value) != {"puppet_left", "puppet_right"}:
        raise ValueError("EEF data must contain exactly puppet_left and puppet_right")
    result: dict[str, np.ndarray] = {}
    for arm in ("puppet_left", "puppet_right"):
        vector = np.asarray(value[arm], dtype=np.float32)
        if vector.shape != (EEF_DIMENSION,):
            raise ValueError(f"EEF {arm} must contain exactly six values")
        if not np.isfinite(vector).all():
            raise ValueError(f"EEF {arm} values must be finite")
        result[arm] = vector.copy()
    return result


def _coerce_images(images: dict[str, bytes]) -> dict[str, bytes]:
    if set(images) != set(IMAGE_SENSORS):
        raise ValueError(f"images must contain exactly {', '.join(IMAGE_SENSORS)}")
    copied: dict[str, bytes] = {}
    for camera in IMAGE_SENSORS:
        image_bytes = images[camera]
        if not isinstance(image_bytes, bytes):
            raise ValueError(f"{camera} image must be JPEG bytes")
        copied[camera] = bytes(image_bytes)
    return copied


def _coerce_sensor_mapping(
    values: dict[str, float],
    field: str,
    required_sensors: tuple[str, ...] = REQUIRED_SENSORS,
) -> dict[str, float]:
    if set(values) != set(required_sensors):
        raise ValueError(f"{field} must contain exactly {', '.join(required_sensors)}")
    return {sensor: _coerce_finite_float(values[sensor], f"{field}.{sensor}") for sensor in required_sensors}


def _coerce_finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        coerced = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(coerced):
        raise ValueError(f"{field} must be finite")
    return coerced


def _decode_jpeg_array(encoded: Any) -> bool:
    array = np.asarray(encoded, dtype=np.uint8)
    if array.ndim != 1 or array.size == 0:
        return False
    decoded = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return decoded is not None and decoded.size > 0


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _ensure_json_serializable(value: Any, field: str) -> None:
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be JSON serializable") from exc
