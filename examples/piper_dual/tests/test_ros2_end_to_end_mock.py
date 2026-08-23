"""No-hardware end-to-end ROS 2 mock flow through public collection and rendering interfaces."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import base64
import json
import sys

import cv2
import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collect_data_ros2 import CollectorController
from frame_synchronizer import FRAME_DIMENSION
from frame_synchronizer import IMAGE_SENSORS
from frame_synchronizer import JOINT_SENSORS
from frame_synchronizer import REQUIRED_SENSORS
from frame_synchronizer import SynchronizedFrame
from render_dataset import COMBINED_VIDEO_FILENAME
from render_dataset import QUALITY_FILENAME
from render_dataset import detect_schema
from render_dataset import render_episode
from ros2_backend import Ros2BackendClient
from streaming_hdf5 import SCHEMA_VERSION
from streaming_hdf5 import StreamingEpisodeWriter
from streaming_hdf5 import validate_episode
from ros2_contract import BridgeContract
from ros2_contract import EndpointPlan
from control_stack import load_profile_config
from control_stack import profile_for


CAMERA_COLORS: dict[str, tuple[int, int, int]] = {
    "cam_high": (250, 30, 20),
    "cam_left_wrist": (20, 250, 30),
    "cam_right_wrist": (30, 20, 250),
}
SENSOR_ORDER = IMAGE_SENSORS + JOINT_SENSORS


def _jpeg_bytes(color: tuple[int, int, int], *, index: int) -> bytes:
    image = np.zeros((36, 48, 3), dtype=np.uint8)
    image[:] = color
    cv2.putText(image, str(index), (3, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    assert ok
    return bytes(encoded)


def _joint_values(sensor: str, index: int) -> list[float]:
    base = {
        "puppet_left": 0.0,
        "puppet_right": 10.0,
        "master_left": 100.0,
        "master_right": 110.0,
    }[sensor]
    return [base + float(index) + joint * 0.01 for joint in range(7)]


def _sensor_event(sensor: str, timestamp: float, index: int) -> dict[str, object]:
    if sensor in IMAGE_SENSORS:
        payload = _jpeg_bytes(CAMERA_COLORS[sensor], index=index)
        return {
            "type": "sensor",
            "sensor": sensor,
            "timestamp": timestamp,
            "jpeg_b64": base64.b64encode(payload).decode("ascii"),
        }
    return {"type": "sensor", "sensor": sensor, "timestamp": timestamp, "values": _joint_values(sensor, index)}


def _ten_frame_bridge_events() -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for index in range(10):
        timestamp = 50.0 + index * 0.1
        for sensor in SENSOR_ORDER:
            events.append(_sensor_event(sensor, timestamp, index))
    return events


class InProcessBridgeBackend(Ros2BackendClient):
    """Test backend that exercises Ros2BackendClient decoding/synchronizer boundary without a process."""

    def __init__(self, events: list[dict[str, object]], *, max_sync_error: float = 0.03) -> None:
        super().__init__(bridge_python=Path(sys.executable), config=Path("mock-ros2-config.yaml"), dry_run=True)
        self.events = deque(events)
        self.started = False
        self.closed = False
        self.clear_calls = 0
        self.max_sync_error = max_sync_error
        self.synchronizer.max_error = max_sync_error
        self._started = True

    def start(self) -> None:
        self.started = True
        self._started = True

    def clear_buffers(self) -> None:
        self.clear_calls += 1
        self._clear_frame_state()

    def close(self) -> None:
        self.closed = True

    def next_frame(self, timeout: float) -> SynchronizedFrame:
        del timeout
        while self.events:
            self._handle_message(self.events.popleft())
            try:
                return self._frame_queue.get_nowait()
            except Exception:
                continue
        raise EOFError("bridge EOF before a complete synchronized frame")


def _read_video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened(), str(path)
    try:
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            assert frame is not None
            count += 1
        return count
    finally:
        capture.release()


def _run_mock_collection_and_render(tmp_path: Path) -> tuple[Path, Path, dict[str, object], list[dict]]:
    backend = InProcessBridgeBackend(_ten_frame_bridge_events())
    backend.start()
    reports: list[dict] = []
    controller = CollectorController(
        backend=backend,
        output_dir=tmp_path / "episodes",
        prompt="Fold the towel",
        jpeg_quality=90,
        writer_factory=StreamingEpisodeWriter.open,
        validator=validate_episode,
        report_sink=reports.append,
        preview=None,
        render_after_save=False,
    )

    controller.start_episode()
    for _ in range(10):
        controller.process_frame(backend.next_frame(timeout=0.1))
    final_path = controller.stop_episode()
    controller.close()

    render_dir = tmp_path / "rendered"
    render_report = render_episode(final_path, render_dir, fps=10, make_plots=False)
    return final_path, render_dir, render_report, reports


def test_ten_frame_mock_flow_writes_self_contained_hdf5_four_readable_videos_and_quality_report(tmp_path: Path) -> None:
    final_path, render_dir, render_report, collection_reports = _run_mock_collection_and_render(tmp_path)

    assert final_path == tmp_path / "episodes" / "episode_000000.hdf5"
    assert final_path.is_file()
    assert detect_schema(final_path) == SCHEMA_VERSION
    validation = validate_episode(final_path)
    assert validation["errors"] == []
    assert validation["frame_count"] == 10
    assert validation["image_decode_counts"] == {camera: 10 for camera in IMAGE_SENSORS}
    assert collection_reports[-1]["frame_count"] == 10

    with h5py.File(final_path, "r") as episode:
        assert episode.attrs["schema_version"] == SCHEMA_VERSION
        assert episode.attrs["frame_count"] == 10
        metadata = json.loads(episode.attrs["metadata_json"])
        assert metadata["prompt"] == "Fold the towel"
        assert metadata["collector"] == "collect_data_ros2.py"
        assert metadata["source"] == "ros2_bridge"
        assert episode["/observations/state"].shape == (10, FRAME_DIMENSION)
        assert episode["/action"].shape == (10, FRAME_DIMENSION)
        assert set(episode["/observations/images"].keys()) == set(IMAGE_SENSORS)

    expected_video_names = [f"{camera}.mp4" for camera in IMAGE_SENSORS] + [COMBINED_VIDEO_FILENAME]
    assert list(render_report["video_paths"].keys()) == list(IMAGE_SENSORS) + [COMBINED_VIDEO_FILENAME]
    for video_name in expected_video_names:
        path = render_dir / video_name
        assert path.is_file(), path
        assert _read_video_frame_count(path) == 10

    quality_path = render_dir / QUALITY_FILENAME
    assert quality_path.is_file()
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    assert quality == render_report
    assert quality["schema_version"] == SCHEMA_VERSION
    assert quality["frame_count"] == 10
    assert quality["image_decode_counts"] == {camera: 10 for camera in IMAGE_SENSORS}
    assert quality["errors"] == []
    assert quality["state_action_plot"]["status"] == "skipped"


def test_incomplete_frame_surfaces_explicit_eof_and_publishes_no_false_final_output(tmp_path: Path) -> None:
    events = _ten_frame_bridge_events()[: len(SENSOR_ORDER) - 1]
    backend = InProcessBridgeBackend(events)
    backend.start()
    controller = CollectorController(
        backend=backend,
        output_dir=tmp_path,
        prompt="Fold the towel",
        writer_factory=StreamingEpisodeWriter.open,
        validator=validate_episode,
        preview=None,
    )

    controller.start_episode()
    with pytest.raises(EOFError, match="bridge EOF before a complete synchronized frame"):
        backend.next_frame(timeout=0.1)
    result = controller.stop_episode()
    controller.close()

    assert result == Path()
    assert not (tmp_path / "episode_000000.hdf5").exists()
    assert not (tmp_path / "episode_000000.hdf5.partial").exists()


def test_malformed_jpeg_fails_finalization_explicitly_and_publishes_no_false_final_output(tmp_path: Path, capsys) -> None:
    events = _ten_frame_bridge_events()[: len(SENSOR_ORDER)]
    for event in events:
        if event.get("sensor") == "cam_left_wrist":
            event["jpeg_b64"] = base64.b64encode(b"not a jpeg").decode("ascii")
    backend = InProcessBridgeBackend(events)
    backend.start()
    controller = CollectorController(
        backend=backend,
        output_dir=tmp_path,
        prompt="Fold the towel",
        writer_factory=StreamingEpisodeWriter.open,
        validator=validate_episode,
        preview=None,
    )

    controller.start_episode()
    controller.process_frame(backend.next_frame(timeout=0.1))
    result = controller.stop_episode()
    controller.close()

    assert result == Path()
    output = capsys.readouterr().out
    assert "Failed to finalize" in output
    assert "cam_left_wrist" in output
    assert "not decodable JPEG bytes" in output
    assert not (tmp_path / "episode_000000.hdf5").exists()
    assert not (tmp_path / "episode_000000.hdf5.partial").exists()


def test_stale_timestamp_surfaces_explicit_eof_and_does_not_emit_or_publish_partial_frame(tmp_path: Path) -> None:
    events = []
    timestamp = 70.0
    for sensor in SENSOR_ORDER:
        offset = 0.25 if sensor == "master_right" else 0.0
        events.append(_sensor_event(sensor, timestamp + offset, 0))
    backend = InProcessBridgeBackend(events, max_sync_error=0.03)
    backend.start()
    controller = CollectorController(
        backend=backend,
        output_dir=tmp_path,
        prompt="Fold the towel",
        writer_factory=StreamingEpisodeWriter.open,
        validator=validate_episode,
        preview=None,
    )

    controller.start_episode()
    with pytest.raises(EOFError, match="bridge EOF before a complete synchronized frame"):
        backend.next_frame(timeout=0.1)
    result = controller.stop_episode()
    controller.close()

    assert backend.synchronizer.rejected_stale == 1
    assert result == Path()
    assert not (tmp_path / "episode_000000.hdf5").exists()
    assert not (tmp_path / "episode_000000.hdf5.partial").exists()


def test_bridge_eof_after_some_frames_aborts_open_writer_with_no_false_final_output(tmp_path: Path) -> None:
    backend = InProcessBridgeBackend(_ten_frame_bridge_events()[: 2 * len(SENSOR_ORDER)])
    backend.start()
    controller = CollectorController(
        backend=backend,
        output_dir=tmp_path,
        prompt="Fold the towel",
        writer_factory=StreamingEpisodeWriter.open,
        validator=validate_episode,
        preview=None,
    )

    controller.start_episode()
    controller.process_frame(backend.next_frame(timeout=0.1))
    controller.process_frame(backend.next_frame(timeout=0.1))
    with pytest.raises(EOFError, match="bridge EOF before a complete synchronized frame"):
        backend.next_frame(timeout=0.1)
    controller.close()

    assert not (tmp_path / "episode_000000.hdf5").exists()
    assert not (tmp_path / "episode_000000.hdf5.partial").exists()


def test_interrupted_finalization_aborts_partial_and_does_not_publish_final_output(tmp_path: Path, capsys) -> None:
    events = _ten_frame_bridge_events()[: len(SENSOR_ORDER)]
    backend = InProcessBridgeBackend(events)
    backend.start()

    class InterruptingWriter:
        def __init__(self, inner: StreamingEpisodeWriter) -> None:
            self.inner = inner
            self.partial_path = inner.partial_path

        def append(self, frame: SynchronizedFrame) -> None:
            self.inner.append(frame)

        def finalize(self) -> Path:
            raise KeyboardInterrupt("operator interrupted finalization")

        def abort(self, remove_partial: bool = True) -> None:
            self.inner.abort(remove_partial=remove_partial)

    def writer_factory(path: Path, *, metadata: dict, jpeg_quality: int, **kwargs) -> InterruptingWriter:
        return InterruptingWriter(StreamingEpisodeWriter.open(path, metadata=metadata, jpeg_quality=jpeg_quality, **kwargs))

    controller = CollectorController(
        backend=backend,
        output_dir=tmp_path,
        prompt="Fold the towel",
        writer_factory=writer_factory,
        validator=validate_episode,
        preview=None,
    )

    controller.start_episode()
    controller.process_frame(backend.next_frame(timeout=0.1))
    with pytest.raises(KeyboardInterrupt, match="operator interrupted finalization"):
        controller.stop_episode()
    controller.close()

    assert not (tmp_path / "episode_000000.hdf5").exists()
    assert not (tmp_path / "episode_000000.hdf5.partial").exists()


STACK_IDS = ("local-ros", "official-ros", "direct-sdk")
EXPECTED_JOINT_TOPIC_ALIASES = {
    "local-ros": {
        "/puppet/joint_left": "puppet_left",
        "/puppet/joint_right": "puppet_right",
        "/master/joint_left": "master_left",
        "/master/joint_right": "master_right",
    },
    "official-ros": {
        "/joint_left": "puppet_left",
        "/joint_right": "puppet_right",
        "/joint_states_ctrl_left": "master_left",
        "/joint_states_ctrl_right": "master_right",
    },
    "direct-sdk": {
        "/direct_sdk/joint_left": "puppet_left",
        "/direct_sdk/joint_right": "puppet_right",
        "/direct_sdk/joint_ctrl_left": "master_left",
        "/direct_sdk/joint_ctrl_right": "master_right",
    },
}


def _ten_offset_frame_events() -> list[dict[str, object]]:
    offsets = {sensor: index * 0.001 for index, sensor in enumerate(SENSOR_ORDER)}
    events: list[dict[str, object]] = []
    for frame_index in range(10):
        reference = 80.0 + frame_index * 0.1
        events.extend(_sensor_event(sensor, reference + offsets[sensor], frame_index) for sensor in SENSOR_ORDER)
    return events


@pytest.mark.parametrize("stack_id", STACK_IDS)
def test_profile_yaml_mock_flow_normalizes_ten_in_process_backend_frames(stack_id: str) -> None:
    _profile, config = load_profile_config(stack_id, profile_for(stack_id).config_path)
    contract = BridgeContract.from_mapping(config)
    endpoint_plan = EndpointPlan.for_mode(
        contract,
        dry_run=True,
        publish_actions=True,
        eef_control=False,
        include_eef=False,
    )
    backend = InProcessBridgeBackend(_ten_offset_frame_events(), max_sync_error=0.03)
    backend.start()

    frames = [backend.next_frame(timeout=0.1) for _ in range(10)]
    states = np.stack([frame.state for frame in frames])
    actions = np.stack([frame.action for frame in frames])

    assert contract.image_topics == {
        "cam_high": "/camera_f_l/color/image_raw",
        "cam_left_wrist": "/camera_l/color/image_raw",
        "cam_right_wrist": "/camera_r/color/image_raw",
    }
    assert contract.joint_topics == EXPECTED_JOINT_TOPIC_ALIASES[stack_id]
    assert tuple(contract.image_topics) + tuple(contract.joint_topics.values()) == SENSOR_ORDER
    assert endpoint_plan.action_publishers == ()
    assert states.shape == (10, FRAME_DIMENSION)
    assert actions.shape == (10, FRAME_DIMENSION)
    assert [frame.timestamp for frame in frames] == pytest.approx([80.0 + index * 0.1 for index in range(10)])
    assert all(tuple(frame.sensor_timestamps) == SENSOR_ORDER for frame in frames)
    assert max(abs(value) for frame in frames for value in frame.sync_error.values()) <= 0.03
    assert backend.synchronizer.accepted_frames == 10
