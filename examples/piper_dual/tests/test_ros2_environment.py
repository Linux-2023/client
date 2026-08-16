"""Focused tests for the safe ROS 2 dual-arm environment."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

workspace_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(workspace_root / "packages/openpi-client/src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frame_synchronizer import IMAGE_SENSORS
from frame_synchronizer import REQUIRED_SENSORS
from frame_synchronizer import SynchronizedFrame
from observation_adapter import DEFAULT_CAMERA_SPECS
from ros2_environment import Ros2DualEnvironment


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:] = np.array(color, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _make_frame(
    timestamp: float,
    *,
    missing_cameras: set[str] | None = None,
    repeat_action: np.ndarray | None = None,
) -> SynchronizedFrame:
    missing = missing_cameras or set()
    images = {
        spec.name: _jpeg_bytes((10 + index, 20 + index, 30 + index))
        for index, spec in enumerate(DEFAULT_CAMERA_SPECS)
        if spec.name not in missing
    }
    state = np.arange(14, dtype=np.float32)
    action = np.arange(14, dtype=np.float32) if repeat_action is None else np.asarray(repeat_action, dtype=np.float32)
    sensor_timestamps = {sensor: timestamp + index * 0.01 for index, sensor in enumerate(REQUIRED_SENSORS)}
    sync_error = {sensor: index * 0.001 for index, sensor in enumerate(REQUIRED_SENSORS)}
    return SynchronizedFrame(
        timestamp=timestamp,
        images=images,
        state=state,
        action=action,
        sensor_timestamps=sensor_timestamps,
        sync_error=sync_error,
    )


class FakeBackend:
    def __init__(
        self,
        frames: list[SynchronizedFrame] | tuple[SynchronizedFrame, ...] = (),
        *,
        repeat_last: bool = False,
        next_frame_error: Exception | None = None,
    ) -> None:
        self.frames = deque(frames)
        self.repeat_last = repeat_last
        self.next_frame_error = next_frame_error
        self.last_frame = frames[-1] if frames else None
        self.start_calls = 0
        self.clear_calls = 0
        self.close_calls = 0
        self.next_frame_calls = 0
        self.publish_action_calls: list[np.ndarray] = []

    def start(self) -> None:
        self.start_calls += 1

    def clear_buffers(self) -> None:
        self.clear_calls += 1

    def next_frame(self, timeout: float) -> SynchronizedFrame:
        self.next_frame_calls += 1
        if self.next_frame_error is not None:
            raise self.next_frame_error
        if self.frames:
            self.last_frame = self.frames.popleft()
            return self.last_frame
        if self.repeat_last and self.last_frame is not None:
            return self.last_frame
        raise TimeoutError("Timed out waiting for a synchronized frame")

    def publish_action(self, action: np.ndarray) -> None:
        self.publish_action_calls.append(np.asarray(action))

    def close(self) -> None:
        self.close_calls += 1


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend([_make_frame(1.0)])


def test_reset_starts_backend_and_clears_buffers_without_publishing_actions(backend: FakeBackend) -> None:
    env = Ros2DualEnvironment(backend=backend, prompt="Fold the towel")

    env.reset()

    assert backend.start_calls == 1
    assert backend.clear_calls == 1
    assert backend.publish_action_calls == []
    assert env.is_episode_complete() is False


def test_get_observation_maps_ros2_frame_to_model_contract() -> None:
    backend = FakeBackend([_make_frame(2.0)])
    env = Ros2DualEnvironment(backend=backend, prompt="stack", watchdog_timeout=0.1)

    env.reset()
    observation = env.get_observation()

    assert set(observation) == {"observation.state", "images", "timestamps", "sync_error", "task"}
    np.testing.assert_array_equal(observation["observation.state"], np.arange(14, dtype=np.float32))
    assert observation["observation.state"].dtype == np.float32
    assert observation["task"] == "stack"
    assert set(observation["images"]) == {spec.name for spec in DEFAULT_CAMERA_SPECS}
    for image in observation["images"].values():
        assert image.shape == (3, 224, 224)
        assert image.dtype == np.uint8


def test_get_observation_raises_when_required_camera_is_missing() -> None:
    backend = FakeBackend([_make_frame(3.0, missing_cameras={"cam_right_wrist"})])
    env = Ros2DualEnvironment(backend=backend, prompt="stack", watchdog_timeout=0.1)

    env.reset()

    with pytest.raises(ValueError, match="cam_right_wrist"):
        env.get_observation()


@pytest.mark.parametrize(
    "action",
    [
        [0.0] * 13,
        [0.0] * 15,
        [0.0] * 13 + [np.nan],
        [0.0] * 13 + [np.inf],
        [0.0] * 13 + [-np.inf],
    ],
)
def test_apply_action_rejects_invalid_14d_actions(action: list[float]) -> None:
    backend = FakeBackend()
    env = Ros2DualEnvironment(backend=backend, dry_run=True, publish_actions=False)

    env.reset()

    with pytest.raises(ValueError):
        env.apply_action({"actions": action})

    assert backend.publish_action_calls == []

def test_invalid_action_marks_episode_complete_and_disables_future_publish() -> None:
    backend = FakeBackend()
    env = Ros2DualEnvironment(backend=backend, dry_run=False, publish_actions=True)

    env.reset()

    with pytest.raises(ValueError):
        env.apply_action({"actions": [0.0] * 13})

    assert env.is_episode_complete() is True
    env.apply_action({"actions": np.arange(14, dtype=np.float32)})
    assert backend.publish_action_calls == []


def test_apply_action_does_not_publish_in_dry_run() -> None:
    backend = FakeBackend()
    env = Ros2DualEnvironment(backend=backend, dry_run=True, publish_actions=False)

    env.reset()
    env.apply_action({"actions": np.arange(14, dtype=np.float64)})

    assert backend.publish_action_calls == []


def test_apply_action_publishes_only_when_explicitly_enabled() -> None:
    backend = FakeBackend()
    env = Ros2DualEnvironment(backend=backend, dry_run=False, publish_actions=True)

    env.reset()
    env.apply_action({"actions": np.arange(14, dtype=np.float64)})

    assert len(backend.publish_action_calls) == 1
    np.testing.assert_array_equal(backend.publish_action_calls[0], np.arange(14, dtype=np.float32))
    assert backend.publish_action_calls[0].dtype == np.float32

def test_apply_action_enforces_configured_per_step_delta_limit() -> None:
    backend = FakeBackend()
    env = Ros2DualEnvironment(backend=backend, dry_run=False, publish_actions=True, max_action_delta=0.1)

    env.reset()
    env.apply_action({"actions": np.zeros(14, dtype=np.float32)})

    with pytest.raises(ValueError, match="max_action_delta"):
        env.apply_action({"actions": np.ones(14, dtype=np.float32)})

    assert env.is_episode_complete() is True
    assert len(backend.publish_action_calls) == 1



def test_stale_observation_watchdog_raises_when_timestamp_stops_advancing() -> None:
    backend = FakeBackend([_make_frame(4.0)], repeat_last=True)
    env = Ros2DualEnvironment(backend=backend, prompt="stack", watchdog_timeout=0.01)

    env.reset()
    first = env.get_observation()
    assert first["task"] == "stack"

    with pytest.raises(TimeoutError, match="stale observation"):
        env.get_observation()

    assert env.is_episode_complete() is True


def test_get_observation_surfaces_bridge_exit_without_deadlock() -> None:
    backend = FakeBackend(next_frame_error=RuntimeError("bridge exited with status 23"))
    env = Ros2DualEnvironment(backend=backend, prompt="stack", watchdog_timeout=0.1)

    env.reset()

    with pytest.raises(RuntimeError, match="bridge exited with status 23"):
        env.get_observation()

    assert env.is_episode_complete() is True


def test_close_shuts_down_bridge_cleanly_without_publishing() -> None:
    backend = FakeBackend([_make_frame(5.0)])
    env = Ros2DualEnvironment(backend=backend, dry_run=True, publish_actions=False)

    env.reset()
    env.close()
    env.close()

    assert backend.close_calls == 1
    assert backend.publish_action_calls == []
    assert env.is_episode_complete() is True


def test_ros2_environment_module_does_not_import_sdk_hardware_modules() -> None:
    forbidden = {"piper_sdk", "piper_controller", "piper_dual_controller", "cameras"}

    assert forbidden.isdisjoint(sys.modules)

    backend = FakeBackend([_make_frame(6.0)])
    env = Ros2DualEnvironment(backend=backend, prompt="stack")
    env.reset()

    assert forbidden.isdisjoint(sys.modules)
