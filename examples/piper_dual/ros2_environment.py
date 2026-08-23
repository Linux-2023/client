"""Safe ROS 2 dual-arm Piper environment wrapper."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from action_adapter import ActionAdapter
from observation_adapter import ObservationAdapter
from ros2_backend import Ros2BackendClient
from ros2_contract import BridgeContract

DEFAULT_BRIDGE_PYTHON = Path("/usr/bin/python3")
DEFAULT_ROS2_CONFIG = Path(__file__).resolve().with_name("ros2_piper_dual.yaml")
DEFAULT_FRAME_TIMEOUT = 0.1
DEFAULT_WATCHDOG_TIMEOUT = 5.0


class Ros2DualEnvironment(_environment.Environment):
    """Environment adapter that speaks to the safe ROS 2 bridge backend."""

    def __init__(
        self,
        *,
        backend: Any | None = None,
        bridge_python: Path = DEFAULT_BRIDGE_PYTHON,
        ros2_config: Path = DEFAULT_ROS2_CONFIG,
        dry_run: bool = True,
        publish_actions: bool = False,
        observation_adapter: ObservationAdapter | None = None,
        action_adapter: ActionAdapter | None = None,
        prompt: str = "",
        watchdog_timeout: float = DEFAULT_WATCHDOG_TIMEOUT,
        frame_timeout: float = DEFAULT_FRAME_TIMEOUT,
        max_episode_steps: int = 500,
        max_action_delta: float | None = None,
        control_stack: str = "official-ros",
        **_: Any,
    ) -> None:
        if type(prompt) is not str:
            raise ValueError("prompt must be a string")
        if dry_run and publish_actions:
            raise ValueError("publish_actions requires dry_run=False")

        contract = BridgeContract.from_mapping(self._load_config(ros2_config))
        if max_action_delta is None:
            max_action_delta_value = contract.control.max_action_delta
        else:
            max_action_delta_value = float(max_action_delta)
            if not np.isfinite(max_action_delta_value) or max_action_delta_value < 0:
                raise ValueError("max_action_delta must be finite non-negative")

        self._backend = backend or Ros2BackendClient(
            bridge_python=bridge_python,
            config=ros2_config,
            dry_run=dry_run,
            publish_actions=publish_actions,
            control_stack=control_stack,
        )
        self._observation_adapter = observation_adapter or ObservationAdapter(task=prompt)
        self._action_adapter = action_adapter or ActionAdapter()
        self._prompt = prompt
        self._watchdog_timeout = float(watchdog_timeout)
        self._frame_timeout = float(frame_timeout)
        self._max_episode_steps = int(max_episode_steps)
        self._dry_run = bool(dry_run)
        self._publish_actions_requested = bool(publish_actions) and not self._dry_run
        self._publish_actions_locked = False
        self._publish_actions = self._publish_actions_requested
        self._max_action_delta = max_action_delta_value
        self._done = True
        self._closed = False
        self._step_count = 0
        self._last_observation_timestamp: float | None = None
        self._previous_action: np.ndarray | None = None

    @staticmethod
    def _load_config(ros2_config: Path) -> dict[str, Any]:
        import yaml

        config_path = Path(ros2_config)
        with config_path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, dict):
            raise ValueError("ROS 2 config must be a mapping")
        return data

    @override
    def reset(self) -> None:
        self._ensure_open()
        try:
            self._backend.start()
            self._backend.clear_buffers()
        except Exception:
            self._fail_episode()
            raise
        self._done = False
        self._step_count = 0
        self._last_observation_timestamp = None
        self._previous_action = None
        self._publish_actions = self._publish_actions_requested and not self._publish_actions_locked

    @override
    def is_episode_complete(self) -> bool:
        if self._done or self._step_count >= self._max_episode_steps:
            self._done = True
        return self._done

    @override
    def get_observation(self) -> dict[str, Any]:
        self._ensure_open()
        deadline = time.monotonic() + self._watchdog_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._fail_episode()
                raise TimeoutError("stale observation watchdog timed out")

            timeout = min(self._frame_timeout, remaining)
            try:
                frame = self._backend.next_frame(timeout=timeout)
            except TimeoutError:
                continue
            except Exception:
                self._fail_episode()
                raise

            try:
                timestamp = float(frame.timestamp)
                if self._last_observation_timestamp is not None and timestamp <= self._last_observation_timestamp:
                    time.sleep(min(0.001, max(0.0, deadline - time.monotonic())))
                    continue
                observation = self._observation_adapter.adapt(
                    {
                        "state": frame.state,
                        "action": frame.action,
                        "eef": frame.eef,
                        "images": frame.images,
                        "timestamps": frame.sensor_timestamps,
                        "sync_error": self._sync_error_scalar(frame.sync_error),
                        "task": self._prompt,
                    }
                )
                observation["state"] = observation["observation.state"]
                observation["prompt"] = observation["task"]
            except Exception:
                self._fail_episode()
                raise

            self._last_observation_timestamp = timestamp
            return observation

    @override
    def apply_action(self, action: dict[str, Any]) -> None:
        self._ensure_open()
        if self._publish_actions_locked:
            raise RuntimeError(
                "ROS 2 environment is locked after a fatal fault; create a new environment before applying actions."
            )
        if self._done:
            raise RuntimeError("ROS 2 episode is complete; call reset() before applying actions.")
        if not isinstance(action, dict):
            self._fail_episode()
            raise ValueError("Action must be a JSON object")

        raw_action = action.get("actions")
        if raw_action is None:
            self._fail_episode()
            raise ValueError("Action dict must contain 'actions' key with joint positions.")

        try:
            validated = self._action_adapter.validate(raw_action)
        except Exception:
            self._fail_episode()
            raise
        validated = np.array(validated, dtype=np.float32, copy=True)
        if self._max_action_delta is not None and self._previous_action is not None:
            delta = float(np.max(np.abs(validated - self._previous_action)))
            if delta > self._max_action_delta:
                self._fail_episode()
                raise ValueError(
                    f"Action delta {delta:.6f} exceeds max_action_delta {self._max_action_delta:.6f}"
                )

        if self._publish_actions:
            try:
                self._backend.publish_action(validated.copy())
            except Exception:
                self._fail_episode()
                raise

        self._previous_action = validated.copy()
        self._step_count += 1
        if self._step_count >= self._max_episode_steps:
            self._done = True

    @override
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._done = True
        self._publish_actions = False
        self._backend.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ROS 2 environment has been closed")

    def _fail_episode(self) -> None:
        self._done = True
        self._publish_actions_locked = True
        self._publish_actions = False

    def _sync_error_scalar(self, sync_error: Any) -> float:
        raw_values = sync_error.values() if isinstance(sync_error, dict) else (sync_error,)
        values: list[float] = []
        for raw_value in raw_values:
            if isinstance(raw_value, bool):
                raise ValueError("sync_error values must be finite numbers")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("sync_error values must be finite numbers") from exc
            if not np.isfinite(value):
                raise ValueError("sync_error values must be finite")
            values.append(abs(value))
        if not values:
            raise ValueError("sync_error must include at least one value")
        return max(values)
