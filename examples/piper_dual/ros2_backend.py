"""Backend client for the Piper dual ROS 2 bridge process."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import base64
import json
import math
import queue
import subprocess
import threading
import time
from typing import Any

import numpy as np

from action_adapter import ActionAdapter
from eef_action_adapter import EefActionAdapter
from frame_synchronizer import EEF_DIMENSION
from frame_synchronizer import EEF_SENSORS
from frame_synchronizer import FrameSynchronizer
from frame_synchronizer import JOINT_DIMENSION
from frame_synchronizer import SynchronizedFrame
from ros2_protocol import decode_message
from ros2_protocol import encode_message

BRIDGE_SCRIPT = Path(__file__).resolve().with_name("ros2_bridge_process.py")
LOCAL_BRIDGE_SCRIPT = Path(__file__).resolve().with_name("ros2_local_bridge_process.py")
DEFAULT_QUEUE_SIZE = 8
DEFAULT_SYNC_ERROR = 0.03
DEFAULT_BUFFER_SECONDS = 2.0
STDERR_LOG_LIMIT = 32
PROCESS_SHUTDOWN_TIMEOUT = 5.0
_ERROR_SENTINEL = object()


@dataclass(frozen=True, slots=True)
class _BackendError:
    message: str
    priority: int


class Ros2BackendClient:
    """Launch and supervise the ROS 2 bridge sidecar."""

    def __init__(
        self,
        bridge_python: Path,
        config: Path,
        publish_actions: bool = False,
        dry_run: bool = True,
        *,
        eef_left_topic: str | None = None,
        eef_right_topic: str | None = None,
        action_adapter: Any | None = None,
        eef_control: bool = False,
        eef_left_action_topic: str = "/pos_left_cmd",
        eef_right_action_topic: str = "/pos_right_cmd",
        control_stack: str = "official-ros",
    ) -> None:
        if (eef_left_topic is None) != (eef_right_topic is None):
            raise ValueError("EEF left and right topics must be provided together")
        if eef_control and action_adapter is None:
            action_adapter = EefActionAdapter()
        self.bridge_python = Path(bridge_python)
        self.config = Path(config)
        self.control_stack = str(control_stack)
        self.publish_actions_enabled = bool(publish_actions)
        self.dry_run = bool(dry_run)
        self.eef_left_topic = eef_left_topic
        self.eef_right_topic = eef_right_topic
        self.include_eef = eef_left_topic is not None or bool(eef_control)
        self.eef_control = bool(eef_control)
        self.eef_left_action_topic = str(eef_left_action_topic)
        self.eef_right_action_topic = str(eef_right_action_topic)
        self.complete_frame_queue_size = DEFAULT_QUEUE_SIZE
        self.max_sync_error = DEFAULT_SYNC_ERROR
        self.max_buffer_seconds = DEFAULT_BUFFER_SECONDS
        self.action_adapter = action_adapter or ActionAdapter()
        self.synchronizer = FrameSynchronizer(
            self.max_sync_error,
            self.max_buffer_seconds,
            include_eef=self.include_eef,
        )
        self.process: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_queue: queue.Queue[SynchronizedFrame | object] = queue.Queue(maxsize=self.complete_frame_queue_size)
        self._stdin_lock = threading.Lock()
        self._error_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._clear_lock = threading.Lock()
        self._barrier_condition = threading.Condition()
        self._backend_error: _BackendError | None = None
        self._stderr_lines: deque[str] = deque(maxlen=STDERR_LOG_LIMIT)
        self._pending_ping_barrier_id: str | None = None
        self._ping_barrier_reached = False
        self._ping_barrier_released = False
        self._dropped_complete_frames = 0
        self.hardware_fault: dict[str, Any] | None = None
        self._started = False
        self._closed = False

    @property
    def dropped_complete_frames(self) -> int:
        return self._dropped_complete_frames

    @property
    def stderr_log(self) -> tuple[str, ...]:
        return tuple(self._stderr_lines)

    def start(self) -> None:
        if self._started:
            return
        if self.dry_run and self.publish_actions_enabled:
            raise ValueError("Unsafe configuration: dry_run=True together with publish_actions=True")
        bridge_script = LOCAL_BRIDGE_SCRIPT if self.control_stack == "local-ros" else BRIDGE_SCRIPT
        args = [str(self.bridge_python), str(bridge_script), "--config", str(self.config)]
        if self.control_stack != "local-ros":
            args.extend(["--control-stack", self.control_stack])
        if self.dry_run:
            args.append("--dry-run")
        if self.publish_actions_enabled:
            args.append("--publish-actions")
        if self.include_eef:
            args.append("--include-eef")
        if self.eef_left_topic is not None:
            args.extend(["--eef-left-topic", str(self.eef_left_topic), "--eef-right-topic", str(self.eef_right_topic)])
        if self.eef_control:
            args.extend([
                "--eef-control",
                "--eef-left-action-topic",
                self.eef_left_action_topic,
                "--eef-right-action-topic",
                self.eef_right_action_topic,
            ])
        self.process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._started = True
        self._closed = False
        self._stop_event.clear()
        self._reader_thread = threading.Thread(target=self._reader_loop, name="piper-bridge-reader", daemon=True)
        self._stderr_thread = threading.Thread(target=self._stderr_loop, name="piper-bridge-stderr", daemon=True)
        self._reader_thread.start()
        self._stderr_thread.start()

    def next_frame(self, timeout: float) -> SynchronizedFrame:
        self._ensure_started()
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = time.monotonic() + timeout
        while True:
            self._raise_if_error()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if self.process is not None and self.process.poll() is not None:
                    self._raise_if_error(default=f"bridge exited before producing a synchronized frame (status {self.process.returncode})")
                raise TimeoutError("Timed out waiting for a synchronized frame")
            try:
                frame = self._frame_queue.get(timeout=remaining)
            except queue.Empty as exc:
                self._raise_if_error()
                if self.process is not None and self.process.poll() is not None:
                    self._raise_if_error(default=f"bridge exited before producing a synchronized frame (status {self.process.returncode})")
                raise TimeoutError("Timed out waiting for a synchronized frame") from exc
            if frame is _ERROR_SENTINEL:
                self._raise_if_error(default="Bridge failed while waiting for a synchronized frame")
                continue
            self._raise_if_error()
            return frame

    def publish_action(self, action: np.ndarray) -> None:
        self._ensure_started()
        self._raise_if_error()
        request = self.action_adapter.to_bridge_request(action)
        self._send_request(request)

    def clear_buffers(self) -> None:
        self._ensure_started()
        self._raise_if_error()
        with self._clear_lock:
            self._clear_frame_state()
            barrier_id = str(time.monotonic_ns())
            with self._barrier_condition:
                self._pending_ping_barrier_id = barrier_id
                self._ping_barrier_reached = False
                self._ping_barrier_released = False
            try:
                self._send_request({"type": "ping", "id": barrier_id})
                self._wait_for_ping_barrier(barrier_id)
                self._clear_frame_state()
            finally:
                self._release_ping_barrier(barrier_id)
            self._raise_if_error()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.process is not None:
            try:
                self._send_request({"type": "stop"})
            except Exception:
                pass
            try:
                if self.process.stdin is not None:
                    self.process.stdin.close()
            except Exception:
                pass
            try:
                self.process.wait(timeout=PROCESS_SHUTDOWN_TIMEOUT)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=PROCESS_SHUTDOWN_TIMEOUT)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=PROCESS_SHUTDOWN_TIMEOUT)
        self._stop_event.set()
        with self._barrier_condition:
            self._barrier_condition.notify_all()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=PROCESS_SHUTDOWN_TIMEOUT)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=PROCESS_SHUTDOWN_TIMEOUT)

    def _reader_loop(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            for line in self.process.stdout:
                try:
                    message = decode_message(line)
                except ValueError as exc:
                    self._set_error(f"Protocol message is not valid JSON: {exc}")
                    break
                try:
                    self._handle_message(message)
                except Exception as exc:  # noqa: BLE001 - bridge failures must be surfaced verbosely.
                    self._set_error(str(exc))
                    break
            else:
                try:
                    returncode = self.process.wait(timeout=0.1) if self.process is not None else None
                except subprocess.TimeoutExpired:
                    self._set_error(self._format_exit_error("stdout closed before bridge exited"), priority=10)
                else:
                    if returncode is not None and returncode != 0:
                        self._set_error(self._format_exit_error(f"bridge exited with status {returncode}"), priority=20)
                    else:
                        self._set_error(self._format_exit_error("stdout closed before bridge exited"), priority=10)
        finally:
            self._stop_event.set()
            if self.process is not None:
                returncode = self.process.poll()
                if returncode is not None and returncode != 0:
                    self._set_error(self._format_exit_error(f"bridge exited with status {returncode}"), priority=20)

    def _stderr_loop(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for line in self.process.stderr:
            self._stderr_lines.append(line.rstrip("\n"))

    def _handle_message(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        if message_type == "status":
            self._handle_status_message(message)
            return
        if message_type == "error":
            raise RuntimeError(str(message.get("message", "bridge error")))
        if message_type != "sensor":
            raise RuntimeError(f"Unexpected bridge message type: {message_type!r}")

        sensor = message.get("sensor")
        timestamp = message.get("timestamp")
        if not isinstance(sensor, str):
            raise RuntimeError("sensor event is missing sensor name")

        if "jpeg_b64" in message:
            value = self._decode_jpeg_payload(message["jpeg_b64"])
        elif "values" in message:
            if sensor in EEF_SENSORS:
                value = self._decode_vector_payload(message["values"], EEF_DIMENSION, "EEF")
            else:
                value = self._decode_vector_payload(message["values"], JOINT_DIMENSION, "joint")
        else:
            raise RuntimeError("sensor event must contain jpeg_b64 or values")

        with self._frame_lock:
            self.synchronizer.push(sensor, timestamp, value)
            frame = self.synchronizer.try_sync()
            if frame is not None:
                self._enqueue_frame(frame)

    def _enqueue_frame(self, frame: SynchronizedFrame) -> None:
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            self._frame_queue.put_nowait(frame)
            self._dropped_complete_frames += 1

    def _decode_jpeg_payload(self, payload: object) -> bytes:
        if not isinstance(payload, str):
            raise RuntimeError("sensor image payload must be base64 text")
        try:
            return base64.b64decode(payload.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError) as exc:
            raise RuntimeError("sensor image payload is not valid base64") from exc

    def _decode_joint_payload(self, payload: object) -> np.ndarray:
        return self._decode_vector_payload(payload, JOINT_DIMENSION, "joint")

    def _decode_vector_payload(self, payload: object, dimension: int, label: str) -> np.ndarray:
        if not isinstance(payload, (list, tuple)):
            raise RuntimeError(f"{label} sensor values must be a list or tuple")
        if len(payload) != dimension:
            word = "six" if dimension == EEF_DIMENSION else "seven"
            raise RuntimeError(f"{label} sensor values must contain exactly {word} values")

        values: list[float] = []
        for raw_value in payload:
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise RuntimeError(f"{label} sensor values must be real JSON numbers")
            value = float(raw_value)
            if not math.isfinite(value):
                raise RuntimeError(f"{label} sensor values must be finite")
            values.append(value)
        return np.asarray(values, dtype=np.float32)

    def _handle_status_message(self, message: dict[str, Any]) -> None:
        state = message.get("state")
        if state == "hardware_fault":
            self._record_hardware_fault(message)
            return
        if state != "pong":
            return
        metadata = message.get("metadata")
        pong_id = metadata.get("id") if isinstance(metadata, dict) else message.get("id")
        with self._barrier_condition:
            pending_id = self._pending_ping_barrier_id
            if pending_id is None:
                return
            if pong_id is not None and str(pong_id) != pending_id:
                return
            self._ping_barrier_reached = True
            self._barrier_condition.notify_all()
            while self._pending_ping_barrier_id == pending_id and not self._ping_barrier_released and not self._stop_event.is_set():
                self._barrier_condition.wait(timeout=0.05)


    def _record_hardware_fault(self, message: dict[str, Any]) -> None:
        metadata = message.get("metadata")
        fault_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        self.hardware_fault = fault_metadata
        reason = fault_metadata.get("reason", "hardware fault")
        if not self.dry_run:
            self._set_error(f"hardware fault: {reason}", priority=30)

    def _clear_frame_state(self) -> None:
        with self._frame_lock:
            self.synchronizer.clear()
            while True:
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break

    def _wait_for_ping_barrier(self, barrier_id: str) -> None:
        deadline = time.monotonic() + PROCESS_SHUTDOWN_TIMEOUT
        with self._barrier_condition:
            while self._pending_ping_barrier_id == barrier_id and not self._ping_barrier_reached:
                self._raise_if_error()
                if self.process is not None and self.process.poll() is not None:
                    self._raise_if_error(default=f"bridge exited before acknowledging ping barrier (status {self.process.returncode})")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for bridge ping barrier")
                self._barrier_condition.wait(timeout=min(0.05, remaining))

    def _release_ping_barrier(self, barrier_id: str | None = None) -> None:
        with self._barrier_condition:
            if barrier_id is None or self._pending_ping_barrier_id == barrier_id:
                self._pending_ping_barrier_id = None
                self._ping_barrier_released = True
                self._barrier_condition.notify_all()

    def _send_request(self, request: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Bridge process has not been started")
        line = encode_message(request)
        with self._stdin_lock:
            if self.process.stdin.closed:
                raise RuntimeError("Bridge stdin is closed")
            self.process.stdin.write(line)
            self.process.stdin.flush()

    def _set_error(self, message: str, priority: int = 10) -> None:
        changed = False
        with self._error_lock:
            if self._backend_error is None or priority >= self._backend_error.priority:
                self._backend_error = _BackendError(message, priority)
                changed = True
        if changed:
            self._wake_frame_waiters()
            with self._barrier_condition:
                self._barrier_condition.notify_all()

    def _raise_if_error(self, default: str | None = None) -> None:
        with self._error_lock:
            error = self._backend_error
        if error is not None:
            raise RuntimeError(error.message)
        if default is not None:
            raise RuntimeError(default)

    def _wake_frame_waiters(self) -> None:
        try:
            self._frame_queue.put_nowait(_ERROR_SENTINEL)
            return
        except queue.Full:
            pass
        try:
            self._frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._frame_queue.put_nowait(_ERROR_SENTINEL)
        except queue.Full:
            pass

    def _format_exit_error(self, message: str) -> str:
        stderr = "; ".join(self._stderr_lines)
        if stderr:
            return f"{message}: {stderr}"
        return message

    def _ensure_started(self) -> None:
        if not self._started or self.process is None:
            raise RuntimeError("Bridge process has not been started")

    def _validate_queue_size(self, value: int) -> int:
        if type(value) is not int or value <= 0:
            raise ValueError("complete_frame_queue_size must be a positive integer")
        return value
