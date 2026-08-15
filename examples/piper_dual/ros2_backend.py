"""Backend client for the Piper dual ROS 2 bridge process."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import base64
import json
import queue
import subprocess
import threading
import time
from typing import Any

import numpy as np

from action_adapter import ActionAdapter
from frame_synchronizer import FrameSynchronizer
from frame_synchronizer import SynchronizedFrame
from ros2_protocol import decode_message
from ros2_protocol import encode_message

BRIDGE_SCRIPT = Path(__file__).resolve().with_name("ros2_bridge_process.py")
DEFAULT_QUEUE_SIZE = 8
DEFAULT_SYNC_ERROR = 0.03
DEFAULT_BUFFER_SECONDS = 2.0
STDERR_LOG_LIMIT = 32
PROCESS_SHUTDOWN_TIMEOUT = 5.0


@dataclass(frozen=True, slots=True)
class _BackendError:
    message: str
    priority: int


class Ros2BackendClient:
    """Launch and supervise the ROS 2 bridge sidecar."""

    def __init__(self, bridge_python: Path, config: Path, publish_actions: bool = False, dry_run: bool = True) -> None:
        self.bridge_python = Path(bridge_python)
        self.config = Path(config)
        self.publish_actions_enabled = bool(publish_actions)
        self.dry_run = bool(dry_run)
        self.complete_frame_queue_size = DEFAULT_QUEUE_SIZE
        self.max_sync_error = DEFAULT_SYNC_ERROR
        self.max_buffer_seconds = DEFAULT_BUFFER_SECONDS
        self.action_adapter = ActionAdapter()
        self.synchronizer = FrameSynchronizer(self.max_sync_error, self.max_buffer_seconds)
        self.process: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_queue: queue.Queue[SynchronizedFrame] = queue.Queue(maxsize=self.complete_frame_queue_size)
        self._stdin_lock = threading.Lock()
        self._error_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._backend_error: _BackendError | None = None
        self._stderr_lines: deque[str] = deque(maxlen=STDERR_LOG_LIMIT)
        self._dropped_complete_frames = 0
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
        args = [str(self.bridge_python), str(BRIDGE_SCRIPT), "--config", str(self.config)]
        if self.dry_run:
            args.append("--dry-run")
        if self.publish_actions_enabled:
            args.append("--publish-actions")
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
        self._raise_if_error()
        try:
            frame = self._frame_queue.get(timeout=timeout)
        except queue.Empty as exc:
            self._raise_if_error()
            if self.process is not None and self.process.poll() is not None:
                self._raise_if_error(default=f"bridge exited before producing a synchronized frame (status {self.process.returncode})")
            raise TimeoutError("Timed out waiting for a synchronized frame") from exc
        self._raise_if_error()
        return frame

    def publish_action(self, action: np.ndarray) -> None:
        self._ensure_started()
        request = self.action_adapter.to_bridge_request(action)
        self._send_request(request)

    def clear_buffers(self) -> None:
        with self._frame_lock:
            self.synchronizer.clear()
            while True:
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break

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
            value = np.asarray(message["values"], dtype=np.float32)
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
        with self._error_lock:
            if self._backend_error is None or priority >= self._backend_error.priority:
                self._backend_error = _BackendError(message, priority)

    def _raise_if_error(self, default: str | None = None) -> None:
        if self._backend_error is not None:
            raise RuntimeError(self._backend_error.message)
        if default is not None:
            raise RuntimeError(default)

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
