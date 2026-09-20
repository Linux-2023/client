"""Bounded, asynchronous recording of the synchronized policy observation views."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import queue
import threading
import time
import uuid

import cv2
import imageio.v2 as imageio
import numpy as np
from openpi_client.runtime.subscriber import Subscriber

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def compose_tshape(images: dict) -> np.ndarray:
    """RGB: full-width front above left/right wrists, without swapping or mirroring."""
    frames = []
    for name in CAMERAS:
        frame = np.asarray(images[name])
        if frame.ndim != 3 or frame.shape[0] != 3 or frame.dtype != np.uint8:
            raise ValueError(f"{name}: expected uint8 CHW RGB")
        frames.append(np.transpose(frame, (1, 2, 0)))
    h, w = frames[0].shape[:2]
    top = cv2.resize(frames[0], (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
    wrists = [cv2.resize(frame, (w, h)) for frame in frames[1:]]
    return np.concatenate((top, np.concatenate(wrists, axis=1)), axis=0)


class TShapeVideoSaver(Subscriber):
    """One MP4 + JSONL per episode; worker encoding never waits in the action loop.

    MP4 uses configured nominal FPS. JSONL retains actual receipt/source times;
    pauses or dropped recording frames must be assessed using those timestamps.
    SIGINT/SIGTERM are finalized by the entrypoint; SIGKILL cannot be finalized.
    """

    def __init__(self, out_dir: Path, fps: float = 30, queue_size: int = 64):
        if not math.isfinite(fps) or fps <= 0 or queue_size <= 0:
            raise ValueError("fps and queue_size must be positive")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.queue_size = queue_size
        self._thread = None
        self._error = None
        self._step = 0
        self.dropped = 0

    def on_episode_start(self):
        self.close()
        self._queue = queue.Queue(self.queue_size)
        self._stop = threading.Event()
        self._error = None
        self._step = self.dropped = 0
        name = f"tshape_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.video_path = self.out_dir / f"{name}.mp4"
        self.metadata_path = self.out_dir / f"{name}.jsonl"
        self._thread = threading.Thread(target=self._write, name="tshape-video", daemon=True)
        self._thread.start()
        logging.info("T-shape recording: %s", self.video_path)

    def on_step(self, observation, action):
        if self._error is not None:
            raise RuntimeError("T-shape video writer failed") from self._error
        if self._thread is None:
            raise RuntimeError("episode recording has not started")
        metadata = {
            "event": "frame", "step": self._step,
            "timestamp_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(),
            "sensor_timestamps": {k: float(v) for k, v in observation.get("timestamps", {}).items()},
            "sync_error": float(observation.get("sync_error", 0)),
            "state": np.asarray(observation.get("state", [])).tolist(),
            "action": np.asarray(action.get("actions", [])).tolist(),
            "dropped_before": self.dropped,
        }
        self._step += 1
        # Only the small model observation is copied; no extra ROS image subscription.
        frame = compose_tshape(observation["images"])
        try:
            self._queue.put_nowait((frame, metadata))
        except queue.Full:
            self.dropped += 1
            if self.dropped == 1:
                logging.warning("Video encoder backlog: dropping recording frames; see JSONL")

    def _write(self):
        writer = None
        try:
            with self.metadata_path.open("x") as stream:
                stream.write(json.dumps({"event": "start", "fps": self.fps,
                    "layout": "front full width above left wrist / right wrist",
                    "source": "synchronized policy observation, CHW RGB; model resolution",
                    "timebase": "nominal FPS playback; actual sensor/host timestamps per frame",
                    "timestamp_ns": time.time_ns()}) + "\n")
                stream.flush()
                count = 0
                while not self._stop.is_set() or not self._queue.empty():
                    try:
                        frame, metadata = self._queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if writer is None:
                        writer = imageio.get_writer(str(self.video_path), fps=self.fps,
                            codec="libx264", pixelformat="yuv420p", macro_block_size=2,
                            ffmpeg_params=["-preset", "ultrafast", "-crf", "20", "-threads", "1"])
                    writer.append_data(frame)
                    metadata["video_frame"] = count
                    stream.write(json.dumps(metadata, allow_nan=False) + "\n")
                    stream.flush()
                    count += 1
                if writer is not None:
                    writer.close()
                    writer = None
                stream.write(json.dumps({"event": "end", "frames": count,
                    "steps": self._step, "dropped_frames": self.dropped,
                    "timestamp_ns": time.time_ns()}) + "\n")
        except Exception as exc:
            self._error = exc
            logging.exception("T-shape recording failed")
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception as exc:
                    self._error = exc

    def on_episode_end(self):
        self.close()

    def close(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
            if self._error is not None:
                raise RuntimeError("T-shape video writer failed") from self._error
