#!/usr/bin/env python3
"""Preview-first ROS 2 collector for Piper dual-arm synchronized episodes."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import timezone
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

from frame_synchronizer import IMAGE_SENSORS
from frame_synchronizer import SynchronizedFrame
from observation_adapter import ObservationAdapter
from ros2_backend import DEFAULT_SYNC_ERROR
from ros2_backend import Ros2BackendClient
from streaming_hdf5 import StreamingEpisodeWriter
from streaming_hdf5 import validate_episode

CAMERA_ORDER = tuple(IMAGE_SENSORS)
DEFAULT_BRIDGE_PYTHON = Path("/usr/bin/python3")
DEFAULT_FRAME_TIMEOUT_SECONDS = 0.1


class CollectorState(Enum):
    PREVIEW = "PREVIEW"
    RECORDING = "RECORDING"
    FINALIZING = "FINALIZING"
    EXIT = "EXIT"


class PreviewWindow:
    """Small OpenCV preview wrapper kept separate from collector state logic."""

    def __init__(
        self,
        adapter: ObservationAdapter | None = None,
        *,
        cv2_module: Any = cv2,
        window_name: str = "Piper ROS 2 Collector",
    ) -> None:
        self._adapter = adapter or ObservationAdapter()
        self._cv2 = cv2_module
        self._window_name = window_name

    def render(self, frame: SynchronizedFrame, state: CollectorState, prompt: str) -> str | None:
        panels = [self._decode_panel(frame.images[camera], camera) for camera in CAMERA_ORDER]
        image_row = np.hstack(panels)
        banner = self._banner(image_row.shape[1], state, prompt)
        canvas = np.vstack([image_row, banner])
        self._cv2.imshow(self._window_name, canvas)
        raw_key = self._cv2.waitKey(1) & 0xFF
        if raw_key in (ord("s"), ord("e"), ord("q")):
            return chr(raw_key)
        return None

    def close(self) -> None:
        try:
            self._cv2.destroyWindow(self._window_name)
        except Exception:
            pass

    def _decode_panel(self, jpeg: bytes, camera: str) -> np.ndarray:
        rgb = self._adapter.decode_image(jpeg)
        bgr = self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR)
        label_color = (0, 0, 255) if camera == "cam_high" else (0, 255, 0)
        self._cv2.putText(bgr, camera, (8, 22), self._cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 2)
        return bgr

    def _banner(self, width: int, state: CollectorState, prompt: str) -> np.ndarray:
        banner = np.zeros((82, width, 3), dtype=np.uint8)
        state_color = (0, 0, 255) if state is CollectorState.RECORDING else (0, 200, 0)
        self._cv2.putText(
            banner,
            f"State: {state.value}   Keys: [s] start  [e] end/save  [q] quit",
            (10, 28),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            state_color,
            2,
        )
        self._cv2.putText(
            banner,
            f"Prompt: {prompt}",
            (10, 60),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        return banner


class CollectorController:
    """Coordinate preview, recording, finalization, and shutdown."""

    def __init__(
        self,
        *,
        backend: Ros2BackendClient,
        output_dir: Path,
        prompt: str,
        jpeg_quality: int = 90,
        writer_factory: Callable[..., StreamingEpisodeWriter] = StreamingEpisodeWriter.open,
        validator: Callable[[Path], dict] = validate_episode,
        report_sink: Callable[[dict], None] | None = None,
        preview: PreviewWindow | None = None,
        render_after_save: bool = False,
        renderer: Callable[[Path], None] | None = None,
    ) -> None:
        self.backend = backend
        self.output_dir = Path(output_dir)
        self.prompt = prompt
        self.jpeg_quality = int(jpeg_quality)
        self._writer_factory = writer_factory
        self._validator = validator
        self._report_sink = report_sink or _print_validation_report
        self._preview = preview
        self._render_after_save = bool(render_after_save)
        self._renderer = renderer or _render_episode
        self.state = CollectorState.PREVIEW
        self._writer: StreamingEpisodeWriter | None = None
        self._current_path: Path | None = None
        self._active_frame_count = 0
        self._next_episode_index = 0
        self._closed = False

    @property
    def writer_open(self) -> bool:
        return self._writer is not None

    def start_episode(self) -> None:
        if self.state is CollectorState.EXIT:
            return
        if self.state is not CollectorState.PREVIEW:
            return
        self.backend.clear_buffers()
        final_path = self._next_episode_path()
        metadata = {
            "prompt": self.prompt,
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "collector": "collect_data_ros2.py",
            "source": "ros2_bridge",
        }
        self._writer = self._writer_factory(final_path, metadata=metadata, jpeg_quality=self.jpeg_quality)
        self._current_path = final_path
        self._active_frame_count = 0
        self.state = CollectorState.RECORDING
        print(f"Recording {final_path.with_suffix(final_path.suffix + '.partial')} (press e to finalize, q to abort)")

    def stop_episode(self) -> Path:
        if self.state is not CollectorState.RECORDING or self._writer is None:
            if self._current_path is None:
                return Path()
            return self._current_path
        self.state = CollectorState.FINALIZING
        writer = self._writer
        if self._active_frame_count == 0:
            return self._abort_current_episode(writer, "episode contains zero frames", remove_partial=True)
        try:
            final_path = writer.finalize()
        except ValueError as exc:
            return self._abort_current_episode(writer, str(exc), remove_partial=True)
        self._writer = None
        self._active_frame_count = 0
        report = self._validator(final_path)
        self._report_sink(report)
        if self._render_after_save:
            self._renderer(final_path)
        self._current_path = final_path
        self.state = CollectorState.PREVIEW
        print(f"Finalized {final_path} (press s for another episode, q to quit)")
        return final_path

    def _abort_current_episode(self, writer: StreamingEpisodeWriter, reason: str, *, remove_partial: bool) -> Path:
        partial_path = writer.partial_path
        try:
            writer.abort(remove_partial=remove_partial)
        finally:
            self._writer = None
            self._current_path = None
            self._active_frame_count = 0
            self.state = CollectorState.PREVIEW
        policy = "Removed" if remove_partial else "Retained"
        print(f"Failed to finalize episode: {reason}. {policy} partial {partial_path}. Press s to try another episode, q to quit.")
        return Path()

    def handle_key(self, key: str | None) -> CollectorState:
        if key is None:
            return self.state
        normalized = key.lower()
        if normalized == "s":
            self.start_episode()
        elif normalized == "e":
            self.stop_episode()
        elif normalized == "q":
            self.state = CollectorState.EXIT
            self.close()
        return self.state

    def process_frame(self, frame: SynchronizedFrame) -> str | None:
        key = self._preview.render(frame, self.state, self.prompt) if self._preview is not None else None
        if self.state is CollectorState.RECORDING and self._writer is not None:
            self._writer.append(frame)
            self._active_frame_count += 1
        if key is not None:
            self.handle_key(key)
        return key

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.state = CollectorState.EXIT
        if self._writer is not None:
            self._writer.abort(remove_partial=True)
            self._writer = None
            self._active_frame_count = 0
        if self._preview is not None:
            self._preview.close()
        self.backend.close()

    def _next_episode_path(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        index = self._next_episode_index
        while True:
            final_path = self.output_dir / f"episode_{index:06d}.hdf5"
            partial_path = final_path.with_suffix(f"{final_path.suffix}.partial")
            index += 1
            if not final_path.exists() and not partial_path.exists():
                self._next_episode_index = index
                return final_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview synchronized Piper dual ROS 2 frames and save multi-episode self-contained HDF5 files."
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for episode_XXXXXX.hdf5 outputs")
    parser.add_argument("--prompt", required=True, help="Task prompt stored in each episode metadata")
    parser.add_argument("--config", required=True, type=Path, help="ROS 2 bridge config YAML/JSON path")
    parser.add_argument(
        "--bridge-python",
        type=Path,
        default=DEFAULT_BRIDGE_PYTHON,
        help="Python interpreter used to launch the ROS 2 bridge (default: /usr/bin/python3)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality metadata recorded by the streaming writer (1-100, default: 90)",
    )
    parser.add_argument(
        "--max-sync-error-ms",
        type=float,
        default=DEFAULT_SYNC_ERROR * 1000.0,
        help="Maximum sensor timestamp mismatch in milliseconds (default: 30)",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep ROS 2 bridge in no-action-publishing mode (default: true)",
    )
    parser.add_argument(
        "--publish-actions",
        action="store_true",
        default=False,
        help="Enable validated action publishing in the bridge; unsafe with --dry-run and disabled by default",
    )
    parser.add_argument(
        "--render-after-save",
        action="store_true",
        default=False,
        help="Render a preview MP4 next to each finalized episode",
    )
    return parser


def run_collection_loop(controller: CollectorController, *, frame_timeout: float = DEFAULT_FRAME_TIMEOUT_SECONDS) -> None:
    try:
        while controller.state is not CollectorState.EXIT:
            try:
                frame = controller.backend.next_frame(timeout=frame_timeout)
            except TimeoutError:
                continue
            controller.process_frame(frame)
    except Exception:
        controller.close()
        raise
    finally:
        controller.close()


def create_controller(args: argparse.Namespace) -> CollectorController:
    if args.dry_run and args.publish_actions:
        raise ValueError("Unsafe configuration: --dry-run cannot be combined with --publish-actions")
    backend = Ros2BackendClient(
        bridge_python=args.bridge_python,
        config=args.config,
        publish_actions=args.publish_actions,
        dry_run=args.dry_run,
    )
    try:
        backend.max_sync_error = float(args.max_sync_error_ms) / 1000.0
        backend.synchronizer.max_error = backend.max_sync_error
        backend.start()
        adapter = ObservationAdapter(task=args.prompt)
        preview = PreviewWindow(adapter=adapter)
    except Exception:
        backend.close()
        raise
    return CollectorController(
        backend=backend,
        output_dir=args.output_dir,
        prompt=args.prompt,
        jpeg_quality=args.jpeg_quality,
        preview=preview,
        render_after_save=args.render_after_save,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    controller = create_controller(args)
    print("Previewing synchronized ROS 2 frames. Press s to record, e to finalize, q to quit.")
    try:
        run_collection_loop(controller)
    except KeyboardInterrupt:
        controller.close()
        print("Interrupted; closed collector cleanly.")
    return 0


def _print_validation_report(report: dict) -> None:
    print("Validation report:")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


def _render_episode(path: Path) -> None:
    output_path = path.with_suffix(".preview.mp4")
    prompt = ""
    adapter = ObservationAdapter()
    with h5py.File(path, "r") as episode:
        images = episode["observations/images"]
        if "metadata_json" in episode.attrs:
            try:
                metadata = json.loads(episode.attrs["metadata_json"])
                if isinstance(metadata, dict) and isinstance(metadata.get("prompt"), str):
                    prompt = metadata["prompt"]
            except Exception:
                prompt = ""
        frame_count = int(episode.attrs.get("frame_count", len(images[CAMERA_ORDER[0]])))
        if frame_count <= 0:
            print(f"Skipping render for zero-frame episode: {path}")
            return
        writer = None
        try:
            for index in range(frame_count):
                panels: list[np.ndarray] = []
                for camera in CAMERA_ORDER:
                    raw = images[camera][index]
                    frame_rgb = adapter.decode_image(bytes(raw))
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    label_color = (0, 0, 255) if camera == "cam_high" else (0, 255, 0)
                    cv2.putText(frame_bgr, camera, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 2)
                    panels.append(frame_bgr)
                row = np.hstack(panels)
                banner = np.zeros((72, row.shape[1], 3), dtype=np.uint8)
                cv2.putText(
                    banner,
                    f"{path.name} frame {index + 1}/{frame_count}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )
                if prompt:
                    cv2.putText(
                        banner,
                        f"Prompt: {prompt}",
                        (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                canvas = np.vstack([row, banner])
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(output_path), fourcc, 30.0, (canvas.shape[1], canvas.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open video writer for {output_path}")
                writer.write(canvas)
        finally:
            if writer is not None:
                writer.release()
    print(f"Rendered preview video to {output_path}")


if __name__ == "__main__":
    raise SystemExit(main())
