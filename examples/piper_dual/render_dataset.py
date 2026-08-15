#!/usr/bin/env python3
"""Render self-contained Piper dual HDF5 episodes into MP4 views and quality reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

from frame_synchronizer import FRAME_DIMENSION
from frame_synchronizer import IMAGE_SENSORS
from frame_synchronizer import REQUIRED_SENSORS
from streaming_hdf5 import SCHEMA_VERSION
from streaming_hdf5 import validate_episode

CAMERA_ORDER = IMAGE_SENSORS
QUALITY_FILENAME = "quality.json"
STATE_ACTION_PLOT_FILENAME = "state_action.png"
COMBINED_VIDEO_FILENAME = "views_3x1.mp4"
BANNER_HEIGHT = 88


def render_episode(path: Path, output_dir: Path, fps: int = 30, make_plots: bool = True) -> dict[str, Any]:
    """Render a self-contained Piper dual episode into four MP4s and a quality report."""

    episode_path = Path(path)
    fps = int(fps)
    if fps <= 0:
        raise ValueError("fps must be positive")

    report = _make_quality_report(episode_path, fps)
    if report["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"legacy schema; use existing visualizer (expected {SCHEMA_VERSION!r}, found {report['schema_version']!r})"
        )
    if report["frame_count"] <= 0:
        raise ValueError("episode contains zero frames")

    render_dir = Path(output_dir)
    render_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(episode_path, "r") as episode:
        metadata = _load_metadata_json(episode.attrs.get("metadata_json"))
        task = _extract_task(metadata, episode)
        timestamps = np.asarray(episode["/observations/timestamp"], dtype=np.float32)
        states = np.asarray(episode["/observations/state"], dtype=np.float32)
        actions = np.asarray(episode["/action"], dtype=np.float32)
        sync_error = _load_sync_error(episode)
        images_group = episode["/observations/images"]

        camera_sizes = _discover_camera_sizes(images_group, report["frame_count"])
        target_height = min(height for _width, height in camera_sizes.values())
        combined_width = sum(_resize_width_for_height(size, target_height) for size in camera_sizes.values())
        combined_size = (combined_width, target_height + BANNER_HEIGHT)

        writers = {
            camera: _open_video_writer(render_dir / f"{camera}.mp4", camera_sizes[camera], fps)
            for camera in CAMERA_ORDER
        }
        writers[COMBINED_VIDEO_FILENAME] = _open_video_writer(render_dir / COMBINED_VIDEO_FILENAME, combined_size, fps)

        try:
            for index in range(report["frame_count"]):
                per_camera_frames: dict[str, np.ndarray] = {}
                for camera in CAMERA_ORDER:
                    decoded = _decode_jpeg(images_group[camera][index])
                    if decoded is None:
                        decoded = _placeholder_frame(camera_sizes[camera], camera, index, report["frame_count"])
                    resized = _ensure_size(decoded, camera_sizes[camera])
                    per_camera_frames[camera] = resized
                    writers[camera].write(_rgb_to_bgr(resized))

                combined = _compose_views_row(
                    per_camera_frames,
                    target_height=target_height,
                    episode_name=episode_path.name,
                    frame_index=index,
                    frame_count=report["frame_count"],
                    timestamp=float(timestamps[index]),
                    task=task,
                )
                writers[COMBINED_VIDEO_FILENAME].write(_rgb_to_bgr(combined))
        finally:
            for writer in writers.values():
                writer.release()

    report["state_action_plot"] = _render_state_action_plot(
        render_dir / STATE_ACTION_PLOT_FILENAME,
        timestamps=timestamps,
        states=states,
        actions=actions,
        sync_error=sync_error,
        task=task,
        make_plots=make_plots,
    )
    report["video_paths"] = {name: str(render_dir / name) for name in list(CAMERA_ORDER) + [COMBINED_VIDEO_FILENAME]}
    report["quality_path"] = str(render_dir / QUALITY_FILENAME)
    _write_quality_report(render_dir / QUALITY_FILENAME, report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a self-contained Piper dual HDF5 episode")
    parser.add_argument("--input", required=True, type=Path, help="Input self-contained HDF5 episode")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for rendered MP4s and quality.json")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip state_action.png while still writing videos and quality.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        report = render_episode(args.input, args.output_dir, fps=args.fps, make_plots=not args.no_plots)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def _make_quality_report(path: Path, fps: int) -> dict[str, Any]:
    validation = validate_episode(path)
    return {
        "path": str(path),
        "fps": int(fps),
        "schema_version": validation["schema_version"],
        "frame_count": int(validation["frame_count"]),
        "image_decode_counts": {camera: int(count) for camera, count in validation["image_decode_counts"].items()},
        "errors": list(validation["errors"]),
    }


def _load_metadata_json(metadata_json: Any) -> dict[str, Any]:
    if metadata_json is None:
        return {}
    if isinstance(metadata_json, bytes):
        metadata_json = metadata_json.decode("utf-8", errors="replace")
    if not isinstance(metadata_json, str):
        return {}
    try:
        parsed = json.loads(metadata_json)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_task(metadata: dict[str, Any], episode: h5py.File) -> str:
    for key in ("prompt", "task"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    if "task" in episode:
        dataset = episode["task"]
        if len(dataset) > 0:
            first = dataset[0]
            if isinstance(first, bytes):
                return first.decode("utf-8", errors="replace")
            if isinstance(first, np.bytes_):
                return bytes(first).decode("utf-8", errors="replace")
            return str(first)
    return ""


def _load_sync_error(episode: h5py.File) -> np.ndarray:
    return np.stack([np.asarray(episode[f"/observations/sync_error/{sensor}"], dtype=np.float32) for sensor in REQUIRED_SENSORS])


def _discover_camera_sizes(images_group: h5py.Group, frame_count: int) -> dict[str, tuple[int, int]]:
    sizes: dict[str, tuple[int, int] | None] = {camera: None for camera in CAMERA_ORDER}
    for camera in CAMERA_ORDER:
        dataset = images_group[camera]
        for index in range(frame_count):
            decoded = _decode_jpeg(dataset[index])
            if decoded is not None:
                sizes[camera] = (int(decoded.shape[1]), int(decoded.shape[0]))
                break
    fallback = next((size for size in sizes.values() if size is not None), None)
    if fallback is None:
        raise ValueError("episode contains no decodable JPEG frames")
    return {camera: (sizes[camera] or fallback) for camera in CAMERA_ORDER}


def _render_state_action_plot(
    output_path: Path,
    *,
    timestamps: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    sync_error: np.ndarray,
    task: str,
    make_plots: bool,
) -> dict[str, Any]:
    if not make_plots:
        return {"status": "skipped", "reason": "plots disabled via --no-plots"}

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment-dependent
        return {"status": "skipped", "reason": f"matplotlib unavailable: {exc}"}

    try:
        fig, (ax_state, ax_action) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        _plot_joint_panel(ax_state, timestamps, states, "State")
        _plot_joint_panel(ax_action, timestamps, actions, "Action")
        title = "Piper dual episode state/action summary"
        if task:
            title = f"{title} — {task}"
        fig.suptitle(title, fontsize=16, fontweight="bold")
        fig.text(0.5, 0.01, _format_sync_error_summary(sync_error), ha="center", va="bottom", fontsize=8)
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting failure is environment dependent
        try:
            plt.close("all")
        except Exception:
            pass
        return {"status": "skipped", "reason": f"state_action plot failed: {exc}"}

    return {"status": "created", "path": str(output_path)}


def _plot_joint_panel(ax: Any, timestamps: np.ndarray, values: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    left_labels = [f"left {index + 1}" for index in range(FRAME_DIMENSION // 2)]
    right_labels = [f"right {index + 1}" for index in range(FRAME_DIMENSION // 2)]
    left_colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, FRAME_DIMENSION // 2))
    right_colors = plt.get_cmap("Set2")(np.linspace(0.0, 1.0, FRAME_DIMENSION // 2))

    for index in range(FRAME_DIMENSION // 2):
        ax.plot(timestamps, values[:, index], label=left_labels[index], color=left_colors[index], linewidth=1.2)
    for index in range(FRAME_DIMENSION // 2, FRAME_DIMENSION):
        ax.plot(
            timestamps,
            values[:, index],
            label=right_labels[index - FRAME_DIMENSION // 2],
            color=right_colors[index - FRAME_DIMENSION // 2],
            linewidth=1.2,
            linestyle="--",
        )

    ax.set_title(f"{title}: left/right 7D blocks", fontsize=13)
    ax.set_ylabel("Value", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=4, fontsize=7, frameon=False)


def _format_sync_error_summary(sync_error: np.ndarray) -> str:
    rows = []
    for index, sensor in enumerate(REQUIRED_SENSORS):
        values = np.asarray(sync_error[index], dtype=np.float32)
        rows.append(f"{sensor}: mean={float(np.mean(np.abs(values))):.4f}s max={float(np.max(np.abs(values))):.4f}s")
    return "Sync error summary — " + "; ".join(rows)


def _open_video_writer(path: Path, size: tuple[int, int], fps: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), size)
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")
    return writer


def _resize_width_for_height(size: tuple[int, int], target_height: int) -> int:
    width, height = size
    if height == target_height:
        return width
    return max(1, int(round(width * (target_height / float(height)))))


def _ensure_size(frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    target_width, target_height = target_size
    if frame.shape[1] == target_width and frame.shape[0] == target_height:
        return frame
    interpolation = cv2.INTER_AREA if frame.shape[0] > target_height else cv2.INTER_LINEAR
    return cv2.resize(frame, (target_width, target_height), interpolation=interpolation)


def _rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def _compose_views_row(
    camera_frames: dict[str, np.ndarray],
    *,
    target_height: int,
    episode_name: str,
    frame_index: int,
    frame_count: int,
    timestamp: float,
    task: str,
) -> np.ndarray:
    panels = []
    for camera in CAMERA_ORDER:
        frame = camera_frames[camera]
        resized = _resize_to_height(frame, target_height)
        panels.append(resized)

    row = np.hstack(panels)
    banner = np.zeros((BANNER_HEIGHT, row.shape[1], 3), dtype=np.uint8)
    _draw_banner_text(
        banner,
        [
            f"Episode: {episode_name}",
            "Views: cam_high | cam_left_wrist | cam_right_wrist",
            f"Frame: {frame_index + 1}/{frame_count}  Timestamp: {timestamp:.3f}s  Task: {task if task else '(none)'}",
        ],
    )
    return np.vstack([row, banner])


def _resize_to_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    if frame.shape[0] == target_height:
        return frame
    target_width = max(1, int(round(frame.shape[1] * (target_height / float(frame.shape[0])))))
    interpolation = cv2.INTER_AREA if frame.shape[0] > target_height else cv2.INTER_LINEAR
    return cv2.resize(frame, (target_width, target_height), interpolation=interpolation)


def _draw_banner_text(banner: np.ndarray, lines: list[str]) -> None:
    y = 28
    for line in lines:
        cv2.putText(banner, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26


def _annotate_frame(frame: np.ndarray, label: str, color: tuple[int, int, int]) -> None:
    cv2.putText(frame, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _placeholder_frame(size: tuple[int, int], camera: str, index: int, frame_count: int) -> np.ndarray:
    width, height = size
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    _draw_banner_text(frame, [camera, f"decode failed frame {index + 1}/{frame_count}"])
    return frame


def _decode_jpeg(value: Any) -> np.ndarray | None:
    try:
        if isinstance(value, np.ndarray):
            buffer = value.astype(np.uint8, copy=False).reshape(-1)
        else:
            buffer = np.frombuffer(value, dtype=np.uint8)
    except TypeError:
        try:
            buffer = np.frombuffer(bytes(value), dtype=np.uint8)
        except Exception:
            return None
    except Exception:
        return None
    if buffer.size == 0:
        return None
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded is None or decoded.size == 0:
        return None
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def _write_quality_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
