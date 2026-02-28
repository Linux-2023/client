"""Capture and save one sharp image from three cameras.

Usage examples:
  python examples/piper_dual/save_three_cameras.py \
    --camera-ids 0 1 2 \
    --output-dir outputs/three_cams

  # Mix USB and RealSense
  python examples/piper_dual/save_three_cameras.py \
    --camera-types usb realsense usb \
    --camera-ids 0 128422271347 2 \
    --output-dir outputs/three_cams
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

from cameras import RealSenseCamera, USBCamera


def _focus_score(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _pick_sharpest_frame(
    read_fn,
    attempts: int,
    warmup_frames: int,
    delay_s: float,
) -> np.ndarray:
    # Warm up to stabilize auto-exposure/auto-focus
    for _ in range(warmup_frames):
        _ = read_fn()
        if delay_s > 0:
            time.sleep(delay_s)

    best_frame = None
    best_score = -1.0
    for _ in range(attempts):
        frame = read_fn()
        if frame is None:
            if delay_s > 0:
                time.sleep(delay_s)
            continue
        score = _focus_score(frame)
        if score > best_score:
            best_score = score
            best_frame = frame
        if delay_s > 0:
            time.sleep(delay_s)

    if best_frame is None:
        raise RuntimeError("Failed to capture a valid frame from camera")
    return best_frame


def _make_camera(camera_type: str, camera_id: int, width: int, height: int, fps: int):
    if camera_type == "usb":
        return USBCamera(camera_id=camera_id, width=width, height=height, fps=fps)
    if camera_type == "realsense":
        return RealSenseCamera(width=width, height=height, fps=fps, serial_number=str(camera_id))
    raise ValueError(f"Unsupported camera type: {camera_type}")


def _save_frame(image_bgr: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # PNG is lossless for clarity
    ok = cv2.imwrite(str(output_path), image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not ok:
        raise RuntimeError(f"Failed to write image to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Save one sharp frame from three cameras.")
    parser.add_argument(
        "--camera-ids",
        type=int,
        nargs=3,
        default=[0, 1, 2],
        help="Three camera IDs (USB index or RealSense serial number).",
    )
    parser.add_argument(
        "--camera-types",
        type=str,
        nargs=3,
        default=["usb", "usb", "usb"],
        help="Three camera types: usb or realsense.",
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs=3,
        default=["cam_1", "cam_2", "cam_3"],
        help="Names for the output files.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Capture width.")
    parser.add_argument("--height", type=int, default=720, help="Capture height.")
    parser.add_argument("--fps", type=int, default=30, help="Capture FPS.")
    parser.add_argument(
        "--attempts",
        type=int,
        default=10,
        help="Number of frames to evaluate for sharpness.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=10,
        help="Warmup frames to stabilize exposure/focus.",
    )
    parser.add_argument(
        "--delay-s",
        type=float,
        default=0.03,
        help="Delay between frames (seconds).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/three_cams",
        help="Output directory.",
    )

    args = parser.parse_args()

    if len(args.camera_ids) != 3 or len(args.camera_types) != 3 or len(args.names) != 3:
        raise ValueError("camera-ids, camera-types, and names must each have exactly 3 items")

    output_dir = Path(args.output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    cameras = []
    try:
        for cam_type, cam_id in zip(args.camera_types, args.camera_ids, strict=True):
            cam = _make_camera(cam_type, cam_id, args.width, args.height, args.fps)
            cam.start()
            cameras.append(cam)

        for cam, name in zip(cameras, args.names, strict=True):
            frame = _pick_sharpest_frame(
                cam.read,
                attempts=args.attempts,
                warmup_frames=args.warmup_frames,
                delay_s=args.delay_s,
            )
            output_path = output_dir / f"{name}_{timestamp}.png"
            _save_frame(frame, output_path)
            print(f"Saved: {output_path}")
    finally:
        for cam in cameras:
            try:
                cam.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
