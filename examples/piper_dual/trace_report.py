#!/usr/bin/python3
"""关节 trace 报告：有效步频 / 跟随误差 / 滞后时间 / 换块跳变。

读取 examples/piper_dual 运行产生的 data/piper_dual/videos/joint_traces/episode_*.jsonl
（每行 = {timestamp, step, server_action[14], actual_joint[14], tracking_error[14]}）。

用法:
  python trace_report.py                          # 最近 5 个 episode
  python trace_report.py --last 10
  python trace_report.py --files a.jsonl b.jsonl   # 指定文件对比
  python trace_report.py --dir <joint_traces 目录>
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def load(path: pathlib.Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} 为空")
    ts = np.array([r["timestamp"] for r in rows])
    act = np.array([r["server_action"] for r in rows], dtype=float)
    err = np.array([r["tracking_error"] for r in rows], dtype=float)
    return ts, act, err


def analyze(path: pathlib.Path) -> dict:
    ts, act, err = load(path)
    dt = np.diff(ts)
    median_dt = float(np.median(dt)) if len(dt) else float("nan")
    fps = 1.0 / median_dt if median_dt else float("nan")
    d_act = np.linalg.norm(np.diff(act, axis=0), axis=1) if len(act) > 1 else np.array([0.0])
    err_norm = np.linalg.norm(err, axis=1)
    speed = d_act * fps  # 指令速度近似（rad/s）
    moving = speed > 0.05
    # 滞后时间 ≈ |误差| / |指令速度|（只统计在动的步）
    lag_ms = np.median(err_norm[:-1][moving] / np.maximum(speed[moving], 1e-6)) * 1000 if moving.any() else float("nan")
    return {
        "file": path.name,
        "steps": len(ts),
        "duration": float(ts[-1] - ts[0]),
        "fps": fps,
        "max_dt_ms": float(dt.max() * 1000) if len(dt) else 0.0,
        "stalls": int((dt > 1.5 * median_dt).sum()) if len(dt) else 0,
        "step_len": float(np.median(d_act)),
        "err_median": float(np.median(err_norm)),
        "err_p95": float(np.percentile(err_norm, 95)),
        "err_max": float(err_norm.max()),
        "lag_ms": float(lag_ms),
        "rms_per_joint": np.sqrt((err**2).mean(axis=0)),
        "top": [
            (int(i + 1), float(err_norm[i]), float(d_act[i]))
            for i in np.argsort(err_norm[:-1])[-3:][::-1]
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report tracking quality from joint traces")
    parser.add_argument("--dir", default="data/piper_dual/videos/joint_traces")
    parser.add_argument("--last", type=int, default=5, help="按 mtime 取最近 N 个 episode")
    parser.add_argument("--files", nargs="*", default=None, help="显式指定 trace 文件")
    args = parser.parse_args(argv)

    if args.files:
        paths = [pathlib.Path(p) for p in args.files]
    else:
        directory = pathlib.Path(args.dir)
        candidates = sorted(directory.glob("episode_[0-9]*.jsonl"), key=lambda p: p.stat().st_mtime)
        paths = candidates[-args.last :]
    if not paths:
        print(f"没有找到 trace 文件（默认目录 {args.dir}）")
        return 2

    names = [f"L{i}" for i in range(1, 7)] + ["Lgrip"] + [f"R{i}" for i in range(1, 7)] + ["Rgrip"]
    results = [analyze(p) for p in paths]

    print(f"{'episode':>22} {'fps':>6} {'步数':>6} {'时长s':>7} {'stall':>6} {'最大dt':>7} "
          f"{'步长':>7} {'err中位':>8} {'err p95':>8} {'滞后ms':>7}")
    for r in results:
        print(
            f"{r['file']:>22} {r['fps']:6.2f} {r['steps']:6d} {r['duration']:7.1f} {r['stalls']:6d} "
            f"{r['max_dt_ms']:6.0f}m {r['step_len']:7.4f} {r['err_median']:8.4f} {r['err_p95']:8.4f} {r['lag_ms']:7.0f}"
        )

    latest = results[-1]
    print(f"\n最近一次 {latest['file']} 各关节跟随误差 RMS（rad / grip m）：")
    for name, value in zip(names, latest["rms_per_joint"]):
        bar = "█" * min(40, int(value * 200))
        print(f"  {name:6s} {value:6.4f} {bar}")

    print("\n各 episode 误差最大的 3 步（步号, |误差|, 该步指令变化 ||Δa||）—— 误差大且 Δa 也大 = 换块/阶跃，"
          "误差大但 Δa 小 = 持续滞后（控制器跟不上）：")
    for r in results:
        top = "  ".join(f"#{s}(|e|={e:.2f},Δa={d:.2f})" for s, e, d in r["top"])
        print(f"  {r['file']:>22}: {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
