#!/usr/bin/env python3
"""根据实测推理延迟，给出 fps / action_horizon / actions_during_latency 的建议值。

关系（见 packages/openpi-client/src/openpi_client/action_chunk_broker.py）：
  推理频率 ≈ fps / action_horizon        （_cur_step 递增到 action_horizon 且上一轮推理结束才触发下一轮）
  稳定条件：action_horizon / fps > 单次推理耗时，否则会退化成"一结束就立刻下一轮"，
           期间客户端保持块内最后一个绝对目标（表现为停顿）
  actions_during_latency ≈ 单次推理耗时 × fps   （RTC 拼接点对齐；偏小会在换块处产生跳变）
  fps 必须等于训练数据帧率，否则轨迹被时间缩放（data_fps / fps 倍速）

用法（用 client 的 venv python，不需要 ROS）：
  env -u PYTHONPATH .venv/bin/python examples/piper_dual/broker_params.py
  env -u PYTHONPATH .venv/bin/python examples/piper_dual/broker_params.py --samples 8
  env -u PYTHONPATH .venv/bin/python examples/piper_dual/broker_params.py --log /tmp/run.log
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import statistics
import sys
import time

MAX_HORIZON = 50  # 服务端返回的 chunk 长度（模型 action_horizon=50）


def _dataset_fps(dataset: str, roots: list[pathlib.Path]) -> int | None:
    for root in roots:
        for candidate in (root / dataset, root / "local" / dataset):
            info = candidate / "meta" / "info.json"
            if info.exists():
                data = json.loads(info.read_text())
                fps = data.get("fps")
                print(f"[data] {info} -> fps={fps}")
                return int(fps) if fps else None
    return None


def _latency_from_log(path: str) -> list[float]:
    text = pathlib.Path(path).read_text(errors="ignore")
    values = [float(m) / 1000.0 for m in re.findall(r"infer_latency=([0-9.]+)", text)]
    print(f"[log] {path}: 解析到 {len(values)} 条 infer_latency")
    return values


def _latency_from_probe(host: str, port: int, prompt: str, samples: int) -> list[float]:
    import numpy as np
    from openpi_client import websocket_client_policy as wcp

    client = wcp.WebsocketClientPolicy(host=host, port=port)
    obs = {
        "state": np.zeros(14, dtype=np.float32),
        "images": {
            "cam_high": np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
        },
        "prompt": prompt,
    }
    latencies: list[float] = []
    for index in range(samples + 1):
        start = time.time()
        out = client.infer(obs)
        elapsed = time.time() - start
        shape = np.asarray(out["actions"]).shape
        if index == 0:
            print(f"[probe] 预热调用 {elapsed:.3f}s actions{shape}（不计入统计）")
            continue
        latencies.append(elapsed)
        print(f"[probe] 第 {index} 次: {elapsed:.3f}s actions{shape}")
    return latencies


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Suggest fps / action_horizon / actions_during_latency")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="Pass the playing cards and sunflower seeds to the guests.")
    parser.add_argument("--samples", type=int, default=5, help="探测次数（不含预热）")
    parser.add_argument("--log", default=None, help="改为从日志解析 infer_latency=（不探测）")
    parser.add_argument("--dataset", default="piper_dual_playing_cards_1_500_joint")
    parser.add_argument("--data-fps", type=int, default=30, help="数据集帧率兜底值")
    parser.add_argument(
        "--target-chunk-seconds", type=float, default=0.75,
        help="期望的块寿命（秒）：越长换块越少越平滑、但计划越陈旧；默认 0.75s",
    )
    args = parser.parse_args(argv)

    roots = [
        pathlib.Path(os.environ.get("HF_LEROBOT_HOME", "/home/agilex/lgd/lerobot_datasets")),
        pathlib.Path("/home/agilex/lgd/lerobot_datasets"),
        pathlib.Path.home() / ".cache" / "huggingface" / "lerobot",
    ]
    data_fps = _dataset_fps(args.dataset, roots) or args.data_fps
    print(f"[data] 使用数据帧率 {data_fps} fps（动作时间步 = 1/{data_fps} s）")

    try:
        latencies = _latency_from_log(args.log) if args.log else _latency_from_probe(
            args.host, args.port, args.prompt, args.samples
        )
    except Exception as exc:  # noqa: BLE001 - 现场探测失败要给人看得懂的信息
        print(f"\n[错误] 取延迟失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("  服务端没起？或改用 --log <客户端stdout文件>", file=sys.stderr)
        return 2
    if not latencies:
        print("\n[错误] 没有拿到任何延迟样本", file=sys.stderr)
        return 2

    mean_s = statistics.fmean(latencies)
    p95_s = sorted(latencies)[max(0, math.ceil(0.95 * len(latencies)) - 1)]
    print(
        f"\n[latency] n={len(latencies)} mean={mean_s * 1000:.0f} ms  p95={p95_s * 1000:.0f} ms  "
        f"min={min(latencies) * 1000:.0f} ms  max={max(latencies) * 1000:.0f} ms"
    )

    horizon_min = _clamp(math.ceil(1.25 * p95_s * data_fps), 2, MAX_HORIZON)
    horizon = _clamp(max(horizon_min, round(args.target_chunk_seconds * data_fps)), horizon_min, MAX_HORIZON)
    d = _clamp(round(mean_s * data_fps), 1, horizon - 1)
    chunk_life = horizon / data_fps
    infer_hz = data_fps / horizon

    print(f"\n=== 建议（按 p95 延迟留 25% 余量，horizon 上限 {MAX_HORIZON}） ===")
    print(f"  稳定下限 horizon ≥ {horizon_min}（{horizon_min / data_fps:.2f}s > p95 {p95_s:.2f}s）")
    print(f"  --fps {data_fps}                    # 与数据帧率一致，动作速度才不被缩放")
    print(f"  --action-horizon {horizon}                # 块寿命 {chunk_life:.2f}s > p95 延迟 {p95_s:.2f}s，余量 {chunk_life - p95_s:+.2f}s")
    print(f"  --actions-during-latency {d}        # ≈ 平均延迟 {mean_s * 1000:.0f}ms × {data_fps}fps = {mean_s * data_fps:.1f}")
    print(f"  → 推理频率 ≈ {infer_hz:.2f} Hz（每 {chunk_life:.2f}s 一次），关节目标下发 {data_fps} Hz")

    print("\n=== 其它组合对比 ===")
    print("  fps  horizon  d   下发Hz  chunk寿命  推理Hz  说明")
    for fps, hz, dd, note in [
        (20, 15, 8, "你当前默认"),
        (data_fps, 15, 8, "只把 fps 对齐数据"),
        (data_fps, horizon, d, "上面建议值"),
        (20, max(2, math.ceil(1.25 * p95_s * 20)), max(1, round(mean_s * 20)), "保持 20fps 时"),
        (data_fps, data_fps, max(1, round(mean_s * data_fps)), "1 秒块（最平顺）"),
    ]:
        life = hz / fps
        flag = "OK" if life > p95_s else "不稳(块<延迟)"
        print(f"  {fps:4d} {hz:7d}  {dd:3d}  {fps:5d}  {life:8.2f}s  {fps / hz:6.2f}  {note}｜{flag}")

    print("\n可直接粘贴（按你的常用命令改好参数）：")
    print(
        f"  python examples/piper_dual/main_dual_ros.py --host {args.host} --port {args.port} \\\n"
        f"    --bridge-python /usr/bin/python3 --ros2-config examples/piper_dual/ros2_piper_dual_local.yaml \\\n"
        f"    --control-stack local-ros --prompt \"{args.prompt}\" \\\n"
        f"    --fps {data_fps} --action-horizon {horizon} --actions-during-latency {d} \\\n"
        "    --no-dry-run --publish-actions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
