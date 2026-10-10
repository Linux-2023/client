# 固定前缀的 RTC + TOPP 实验版

`main_dual_topp_rtc_ros.py` 默认 `--topp-rtc-backend committed`。
纯 RTC `main_dual_ros.py` 和非 RTC TOPP `main_dual_topp_ros.py` 未修改。
旧 RTC policy 保留；修改前入口/说明备份在 `backups/topp_rtc_before_committed/`。

```bash
python examples/piper_dual/main_dual_topp_rtc_ros.py \
  --host 127.0.0.1 --port 8000 \
  --bridge-python /usr/bin/python3 \
  --ros2-config examples/piper_dual/ros2_piper_dual_local.yaml \
  --control-stack local-ros \
  --prompt "Pass the playing cards and sunflower seeds to the guests." \
  --max-action-delta 20 --no-dry-run --publish-actions
```

上面的发布开关会允许实机动作。本次开发仅离线测试，没有连接机器人或真实推理服务。

## 与上一版不同

1. 请求发出时确定未来接续 tick，冻结此前旧轨迹；RTC 前缀就是这段计划的固定模型频率采样。
2. 同一个后台线程完成推理、TOPP 求解、加密检查；发送循环只评估旧计划、检查结果、切换计划。
3. 结果提前完成也必须等到约定 tick；晚到则丢弃，继续旧停止轨迹，不临时移动接缝。
4. 接缝位于模型频率与输出频率的共同时间网格，q/dq 是旧计划在该时刻的值。
   若频率组合使共同网格超过预测窗口，启动时报错；15/30/60 Hz 组合可用。
5. 模型返回块的 `[0,d)` 属于已承诺前缀，索引 d 的位置以旧计划为准；
   只对 `[d+1,d+1+action_horizon)` 重新规划，最多22点，不再默认规划整块50点。
6. 本地服务端使用 `H-s` 表示软前缀末端。这里 `s=H-(d+soft_prefix_steps)`，
   默认软过渡3点，避免旧版固定 s=22 导致28点旧前缀占满短规划窗口。
   此参数契约已核对本地 pi0.py / transforms.py；远端服务必须支持同一契约。

`d` 覆盖预计推理+规划+余量，从最近10次耗时取最大值；初始推理预算为 actions-during-latency / 模型频率。
规划预算至少 `--topp-plan-budget`（默认0.2秒），余量 `--topp-planning-margin` 默认0.05秒。
`--topp-commit-seconds` 默认0.1秒，在新模式中是总承诺窗口的最短值，不是返回后追加缓冲。
窗口按共同网格向上取整，并受模型窗口限制。预算被截断时日志 `budget_saturated=true`，
网络或求解持续超过可用窗口时仍可能无法接续。
初始15 Hz、d初值5时，通常需要约0.6秒承诺窗口；这是为异步接续预留的时间，不是模型推理变快。

## 保留的限制

- 每次调用仍只推进一个输出 tick，其他环节阻塞时不会跳帧追赶。这里承诺的是计划时间，
  不保证墙钟硬实时；后台线程仍可能有 GIL/CPU 竞争，观测、记录、ROS也可能拖慢循环。
- 只固定前缀；未承诺尾部仍被TOPP重定时。没有证明与原RTC完全等价，也没有修改模型训练。
- 无法带速接入新路径时继续旧计划减速，到终点后用新观测请求，不在运动中清零速度。
- 每个计划仍有零速终点；若新计划迟到、路径冲突或任务本来需要反向，仍可能停顿。
- 仍是三次样条路径，可能偏离原折线；没有位置/碰撞/jerk约束，没有实际跟踪误差闭环恢复。
- 当前用户参数沿用：关节v=4、a=2，夹爪v=0.05、a=0.2，路径频率15；输出fps继承原入口。
  这些不是官方限值，也不是经实机验证的推荐值。
- `--topp-rolling`/`--no-topp-rolling` 只作用于 legacy；新模式固定使用短窗口。

## 对比与日志

保留上次实现：同一命令加 `--topp-rtc-backend legacy`。纯RTC用 `main_dual_ros.py --use-rtc --use-async`。
比较时保持fps、速度/加速度、模型、任务和初始姿态一致，重点记录成功率、停顿次数和跟踪误差。

新日志 `--out-dir/topp_rtc_committed_metrics.jsonl`：
- request：rtc_s、rtc_d、target_tick、committed_s、预算是否截断。
- worker_done：后台推理、规划耗时和完成tick。
- adopted：实际接续tick、规划点数、起始速度、轨迹时长。
- rejected：失败/错过接续时刻，是否等待旧计划停止后恢复。

策略返回 `_topp_trace` 另含 policy_ms 与 call_interval_ms，供离线基准读取；
现有 ROS 记录器不保证持久化这两个新字段。

## 离线验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  examples/piper_dual/tests/test_topp_committed_rtc_policy.py \
  examples/piper_dual/tests/test_topp_rtc_policy.py \
  examples/piper_dual/tests/test_topp_adaptive_planner.py \
  examples/piper_dual/tests/test_topp_policy.py \
  examples/piper_dual/tests/test_main_dual_topp_ros.py

.venv/bin/python examples/piper_dual/benchmark_topp_rtc_dispatch.py \
  --input data/piper_dual/videos/topp_diagnostics/last_rejected_plan.npz
```

46项测试，包括不同频率的固定前缀、真实TOPP非零速度接续、后台阻塞时持续输出、晚到丢弃、
停下后的重试、网络错误和重置。没有真实模型闭环或实机测试。
本机一次6秒回放（模拟250ms推理，重复保存的动作块，不是真实RTC模型）：90次输出，
主线程策略调用P95=0.45ms、最大0.78ms，调用间隔P95=67.31ms。
3次采用、3次非零速度接续拒绝。该测试支持“求解移出发送循环”的结论，
不支持“消除所有停顿”或“任务成功率提高”的结论。报告：`data/topp_rtc_dispatch_benchmark.json`。
