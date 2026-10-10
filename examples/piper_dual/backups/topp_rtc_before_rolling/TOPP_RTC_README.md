# RTC + TOPP-RA 实验入口

`main_dual_topp_rtc_ros.py` 默认 `use_rtc=True`、异步推理。
保留现有 `main_dual_topp_ros.py`、`topp_policy.py`、原入口及 RTC broker，不修改硬件驱动。
新增 `topp_rtc_policy.py` 使用现有 Planner，独立实现 RTC 调度。

```bash
python examples/piper_dual/main_dual_topp_rtc_ros.py \
  --host 127.0.0.1 --port 8000 \
  --bridge-python /usr/bin/python3 \
  --ros2-config examples/piper_dual/ros2_piper_dual_local.yaml \
  --control-stack local-ros \
  --prompt "Pass the playing cards and sunflower seeds to the guests." \
  --max-action-delta 20 --no-dry-run --publish-actions
```

默认仍 dry-run；上面命令明确开启实际发布。需已有机械臂驱动、相机节点、支持本地 RTC 接口的推理服务器。
未进行实机或真实模型服务验证。只对本地源码中的 RTC 契约做了离线核对与模拟集成测试；
客户端不能自动证明远端服务启用了 RTC，也不能证明其模型/变换与本地一致。

## 时间与 RTC 协议

- 固定模型/路径频率 `--topp-path-fps`，与输出 `--fps` 分开。
- 发请求时，按 `1/topp-path-fps` 重新采样当前 TOPP 计划的未来位置，发送
  `rtc_obs={prev_action, prev_state, s, d}`。位置为绝对 SI 单位，由服务端原变换转换到 delta/归一化空间。
- `prev_action` 为 `(H,14)`；计划结束后重复终点，初始无计划时重复实测起始位置。
  `prev_state` 为本次观测状态；模型起步时仍假设机械臂静止。
- `s=action_horizon`。本地模型以 `H-s` 作为软前缀长度，`d` 作为延迟前缀长度。
  首次 d 来自 `--actions-during-latency`，后续取上次推理耗时乘模型频率并向上取整，限制到 `H-s-1`。
  这是延迟估计，不是保证；长延迟可能超过前缀覆盖范围。
- 推理期间继续输出旧 TOPP 计划。收到新块后，以等待期间的输出 tick 数 / 输出 fps
  换算计划时间，再乘模型频率向下取整得到过期索引，裁掉该前缀，然后取最多 action_horizon 点重新规划。
  第一次请求期间没有旧计划运动，不裁剪初始块。边界量化误差小于一个模型时间步。
- 只按实际调用推进的逻辑时间裁剪，不按墙钟跳过未执行动作；循环阻塞仍会拉长真实执行时间。
- 新 TOPP 从旧计划当前 q/dq 接续。不可行、格式错误或整个返回块过期时，保留旧减速轨迹；
  服务器异常退出。没有把未规划目标直接发往机械臂。
- 不发送原 broker 的顶层 prev_action/GPR 条件。这里 RTC 使用的是重采样计划，而非旧原始模型块。

## 默认值与边界

新增时沿用当前版本：fps=30，topp-path-fps=30，关节 v=4 rad/s、a=1 rad/s²，
夹爪 v=0.05 m/s、a=0.2 m/s²，action-horizon=22，预取剩余时间=0.4s，初始 d=5。
这些是用户当前代码中的规划默认值，不是官方硬件额定值或实机验证结果。
`--rtc-model-horizon` 默认50，必须与服务端模型输出长度一致；不一致的块会被拒绝并记录。
`--no-use-rtc` 和 `--no-use-async` 在这个入口被拒绝；非 RTC 请使用保留的旧 TOPP 入口。

RTC 在固定模型时间网格上约束生成，TOPP 之后仍会改变动作持续时间，因此不是原始 RTC 的严格时序等价实现。
没有保证整个延迟前缀被逐点原样执行，没有 jerk/碰撞约束，没有实际速度闭环或跟踪误差恢复。
轨迹位置/速度衔接是计划层面的连续，不代表实际机械臂响应连续。
提高输出 fps 不会改变给定 q/dq 和块的路径，但仍影响闭环观测和接续时机。

## 日志与验证

`--out-dir/topp_rtc_metrics.jsonl` 包含 request 的 s/d、两种频率，以及 adopted 的裁剪步数、
推理/规划耗时、轨迹时长，或 rejected 原因。可与旧 TOPP 分开比较。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  examples/piper_dual/tests/test_topp_rtc_policy.py \
  examples/piper_dual/tests/test_topp_policy.py \
  examples/piper_dual/tests/test_main_dual_topp_ros.py
```

依赖沿用已安装的 toppra 0.6.3；25 项测试通过，不包含实机测试。
