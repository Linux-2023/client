# 当前默认版本说明

RTC 入口现默认使用 `--topp-rtc-backend committed`：固定前缀、后台求解、短窗口。
当前行为、参数和验证结果请先看 [TOPP_RTC_COMMITTED.md](TOPP_RTC_COMMITTED.md)。
以下为旧 `--topp-rtc-backend legacy` 的实现记录，不描述当前默认调度。

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
  换算计划时间，再乘模型频率向下取整得到过期索引，裁掉该前缀，默认滚动模式使用剩余完整预测块规划；关闭滚动模式时只取最多 action_horizon 点。
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

依赖沿用已安装的 toppra 0.6.3；28 项测试通过，不包含实机测试。


## 滚动接续（默认开启）

2026-09-30 增加，修改前的 RTC 入口、policy 和说明保存在
`backups/topp_rtc_before_rolling/`，原非 RTC TOPP 文件未修改。

- `--topp-rolling` 默认开启；`--no-topp-rolling` 恢复本次修改前的短块策略，便于对比。
- 规划使用裁掉过期前缀后的完整预测（最多50点），而不是只规划22点。
  action-horizon 仍用于 RTC 的 s，并限制最长多久请求下一次推理，不再是轨迹的截断长度。
- 在400个采样点上，取最后一个达到归一化峰值速度98%的时刻，作为末尾减速的近似参考。
  这只是保守启发式，不是多关节轨迹的精确减速起点；复杂路径可能仍有内部停顿。
- 提前量取原 prefetch 与（上次推理耗时 + 上次规划耗时 + 规划余量 + 接续缓冲）的较大值。
  最早下一输出周期请求，最晚 action-horizon / topp-path-fps 秒请求。
- `--topp-commit-seconds` 默认0.1秒：收到新结果后保留旧轨迹的精确缓冲段，
  从缓冲末端 q/dq 开始规划新尾段。对应的模型前缀也同步跳过，避免重复。
  缓冲按模型步长向上取整，且受旧计划剩余时间和新块长度约束。
- `--topp-planning-margin` 默认0.05秒，用于提前请求的余量。
- 不可行时不强接，保持旧轨迹并在0.1秒计划时间后重试；计划仍以零速结束。
- 日志新增 schedule：request_at_s、brake_estimate_s、lead_s；adopted 包含 rolling 标识。

这次没有将求解移入后台线程：规划期间输出循环仍会阻塞。完整路径的计算成本可能更高。
也不保证反向动作、推理迟到、预测不相容时不停顿。更长预测尾部会参与路径规划，
可能影响动作时序和夹爪事件；需要与原策略做实际任务对比，不能由离线测试认定任务效果改善。

离线新增测试覆盖完整预测终点、提前请求、精确保留缓冲、非零接续速度、速度/加速度数值检验、
推理一直不返回时的终点保持。未启动实机动作或真实推理服务。

## 加速度检查失败的自适应重算

RTC 入口现使用独立 `topp_adaptive_planner.py`，非 RTC Planner 不变。
检查失败时按基础网格点数的1、4、16倍最多求解三次，保留相同路径、速度/加速度上限及起始速度。
检查覆盖每个分段的内部及边界两侧，仍是数值检验而非连续域证明。
成功时记录 `refined solution accepted`；失败原因包含关节/夹爪、实际值、上限、倍数和时间。
全部失败后仍保留旧停止轨迹，并将最近一次失败输入覆盖保存到
`--out-dir/topp_diagnostics/last_rejected_plan.npz`，用于离线复现。
加密会增加同步规划耗时，不能据此保证消除发送卡顿。未进行实机验证。
12组模拟转向路径在旧检查下均拒绝，新实现均第二次通过；加入回归测试后共32项通过。

## 接续失败后的恢复

修正了轨迹到终点后 `elapsed + 0.1` 重试时间不可达的问题：普通拒绝改用输出 tick 计时重试。
非零速度接续不可行时记录 `recover_at_rest`，继续旧轨迹直到其有效零速终点，再从新观测请求新块。
已经到终点的计划视为静止请求，等待期间不会按移动轨迹裁掉返回块的前缀。
这解决永久不再请求的问题，但不保证消除不可行接续本身造成的停顿。
本地保存的真实失败输入可复现非零速度不可行；把同一输入的起始速度设零做离线对照后可解，
此对照不是实机恢复验证，实机采用的是走完旧轨迹后重新观测，不会在运动中直接清零计划速度。
新增三个调度回归测试；共35项离线测试通过。
