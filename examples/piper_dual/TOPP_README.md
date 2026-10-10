# 实验版 TOPP-RA joint 部署

新增入口 `main_dual_topp_ros.py`；原入口、原 broker、原 ROS 环境和硬件驱动均不改。
这是上层轨迹时间参数化，不切换硬件 J/JS 模式，不发送初始化回零。
默认仍不发布动作。建议先去掉下例的 `--no-dry-run --publish-actions` 检查规划日志。

```bash
python examples/piper_dual/main_dual_topp_ros.py \
  --host 127.0.0.1 --port 8000 \
  --bridge-python /usr/bin/python3 \
  --ros2-config examples/piper_dual/ros2_piper_dual_local.yaml \
  --control-stack local-ros \
  --prompt "Pass the playing cards and sunflower seeds to the guests." \
  --max-action-delta 20 --no-dry-run --publish-actions
```

## 算法与边界

- 使用真实 TOPP-RA 0.6.3（seidel 求解器），非简单低通滤波或插值冒充 TOPP。
- 对 14 维联合路径计时：左六关节 rad + 左夹爪 m + 右六关节 rad + 右夹爪 m。
  双臂和夹爪共用时间参数；夹爪另有 SI 速度/加速度约束。
- 首帧以实时 observation.state 为起点，计划初速为零。启动时应处于静止状态；没有估计实际初始速度。
- 默认后台异步推理，取每块前 action-horizon 个点（默认 22），自然三次样条构造路径。
- 新块到达时，以旧计划的当前位置/速度作为新路径起始位置/切线，再指定相应路径速度，
  保持数学轨迹的 C1 连续。不是把每块独立做零速起止后拼接。
- 新路径可能不可行；此时拒绝新块、继续旧计划减速到零，再用新观测请求下一块。
  不发送原始未规划目标，不强行清零速度。服务器错误则通过原 runtime 退出处理。
- 每份计划尾端速度为零，推理迟到时可以走完旧轨迹并保持。无法保证网络卡顿时仍连续运动。
- 标准 TOPP-RA 不保证加速度跨块连续或 jerk 有界；样条可能偏离折线、过冲。
  没有碰撞、动力学/力矩或位置限位约束，也未处理实际跟踪误差/扰动恢复。
- 内部密集采样核对速度/加速度，属于数值检验，不是连续域严格证明。
- TOPP 可缩短或拉长原动作时长，会改变模型预测的时序和夹爪事件时机；闭环效果尚未实机验证。

## 参数

| 参数 | 默认值 | 单位/含义 |
|---|---:|---|
| `--topp-joint-velocity` | 0.5 | rad/s，每个关节 |
| `--topp-joint-acceleration` | 1.0 | rad/s²，每个关节 |
| `--topp-gripper-velocity` | 0.05 | m/s，每个夹爪 |
| `--topp-gripper-acceleration` | 0.2 | m/s²，每个夹爪 |
| `--topp-prefetch-seconds` | 0.4 | 旧计划剩余多少秒时请求下一块 |
| `--topp-gridpoints` | 200 | 路径离散网格基础点数，另外加入样条节点 |
| `--fps` | 15 | 原 runtime 输出频率 |
| `--topp-path-fps` | 15 | 路径节点名义频率，独立于输出频率 |

这些是本实验规划参数，不是 Piper 官方额定限值。速度/加速度约束是 TOPP 问题本身的输入，
只作用于这个新入口，不给官方 SDK 添加限制。

保留原参数 `--max-action-delta`、发布开关、相机/ROS 配置和记录订阅器。
原 `--actions-during-latency` 在新入口不使用，由时间预取参数替代。
RTC 默认关闭，显式 `--use-rtc` 会报错：重定时后原 RTC 的 s/d 索引语义不成立，不能直接沿用。
`--no-use-async` 可用于调试，但推理会阻塞发送。不发送旧 broker 的 prev_action/GPR 条件。

## 延迟与记录

规划目前在输出循环同步执行，推理在后台执行。每次只前进 1/fps 的计划时间；
不按积压墙钟时间追赶跳点，慢循环会拉长实际执行时间。
旧计划剩余 0.4 秒触发预取，并不等于额外固定等待 0.4 秒。
计算耗时若超过循环剩余预算，会推迟该帧；观测同步、ROS 和硬件还会贡献额外延迟。
默认输出 15 Hz，未声称有 100/200 Hz 的独立高频执行线程。

`--out-dir` 下 `topp_metrics.jsonl` 记录每块推理耗时、规划耗时、轨迹时长、名义时长和拒绝原因。
本机 40 组模拟静止起步路径：中位 12.5 ms、P95 15.1 ms、最大 18.7 ms（一次测量）。
这不是实机端到端延迟，也不包含初次导入开销。报告在 `data/topp_benchmark.json`。

## 依赖与离线验证

本次已将 `toppra==0.6.3` 安装到 `third_party/toppra_deps`，使用项目 `.venv` 的 Python 3.11 编译，
没有升级现有环境。新模块在环境无 toppra 时自动查找这个目录。
换 Python ABI 或机器时，用对应解释器安装 toppra 0.6.3；还需要 numpy/scipy（现有环境已有）。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  examples/piper_dual/tests/test_topp_policy.py \
  examples/piper_dual/tests/test_main_dual_topp_ros.py
.venv/bin/python examples/piper_dual/benchmark_topp.py
# 可选：用自己的 SI 单位动作块数组 (chunks, points, 14) 离线计时
.venv/bin/python examples/piper_dual/benchmark_topp.py --input actions.npy
```

关闭 pytest 自动插件加载，是为避免 ROS Python 3.10 插件污染项目 Python 3.11 测试。
官方算法/API：https://hungpham2511.github.io/toppra/python_api.html

输出频率与路径频率已拆分：只调整 `--fps` 不再改变相同输入位置/速度和动作块的规划路径。
`--topp-path-fps` 保持默认即可；它不是机械臂提速旋钮。闭环观测与动作块接续时机仍会随输出频率变化。
