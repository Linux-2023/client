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

## 接缝过渡：减少带速接续失败（当前默认开启）

新模块 `topp_transition_planner.py` 只用于 committed 后端，默认 `--topp-transition`。
`--no-topp-transition` 恢复本次优化前的 committed 规划器；legacy 与非RTC TOPP不变。
修改前的入口/调度器/说明保存在 `backups/topp_rtc_before_transition/`。

先尝试原有带速TOPP；若初始加速度区间无交集，直接跳过不可能成功的网格重算。
若原路径无法带速接续，生成一个从相同位置静止起步的TOPP参考，叠加短五次多项式位置修正：
`e(t)=dq*T*(u-6u³+8u⁴-3u⁵)`，其中 `u=t/T`，仅作用于0到T。
组合轨迹初始速度仍是原 dq，**不会在运动中将速度清零**；到T时修正的位置、速度、加速度均为零。
可行的直接接续保持原样；修正只发生在承诺时刻之后，固定前缀不变。

- 过渡时长搜索0.1到0.6秒，`--topp-transition-max-seconds` 可调整上限。
- 必要时参考轨迹统一放慢至原时长的1.15、1.35或1.6倍，为加速度修正留余量。
  所以这是减少停车次数的尝试，不承诺每段都更快。
- 检查组合轨迹的速度、加速度，不提高用户原上限；检查仍是密集数值检查，不是连续域严格证明。
- 额外位置偏移相对“同一时间尺度的静止起步TOPP参考”，不是相对原模型折线的总误差。
  解析最大偏移为 `abs(dq)*T*16/81`，默认关节0.03rad、夹爪0.002m，
  参数分别为 `--topp-transition-joint-deviation` 和 `--topp-transition-gripper-deviation`。
- 参考样条自身仍可能偏离模型折线。这里没有保证工具末端笛卡尔偏差、碰撞距离、跨块加速度或jerk连续。
- 所有候选均失败仍保留旧停止轨迹，不能保证彻底消除停顿。

worker_done/adopted日志增加 transition 字段，记录 used、blend_s、time_scale、
max_joint_offset_rad、max_gripper_offset_m 和直接接续失败原因。

保存的真实失败输入已复制为离线测试夹具 `tests/fixtures/rtc_nonzero_handoff_failure.npz`。
该样本原规划失败；新规划用0.2秒过渡、1.15倍时长通过，额外关节偏移最大0.00154rad，
夹爪0.000179m；初始q/dq不变，过渡期间没有所有关节同时零速的停点。

相同固定动作块、模拟250ms推理的6秒回放（不是实际模型闭环）：
旧版采用5块/拒绝4次；新版本采用9块/拒绝0次，其中7次非零速度接续。
新版本主线程策略调用P95约0.65ms、最大约1.00ms。54项原/新增测试后又加入直接接续保持测试，共55项。
这些结果不能替代实际抓取成功率、跟踪误差和停顿次数的对比。没有启动机械臂动作。

```bash
# 比较固定动作回放，不启动机器人
.venv/bin/python examples/piper_dual/benchmark_topp_rtc_dispatch.py \
  --input examples/piper_dual/tests/fixtures/rtc_nonzero_handoff_failure.npz \
  --out data/topp_rtc_transition_benchmark.json
.venv/bin/python examples/piper_dual/benchmark_topp_rtc_dispatch.py --no-transition \
  --input examples/piper_dual/tests/fixtures/rtc_nonzero_handoff_failure.npz \
  --out data/topp_rtc_without_transition_benchmark.json
# 新增回归测试
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  examples/piper_dual/tests/test_topp_transition_planner.py
```

## 过渡失败后的有限运动中重试

最新实测日志仍有15次过渡失败、28次采用中16次零速起步，42次后台完成均未超时。
因此没有证据将这些停顿归因于发送循环或截止时间；不能声称过渡已消除停顿。

当前恢复调度允许每个已采用计划最多一次运动中重试：必须等原承诺时刻过去，
旧轨迹必须仍留有完整推理/规划预算，预计接入时刻速度必须非零且预算未被模型窗口截断。
否则仍沿旧计划减速；二次失败不再反复重试。旧承诺前缀不会被提前撤销。
失败请求的完整规划输入现保存到 `--out-dir/topp_diagnostics/last_committed_failure.npz`，
包括q/dq、实际使用的22点动作、限值、频率和失败原因（覆盖最近一次）。
这补足了此前过渡失败可能没有覆盖旧诊断文件的问题，供后续针对真实新失败离线复现。
日志 rejected 增加 retry_moving、retry_at_tick；58项离线测试通过，尚无本次调度修改后的实机结果。

## 候选约束诊断

诊断不调整约束、候选顺序或接受条件。每个进入搜索的候选记录 blend_s、time_scale、
checked、not_checked、status 和 violations。每个超限项包含 constraint、axis、带符号actual、
magnitude、limit、ratio、unit、time_s；关节用rad相关单位，夹爪用m相关单位。
偏移量用解析峰值，time_s为空；非有限值显式标为nonfinite，不伪造超限倍数。
速度不合格的候选仍检查加速度，避免只报告第一项而漏掉其他约束；
时长或偏移先被拒绝的候选不计算动力学，明确列为not_checked，不能当成通过。

`transition.diagnostics` 随 worker_done 的成功和失败结果一起进入运行日志，汇总：
- evaluated_candidates/rejected_candidates：检查/拒绝的候选数。
- violation_counts：各约束的“候选×轴”超限项数，不是候选数。
- axis_counts：各轴的超限项数。
- worst_by_constraint：每类约束最严重的超限值、限值、倍数及候选参数。
- diagnostic_id：对应明细文件中的同一条记录。

默认输出目录下的 `topp_diagnostics/transition_candidates.jsonl` 追加保存各次过渡搜索的完整明细，
`last_transition_failure.json` 保存最近一次最终失败（后续成功不会清除此历史失败文件，需看diagnostic_id）。
可使用 `python -m json.tool data/piper_dual/videos/topp_diagnostics/last_transition_failure.json` 查看。
未进入过渡搜索的失败会记录error及零候选，不能误认为已评估全部过渡方案。

duration超限特指“候选过渡长于参考轨迹可用时间”；search.max_blend_seconds是搜索边界，
超出该边界的方案没有尝试，不能由失败诊断证明必须延长过渡时长。
偏移上限仍仅相对TOPP参考，不是相对原模型点的总偏移。

真实旧夹具离线重放示例（不是最新实机）：8个候选中7个被拒绝，均涉及加速度，
最严重项为0.1秒/1倍时长候选的left.joint2：-2.93386 rad/s²，上限2，倍数1.46693。
第8个候选通过。报告另存 `data/topp_transition_diagnostic_replay/`，没有覆盖运行诊断。
新增诊断归因、单位、未检查标记、失败文件与成功搜索记录测试，共62项离线测试通过。
