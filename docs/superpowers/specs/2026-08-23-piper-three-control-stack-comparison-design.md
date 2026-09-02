# Piper 三控制栈统一真机对比设计

## 目标

让同一个 `examples/piper_dual/main_dual.py`、同一个 PI05 policy server、同一个 prompt 和同一套 14 维动作约束，依次驱动以下三套互斥控制栈：

1. `local-ros`：`/home/agilex/piper_ros` 当前本地 ROS 2 控制栈；
2. `official-ros`：`/home/agilex/piper_ros/.worktrees/piper-official-humble` 官方 Humble 候选栈；
3. `direct-sdk`：复用 `/home/agilex/lgd/control_your_robot/src/robot/controller/Piper_controller.py` 的 `piper_sdk.C_PiperInterface_V2` 直接控制栈，通过独立 ROS 2 适配进程接入统一客户端。

用户通过三次真机运行和分别保存的视频肉眼比较任务效果。系统不实现自动评分或排名。

## 非目标

- 不比较 EEF 与关节动作空间；三套栈都使用同一 14 维关节加夹爪策略契约。
- 不比较 `PiperAgx_controller.py`、MIT 模式或重力补偿遥操作。
- 不让 `main_dual.py` 配置 CAN、启动控制栈、使能或复位机械臂。
- 不同时运行两套控制栈，不让两个进程同时占用 `can_left` 或 `can_right`。
- 不修改原本地 ROS 控制实现、官方 `piper_single_ctrl` 或 `control_your_robot/Piper_controller.py`。
- 不把 mock、dry-run 或单元测试描述为真机验收。

## 现状与问题

`main_dual.py` 已统一 PI05 policy/runtime，并支持 `sdk` 与 `ros2` 环境，但现有 ROS bridge 已迁移到官方话题契约。三套目标栈的接口不同：

| 栈 | 状态/目标话题 | 动作话题 | 关节名 | CAN 所有者 |
|---|---|---|---|---|
| `local-ros` | `/puppet/joint_left/right`、`/master/joint_left/right` | `/joint_left_states`、`/joint_right_states` | `joint0..joint6` | 本地 `piper_single_ctrl` |
| `official-ros` | `/joint_left/right`、`/joint_states_ctrl_left/right` | `/joint_ctrl_cmd_left/right` | `joint1..joint6,gripper` | 官方 `piper_single_ctrl` |
| `direct-sdk` | 当前不存在稳定 ROS 接口 | 直接 `JointCtrl` | Python 数组 | 外部直接 SDK 适配进程 |

因此不能只替换启动命令后继续使用同一份官方 YAML。需要让每套栈先归一化到同一客户端契约。

## 用户接口

保留 `main_dual.py` 作为唯一策略入口，新增：

```text
--control-stack local-ros|official-ros|direct-sdk
```

三轮策略命令仅允许 `--control-stack` 和 `--run-tag` 不同。例如：

```bash
python examples/piper_dual/main_dual.py \
  --backend ros2 \
  --control-stack official-ros \
  --host 127.0.0.1 \
  --port 8000 \
  --prompt "Stack_the_paper_cups_together." \
  --fps 30 \
  --num-steps 1600 \
  --no-dry-run \
  --publish-actions \
  --max-action-delta 0.05 \
  --run-tag official-ros
```

约束：

- `--control-stack` 只对 `--backend ros2` 有效；其他组合在创建环境前失败。
- 默认仍保持安全且兼容：现有未指定 `--control-stack` 的 ROS 路径使用 `official-ros`，但仍遵守既有 dry-run 默认和发布安全门。
- 每个 profile 的 YAML 写入不可变 `control_stack_id`。显式 `--ros2-config` 覆盖时，该 ID 必须与 `--control-stack` 一致，否则拒绝启动。
- `main_dual_ros.py` 继续作为官方栈专用入口；三栈对比使用 `main_dual.py`。

## 统一数据契约

三套 profile 对客户端提供完全相同的数据：

```text
observation.state = [left J1..J6, left gripper, right J1..J6, right gripper]
action            = [left J1..J6, left gripper, right J1..J6, right gripper]
joint units        = rad
gripper units      = m
shape              = (14,)
```

相机映射保持不变：

```text
/camera_f_l/color/image_raw -> cam_high
/camera_l/color/image_raw   -> cam_left_wrist
/camera_r/color/image_raw   -> cam_right_wrist
```

所有 profile 使用相同的：

- `max_sync_error_ms=30`；
- `gripper_effort=0.5`；
- `max_action_delta=0.05`；
- 状态 watchdog；
- 动作有限值、维度和单步变化校验；
- 致命故障后永久锁定当前进程的动作发布。

速度是本次有意保留的控制栈差异：`local-ros` 与 `official-ros` 通过 `JointState.velocity[-1]` 使用 `speed_percent=30`；`direct-sdk` 原样调用第三方 `PiperController.set_joint()`，其 `MotionCtrl_2` 固定使用 100%。结果必须明确标注该差异，不能把肉眼差异只归因于 ROS 与直接 SDK 通信路径。

不同底层的关节名、话题名和单位转换只存在于 profile/适配器内部，不泄漏给策略和 runtime。

## 组件设计

### 1. 控制栈 profile

新增一个小型不可变 profile 层，集中定义：

- `control_stack_id`；
- 默认 YAML 路径；
- 必需与禁止的 ROS 图指纹；
- 预期节点、话题和消息类型；
- 关节名称约定；
- 动作 publisher 端点。

`main_dual.py` 只负责选择 profile 并把配置交给现有 `Ros2DualEnvironment`。不在入口中写三份分支化控制逻辑。

### 2. `local-ros` profile

新增本地旧栈 YAML，映射：

```text
feedback:
  /puppet/joint_left  -> puppet_left
  /puppet/joint_right -> puppet_right
command echo:
  /master/joint_left  -> master_left
  /master/joint_right -> master_right
action:
  left  -> /joint_left_states
  right -> /joint_right_states
status:
  left  -> /piper_left_ctrl_node/arm_status
  right -> /piper_right_ctrl_node/arm_status
joint names:
  joint0..joint6
```

bridge 继续把输入归一化为 J1–J6 加夹爪，不改变旧节点。

### 3. `official-ros` profile

复用并标识当前 `ros2_piper_dual.yaml`：

```text
feedback:
  /joint_left  -> puppet_left
  /joint_right -> puppet_right
command echo:
  /joint_states_ctrl_left  -> master_left
  /joint_states_ctrl_right -> master_right
action:
  left  -> /joint_ctrl_cmd_left
  right -> /joint_ctrl_cmd_right
status:
  /arm_status_left
  /arm_status_right
joint names:
  joint1..joint6,gripper
```

官方控制节点保持不改。

### 4. `direct-sdk` 外部 ROS 2 适配进程

新增独立进程；文件位于 client 示例目录，但通过 `PYTHONPATH=/home/agilex/lgd/control_your_robot/src` 导入并使用原 `PiperController`。不修改第三方仓库。

进程参数包括：

```text
--left-can can_left
--right-can can_right
--allow-enable
```

安全行为：

- 缺少 `--allow-enable` 时不得构造 `PiperController`、不得打开 CAN、不得使能；只打印明确错误并退出。
- 指定 `--allow-enable` 后，分别调用原控制器的 `set_up(can)`；这会连接并使能真机，因此必须由操作者在独立终端显式执行。
- 只接受通过统一 bridge 验证后的 7+7 有限动作。
- 关节动作以弧度原样交给 `PiperController.set_joint()`；统一契约中的夹爪米值先校验在 `[0.0, 0.07]`，再除以 `0.07` 转成 `PiperController.set_gripper()` 所需的 `[0.0, 1.0]` 开度。
- `PiperController.get_state()["gripper"]` 实际返回归一化开度；适配器必须乘以 `0.07` 后才可发布为统一契约的夹爪米值。
- `direct-sdk` 保留 `PiperController.set_joint()` 内部固定 100% 速度；两套 ROS profile 使用 30%。这是用户明确选择的被测差异，结果元数据必须记录。
- 反馈来自 `PiperController.get_state()`；不得用目标值冒充实际反馈。
- 状态消息在六个 driver 全部使能时复制底层 SDK 的真实 `ctrl_mode`、`arm_status`、`mode_feed`、`motion_status` 和 `err_code`。任一 driver 未使能或字段缺失时，适配器保留实际字段并将对外 `ctrl_mode` 明确投影为非就绪值 `0`、锁定动作并记录原因；不得发布健康占位值。
- 适配器同时发布 `/direct_sdk/joint_ctrl_left/right` 命令回显；它表示最近一次实际接受并发送的米制 7 维动作。未发送动作前使用启动时真实反馈作为初值。
- 使用独立 `/direct_sdk/...` 命名空间，避免和任一 ROS Piper 栈碰撞。
- 收到 SIGINT/SIGTERM 后停止接收新动作；尽最大可能执行双臂 `DisableArm(7)` 并关闭连接。任何一侧清理失败必须打印并返回非零状态，不能报告成功。

### 5. ROS 图身份与冲突检查

bridge 在创建动作 publisher 前验证所选 profile：

- 所有必需状态/命令回显话题存在且消息类型正确；
- 禁止的其他栈指纹不存在；
- 每个动作端点的订阅者数量与预期一致；
- 左右状态均新鲜、无故障并处于允许控制状态；
- direct SDK profile 还要求外部节点身份为 `piper_direct_sdk_adapter`。

典型指纹：

- `local-ros`：需要 `/puppet/joint_left`；禁止 `/joint_left` 和 `/direct_sdk/joint_left`。
- `official-ros`：需要 `/joint_left`；禁止 `/puppet/joint_left` 和 `/direct_sdk/joint_left`。
- `direct-sdk`：需要 `/direct_sdk/joint_left`；禁止本地与官方 Piper 状态指纹。

不满足时 dry-run 可以打印诊断但不能创建 publisher；live 模式直接失败。这样不会因 source 顺序或同名 `piper` 包误连另一套控制栈。

## 控制栈生命周期

三套底层栈均由操作者在另一个终端手动启动和停止。`main_dual.py` 只连接已运行的栈。

切换顺序固定：

1. 停止 `main_dual.py`，等待 bridge 退出；
2. 停止当前底层控制进程；
3. 确认没有 `piper_single_ctrl`、direct SDK adapter 或 bridge 残留；
4. 确认 `can_left`、`can_right` 都是 1 Mbps 且 `ERROR-ACTIVE`；
5. 启动下一套底层控制栈；
6. 先观察左右真实状态与急停；
7. 使用对应 `--control-stack` 启动同一策略任务。

禁止用当前 `start_multi_piper.sh` 作为公平实验入口，因为它同时配置 CAN，并让旧 launch 使用默认 `auto_enable=true`。本地与官方 ROS 栈都必须使用显式 `auto_enable:=false` 的 launch 命令，再由操作者显式使能。direct SDK 通过外部进程的 `--allow-enable` 明确确认使能副作用。

## 三轮真机比较协议

用户选择直接策略任务，不增加固定轨迹校准阶段。每轮必须保持：

- 同一个 PI05 server/checkpoint；
- 同一个 prompt；
- 同一个 `fps`、action horizon、异步/RTC 设置和步数；
- 同一个 `max_action_delta=0.05`；
- 同一相机位置、曝光设置和图像预处理；
- 同一机械臂起始姿态；
- 同一物体初始位置；
- 无其他控制进程占用 CAN。
- 被测速度差异固定记录为 `local-ros=30%`、`official-ros=30%`、`direct-sdk=100%`；不在三轮之间调整；

每轮使用独立 `run-tag`：

```text
local-ros
official-ros
direct-sdk
```

每个非空 `run-tag` 使用独立输出子目录；视频采用实际客户端 `fps`，避免硬编码播放速度影响肉眼判断。原子写入最小 `run_metadata.json`：stack ID、起止时间、prompt、客户端参数、策略地址、视频 fps、有效速度口径、退出码和退出原因。用户通过视频和现场表现肉眼比较；不增加自动成功率、延迟或轨迹评分系统。

## 失败与安全处理

以下任一条件立即停止当前 episode 并永久锁定动作发布，必须重启客户端后才能恢复：

- 左右任一状态缺失、过期或报告急停/碰撞/驱动异常；
- 栈身份或话题类型与 `--control-stack` 不一致；
- 检测到另一控制栈的 ROS 指纹；
- 14 维动作维度错误、包含非有限值或超过单步变化限制；
- bridge/adapter EOF、写入失败或 CAN 反馈中断；
- 左右任一侧 direct SDK 初始化、使能或清理失败。

发生故障后不能自动重试动作、自动切换栈或自动解除故障锁。操作者先停止所有进程并检查机械臂/CAN。

## 测试设计

自动测试不得连接 CAN 或使能机械臂，覆盖：

1. `--control-stack` 解析、默认值及与 `--backend` 的非法组合；
2. 三个 profile 的完整端点、关节名和 `control_stack_id`；
3. 自定义 YAML 与所选 stack ID 不一致时拒绝；
4. 三套 ROS 图指纹的成功、缺失、错误类型和冲突拒绝；
5. dry-run 在任何 profile 下创建零动作 publisher；
6. 三套 profile 消息均归一化为同一 14 维 rad/m 契约；
7. direct SDK adapter 在缺少 `--allow-enable` 时不实例化控制器；
8. fake `PiperController` 下的反馈、命令、真实状态故障和左右初始化失败；
9. direct SDK 夹爪反馈执行 `normalized * 0.07 -> m`，动作执行 `m / 0.07 -> normalized`，拒绝范围外输入；
10. direct SDK 动作只调用对应侧 `set_joint`/`set_gripper`，保留原生 100% 速度及动作限幅；
11. SIGINT/SIGTERM 后停止动作并尝试双臂清理；
12. 三个 run tag 和包含实际速度口径的最小比较元数据分别保存；
13. 现有官方 ROS、旧 `sdk` backend 和 `main_dual_ros.py` 回归测试保持通过。

## 真机验收

真机验收由操作者逐栈执行，不纳入自动测试：

1. 证明当前只运行一套底层控制进程；
2. 观察左右 arm status、反馈频率和初始 14 维状态；
3. 使用相同策略参数完成一轮任务；
4. 以 Ctrl-C 正常停止并确认无新动作；
5. 保存带 stack ID 的视频；
6. 停止底层栈后再开始下一轮。

只有三轮都实际完成后，才能声称完成真机三栈比较。实现完成、mock 测试通过或单套真机运行均不能替代该结论。

## 验收标准

- 同一个 `main_dual.py` 命令模板通过 `--control-stack` 选择三套栈。
- 三套栈向策略暴露相同的 14 维状态/动作、rad/m 单位、相机和安全限制。
- 速度不是归一化变量：本地/官方 ROS 为 30%，direct SDK 保留原生 100%；报告必须显式标注，避免错误归因。
- `main_dual.py` 不启动、使能或复位任何底层栈。
- 选择错误、配置错配或并发控制栈在发布动作前失败。
- direct SDK 复用 `control_your_robot` 的 `PiperController`，且第三方源码保持不改。
- 三轮输出按 stack ID 分离，足够供用户肉眼比较。
- 自动验证不打开 CAN；真机结果单独记录，不从测试推断。
