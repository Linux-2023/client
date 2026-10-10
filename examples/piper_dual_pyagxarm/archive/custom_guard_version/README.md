# 基于 pyAgxArm 的 Piper 双臂 local-ros 推理

新增驱动位于本目录；`examples/piper_dual` 和 `/home/agilex/piper_ros` 原文件保留。
复用原 OpenPI 客户端、RTC/action chunk broker、三相机、14 维状态/动作、视频及关节记录。
实际 SDK 替换发生在 ROS → CAN 层。仅运行新入口而仍启动旧 ROS 驱动，底层仍然是旧 SDK。

## 版本与差异

本次检查：2026-09-29。官方源码：
[pyAgxArm](https://github.com/agilexrobotics/pyAgxArm)，
固定检查版本 `841a625f5f4920e776f20b934eb13048b747e6d0`（包版本 1.0.0）。
本地下载位置：`third_party/pyAgxArm`，未修改上游代码。

| 项目 | 本地现状 | pyAgxArm / 本次实现 |
| --- | --- | --- |
| 实际驱动 SDK | `/usr/bin/python3` 导入 `~/.local/lib/python3.10/site-packages/piper_sdk`，版本 **0.5.0** | 驱动优先导入本地 `third_party/pyAgxArm` |
| 仓库副本 | `third_party/piper_sdk` 为 **0.6.1**，提交 `3563428c43f62b0b362fe9a9341d304452c88402` | 保留，不卸载、不替换 |
| 创建对象 | 本地 ROS 节点使用 `C_PiperInterface`；其他直连代码使用 `_V2` | `create_agx_arm_config` + `AgxArmFactory.create_arm` |
| 机型/固件 | Piper 专用接口，版本入口较粗 | Piper/H/L/X/Nero 的工厂及固件分支；本驱动允许 Piper 系列 |
| 关节控制 | `MotionCtrl_2(1,1,speed,0)` + `JointCtrl`，整数 0.001° | `set_speed_percent` + `move_j`，rad |
| 反馈 | `GetArmJointMsgs().joint_state` 等包装 | `MessageAbstract.msg / timestamp / hz`，可能返回 None |
| 夹爪 | `GripperCtrl`，位置整数 µm、effort ×1000 | 独立 `init_effector(AGX_GRIPPER)`；`move_gripper_m`，m / force |
| 单位细节 | 本地 ROS 用 57324.840764 和 0.017444 近似转换 | SDK 使用更精确转换，可能有小幅数值差异；不是完全相同的轨迹 |
| 运动模式 | 当前默认 MOVE J，另有本地 MIT 实验扩展 | 本版只使用 `move_j`；`move_js` 是 MIT 直通语义，不能当等价替换 |
| 异常处理 | 原 local bridge 不采用 official stack 的状态锁 | 新驱动检查双臂状态/反馈时效，失败停止两臂并退出 |
| 限位 | 原 local 节点主要转换数据 | 关节目标交由官方 `move_j()` 处理，沿用 SDK 默认关闭模型限位的设置（仍有通用 ±2π 限幅）；夹爪米值交给官方 `move_gripper_m()`，仅显式指定 `--gripper-max` 时额外检查宽度 |

官方接口与固件依据：
[Piper API](https://github.com/agilexrobotics/pyAgxArm/blob/841a625f5f4920e776f20b934eb13048b747e6d0/docs/piper/piper_api.md)，
[固件矩阵](https://github.com/agilexrobotics/pyAgxArm/blob/841a625f5f4920e776f20b934eb13048b747e6d0/docs/piper/firmware_reference.md)。
官方文档将新夹爪参数 `force` 标为 N；旧本地桥接注释将 effort 标为 N·m。
本实现维持旧 **数值和 CAN 缩放**（默认 1，夹到 0.5..3），不把这两个文档标注当作物理单位已校准的证明。

## 实现计划与完成情况

1. 已完成：核对实际加载的 SDK、ROS 话题、动作单位和 CLI。
2. 已完成：新增 `arm.py`，封装 pyAgxArm、反馈有效性、双臂故障锁和命令超时。
3. 已完成：新增 `local_driver.py`，兼容原 local-ros 的关节、状态、命令话题。
4. 已完成：新增 `main_dual_ros.py` 入口，复用原推理代码；原命令无需改动也可使用新驱动。
5. 已完成：真实 SDK + Python virtual CAN 测试、ROS 接线模拟测试、原链路回归。
6. 型号/固件已由用户确认：两臂均为普通 Piper、S-V1.9-1，对应 `piper` / `v189`。
   待现场验收：只读反馈、低速动作和真实策略任务。此次未使能或驱动实体机械臂。

## 启动

保持相机和策略服务器原有启动方式。停止旧的双臂 ROS 控制 launch 后再启动新驱动；
同一 CAN 接口不能同时运行旧驱动、直连 SDK 控制器和新驱动。
本驱动检查 `/joint_*_states` 是否已有订阅者，但无法检测所有不通过 ROS 的 CAN 写入者。

先按真实固件选择参数，两臂可不同：

| 实际固件 | 参数值 |
| --- | --- |
| ≤ S-V1.8-2 | `default` |
| S-V1.8-3 ～ S-V1.8-7 | `v183` |
| S-V1.8-8 | `v188` |
| ≥ S-V1.8-9 | `v189` |

本机两臂已确认是普通 Piper、S-V1.9-1。`run_driver.sh` 默认传入
`--model piper --left-firmware v189 --right-firmware v189`，无需重复填写。
这里 `v189` 是 SDK 协议分支名，并不表示实体固件是 S-V1.8-9。
切换硬件时可在脚本后显式覆盖；直接运行 `local_driver.py` 仍需填写两臂固件参数。

终端 1，先只读检查（不发送使能或动作，不订阅动作话题）：

```bash
cd /home/agilex/client
bash examples/piper_dual_pyagxarm/run_driver.sh \
  --left-can-port can_left --right-can-port can_right
```

确认反馈正常后退出，只在准备实体运行时加入 `--enable`：

```bash
bash examples/piper_dual_pyagxarm/run_driver.sh \
  --left-can-port can_left --right-can-port can_right --enable
```

驱动启动不会回零，不会自动发送夹爪闭合命令；`--enable` 会使能关节并切换 MOVE J。
保持驱动终端可见，以便查看初始化、反馈、故障日志。

如果启动报告 `arm_status=1`（电子急停，可能由上一轮驱动退出时的停止命令留下），
确认机械臂可以恢复且停止原因已排除后，显式使用：

```bash
bash examples/piper_dual_pyagxarm/run_driver.sh --enable --reset-stop
```

`--reset-stop` 只对状态报告为电子停止且静止待机的臂发送一次 SDK `reset()`，
两侧状态先检查，有其他故障码时拒绝复位。恢复后仍需正常反馈和使能检查通过才接受动作。
不带该选项不会清急停。仅连接/检查反馈期间失败，不再额外发送电子停止；
一旦开始复位、切模式或使能，异常清理仍会停止两臂。

终端 2，你原来的核心命令可以**原样保留**：

```bash
cd /home/agilex/client
python examples/piper_dual/main_dual_ros.py \
  --host 127.0.0.1 \
  --port 8000 \
  --bridge-python /usr/bin/python3 \
  --ros2-config examples/piper_dual/ros2_piper_dual_local.yaml \
  --control-stack local-ros \
  --prompt "Pass the playing cards and sunflower seeds to the guests." \
  --max-action-delta 20 \
  --no-dry-run \
  --publish-actions
```

也可把上面入口替换为 `examples/piper_dual_pyagxarm/main_dual_ros.py`，其余参数全部相同。
该入口默认补齐 local-ros 及原 local YAML；不自动启动硬件驱动。
`--max-action-delta 20` 保留原含义：对模型动作的 rad/m 数值做差值检查，**不是 20 度速度限制**。
YAML 中 `speed_percent: 100` 仍传入 `velocity[6]`，新驱动保持原 30..100 的夹取逻辑。

## 数据及停止语义

### return_zero

`~/.local/bin/return_zero` 已切换到本目录的 `go_zero_dual.py`，通过 ROS 图识别
`pyagxarm_dual_local_driver` 或旧 `piper_left/right_ctrl_node`，不再解析 `ros2 topic info` 输出。
旧包装脚本保存在 `~/.local/bin/return_zero.legacy-piper-sdk`，原回零 Python 文件未修改。

```bash
# 终端 1：先启动驱动
bash examples/piper_dual_pyagxarm/run_driver.sh --enable
# 终端 2：先检查计划，再执行（执行会双臂回零并闭合夹爪）
return_zero --dry-run
return_zero --seconds 6
```

先停止推理/遥操作的动作发布者。工具检查唯一订阅者，要求双侧完整新鲜反馈，
缺少反馈时拒绝执行，不会直接跳零；斜坡按真实时间节拍运行。
旧 launch 自带的 `piper_pinocchio_ik` 可保留：工具通过 ROS 图检查其实际订阅的
`PosCmd` 话题，只有这些输入都没有发布者时才允许回零，并在每次发送目标前重新检查。
发现末端输入发布者、其他关节发布者或同名回零进程时停止发布，并报告具体节点/话题。
pyAgxArm 当前支持双臂回零；`--arm left/right` 只适用于旧独立驱动，
新驱动会明确拒绝单臂模式，避免另一侧命令流缺失触发双臂 watchdog。
正常回零完成后，工具通过 `/pyagxarm_dual/finish_zero` 通知驱动；驱动核对两臂最后目标均为零且新鲜反馈已收敛，结束当前命令流的 watchdog 计时并继续使能待命。下一项任务的首条动作重新启动双臂 watchdog。动作途中断流、回零失败或手动中断仍保留默认 5 秒超时停止；旧版驱动没有此完成服务，需要先重启到新版。
`--hold --timeout 60` 可在到位后继续发布指定时间，然后同样停止。
`--dry-run` 不发布动作；订阅者为 0 时需先启动带 `--enable` 的驱动，并检查 ROS_DOMAIN_ID/网络环境。
旧 Piper 驱动同样受支持，不需要改为 pyAgxArm。工具默认等待 DDS 发现最多 10 秒，
可用 `--discovery-timeout 20` 延长；启动会打印当前 ROS 域/本机发现限制/RMW，
超时会打印实际可见节点和关节话题。固定等待 1 秒导致的过早判定已移除。
订阅者数量为 1 但名称为 `_NODE_NAME_UNKNOWN_` 时，同样继续等待 DDS 节点元数据补齐，
不会立即判为不支持，也不会放行身份未知的控制端点。后续检查复用已确认的订阅端点快照。
两个终端必须使用一致的发现配置；本机 `.bashrc` 当前设置为 `ROS_DOMAIN_ID=222`、
`ROS_LOCALHOST_ONLY=1`，但旧终端或启动脚本可能覆盖这些值，以各自实际环境为准。

| ROS 话题 | 新驱动语义 |
| --- | --- |
| `/puppet/joint_left`, `/puppet/joint_right` | 测量关节 rad + 夹爪宽度 m |
| `/joint_left_states`, `/joint_right_states` | 同序七维命令；velocity[6] 为速度百分比，effort[6] 为夹爪力数值 |
| `/master/joint_left`, `/master/joint_right` | 最近成功发送的命令回显；首条命令前用测量状态初始化，满足原同步器的启动条件 |
| `/piper_left_ctrl_node/arm_status`, `/piper_right_ctrl_node/arm_status` | pyAgxArm 的真实状态字段映射 |

`/master/*` 在本版是软件命令回显，**不代表实体主臂反馈或 CAN 应答确认**，不要把它当遥操作采集标签使用。
发布时使用 ROS 当前时间，先检查 SDK 原始 CAN 时间戳，过期反馈不会重新包装成正常帧。
检查三个关节对帧的时效需要访问固定 SDK 版本的 `_parser.joint_12/34/56`；升级 SDK 时须复测。

默认反馈超时 0.5 秒；首条动作后，任一臂连续 5 秒没有新命令会触发共同停止。
可用 `--feedback-timeout` / `--command-timeout` 调整。驱动异常或退出时，如果以 `--enable` 启动，
调用两臂 `electronic_emergency_stop()` 后断开连接；不执行回零或掉电。
电子停止可能需要按现场流程恢复，再重启；本版不自动清故障/复位。
ROS 双臂话题分别到达，CAN 发送不具备事务原子性；单侧发送失败会立即尝试停止两臂。

本次交付聚焦请求中的关节推理命令。原 EEF/Pinocchio IK、MIT 参数实验、遥操作采集、归零及夹爪标零工具
继续留在原版，未声称已移植到新驱动。新驱动不发布 EEF 话题。

## 依赖、验证、回退

### 运行中出现 `expected CAN control / MOVE J`

`ctrl_mode=0` 表示待机，`mode_feedback=0` 表示 MOVE P。
首次动作之前，驱动允许 `ctrl_mode=0, mode_feedback=0` 的待机状态继续等待，
要求故障码、示教状态及运动状态均为零、六个电机仍使能、反馈仍新鲜。
等待期间不发模式保持或位置目标。收到首条合法动作后，向待机臂请求 MOVE J，
两臂都通过检查且请求后产生的新模式反馈在 0.5 秒内到达，才发送动作；不自动复位或重新使能。
首条动作发出后，退出 CAN/MOVE J 仍会停止两臂并退出。无模式确认也会停止，不能跳过检查。
不能仅凭待机帧断定造成模式变化的固件或外部原因。新增错误上下文包含左右侧、
`waiting for first action` / `action stream started`、该侧已发送命令数以及流启动后的 watchdog 时间。
退出日志同时报告停止请求及发送失败信息；停止请求不等同于硬件应答确认。
排查时保留完整日志，并记录是否已经启动推理、期间是否复位/断电或运行其他 CAN 控制程序。

### `arm_status=5, err_code=14`

表示主控报告关节通信异常，`14 = 0x000E` 对应关节 2、3、4 的通信故障位。
协议中低字节为通信故障，高字节为关节角度限位；ROS 状态投影已按固定 SDK 源码校正。
若 `commands_sent=0`，说明发生在发送策略/回零目标之前，但启动仍曾发送复位、模式和使能指令。
不能仅凭该帧确定是供电、臂内连接还是固件/驱动交互问题。不要以循环复位或忽略故障码作为修复。
确认不存在并行 CAN 控制进程，并在按设备操作流程停止后检查供电和连接；记录相同硬件下旧驱动
是否也报告通信故障，以便进一步定位。

### 回零持续发布，但角度不变

不能把 `control_mode=joint` 当作完整运动配置。工具现在还读取并校验
`joint_is_mit_mode`、`joint_move_mode`、`joint_speed_override`。
本次现场只读采样发现左臂 `joint_is_mit_mode=173`、`joint_speed_override=0`：
前者选择 MIT 标志，后者覆盖命令速度为零；右臂这两项为正常的 `0` / `-1`。
CAN 采样时两侧六个电机均使能、状态无故障，不能将不动简单解释成电机未使能。
停止动作发布后，可将左臂参数恢复普通 MOVE J 配置：

```bash
ros2 param set /piper_left_ctrl_node joint_is_mit_mode 0
ros2 param set /piper_left_ctrl_node joint_speed_override -1
```

这两项在现有旧驱动中只影响后续指令；工具不会自动修改在线模式参数。
只读采集工具（不发动作/使能/复位）：

```bash
source /opt/ros/humble/setup.bash
source /home/agilex/piper_ros/install/setup.bash
/usr/bin/python3 examples/piper_dual_pyagxarm/diagnose_ros.py
/usr/bin/python3 examples/piper_dual_pyagxarm/diagnose_can.py
```

右臂参数正常但用户上次运行也未移动，仍需执行期间的命令接收/CAN 发送证据进一步定位；
事后无故障状态不证明回零命令已经被驱动执行。

本机 `/usr/bin/python3` 已具备 python-can 4.6.1、ROS Humble 和 piper_msgs；没有替换系统包。
`run_driver.sh` 加载原 ROS 安装环境；OpenPI 继续使用原虚拟环境 Python。
重建机器时需要单独检出固定 SDK：

```bash
git clone https://github.com/agilexrobotics/pyAgxArm.git third_party/pyAgxArm
git -C third_party/pyAgxArm checkout 841a625f5f4920e776f20b934eb13048b747e6d0
```

离线验证命令（不访问实体 CAN）：

```bash
/usr/bin/python3 -m pytest -q examples/piper_dual_pyagxarm/tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  examples/piper_dual/tests/test_main_dual_ros.py \
  examples/piper_dual/tests/test_ros2_environment.py \
  examples/piper_dual/tests/test_ros2_backend.py
/usr/bin/python3 -m pytest -q examples/piper_dual/tests/test_ros2_local_bridge_codec.py
```

pytest 自动插件加载需对 OpenPI Python 禁用，避免误加载 ROS 的 Python 3.10 测试插件。
ROS bridge 的测试用系统 Python 执行。

本次结果：新版本 25 项测试通过（含四种固件的真实 SDK 虚拟 CAN 编解码），
原入口/环境/backend 回归 87 项通过，原 local bridge codec 2 项通过，共 **114 项**。
系统 Python 的 ROS 消息导入、Python 编译、shell 语法检查通过；
原 `examples/piper_dual` 的 116 个文件经前后 SHA-256 核对，无内容变化。
这不等于实体双臂、相机和在线策略服务器已经完成整机验收。

回退：停止策略进程和新驱动，按现场停止恢复流程处理机械臂，再按原方式启动
`piper_ros` 双臂控制 launch，运行原核心命令。无需还原代码、卸载包或改 YAML。
