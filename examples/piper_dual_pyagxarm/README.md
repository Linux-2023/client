# pyAgxArm 双臂 ROS 桥

底层使用本地官方仓库 `third_party/pyAgxArm`，固定提交
`841a625f5f4920e776f20b934eb13048b747e6d0`。
硬件配置：普通 Piper，S-V1.9-1，使用官方 `v189` 驱动。
ROS 桥是本项目代码，不是官方 ROS 节点；原 `examples/piper_dual` 保留。

## 启动

先停止旧版双臂驱动，保持单一 CAN 控制来源：

```bash
bash examples/piper_dual_pyagxarm/run_driver.sh --enable
```

需要显式恢复时使用 `--enable --reset-stop`，仅对实际报告电子停止的手臂调用一次官方 `reset()`；等待恢复后新鲜的未使能反馈，再调用使能。
`--enable` 循环调用官方 `enable()`，并要求调用后新收到的六个电机反馈全部已使能，避免缓存状态造成提前成功。
`--startup-timeout` 默认 10 秒，只限制启动等待，不控制运动。

## 动作处理

ROS 七维 position：前六项 rad，最后一项夹爪宽度 m。
velocity[6] 转为整数百分比传给 `set_speed_percent()`；effort[6] 原值传给夹爪接口。
无 velocity/effort 时默认 30% 和 1.0。速度不变时不重复发送设置。
关节、夹爪直接调用官方 `move_j()`、`move_gripper_m()`。
范围处理和接口异常遵循官方 SDK；保留七维、有限数值、关节顺序检查。

没有自加的关节/夹爪限位、速度/夹爪力截断、模式 ACK 前置判断、双臂故障锁、
命令 watchdog 或退出自动电子急停。停止发布、Ctrl+C 和进程退出只结束发送/连接，
不代表机械臂立即停下；控制器可能继续执行最后目标。
反馈中故障字段原样发布；缺失、过期或非米制夹爪反馈不伪装为新鲜 ROS 状态，
但不会因此向另一臂发送急停。启动时仍检查 ROS 命令话题没有其他驱动。
上层原推理客户端的动作检查保持原样，例如用户指定的 `--max-action-delta`。

## 回零与推理

```bash
return_zero --seconds 6 --speed-percent 30
```

回零工具根据反馈报告是否到位，驱动保持运行。
原 `examples/piper_dual/main_dual_ros.py --control-stack local-ros` 命令继续使用，
相机节点和推理服务需另外运行。旧版额外保护实现及测试备份在
`archive/custom_guard_version/`，不会由启动脚本加载。

## 离线验证

```bash
source /opt/ros/humble/setup.bash
source /home/agilex/piper_ros/install/setup.bash
/usr/bin/python3 -m pytest -q examples/piper_dual_pyagxarm/tests
```

测试包含官方 SDK 的虚拟 CAN 编码和 ROS 话题适配，不代表实机推理已经验证。

## MOVE JS 选项

```bash
bash examples/piper_dual_pyagxarm/run_driver.sh --enable --motion-mode js
```

默认 `--motion-mode j`；`js` 直接调用官方 `move_js()`，六关节仍为 rad。
官方 JS 是 MIT 快速响应方式，没有 MOVE J 的轨迹平滑；不要把速度百分比当作 JS 的可靠速度上限。
虚拟 CAN 已验证模式帧为 MOVE J 字段 1、MIT 标志 0xAD；普通 J 标志为 0。
2026-09-29 做过双臂零位保持 2 秒验证，反馈无故障，未验证非零轨迹或完整推理。
SDK SocketCAN 默认 local_loopback=False，因此另一 SocketCAN 监听器看不到本机发送帧；
不能用监听器里缺少 TX 帧推断驱动没有下发。

## EEF 数据采集

驱动发布 `/puppet/end_pose_left`、`/puppet/end_pose_right`（PoseStamped），
来自官方 `get_flange_pose()` 的米/弧度反馈；RPY 按 XYZ 外旋转四元数。
关节和 EEF 共用 ROS 发布时刻，读取前检查各组 CAN 反馈的新鲜度。
旧进程未重启时，可运行 `eef_feedback.py` 只读补发；新版驱动发现已有
EEF 发布者会沿用它，避免重复发布。若要迁回驱动内发布，先关闭只读节点再重启驱动。
注意本桥的 `/master/joint_*` 是命令回显（无命令时为实际位置），不是独立主臂测量；
主从遥操作采集时需要核对 action 的来源，不能把这个回显当作主臂传感器数据。
