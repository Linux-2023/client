# ROS2 专用 Piper 部署入口设计

## 目标
新增 `examples/piper_dual/main_dual_ros.py`，提供仅面向 ROS2 bridge + 远程 PI05 WebSocket server 的双臂 Piper 部署入口。原 `main_dual.py` 保持不变，继续承担 SDK/ROS2 兼容入口职责。

## 范围
入口只暴露远程策略连接、Runtime 参数、ROS2 bridge 配置和动作发布安全门。复用现有 `Ros2DualEnvironment`、`WebsocketClientPolicy`、action broker、`PolicyAgent`、`Runtime`、`VideoSaver` 与 `RobotStatePlotter`；不复制 SDK 硬件实现。

删除的职责和参数：SDK backend 选择、CAN 端口、USB/RealSense camera ID、tele_mode、gripper_norm、SDK record_mode，以及所有 SDK 专用环境分支。

## 参数与安全
- `--host`、`--port`：PI05 WebSocket server 地址。
- `--bridge-python`、`--ros2-config`：ROS2 bridge interpreter/config。
- `--dry-run/--no-dry-run`：默认 dry-run。
- `--publish-actions`：只允许与 `--no-dry-run` 同时使用。
- `--max-action-delta`：可选单步动作变化上限。
- `--prompt`、`--fps`、`--num-steps`、`--num-episodes`、`--out-dir`、`--action-horizon`、`--actions-during-latency`、`--use-async`、`--use-rtc`、`--run-tag`：运行时参数。

非法安全组合在创建环境前返回退出码 2；运行时错误返回 1；Ctrl-C 返回 130，并关闭 runtime/environment。

## 数据流
1. 入口解析参数并校验动作发布门。
2. 创建 `Ros2DualEnvironment`，其 sidecar 使用 `/usr/bin/python3` 和 YAML 配置连接官方 ROS2 节点。
3. 创建 `WebsocketClientPolicy`，按需包裹 `ActionChunkBroker` 或 `ActionChunkBroker_RTC`。
4. 创建 `Runtime`，使用 `PolicyAgent` 获取动作，交给 ROS2 environment 做 14 维校验、限幅和可选发布。
5. Runtime 结束或收到信号后关闭资源。

## 验证
新增入口测试覆盖：默认仅 ROS2 参数、帮助信息不含 SDK/CAN/camera 参数、安全门、参数解析、ROS2 environment 构造以及 policy/runtime 构造契约。执行 Python client 测试、CLI help 和双解释器 `py_compile`；不启动物理 ROS2 节点、不发布真实动作。
