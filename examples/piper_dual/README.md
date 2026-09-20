## Piper 双臂机器人环境

本目录包含双臂 Piper 机械臂的数据采集和模型部署代码。

### 一、环境构建

在松灵 Piper 双臂机械臂上构建真机部署环境，安装方式如下：

```bash
sudo apt install can-utils 
uv venv --python 3.10 examples/piper_dual/.venv
source examples/piper_dual/.venv/bin/activate
uv pip install opencv-python python-can piper_sdk pyrealsense2 imageio[pyav] tyro matplotlib h5py
uv pip install -e packages/openpi-client
```

### 二、数据收集

#### 1. 硬件连接
**请在以下操作前将双臂机械臂掰回初始状态。**

真机需要先连接硬件：
- 将双臂机械臂、RealSense 相机、两个腕部 USB 相机的接口都接入 PC
- 机械臂需要给左右两个主臂和从臂都接入电源
- 通过主臂遥操作控制机械臂，PC 负责从从臂读取数据

通过 CAN 连接双臂机械臂：
```bash
# 确定是否能被 CAN 工具检测
bash third_party/piper_sdk/piper_sdk/find_all_can_port.sh

# 连接双臂（需要两个 CAN 接口）
bash third_party/piper_sdk/piper_sdk/can_muti_activate.sh
```

激活虚拟环境：
```bash
source examples/piper_dual/.venv/bin/activate
```

测试硬件连接：
```bash
python examples/piper_dual/test_env.py
```

#### 2. 相机 ID 查询

USB 相机 ID 查询：
```bash
ls /dev/video*
sudo apt install v4l-utils
sudo v4l2-ctl --list-devices
```

RealSense 相机序列号查询：
```bash
rs-enumerate-devices | grep "Serial Number"
```

#### 3. ROS 2 preview-first 数据收集

ROS 2 采集器只连接已经运行的官方 ROS 2 节点，负责预览同步帧并将每段轨迹保存为自包含 HDF5。它不会配置 CAN、不会初始化或使能物理机械臂；CAN 激活、相机节点和 Piper 节点仍必须按官方流程在独立终端中提前启动。

终端 1：启动三路 RealSense 彩色相机节点：
```bash
source /opt/ros/humble/setup.bash
source /home/agilex/camera_ros/install/setup.bash
bash /home/agilex/camera_ros/scripts/start_realsense_3cam_color.sh
```

终端 2：启动双臂 Piper ROS 2 硬件节点：
```bash
source /opt/ros/humble/setup.bash
source /home/agilex/piper_ros/install/setup.bash
cd /home/agilex/piper_ros
bash start_multi_piper.sh
```

终端 3：在 client Python 环境中运行采集器（默认安全 dry-run，不发布动作）：
```bash
cd /home/agilex/client
python examples/piper_dual/collect_data_ros2.py \
  --output-dir /home/agilex/piper_dual_dataset \
  --prompt "Fold the towel" \
  --config examples/piper_dual/ros2_piper_dual.yaml \
  --bridge-python /usr/bin/python3 \
  --dry-run
```

需要物理发布时，显式切换为：
```bash
cd /home/agilex/client
python examples/piper_dual/collect_data_ros2.py \
  --output-dir /home/agilex/piper_dual_dataset \
  --prompt "Fold the towel" \
  --config examples/piper_dual/ros2_piper_dual.yaml \
  --bridge-python /usr/bin/python3 \
  --no-dry-run \
  --publish-actions
```

参数说明：
- `--output-dir`: 保存 `episode_000000.hdf5`、`episode_000001.hdf5` 等文件的目录；录制过程中先写入同名 `.hdf5.partial`，成功 finalize 后原子发布为 `.hdf5`。
- `--prompt`: 写入每个 episode metadata 的任务指令。
- `--config`: ROS 2 bridge 配置文件，例如 `examples/piper_dual/ros2_piper_dual.yaml`。
- `--bridge-python`: 启动 ROS 2 bridge 的 Python，默认 `/usr/bin/python3`，用于使用 ROS 2 Humble 环境。
- `--jpeg-quality`: 写入 HDF5 metadata 的 JPEG 质量值。
- `--max-sync-error-ms`: 同步帧允许的最大传感器时间误差。
- `--dry-run`: 默认启用，保证采集器/bridge 不发布动作。
- `--publish-actions`: 默认关闭；只有明确需要且不使用 `--dry-run` 时才会启用动作发布。
- `--render-after-save`: 每次按 `s` finalize 后在 HDF5 旁边导出 `.preview.mp4` 三相机预览视频。

窗口会预览 `cam_high`、`cam_left_wrist`、`cam_right_wrist` 三路相机并显示状态与 prompt。按键说明：
- `s`: 在 PREVIEW 开始录制；在 RECORDING 时按 `s` 或 `e` 都会结束当前 episode、finalize/validate HDF5，可选渲染，然后回到 PREVIEW；如果当前 episode 为空或 HDF5 验证失败，采集器会显示失败原因、删除未发布的 `.hdf5.partial`，并回到 PREVIEW，不会关闭 ROS 2 backend。
- `e`: 在 PREVIEW 下无操作；仅在 RECORDING 时作为 `s` 的别名完成 finalize。
- `q`: 退出；如果有未 finalize 的 partial 文件会 abort，不会发布最终 HDF5。


每个 finalize 后的 HDF5 都是自包含文件，包含三路 JPEG 图像、14 维 state/action、时间戳、同步误差和采集 metadata，可独立复制和验证。它们的根属性固定包含 `schema_version`、`frame_count`、`jpeg_quality`、`metadata_json`、`camera_mapping_json`、`units_json`、`image_preprocessing_json` 和 `timing_counters_json`；结构检测不依赖文件名。

#### 4. ROS 2 EEF 数据收集（关节角度 + puppet 末端位姿）

EEF 采集器在现有关节 state/action 基础上，额外同步并保存左右 puppet 臂的 `geometry_msgs/msg/PoseStamped` 末端位姿。Piper ROS 节点默认话题为 `/puppet/end_pose_left` 和 `/puppet/end_pose_right`；默认位置单位为米，姿态转换为弧度制 `[x, y, z, roll, pitch, yaw]`。

```bash
cd /home/agilex/client/.worktrees/piper-dual-ros2-migration
source /opt/ros/humble/setup.bash
source /home/agilex/piper_ros/install/setup.bash
source .venv/bin/activate

python examples/piper_dual/collect_data_eef_ros2.py \
  --output-dir /home/agilex/piper_dual_dataset/fold_towel_eef \
  --prompt "Fold the towel" \
  --config examples/piper_dual/ros2_piper_dual.yaml \
  --bridge-python /usr/bin/python3 \
  --max-sync-error-ms 30 \
  --dry-run
```

可通过 `--eef-left-topic` 和 `--eef-right-topic` 覆盖 EEF 话题。三路相机、四路关节流和两路 EEF 必须共同满足 `--max-sync-error-ms`，否则该参考帧不会写入；不会填充 NaN 或另写独立 EEF 时间序列。

EEF episode 仍保持 `/observations/state` 和 `/action` 的 14 维关节格式，并额外包含：

```text
/observations/eef/puppet_left     (N, 6) float32  [x,y,z,roll,pitch,yaw]
/observations/eef/puppet_right    (N, 6) float32  [x,y,z,roll,pitch,yaw]
```

`metadata_json.eef.enabled` 标记 EEF 文件；对应 EEF 传感器的时间戳和同步误差位于 `/observations/sensor_timestamps`、`/observations/sync_error`。当前 `convert_ros2_piper_data_to_lerobot.py` 保持原行为，不会把这些原始 EEF 数据加入 LeRobot state。

将 EEF episode 转为 shifted LeRobot 数据集时，使用独立脚本；原始 HDF5 的 6D RPY 数据不会被修改：

```bash
cd /home/agilex/client/.worktrees/piper-dual-ros2-migration
.venv/bin/python examples/piper_dual/utils/convert_ros2_piper_data_to_lerobot_eef_shifted.py \
  --raw-dir /home/agilex/piper_dual_dataset/stack_cups_eef \
  --repo-id local/piper_dual_stack_cups_eef_shifted \
  --mode image
```

每个原始 EEF `[x,y,z,roll,pitch,yaw]` 按 ROS/tf2 的 XYZ RPY 约定构造 `R = Rz(yaw) Ry(pitch) Rx(roll)`，再按列保存 `rot6d=[r11,r21,r31,r12,r22,r32]`。LeRobot 的 `observation.state` 和 `action` 均为 20 维：

```text
[left_xyz, left_rot6d, left_gripper, right_xyz, right_rot6d, right_gripper]
```

时序映射与关节 shifted 转换器一致：`state[t]` 使用当前帧左右 EEF，`action[t]` 使用下一帧左右 EEF；两者的左右 gripper 均使用当前帧 master action 的 gripper。图像和 task 使用当前帧，每个 N 帧 episode 输出 N-1 帧。脚本支持 `--episodes 0,1,2` 和 `--overwrite`。原始 `convert_ros2_piper_data_to_lerobot.py` 与关节版 `convert_ros2_piper_data_to_lerobot_shifted.py` 均保持不变。

#### 官方 legacy data_tools 流程（raw capture / sync / HDF5 / replay）

来自 `/home/agilex/data_ros/src/data_tools/README.md` 的官方 legacy 命令仍然可用；本 checkout 里对应的 setup 入口是：

```bash
source /opt/ros/humble/setup.bash
source /home/agilex/data_ros/install/setup.bash
cd /home/agilex/data_ros/src/data_tools/scripts

# raw capture
ros2 launch data_tools run_data_capture.launch.py type:=aloha datasetDir:={data_path} episodeIndex:=0

# sync
python3 data_sync.py --type aloha --datasetDir {data_path}

# HDF5
python3 data_to_hdf5.py --type aloha --useCameraPointCloud "" --datasetDir {data_path} --useIndex "" --datasetTargetDir {hdf5_saving_path}

# replay / publish
ros2 launch data_tools run_data_publish.launch.py type:=aloha datasetDir:={data_path} episodeIndex:=0
python3 data_publish.py --type aloha --datasetDir {hdf5_path}
```

#### 4. 旧版直连数据收集

运行旧版直连数据采集脚本：
```bash
python examples/piper_dual/collect_data.py \
    --prompt "pick up the object" \
    --left_can_port can_left \
    --right_can_port can_right \
    --high_camera_id 148522073709 \
    --left_wrist_camera_id 6 \
    --right_wrist_camera_id 8
```

参数说明：
- `--prompt`: 任务指令文本
- `--left_can_port`: 左臂 CAN 端口（默认: can_left）
- `--right_can_port`: 右臂 CAN 端口（默认: can_right）
- `--high_camera_id`: RealSense 相机序列号（可选，默认使用第一个可用的）
- `--left_wrist_camera_id`: 左腕 USB 相机 ID（默认: 6）
- `--right_wrist_camera_id`: 右腕 USB 相机 ID（默认: 8）

也可以从预设任务中选择：
```bash
python examples/piper_dual/collect_data.py --task_type "pick" --prompt_index 0
```

运行后，程序会实时可视化三个摄像机视角，窗口下方显示任务 prompt。操作说明：
- 按 `s` 开始录制（窗口文字由绿变红）
- 按 `q` 停止录制并保存轨迹（文字由红变绿）
- 按 `ESC` 退出程序

录制的数据保存在 `./recorded_data_dual` 文件夹下。

#### 5. 数据集转化

将录制的 HDF5 格式转化为 LeRobot 格式：
```bash
export HF_LEROBOT_HOME="./datasets/piper_dual_lerobot"
uv run examples/piper_dual/utils/convert_piper_data_to_lerobot.py     --raw_dir ./recorded_data_dual     --repo_id piper_dual_lerobot
```

#### 6. 数据集可视化

使用 Rerun 可视化转化后的 LeRobot 数据集：
```bash
export HF_LEROBOT_HOME="./datasets/piper_dual_lerobot"
uv run examples/piper_dual/utils/vis_lerobot_datasets.py \
    --repo-id piper_dual_lerobot \
    --root ./datasets/piper_dual_lerobot/piper_dual_lerobot \
    --episode 0
```

#### 7. 自包含 HDF5 渲染

对 `piper_dual_ros2_v1` 自包含 HDF5，可以直接导出三个单独相机视频、固定顺序三视图视频和质量报告：

```bash
cd /home/agilex/client
python examples/piper_dual/render_dataset.py \
    --input /path/to/episode.hdf5 \
    --output-dir /path/to/rendered_episode \
    --fps 30 \
    --no-plots
```

输出会生成 `cam_high.mp4`、`cam_left_wrist.mp4`、`cam_right_wrist.mp4`、`views_3x1.mp4`、`quality.json`，并在允许绘图时额外生成 `state_action.png`。`quality.json` 对应 `render_episode()` 返回值中的 `schema_version`、`frame_count`、`image_decode_counts`、`errors`、`state_action_plot`、`video_paths` 和 `quality_path`。
自包含 HDF5 的根属性固定为 `schema_version`、`frame_count`、`jpeg_quality`、`metadata_json`、`camera_mapping_json`、`units_json`、`image_preprocessing_json` 和 `timing_counters_json`；`detect_schema()` 只会返回 `piper_dual_ros2_v1` 或 `legacy_official_hdf5`，损坏/混杂文件会直接报错。
`visualize_hdf5.py` 检测到 `piper_dual_ros2_v1` 时会自动转调到该渲染器；旧版 HDF5 仍保持原来的关节曲线和单相机视频逻辑。

#### 8. 三栈手动比较 runbook

任务 7 的手动比较只允许三个运行标签：`local-ros`、`official-ros`、`direct-sdk`。非空 `--run-tag` 必须与 `--control-stack` 解析结果完全一致；每个标签都写入自己的 `out_dir/<run-tag>` 目录，视频、图表、CSV 和 `run_metadata.json` 不会互相混放。

本节以 `main_dual.py` 为入口，`local-ros` 与 `official-ros` 通过 ROS 2 bridge 控制，`direct-sdk` 通过直连适配器控制。固定测试速度不同：`local-ros=30%`、`official-ros=30%`、`direct-sdk=100%`。这不是等速对比实验，不能把它写成公平的同速比较；每次运行的 `run_metadata.json` 都会记录 `effective_speed_percent`。

**本地 ROS 栈（local-ros，30%）**

终端 1：启动本地 ROS 2 控制栈，保持 `auto_enable:=false`：

```bash
cd /home/agilex/piper_ros
bash can_config.sh
source /opt/ros/humble/setup.bash
source /home/agilex/piper_ros/install/setup.bash
ros2 launch piper start_two_piper.launch.py \
  can_left_port:=can_left can_right_port:=can_right auto_enable:=false
```

终端 2：确认左右状态话题，再显式 enable 左右臂：

```bash
source /opt/ros/humble/setup.bash
source /home/agilex/piper_ros/install/setup.bash
ros2 topic echo --once /arm_status_left
ros2 topic echo --once /arm_status_right
ros2 service call /piper_left_ctrl_node/enable_srv piper_msgs/srv/Enable "{enable_request: true}"
ros2 service call /piper_right_ctrl_node/enable_srv piper_msgs/srv/Enable "{enable_request: true}"
```

终端 3：启动匹配标签的客户端：

```bash
cd /home/agilex/client/.worktrees/piper-dual-ros2-migration
.venv/bin/python examples/piper_dual/main_dual.py \
  --backend ros2 \
  --control-stack local-ros \
  --ros2-config examples/piper_dual/ros2_piper_dual_local.yaml \
  --run-tag local-ros \
  --fps 30 \
  --dry-run
```

确认 dry-run、状态 echo 和双臂 enable 后，才允许把最后一行替换为 `--no-dry-run --publish-actions`。

**官方 ROS 栈（official-ros，30%）**

终端 1：只 source 隔离的官方 install，并保持 `auto_enable:=false`：

```bash
cd /home/agilex/piper_ros/.worktrees/piper-official-humble
source /opt/ros/humble/setup.bash
source /home/agilex/piper_ros/.worktrees/piper-official-humble/install-official/setup.bash
ros2 launch piper_official_bringup start_two_piper_official.launch.py \
  can_left_port:=can_left can_right_port:=can_right auto_enable:=false
```

终端 2：确认官方左右状态话题，再显式 enable 左右臂：

```bash
source /opt/ros/humble/setup.bash
source /home/agilex/piper_ros/.worktrees/piper-official-humble/install-official/setup.bash
ros2 topic echo --once /arm_status_left
ros2 topic echo --once /arm_status_right
ros2 service call /piper_left_ctrl_node/enable_srv piper_msgs/srv/Enable "{enable_request: true}"
ros2 service call /piper_right_ctrl_node/enable_srv piper_msgs/srv/Enable "{enable_request: true}"
```

终端 3：启动匹配标签的客户端：

```bash
cd /home/agilex/client/.worktrees/piper-dual-ros2-migration
.venv/bin/python examples/piper_dual/main_dual.py \
  --backend ros2 \
  --control-stack official-ros \
  --ros2-config examples/piper_dual/ros2_piper_dual.yaml \
  --run-tag official-ros \
  --fps 30 \
  --dry-run
```

确认 dry-run、状态 echo 和双臂 enable 后，才允许把最后一行替换为 `--no-dry-run --publish-actions`。

**直连 SDK 栈（direct-sdk，100%）**

终端 1：source 官方 message install，加入直连 SDK 依赖路径，再启动适配器：

```bash
cd /home/agilex/client/.worktrees/piper-dual-ros2-migration
source /opt/ros/humble/setup.bash
source /home/agilex/piper_ros/.worktrees/piper-official-humble/install-official/setup.bash
export PYTHONPATH=/home/agilex/lgd/control_your_robot/src:$PYTHONPATH
/usr/bin/python3 examples/piper_dual/piper_direct_sdk_adapter.py \
  --left-can can_left --right-can can_right --allow-enable
```

终端 2：启动匹配标签的客户端：

```bash
cd /home/agilex/client/.worktrees/piper-dual-ros2-migration
.venv/bin/python examples/piper_dual/main_dual.py \
  --backend ros2 \
  --control-stack direct-sdk \
  --ros2-config examples/piper_dual/ros2_piper_dual_direct_sdk.yaml \
  --run-tag direct-sdk \
  --fps 30 \
  --dry-run
```

确认适配器、dry-run 和机械臂状态后，才允许把最后一行替换为 `--no-dry-run --publish-actions`。

**切换/停止顺序**

每次切换栈前都必须先停止客户端，再停止当前控制栈，然后检查残留进程和 CAN 状态；没有旧 owner 且 CAN 名称/bitrate 正确前，不允许启动下一栈。

```bash
# Stop policy/client first with Ctrl-C.
# Stop the active ROS launch or direct adapter with Ctrl-C.
pgrep -af 'piper_single_ctrl|piper_direct_sdk_adapter|main_dual.py|ros2_bridge_process'
ip -details link show type can
```

`run_metadata.json` 必须记录 `control_stack`、`run_tag`、`prompt`、policy host/port、runtime/config 选项、`video_fps`、`effective_speed_percent`、开始/结束时间与退出码/原因；三个栈都必须把自己的 run metadata 留在各自的 run-tag 子目录中。




### 三、模型部署

#### 1. 连接硬件
**请在以下操作前将双臂机械臂掰回初始状态。**

注意：部署时只给从臂接入电源，主臂需要断电，否则无法控制机械臂。

```bash
# 检测 CAN 设备
bash third_party/piper_sdk/piper_sdk/find_all_can_port.sh

# 连接双臂
bash third_party/piper_sdk/piper_sdk/can_muti_activate.sh
```

#### 2. 启动推理服务器

在服务器端启动 ZR-0 模型服务：
```bash
cd ZR-0
python server.py \
    --dataset_entry piper_dual \
    --ckpt_dir /path/to/your/checkpoint \
    --port 8000
```

#### 3. 运行客户端

**方式一：远程推理模式（连接推理服务器）**
```bash
source examples/piper_dual/.venv/bin/activate

python examples/piper_dual/main_dual.py \
    --mode remote \
    --host 0.0.0.0 \
    --port 8000 \
    --prompt "pick up the object" \
    --left_can_port can_left \
    --right_can_port can_right \
    --high_camera_id 148522073709 \
    --left_wrist_camera_id 6 \
    --right_wrist_camera_id 8
```

#### 4. ROS 2 专用部署入口（默认 dry-run）

`main_dual_ros.py` 只负责 ROS 2 bridge 和远程 PI05 server，不包含 SDK、CAN 或 USB/RealSense 设备初始化。相机节点和 Piper ROS 2 节点必须先在独立终端启动。

预览和同步验证时保持 dry-run：
```bash
cd /home/agilex/client/.worktrees/piper-dual-ros2-migration
source .venv/bin/activate
python examples/piper_dual/main_dual_ros.py \
  --host 127.0.0.1 \
  --port 8000 \
  --bridge-python /usr/bin/python3 \
  --ros2-config examples/piper_dual/ros2_piper_dual.yaml \
  --prompt "Fold the towel" \
  --dry-run
```

只有完成相机话题、Piper 节点、CAN、急停和低风险动作策略检查后，才显式允许动作发布：
```bash
cd /home/agilex/client/.worktrees/piper-dual-ros2-migration
source .venv/bin/activate
python examples/piper_dual/main_dual_ros.py \
  --host 127.0.0.1 \
  --port 8000 \
  --bridge-python /usr/bin/python3 \
  --ros2-config examples/piper_dual/ros2_piper_dual.yaml \
  --prompt "Fold the towel" \
  --no-dry-run \
  --publish-actions
```

安全约束：
- `--dry-run` 是默认值；它不会发布物理动作。
- `--publish-actions` 必须与 `--no-dry-run` 同时提供，否则入口在创建环境前退出。
- 客户端不会启动、使能或复位真实机械臂；官方 ROS 2 节点必须预先运行。
- `Ros2DualEnvironment` 发生致命故障后会锁定当前实例的动作发布，必须创建新实例。


#### 5. EEF rot6d ROS 2 专用部署入口

`main_dual_eef_ros.py` 使用 20D EEF policy：

```text
[left_xyz, left_rot6d, left_gripper,
 right_xyz, right_rot6d, right_gripper]
```

其中 `rot6d` 按列存储旋转矩阵前两列；client 使用 Gram-Schmidt 恢复旋转矩阵并转为 Piper `PosCmd` 的 `[x,y,z,roll,pitch,yaw,gripper]`。EEF 观测话题类型为 `geometry_msgs/msg/PoseStamped`，动作话题类型为 `piper_msgs/msg/PosCmd`。

先运行 dry-run 验证策略输入和同步帧：

```bash
cd /home/agilex/client/.worktrees/piper-dual-ros2-migration
source /opt/ros/humble/setup.bash
source /home/agilex/camera_ros/install/setup.bash
source /home/agilex/piper_ros/install/setup.bash
source .venv/bin/activate

python examples/piper_dual/main_dual_eef_ros.py \
  --host 127.0.0.1 \
  --port 8000 \
  --bridge-python /usr/bin/python3 \
  --ros2-config examples/piper_dual/ros2_piper_dual.yaml \
  --prompt "Stack_the_paper_cups_together." \
  --dry-run
```

确认策略输出、EEF 姿态、动作限幅和急停后，才允许真实 PosCmd 发布：

```bash
python examples/piper_dual/main_dual_eef_ros.py \
  --host 127.0.0.1 \
  --port 8000 \
  --bridge-python /usr/bin/python3 \
  --ros2-config examples/piper_dual/ros2_piper_dual.yaml \
  --prompt "Stack_the_paper_cups_together." \
  --no-dry-run \
  --publish-actions \
  --max-action-delta 0.05
```

默认 topic：

```text
EEF observation: /puppet/end_pose_left, /puppet/end_pose_right
EEF action:      /pos_left_cmd, /pos_right_cmd
```

动作发布开关、故障锁定和 cleanup 行为与关节入口一致；入口不会启动、使能或复位机械臂。建议先用 `ros2 topic type` 验证上述四个 topic 的消息类型，再进行真实发布。

#### 6. EEF XYZ+RPY ROS 2 专用部署入口

`main_dual_eef_xyz3d_ros.py` 使用 14D EEF policy：

```text
[left_xyz, left_rpy, left_gripper,
 right_xyz, right_rpy, right_gripper]
```

其中位置和夹爪使用米，roll、pitch、yaw 使用弧度。服务端使用 checkpoint 匹配的 14D 归一化统计，并将模型内部 32D 输出裁剪成 14D：

```bash
cd /home/agilex/client/.worktrees/piper-dual-ros2-migration
source .venv/bin/activate
python scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_piper_dual_stack_cups_eef_xyz3d \
  --policy.dir=/home/agilex/checkpoints/pi05_piper_dual_stack_cups_eef_xyz3d_jax_step10000_pytorch
```

XYZ3D 服务端会保留客户端传入的 `rtc_obs`，将其中绝对 EEF 前缀按 action 统计归一化、补齐到模型维度，再交给 RTC 引导；不会转换成关节增量或相对位姿。基础 XYZ3D 和 `_100` 配置共用该适配器。没有 `rtc_obs` 的首次推理或关闭 RTC 的请求仍走普通采样。

更新服务端适配器代码后，需要先停止推理客户端，再停止并用原 checkpoint/config 命令重启 `scripts/serve_policy.py`，仅重启客户端不会让旧服务加载新代码。随后客户端使用 `--use-async --use-rtc`；已有默认值也可继续使用。PyTorch 服务端从带前缀的后续请求开始输出 `overlap part error:`，可辅助确认进入了 RTC 分支，但该数值不是机械臂跟踪误差或平滑性保证。首次启用真实 RTC 后重新记录短时运行，检查推理耗时、chunk 边界和动作跳变；无需修改模型权重或重建 ROS 驱动。

先运行 dry-run 检查同步帧、14D 策略输出和动作变化：

```bash
source /opt/ros/humble/setup.bash
source /home/agilex/camera_ros/install/setup.bash
source /home/agilex/piper_ros/install/setup.bash
cd /home/agilex/client/.worktrees/piper-dual-ros2-migration
source .venv/bin/activate
python examples/piper_dual/main_dual_eef_xyz3d_ros.py \
  --host 127.0.0.1 \
  --port 8000 \
  --bridge-python /usr/bin/python3 \
  --ros2-config examples/piper_dual/ros2_piper_dual.yaml \
  --prompt "Stack_the_paper_cups_together." \
  --action-horizon 50 \
  --dry-run
```

确认 dry-run 输出、动作限幅和急停后，才可把 `--dry-run` 替换成 `--no-dry-run --publish-actions`。默认 EEF observation/action topic、故障锁定和 cleanup 行为与 rot6d EEF 入口一致。

#### 7. XYZ3D RTC 轨迹诊断与驱动精度回退

客户端记录默认关闭。给 XYZ3D 推理命令追加 `--trace-dir /home/agilex/eef_traces/run_sdk_001` 后，会在构造机器人环境前独占创建 `client_metadata.json` 和 `client.jsonl`；已有同名文件时拒绝启动，不会覆盖旧实验。每次实验使用新目录。`client.jsonl` 的 `action_submitted` 是 `environment.apply_action` 返回后的策略动作，不代表机械臂已经执行；dry-run 也会记录，因此必须结合 metadata 中的发布开关和 ROS 记录判断。

RTC `_chunk_trace` 包含 `chunk_id`、原始 chunk 内的 `chunk_step`、`chunk_boundary`、`skipped_steps`、选择动作时的双时间戳、推理耗时和 RTC 参数。它在选择该动作时确定，不会被异步线程之后的 chunk 切换覆盖。`observation_state` 是动作选择前的 EEF 观测，不是同步的关节跟踪误差。

本地驱动 `/home/agilex/piper_ros/src/piper/piper/piper_ctrl_single_node.py` 的 `eef_position_quantization` 启动参数支持：

- `sdk`（默认）：`round(position_m * 1_000_000)`，保留 SDK 的 0.001 mm 指令分辨率；不是机械臂实际定位精度承诺。
- `legacy_mm`：严格恢复原先 `round(position_m * 1000) * 1000` 的整毫米取整，便于对照或回退。
- `eef_command_trace:=true`：发布左右 `std_msgs/msg/String` 诊断话题，默认关闭。不改 MOVE P、速度、姿态或夹爪控制参数。

这两个参数在启动时读取。只修改源码不会更新已运行的进程。需要更新安装包时，在系统 ROS Python 环境中执行 `colcon build --packages-select piper`；它只构建安装，不负责停止或重启驱动。安全停机后，由操作者停止原控制 launch，确认没有重复 CAN owner，再重新启动。以下示例禁止自动使能，后续使能沿用已验证的人工流程：

```bash
source /opt/ros/humble/setup.bash
source /home/agilex/piper_ros/install/setup.bash
ros2 launch piper start_two_piper.launch.py \
  can_left_port:=can_left can_right_port:=can_right \
  auto_enable:=false \
  eef_position_quantization:=sdk eef_command_trace:=true
```

恢复旧取整方式：安全停止推理和驱动，再用同样的启动命令把 `eef_position_quantization:=sdk` 改为 `eef_position_quantization:=legacy_mm`。保持其他参数不变并使用新的记录目录；不要依赖运行时 `ros2 param set` 切换。完全关闭诊断则使用 `eef_command_trace:=false`，并去掉客户端 `--trace-dir`。

先在独立终端启动只读 ROS 采集器，再运行推理。采集器不调用机械臂驱动、不连接 CAN、不发布控制指令、不使能机械臂：

```bash
source /opt/ros/humble/setup.bash
source /home/agilex/piper_ros/install/setup.bash
cd /home/agilex/client/.worktrees/piper-dual-ros2-migration
/usr/bin/python3 examples/piper_dual/record_eef_ros2_trace.py \
  --output-dir /home/agilex/eef_traces/run_sdk_001
```

看到 `READY` 后，在原推理命令末尾追加同一个目录：

```text
--trace-dir /home/agilex/eef_traces/run_sdk_001
```

结束时先停止推理，等待反馈到达后再 Ctrl-C 停止采集器；它会 flush 并输出各路计数和缺失流。也可加 `--duration 120` 限时录制，但时长包含等待推理启动的时间。

目录中的 `ros.jsonl` 同时记录八路数据：

| 话题 | 记录内容 |
| --- | --- |
| `/pos_left_cmd`、`/pos_right_cmd` | 原始 XYZ+RPY、夹爪、mode 字段；单位 m/rad/m |
| `/piper_left_ctrl_node/eef_command_trace`、`/piper_right_ctrl_node/eef_command_trace` | 驱动收到的输入、精度模式、实际传给 `EndPoseCtrl` 的六个整数、序号与接收时间；SDK XYZ 单位 0.001 mm，角度单位 0.001 度 |
| `/puppet/joint_left`、`/puppet/joint_right` | 原始关节位置、速度、effort、名称和 header 时间戳 |
| `/puppet/end_pose_left`、`/puppet/end_pose_right` | 原始末端位置、四元数、header 时间戳 |

诊断中的 `dispatched=true` 只表示调用了 `EndPoseCtrl`，不是 SDK/CAN 成功或实际运动确认；false 时 `sdk_pose` 是计算出的候选值，未下发。关闭使能时也可收到此诊断。若没发现诊断发布者或结束时没有有效诊断消息，采集器会明确警告，不能视为完整捕获。

同机数据可用 `timestamp_ns`（Unix wall clock）和 `monotonic_ns` 对齐；带 header 的反馈额外保留 `source_timestamp_ns`，生产者时钟可能不同。PosCmd 本身无 header/唯一编号，只能结合时序和数值匹配，不能虚构跨层一一对应。订阅 QoS 为 best-effort，兼容可靠及 best-effort 发布者；结束计数不证明 DDS 零丢包。诊断序号可辅助发现缺口。原始 EEF 与关节数据必须分别分析，不可直接相减。

`--max-action-delta` 仍是原有 14D 原始数值相邻差的拒绝阈值，不是平滑器；XYZ 使用米、RPY 使用弧度，`20` 不是 20 mm。采集功能不会改变该阈值或任何现有安全检查。

#### 8. RTC 姿态误差双方案 A/B 实测

服务端新增 `--rtc-orientation-mode`，必须放在 `policy:checkpoint` 前面。默认 `legacy` 保持原公式，便于回退和对照；需要显式选择新方案：

| 模式 | 姿态误差 | 限制 |
| --- | --- | --- |
| `legacy` | 归一化 RPY 普通向量差 | 不理解角度周期，仅作为原实现对照 |
| `wrapped-rpy`（方案一） | 用 checkpoint action 统计还原弧度差，`atan2(sin(delta),cos(delta))` 后换回归一化单位 | 处理逐轴 2π 跨界，不解决耦合欧拉角等价表示或奇异性 |
| `so3`（方案二） | RPY 转四元数，对最短旋转角的平方损失求物理 RPY 梯度，再除以归一化比例 | 损失尊重完整旋转等价性；更新仍使用预测 RPY 坐标，在万向锁处退化；恰好 π 时方向不唯一 |

两种新模式只作用于双臂 RPY 六个维度，XYZ、夹爪、padding、RTC 时间权重及执行调度不变。它们都不反向传播整个去噪网络，SO(3) 只对小规模姿态几何计算求导。仅支持 PyTorch + `LeRobotPiperEefXyz3dDataConfig` 的绝对 14D XYZ+RPY checkpoint；错误配置或无效姿态归一化统计会报错，不会静默切换方案。无需重新训练、转换权重或重建 ROS 驱动。它们不是最终输出滤波器，不能保证消除模型原始错误或 RTC 权重为零处的姿态尖峰。

先停止客户端，再停止原模型服务，按方案一启动（checkpoint/config 必须与任务匹配）：

```bash
cd /home/agilex/client/.worktrees/piper-dual-ros2-migration
source .venv/bin/activate
python scripts/serve_policy.py \
  --port=8000 \
  --rtc-orientation-mode wrapped-rpy \
  policy:checkpoint \
  --policy.config=pi05_piper_dual_stack_cups_eef_xyz3d_100 \
  --policy.dir=/home/agilex/formal_expriments/PyToch_checkpoints/pi05_piper_dual_stack_cups_eef_xyz3d_100_step30000_pytorch
```

方案二使用同一命令，仅把 `wrapped-rpy` 改为 `so3`。回退时改为 `legacy`。每次切换必须重启服务，不能仅重启客户端；不要同时启动两个占用同一端口的服务。

每段实验先启动只读采集器，等待 `READY`，然后运行原 XYZ3D 客户端。采集器 `--output-dir` 与客户端 `--trace-dir` 使用同一个新目录，例如方案一 `/home/agilex/eef_traces/run_wrapped_rpy_001`，方案二 `/home/agilex/eef_traces/run_so3_001`。保留 `--use-async --use-rtc --fps 30 --action-horizon 20 --actions-during-latency 12`，同任务、同 checkpoint、尽量一致的初始场景。先 dry-run 用独立目录确认，再做人工看护下的短时实测；不要同时调整速度、动作限幅或其他控制参数。需要完整 SDK 参数时，在安全启动驱动时设置 `eef_command_trace:=true`。

客户端连接服务后，在 `client.jsonl` 首部即时 flush 一条 `server_metadata` 事件，保存服务端报告的 `rtc_orientation.mode`、`policy_config`、`checkpoint_dir`，以及新方案采用的姿态索引、归一化比例、偏移和梯度语义。初始 `client_metadata.json` 保持独占且不重写。空的服务端 metadata 表示未知来源，不能推断成 legacy；目录名也不是实际方案证据。后续按服务端 provenance、真实姿态角差、chunk 边界和耗时比较，不把 SDK 调用或目标跳变当作机械臂已经执行。

现有服务端 `overlap part error:` 仍打印归一化动作的普通向量差，未改成旋转距离。等价角度在新模式下可以保留不同数值分支，因此不要用该日志的大小比较三种模式优劣；应使用记录姿态之间的 SO(3) 角差、实测关节响应、任务结果及延迟。SO(3) 模式也未改变模型输出的 RPY 表示，不能保证在欧拉角奇异位形或所有模型异常上都优于周期角度方案。

### 六、文件结构

```
examples/piper_dual/
├── README.md                 # 本文档
├── cameras.py                # 相机封装（RealSense + USB）
├── collect_data.py           # 数据采集脚本
├── collect_data_with_intervention.py  # 带干预的数据采集
├── env_dual.py               # 双臂环境类
├── main_dual.py              # 双臂模型部署入口
├── piper_dual_controller.py  # 双臂控制器
├── saver.py                  # 视频保存器
├── test_env.py               # 环境测试脚本
└── utils/
    ├── convert_piper_data_to_lerobot.py  # 数据格式转换
    └── vis_lerobot_datasets.py           # 数据可视化
```

### 六、故障排除

1. **相机连接问题**
   - 检查 USB 连接是否牢固
   - 使用 `v4l2-ctl --list-devices` 确认相机 ID
   - 尝试更换 USB 端口（使用主板直连端口而非 HUB）

2. **CAN 总线问题**
   - 运行 `bash third_party/piper_sdk/piper_sdk/find_all_can_port.sh` 检测
   - 确保 CAN 接口已正确激活
   - 检查机械臂电源连接

3. **机械臂不响应**
   - 确认从臂已上电，主臂已断电（部署时）
   - 检查 `tele_mode` 参数设置
   - 重启机械臂后重新运行 CAN 激活脚本

### XYZ3D 推理的 T 形调试录像（2026-09-20）

`main_dual_eef_xyz3d_ros.py` 默认将当前客户端使用的三路同步观测写到
`--out-dir`：上方整幅主视角，下方左腕/右腕，RGB 不翻转、不交换左右。
每路源图为模型观测分辨率（通常 224×224），拼接输出为 448×672；不是原始
1280×720 高清录像，也不是独立以 30 Hz 连续采集的相机录像。

沿用原有启动命令，增加或修改这两个路径即可（每次 trace 目录必须全新）：

```bash
--out-dir /home/agilex/piper_deploy/runs/<本次运行>/videos \
--trace-dir /home/agilex/piper_deploy/runs/<本次运行>/trace
```

每个 episode 独立生成 `tshape_<时间>_<唯一ID>.mp4` 和同名 `.jsonl`。
视频播放速率跟随 `--fps`；JSONL 保存视频帧号、episode 内 step、三路源时间戳、
主机 wall/monotonic 时间、同步误差、观测 state、提交的 action 和录像丢帧计数。
视频帧来自动作提交成功后的 runtime 回调（dry-run 中代表提交到禁止动作发布的
环境），不能视为硬件已执行动作；拒绝的动作及推理等待期间没有额外视频帧。
若实际推理低于设定 FPS 或出现停顿，MP4 时长会短于真实时长，请按 JSONL
时间戳和 trace 排查，不要据视频播放速度估算机械臂速度。

编码使用后台线程和有界队列，不在动作循环等待编码；队列满时丢弃录像帧并
记录数量。正常结束、Ctrl-C、SIGTERM、Python 异常都会收尾已排队的视频；
SIGKILL、断电和编码/磁盘故障不保证 MP4 完整。原有其他部署入口的录像不变。
使用 `/home/agilex/piper_deploy/04_client.sh` 时，其已有 `--out-dir` 会自动接收
上述视频，不需要改相机/机械臂启动方式。此改动只在下一次启动客户端时加载。
