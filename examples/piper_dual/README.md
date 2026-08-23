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
- `s`: 在 PREVIEW 开始录制；在 RECORDING 再按一次 `s` 结束当前 episode、finalize/validate HDF5，可选渲染，然后回到 PREVIEW；如果当前 episode 为空或 HDF5 验证失败，采集器会显示失败原因、删除未发布的 `.hdf5.partial`，并回到 PREVIEW，不会关闭 ROS 2 backend。
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
