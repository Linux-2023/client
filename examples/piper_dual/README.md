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
- `--render-after-save`: 每次按 `e` finalize 后在 HDF5 旁边导出 `.preview.mp4` 三相机预览视频。

窗口会预览 `cam_high`、`cam_left_wrist`、`cam_right_wrist` 三路相机并显示状态与 prompt。按键说明：
- `s`: 从 PREVIEW 进入 RECORDING；采集器先执行 ROS 2 同步缓存清空 barrier，然后打开下一个唯一 `.hdf5.partial` 文件。
- `e`: 结束当前 episode，停止追加帧，finalize/validate HDF5，可选渲染，然后回到 PREVIEW，可继续按 `s` 录制下一段；如果当前 episode 为空或 HDF5 验证失败，采集器会显示失败原因、删除未发布的 `.hdf5.partial`，并回到 PREVIEW，不会关闭 ROS 2 backend。
- `q`: 退出；如果有未 finalize 的 partial 文件会 abort，不会发布最终 HDF5。

每个 finalize 后的 HDF5 都是自包含文件，包含三路 JPEG 图像、14 维 state/action、时间戳、同步误差和采集 metadata，可独立复制和验证。它们的根属性固定包含 `schema_version`、`frame_count`、`jpeg_quality`、`metadata_json`、`camera_mapping_json`、`units_json`、`image_preprocessing_json` 和 `timing_counters_json`；结构检测不依赖文件名。

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

#### 4. ROS 2 安全后端（默认 dry-run）

`main_dual.py` 支持安全选择 ROS 2 后端。预览阶段请保持 dry-run：

```bash
cd /home/agilex/client
python examples/piper_dual/main_dual.py \
  --backend ros2 \
  --bridge-python /usr/bin/python3 \
  --ros2-config examples/piper_dual/ros2_piper_dual.yaml \
  --dry-run
```

需要显式物理发布时，先完成安全检查，再使用：
```bash
cd /home/agilex/client
python examples/piper_dual/main_dual.py \
  --backend ros2 \
  --bridge-python /usr/bin/python3 \
  --ros2-config examples/piper_dual/ros2_piper_dual.yaml \
  --no-dry-run \
  --publish-actions
```

此命令只在确认相机、CAN、急停和低风险动作策略后使用；否则保持 `--dry-run`。

安全约束：
- `sdk` 仍然是默认后端，旧流程不变。
- ROS 2 默认不会发布动作；只做预览和同步帧验证。
- 只有显式提供 `--backend ros2 --no-dry-run --publish-actions` 才允许动作发布。
- 一旦 `Ros2DualEnvironment` 在 `reset()` 阶段的 `start()` / `clear_buffers()` 失败，或在观测/动作阶段触发致命故障，同一个实例会永久锁定动作发布；即使 `reset()` 也不会恢复，必须创建新的环境/bridge 实例才能重新发布动作。
- 如果只是验证同步和观察映射，请保持 `--dry-run`。
- 不要在客户端脚本里启用或复位真实机械臂；ROS 2 bridge 和机械臂节点必须先按官方流程在独立终端启动。

### 五、文件结构

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
