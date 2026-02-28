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

#### 3. 数据收集

运行数据采集脚本：
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

#### 4. 数据集转化

将录制的 HDF5 格式转化为 LeRobot 格式：
```bash
export HF_LEROBOT_HOME="./datasets/piper_dual_lerobot"
uv run examples/piper_dual/utils/convert_piper_data_to_lerobot.py     --raw_dir ./recorded_data_dual     --repo_id piper_dual_lerobot
```

#### 5. 数据集可视化

使用 Rerun 可视化转化后的 LeRobot 数据集：
```bash
export HF_LEROBOT_HOME="./datasets/piper_dual_lerobot"
uv run examples/piper_dual/utils/vis_lerobot_datasets.py \
    --repo-id piper_dual_lerobot \
    --root ./datasets/piper_dual_lerobot/piper_dual_lerobot \
    --episode 0
```

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

### 四、文件结构

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

### 五、故障排除

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
