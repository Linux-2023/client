# 准备活动

1. 开启电小二电源（连接机械臂电源，开启后机械臂可以开始主从臂控制）
2. 打开底座的开关（连接显示器电源，打开后显示器正常打开）

## 查看RealSense相机画面

```
realsense-viewer
```
备注三个相机的序列号
```
左臂：254622072400
右臂：254522074588
中间：151222073259
```
## 相机启动
在终端运行命令
```
source /opt/ros/humble/setup.bash 

source /home/agilex/camera_ros/install/setup.bash

bash /home/agilex/camera_ros/scripts/start_realsense_3cam_color.sh
```
成功后输出类似：
```
[realsense2_camera_node-3] [INFO] [1786782573.887138195] [camera_r]: Open profile: stream_type: Color(0), Format: RGB8, Width: 1280, Height: 720, FPS: 30
[realsense2_camera_node-3] [INFO] [1786782573.889615666] [camera_r]: RealSense Node Is Up!
```
可以开启一个新的终端通过
```
ros2 topic list
```
查看是否订阅
```
...
/camera_f_l/color/image_raw
...
/camera_l/color/image_raw
...
/camera_r/color/image_raw
```

## 使能机械臂CAN口

新建终端，查看当前插入的CAN模块

```
ip -br link | grep can
```
正常是三个CAN接口（can0,can1,can2），分别为底座，右臂，左臂

切换到`cd /home/agilex/piper_ros/`

运行查询和配置脚本
```
bash find_all_can_port.sh 
bash can_config.sh
```
显示重新命名（can_left,can_right）并成功激活CAN口

## 启动机械臂

旧 `start_multi_piper.sh`（piper_sdk + piper_ros 节点）已弃用，底层 SDK 换成 `third_party/pyAgxArm`，改用本仓库自带的 pyAgxArm 双臂 ROS 桥启动：

```
cd /home/agilex/client
bash examples/piper_dual_pyagxarm/run_driver.sh --enable --motion-mode js
```

脚本内部已自动 `source` ROS 环境，并固定 `--model piper --left-firmware v189 --right-firmware v189`（S-V1.9-1 固件）。
- `--enable`：循环使能电机，直到六个电机反馈全部已使能；急停后需用 `--enable --reset-stop` 恢复
- `--motion-mode js`：官方 `move_js()`（MIT 快速响应）；默认 `j` 为 `move_j()`（MOVE J 带轨迹平滑）
- `--startup-timeout` 默认 10 秒，只限制启动等待

可以开启一个新的终端通过
```
ros2 topic list
```
查看是否订阅（主臂和从臂）
```
...
/piper_left_ctrl_node/arm_status
...
/piper_right_ctrl_node/arm_status
...
/puppet/joint_left
/puppet/joint_right
...
/master/joint_left
/master/joint_right
...
/puppet/end_pose_left
/puppet/end_pose_right
...
```
注意：本桥接的 `/master/joint_*` 是命令回显（无命令时为实际位置），不是独立主臂测量。
# 官方提供的命令（采集数据流程）
## 采集一条数据（原始格式）

新建一个终端，运行
```
source /opt/ros/humble/setup.bash 

source /home/agilex/camera_ros/install/setup.bash

ros2 launch data_tools run_data_capture.launch.py \
  type:=aloha \
  datasetDir:=/home/agilex/lgd_data \
  episodeIndex:=0
```

## 数据格式转换（HDF5）

新建终端，激活conda环境，并切换到新的目录
```
conda activate data
cd /home/agilex/data_ros/src/data_tools/scripts
```
同步数据集（？）
```
python3 data_sync.py \
  --type aloha \
  --datasetDir /home/agilex/lgd_data
```
运行数据格式转换脚本
```
python3 data_to_hdf5.py \
  --type aloha \
  --datasetDir /home/agilex/lgd_data
```
## 可视化数据

当前仓库未提供 `visualize_ros2_episodes.py`，可直接用下面的 `render_hdf5_batch.sh` 渲染三视角视频：
```
cd /home/agilex/client
source .venv/bin/activate

./examples/piper_dual/render_hdf5_batch.sh \
  --input-dir /home/agilex/lgd_data \
  --output-dir /home/agilex/visualizations/episodes_0 \
  --fps 30
```

## 重放采集的数据

```
source /opt/ros/humble/setup.bash
source /home/agilex/data_ros/install/setup.bash
conda activate data

python3 /home/agilex/data_ros/src/data_tools/scripts/data_publish.py \
  --datasetDir /home/agilex/lgd_data/episode0/episode0.hdf5 \
  --publish_rate 30
```

# 优化版采数据流程

> 本机驱动发布的是 `local-ros` 栈话题，因此以下采集/部署命令必须显式使用 `--control-stack local-ros` 并配套 `ros2_piper_dual_local.yaml`；默认的 `official-ros` 栈在本机无法通过契约校验。

开启采集数据
新开一个终端
```
source /opt/ros/humble/setup.bash
source /home/agilex/piper_ros/install/setup.bash
cd /home/agilex/client
source .venv/bin/activate
```
```
## 不带eef数据
python examples/piper_dual/collect_data_ros2.py \
  --output-dir /home/agilex/piper_dual_dataset/fold_towel \
  --prompt "Fold the towel" \
  --config examples/piper_dual/ros2_piper_dual_local.yaml \
  --control-stack local-ros \
  --bridge-python /usr/bin/python3 \
  --max-sync-error-ms 30
```
采集带eef的数据（同时记录 puppet 与 master EEF）
```
source /opt/ros/humble/setup.bash
source /home/agilex/piper_ros/install/setup.bash
cd /home/agilex/client
source .venv/bin/activate

python examples/piper_dual/collect_data_eef_ros2.py \
  --output-dir /home/agilex/piper_dual_dataset/stack_cups_eef \
  --prompt "Stack_the_paper_cups_together." \
  --config examples/piper_dual/ros2_piper_dual_local.yaml \
  --control-stack local-ros \
  --bridge-python /usr/bin/python3 \
  --max-sync-error-ms 30
```

将数据转换为lerobot格式（只有主臂action和从臂state关节角度，无eef）
```
export HF_LEROBOT_HOME=/home/agilex/lgd/lerobot_datasets/
cd /home/agilex/client
source .venv/bin/activate

.venv/bin/python examples/piper_dual/utils/convert_ros2_piper_data_to_lerobot.py \
  --raw-dir /home/agilex/piper_dual_dataset/stack_cups \
  --repo-id local/piper_dual_stack_cups_ros2 \
  --mode image
```
从臂(state+action)加主臂夹爪的数据格式转换（关节角度）
```
export HF_LEROBOT_HOME=/home/agilex/lgd/lerobot_datasets/
cd /home/agilex/client
source .venv/bin/activate

.venv/bin/python examples/piper_dual/utils/convert_ros2_piper_data_to_lerobot_shifted.py \
  --raw-dir /home/agilex/piper_dual_dataset/stack_cups \
  --repo-id local/piper_dual_stack_cups_ros2_shifted \
  --mode image
```
eef数据格式转换

第一种转换为xyz+rot6d（非原生控制）
```
export HF_LEROBOT_HOME=/home/agilex/lgd/lerobot_datasets/
cd /home/agilex/client
source .venv/bin/activate

.venv/bin/python examples/piper_dual/utils/convert_ros2_piper_data_to_lerobot_eef_shifted.py \
  --raw-dir /home/agilex/piper_dual_dataset/stack_cups_eef \
  --repo-id local/piper_dual_stack_cups_eef_rot6d_shifted \
  --mode image
```
第二种转换为xyz+rot3d（原生控制）
```
export HF_LEROBOT_HOME=/home/agilex/lgd/lerobot_datasets/
cd /home/agilex/client
source .venv/bin/activate

.venv/bin/python examples/piper_dual/utils/convert_ros2_piper_data_to_lerobot_eef_xyz3d_shifted.py \
  --raw-dir /home/agilex/piper_dual_dataset/stack_cups_eef \
  --repo-id local/piper_dual_stack_cups_eef_xyz3d_test \
  --mode image
```

第三种转换为xyz+rot3d，action 使用当前帧 master 关节向量（不丢帧）
```
export HF_LEROBOT_HOME=/home/agilex/lgd/lerobot_datasets/
cd /home/agilex/client
source .venv/bin/activate

.venv/bin/python examples/piper_dual/utils/convert_ros2_piper_data_to_lerobot_eef_xyz3d_master_action.py \
  --raw-dir /home/agilex/piper_dual_dataset/stack_cups_eef \
  --repo-id local/piper_dual_stack_cups_eef_xyz3d_master_action \
  --mode image
```

上传到HF的命令
```
export HF_TOKEN='hf_你的新Token'

cd /home/agilex/.cache/huggingface/lerobot/local

env -u ALL_PROXY -u all_proxy python -c '
import os
from huggingface_hub import HfApi

repo_id = "HITdongdong/piper_dual_stack_cups_ros2"
folder = "piper_dual_stack_cups_ros2"
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(
    repo_id=repo_id,
    repo_type="dataset",
    private=False,
    exist_ok=True,
)
result = api.upload_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path=folder,
    allow_patterns=["data/**", "meta/**"],
    commit_message="Upload LeRobot dataset",
)
print(result)
'
```

# 模型部署流程

从HF上拉取训练好的模型到本地
服务器上checkpoint保存路径：
```
-Stack_cups：                                                                             
   /pfs/pfs-7jnepv/lgd/SANE/checkpoints/pi05_piper_dual_stack_cups_eef_xyz3d_100/stack_cups_eef_xyz3d_100_50k_20260825T064651Z/                                                     
 -Fold_towel：                                                                             
   /pfs/pfs-7jnepv/lgd/SANE/checkpointspi05_piper_dual_fold_towel_eef_xyz3d_100/fold_towel_eef_xyz3d_100_50k_20260825T070133Z/ 
```

先将JAX模型转换为PyTorch模型
```
cd /home/agilex/client
source .venv/bin/activate

python examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir /home/agilex/checkpoints/piper_dual_stack_cups_pi05_25000_upload \
  --config_name pi05_piper_dual_stack_cups \
  --output_path /home/agilex/checkpoints/piper_dual_stack_cups_pi05_25000_pytorch \
  --precision bfloat16
```
> 注意：`--config_name` 必须与训练数据集/任务一致，例如 eef_xyz3d 数据集对应 `pi05_piper_dual_*_eef_xyz3d_100`。

复制一下归一化
```
cp -a \
  /home/agilex/checkpoints/piper_dual_stack_cups_pi05_25000_upload/assets \
  /home/agilex/checkpoints/piper_dual_stack_cups_pi05_25000_pytorch/
```

server端：
```
cd /home/agilex/client
source .venv/bin/activate

python scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_piper_dual_stack_cups \
  --policy.dir=/home/agilex/checkpoints/piper_dual_stack_cups_pi05_25000_pytorch
```
> `--policy.config` 要与转换时使用的 `--config_name` 一致。
client端：关节状态控制
```
cd /home/agilex/client
source /opt/ros/humble/setup.bash
source /home/agilex/camera_ros/install/setup.bash
source /home/agilex/piper_ros/install/setup.bash
source .venv/bin/activate

python examples/piper_dual/main_dual_ros.py \
  --host 127.0.0.1 \
  --port 8000 \
  --bridge-python /usr/bin/python3 \
  --ros2-config examples/piper_dual/ros2_piper_dual_local.yaml \
  --control-stack local-ros \
  --prompt "Pass the playing cards and sunflower seeds to the guests." \
  --no-dry-run \
  --publish-actions \
  --num-steps 16000 \
  --num-episodes 1
```
client端：EEF控制（xyz+RPY）
```
python examples/piper_dual/main_dual_eef_xyz3d_ros.py \
  --host 127.0.0.1 \
  --port 8000 \
  --bridge-python /usr/bin/python3 \
  --ros2-config examples/piper_dual/ros2_piper_dual_local.yaml \
  --control-stack local-ros \
  --eef-left-topic /puppet/end_pose_left \
  --eef-right-topic /puppet/end_pose_right \
  --eef-left-action-topic /pos_left_cmd \
  --eef-right-action-topic /pos_right_cmd \
  --prompt "Clean up playing cards and the sunflower seed shells on the desk" \
  --max-action-delta 20 \
  --no-dry-run \
  --publish-actions
```



## 实验性 TOPP-RA 部署

仓库新增了两个实验性入口，用于在模型动作块之间做时间最优轨迹参数化。默认仍是 dry-run，先去掉 `--no-dry-run --publish-actions` 检查规划日志。

### 非 RTC 版

```
cd /home/agilex/client
source /opt/ros/humble/setup.bash
source /home/agilex/piper_ros/install/setup.bash
source .venv/bin/activate

python examples/piper_dual/main_dual_topp_ros.py \
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

> 该入口不支持 `--use-rtc`；详细参数与边界见 `examples/piper_dual/TOPP_README.md`。

### RTC 版

```
cd /home/agilex/client
source /opt/ros/humble/setup.bash
source /home/agilex/piper_ros/install/setup.bash
source .venv/bin/activate

python examples/piper_dual/main_dual_topp_rtc_ros.py \
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

> 当前默认 backend 为 `committed`，详细说明见 `examples/piper_dual/TOPP_RTC_COMMITTED.md`。

## 常用工具

### 双臂回零位

直接运行包装命令（自动 source ROS 环境，底层调用 `examples/piper_dual_pyagxarm/go_zero_dual.py`，
通过正在运行的驱动下发关节命令，不直接占用 CAN）：

```
return_zero --seconds 6 --speed-percent 30
```

- `--seconds 6`：用 6 秒平滑走到零位（默认 6；0 = 直接跳到位）
- `--arm left`：只回左臂；`--dry-run` 只看计划不下发
- 要求驱动保持运行；到位后停止发布，机械臂保持最后位姿

### 设置夹爪零点

必须在系统 Python 下运行，且目标臂的 `piper_*_ctrl_node` 需先停止：

```
cd /home/agilex/client
source /opt/ros/humble/setup.bash

/usr/bin/python3 examples/piper_dual/set_gripper_zero.py \
  --can can_left \
  --read-only

# 确认位置正确后，再加 --confirm 真正写入
/usr/bin/python3 examples/piper_dual/set_gripper_zero.py \
  --can can_left \
  --confirm
```

## 配合Sim2Video的工作

将采集的数据进行可视化，输出三视角的视频（头部，腕部），并保存当前的相机参数，输出action（关节状态加上eef,对齐时间戳）
```
cd /home/agilex/client
./examples/piper_dual/render_hdf5_batch.sh \
  --input-dir /path/to/hdf5 \
  --output-dir /path/to/rendered \
  --fps 30
```
