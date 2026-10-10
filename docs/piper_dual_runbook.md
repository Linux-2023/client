# Piper 双臂现场作业手册（修正版）

适用仓库：`/home/agilex/client`，分支 `piper-dual-ros2-migration`，commit `3563428`。
本手册取代旧的现场 runbook；凡与旧文档冲突处，以本手册为准。

阅读约定：

- 所有命令均为绝对路径风格，可直接复制粘贴。
- 凡标注「待现场确认」的条目，属于尚未实测的推断，必须由现场操作者按第 14 节逐条核查后填写。
- 本手册不写「已验证」字样，除非对应结论确为实测所得。

---

## 1. 修订摘要

| 序号 | 原写法 | 修正后 | 原因 |
|---|---|---|---|
| 1 | `cd /home/agilex/client/.worktrees/piper-dual-ros2-migration` | `cd /home/agilex/client` | 该 worktree 不存在；`git worktree list` 仅 `/home/agilex/client` |
| 2 | 采集/关节部署命令缺 `--control-stack local-ros`，且 `--config` 用 `ros2_piper_dual.yaml` | 改为 `--control-stack local-ros --config examples/piper_dual/ros2_piper_dual_local.yaml` | 否则启动即抛 `missing required topic /joint_left` 或 `found forbidden topic /puppet/joint_left` |
| 3 | `main_dual_eef_xyz3d_ros.py` 那条命令 | 保留不变（`--control-stack local-ros` + `_local.yaml`），并作为范例 | 原写法本就正确；其余命令一律向它对齐 |
| 4 | 相机启动脚本 `/home/agilex/camera_ros/scripts/start_realsense_3cam_color.sh`（旧名不存在） | 使用本次新建脚本 `/home/agilex/camera_ros/scripts/start_realsense_3cam_color.sh` | 旧名文件不存在，启动会直接失败 |
| 5 | 「可视化数据」用 `visualize_ros2_episodes.py` | 改用 `examples/piper_dual/render_hdf5_batch.sh` | `/home/agilex/camera_ros/src/deploy/visualize_ros2_episodes.py` 不存在 |
| 6 | `--num-episodes 1-` | `--num-episodes 1` | `1-` 非法（参数为 `type=int`），会解析报错 |
| 7 | `cp -a ..._pytorch_exact/` | 拷贝到实际 `--output_path` 指定的目录 | runbook 自相矛盾，源目录名与实际输出目录名不一致 |
| 8 | 转换命令未统一数据集根目录 | 统一 `export HF_LEROBOT_HOME=/home/agilex/lgd/lerobot_datasets/` | 否则第一条写入默认 `~/.cache/huggingface/lerobot`，数据分落两处 |
| 9 | 模型转换/服务用 `config_name=pi05_piper_dual_stack_cups` | 用与训练数据集一致的 `config_name`（eef_xyz3d 数据集训练 → `pi05_piper_dual_*_eef_xyz3d_100`） | `config_name` 与训练数据不一致会导致归一化统计/动作维度对不上 |
| 10 | 「启动机械臂」终端只写 `bash start_multi_piper.sh` | 先 `source /opt/ros/humble/setup.bash` 和 `source /home/agilex/piper_ros/install/setup.bash` | `start_multi_piper.sh` 内部不 source ROS |
| 11 | 官方 legacy 输出 `/home/agilex/lgd_data`、部署用 `/home/agilex/checkpoints` 未提及是否需创建 | 本次已 `mkdir -p` 创建（首次运行仍需确认脚本是否自行创建） | 之前这两个目录不存在，官方流程脚本会找不到落点 |

---

## 2. 控制栈契约

### 2.1 三栈对照表

`examples/piper_dual/control_stack.py` 定义三套栈，**每套栈只能配对应的 yaml**。`ros2_contract.py:151-159` 会在启动时对 live ROS graph 做硬校验：缺 required 话题 → `ValueError: ... missing required topic ...`；出现 forbidden 话题 → 同样报错。

| 栈 id | 配置文件 | required 话题 | forbidden 话题 | 动作话题 |
|---|---|---|---|---|
| `local-ros` | `ros2_piper_dual_local.yaml` | `/puppet/joint_left`、`/puppet/joint_right`、`/master/joint_left`、`/master/joint_right`、`/piper_left_ctrl_node/arm_status`、`/piper_right_ctrl_node/arm_status` | `/joint_left`、`/joint_right`、`/direct_sdk/joint_*` | `/joint_left_states`、`/joint_right_states` |
| `official-ros` | `ros2_piper_dual.yaml` | `/joint_left`、`/joint_right`、`/joint_states_ctrl_left`、`/joint_states_ctrl_right`、`/arm_status_left`、`/arm_status_right` | `/puppet/joint_left`、`/puppet/joint_right`、`/direct_sdk/joint_*` | `/joint_ctrl_cmd_left`、`/joint_ctrl_cmd_right` |
| `direct-sdk` | `ros2_piper_dual_direct_sdk.yaml` | `/direct_sdk/joint_left`、`/direct_sdk/joint_right`、`/direct_sdk/joint_ctrl_left`、`/direct_sdk/joint_ctrl_right`、`/direct_sdk/arm_status_left`、`/direct_sdk/arm_status_right` | `/puppet/joint_*`、`/joint_left`、`/joint_right` | `/direct_sdk/joint_cmd_left`、`/direct_sdk/joint_cmd_right`（需要节点 `piper_direct_sdk_adapter`） |

### 2.2 本机实机对应关系（实测）

`/home/agilex/piper_ros` 驱动发布 `/puppet/joint_*`、`/master/joint_*`、`/piper_*_ctrl_node/arm_status`、`/puppet/end_pose_left|right`，并 remap 订阅 `/joint_left_states`、`/joint_right_states`、`/pos_left_cmd`、`/pos_right_cmd`。

**结论：本机只有 `local-ros` 栈可用。** 全机 grep 确认没有任何发布 `/joint_left`、`/joint_states_ctrl_left`、`/arm_status_left`、`/end_pose_stamped_left` 的节点。

### 2.3 选择规则

- 各脚本 `--control-stack` 默认值均为 `official-ros`（`collect_data_ros2.py:274-277`、`collect_data_eef_ros2.py:77-79`、`main_dual_ros.py:43`、`main_dual_eef_xyz3d_ros.py:78`）。**默认值在本机不可用**，凡配本机实机的命令必须显式写 `--control-stack local-ros` 且 `--config examples/piper_dual/ros2_piper_dual_local.yaml`。
- 换 official-ros 栈（需在配套驱动上运行）时，必须同时换 `ros2_piper_dual.yaml` 与 `--control-stack official-ros`；direct-sdk 栈同理换其专用 yaml，并确保 `piper_direct_sdk_adapter` 节点在跑。
- 栈与 yaml 必须成对出现；混搭会在启动瞬间被契约校验拒绝。

---

## 3. 环境与路径总表

### 3.1 基础环境

| 项 | 路径 / 值 | 状态 |
|---|---|---|
| 代码仓库 | `/home/agilex/client` | 存在（分支 `piper-dual-ros2-migration`，commit `3563428`） |
| worktree | `/home/agilex/client/.worktrees/piper-dual-ros2-migration` | 不存在；`git worktree list` 仅 `/home/agilex/client` |
| Python 环境 | `/home/agilex/client/.venv` | 存在，CPython 3.11.14（uv 托管），`uv sync` 已完成，`uv sync --dry-run` → `Would make no changes` |
| ROS | `/opt/ros/humble/setup.bash` | 存在，Humble；ROS 侧 Python 3.10.12，与 venv 3.11 不兼容 |
| 桥接 Python | `/usr/bin/python3` | 存在，3.10.12（采集/部署脚本的 `--bridge-python` 用它） |
| conda 环境 | `/home/agilex/miniforge3/envs/data` | 存在，Python 3.10.0；h5py 3.16.0 / numpy 1.23.5 / PIL 11.3.0；可 `import rclpy`（前提先 source ROS） |
| direct-sdk 专用 venv | `examples/piper_dual/.venv` | 不存在（README 里给 direct-sdk 直连用的 3.10 环境未创建；走 ROS 桥接不需要它） |

### 3.2 工具与包路径（存在）

| 路径 | 说明 |
|---|---|
| `/home/agilex/camera_ros/install/setup.bash` | 相机 ROS 包安装结果 |
| `/home/agilex/camera_ros/scripts/` | 相机脚本目录 |
| `/home/agilex/piper_ros/install/setup.bash` | 机械臂 ROS 包安装结果 |
| `/home/agilex/piper_ros/find_all_can_port.sh` | CAN 口枚举 |
| `/home/agilex/piper_ros/can_config.sh` | CAN 激活配置 |
| `/home/agilex/piper_ros/can_activate.sh` | CAN 单口激活 |
| `/home/agilex/piper_ros/start_multi_piper.sh` | 双臂启动 |
| `/home/agilex/data_ros/install/setup.bash` | 官方 legacy 采集包安装结果 |
| `/home/agilex/data_ros/src/data_tools/scripts/{data_sync.py,data_to_hdf5.py,data_publish.py}` | 官方 legacy 三步脚本 |
| `/home/agilex/data_ros/src/data_tools/launch/{run_data_capture.launch.py,run_data_publish.launch.py}` | 官方 legacy launch |
| `realsense-viewer`、`rs-enumerate-devices`、`v4l2-ctl` | 相机排查工具 |

### 3.3 本次已创建

| 路径 | 说明 |
|---|---|
| `/home/agilex/camera_ros/scripts/start_realsense_3cam_color.sh` | 本次新建的三路彩色相机启动脚本 |
| `/home/agilex/camera_ros/.../three_camera_color.launch.py` | 本次新建的三路彩色 launch（与上脚本配套） |
| `/home/agilex/lgd_data` | 官方 legacy 采集输出根目录（本次 `mkdir -p`） |
| `/home/agilex/piper_dual_dataset` | 本地栈采集输出根目录（本次 `mkdir -p`） |
| `/home/agilex/checkpoints` | 模型 checkpoint 根目录（本次 `mkdir -p`） |
| `/home/agilex/lgd/lerobot_datasets` | LeRobot 数据集根目录（本次 `mkdir -p`） |

### 3.4 仍缺 / 需注意

| 路径 | 状态 |
|---|---|
| `/home/agilex/.cache/huggingface` | 仍未创建；由首次 HF 写入时自动生成 |
| `/home/agilex/camera_ros/src/deploy/visualize_ros2_episodes.py` | 不存在，可视化改用 `render_hdf5_batch.sh` |
| `examples/piper_dual/.venv` | 不存在；仅 direct-sdk 直连路径需要 |
| `/home/agilex/checkpoints/*` 内的模型目录 | 需现场确认是否已拉到本地（见第 14 节） |

### 3.5 环境变量（实测）

- `~/.bashrc` 含：`export ROS_DOMAIN_ID=222`；`http_proxy/https_proxy/all_proxy=http://127.0.0.1:7897`，`no_proxy=localhost,127.0.0.1,::1`。
- 当前 shell：`HF_TOKEN` 未设置；`HF_LEROBOT_HOME`、`OPENPI_DATA_HOME` 未设置；代理变量已设置。
- HF 上传命令使用 `env -u ALL_PROXY -u all_proxy` 是有原因的：代理会干扰 HF 直连。
- **ROS 相关终端必须都带 `ROS_DOMAIN_ID=222`**（`.bashrc` 已设；若用 `env -u` 清环境或非交互 shell，需手动 export，否则 `ros2 topic list` 看不到对方）。

---

## 4. 准备活动

### 4.1 终端约定

每个 ROS 终端都必须先 source ROS，再进入仓库：

```bash
source /opt/ros/humble/setup.bash        # 每个 ROS 终端都要
cd /home/agilex/client
```

约定：

- 采集/部署脚本走 ROS 桥接，桥接进程用 `/usr/bin/python3`（3.10.12），venv 进程用 3.11；不要试图让一个解释器同时满足两者。
- 启动机械臂的脚本内部不 source ROS，终端必须自己 source（见 6.1）。
- 非交互 shell 或清理过环境的终端，需手动 `export ROS_DOMAIN_ID=222`。

### 4.2 电源与底座

- 先接通双臂电源与相机供电，再执行 CAN 激活与节点启动；未上电时 CAN 口会停在 DOWN。
- 底座/台面按任务要求摆放（如 stack cups、fold towel），并留出双臂行程与相机视野，避免遮挡。
- 上电后如需断电重启，先停 ROS 节点，再断机械臂电源。

### 4.3 环境变量提醒

- `ROS_DOMAIN_ID=222`：所有 ROS 终端一致，否则互相看不见话题。
- `HF_LEROBOT_HOME`：转换与上传阶段统一 export（见第 9、11 节）。
- 代理变量：ROS 本地通信走 `no_proxy` 白名单；HF 上传时临时取消 ALL_PROXY。

---

## 5. 相机

### 5.1 串号采集

在线 3 台（`rs-enumerate-devices`）：`254622074071`、`254522075411`、`254522077523`（均为 Intel RealSense D435）。

旧 runbook 记录的 `254622072400 / 254522074588 / 151222073259` 与在线设备**全不匹配**，作废。

正规串号采集方式（仓库自带）：

```bash
source /opt/ros/humble/setup.bash          # rs-enumerate-devices 位于 /opt/ros/humble/bin
python3 /home/agilex/camera_ros/scripts/setup_realsense_serials.py
```

该脚本会要求「只插一台相机」逐个录 front/left/right。

注意：该工具的结果写入的是 `camera_ros/scripts/start_realsense_camera.sh`（**4 相机版**，且其中 front 串号 `254622073589` 当前不在线），**不会**修改本手册使用的 `start_realsense_3cam_color.sh`。重新采集后，需把 front / left / right 三个值手动同步到第 5.3 节的脚本默认值（或运行时用 `FRONT_SERIAL=` / `LEFT_SERIAL=` / `RIGHT_SERIAL=` 覆盖）。

### 5.2 启动三路彩色相机

**启动前先做预检**（这是最常踩的坑：少一台相机 → 采集器不会出预览窗口）：

```bash
source /opt/ros/humble/setup.bash
rs-enumerate-devices -s
```

期望输出 3 行 D435（串号 `254622074071` / `254522077523` / `254522075411`）。少于 3 台就先修硬件（USB 线/供电/换口），别急着启动。

```bash
source /home/agilex/camera_ros/install/setup.bash
bash /home/agilex/camera_ros/scripts/start_realsense_3cam_color.sh
```

脚本会先校验三个串号都在线：缺相机时**拒绝启动**并打印排查清单；临时只用在线相机调试可加 `ALLOW_PARTIAL=1`（缺的那台会在 ROS 侧报 `No device connected`，且采集器凑不齐同步帧）。

### 5.3 默认映射（待现场确认）

**2026-09-21 实测记录**：启动脚本后，各节点的 `serial_no` 参数与下表一致（`ros2 param get /camera_f_l serial_no` → `_254622074071`，`/camera_l` → `_254522077523`，`/camera_r` → `_254522075411`），三路 `/camera_*_color/image_raw` 话题均发布。即**"节点 ↔ 串号"绑定已实测确认**；但"物理上哪台是中间/高视角、哪台是左腕、哪台是右腕"仍需挡镜头确认（见第 14 节）。
同次实测中发现：三相机启动约 1 分钟后 USB 上所有外设（3 相机 + 2 个 CAN 适配器）同时掉线（realsense 报 `The device has been disconnected!`，`lsusb` 只剩 root hub），属于供电/接线问题，需先按第 4.2 节上电。

本次新建的 `start_realsense_3cam_color.sh` + `three_camera_color.launch.py` 默认映射如下，**必须现场确认**：

| 角色 | ROS 命名空间 | 客户端字段 | 串号 | 备注 |
|---|---|---|---|---|
| front | `camera_f_l` | `cam_high` | `254622074071` | 旧记录里的 front_right，唯一剩余在线设备 |
| left | `camera_l` | `cam_left_wrist` | `254522077523` | 与既有记录一致 |
| right | `camera_r` | `cam_right_wrist` | `254522075411` | 与既有记录一致 |

对照：既有记录 `camera_ros/scripts/start_realsense_camera.sh`（2026-06-11，比 launch 文件新）的分配为 `front=254622073589`（**当前离线**）、`left=254522077523`、`right=254522075411`、`front_right=254622074071`。

### 5.4 图像方向（旋转）

**2026-09-21 实测**：wrapper 的 `rotation_filter` 在**本机确实对彩色流生效**（系统 librealsense 2.56.5；`base_realsense_node.cpp` 把它注册到 DEPTH/COLOR/INFRARED，`named_filter.cpp` 的 `registerDynamicOptions` 把 `rotation_filter.rotation` 映射到 `RS2_OPTION_ROTATION`，`enable=true` 才会 process）。判定实验（同场景取帧，开关滤镜对比）：噪声地板 `mean|A-A2|=2.48`，关掉滤镜后 `mean|A-B|=58.75`、`mean|rot180(A)-B|=2.61` → 即关掉滤镜正好使画面相对开启时旋转 180°。

由此修掉一个沿用错误：老 4 相机配置 `multi_camera.launch.py` 给 `camera_f_l` 设了 `rotation_filter.enable=true, rotation=180.0`，在本机会把**正装的**头部相机画面转倒。三相机 launch 已改为默认 **`front_rotation_enable=false`（0°）**；若将来相机倒装，启动时传 `front_rotation_enable:=true front_rotation:=180.0`，或运行时改参数：

```bash
ros2 param set /camera_f_l rotation_filter.rotation 180.0
ros2 param set /camera_f_l rotation_filter.enable true
```

**架构要点（决定"该在哪一层转"）**：HDF5 里存的是**桥接层收到的原始 JPEG**；客户端 yaml 的 `rotate_degrees` 只作用于 `observation_adapter.py`（→ 预览窗口与推理输入），**不影响录制数据**。因此：

- 希望"录制数据 / 推理输入 / 训练数据"三者同向 → 方向应在 **ROS 层**（相机节点或重发布节点）统一解决，yaml 的 `rotate_degrees` 保持 0；
- 若确实要在客户端转，必须把 yaml 的 `rotate_degrees` 设成同一个值，否则会出现「训练数据方向 ≠ 推理输入方向」的隐性错配。

### 5.5 话题核查

| 客户端字段 | 话题 |
|---|---|
| `cam_high` | `/camera_f_l/color/image_raw` |
| `cam_left_wrist` | `/camera_l/color/image_raw` |
| `cam_right_wrist` | `/camera_r/color/image_raw` |

客户端两个 yaml 的相机话题**不改**，与新脚本一致。核查方法：`ros2 topic hz /camera_l/color/image_raw` 等三路各自独立观察，逐一遮挡镜头看哪一路帧率掉零。

```bash
ros2 topic hz /camera_f_l/color/image_raw
ros2 topic hz /camera_l/color/image_raw
ros2 topic hz /camera_r/color/image_raw
```

---

## 6. 机械臂

### 6.1 CAN 使能与启动

`ip -br link | grep can` **2026-09-21 实测**：执行 `can_config.sh` 后为 `can_left`、`can_right`（均 `UP`、`bitrate 1000000`、`ERROR-ACTIVE`）；执行前为 `can1`、`can2`（`DOWN`），底座那路 `can0`（USB `1-4:1.0`）当时未插。

`/home/agilex/piper_ros/can_config.sh` 内容：`can_activate.sh can_left 1000000 "1-13:1.0"`、`can_activate.sh can_right 1000000 "1-12:1.0"`、`can_activate.sh can0 500000 "1-4:1.0"`。

角色 ↔ USB 端口 ↔ 接口名（脚本声明 + 实测）：

| 角色 | `can_config.sh` 声明的 USB 地址 | 重命名后接口名（实测其 USB 路径） | 重命名前的临时名 |
|---|---|---|---|
| 左臂 | `1-13:1.0` | `can_left` → 实测 `1-13:1.0` | `can2` |
| 右臂 | `1-12:1.0` | `can_right` → 实测 `1-12:1.0` | `can1` |
| 底座 | `1-4:1.0` | `can0`（该端口当时未插） | `can0` |

`can_activate.sh` 用 `sudo ethtool -i <iface>` 的 `bus-info` 与传入地址**字符串相等**来选设备并重命名，因此「接口名 ↔ USB 地址」由脚本强制保证，不是人工推断；但「USB 端口 ↔ 物理左/右臂」需要行为验证：起节点后移动**物理左主臂**，确认变化的是 `/master/joint_left`（若变化的却是 `/master/joint_right`，说明左右接反，需把 `can_config.sh` 的 left/right 两行对调）。免 sudo 读取同一标识：`readlink -f /sys/class/net/can_left/device`（输出末尾即 `X-Y:Z.W`，与 `ethtool -i` 的 bus-info 同义）。

`/home/agilex/piper_ros/start_multi_piper.sh` 内容：`conda deactivate` → `bash can_config.sh` → `ros2 launch piper start_two_piper.launch.py`（**内部不 source ROS**，所以终端里必须先 source）。

```bash
cd /home/agilex/piper_ros
source /opt/ros/humble/setup.bash
source /home/agilex/piper_ros/install/setup.bash
ip -br link | grep can
bash find_all_can_port.sh
bash can_config.sh
bash start_multi_piper.sh
```

### 6.2 enable 服务

`start_two_piper.launch.py` 启动节点 `piper_left_ctrl_node`、`piper_right_ctrl_node`。`auto_enable` **默认值为 `true`**（`start_multi_piper.sh` 不覆盖它 → 一 launch 就使能双臂）；要"先起节点、稍后再使能"就显式传 `auto_enable:=false`。

**注意（2026-09-21 实测）**：本版节点的服务名是**相对名** `'enable_srv'`（`piper_ctrl_single_node.py:55`），注册结果是 **`/enable_srv`** —— 两个臂节点**同名**（node 名不参与服务名解析）。因此旧文档里的 `/piper_left_ctrl_node/enable_srv` **不存在**；且调用 `/enable_srv` 只会命中其中一个节点（命中哪个不确定）。

确定地使能**双**臂：两个节点都订阅 `/enable_flag`（`std_msgs/Bool`），发布一次 `true` 会同时送达两者：

```bash
ros2 topic pub --once /enable_flag std_msgs/msg/Bool "{data: true}"     # 使能双臂
ros2 topic pub --once /enable_flag std_msgs/msg/Bool "{data: false}"    # 失能双臂
```

服务路径仍可用（会校验 6 个电机 `driver_enable_status` 后才置位），但只作用于其中一个臂：`ros2 service call /enable_srv piper_msgs/srv/Enable "{enable_request: true}"`。

另：节点本身**不做**主臂→从臂跟随 —— 源码里 `JointCtrl()` 只在收到 ROS 关节命令（`/joint_*_states`）时被调用，200 Hz 循环只读状态并发布 `/puppet/joint_*`、`/master/joint_*`。因此从臂是否随主臂运动取决于使能后的臂端行为，采集前必须实测确认（见第 14 节）。

### 6.3 话题核查

节点话题（实测源码 `piper_ctrl_single_node.py:49-53`）：

| 话题 | 含义 |
|---|---|
| `/puppet/joint_left`、`/puppet/joint_right` | 从臂关节 |
| `/master/joint_left`、`/master/joint_right` | 主臂关节 |
| `/piper_left_ctrl_node/arm_status`、`/piper_right_ctrl_node/arm_status` | 臂状态 |
| `/puppet/end_pose_left`、`/puppet/end_pose_right` | 末端位姿 |

核查命令：

```bash
ros2 node list
ros2 topic echo --once /piper_left_ctrl_node/arm_status
```

这三个 required 话题 + 动作话题的对应关系即 `local-ros` 栈契约（见第 2 节）；节点起来后这些话题必须全部在线，否则采集/部署脚本会在契约校验处直接报错。

注意：`/puppet/end_pose_left|right` 是**被测臂（从臂）**的末端位姿反馈（`end_pose_stamp_pub` 硬编码 `puppet` 前缀），`/master/joint_left|right` 是控制帧回显。**不存在 `/master/end_pose_*` 话题**：firmware 不提供主臂末端位姿反馈，主臂 EEF 只能由主臂关节 FK 推导（见 7.2）。

---

## 7. 数据采集（优化版，本地栈）

### 7.1 关节控制采集

```bash
cd /home/agilex/client
source /opt/ros/humble/setup.bash
source /home/agilex/piper_ros/install/setup.bash
source .venv/bin/activate
python examples/piper_dual/collect_data_ros2.py \
  --output-dir /home/agilex/piper_dual_dataset/fold_towel \
  --prompt "Fold the towel" \
  --control-stack local-ros \
  --config examples/piper_dual/ros2_piper_dual_local.yaml \
  --bridge-python /usr/bin/python3 \
  --max-sync-error-ms 30
```

### 7.2 含 EEF 的采集

```bash
python examples/piper_dual/collect_data_eef_ros2.py \
  --output-dir /home/agilex/piper_dual_dataset/stack_cups_eef \
  --prompt "Stack_the_paper_cups_together." \
  --control-stack local-ros \
  --config examples/piper_dual/ros2_piper_dual_local.yaml \
  --bridge-python /usr/bin/python3 \
  --max-sync-error-ms 30 \
  --eef-left-topic /puppet/end_pose_left \
  --eef-right-topic /puppet/end_pose_right
```

采集结果（`schema_version = piper_dual_ros2_v1`）：`/observations/state` = 从臂 14 维关节，`/action` = 主臂 14 维关节，另加四个 6 维 EEF 数据集（`[x, y, z, roll, pitch, yaw]`，米 + 弧度）：

| 数据集 | 来源 | 备注 |
|---|---|---|
| `/observations/eef/puppet_left`、`puppet_right` | `/puppet/end_pose_left`、`right` 话题（firmware 末端位姿反馈） | 参与 `--max-sync-error-ms` 同步，自带 `sensor_timestamps/sync_error` 条目 |
| `/observations/eef/master_left`、`master_right` | FK(同一帧 `/action` 的左/右 7 维主臂关节)，vendored Piper DH（`dh_is_offset=1`，link6） | 与 `action` 同帧同时间戳，不重复写时间戳/误差，看 `sensor_timestamps/master_left|right` |

**主臂 EEF 为什么是推导值**：`piper_ctrl_single_node` 只发布被测臂的 `/puppet/end_pose_*`（`end_pose_stamp_pub` 硬编码 `puppet` 前缀），`/master/joint_*` 是控制帧回显，firmware 没有主臂末端位姿反馈帧，所以 `frame_synchronizer` 用 `master_eef.MasterEefKinematics` 在同一帧内由主臂关节算出主臂 EEF，不需要新话题、也不改驱动。该 DH 模型对 `playing_cards_2` 全部 100 episode（148892 帧）逐帧复现 firmware 末端位姿反馈：位置误差 mean 0.43 mm（左）/ 0.21 mm（右）、p95 2.8 mm / 1.4 mm，姿态 mean 0.15° / 0.40°。两处已知偏差：关节样本与位姿样本之间存在最长 30 ms 偏差（高速运动时表现为毫米级差异）；pitch ≈ ±90° 简并点上 roll/yaw 拆分不唯一（两种拆分代表同一旋转）。

`metadata_json.eef.master.source = "fk_from_master_joints"` 标记主臂 EEF 为 FK 推导（`inputs = ["/action"]`、`kinematics.link = "link6"`、`kinematics.dh_is_offset = 1`）。`validate_episode` 对声明了 `eef.master` 的文件强制要求四个 EEF 数据集；改动前采集的两路 puppet-only 文件仍按原规则校验通过。`export_episode_json.py` 仍是 puppet-only 导出；EEF 类转换脚本不受影响（已用新格式 episode 实跑 `_eef_xyz3d_master_action` 与 `_eef_xyz3d_shifted` 验证）。

### 7.3 参数说明

`collect_data_ros2.py` 可用参数：`--output-dir --prompt --config --bridge-python --control-stack --max-sync-error-ms --dry-run/--no-dry-run --publish-actions --jpeg-quality --render-after-save`。

- **`--dry-run` 默认 True**：不下发动作，只做采集链路联调。要真机执行必须显式 `--no-dry-run`。
- `--publish-actions` 与 `--dry-run` 同时给会报错，二者互斥。
- `collect_data_eef_ros2.py` 另有 `--eef-left-topic`（默认 `/puppet/end_pose_left`）、`--eef-right-topic`（默认 `/puppet/end_pose_right`）。
- `--bridge-python /usr/bin/python3` 指向 ROS 侧 3.10 解释器；venv 是 3.11，不能替代。

### 7.4 单集 finalize 的校验范围（2026-09-23 调整）

按 `s`/`e` 结束一集时，finalize 只做**结构校验**：dataset 齐全性、形状/dtype/可扩展性、数值有限性、时间戳单调性；**不再逐帧解码 JPEG**，也不再在 finalize 之后再跑一遍校验、不再打印 `Validation report`。

实测（`playing_cards_2/episode_000000.hdf5`，1818 帧）：全量 `validate_episode`（含逐帧解码）**25.67 s**，结构校验 **0.58 s**，JPEG 解码占 97.7%；旧流程每条数据要跑两遍全量校验 ≈ 51 s。现在每条 ≈ 0.6 s。

- 完整校验仍然可用，且离线路径默认就是全量：`streaming_hdf5.validate_episode(path)`（`render_dataset.py`、转换脚本的 `--deep-validate` 都走全量）。
- 失效模式变化：**「图像字节不可解码」（相机流中断/桥接传了非 JPEG）不再在 finalize 当场拦下**，会先发布成 `.hdf5`，由离线全量校验或渲染时暴露。其余结构性问题（空集、缺 dataset、长度/形状不符、NaN、时间戳回退）仍在 finalize 处 fail closed 并删除未发布的 `.hdf5.partial`。
- 若某次采集后想立刻确认图像可解码：`python -c "import sys; sys.path.insert(0,'examples/piper_dual'); from streaming_hdf5 import validate_episode; print(validate_episode('episode_000000.hdf5')['errors'])"`。

### 7.5 键位

窗口预览三路相机：

| 键 | 行为 |
|---|---|
| `s` | 开始录制；录制中按 `s` 或 `e` 结束并 finalize |
| `q` | 退出（若有未 finalize 的 partial 会 abort） |

---

## 8. 数据采集（官方 legacy 流程）

三步流程，两个终端。

终端 1（采集）：

```bash
source /opt/ros/humble/setup.bash
source /home/agilex/camera_ros/install/setup.bash
ros2 launch data_tools run_data_capture.launch.py type:=aloha datasetDir:=/home/agilex/lgd_data episodeIndex:=0
```

终端 2（同步 + 转 HDF5）：

```bash
source /opt/ros/humble/setup.bash
source /home/agilex/data_ros/install/setup.bash
conda activate data
cd /home/agilex/data_ros/src/data_tools/scripts
python3 data_sync.py --type aloha --datasetDir /home/agilex/lgd_data
python3 data_to_hdf5.py --type aloha --datasetDir /home/agilex/lgd_data
```

重放：

```bash
python3 data_publish.py --datasetDir /home/agilex/lgd_data/episode0/episode0.hdf5 --publish_rate 30
```

参数核对：`data_sync.py --type --datasetDir`；`data_to_hdf5.py --type --datasetDir --targetDir`；`data_publish.py --datasetDir --publish_rate --publishIndex --type`（`--publish_rate` 确实存在）。

注意：`/home/agilex/lgd_data` 本次才创建，首次运行需确认官方脚本是否会自行创建下级目录（见第 14 节）。

---

## 9. 转换到 LeRobot 格式（脚本与语义）

转换前**必须先统一数据集根目录**：

```bash
export HF_LEROBOT_HOME=/home/agilex/lgd/lerobot_datasets/
cd /home/agilex/client
```

`examples/piper_dual/utils/` 下 6 个脚本的语义（**2026-09-22 全部用合成 episode 实测验证**，数值与下表逐项吻合；"N" = 源 episode 帧数）：

| 脚本 | `observation.state` | `action` | 维度 | 帧数 |
|---|---|---|---|---|
| `convert_ros2_piper_data_to_lerobot.py`（基础） | 从臂当前关节（含**从臂**夹爪） | 主臂当前关节（含**主臂**夹爪） | 14 | N |
| `..._shifted.py` | 从臂当前关节 + **主臂当前夹爪** | **从臂 t+1 关节** + 主臂当前夹爪 | 14 | N−1 |
| `..._eef_shifted.py` | 从臂 EEF(t) `xyz+rot6d` + 主臂夹爪(t) | 从臂 EEF(t+1) `xyz+rot6d` + 主臂夹爪(t) | 20 | N−1 |
| `..._eef_xyz3d_shifted.py` | 从臂 EEF(t) `xyz+RPY` + 主臂夹爪(t) | 从臂 EEF(t+1) `xyz+RPY` + 主臂夹爪(t) | 14 | N−1 |
| `..._eef_xyz3d_master_action.py`（新增） | 同上一行 | **主臂当前关节 + 主臂自己的夹爪**（原始 `/action` 行） | 14 | **N** |
| `..._combined.py` | 同 `_shifted` | 同 `_shifted`（**EEF 位姿字段被忽略**） | 14 | 各 N−1 |

要点：
- **shifted 系列一定丢最后一帧**（N→N−1）；新增的 `_eef_xyz3d_master_action` 因 action 取同帧主臂向量，**不丢帧**。
- EEF 类脚本（3 个）要求源 HDF5 带 `/observations/eef/puppet_left|puppet_right` 且 `metadata_json.eef.enabled=true`，即必须用 `collect_data_eef_ros2.py` 采集；否则报 `metadata_json.eef.enabled must be true for EEF conversion`。
- 现有 6 个脚本都**不读**新增的 `/observations/eef/master_left|right`（主臂 EEF 由 FK 推导，见 7.2）；这些数据集只作观测/后续 EEF-action 训练来源，对上述转换脚本的输出无影响。要按主臂 EEF 训练需另写转换脚本。
- 特征名：EEF 类 state = `eef_puppet_{left,right}_{x,y,z,roll,pitch,yaw}` + `{left,right}_gripper`；`_eef_xyz3d_master_action` 的 action 名为 14 个关节名（`left_waist … right_gripper`）。
- 共有参数：`--raw-dir --repo-id --mode {image,video}`，另有 `--episodes 0,1,2`、`--overwrite`、`--deep-validate`。
- **预检默认只读 metadata**（`--deep-validate` 可切回全量解码）：结构性错误（缺 dataset、形状不符、非 EEF、prompt 为空）仍在写任何文件前全部拦下，耗时 ~1 ms/集（实测 100 集 / 97,528 帧用 **90 ms**；旧的"全部解码"预检约 15 s/集≈25 分钟）。代价是 JPEG 本身损坏这类问题会跑到该集才暴露——异常时脚本会删除半成品目录，不会留下脏数据集。
- 转换过程逐集打印 `[k/N] episode_xxxxxx.hdf5: N frames`，第一集写完立即可见。

### 9.1 关节

```bash
.venv/bin/python examples/piper_dual/utils/convert_ros2_piper_data_to_lerobot_shifted.py \
  --raw-dir /home/agilex/piper_dual_dataset/stack_cups \
  --repo-id local/piper_dual_stack_cups_ros2_shifted --mode image
```

### 9.2 EEF → xyz+rot6d

```bash
.venv/bin/python examples/piper_dual/utils/convert_ros2_piper_data_to_lerobot_eef_shifted.py \
  --raw-dir /home/agilex/piper_dual_dataset/stack_cups_eef \
  --repo-id local/piper_dual_stack_cups_eef_rot6d_shifted --mode image
```

### 9.3 EEF → xyz+RPY（原生控制）

```bash
.venv/bin/python examples/piper_dual/utils/convert_ros2_piper_data_to_lerobot_eef_xyz3d_shifted.py \
  --raw-dir /home/agilex/piper_dual_dataset/stack_cups_eef \
  --repo-id local/piper_dual_stack_cups_eef_xyz3d --mode image
```

### 9.4 EEF state + 主臂关节 action（不丢帧）

```bash
.venv/bin/python examples/piper_dual/utils/convert_ros2_piper_data_to_lerobot_eef_xyz3d_master_action.py \
  --raw-dir /home/agilex/piper_dual_dataset/stack_cups_eef \
  --repo-id local/piper_dual_stack_cups_eef_xyz3d_master_action --mode image
```

用途：观察量是**从臂 EEF 位姿**（原生 `xyz+RPY`），动作标签是**主臂当前关节向量**（含主臂自己的夹爪，即采集时的 `/action` 原值）。因为 state/action 同帧，**输出帧数 = 源帧数 N**（与 9.3 的 N−1 不同）。

**必须统一 `HF_LEROBOT_HOME`**：否则未 export 的那条会写到默认 `~/.cache/huggingface/lerobot`，同一批数据分落两处，后续上传与训练找不到。

---

## 10. 渲染与可视化

旧文档的 `visualize_ros2_episodes.py` **不存在**（路径 `/home/agilex/camera_ros/src/deploy/visualize_ros2_episodes.py`），可视化改用 `examples/piper_dual/render_hdf5_batch.sh`（可执行，参数 `--input-dir --output-dir --fps`）。

```bash
cd /home/agilex/client
./examples/piper_dual/render_hdf5_batch.sh \
  --input-dir /home/agilex/piper_dual_dataset/stack_cups_eef \
  --output-dir /home/agilex/piper_dual_dataset/stack_cups_eef_rendered \
  --fps 30
```

产出：自包含 HDF5 → 三视角视频 + `quality.json`。

---

## 11. 上传 HuggingFace

```bash
export HF_TOKEN='hf_你的Token'
cd /home/agilex/lgd/lerobot_datasets/local      # 与 HF_LEROBOT_HOME 一致，不再是 ~/.cache/huggingface/lerobot/local
env -u ALL_PROXY -u all_proxy python - <<'PY'
import os
from huggingface_hub import HfApi
repo_id = "HITdongdong/piper_dual_stack_cups_ros2"
folder = "piper_dual_stack_cups_ros2"
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
print(api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=folder,
                        allow_patterns=["data/**", "meta/**"], commit_message="Upload LeRobot dataset"))
PY
```

要点：

- 上传目录为 `/home/agilex/lgd/lerobot_datasets/local`，与第 9 节 export 的 `HF_LEROBOT_HOME` 保持一致；旧文档里的 `~/.cache/huggingface/lerobot/local` 已不再正确。
- `env -u ALL_PROXY -u all_proxy` 用于临时取消代理，代理会干扰 HF 直连。
- `HF_TOKEN` 当前未设置，必须先 export；`/home/agilex/.cache/huggingface` 由首次 HF 写入时自动生成。

---

## 12. 模型部署

**先看数据集的 action 空间，再选客户端**（搞反会把关节角当末端位姿下发）：

| 数据集 action 空间 | 用哪个 client | 下发话题 | 臂节点回调 → 底层调用 |
|---|---|---|---|
| **关节量**（每臂 6 关节 + 夹爪，两臂 14 维）——如 `_shifted`、`_eef_xyz3d_master_action` | `examples/piper_dual/main_dual_ros.py`（见 12.4） | `/joint_left_states`、`/joint_right_states` | `joint_callback` → `JointCtrl(...)` |
| **末端位姿**（xyz+rpy+夹爪）——如 `_eef_shifted`、`_eef_xyz3d_shifted` | `examples/piper_dual/main_dual_eef_xyz3d_ros.py`（见 12.5） | `/pos_left_cmd`、`/pos_right_cmd` | `pos_callback` → `EndPoseCtrl(x,y,z,rx,ry,rz)` |

注意：**state 的空间与 action 的空间可以不同**。例如 `_eef_xyz3d_master_action` 数据集是「state = 从臂末端位姿，action = 主臂关节量」，部署时按 **action** 选关节客户端。

### 12.1 jax → torch

```bash
cd /home/agilex/client
source .venv/bin/activate
# 1) jax → torch（config_name 必须与训练一致）
python examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir /home/agilex/checkpoints/piper_dual_stack_cups_pi05_25000_upload \
  --config_name pi05_piper_dual_stack_cups_eef_xyz3d_100 \
  --output_path /home/agilex/checkpoints/piper_dual_stack_cups_pi05_25000_pytorch \
  --precision bfloat16
```

参数：`--checkpoint_dir --config_name --output_path --precision`（下划线写法实测可用，kebab 亦可），`--precision` 取值 `float32|bfloat16|float16`（默认 bfloat16）。

### 12.2 拷贝归一化参数（必须）

```bash
# 2) 拷贝归一化参数（必须）
cp -a /home/agilex/checkpoints/piper_dual_stack_cups_pi05_25000_upload/assets \
      /home/agilex/checkpoints/piper_dual_stack_cups_pi05_25000_pytorch/
```

目标目录必须是实际 `--output_path`（旧文档写成 `..._pytorch_exact/`，与实际输出目录不一致）。

### 12.3 serve

```bash
# 3) server
python scripts/serve_policy.py --port=8000 policy:checkpoint \
  --policy.config=pi05_piper_dual_stack_cups_eef_xyz3d_100 \
  --policy.dir=/home/agilex/checkpoints/piper_dual_stack_cups_pi05_25000_pytorch
```

`scripts/serve_policy.py policy:checkpoint --policy.config=... --policy.dir=...` 语法合法（tyro 子命令，`--help` 实测通过）。

### 12.4 client（关节）

```bash
# 4) client：关节控制（本地栈）
python examples/piper_dual/main_dual_ros.py \
  --host 127.0.0.1 --port 8000 --bridge-python /usr/bin/python3 \
  --ros2-config examples/piper_dual/ros2_piper_dual_local.yaml \
  --control-stack local-ros \
  --prompt "Stack_the_paper_cups_together." \
  --no-dry-run --publish-actions --num-steps 800 --num-episodes 1
```

### 12.5 client（EEF，xyz+RPY 原生控制）

```bash
# 5) client：EEF（xyz+RPY，原生控制）
python examples/piper_dual/main_dual_eef_xyz3d_ros.py \
  --host 127.0.0.1 --port 8000 --bridge-python /usr/bin/python3 \
  --ros2-config examples/piper_dual/ros2_piper_dual_local.yaml --control-stack local-ros \
  --eef-left-topic /puppet/end_pose_left --eef-right-topic /puppet/end_pose_right \
  --eef-left-action-topic /pos_left_cmd --eef-right-action-topic /pos_right_cmd \
  --prompt "Stack_the_paper_cups_together." --max-action-delta 20 \
  --no-dry-run --publish-actions
```

`main_dual_eef_xyz3d_ros.py` 是唯一旧文档就正确的命令，其余命令一律向它对齐（`--control-stack local-ros` + `_local.yaml`）。

### 12.6 config 名一致性检查表

训练/服务配置名存在于 `src/openpi/training/config.py`：

| config_name | 行号 | 对应数据 |
|---|---|---|
| `pi05_piper_dual_stack_cups` | 876 | 关节/基础 stack cups |
| `pi05_piper_dual_stack_cups_shifted` | 908 | 关节 shifted |
| `pi05_piper_dual_stack_cups_eef_xyz3d_100` | 987 | EEF xyz3d |
| `pi05_piper_dual_fold_towel_eef_xyz3d_100` | 1079 | EEF xyz3d fold towel |

检查规则：**eef_xyz3d 数据集训练的模型必须用 `pi05_piper_dual_*_eef_xyz3d_100`**，不能用 `pi05_piper_dual_stack_cups`。转换脚本的 `--config_name` 与 `serve_policy.py` 的 `--policy.config` 必须完全一致，否则归一化统计与动作维度对不上。

---

## 13. 故障排查

| 症状 | 根因 | 处置 |
|---|---|---|
| 启动即报 `ValueError: ... missing required topic /joint_left` 或 `found forbidden topic /puppet/joint_left` | 控制栈与 yaml 不匹配（脚本默认 `official-ros`） | 显式加 `--control-stack local-ros` 与 `--config examples/piper_dual/ros2_piper_dual_local.yaml`；栈与 yaml 必须成对（第 2 节） |
| `uv run pytest` 收集阶段直接崩：`No module named 'lark'` / `PluginValidationError: unknown hook ...` | 终端已 `source /opt/ros/humble/setup.bash`，`PYTHONPATH` 把 ROS(py3.10) 的 pytest 插件 `launch_testing` / `launch_ros` 带给了 uv 环境(3.11) | 用 `env -u PYTHONPATH uv run pytest ...`，或加 `-p no:launch_ros -p no:launch_testing` |
| ROS 集成测试报 `rclpy._rclpy_pybind11` 导入/架构错误 | `examples/piper_dual/tests/` 的 3 个 ROS 集成测试在 3.11 venv 下无法运行（该扩展是 cpython-310） | 只能在 `/usr/bin/python3` + ROS 下跑 |
| 采集/部署脚本 import `rclpy` 失败 | 用 venv 的 3.11 跑了 ROS 侧代码 | 桥接进程用 `--bridge-python /usr/bin/python3`；直接跑 ROS 脚本用 conda `data` 环境并先 source ROS |
| 上传 HF 卡住 / 连接超时 | 代理变量干扰 HF 直连 | 用 `env -u ALL_PROXY -u all_proxy` 执行上传 |
| `HF_TOKEN` 未设置导致上传 401/403 | 当前 shell 未 export | `export HF_TOKEN='hf_你的Token'` |
| `ros2 topic list` 看不到对端话题 | `ROS_DOMAIN_ID` 不一致（.bashrc 设为 222；非交互/清理过环境的 shell 会丢） | 每个 ROS 终端确认 `export ROS_DOMAIN_ID=222` |
| `--num-episodes 1-` 解析报错 | 参数为 `type=int`，`1-` 非法 | 写 `--num-episodes 1` |
| 客户端跑完但没有实际动作 | `--dry-run` **默认 True**（不下发动作） | 真机执行加 `--no-dry-run`；注意 `--publish-actions` 与 `--dry-run` 互斥 |
| 数据集分落两处，上传脚本找不到 | 未统一 `HF_LEROBOT_HOME`，某条转换写到了 `~/.cache/huggingface/lerobot` | 转换前统一 `export HF_LEROBOT_HOME=/home/agilex/lgd/lerobot_datasets/`，上传目录用 `/home/agilex/lgd/lerobot_datasets/local` |
| 可视化脚本找不到 | `visualize_ros2_episodes.py` 不存在 | 改用 `examples/piper_dual/render_hdf5_batch.sh` |
| 机械臂启动脚本无反应/找不到 launch | `start_multi_piper.sh` 内部不 source ROS | 先 `source /opt/ros/humble/setup.bash` 和 `source /home/agilex/piper_ros/install/setup.bash` |
| 模型输出维度/归一化异常 | `config_name` 与训练数据集不一致 | 按 12.6 表核对，改成 `pi05_piper_dual_*_eef_xyz3d_100` 等匹配名 |
| 采集器启动后随即在 `cv2.imshow` 崩：`The function is not implemented. Rebuild the library with ... GTK+` | venv 中 `opencv-python-headless`（lerobot 依赖）覆盖了 GUI 版 `opencv-python`（openpi 依赖），磁盘上的 `cv2.abi3.so` 不链接 GUI 库 | 用 GUI 版覆盖安装：`uv pip install --python .venv/bin/python --force-reinstall --no-deps opencv-python==4.11.0.86`；复核 `ldd .venv/lib/python3.11/site-packages/cv2/cv2.abi3.so \| grep -cE "libGL\|libgtk\|libX11"` 应 > 0（0 即为 headless 生效） |
| 采集器只打印 `Previewing synchronized ROS 2 frames...` 但**不出预览窗口** | 同步帧永远凑不齐：EEF 采集器要求 3 路相机 + 4 路关节 + 2 路 EEF 全部落在 `--max-sync-error-ms` 内；缺任一相机即永远不渲染（窗口是渲染时才创建的） | 先 `rs-enumerate-devices -s` 确认 3 台在线，再 `ros2 topic hz /camera_{f_l,l,r}/color/image_raw` 确认三路都有数据；补上缺失的相机/话题后重启采集器 |
| 相机脚本报「相机未被 USB 枚举到」 | 该相机 USB 掉线（腕部相机随臂运动，线缆易被拉扯）或串号变更 | 按脚本打印的排查清单处理；串号确实变了就用 `setup_realsense_serials.py` 重采并更新脚本默认值 |
| 预览窗口打不开（`cannot open display` / 无窗口） | 进程环境缺少 `DISPLAY` 或 `XAUTHORITY`（非登录 shell、后台/托管启动时常见） | 传入 `DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority`；先用 `.venv/bin/python -c "import cv2,numpy as np; cv2.namedWindow('t'); cv2.imshow('t',np.zeros((10,10,3),np.uint8)); cv2.waitKey(1)"` 自检 |

---

## 14. 实机核查占位清单

- [ ] 三台相机串号与角色映射
  - 命令：`rs-enumerate-devices | grep -A1 "Serial Number"`
  - 期望：与 `start_realsense_3cam_color.sh` 中 FRONT/LEFT/RIGHT 一致
  - 实际：______
- [x] 三路彩色话题帧率（**2026-09-21 实测**：29.1 / 27.2 / 30.1 Hz，`ros2 topic hz`）
  - 命令：`ros2 topic hz /camera_f_l/color/image_raw`、`/camera_l/color/image_raw`、`/camera_r/color/image_raw`
  - 结果：三路均在 27–30 Hz
- [ ] 头部相机方向是否正确（当前 `front_rotation_enable=false`）
  - 命令：看采集器预览窗口的 cam_high 画面，或 `ros2 param get /camera_f_l rotation_filter.enable`
  - 期望：头部视角正立；若倒立则把该参数改回 `true` 且 `rotation_filter.rotation=180.0`
  - 实际：______
- [ ] `cam_high` 是否确为「中间/高视角」
  - 命令：遮挡法（逐一遮挡镜头观察哪一路掉帧/画面变化）
  - 期望：`cam_high` 对应中间/高视角那一路
  - 实际：______
- [ ] CAN 三个口激活状态
  - 命令：`ip -br link | grep can`
  - 期望：can_left / can_right / can0 为 UP
  - 实际：______
- [ ] 机械臂节点与话题
  - 命令：`ros2 node list`、`ros2 topic echo --once /piper_left_ctrl_node/arm_status`
  - 期望：`piper_left_ctrl_node`、`piper_right_ctrl_node` 在线且有状态输出
  - 实际：______
- [x] `auto_enable` 默认值（**2026-09-21 实测**：`start_two_piper.launch.py` 的 `default_value='true'`）
  - 命令：`grep -n auto_enable /home/agilex/piper_ros/src/piper/launch/start_two_piper.launch.py`
  - 结果：默认 `true` → 一 launch 即使能双臂；需要"后使能"请用 `auto_enable:=false` 启动（本手册 6.1 的命令即如此）
- [ ] 使能后从臂是否随主臂运动（`/enable_flag` 发 true 后推主臂）
  - 命令：`ros2 topic pub --once /enable_flag std_msgs/msg/Bool "{data: true}"` 后推动主臂，同时 `ros2 topic echo /puppet/joint_left`
  - 期望：`/puppet/joint_left` 随主臂变化；若不动，说明需要额外的遥操作命令源（节点本身不做跟随）
  - 实际：______
- [ ] 采集 dry-run 跑通
  - 命令：第 7.1 节命令（保持 `--dry-run` 默认）+ 按 `s` 录制
  - 期望：生成 `episode_000000.hdf5`
  - 实际：______
- [ ] 模型 checkpoint 是否已拉到本地
  - 命令：`ls /home/agilex/checkpoints`
  - 期望：存在 `piper_dual_stack_cups_pi05_25000_upload` 等目录
  - 实际：______
- [ ] `HF_TOKEN` 是否配置
  - 命令：`printenv HF_TOKEN >/dev/null && echo SET`
  - 期望：输出 `SET`
  - 实际：______
