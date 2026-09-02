# 五任务 EEF XYZ3D 100-Episode PI0.5 微调设计

## 目标

对五套已经转换为 LeRobot v2.1 的双臂 EEF XYZ3D 数据分别执行同构流程：上传为公开 Hugging Face 数据集、由训练服务器从 Hugging Face 拉取、计算独立归一化统计，并启动五个彼此隔离的 PI0.5 50,000-step 微调实验。

## 数据集

| 任务 | 本地目录 | Hugging Face 仓库 | 任务文本 |
|---|---|---|---|
| 叠纸杯 | `/home/agilex/lgd/lerobot_datasets/local/piper_dual_stack_cups_eef_xyz3d_100` | `HITdongdong/piper_dual_stack_cups_eef_xyz3d_100` | `Stack_the_paper_cups_together.` |
| 叠毛巾 | `/home/agilex/lgd/lerobot_datasets/local/piper_dual_fold_towel_eef_xyz3d_100` | `HITdongdong/piper_dual_fold_towel_eef_xyz3d_100` | `Fold_the_towel.` |
| 击鼓 | `/home/agilex/lgd/lerobot_datasets/local/piper_dual_beat_drum_eef_xyz3d_100` | `HITdongdong/piper_dual_beat_drum_eef_xyz3d_100` | `Beat_the_drum_three_times.` |
| 放积木入抽屉 | `/home/agilex/lgd/lerobot_datasets/local/piper_dual_block_drawer_eef_xyz3d_100` | `HITdongdong/piper_dual_block_drawer_eef_xyz3d_100` | `Put_the_block_in_the_drawer.` |
| 称苹果 | `/home/agilex/lgd/lerobot_datasets/local/piper_dual_weigh_apple_eef_xyz3d_100` | `HITdongdong/piper_dual_weigh_apple_eef_xyz3d_100` | `Weigh_the_apple.` |

五套数据均为 `piper_dual_ros2_eef`、100 episodes、30 FPS、14D observation/action、三路 224×224 相机，图像直接嵌入 Parquet。逐 episode 校验已确认 schema 一致、frame 索引连续、timestamp 单调、task index 为 0，且 `episodes.jsonl` 与 `episodes_stats.jsonl` 均覆盖 0–99。

- 叠纸杯：100 Parquet，61,758 frames，约 5.1 GiB。
- 叠毛巾：100 Parquet，82,048 frames，约 7.7 GiB。
- 击鼓：100 Parquet，30,501 frames，约 2.7 GiB。
- 放积木入抽屉：100 Parquet，54,204 frames，约 4.7 GiB。
- 称苹果：100 Parquet，50,067 frames，约 4.2 GiB。

## 仓库与传输

五个目标均创建为公开 `dataset` 仓库。上传不改写本地数据；大文件通过 `huggingface_hub` 的大目录上传能力分片和重试。每套数据在上传前冻结相对路径及逻辑字节数清单，上传后使用 HF API 获取全部远端 sibling 元数据，并对相对路径、文件大小、公开状态逐项比较。

服务器必须从对应 HF dataset repo 下载到独立路径，不用本地到服务器的直接复制代替这一验收。下载后重新验证 `meta/info.json`、100 个 Parquet、总帧数和任务文本。

## 训练配置

在现有 `LeRobotPiperEefXyz3dDataConfig` 和 `piper_eef_xyz3d_policy` 之上增加五个命名配置，不引入第二套 transform：

- `pi05_piper_dual_stack_cups_eef_xyz3d_100`
- `pi05_piper_dual_fold_towel_eef_xyz3d_100`
- `pi05_piper_dual_beat_drum_eef_xyz3d_100`
- `pi05_piper_dual_block_drawer_eef_xyz3d_100`
- `pi05_piper_dual_weigh_apple_eef_xyz3d_100`

共同契约：

- `Pi0Config(pi05=True, action_dim=32, action_horizon=50, discrete_state_input=False)`
- 全局 `batch_size=128`
- `num_workers=0`
- `num_train_steps=50_000`
- `save_interval=10_000`
- `keep_period=10_000`
- `fsdp_devices=4`
- PI0.5 base params 作为初始权重
- 各自的 repo ID、default prompt、norm stats、W&B run、实验名和 checkpoint 目录

OpenPI 训练循环为零基步编号。完成 50,000 次更新的最终 checkpoint 目录预计为 `49999`；周期节点为 `10000`、`20000`、`30000`、`40000`。不得通过多跑一步或重命名制造 `50000`。

## GPU 调度

最近一次观测中，9025 和 9026 上各 8 张 NVIDIA B20Z 均被其他训练占用；9022 连接超时，9024 本轮复查不可用。实现不得抢占、终止或复用他人进程。

每次启动前重新查询 GPU UUID、显存、利用率和 compute process。一个实验只在同一节点存在完整 4 张空闲 GPU 时启动。若多个节点分别有 4 张空闲卡，可并行启动；否则顺序启动。没有满足门槛的资源时，完成上传、下载、配置、norm stats 等所有可达工作，将训练标记为等待 GPU，而不是降低 FSDP 数或擅自改变 batch。

## 验证与故障处理

每个新增配置均采用测试先行：测试先证明新名字不存在，再增加最小配置并验证 model/data/training 字段以及 14D→32D padding/32D→14D unpadding 契约。

每个正式实验启动前：

1. 远端解析配置并输出关键字段。
2. 计算并验证专属 norm stats。
3. 构造实际 data loader batch，检查图像、state、action 和 prompt。
4. 执行短启动烟测，观察初始权重加载和首个有限 loss。
5. 再以独立日志和进程启动 50k 训练。

运行中验证每个 checkpoint 的 Orbax 完整标志、params、assets 和无临时目录。最终验收要求五个实验均到 `49999`、无 NaN/OOM/ENOSPC，并报告 checkpoint 路径、最终 loss、W&B URL、日志路径与数据仓库 URL。

## 安全边界

- HF token 不写入仓库、日志、命令输出或设计文档。
- 不删除或覆盖既有 checkpoint、dataset cache 或他人训练进程。
- 新实验使用唯一 exp name；目标目录存在时停止并检查，不使用 `--overwrite`。
- 上传/下载、配置、norm stats、烟测和正式训练每一层均以实际结果验收，不以进程创建或命令返回 0 代替行为验证。
