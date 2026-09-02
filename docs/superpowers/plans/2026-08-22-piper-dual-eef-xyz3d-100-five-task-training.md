# 五任务 EEF XYZ3D 100-Episode PI0.5 微调实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 stack-cups、fold-towel、beat-drum、block-drawer 与 weigh-apple 五套 100-episode EEF XYZ3D LeRobot 数据分别发布到公开 HF，并在可用的 4-GPU 资源上启动五个独立的 PI0.5 50k 微调实验。

**Architecture:** 代码侧增加五个复用现有 XYZ3D data factory/policy 的 `TrainConfig`，测试固定配置契约。运维侧用上传前后文件清单校验 HF，再由共享文件系统上的训练环境从 HF 下载并校验，计算各自 norm stats；GPU 门禁通过后分别烟测并后台运行。

**Tech Stack:** Python 3.11、pytest、OpenPI/JAX、LeRobot v2.1、PyArrow、huggingface_hub、Orbax、W&B、NVIDIA B20Z/FSDP。

## Global Constraints

- 五个数据集及训练实验必须独立，禁止合并 task index、norm stats 或 checkpoint。
- HF repos 必须公开：`HITdongdong/piper_dual_stack_cups_eef_xyz3d_100`、`HITdongdong/piper_dual_fold_towel_eef_xyz3d_100`、`HITdongdong/piper_dual_beat_drum_eef_xyz3d_100`、`HITdongdong/piper_dual_block_drawer_eef_xyz3d_100` 与 `HITdongdong/piper_dual_weigh_apple_eef_xyz3d_100`。
- 五个实验均为 `action_horizon=50`、全局 `batch_size=128`、`num_train_steps=50_000`、`save_interval=10_000`、`keep_period=10_000`、`fsdp_devices=4`。
- 禁止抢占或终止他人 GPU 进程；没有完整 4 张空闲卡时不得启动正式训练。
- token 不落盘、不进入 Git、不回显。
- OpenPI 零基最终 checkpoint 为 `49999`，不伪造 `50000`。

---

### Task 1: 固定双训练配置契约

**Files:**
- Modify: `tests/test_eef_xyz3d_config.py`
- Modify: `src/openpi/training/config.py`

**Interfaces:**
- Consumes: `config.get_config(name: str) -> TrainConfig`、`LeRobotPiperEefXyz3dDataConfig`。
- Produces: 配置名 `pi05_piper_dual_stack_cups_eef_xyz3d_100` 与 `pi05_piper_dual_fold_towel_eef_xyz3d_100`。

- [ ] **Step 1: 写入失败测试**

在 `tests/test_eef_xyz3d_config.py` 增加参数化契约：

```python
@pytest.mark.parametrize(
    ("name", "repo_id", "prompt"),
    [
        (
            "pi05_piper_dual_stack_cups_eef_xyz3d_100",
            "HITdongdong/piper_dual_stack_cups_eef_xyz3d_100",
            "Stack the paper cups together.",
        ),
        (
            "pi05_piper_dual_fold_towel_eef_xyz3d_100",
            "HITdongdong/piper_dual_fold_towel_eef_xyz3d_100",
            "Fold the towel.",
        ),
    ],
)
def test_eef_xyz3d_100_configs_match_training_contract(name, repo_id, prompt):
    cfg = config.get_config(name)
    data = cfg.data.create(cfg.assets_dirs, cfg.model)

    assert cfg.model.pi05 is True
    assert cfg.model.action_dim == 32
    assert cfg.model.action_horizon == 50
    assert cfg.model.discrete_state_input is False
    assert cfg.data.__class__.__name__ == "LeRobotPiperEefXyz3dDataConfig"
    assert cfg.data.repo_id == repo_id
    assert cfg.data.default_prompt == prompt
    assert cfg.batch_size == 128
    assert cfg.num_workers == 0
    assert cfg.num_train_steps == 50_000
    assert cfg.save_interval == 10_000
    assert cfg.keep_period == 10_000
    assert cfg.fsdp_devices == 4
    assert data.action_sequence_keys == ("action",)
```

- [ ] **Step 2: 验证 RED**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_eef_xyz3d_config.py::test_eef_xyz3d_100_configs_match_training_contract -q
```

Expected: 两个参数均因 `Config ... not found` 失败。

- [ ] **Step 3: 增加最小配置**

在现有 `pi05_piper_dual_stack_cups_eef_xyz3d` 后加入两个 `TrainConfig`。每个均使用：

```python
model=pi0_config.Pi0Config(
    pi05=True,
    action_dim=32,
    action_horizon=50,
    discrete_state_input=False,
),
data=LeRobotPiperEefXyz3dDataConfig(
    repo_id=repo_id,
    base_config=DataConfig(prompt_from_task=True),
    default_prompt=prompt,
),
weight_loader=weight_loaders.CheckpointWeightLoader(
    "/pfs/pfs-7jnepv/lgd/.cache/openpi/openpi-assets/checkpoints/pi05_base/params"
),
batch_size=128,
num_workers=0,
num_train_steps=50_000,
save_interval=10_000,
keep_period=10_000,
fsdp_devices=4,
```

- [ ] **Step 4: 验证 GREEN 和回归**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_eef_xyz3d_config.py tests/test_convert_jax_model_to_pytorch.py -q
```

Expected: 全部通过。

- [ ] **Step 5: 提交配置**

```bash
git add src/openpi/training/config.py tests/test_eef_xyz3d_config.py
git commit -m "feat: add xyz3d 100-episode training configs"
```

### Task 2: 发布并逐文件验证两个公开 HF 数据集

**Files:**
- Read-only source: `/home/agilex/lgd/lerobot_datasets/local/piper_dual_stack_cups_eef_xyz3d_100`
- Read-only source: `/home/agilex/lgd/lerobot_datasets/local/piper_dual_fold_towel_eef_xyz3d_100`
- Create outside Git: `/home/agilex/lgd/logs/eef_xyz3d_100_publication/local_manifests.json`
- Create outside Git: `/home/agilex/lgd/logs/eef_xyz3d_100_publication/hf_audit.json`

**Interfaces:**
- Consumes: two validated LeRobot directories and process-scoped HF authentication.
- Produces: two public dataset repos whose path/size sets exactly equal local manifests.

- [ ] **Step 1: 冻结本地清单**

对两个 root 递归记录每个普通文件的相对 POSIX path 与 `st_size`，并记录 `meta/info.json` 的 total episodes/frames。忽略上传客户端自己生成的 `.cache/huggingface`，但不得忽略数据文件。

- [ ] **Step 2: 核验认证而不输出 token**

通过 `HfApi().whoami(token=True)` 仅输出账号名；要求为 `HITdongdong`。认证失败则停止上传，不把 token 放到命令行。

- [ ] **Step 3: 创建公开仓库并上传**

对每个 repo 执行：

```python
api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
api.upload_large_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path=str(local_root),
)
```

若目标仓库预先存在，先审计内容；只允许补齐/更新与本地同名数据文件，不删除未知远端文件，遇到冲突停止报告。

- [ ] **Step 4: 远端精确审计**

调用 `dataset_info(repo_id, files_metadata=True)`，断言 `private is False`；对每个清单 path 查找同名 sibling 并比较 size。远端允许 HF 自动生成的 `.gitattributes`，除此以外不得出现未知数据文件。

- [ ] **Step 5: 保存无密钥审计记录**

记录 repo URL、commit SHA、文件数、逻辑字节数、公开状态及验证时间；不得记录 token 或 authorization header。

### Task 3: 同步代码并从 HF 下载数据到服务器

**Files:**
- Remote code: `/pfs/pfs-7jnepv/lgd/SANE/src/openpi/training/config.py`
- Remote test: `/pfs/pfs-7jnepv/lgd/SANE/tests/test_eef_xyz3d_config.py`
- Remote datasets: `/pfs/pfs-7jnepv/lgd/datasets/HITdongdong/piper_dual_stack_cups_eef_xyz3d_100`
- Remote datasets: `/pfs/pfs-7jnepv/lgd/datasets/HITdongdong/piper_dual_fold_towel_eef_xyz3d_100`

**Interfaces:**
- Consumes: Task 1 commit and Task 2 public repos.
- Produces: server-resolvable configs and HF-origin datasets matching 61,758/82,048 frames.

- [ ] **Step 1: 同步 Task 1 精确变更**

在远端 Git 工作区无冲突的前提下应用 Task 1 commit；若远端有并发未提交修改，先保存状态并只应用两文件的目标 hunks，不覆盖用户工作。

- [ ] **Step 2: 远端运行同一测试**

```bash
cd /pfs/pfs-7jnepv/lgd/SANE
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest tests/test_eef_xyz3d_config.py -q
```

Expected: 全部通过。

- [ ] **Step 3: 从 HF 下载两个快照**

使用远端 `huggingface_hub.snapshot_download(repo_type="dataset", local_dir=...)`；记录解析到的 commit SHA。目标存在时先比较 manifest，不盲目覆盖。

- [ ] **Step 4: 远端内容验证**

使用 PyArrow 检查两个目录各有 100 个 Parquet、统一 schema、episode index 0–99；断言 cups rows 为 61,758、towel rows 为 82,048，并检查各自 `tasks.jsonl`。

### Task 4: 计算独立 norm stats 并验证配置数据链路

**Files:**
- Remote assets: `/pfs/pfs-7jnepv/lgd/SANE/assets/HITdongdong/piper_dual_stack_cups_eef_xyz3d_100/norm_stats.json`
- Remote assets: `/pfs/pfs-7jnepv/lgd/SANE/assets/HITdongdong/piper_dual_fold_towel_eef_xyz3d_100/norm_stats.json`

**Interfaces:**
- Consumes: Task 3 configs and downloaded datasets.
- Produces: each config's own finite 14D state/action statistics.

- [ ] **Step 1: 解析并打印配置关键字段**

分别 `get_config`，断言 repo ID、horizon 50、batch 128、steps 50k、save/keep 10k、FSDP 4。

- [ ] **Step 2: 运行 norm stats 工具**

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 .venv/bin/python scripts/compute_norm_stats.py --config-name pi05_piper_dual_stack_cups_eef_xyz3d_100
XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 .venv/bin/python scripts/compute_norm_stats.py --config-name pi05_piper_dual_fold_towel_eef_xyz3d_100
```

一次只运行一个，避免争用 GPU/内存；若工具只需 CPU，则显式使用 CPU backend。

- [ ] **Step 3: 验证统计文件**

解析两个 JSON，检查 state/action 统计均为 14D、所有数值 finite、标准差/quantile 合法，且 asset path 分属各自 repo ID。

- [ ] **Step 4: 实际加载首个 batch**

使用每个配置构建 data loader，取得一个 batch；断言全局 batch 128，state pad 后 32D，actions 为 `(128, 50, 32)`，三路图像和非空 prompt 存在。

### Task 5: GPU 门禁、烟测和正式训练

**Files:**
- Remote checkpoints: `/pfs/pfs-7jnepv/lgd/SANE/checkpoints/pi05_piper_dual_stack_cups_eef_xyz3d_100/<exp_name>`
- Remote checkpoints: `/pfs/pfs-7jnepv/lgd/SANE/checkpoints/pi05_piper_dual_fold_towel_eef_xyz3d_100/<exp_name>`
- Remote logs: `/pfs/pfs-7jnepv/lgd/logs/eef_xyz3d_100_training/`

**Interfaces:**
- Consumes: Task 4 verified configs/assets and exactly four idle GPUs per experiment.
- Produces: two independent 50k processes or an explicit resource-blocked state with all prerequisites complete.

- [ ] **Step 1: 重新盘点所有可连接节点 GPU**

使用 `nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits` 及 compute-app 查询。空闲要求：无 compute process、显存使用低于 1 GiB、利用率为 0；必须同一节点至少四张。

- [ ] **Step 2: 检查唯一输出目录**

生成包含 task 与 UTC timestamp 的 exp name。若 checkpoint 目录已存在，停止而不是传 `--overwrite`。检查 GPFS 剩余空间足够保存 5 个完整 checkpoint/实验。

- [ ] **Step 3: 每个配置执行短烟测**

在隔离临时 checkpoint 目录运行训练到至少产生一个 finite loss；验证 PI0.5 base params 加载、4 个 FSDP device、实际 batch shape 与 W&B 初始化。烟测完成后优雅停止并删除仅由本次创建的临时烟测目录。

- [ ] **Step 4: 启动正式训练**

在各自日志中运行：

```bash
CUDA_VISIBLE_DEVICES=<four-idle-indices> XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  .venv/bin/python scripts/train.py <config-name> --exp-name <unique-exp-name>
```

若只有一个可用 4-GPU 组，先启动一个，第二个保持资源等待；不得改成少于 4 GPU 或改变 batch 128。

- [ ] **Step 5: 验证运行态**

确认进程存活、命令行配置名正确、GPU UUID 对应所选空闲卡、首个 finite loss 已出现、无 OOM/NaN/ENOSPC，并记录 W&B run ID 与 PID。

### Task 6: 节点与最终验收

**Files:**
- Read: Task 5 logs/checkpoints
- Create outside Git: `/pfs/pfs-7jnepv/lgd/logs/eef_xyz3d_100_training/final_audit.json`

**Interfaces:**
- Consumes: Task 5 training processes.
- Produces: evidence-backed checkpoint audit for both tasks.

- [ ] **Step 1: 监控语义状态而非固定睡眠**

基于日志最新 step、进程存活、checkpoint 完整标志和 hard-failure regex 判断状态；不以轮询次数或预计耗时判断完成。

- [ ] **Step 2: 验证周期 checkpoint**

每个实验对 `10000`、`20000`、`30000`、`40000` 逐项检查 Orbax metadata、params、sharding 与 assets；禁止残留临时保存目录。

- [ ] **Step 3: 验证最终 checkpoint**

确认训练完成 50,000 次更新并生成完整 `49999`；日志无 NaN、OOM、ENOSPC、RESOURCE_EXHAUSTED 或未处理 traceback。

- [ ] **Step 4: 写入最终审计并交付**

每个实验记录 config、dataset HF URL/SHA、checkpoint 路径、完成节点、最后 finite loss、W&B URL、日志路径和验证时间。最终回复严格区分“已启动”“运行中”“已完成”；只有实际到 `49999` 才称完成。

### Task 7: 固定 beat-drum 训练配置契约

**Files:**
- Modify: `tests/test_eef_xyz3d_config.py`
- Modify: `src/openpi/training/config.py`

**Interfaces:**
- Consumes: Task 1 的参数化 XYZ3D 100-episode 配置测试与 `LeRobotPiperEefXyz3dDataConfig`。
- Produces: 配置名 `pi05_piper_dual_beat_drum_eef_xyz3d_100`。

- [ ] **Step 1: 写入失败测试**

在 `test_eef_xyz3d_100_configs_match_training_contract` 的参数列表中加入：

```python
(
    "pi05_piper_dual_beat_drum_eef_xyz3d_100",
    "HITdongdong/piper_dual_beat_drum_eef_xyz3d_100",
    "Beat the drum three times.",
),
```

- [ ] **Step 2: 验证 RED**

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_eef_xyz3d_config.py::test_eef_xyz3d_100_configs_match_training_contract -k beat_drum -q
```

Expected: 因 `Config 'pi05_piper_dual_beat_drum_eef_xyz3d_100' not found` 失败。

- [ ] **Step 3: 增加最小配置**

复用 Task 1 已建立的配置模式，唯一任务特定字段为：

```python
name="pi05_piper_dual_beat_drum_eef_xyz3d_100"
repo_id="HITdongdong/piper_dual_beat_drum_eef_xyz3d_100"
default_prompt="Beat the drum three times."
```

模型与训练字段必须为 `pi05=True`、`action_dim=32`、`action_horizon=50`、`discrete_state_input=False`、`batch_size=128`、`num_workers=0`、`num_train_steps=50_000`、`save_interval=10_000`、`keep_period=10_000`、`fsdp_devices=4`，并使用与 Task 1 相同的 PI0.5 base params loader。

- [ ] **Step 4: 验证 GREEN 和回归**

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_eef_xyz3d_config.py tests/test_convert_jax_model_to_pytorch.py -q
```

Expected: 全部通过，参数化测试覆盖三个配置。

- [ ] **Step 5: 提交配置**

```bash
git add src/openpi/training/config.py tests/test_eef_xyz3d_config.py
git commit -m "feat: add beat drum xyz3d training config"
```

### Task 8: 发布并逐文件验证 beat-drum HF 数据集

**Files:**
- Read-only source: `/home/agilex/lgd/lerobot_datasets/local/piper_dual_beat_drum_eef_xyz3d_100`
- Modify outside Git: `/home/agilex/lgd/logs/eef_xyz3d_100_publication/local_manifests.json`
- Modify outside Git: `/home/agilex/lgd/logs/eef_xyz3d_100_publication/hf_audit.json`

**Interfaces:**
- Consumes: 已验证的 100-episode beat-drum LeRobot 目录与进程级 HF 认证。
- Produces: 公开 dataset repo `HITdongdong/piper_dual_beat_drum_eef_xyz3d_100`，远端路径/大小集合与本地清单精确相等。

- [ ] **Step 1: 冻结本地清单**

递归记录每个普通文件的相对 POSIX path 与 `st_size`，排除仅由上传客户端生成的 `.cache/huggingface`。断言 100 个 Parquet、30,501 frames、任务文本 `Beat_the_drum_three_times.`。

- [ ] **Step 2: 安全创建并上传公开仓库**

使用 `HfApi(token=True)` 的现有缓存认证；不读取或输出 token。创建 `repo_type="dataset"`、`private=False` 的目标仓库，再用 `upload_large_folder` 上传。仓库已存在时先审计，未知或冲突远端文件导致停止，禁止删除。

- [ ] **Step 3: 远端精确审计**

通过 `dataset_info(files_metadata=True)` 断言 `private is False`，逐项比较全部相对路径和 size；只允许 `.gitattributes` 为额外文件。把 repo URL、SHA、文件数、字节数和时间写入两个无密钥审计 JSON。

### Task 9: 远端同步并从 HF 下载 beat-drum 数据

**Files:**
- Remote code: `/pfs/pfs-7jnepv/lgd/SANE/src/openpi/training/config.py`
- Remote test: `/pfs/pfs-7jnepv/lgd/SANE/tests/test_eef_xyz3d_config.py`
- Remote dataset: `/pfs/pfs-7jnepv/lgd/datasets/HITdongdong/piper_dual_beat_drum_eef_xyz3d_100`

**Interfaces:**
- Consumes: Task 7 commit 与 Task 8 的公开 repo/SHA。
- Produces: 远端可解析配置和从 HF 获取的 100-episode、30,501-frame 数据目录。

- [ ] **Step 1: 同步配置并运行测试**

仅在不覆盖远端并发修改的前提下应用 Task 7 变更，然后运行：

```bash
cd /pfs/pfs-7jnepv/lgd/SANE
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest tests/test_eef_xyz3d_config.py -q
```

- [ ] **Step 2: 从 HF 下载并验证**

使用 `snapshot_download(repo_type="dataset", local_dir=...)`，记录解析 SHA。使用 PyArrow 断言 100 个 Parquet、统一 schema、episode 0–99、总 rows 30,501、连续 frame index、单调 timestamp、task index 0 和正确任务文本。

### Task 10: 计算并验证 beat-drum norm stats

**Files:**
- Remote asset: `/pfs/pfs-7jnepv/lgd/SANE/assets/HITdongdong/piper_dual_beat_drum_eef_xyz3d_100/norm_stats.json`

**Interfaces:**
- Consumes: Task 9 远端配置与 HF 数据。
- Produces: 独立、finite 的 14D state/action 统计与可加载 batch。

- [ ] **Step 1: 计算统计**

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 .venv/bin/python scripts/compute_norm_stats.py --config-name pi05_piper_dual_beat_drum_eef_xyz3d_100
```

- [ ] **Step 2: 验证统计和实际 batch**

检查 state/action 统计均为 14D且全部 finite；通过配置构造一个实际 batch，断言全局 batch 128、state pad 到 32D、actions `(128, 50, 32)`、三路图像和非空 beat-drum prompt。

### Task 11: GPU 门禁、启动并审计 beat-drum 训练

**Files:**
- Remote checkpoints: `/pfs/pfs-7jnepv/lgd/SANE/checkpoints/pi05_piper_dual_beat_drum_eef_xyz3d_100/<exp_name>`
- Remote logs: `/pfs/pfs-7jnepv/lgd/logs/eef_xyz3d_100_training/`
- Modify outside Git: `/pfs/pfs-7jnepv/lgd/logs/eef_xyz3d_100_training/final_audit.json`

**Interfaces:**
- Consumes: Task 10 通过验证的配置/assets 与同一节点恰好四张空闲 GPU。
- Produces: 独立 50k beat-drum 训练或明确的 GPU 资源等待状态。

- [ ] **Step 1: 重新执行 GPU 与磁盘门禁**

空闲条件为无 compute process、显存使用低于 1 GiB、利用率为 0；必须同一节点至少四张。不得抢占他人进程或更改 FSDP/batch。检查唯一 exp name、目标目录不存在且 GPFS 空间足够。

- [ ] **Step 2: 烟测并启动正式训练**

先运行到首个 finite loss，验证 base params、4 个 FSDP device、batch shape 和 W&B。成功后用四张已验证空闲卡启动配置 `pi05_piper_dual_beat_drum_eef_xyz3d_100`，保存独立 PID、日志和 W&B ID。

- [ ] **Step 3: 节点与最终验收**

逐项验证完整 `10000`、`20000`、`30000`、`40000` 与最终 `49999` 的 Orbax metadata、params、sharding、assets 和无临时目录；日志不得含 NaN/OOM/ENOSPC/RESOURCE_EXHAUSTED/未处理 traceback。只有实际到 `49999` 才记录为完成。

### Task 12: 固定 block-drawer 训练配置契约

**Files:**
- Modify: `tests/test_eef_xyz3d_config.py`
- Modify: `src/openpi/training/config.py`

**Interfaces:**
- Consumes: 现有 XYZ3D 100-episode 参数化测试和 `LeRobotPiperEefXyz3dDataConfig`。
- Produces: `pi05_piper_dual_block_drawer_eef_xyz3d_100`。

- [ ] **Step 1: 写入失败测试**

在参数列表加入：

```python
(
    "pi05_piper_dual_block_drawer_eef_xyz3d_100",
    "HITdongdong/piper_dual_block_drawer_eef_xyz3d_100",
    "Put the block in the drawer.",
),
```

- [ ] **Step 2: 验证 RED**

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_eef_xyz3d_config.py::test_eef_xyz3d_100_configs_match_training_contract -k block_drawer -q
```

Expected: `Config 'pi05_piper_dual_block_drawer_eef_xyz3d_100' not found`。

- [ ] **Step 3: 增加最小配置**

任务特定字段为 `name="pi05_piper_dual_block_drawer_eef_xyz3d_100"`、`repo_id="HITdongdong/piper_dual_block_drawer_eef_xyz3d_100"`、`default_prompt="Put the block in the drawer."`。其余模型、loader 和训练字段必须与另外三个 100-episode 配置完全一致。

- [ ] **Step 4: 验证并提交**

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_eef_xyz3d_config.py tests/test_convert_jax_model_to_pytorch.py -q
git add src/openpi/training/config.py tests/test_eef_xyz3d_config.py
git commit -m "feat: add block drawer xyz3d training config"
```

### Task 13: 发布并逐文件验证 block-drawer HF 数据集

**Files:**
- Read-only source: `/home/agilex/lgd/lerobot_datasets/local/piper_dual_block_drawer_eef_xyz3d_100`
- Modify outside Git: `/home/agilex/lgd/logs/eef_xyz3d_100_publication/local_manifests.json`
- Modify outside Git: `/home/agilex/lgd/logs/eef_xyz3d_100_publication/hf_audit.json`
- Modify outside Git: `/home/agilex/lgd/logs/eef_xyz3d_100_publication/source_preservation_audit.json`

**Interfaces:**
- Consumes: 已验证的 block-drawer 数据与缓存 HF 认证。
- Produces: 公开 repo `HITdongdong/piper_dual_block_drawer_eef_xyz3d_100`。

- [ ] **Step 1: 冻结并上传**

冻结 path/size/mtime，断言 100 Parquet、54,204 frames、任务 `Put_the_block_in_the_drawer.`。安全预审远端后用 `upload_large_folder` 上传公开 dataset repo；不得输出 token 或代理值。

- [ ] **Step 2: 精确远端与源保全审计**

远端除 `.gitattributes` 外须与 104 个本地非缓存文件逐 path/size 相等。上传后非缓存源 path/size/mtime 必须与冻结清单一致，仅允许 `.cache/huggingface/**` 变化。合并写入三个 JSON，保留前三个数据集条目。

### Task 14: 远端同步并从 HF 下载 block-drawer

**Files:**
- Remote config/test: `/pfs/pfs-7jnepv/lgd/SANE/src/openpi/training/config.py`, `/pfs/pfs-7jnepv/lgd/SANE/tests/test_eef_xyz3d_config.py`
- Remote dataset: `/pfs/pfs-7jnepv/lgd/datasets/HITdongdong/piper_dual_block_drawer_eef_xyz3d_100`

- [ ] **Step 1: 精确同步并测试**

保留远端并发修改，先备份，再只加入 block-drawer 配置/测试行；远端 focused test 必须通过。

- [ ] **Step 2: 固定 revision 下载与验证**

以 `snapshot_download(repo_type="dataset")` 从公开 HF 下载固定 SHA。PyArrow 逐 episode 验证 100 Parquet、54,204 rows、schema、索引、timestamp、task；冲突检查后绑定到 LeRobot 默认 cache 路径并用 `LeRobotDatasetMetadata` 离线解析。

### Task 15: 计算并验证 block-drawer norm stats

- [ ] **Step 1: CPU 计算**

```bash
JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 .venv/bin/python scripts/compute_norm_stats.py --config-name pi05_piper_dual_block_drawer_eef_xyz3d_100
```

- [ ] **Step 2: 统计和 batch 验证**

检查独立 state/action 统计为 14D、finite、quantile 合法；加载实际 batch，断言全局 128、horizon 50、pad 后 state/actions 32D、三路图像和正确 prompt。

### Task 16: GPU 门禁、启动并审计 block-drawer 训练

- [ ] **Step 1: GPU/磁盘门禁**

仅当同一节点四张卡均无 compute process、显存使用低于 1 GiB、利用率为 0 时继续；检查唯一 exp name、目录不存在和 GPFS 空间。不得抢占或降低 batch/FSDP。

- [ ] **Step 2: 烟测与正式启动**

运行至首个 finite loss，验证 base params、4-device FSDP、实际 batch 和 W&B；再启动独立 50k 训练。

- [ ] **Step 3: 节点验收**

验证完整 `10000/20000/30000/40000/49999` 及无 NaN/OOM/ENOSPC/RESOURCE_EXHAUSTED/未处理 traceback；只有实际到 `49999` 才标记完成。

### Task 17: 固定 weigh-apple 训练配置契约

**Files:**
- Modify: `tests/test_eef_xyz3d_config.py`
- Modify: `src/openpi/training/config.py`

- [ ] **Step 1: RED**

向既有参数化测试加入 `("pi05_piper_dual_weigh_apple_eef_xyz3d_100", "HITdongdong/piper_dual_weigh_apple_eef_xyz3d_100", "Weigh the apple.")`，运行 `-k weigh_apple` 并确认因配置不存在而失败。

- [ ] **Step 2: GREEN 与回归**

增加同模式 TrainConfig；任务字段仅为上述 name/repo/prompt，其余字段严格复用 100-episode 契约。运行：

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/test_eef_xyz3d_config.py tests/test_convert_jax_model_to_pytorch.py -q
```

- [ ] **Step 3: 提交**

只提交配置和测试，提交信息 `feat: add weigh apple xyz3d training config`。

### Task 18: 发布并验证 weigh-apple HF 数据集

- [ ] **Step 1: 冻结与上传**

冻结 `/home/agilex/lgd/lerobot_datasets/local/piper_dual_weigh_apple_eef_xyz3d_100` 的非缓存 path/size/mtime，断言 100 Parquet、50,067 frames、任务 `Weigh_the_apple.`；预审并上传公开 repo `HITdongdong/piper_dual_weigh_apple_eef_xyz3d_100`。

- [ ] **Step 2: 远端与源保全审计**

远端除 `.gitattributes` 外须与 104 个 payload 文件逐 path/size 相等。源非缓存 path/size/mtime 不变，仅允许上传客户端 cache 变化。合并审计 JSON 并保留前四个数据集。

### Task 19: 远端同步并从 HF 下载 weigh-apple

- [ ] **Step 1: 精确同步并测试**

备份远端并发修改，只插入 weigh-apple 配置与测试行；focused test 必须通过。

- [ ] **Step 2: 固定 revision 下载与验证**

从公开 HF repo 以 `snapshot_download(repo_type="dataset")` 下载固定 SHA；逐 episode 验证 100 Parquet、50,067 rows、schema、索引、timestamp 和任务，然后 conflict-check 绑定到 LeRobot 默认路径并解析。

### Task 20: 计算并验证 weigh-apple norm stats

- [ ] **Step 1: CPU 计算**

```bash
JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 .venv/bin/python scripts/compute_norm_stats.py --config-name pi05_piper_dual_weigh_apple_eef_xyz3d_100
```

- [ ] **Step 2: 数值与 batch 证据**

验证独立 14D finite stats、真实 `(128,32)` state、`(128,50,32)` actions、三路图像，并逐行证明 prompt token/mask 等于 `Weigh_the_apple.` 的 tokenizer 输出。

### Task 21: GPU 门禁、启动并审计 weigh-apple

- [ ] **Step 1: 严格门禁**

仅同节点四张卡均无 compute process、显存低于 1 GiB、利用率 0 时继续；不得抢占或改变 batch/FSDP。

- [ ] **Step 2: 烟测、正式启动和节点验收**

验证首个 finite loss、base params、FSDP 4、batch、W&B 后启动独立 50k；验收 `10000/20000/30000/40000/49999` 和无硬错误。
