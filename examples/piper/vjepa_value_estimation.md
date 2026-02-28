## 用 SRPO/V-JEPA 生成 GraspAnything 的逐帧价值（HDF5）与可视化

本文档对应脚本：
- `scripts/estimate_vjepa_value_hdf5.py`：从 HDF5 计算逐帧 V-JEPA 价值（支持 `progress_warp` / `segment_match50`）
- `scripts/viz_vjepa_value_curve.py`：交互式查看单条 episode 的图像预览 + 价值曲线
- `scripts/check_segment_match50_clustering.py`：检查 `segment_match50` 的分段成功中心聚类效果（kNN 距离尺度 / noise% / 可选 PCA）
- `scripts/check_segmatch50_diffusion.py`：检查 `segment_match50 + diffusion` 的效果（段混淆矩阵 / success&failure cost 分布 / 示例 cost heatmap）

`estimate_vjepa_value_hdf5.py` 支持两种价值曲线方法（由 `--value-method` 选择）：
- `progress_warp`（默认）：距离→进度 + 末帧缩放（可选单调化）+ 全局单调 warp \(f\\)
- `segment_match50`：50 段成功中心 + 滑窗局部匹配 + 插值生成价值曲线

### 1) 背景与定义

我们用 `siiRL/vjepa2` 的 V-JEPA 世界模型把视频片段映射到潜在表征空间，并用 SRPO 的“成功聚类中心距离”思想构造逐帧价值：

- 在时间步 \(t\)，取前缀片段 \(o_{0:t}\)
- 用 V-JEPA 得到该片段 embedding \(e_t\)
- 计算到最近成功中心的欧氏距离（在成功集的 StandardScaler 标准化空间里，与 DBSCAN 聚类一致）：
  \[
  \\tilde x = (x-\\mu)/\\sigma,\\quad d_t = \min_k \\lVert \\tilde e_t - \\tilde c_k \\rVert_2
  \]
  脚本会基于成功 embedding 的模长统计（`norm_cv` / `ratio_p95_p5`）**自动判断是否需要先做 L2-normalize**，决策与统计记录在 `cluster_centers.npz` 的 `info` 中。
- 先把距离映射成“进度值”（不再使用 sigmoid / \(d_{\\min},d_{\\max}\)）：令 \(d_0=d_{t=0}\)，
  \[
  u_t = \\max(0, d_0 - d_t),\\quad
  V^{base}_t = \\frac{u_t}{\\max(\\epsilon, u_{T-1})}
  \]
  这样保证 \(V^{base}_0=0\)，并且（当 \(u_{T-1}>0\) 时）\(V^{base}_{T-1}=1\)。
- 再对曲线做**末帧缩放**，保证“末帧=全曲线最大值”：
  - 成功轨迹：末帧缩放到 1
  - 失败轨迹（rollouts_failure）：末帧缩放到 `--failure-end-value`（默认 0.5）
  若设置 `--enforce-monotonic`，会额外对曲线做 `cummax` 强制非降（关闭时允许出现“进度倒退”导致的下降）。
- 最后（默认开启）学习一个全局单调变换 \(f\\)（isotonic regression），使 success_ref 的曲线尽量接近线性时间进度：
  \[
  f\\big(V^{mono}\\big) \\approx \\frac{t}{T-1}
  \]
  并把同一个 \(f\\) 应用到 rollouts 的曲线上（缓存为 `<DIR>/warp_f.npz`）。

另外可以用 `--minmax-skip-prefix-k K` 强制 `value_pred[:K]=0`（只影响输出，不影响 \(d_0=d_{t=0}\) 的取值；用于避免起始阶段 shaping 干扰）。

### 2) 数据目录与成功演示的两种定义

输入 HDF5 两类来源：
- 人类遥操作（默认都视为成功演示）：`/home/ztlab/project/ELM/openpi/datasets/GraspAnything/hdf5`
- VLA 自主 rollouts（包含成功/失败）：`/home/ztlab/project/ELM/openpi/datasets/GraspAnything/rollouts_hdf5/{success,failure}`

成功演示集合（success reference）可二选一：
- **方式 A（推荐起步）**：`--success-mode teleop_all`
  成功参考集 = teleop 全部 HDF5（都视为成功演示）
- **方式 B（只用 rollouts）**：`--success-mode rollouts_shortest_p --shortest-p 0.3`
  成功参考集 = rollouts_success 中“裁剪后时长最短的前 p%”
  评分对象 = 剩余 rollouts_success + 全部 rollouts_failure

### 3) 双相机融合（cam_high + cam_left_wrist）

脚本提供 `--camera-mode both`，并支持融合策略 `--camera-fusion`：

- `dist_min`（默认，推荐）：分别算两路距离 \(d_t^{high}, d_t^{wrist}\)，取
  \(d_t = \\min(d_t^{high}, d_t^{wrist})\)
  直觉：**任一视角“看起来像成功”就给高分**，对遮挡更鲁棒。
- `dist_mean`：
  \(d_t = (d_t^{high}+d_t^{wrist})/2\)
  更保守，要求两路都接近成功才高分。
- `emb_mean`：先把 embedding 均值融合，再在融合空间聚类/算距离（更像“同时用两路信息”，但对视角域偏移更敏感）
- `emb_concat`：拼接两路 embedding（维度翻倍），再聚类/算距离（最“信息完整”，但对聚类/尺度更敏感）

### 4) 运行：估计逐帧价值（写缓存）

> 提醒：`--overwrite` / `--no-gui` 这类参数是 **flag**，直接写 `--overwrite` 即表示启用；不要写 `--overwrite true`。
> 
> 依赖方面：脚本会自动将 `siiRL/vjepa2` 加入 `sys.path`，无需手动设置 `PYTHONPATH`；成功聚类（DBSCAN）已内置实现，不依赖 `sklearn`；同时已为 `timm` 的 `drop_path` 提供 fallback（无需额外安装）。强烈建议使用 GPU（`--device auto`）运行。

#### 4.1 以 teleop 作为成功参考、仅对 rollouts 评分（默认）

```bash
cd /home/ztlab/project/ELM/openpi

.venv/bin/python scripts/estimate_vjepa_value_hdf5.py \
  --success-mode teleop_all \
  --score-sources rollouts \
  --camera-mode high \
  --out-cache-dir /home/ztlab/project/ELM/openpi/datasets/GraspAnything/vjepa_value_cache/vjepa_rollouts_from_teleop_v1 \
  --timestep-stride 6 \
  --embed-batch-size 8 \
  --embed-cache-dir ./datasets/GraspAnything/vjepa_value_cache/embedding_cache
```

#### 4.2 双相机联合输出价值（推荐 `dist_min`）

```bash
cd /home/ztlab/project/ELM/openpi

.venv/bin/python scripts/estimate_vjepa_value_hdf5.py \
  --success-mode teleop_all \
  --score-sources rollouts \
  --camera-mode both \
  --camera-fusion dist_min \
  --out-cache-dir /home/ztlab/project/ELM/openpi/datasets/GraspAnything/vjepa_value_cache/vjepa_rollouts_both_dist_min_v2 \
  --timestep-stride 6 \
  --embed-batch-size 8 \
  --minmax-skip-prefix-k 25 \
  --embed-cache-dir ./datasets/GraspAnything/vjepa_value_cache/embedding_cache \
  --max-score-rollouts-failure-episodes 1 \
  --max-score-rollouts-success-episodes 1
```

#### 4.3 只用 rollouts：最短 30% 成功作为成功参考

```bash
cd /home/ztlab/project/ELM/openpi

.venv/bin/python scripts/estimate_vjepa_value_hdf5.py \
  --success-mode rollouts_shortest_p \
  --shortest-p 0.3 \
  --score-sources rollouts \
  --camera-mode both \
  --camera-fusion dist_min \
  --out-cache-dir /home/ztlab/project/ELM/openpi/datasets/GraspAnything/vjepa_value_cache/vjepa_rollouts_short30_both_v1 \
  --timestep-stride 6 \
  --embed-batch-size 8
```

#### 4.4 Segment-Match50（50 段成功中心 + 滑窗局部匹配）

该方法通过 success_ref 构建“每个时间段的成功中心”，再对 rollouts 用滑动窗口做局部匹配，最后用插值生成逐帧价值曲线。

##### 4.4.1 distance / gaussian_nll（基于成功中心距离）

```bash
cd /home/ztlab/project/ELM/openpi

.venv/bin/python scripts/estimate_vjepa_value_hdf5.py \
  --value-method segment_match50 \
  --success-mode teleop_all \
  --teleop-hdf5-dir ./datasets/GraspAnything/success_ref \
  --score-sources rollouts \
  --camera-mode both \
  --camera-fusion dist_min \
  --dbscan-eps 40 \
  --segment-match-num-segments 50 \
  --segment-match-window-stride 6 \
  --segment-match-radius 3 \
  --segment-match-first-fixed \
  --segment-match-value-mode soft_next \
  --segment-match-path viterbi \
  --segment-match-time-prior-success 1.0 \
  --segment-match-back-penalty-success 5.0 \
  --out-cache-dir /home/ztlab/project/ELM/openpi/datasets/GraspAnything/vjepa_value_cache/vjepa_rollouts_segmatch50_both_dist_min_v1 \
  --embed-batch-size 8 \
  --minmax-skip-prefix-k 0 \
  --embed-cache-dir ./datasets/GraspAnything/vjepa_value_cache/embedding_cache \
  --max-score-rollouts-failure-episodes 1 \
  --max-score-rollouts-success-episodes 2
```

##### 4.4.2 diffusion（基于条件扩散去噪误差，替代“到中心的距离”）

该模式会在 `out-cache-dir` 下训练并缓存一个 **条件扩散模型** 来近似每个时间段的成功分布 \(p(x\\mid seg)\)，并用去噪误差作为观测 cost（再配合 Viterbi 解码）：

- `--segment-match-score diffusion`
- `--segment-match-diffusion-pca-dim`：先在成功集标准化空间做 PCA 降维（默认 32，推荐从 32/64 起步）
- `--segment-match-diffusion-timesteps/epochs/hidden-dim/num-layers/lr/batch-size`：训练超参
- `--segment-match-diffusion-t-eval/noise-samples`：推理时用于计算 cost 的噪声步与噪声采样次数（更大更稳，但更慢）

```bash
cd /home/ztlab/project/ELM/openpi

.venv/bin/python scripts/estimate_vjepa_value_hdf5.py \
  --value-method segment_match50 \
  --segment-match-score diffusion \
  --success-mode teleop_all \
  --teleop-hdf5-dir ./datasets/GraspAnything/success_ref \
  --score-sources rollouts \
  --camera-mode both \
  --camera-fusion dist_min \
  --segment-match-num-segments 50 \
  --segment-match-window-stride 6 \
  --segment-match-radius 3 \
  --segment-match-first-fixed \
  --segment-match-value-mode soft_next \
  --segment-match-path viterbi \
  --segment-match-diffusion-pca-dim 32 \
  --segment-match-diffusion-timesteps 100 \
  --segment-match-diffusion-epochs 500 \
  --segment-match-diffusion-hidden-dim 256 \
  --segment-match-diffusion-num-layers 4 \
  --segment-match-diffusion-lr 1e-4 \
  --segment-match-diffusion-batch-size 256 \
  --segment-match-diffusion-t-eval 20 \
  --segment-match-diffusion-noise-samples 2 \
  --out-cache-dir /home/ztlab/project/ELM/openpi/datasets/GraspAnything/vjepa_value_cache/vjepa_rollouts_segmatch50_diffusion_both_dist_min_v1 \
  --embed-batch-size 8 \
  --embed-cache-dir ./datasets/GraspAnything/vjepa_value_cache/embedding_cache
```

#### 4.5 加速建议

- `--timestep-stride`：仅对 `value-method=progress_warp` 生效。比如 30fps 数据用 `6` 代表每 ~0.2s 算一次 prefix embedding，
  脚本会对稀疏时间点的距离做线性插值生成逐帧曲线。
- `--segment-match-window-stride`：仅对 `value-method=segment_match50` 生效。控制滑窗步长（默认 6）。
- `--embed-cache-dir`：公共 embedding 缓存根目录（建议设为固定路径以跨实验复用）。
  - `progress_warp`：缓存 **成功参考末帧** 与 **prefix（按 timestep-stride 采样）** 的 embedding（key: `t_high/emb_high` 等）
  - `segment_match50`：额外缓存 **50 段 embedding**（key: `seg50_*`）与 **滑窗 embedding**（key: `win50s6_*`）
  后续再次运行时会优先命中缓存，仅对缺失的 key 重新跑 V-JEPA。
- `segment_match50 + diffusion`：扩散训练只使用已缓存的 segment/window embeddings，**不会额外调用 V-JEPA**（除非你的 embed cache 缺失对应 key）。
- `--embed-batch-size`：越大吞吐越高，但会更吃显存。
  双相机时，实际一次 forward 的 batch 约为 `2 * embed_batch_size`。
- `--max-success-episodes / --max-score-episodes`：只跑前 N 条用于快速冒烟。

### 5) 输出目录结构

以 `--out-cache-dir <DIR>` 为根目录：

- `<DIR>/cluster_centers.npz`
  （`value-method=progress_warp`）每个 task 的成功聚类中心 + StandardScaler(mean/std) + L2-normalize 决策与统计（numpy 压缩格式）
  注意：中心缓存会带 `centers_cfg_id`（与 success_ref 选择、双相机/融合策略、DBSCAN 参数等绑定），配置变化会自动触发重建，避免误用旧中心。
- `<DIR>/cluster_centers_report.json`
  （`value-method=progress_warp`）成功聚类的可读报告（每个 task/key 的 noise%、kNN 距离尺度、是否 fallback_mean 等）。
- `<DIR>/warp_f.npz`
  （`value-method=progress_warp`）全局单调变换 \(f\\)（isotonic）缓存：用于把 success_ref 的价值曲线校准为接近线性时间进度，并复用到 rollouts。
- `<DIR>/segment_centers.npz`
  （`value-method=segment_match50`）分段成功中心缓存：对每个 task、每个时间段(0..49)聚类得到的成功中心 + 对应 scaler，用于滑窗局部匹配。
- `<DIR>/segment_centers_report.json`
  （`value-method=segment_match50`）成功分段聚类的可读报告（每段的 noise%、kNN 距离尺度、是否 fallback_mean 等），便于调 `--dbscan-eps`。
- `<DIR>/segmatch50_diffusion_scaler.npz`
  （`segment_match_score=diffusion`）diffusion 训练/推理使用的全局 StandardScaler（成功集 mean/std + 是否 L2-normalize 的决策与统计，按 task/key 存储）。
- `<DIR>/segmatch50_diffusion_pca.npz`
  （`segment_match_score=diffusion`）PCA 参数（在标准化空间拟合，按 task/key 存储；0 则不启用 PCA）。
- `<DIR>/segmatch50_diffusion_models.npz` + `<DIR>/segmatch50_diffusion_model__<cfg_id>__*.pt`
  （`segment_match_score=diffusion`）条件扩散模型权重（按 task/key 保存）。cfg 变化会自动重训并写新 cfg_id。
- `<DIR>/segmatch50_diffusion_train.npz`
  （`segment_match_score=diffusion`）训练 latent 数据（用于诊断脚本快速加载；best-effort，数据很大时可能写入失败但不影响主流程）。
- `<DIR>/segmatch50_diffusion_report.json`
  （`segment_match_score=diffusion`）训练摘要（样本数、每段样本计数、loss 轨迹片段、模型路径等）。
- `<DIR>/episode_values/<hdf5_basename>.npz`
  单条 episode 的逐帧数组（按 trim 后长度）：
  - `distance`：float32, shape `[T_trim]`
    - `segment_match_score=diffusion` 时，`distance` 代表 **diffusion 去噪误差**（越小越像成功分布），不再是欧氏距离。
  - `value_pred`：float32, shape `[T_trim]`
  - `trim_start/trim_end`：裁剪区间（相对于原 HDF5 帧索引）
  - 以及 task/source/is_success/camera_mode/camera_fusion 等元数据
- embedding 缓存（若不指定 `--embed-cache-dir`，默认在 `<DIR>/embedding_cache/`；否则在你指定的目录下）：
  - `vjepa_embed_<hash>/<episode_stem>__<hash>.npz`：prefix embedding 缓存
    - `t_high/emb_high`：cam_high 在若干 timestep 的 embedding（float32）
    - `t_wrist/emb_wrist`：cam_left_wrist 在若干 timestep 的 embedding（float32）
- `<DIR>/hdf5_to_vjepa_value_map.jsonl`
  每条 HDF5 的处理记录（written/reused/skipped/error）
- `<DIR>/vjepa_value_metadata.json`
  本次运行参数、warp 配置/统计、路径等
- `<DIR>/estimate_vjepa_value.log`
  终端日志的落盘版本（包含 DBSCAN fallback 警告与距离尺度提示等），便于复现实验与排查问题。

### 6) 运行：可视化单条轨迹

#### 6.1 GUI 模式（推荐）

```bash
cd /home/ztlab/project/ELM/openpi

.venv/bin/python scripts/viz_vjepa_value_curve.py \
  --hdf5-path /home/ztlab/project/ELM/openpi/datasets/GraspAnything/rollouts_hdf5/success/episode_20251225_173045_pick_up_anything_and_put_them_in_the_box.hdf5 \
  --cache-dir /home/ztlab/project/ELM/openpi/datasets/GraspAnything/vjepa_value_cache/vjepa_rollouts_segmatch50_diffusion_both_dist_min_v1 \
  --value-threshold 0.0
```

按键：
- `a/d` 或 `←/→`：前后 1 帧
- `w/s` 或 `↑/↓`：前后 10 帧
- `q` / `ESC`：退出

#### 6.2 无 GUI 导出 PNG

```bash
cd /home/ztlab/project/ELM/openpi

.venv/bin/python scripts/viz_vjepa_value_curve.py \
  --hdf5-path /home/ztlab/project/ELM/openpi/datasets/GraspAnything/rollouts_hdf5/success/episode_20251225_193302_pick_up_anything_and_put_them_in_the_box.hdf5 \
  --cache-dir /home/ztlab/project/ELM/openpi/datasets/GraspAnything/vjepa_value_cache/vjepa_rollouts_from_teleop_v1 \
  --no-gui \
  --out-png ./vjepa_curve.png
```

#### 6.3 导出 MP4

```bash
cd /home/ztlab/project/ELM/openpi

.venv/bin/python scripts/viz_vjepa_value_curve.py \
  --hdf5-path /home/ztlab/project/ELM/openpi/datasets/GraspAnything/rollouts_hdf5/success/episode_20251225_173045_pick_up_anything_and_put_them_in_the_box.hdf5 \
  --cache-dir /home/ztlab/project/ELM/openpi/datasets/GraspAnything/vjepa_value_cache/_segmatch50_smoke_softnext \
  --out-video ./vjepa_curve.mp4 \
  --fps 30
```

#### 6.4 检查 `segment_match50` 的聚类效果（推荐用于调 `--dbscan-eps`）

该脚本会读取 `<DIR>/segment_centers.npz` + `<DIR>/vjepa_value_metadata.json`，自动定位 embedding cache，
并打印每个 segment 的 `noise%` 与 kNN 距离尺度（标准化空间），用于判断 DBSCAN 是否因为 `eps` 过小导致“全是噪声 → fallback_mean”。
该检查 **不需要 GPU**（只读缓存的 embedding）。

**(a) 打印报告**

```bash
cd /home/ztlab/project/ELM/openpi

.venv/bin/python scripts/check_segment_match50_clustering.py \
  --out-cache-dir /home/ztlab/project/ELM/openpi/datasets/GraspAnything/vjepa_value_cache/vjepa_rollouts_segmatch50_both_dist_min_v1
```

**(b) 导出某个 segment 的 PCA(2D) 散点图（按 DBSCAN label 着色）**

```bash
cd /home/ztlab/project/ELM/openpi

.venv/bin/python scripts/check_segment_match50_clustering.py \
  --out-cache-dir /home/ztlab/project/ELM/openpi/datasets/GraspAnything/vjepa_value_cache/vjepa_rollouts_segmatch50_both_dist_min_v1 \
  --plot-key high \
  --plot-segment 8 \
  --out-png ./cluster_pca_high_seg08.png
```

### 7) 参数说明（完整）

#### 7.1 `estimate_vjepa_value_hdf5.py`

- `--teleop-hdf5-dir`：teleop HDF5 目录（成功演示来源）
- `--rollouts-hdf5-dir`：rollouts HDF5 目录（需包含 `success/` 与 `failure/` 子目录）
- `--out-cache-dir`：输出缓存根目录
- `--overwrite`：若设置该 flag，删除并重建 `out-cache-dir`
- `--embed-cache-dir`：公共 embedding 缓存根目录（默认 `<out-cache-dir>/embedding_cache`）

- `--success-mode`：成功参考集定义
  - `teleop_all`：teleop 全部作为成功参考
  - `rollouts_shortest_p`：rollouts_success 最短 p% 作为成功参考
- `--shortest-p`：`rollouts_shortest_p` 的比例（0~1）

- `--score-sources`：哪些数据会被“逐帧评分”并写入 `episode_values/`
  - `rollouts`：仅 rollouts（默认）
  - `teleop`：仅 teleop
  - `all`：两者都评分
  - `teleop`：仅 teleop
  - `all`：两者都评分

- `--value-method`：价值曲线方法
  - `progress_warp`（默认）：距离→进度 + 单调化/末帧缩放 + 全局单调 warp \(f\\)
  - `segment_match50`：50 段成功中心 + 滑窗局部匹配 + 插值生成曲线

- `--camera-mode`：使用哪个相机
  - `high` / `wrist` / `both`
- `--camera-fusion`：`camera-mode=both` 时的融合策略
  - `dist_min` / `dist_mean` / `emb_mean` / `emb_concat`

- `--epsilon`：静止帧裁剪阈值（基于 `--trim-signal` 的 L∞ 变化）
- `--trim-signal`：裁剪时使用哪个信号来判断是否“发生变化”
  - `qpos`（默认，推荐）：使用 `/observations/qpos`
  - `action`：使用 `/action`（legacy）
- `--max-episode-len`：裁剪后长度超过该值的 episode 直接跳过（用于控算力）

- `--vjepa-ckpt`：V-JEPA 权重路径
- `--vjepa-img-size`：embedding 用的 crop 后分辨率（默认 384）
- `--vjepa-num-frames`：每次 embedding 采样帧数（默认 64）
- `--device`：`auto` / `cuda:0` / `cpu`
- `--enable-fp16`：cuda 下是否启用 fp16 autocast

- `--embed-batch-size`：embedding 的 batch size（越大越快但越吃显存；双相机会近似翻倍）
- `--timestep-stride`：仅 `value-method=progress_warp` 使用；每隔多少帧计算一次 prefix embedding（其余用插值补全）
- `--max-success-episodes`：成功参考 episode 的上限（调试用）
- `--max-score-episodes`：评分 episode 的上限（调试用）
- `--max-score-rollouts-failure-episodes`：仅限制 rollouts_failure 参与评分的条数（调试/省算力）
- `--max-score-rollouts-success-episodes`：仅限制 rollouts_success 参与评分的条数（调试/省算力）

- `--dbscan-eps` / `--dbscan-min-samples`：成功聚类（DBSCAN）的超参（SRPO 同款默认）
- `--dbscan-pca-dim`：可选，先在“标准化后的 embedding”上做 PCA 降维到该维度，再进行 DBSCAN（默认 0 表示不降维）

- （仅 `value-method=progress_warp` 时使用）`--value-mapping`：距离→价值的映射方式
  - `linear_offset_scale`（默认，推荐）：以 \(d_0=d_{t=0}\) 为零点，`ReLU(d0-dt)` 后按末帧缩放
  - `sigmoid`（legacy）：旧 SRPO sigmoid（会用到 `--norm-scope/--reward-scale/...` 这组参数）
- （仅 `value-method=progress_warp` 时使用）`--failure-end-value`：失败轨迹（非 success）末帧缩放目标（默认 0.5）
- `--enforce-monotonic / --no-enforce-monotonic`：是否对输出价值曲线做 `cummax` 强制单调（默认关闭；关闭时允许出现“进度倒退”导致的价值下降）
- （仅 `value-method=progress_warp` 时使用）`--warp-enable` / `--no-warp-enable`：是否启用全局单调变换 \(f\\)（默认启用）
- （仅 `value-method=progress_warp` 时使用）`--warp-method`：目前仅支持 `isotonic`
- （仅 `value-method=progress_warp` 时使用）`--success-fit-step-ratio`：拟合 \(f\\) 时对 success_ref 的时间比例采样步长（默认 0.05，即每 5% 一点）

- （仅 `value-method=segment_match50` 时使用）`--segment-match-num-segments`：时间段数量（默认 50）
- （仅 `value-method=segment_match50` 时使用）`--segment-match-window-stride`：滑窗步长（默认 6）
- （仅 `value-method=segment_match50` 时使用）`--segment-match-radius`：局部匹配搜索半径（默认 5，表示 n±5）
- （仅 `value-method=segment_match50` 时使用）`--segment-match-first-fixed`：若设置该 flag，第一个片段强制匹配到第 1 段
- （仅 `value-method=segment_match50` 时使用）`--segment-match-value-mode`：段→价值的方式
  - `hard`：窗口末帧 value = \((n+1)/50\\)
  - `soft_next`（推荐）：窗口末帧在 \((n+1)/50\\) 基础上加入段内连续进度，显著减少长平台
- （仅 `value-method=segment_match50` 时使用）`--segment-match-score`：候选段打分方式（默认 `dist`）
  - `dist`：按标准化欧式距离最小匹配
  - `gaussian_nll`：对高方差段做惩罚（实验性）
- （仅 `value-method=segment_match50` 时使用）`--segment-match-path`：段号匹配路径算法（默认 `viterbi`）
  - `greedy`：逐窗贪心 + 局部搜索（更快但容易局部最优/末端回退）
  - `viterbi`（推荐）：动态规划求全局最优路径（显著减少局部陷阱/末端回退）
- （仅 `value-method=segment_match50` 时使用）`--segment-match-time-prior-success`：仅对 success 轨迹生效的时间先验系数（默认 0.4）
  会在代价里加入 \(|n - n_{expected}(t)|\) 惩罚，抑制“早跳到很后段”和“末端回到很前段”。设为 0 可关闭。
- （仅 `value-method=segment_match50` 时使用）`--segment-match-back-penalty-success`：仅对 success 轨迹生效的回退惩罚（默认 0.0）
  会对 `prev_seg > seg` 的回退按回退段数线性加罚，进一步避免成功轨迹末端回退导致异常缩放。
- （仅 `value-method=segment_match50` 时使用）`--segment-match-failure-value-cap`：失败轨迹最大价值上限（默认 0.8）
  失败轨迹不会被“拉满到 1”，而是（必要时）整体缩小，使整条曲线满足 `max(value_pred) < 0.8`。
- `--minmax-skip-prefix-k`：强制 `value_pred[:K]=0`（只影响输出，不影响 \(d_0\)）

- （仅 `value-mapping=sigmoid` 时使用）`--norm-scope`：距离归一化统计范围
  - `fail_only`：优先只用失败轨迹统计 min/max；若无失败则回退到 all_scored
  - `all_scored`：所有被评分样本共同统计 min/max
- （仅 `value-mapping=sigmoid` 时使用）`--reward-scale` / `--sigmoid-steepness` / `--sigmoid-offset`：sigmoid 超参
- `--norm-eps`：数值 eps（防止除零）
- `--set-success-terminal-reward` / `--success-terminal-reward`：legacy（默认关闭，不建议启用，会破坏“按末帧缩放”的一致性）

#### 7.2 `viz_vjepa_value_curve.py`

- `--hdf5-path`：要可视化的 HDF5 文件路径
- `--cache-dir`：与估计脚本 `--out-cache-dir` 相同
- `--scale`：预览图缩放（目前主要影响 figure 观看体验）
- `--plot-height`：曲线区域高度（像素级意义不强，保留与 `viz_advantage_curve.py` 一致的参数）
- `--value-threshold`：用于把点染色为红/绿的阈值（可选）
- `--no-gui`：不弹窗，直接导出 PNG
- `--out-png`：PNG 输出路径
- `--out-video`：MP4 输出路径
- `--fps`：视频帧率
- `--lookahead`：额外画一条“未来 lookahead 帧”的虚线（便于看进度）
- `--repeat-interval-ms`：按键长按的重复触发间隔


