#!/usr/bin/env python3
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def compute_and_save_cholesky(
    dataset_repo_id: str,
    root_dir: str,
    horizon: int = 50,  # 默认 Chunk size，应与您的训练配置一致
    beta: float = 0.5,
    save_path: str = "action_cholesky.pt", 
    max_samples: int = None
):
    """
    计算动作协方差矩阵的 Cholesky 因子。
    
    Args:
        dataset_repo_id: 数据集 ID (例如: 'pick_anything_100/piper_dual_lerobot')
        root_dir: 数据集根目录
        horizon: 动作预测的时间步长 (Chunk Size)
        beta: 收缩系数 (0~1)，越大越接近原始协方差，越小越接近单位阵
    """
    print(f"Loading dataset: {dataset_repo_id} from {root_dir}")
    
    # 1. 加载数据集
    dataset = LeRobotDataset(
        repo_id=dataset_repo_id,
        root=root_dir,
    )
    
    print(f"Total frames in dataset: {len(dataset)}")
    
    all_actions_flat = []
    
    # 定义有效的数据范围
    # 我们只在一个 Episode 内部进行切片，不跨 Episode
    # LeRobotDataset 已经处理了 Episode 边界，但我们需要手动构建 Chunk
    
    # 遍历所有 Episode
    episode_ids = range(dataset.num_episodes)
    valid_indices = []
    
    print("Collecting valid indices...")
    for ep_idx in episode_ids:
        # 获取该 episode 的帧范围 (start, end)
        from_idx = dataset.episode_data_index["from"][ep_idx].item()
        to_idx = dataset.episode_data_index["to"][ep_idx].item()
        length = to_idx - from_idx
        
        # 只要长度满足 horizon，就进行滑动窗口采样 (或者随机采样)
        # 这里为了计算准确，使用 stride=1 的滑动窗口
        if length >= horizon:
            # range(from_idx, to_idx - horizon + 1)
            # 限制采样数量以防内存爆炸
            step = 1
            if max_samples and len(dataset) > max_samples:
                step = max(1, len(dataset) // max_samples)
                
            indices = list(range(from_idx, to_idx - horizon + 1, step))
            valid_indices.extend(indices)
            
    print(f"Collected {len(valid_indices)} valid action chunks (horizon={horizon})")
    
    if len(valid_indices) == 0:
        raise ValueError(f"No valid chunks found with horizon={horizon}. Dataset episodes might be too short.")

    # 2. 提取动作 Chunk
    # 这一步可能会比较慢，具体取决于磁盘 I/O
    print("Extracting action trajectories...")
    
    # 预分配 tensor 以提高速度 ? 不，list append 比较灵活
    # 但为了不撑爆内存，可以分批处理，但协方差需要所有数据
    # 对于 100 episodes，数据量应该还可以接受
    
    # 直接访问 dataset[idx] 会返回单帧，我们需要 slice
    # LeRobotDataset 对 slice 支持可能有限，我们可以直接访问 hf_dataset 或内部逻辑
    # 更简单的方法：拿到所有 action 数据（如果不大的话），然后在内存切片
    
    # 尝试加载所有 Action 到内存 (如果内存够大)
    # 假设 'action' 是 key
    print("Loading all actions into memory for fast slicing...")
    # 注意：LeRobotDataset[idx] 返回的是 dict，比较慢
    # 直接访问底层的 parquet/arrow 表比较快，或者利用 dataset.hf_dataset
    
    # 比较通用的方式：
    full_actions = dataset.hf_dataset["action"] # shape: [Total_Frames, Action_Dim]
    
    # Handle cases where hf_dataset returns a list of Tensors (common with 'torch' format)
    if isinstance(full_actions, list) and len(full_actions) > 0 and isinstance(full_actions[0], torch.Tensor):
        print("Stacking list of tensors...")
        full_actions = torch.stack(full_actions)
    elif not isinstance(full_actions, torch.Tensor):
        full_actions = torch.tensor(full_actions)
    
    action_dim = full_actions.shape[1]
    print(f"Action Dimension: {action_dim}")
    
    chunks = []
    for start_idx in tqdm(valid_indices):
        end_idx = start_idx + horizon
        # 提取 [start, end) 的片段
        chunk = full_actions[start_idx:end_idx] # [Horizon, Dim]
        
        # 展平 [Horizon * Dim]
        chunk_flat = chunk.view(-1)
        chunks.append(chunk_flat)
        
    all_actions_flat = torch.stack(chunks) # [N_Samples, Horizon * Dim]
    print(f"Sample matrix shape: {all_actions_flat.shape}")

    # 3. 计算协方差
    print("Computing Covariance Matrix...")
    # shape: [Horizon*Dim, Horizon*Dim]
    sigma = torch.cov(all_actions_flat.T) 
    print(f"Covariance Matrix shape: {sigma.shape}")

    # 4. Beta Shrinkage
    print(f"Applying Beta Shrinkage (beta={beta})...")
    identity = torch.eye(sigma.shape[0], device=sigma.device)
    sigma_reg = beta * sigma + (1 - beta) * identity

    # 5. Cholesky 分解
    print("Computing Cholesky Decomposition...")
    try:
        L = torch.linalg.cholesky(sigma_reg)
    except RuntimeError as e:
        print(f"Warning: Cholesky failed ({e}). Adding jitter...")
        jitter = 1e-4 * identity
        L = torch.linalg.cholesky(sigma_reg + jitter)

    # 6. 保存
    print(f"Saving L to {save_path}")
    torch.save(L, save_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="pick_anything_100/piper_dual_lerobot", help="Dataset ID")
    parser.add_argument("--root", type=str, default="/home/ztlab/lgd/openpi_client/pick_anything_100/piper_dual_lerobot", help="Dataset root directory")
    parser.add_argument("--horizon", type=int, default=50, help="Action chunk size (must match training)")
    parser.add_argument("--beta", type=float, default=0.5, help="Shrinkage factor (1.0 = raw covariance)")
    parser.add_argument("--out", type=str, default="action_cholesky.pt", help="Output path")
    
    args = parser.parse_args()
    
    compute_and_save_cholesky(
        dataset_repo_id=args.repo_id,
        root_dir=args.root,
        horizon=args.horizon,
        beta=args.beta,
        save_path=args.out
    )