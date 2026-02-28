#!/usr/bin/env python3
"""
Toy Script: Verify if Diffusion Denoising Error can serve as a Value Metric.

Hypothesis:
  A Diffusion Model trained ONLY on successful trajectories will have low denoising error
  for in-distribution (success) trajectories, and high denoising error for out-of-distribution
  (failure) trajectories.

Workflow:
  1. Load rollouts embeddings from V-JEPA cache.
  2. PCA reduction (1408 -> 32 dim), fitted on success rollouts.
  3. Train a simple 1D Temporal Diffusion Model on Success Rollouts.
  4. Evaluate denoising MSE on Success vs Failure rollouts.
  5. Plot error distributions.
"""

import dataclasses
import json
import math
import random
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# --- Utilities (Reused from viz script) ---

def _resolve_path(p: str | Path | None) -> Path | None:
    if p is None:
        return None
    pp = Path(p)
    return pp if pp.is_absolute() else Path(__file__).resolve().parents[1] / pp


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz_dict(path: Path) -> dict[str, Any]:
    with np.load(str(path), allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def _resample_traj(X: np.ndarray, num_points: int) -> np.ndarray:
    if X.shape[0] < 2:
        return np.repeat(X, num_points, axis=0) if X.shape[0] > 0 else np.zeros((num_points, X.shape[1]))
    old_idx = np.arange(X.shape[0])
    new_idx = np.linspace(0, X.shape[0] - 1, num=num_points)
    X_new = np.zeros((num_points, X.shape[1]), dtype=X.dtype)
    for d in range(X.shape[1]):
        X_new[:, d] = np.interp(new_idx, old_idx, X[:, d])
    return X_new


# --- 1D Temporal U-Net (Simple MLP-Mixer style) ---

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.ff(x))


class SimpleTemporalDenoiser(nn.Module):
    def __init__(self, traj_dim=32, traj_len=50, hidden_dim=256):
        super().__init__()
        self.traj_dim = traj_dim
        self.traj_len = traj_len
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )

        # Input projection
        self.input_proj = nn.Linear(traj_dim, hidden_dim)

        # Temporal mixing (Conv1d) and Channel mixing (MLP)
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                ResidualBlock(hidden_dim),
                # Temporal mixing via simple Conv1d
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.Mish(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                )
            ]) for _ in range(4)
        ])

        self.output_proj = nn.Linear(hidden_dim, traj_dim)

    def forward(self, x, t):
        # x: (B, L, D)
        B, L, D = x.shape
        
        # Embed time
        t_emb = self.time_mlp(t)  # (B, H)
        
        # Project input
        h = self.input_proj(x)  # (B, L, H)
        
        # Add time emb
        h = h + t_emb[:, None, :]

        # Process
        # Reshape for Conv1d: (B, H, L)
        h = h.permute(0, 2, 1)
        
        for mlp_block, conv_block in self.blocks:
            # Channel mixing (applied on last dim H after permute back, or use Conv1x1)
            # Here we just permute back for MLP
            h_t = h.permute(0, 2, 1) # (B, L, H)
            h_t = mlp_block(h_t)
            h = h_t.permute(0, 2, 1) # (B, H, L)
            
            # Temporal mixing
            h = h + conv_block(h)
            
        h = h.permute(0, 2, 1) # (B, L, H)
        out = self.output_proj(h)
        return out


# --- Diffusion Scheduler ---

class LinearNoiseScheduler:
    def __init__(self, num_timesteps=100, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - self.alpha_cumprod)

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alpha_cumprod[timesteps].reshape(-1, 1, 1)
        s2 = self.sqrt_one_minus_alpha_cumprod[timesteps].reshape(-1, 1, 1)
        return s1 * x_start + s2 * x_noise

    def sample(self, model, shape, device):
        # Full loop sampling (not used for this metric, but good for debug)
        model.eval()
        with torch.no_grad():
            img = torch.randn(shape, device=device)
            for i in tqdm(reversed(range(self.num_timesteps)), desc="Sampling", total=self.num_timesteps):
                t = torch.full((shape[0],), i, dtype=torch.long, device=device)
                pred_noise = model(img, t)
                
                alpha = self.alphas[i]
                alpha_hat = self.alpha_cumprod[i]
                beta = self.betas[i]
                
                if i > 0:
                    noise = torch.randn_like(img)
                else:
                    noise = torch.zeros_like(img)
                    
                img = (1 / torch.sqrt(alpha)) * (img - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * pred_noise) + torch.sqrt(beta) * noise
        return img


# --- Main ---

@dataclasses.dataclass
class Args:
    # Path to the V-JEPA value cache dir containing episode_values/*.npz and embed_cache
    cache_dir: Path
    
    # Training settings
    pca_dim: int = 32
    traj_len: int = 50
    epochs: int = 500
    batch_size: int = 64
    lr: float = 1e-4
    seed: int = 42
    
    # Diffusion settings
    diff_steps: int = 100
    
    # Evaluation
    eval_noise_t: int = 20  # At which timestep to evaluate denoising error (small t = fine detail, large t = global structure)


def main(args: Args):
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    cache_dir = Path(args.cache_dir)
    meta_path = cache_dir / "vjepa_value_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    
    meta = _read_json(meta_path)
    embed_cache_dir = _resolve_path(meta.get("embed_cache_dir"))
    if not embed_cache_dir or not embed_cache_dir.exists():
        raise FileNotFoundError(f"Embed cache not found: {embed_cache_dir}")

    # Identify episode npz files
    ep_files = sorted((cache_dir / "episode_values").glob("*.npz"))
    
    success_rollouts = []
    failure_rollouts = []

    print(f"Scanning {len(ep_files)} episodes...")
    
    # We need to map episode npz back to embed cache
    # Since we don't want to reimplement the full matching logic, we assume standard naming or use "hdf5_path" from npz
    
    for ep_npz in tqdm(ep_files, desc="Loading Embeddings"):
        try:
            z = _load_npz_dict(ep_npz)
            is_succ = bool(z["is_success"].item())
            src = str(z.get("source", ""))
            
            # Filter for rollouts
            if not src.startswith("rollouts_"):
                continue
            
            # Reconstruct embed cache path (naive attempt, assuming stem match)
            # ep_npz name is "episode_xxx.hdf5.npz"
            # embed cache name is "episode_xxx__hash.npz" (note: .hdf5 is stripped in stem typically)
            
            stem_name = ep_npz.name.replace(".hdf5.npz", "").replace(".npz", "")
            # Try both with and without .hdf5 just in case
            candidates = list(embed_cache_dir.glob(f"{stem_name}__*.npz"))
            if not candidates:
                 # Try adding .hdf5 back if missing, or removing if present
                 candidates = list(embed_cache_dir.glob(f"{stem_name}.hdf5__*.npz"))
            
            if not candidates:
                # Debug print only if really stuck
                # print(f"Skipping {stem_name}, candidates not found in {embed_cache_dir}")
                continue
            
            # Pick the newest one
            emb_path = max(candidates, key=lambda p: p.stat().st_mtime)
            
            # Load embeddings
            ez = _load_npz_dict(emb_path)
            
            # Extract fused embedding (Both cameras, dist_min usually implies we want fused or concat)
            # For simplicity, let's look for "win50s6_emb_high" and "win50s6_emb_wrist" (assuming default args)
            # We will use whatever keys are available.
            
            keys = [k for k in ez.keys() if "_emb_" in k]
            if not keys:
                continue
            
            # Let's just grab High camera for this toy test, or concat if both exist
            # Assuming series starts with "win"
            win_keys = [k for k in keys if k.startswith("win")]
            if not win_keys:
                continue
                
            # Prefer "winXXsX_emb_high"
            high_k = next((k for k in win_keys if "high" in k), None)
            wrist_k = next((k for k in win_keys if "wrist" in k), None)
            
            parts = []
            if high_k: parts.append(ez[high_k])
            if wrist_k: parts.append(ez[wrist_k])
            
            if not parts:
                continue
                
            # Concat high+wrist
            emb = np.concatenate(parts, axis=1) # (T, D)
            
            # Normalize? Usually V-JEPA is not normalized, but for diffusion input we want standard scale.
            # We will handle standardization after collecting all data.
            
            # Resample to fixed length
            emb_res = _resample_traj(emb, args.traj_len)
            
            if is_succ:
                success_rollouts.append(emb_res)
            else:
                failure_rollouts.append(emb_res)
                
        except Exception as e:
            # print(f"Error loading {ep_npz}: {e}")
            continue

    print(f"Loaded: Success={len(success_rollouts)}, Failure={len(failure_rollouts)}")
    if len(success_rollouts) < 10:
        raise RuntimeError("Not enough success rollouts for training.")

    # 2. PCA Reduction
    from sklearn.decomposition import PCA
    
    # Stack all success for fitting
    X_succ = np.stack(success_rollouts, axis=0) # (N, L, D_orig)
    N, L, D_orig = X_succ.shape
    X_succ_flat = X_succ.reshape(-1, D_orig)
    
    print(f"Fitting PCA ({D_orig} -> {args.pca_dim})...")
    pca = PCA(n_components=args.pca_dim)
    X_succ_pca_flat = pca.fit_transform(X_succ_flat)
    
    # Transform Success
    X_succ_pca = X_succ_pca_flat.reshape(N, L, args.pca_dim)
    
    # Transform Failure
    if failure_rollouts:
        X_fail = np.stack(failure_rollouts, axis=0)
        X_fail_flat = X_fail.reshape(-1, D_orig)
        X_fail_pca = pca.transform(X_fail_flat).reshape(X_fail.shape[0], L, args.pca_dim)
    else:
        X_fail_pca = np.empty((0, L, args.pca_dim))

    # Normalize latent space to N(0, 1) roughly for stable diffusion training
    # Standard Scaler
    scaler_mean = X_succ_pca.mean()
    scaler_std = X_succ_pca.std()
    
    X_succ_norm = (X_succ_pca - scaler_mean) / scaler_std
    if len(X_fail_pca) > 0:
        X_fail_norm = (X_fail_pca - scaler_mean) / scaler_std
    
    # 3. Train Diffusion Model
    print("Training Diffusion Model...")
    
    dataset = TensorDataset(torch.from_numpy(X_succ_norm).float())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = SimpleTemporalDenoiser(traj_dim=args.pca_dim, traj_len=args.traj_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = LinearNoiseScheduler(num_timesteps=args.diff_steps, device=device)
    
    model.train()
    pbar = tqdm(range(args.epochs), desc="Training")
    for epoch in pbar:
        epoch_loss = 0
        for batch in loader:
            x0 = batch[0].to(device) # (B, L, D)
            
            # Sample t
            t = torch.randint(0, args.diff_steps, (x0.shape[0],), device=device)
            
            # Noise
            noise = torch.randn_like(x0)
            
            # Add noise
            xt = scheduler.add_noise(x0, noise, t)
            
            # Predict noise
            noise_pred = model(xt, t)
            
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if epoch % 10 == 0:
            pbar.set_postfix({"loss": epoch_loss / len(loader)})

    # 4. Evaluation
    print("Evaluating Denoising Error...")
    model.eval()
    
    def get_denoising_errors(data_norm, t_eval):
        errors = []
        # Process in batches
        bs = args.batch_size
        with torch.no_grad():
            for i in range(0, len(data_norm), bs):
                batch_np = data_norm[i:i+bs]
                x0 = torch.from_numpy(batch_np).float().to(device)
                
                # Fix t
                t = torch.full((x0.shape[0],), t_eval, dtype=torch.long, device=device)
                
                # Add noise (deterministic seed for comparison? No, average over random noise is better, 
                # but for simple metric single sample is okay-ish if N is large)
                # To be robust, we might want to run this multiple times per sample, but let's stick to 1.
                noise = torch.randn_like(x0)
                xt = scheduler.add_noise(x0, noise, t)
                
                noise_pred = model(xt, t)
                
                # Error: MSE per trajectory
                # shape: (B, L, D) -> mean over L, D -> (B,)
                loss_per_traj = F.mse_loss(noise_pred, noise, reduction='none').mean(dim=(1, 2))
                errors.extend(loss_per_traj.cpu().numpy().tolist())
        return np.array(errors)

    t_eval = args.eval_noise_t
    err_succ = get_denoising_errors(X_succ_norm, t_eval)
    err_fail = get_denoising_errors(X_fail_norm, t_eval) if len(X_fail_norm) > 0 else np.array([])
    
    print(f"Success Mean Error: {err_succ.mean():.4f} +/- {err_succ.std():.4f}")
    if len(err_fail) > 0:
        print(f"Failure Mean Error: {err_fail.mean():.4f} +/- {err_fail.std():.4f}")
        
    # 5. Plot
    plt.figure(figsize=(10, 6))
    plt.hist(err_succ, bins=30, alpha=0.6, label='Success Rollouts', density=True, color='green')
    if len(err_fail) > 0:
        plt.hist(err_fail, bins=30, alpha=0.6, label='Failure Rollouts', density=True, color='red')
    
    plt.title(f"Diffusion Denoising Error (t={t_eval}/{args.diff_steps})")
    plt.xlabel("MSE (Predicted Noise vs True Noise)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = cache_dir / "diffusion_value_test_histogram.png"
    plt.savefig(out_path)
    print(f"Saved histogram to {out_path}")
    
    # Print Separation Metric (e.g. Area Under ROC roughly implied by overlap)
    # Simple heuristic: Success < Threshold < Failure ?
    if len(err_fail) > 0:
        # Check how many failures have error > success_mean
        ratio = (err_fail > err_succ.mean()).mean()
        print(f"Ratio of Failure errors > Mean Success error: {ratio:.2%}")


if __name__ == "__main__":
    tyro.cli(main)

