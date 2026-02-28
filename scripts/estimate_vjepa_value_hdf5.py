#!/usr/bin/env python3
"""
Estimate per-frame V-JEPA value (SRPO-style) for GraspAnything HDF5 episodes.

At timestep t, we build a prefix clip o_{0:t}, embed it with a V-JEPA world model,
compute distance to the nearest "successful" cluster center, then map that distance
to a reward/value using the same sigmoid shaping as SRPO:

  normalized = (d - d_min) / (d_max - d_min)
  value = reward_scale * sigmoid(sigmoid_steepness * (sigmoid_offset - normalized))

Optionally, for successful episodes, the last frame's value can be overridden to 1.0
to match SRPO's success reward.

Outputs under --out-cache-dir:
- vjepa_value_metadata.json
- cluster_centers.npz
- hdf5_to_vjepa_value_map.jsonl
- episode_values/<hdf5_basename>.npz : per-frame arrays (distance/value_pred) for the trimmed episode.

Camera support:
- --camera-mode {high,wrist,both}
- --camera-fusion {dist_min,dist_mean,emb_mean,emb_concat}
  - For dist_*: build separate centers per camera and fuse distances.
  - For emb_*: fuse embeddings then build centers and compute distance in fused space.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any
from typing import Literal

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import tyro


def _trim_static_head_tail(action: np.ndarray, *, epsilon: float) -> tuple[int, int] | None:
    """Return (start_idx, end_idx_inclusive) after trimming static head/tail.

    Static is defined as action[t] == action[t-1] within epsilon (L_inf).
    """
    if action.ndim != 2 or action.shape[0] < 2:
        return None
    diffs = np.max(np.abs(action[1:] - action[:-1]), axis=-1)  # (T-1,)
    changed = diffs > float(epsilon)
    if not np.any(changed):
        return None
    lead_idle = int(np.argmax(changed))
    trail_idle = int(np.argmax(changed[::-1]))
    start = lead_idle + 1
    end = action.shape[0] - trail_idle - 1
    if end <= start:
        return None
    return start, end


def _get_trim_signal(ep: h5py.File, *, trim_signal: str) -> np.ndarray:
    """Return (T,D) array used for trimming static head/tail."""
    sig = str(trim_signal)
    if sig == "qpos":
        for key in ("/observations/qpos", "observations/qpos"):
            if key in ep:
                return np.asarray(ep[key][:], dtype=np.float32)
        raise KeyError("trim_signal=qpos but /observations/qpos not found")
    if sig == "action":
        for key in ("/action", "action"):
            if key in ep:
                return np.asarray(ep[key][:], dtype=np.float32)
        raise KeyError("trim_signal=action but /action not found")
    raise ValueError(f"Unknown trim_signal: {trim_signal}")


def _decode_bytes(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, np.bytes_):
        return bytes(x).decode("utf-8", errors="ignore")
    return str(x)


def _task_from_hdf5(ep: h5py.File, *, fallback_name: str) -> str:
    # Prefer /task dataset (per-frame bytes). In GraspAnything HDF5 it's usually (T,) |Sxx.
    for key in ("/task", "task"):
        if key in ep:
            ds = ep[key]
            if isinstance(ds, h5py.Dataset) and ds.shape:
                return _decode_bytes(ds[0]).strip("\0")
    # Fallback: parse from file stem.
    name = fallback_name
    if name.endswith(".hdf5"):
        name = name[: -len(".hdf5")]
    parts = name.split("_")
    # Heuristic: task string is usually after the timestamp.
    if len(parts) >= 3:
        return "_".join(parts[2:])
    return name


def _load_images_full(ep: h5py.File, cam: str, *, s: int, e: int) -> np.ndarray:
    """Load full trimmed image sequence, returned as uint8 array (T, C, H, W)."""
    # Common schema: /observations/images/<cam> (T, C, H, W) uint8
    ds_path = f"/observations/images/{cam}"
    if ds_path not in ep:
        ds_path = f"observations/images/{cam}"
    ds = ep[ds_path]
    if ds.ndim == 4:
        return np.asarray(ds[s : e + 1], dtype=np.uint8)

    # Fallback: encoded bytes per frame.
    import cv2

    frames = []
    for data in ds[s : e + 1]:
        img = cv2.imdecode(data, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        frames.append(img)
    return np.asarray(frames, dtype=np.uint8)


def _load_images_sample(ep: h5py.File, cam: str, abs_indices: np.ndarray) -> np.ndarray:
    """Load a sampled set of frames, returned as uint8 array (N, C, H, W)."""
    ds_path = f"/observations/images/{cam}"
    if ds_path not in ep:
        ds_path = f"observations/images/{cam}"
    ds = ep[ds_path]
    if ds.ndim == 4:
        idx = np.asarray(abs_indices, dtype=np.int64).reshape(-1)
        # h5py fancy indexing requires indices to be strictly increasing.
        # When we sample with repetition (short segments/windows), idx can contain duplicates
        # or non-monotonic patterns; fall back to per-index reads to preserve order.
        if idx.size <= 1 or bool(np.all(np.diff(idx) > 0)):
            return np.asarray(ds[idx], dtype=np.uint8)
        frames = [np.asarray(ds[int(i)], dtype=np.uint8) for i in idx.tolist()]
        return np.asarray(frames, dtype=np.uint8)

    import cv2

    frames = []
    for i in abs_indices.tolist():
        data = ds[i]
        img = cv2.imdecode(data, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        frames.append(img)
    return np.asarray(frames, dtype=np.uint8)


def _prefix_sample_indices(prefix_len: int, *, n: int) -> np.ndarray:
    if prefix_len <= 0:
        return np.zeros((n,), dtype=np.int64)
    # Use linspace for all cases to ensure monotonic indices even if prefix_len < n.
    return np.linspace(0, max(0, prefix_len - 1), num=int(n), dtype=np.int64)


def _sample_indices_in_range(start: int, end: int, *, n: int, min_index: int | None = None) -> np.ndarray:
    """Uniformly sample n indices in [start, end] (inclusive).

    If the range is shorter than n and min_index is provided, it will try to
    include previous frames (starting from min_index) to reach n frames.
    Otherwise, indices will repeat.
    """
    start_i = int(start)
    end_i = int(end)
    if end_i < start_i:
        end_i = start_i

    length = int(end_i - start_i + 1)
    if length < int(n) and min_index is not None:
        # Try to expand backwards to include previous frames.
        start_i = max(int(min_index), end_i - int(n) + 1)
        length = int(end_i - start_i + 1)

    if length <= 0:
        return np.full((n,), start_i, dtype=np.int64)

    # Use linspace for all cases to ensure monotonic indices even if length < n.
    return (start_i + np.linspace(0, max(0, length - 1), num=int(n), dtype=np.int64)).astype(np.int64)


def _segment_bounds(t_total: int, *, num_segments: int) -> list[tuple[int, int]]:
    """Return inclusive bounds [(s0,e0),...,(sN-1,eN-1)] partitioning [0, t_total-1]."""
    t_total = int(t_total)
    num_segments = int(num_segments)
    if t_total <= 0 or num_segments <= 0:
        return []
    if num_segments == 1:
        return [(0, t_total - 1)]
    # Use integer boundaries based on linspace to keep near-equal lengths.
    cuts = np.linspace(0, t_total, num=num_segments + 1, dtype=np.int64)
    out: list[tuple[int, int]] = []
    for i in range(num_segments):
        a = int(cuts[i])
        b = int(cuts[i + 1]) - 1
        if b < a:
            b = a
        a = max(0, min(t_total - 1, a))
        b = max(0, min(t_total - 1, b))
        out.append((a, b))
    return out


def _min_dist_to_centers(emb: np.ndarray, centers: np.ndarray) -> float:
    """Euclidean distance to nearest center."""
    emb = np.asarray(emb, dtype=np.float32).reshape(1, -1)
    centers = np.asarray(centers, dtype=np.float32)
    if centers.ndim == 1:
        centers = centers.reshape(1, -1)
    # dist^2 = ||e||^2 + ||c||^2 - 2 e·c
    e2 = float(np.sum(emb * emb))
    c2 = np.sum(centers * centers, axis=1)
    dots = (centers @ emb.reshape(-1)).astype(np.float32)
    d2 = np.maximum(c2 + e2 - 2.0 * dots, 0.0)
    return float(np.sqrt(np.min(d2)))


def _min_dist2_to_centers(emb: np.ndarray, centers: np.ndarray) -> float:
    """Squared Euclidean distance to nearest center."""
    emb = np.asarray(emb, dtype=np.float32).reshape(1, -1)
    centers = np.asarray(centers, dtype=np.float32)
    if centers.ndim == 1:
        centers = centers.reshape(1, -1)
    # dist^2 = ||e||^2 + ||c||^2 - 2 e·c
    e2 = float(np.sum(emb * emb))
    c2 = np.sum(centers * centers, axis=1)
    dots = (centers @ emb.reshape(-1)).astype(np.float32)
    d2 = np.maximum(c2 + e2 - 2.0 * dots, 0.0)
    return float(np.min(d2))


def _viterbi_path(
    cost: np.ndarray,
    *,
    radius: int,
    first_fixed: bool,
    back_penalty: float = 0.0,
) -> np.ndarray:
    """Viterbi/DP for segment index path.

    Args:
      cost: (K, N) observation costs (lower is better).
      radius: transition constraint: |n_k - n_{k-1}| <= radius.
      first_fixed: if True, enforce n_0 == 0.
      back_penalty: penalize backward transitions by back_penalty * max(0, prev - cur).

    Returns:
      path: int32 array (K,) with segment indices.
    """
    cost = np.asarray(cost, dtype=np.float32)
    if cost.ndim != 2:
        raise ValueError(f"cost must be (K,N), got {cost.shape}")
    K, N = int(cost.shape[0]), int(cost.shape[1])
    if K <= 0 or N <= 0:
        return np.zeros((0,), dtype=np.int32)

    rad = int(max(0, radius))
    INF = np.float32(1e30)
    dp = np.full((K, N), INF, dtype=np.float32)
    bp = np.full((K, N), -1, dtype=np.int32)

    if bool(first_fixed):
        dp[0, 0] = float(cost[0, 0])
    else:
        dp[0, :] = cost[0, :]

    back_pen = float(max(0.0, back_penalty))
    for k in range(1, K):
        prev_row = dp[k - 1]
        for n in range(N):
            lo = max(0, n - rad)
            hi = min(N - 1, n + rad)
            prev = prev_row[lo : hi + 1]
            m_idx = np.arange(lo, hi + 1, dtype=np.int32)
            if back_pen > 0.0:
                pen = (np.maximum(0, m_idx - int(n)).astype(np.float32) * back_pen).astype(np.float32)
                vals = prev + pen
            else:
                vals = prev
            j_rel = int(np.argmin(vals))
            best_m = int(m_idx[j_rel])
            bp[k, n] = best_m
            dp[k, n] = float(cost[k, n]) + float(vals[j_rel])

    end_n = int(np.argmin(dp[K - 1]))
    path = np.zeros((K,), dtype=np.int32)
    path[K - 1] = end_n
    for k in range(K - 1, 0, -1):
        end_n = int(bp[k, end_n])
        if end_n < 0:
            end_n = int(path[k])
        path[k - 1] = end_n
    if bool(first_fixed):
        path[0] = 0
    return path


class _SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) int64/float
        half = int(self.dim // 2)
        if half <= 0:
            return torch.zeros((t.shape[0], 0), device=t.device, dtype=torch.float32)
        t = t.to(torch.float32)
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / float(max(1, half - 1))))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if int(emb.shape[1]) < int(self.dim):
            emb = torch.cat([emb, torch.zeros((emb.shape[0], int(self.dim) - int(emb.shape[1])), device=t.device, dtype=emb.dtype)], dim=-1)
        return emb


class _ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(dim))
        self.ff = nn.Sequential(
            nn.Linear(int(dim), int(dim) * 4),
            nn.Mish(),
            nn.Linear(int(dim) * 4, int(dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.ff(x))


class _CondDiffusionDenoiser(nn.Module):
    """Simple conditional denoiser for vector embeddings x in R^D conditioned on seg_id."""

    def __init__(self, *, dim_in: int, num_segments: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.dim_in = int(dim_in)
        self.num_segments = int(num_segments)
        h = int(hidden_dim)
        self.in_proj = nn.Linear(int(dim_in), h)
        self.time_mlp = nn.Sequential(
            _SinusoidalPosEmb(h),
            nn.Linear(h, h),
            nn.Mish(),
            nn.Linear(h, h),
        )
        self.seg_emb = nn.Embedding(int(num_segments), h)
        self.blocks = nn.ModuleList([_ResidualMLPBlock(h) for _ in range(int(max(1, num_layers)))])
        self.out_proj = nn.Linear(h, int(dim_in))

    def forward(self, x: torch.Tensor, t: torch.Tensor, seg_id: torch.Tensor) -> torch.Tensor:
        # x: (B,D), t: (B,) long, seg_id: (B,) long
        h = self.in_proj(x)
        h = h + self.time_mlp(t) + self.seg_emb(seg_id)
        for blk in self.blocks:
            h = blk(h)
        return self.out_proj(h)


def _diffusion_precompute(
    *,
    timesteps: int,
    device: torch.device,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> dict[str, torch.Tensor]:
    T = int(max(1, timesteps))
    betas = torch.linspace(float(beta_start), float(beta_end), T, device=device, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_cumprod": alpha_cumprod,
        "sqrt_alpha_cumprod": torch.sqrt(alpha_cumprod),
        "sqrt_one_minus_alpha_cumprod": torch.sqrt(1.0 - alpha_cumprod),
    }


def _fit_diffusion_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
    """Fit a global StandardScaler on success embeddings, with optional L2-normalize decision."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or int(X.shape[0]) <= 0:
        raise ValueError("X must be (N,D) with N>0")
    n = int(X.shape[0])
    norms = np.linalg.norm(X, axis=1).astype(np.float32)
    norm_mean = float(np.mean(norms))
    norm_std = float(np.std(norms))
    norm_cv = float(norm_std / (norm_mean + 1e-12))
    p5, p95 = np.percentile(norms, [5, 95]).astype(np.float32)
    norm_ratio = float(float(p95) / (float(p5) + 1e-12))
    use_l2 = False
    if n >= 20:
        use_l2 = bool((norm_cv > 0.10) or (norm_ratio > 1.20))
    Xp = _l2_normalize(X) if bool(use_l2) else X
    mean = Xp.mean(axis=0, keepdims=False).astype(np.float32)
    std = Xp.std(axis=0, keepdims=False).astype(np.float32)
    std = np.maximum(std, 1e-6).astype(np.float32)
    info = {
        "use_l2_normalize": bool(use_l2),
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "norm_cv": norm_cv,
        "norm_p5": float(p5),
        "norm_p95": float(p95),
        "norm_ratio_p95_p5": norm_ratio,
        "n_success": n,
    }
    return mean, std, bool(use_l2), info


def _fit_pca_for_diffusion(X_std: np.ndarray, *, pca_dim: int) -> dict[str, Any] | None:
    """Fit PCA in standardized space and return a small, serializable dict."""
    X_std = np.asarray(X_std, dtype=np.float32)
    n, d = int(X_std.shape[0]), int(X_std.shape[1])
    k_req = int(pca_dim)
    if k_req <= 0 or n < 3 or d < 2:
        return None
    k = int(min(k_req, d, max(1, n - 1)))
    if k >= d:
        return None
    try:
        from sklearn.decomposition import PCA  # type: ignore

        pca = PCA(n_components=int(k), svd_solver="randomized", random_state=0)
        pca.fit(X_std)
        evr = np.asarray(getattr(pca, "explained_variance_ratio_", []), dtype=np.float32).reshape(-1)
        return {
            "method": "sklearn_pca",
            "dim_req": int(k_req),
            "dim_used": int(k),
            "mean": np.asarray(getattr(pca, "mean_", np.zeros((d,), dtype=np.float32)), dtype=np.float32).reshape(-1),
            "components": np.asarray(getattr(pca, "components_", np.zeros((k, d), dtype=np.float32)), dtype=np.float32),
            "evr_sum": float(np.sum(evr)) if evr.size else None,
        }
    except Exception:
        Xc = (X_std - X_std.mean(axis=0, keepdims=True)).astype(np.float32)
        _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
        W = vt[:k].astype(np.float32)  # (k,d)
        return {
            "method": "svd_pca",
            "dim_req": int(k_req),
            "dim_used": int(k),
            "mean": X_std.mean(axis=0).astype(np.float32).reshape(-1),
            "components": W.astype(np.float32),
            "evr_sum": None,
        }


def _pca_transform_for_diffusion(pca: dict[str, Any] | None, X_std: np.ndarray) -> np.ndarray:
    X_std = np.asarray(X_std, dtype=np.float32)
    if pca is None:
        return X_std
    mean = np.asarray(pca.get("mean", 0.0), dtype=np.float32).reshape(-1)
    comps = np.asarray(pca.get("components", 0.0), dtype=np.float32)
    if comps.ndim != 2:
        raise ValueError("Invalid PCA components")
    return ((X_std - mean) @ comps.T).astype(np.float32)


def _train_cond_diffusion_model(
    X_latent: np.ndarray,
    seg_id: np.ndarray,
    *,
    num_segments: int,
    timesteps: int,
    hidden_dim: int,
    num_layers: int,
    lr: float,
    batch_size: int,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[_CondDiffusionDenoiser, dict[str, Any]]:
    X_latent = np.asarray(X_latent, dtype=np.float32)
    seg_id = np.asarray(seg_id, dtype=np.int64).reshape(-1)
    if X_latent.ndim != 2 or X_latent.shape[0] != seg_id.shape[0]:
        raise ValueError("X_latent must be (N,D) and seg_id must be (N,)")
    n, d = int(X_latent.shape[0]), int(X_latent.shape[1])
    if n <= 0:
        raise ValueError("Empty training data")

    torch.manual_seed(int(seed))
    np.random.seed(int(seed) & 0xFFFF_FFFF)

    model = _CondDiffusionDenoiser(dim_in=int(d), num_segments=int(num_segments), hidden_dim=int(hidden_dim), num_layers=int(num_layers)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))
    sched = _diffusion_precompute(timesteps=int(timesteps), device=device)

    from torch.utils.data import DataLoader, TensorDataset  # local import

    ds = TensorDataset(torch.from_numpy(X_latent).to(torch.float32), torch.from_numpy(seg_id).to(torch.int64))
    dl = DataLoader(ds, batch_size=int(max(1, batch_size)), shuffle=True, drop_last=False)

    losses: list[float] = []
    model.train()
    for _ep in range(int(max(1, epochs))):
        ep_loss = 0.0
        ep_n = 0
        for xb, sb in dl:
            xb = xb.to(device, non_blocking=True)
            sb = sb.to(device, non_blocking=True)
            B = int(xb.shape[0])
            t = torch.randint(0, int(sched["betas"].shape[0]), (B,), device=device, dtype=torch.int64)
            noise = torch.randn_like(xb)
            s1 = sched["sqrt_alpha_cumprod"][t].reshape(-1, 1)
            s2 = sched["sqrt_one_minus_alpha_cumprod"][t].reshape(-1, 1)
            xt = s1 * xb + s2 * noise
            pred = model(xt, t, sb)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * float(B)
            ep_n += B
        losses.append(float(ep_loss / max(1, ep_n)))
    info = {"n_train": int(n), "dim": int(d), "loss_last": float(losses[-1]) if losses else None, "losses": losses[-10:]}
    return model, info


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _srpo_map_distance(
    d: np.ndarray,
    *,
    d_min: float,
    d_max: float,
    reward_scale: float,
    sigmoid_steepness: float,
    sigmoid_offset: float,
    eps: float = 1e-6,
) -> np.ndarray:
    d = np.asarray(d, dtype=np.float32)
    rng = float(d_max - d_min)
    if rng < float(eps):
        normalized = np.full_like(d, 0.5, dtype=np.float32)
    else:
        normalized = (d - float(d_min)) / rng
        normalized = np.clip(normalized, 0.0, 1.0)
    inputs = float(sigmoid_steepness) * (float(sigmoid_offset) - normalized)
    return float(reward_scale) * _sigmoid(inputs)


def _distance_to_value_linear_offset_scale(
    d: np.ndarray, *, eps: float = 1e-6
) -> np.ndarray:
    """Linear distance->value mapping without sigmoid.

    We treat the first timestep (t=0) as baseline and map progress as:
      u[t] = max(0, d0 - d[t])
      v[t] = u[t] / u[-1]   (so v[0]=0 and v[-1]=1 when u[-1]>0)

    This matches the user's requirement:
    - offset so initial value is 0
    - clamp negatives to 0
    - scale so final value is 1
    """
    d = np.asarray(d, dtype=np.float32).reshape(-1)
    if d.size == 0:
        return np.asarray([], dtype=np.float32)
    d0 = float(d[0])
    u = np.maximum(0.0, float(d0) - d).astype(np.float32)
    denom = float(u[-1])
    if denom <= float(eps):
        return np.zeros_like(u, dtype=np.float32)
    return (u / denom).astype(np.float32)


def _cummax_and_scale(v: np.ndarray, *, target_end: float, eps: float = 1e-6) -> np.ndarray:
    """Make v non-decreasing via cummax, then scale so last element == target_end."""
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    if v.size == 0:
        return v.astype(np.float32)
    v2 = np.maximum.accumulate(v).astype(np.float32)
    end = float(v2[-1])
    if end <= float(eps):
        return np.zeros_like(v2, dtype=np.float32)
    return (v2 * (float(target_end) / end)).astype(np.float32)


def _scale_to_end(v: np.ndarray, *, target_end: float, eps: float = 1e-6, clip: bool = True) -> np.ndarray:
    """Scale v so last element == target_end, without enforcing monotonicity."""
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    if v.size == 0:
        return v.astype(np.float32)
    end = float(v[-1])
    if abs(end) <= float(eps):
        return np.zeros_like(v, dtype=np.float32)
    v2 = (v * (float(target_end) / end)).astype(np.float32)
    if clip:
        v2 = np.clip(v2, 0.0, float(target_end)).astype(np.float32)
    return v2


def _sample_timesteps_by_ratio(t_total: int, *, step_ratio: float) -> list[int]:
    """Sample prefix endpoints by time fraction.

    Always includes t=0 and t=t_total-1.
    """
    t_total = int(t_total)
    if t_total <= 0:
        return []
    if t_total == 1:
        return [0]
    r = float(step_ratio)
    if not (0.0 < r <= 1.0):
        raise ValueError(f"success_fit_step_ratio must be in (0,1], got {step_ratio}")
    ts: set[int] = set()
    ts.add(0)
    ts.add(t_total - 1)
    n_steps = int(np.floor(1.0 / r))
    for i in range(1, n_steps):
        frac = float(i) * r
        idx = int(round(frac * float(t_total - 1)))
        ts.add(max(0, min(t_total - 1, idx)))
    # Also ensure endpoint for exact 1.0
    ts.add(t_total - 1)
    return sorted(ts)


def _fit_isotonic_warp(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit a non-decreasing mapping f so that f(x) ~= y.

    Returns (x_breaks, y_breaks, info) where breaks can be applied via np.interp.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if x.size != y.size:
        raise ValueError(f"x and y size mismatch: {x.size} vs {y.size}")
    if x.size < 2:
        xb = np.asarray([0.0, 1.0], dtype=np.float32)
        yb = np.asarray([0.0, 1.0], dtype=np.float32)
        return xb, yb, {"method": "fallback_identity_small", "n": int(x.size)}

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        xb = np.asarray([0.0, 1.0], dtype=np.float32)
        yb = np.asarray([0.0, 1.0], dtype=np.float32)
        return xb, yb, {"method": "fallback_identity_nan", "n": int(x.size)}

    # Clip targets to [0,1] for stability.
    y = np.clip(y, 0.0, 1.0).astype(np.float32)

    # Prefer sklearn if available, but keep a lightweight fallback.
    try:
        from sklearn.isotonic import IsotonicRegression  # type: ignore

        ir = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        ir.fit(x, y)
        xb = np.asarray(ir.X_thresholds_, dtype=np.float32).reshape(-1)
        yb = np.asarray(ir.y_thresholds_, dtype=np.float32).reshape(-1)
        if xb.size < 2:
            xb = np.asarray([0.0, 1.0], dtype=np.float32)
            yb = np.asarray([0.0, 1.0], dtype=np.float32)
        return xb, yb, {"method": "sklearn_isotonic", "n": int(x.size), "n_breaks": int(xb.size)}
    except Exception as ex:  # noqa: BLE001
        # Fallback: simple pooled-adjacent-violators (PAV) on sorted x.
        order = np.argsort(x, kind="mergesort")
        xs = x[order]
        ys = y[order]
        # Merge duplicates in x first (average y).
        uniq_x: list[float] = []
        uniq_y: list[float] = []
        uniq_w: list[float] = []
        i = 0
        while i < xs.size:
            j = i + 1
            while j < xs.size and float(xs[j]) == float(xs[i]):
                j += 1
            w = float(j - i)
            uniq_x.append(float(xs[i]))
            uniq_y.append(float(np.mean(ys[i:j])))
            uniq_w.append(w)
            i = j

        # PAV on blocks.
        by: list[float] = []
        bw: list[float] = []
        bx: list[float] = []
        for x_i, y_i, w_i in zip(uniq_x, uniq_y, uniq_w, strict=True):
            bx.append(float(x_i))
            by.append(float(y_i))
            bw.append(float(w_i))
            while len(by) >= 2 and by[-2] > by[-1]:
                y_new = (by[-2] * bw[-2] + by[-1] * bw[-1]) / (bw[-2] + bw[-1])
                w_new = bw[-2] + bw[-1]
                by[-2] = float(y_new)
                bw[-2] = float(w_new)
                by.pop()
                bw.pop()
                bx.pop()

        xb = np.asarray(bx, dtype=np.float32)
        yb = np.asarray(by, dtype=np.float32)
        if xb.size < 2:
            xb = np.asarray([0.0, 1.0], dtype=np.float32)
            yb = np.asarray([0.0, 1.0], dtype=np.float32)
        return xb, yb, {"method": "pav_fallback", "n": int(x.size), "n_breaks": int(xb.size), "error": str(ex)}


def _apply_warp(v: np.ndarray, *, x_breaks: np.ndarray, y_breaks: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    xb = np.asarray(x_breaks, dtype=np.float32).reshape(-1)
    yb = np.asarray(y_breaks, dtype=np.float32).reshape(-1)
    if xb.size < 2 or yb.size < 2:
        return v.astype(np.float32)
    # Ensure sorted by x.
    order = np.argsort(xb)
    xb = xb[order]
    yb = yb[order]
    return np.interp(v, xb, yb, left=float(yb[0]), right=float(yb[-1])).astype(np.float32)


def _cluster_centers_per_segment(
    embs_by_seg: list[list[np.ndarray]],
    *,
    dbscan_eps: float,
    dbscan_min_samples: int,
    dbscan_pca_dim: int = 0,
) -> tuple[list[np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    """Cluster embeddings for each segment independently.

    Args:
      embs_by_seg: length S list; each element is list of embeddings (D,) for that segment.
    Returns:
      centers_list: length S, each is (K_i, D)
      scaler_list: length S, each is {"mean":(D,), "std":(D,), "l2_normalize":bool}
      info_list:   length S, clustering info per segment
    """
    centers_list: list[np.ndarray] = []
    scaler_list: list[dict[str, Any]] = []
    info_list: list[dict[str, Any]] = []
    for seg_idx, seg_embs in enumerate(embs_by_seg):
        if not seg_embs:
            # empty segment: placeholder
            centers_list.append(np.zeros((1, 1), dtype=np.float32))
            scaler_list.append({"mean": np.zeros((1,), dtype=np.float32), "std": np.ones((1,), dtype=np.float32), "l2_normalize": False})
            info_list.append({"method": "empty", "n_centers": 0, "n_success": 0, "seg_idx": int(seg_idx)})
            continue
        arr = np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in seg_embs], axis=0).astype(np.float32)
        centers, scaler, info = _try_cluster_centers(
            arr,
            dbscan_eps=float(dbscan_eps),
            dbscan_min_samples=int(dbscan_min_samples),
            dbscan_pca_dim=int(dbscan_pca_dim),
        )
        centers_list.append(np.asarray(centers, dtype=np.float32))
        scaler_list.append(scaler)
        info_list.append({**info, "seg_idx": int(seg_idx)})
    return centers_list, scaler_list, info_list


def _l2_normalize(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        n = float(np.linalg.norm(x))
        return x / max(n, float(eps))
    if x.ndim == 2:
        n = np.linalg.norm(x, axis=1, keepdims=True).astype(np.float32)
        return x / np.maximum(n, float(eps))
    raise ValueError(f"Expected 1D or 2D array for l2_normalize, got shape={x.shape}")


def _try_cluster_centers(
    succ_embeddings: np.ndarray,
    *,
    dbscan_eps: float,
    dbscan_min_samples: int,
    dbscan_pca_dim: int = 0,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Return (centers, scaler, info).

    - centers: (K, D) in the *preprocessed embedding space* (raw or L2-normalized).
    - scaler: {"mean": (D,), "std": (D,), "l2_normalize": bool}
      used to compute distances in standardized space: (x - mean) / std
    - info: debug info (method, norm stats, labels...).
    """
    succ_embeddings = np.asarray(succ_embeddings, dtype=np.float32)
    if succ_embeddings.ndim != 2 or succ_embeddings.shape[0] <= 0:
        raise ValueError("succ_embeddings must be (N, D) with N>0")
    n_succ = int(succ_embeddings.shape[0])

    # Heuristic: decide whether we should L2-normalize embeddings before StandardScaler+DBSCAN.
    # Rationale: if embedding norms vary a lot, raw L2 distances may be dominated by magnitude
    # rather than semantic direction. If norms are stable, keep raw embeddings.
    norms = np.linalg.norm(succ_embeddings, axis=1).astype(np.float32)
    norm_mean = float(np.mean(norms))
    norm_std = float(np.std(norms))
    norm_cv = float(norm_std / (norm_mean + 1e-12))
    p5, p95 = np.percentile(norms, [5, 95]).astype(np.float32)
    norm_p5 = float(p5)
    norm_p95 = float(p95)
    norm_ratio = float(norm_p95 / (norm_p5 + 1e-12))

    use_l2_normalize = False
    if n_succ >= 20:
        use_l2_normalize = bool((norm_cv > 0.10) or (norm_ratio > 1.20))

    succ_proc = _l2_normalize(succ_embeddings) if use_l2_normalize else succ_embeddings

    if n_succ == 1:
        mean = succ_proc.reshape(1, -1).astype(np.float32)
        std = np.ones_like(mean, dtype=np.float32)
        centers = succ_proc.copy()
        scaler = {"mean": mean.reshape(-1), "std": std.reshape(-1), "l2_normalize": bool(use_l2_normalize)}
        info = {
            "method": "single",
            "n_centers": 1,
            "n_success": 1,
            "use_l2_normalize": bool(use_l2_normalize),
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "norm_cv": norm_cv,
            "norm_p5": norm_p5,
            "norm_p95": norm_p95,
            "norm_ratio_p95_p5": norm_ratio,
        }
        return centers.astype(np.float32), scaler, info

    # SRPO uses StandardScaler + DBSCAN on successful embeddings. To avoid introducing a hard
    # dependency on scikit-learn, we implement a small DBSCAN (O(N^2)) here. N is typically
    # a few hundred at most, so this is fine.
    eps = float(dbscan_eps)
    min_samples = int(dbscan_min_samples)

    # Standardize
    mean = succ_proc.mean(axis=0, keepdims=True).astype(np.float32)
    std = succ_proc.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6)
    scaled = (succ_proc - mean) / std

    n = int(scaled.shape[0])
    d_full = int(scaled.shape[1])

    # Optional PCA reduction before DBSCAN (in standardized space).
    pca_dim_req = int(dbscan_pca_dim)
    pca_dim_used = 0
    pca_info: dict[str, Any] | None = None
    feats = scaled
    if pca_dim_req > 0 and n >= 3 and d_full >= 2:
        k = int(min(pca_dim_req, d_full, max(1, n - 1)))
        if k < d_full:
            try:
                # Prefer sklearn if available (faster/robust); keep a lightweight fallback.
                from sklearn.decomposition import PCA  # type: ignore

                pca = PCA(n_components=int(k), svd_solver="randomized", random_state=0)
                feats = pca.fit_transform(scaled).astype(np.float32)
                pca_dim_used = int(feats.shape[1])
                evr = np.asarray(getattr(pca, "explained_variance_ratio_", []), dtype=np.float32).reshape(-1)
                pca_info = {
                    "method": "sklearn_pca",
                    "dim_req": int(pca_dim_req),
                    "dim_used": int(pca_dim_used),
                    "evr_sum": (float(np.sum(evr)) if evr.size else None),
                }
            except Exception:
                # Fallback: SVD PCA on centered standardized features.
                Xc = (scaled - scaled.mean(axis=0, keepdims=True)).astype(np.float32)
                _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
                W = vt[: int(k)].T.astype(np.float32)
                feats = (Xc @ W).astype(np.float32)
                pca_dim_used = int(feats.shape[1])
                pca_info = {"method": "svd_pca", "dim_req": int(pca_dim_req), "dim_used": int(pca_dim_used)}

    eps2 = eps * eps
    # Pairwise dist^2 matrix (N,N)
    # dist^2 = ||x||^2 + ||y||^2 - 2x·y
    x2 = np.sum(feats * feats, axis=1, keepdims=True)  # (N,1)
    d2 = x2 + x2.T - 2.0 * (feats @ feats.T)
    d2 = np.maximum(d2, 0.0)

    # Useful debug signal: kNN distance scale in the DBSCAN feature space
    # (standardized full-D, or PCA-reduced standardized).
    # With min_samples=2, DBSCAN needs at least one neighbor within eps (besides self).
    nn_min = nn_med = nn_p90 = None
    knn_k = None
    knn_min = knn_med = knn_p90 = None
    try:
        if n >= 2:
            d2_no_self = d2.copy()
            np.fill_diagonal(d2_no_self, np.inf)
            d = np.sqrt(d2_no_self).astype(np.float32)
            nn = np.min(d, axis=1).astype(np.float32)
            nn_min = float(np.min(nn))
            nn_med = float(np.median(nn))
            nn_p90 = float(np.percentile(nn, 90))

            k_need = int(max(1, int(min_samples) - 1))
            # k-th NN excluding self => index (k-1) in sorted distances.
            idx = int(min(max(0, k_need - 1), n - 2))
            kth = np.partition(d, idx, axis=1)[:, idx].astype(np.float32)
            knn_k = int(k_need)
            knn_min = float(np.min(kth))
            knn_med = float(np.median(kth))
            knn_p90 = float(np.percentile(kth, 90))
    except Exception:
        nn_min = nn_med = nn_p90 = None
        knn_k = None
        knn_min = knn_med = knn_p90 = None
    neighbors = [np.where(d2[i] <= eps2)[0] for i in range(n)]

    visited = np.zeros((n,), dtype=bool)
    labels = np.full((n,), -1, dtype=np.int32)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nbrs = neighbors[i]
        if int(nbrs.shape[0]) < min_samples:
            # noise (keep -1)
            continue
        # Start new cluster
        labels[i] = cluster_id
        seeds = [int(x) for x in nbrs.tolist() if int(x) != i]
        while seeds:
            j = seeds.pop()
            if not visited[j]:
                visited[j] = True
                nbrs_j = neighbors[j]
                if int(nbrs_j.shape[0]) >= min_samples:
                    # Add neighbors (duplicates ok; filtered by visited/labels checks)
                    seeds.extend(int(x) for x in nbrs_j.tolist())
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1

    # Compute centers in original space (equivalent to inverse_transform(mean(scaled_pts))).
    centers: list[np.ndarray] = []
    unique_labels = sorted({int(x) for x in labels.tolist() if int(x) != -1})
    for lab in unique_labels:
        pts = succ_proc[labels == lab]
        if pts.size == 0:
            continue
        centers.append(pts.mean(axis=0).astype(np.float32))

    if not centers:
        center = succ_proc.mean(axis=0, keepdims=True)
        centers_arr = center.astype(np.float32)
        method = "dbscan_fallback_mean"
        n_centers = 1
        # Print a warning to make failures visible in terminal logs.
        # try:
        #     if nn_med is not None:
        #         logging.warning(
        #             f"DBSCAN produced 0 clusters (all noise). Falling back to mean center. "
        #             f"n_success={n_succ} eps={eps:.4g} min_samples={min_samples} "
        #             f"nn_med={nn_med:.2f} (eps/nn_med={eps/(nn_med+1e-9):.4f}). "
        #             f"Consider increasing --dbscan-eps for this embedding space."
        #         )
        #     else:
        #         logging.warning(
        #             f"DBSCAN produced 0 clusters (all noise). Falling back to mean center. "
        #             f"n_success={n_succ} eps={eps:.4g} min_samples={min_samples}. "
        #             f"Consider increasing --dbscan-eps for this embedding space."
        #         )
        # except Exception:
        #     pass
    else:
        centers_arr = np.stack(centers, axis=0).astype(np.float32)
        method = "dbscan_numpy"
        n_centers = int(centers_arr.shape[0])

    scaler = {"mean": mean.reshape(-1), "std": std.reshape(-1), "l2_normalize": bool(use_l2_normalize)}
    info = {
        "method": method,
        "n_centers": int(n_centers),
        "n_success": int(n_succ),
        "use_l2_normalize": bool(use_l2_normalize),
        "dbscan_pca_dim_req": int(pca_dim_req),
        "dbscan_pca_dim_used": int(pca_dim_used),
        "pca": pca_info,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "norm_cv": norm_cv,
        "norm_p5": norm_p5,
        "norm_p95": norm_p95,
        "norm_ratio_p95_p5": norm_ratio,
        "nn_min": nn_min,
        "nn_med": nn_med,
        "nn_p90": nn_p90,
        "knn_k": knn_k,
        "knn_min": knn_min,
        "knn_med": knn_med,
        "knn_p90": knn_p90,
        "labels": labels.tolist(),
    }
    return centers_arr, scaler, info


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _summarize_dbscan_info(info: dict[str, Any]) -> dict[str, Any]:
    """Return a compact summary for logs/reports (drops potentially huge 'labels')."""
    out: dict[str, Any] = {}
    for k in (
        "method",
        "n_centers",
        "n_success",
        "use_l2_normalize",
        "dbscan_pca_dim_req",
        "dbscan_pca_dim_used",
        "norm_mean",
        "norm_std",
        "norm_cv",
        "norm_p5",
        "norm_p95",
        "norm_ratio_p95_p5",
        "nn_min",
        "nn_med",
        "nn_p90",
        "knn_k",
        "knn_min",
        "knn_med",
        "knn_p90",
    ):
        if k in info:
            out[k] = info[k]
    labels = info.get("labels", None)
    if isinstance(labels, (list, tuple)) and labels:
        try:
            lab = np.asarray(labels, dtype=np.int32).reshape(-1)
            n = int(lab.size)
            n_noise = int(np.sum(lab == -1))
            out["n_points"] = n
            out["n_noise"] = n_noise
            out["noise_frac"] = float(n_noise / max(n, 1))
            out["n_clusters"] = int(len({int(x) for x in lab.tolist() if int(x) != -1}))
        except Exception:
            out["noise_frac"] = None
    else:
        out["noise_frac"] = None
    return out


def _sha1_short(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def _load_npz_dict(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        z = np.load(str(path), allow_pickle=True)
        out = {k: z[k] for k in z.files}
        z.close()
        return out
    except Exception:
        return None


def _episode_embed_cache_path(embed_cache_dir: Path, hdf5_path: Path) -> Path:
    st = hdf5_path.stat()
    ep_id = _sha1_short(f"{hdf5_path.resolve()}|{int(st.st_size)}|{int(st.st_mtime_ns)}")
    return embed_cache_dir / f"{hdf5_path.stem}__{ep_id}.npz"


def _hdf5_sig_id(hdf5_path: Path) -> str:
    """Short stable-ish id for an HDF5 file based on path + size + mtime.

    Used to build cache signatures (cluster centers / manifests).
    """
    st = hdf5_path.stat()
    return _sha1_short(f"{hdf5_path.resolve()}|{int(st.st_size)}|{int(st.st_mtime_ns)}")


def _load_episode_embed_maps(
    cache_path: Path,
    *,
    expected_trim_start: int,
    expected_trim_end: int,
    expected_t_total: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Load per-timestep embedding maps from cache.

    Returns:
      (high_map, wrist_map), where maps are {timestep -> embedding(D,)} in float32.
      If cache missing or metadata mismatched, returns empty maps.
    """
    z = _load_npz_dict(cache_path)
    if z is None:
        return {}, {}
    try:
        if int(np.asarray(z.get("trim_start")).item()) != int(expected_trim_start):
            return {}, {}
        if int(np.asarray(z.get("trim_end")).item()) != int(expected_trim_end):
            return {}, {}
        if int(np.asarray(z.get("t_total")).item()) != int(expected_t_total):
            return {}, {}
    except Exception:
        return {}, {}

    high: dict[int, np.ndarray] = {}
    wrist: dict[int, np.ndarray] = {}
    if "t_high" in z and "emb_high" in z:
        t_arr = np.asarray(z["t_high"], dtype=np.int32).reshape(-1)
        emb_arr = np.asarray(z["emb_high"], dtype=np.float32)
        for t, e in zip(t_arr.tolist(), emb_arr, strict=False):
            high[int(t)] = np.asarray(e, dtype=np.float32).reshape(-1)
    if "t_wrist" in z and "emb_wrist" in z:
        t_arr = np.asarray(z["t_wrist"], dtype=np.int32).reshape(-1)
        emb_arr = np.asarray(z["emb_wrist"], dtype=np.float32)
        for t, e in zip(t_arr.tolist(), emb_arr, strict=False):
            wrist[int(t)] = np.asarray(e, dtype=np.float32).reshape(-1)
    return high, wrist


def _load_episode_series_maps(
    cache_path: Path,
    *,
    series: str,
    expected_trim_start: int,
    expected_trim_end: int,
    expected_t_total: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Load camera-specific embedding maps for a named series from cache.

    Series key convention:
      - t_high key:  f\"{series}_t_high\"
      - emb_high:    f\"{series}_emb_high\"
      - t_wrist:     f\"{series}_t_wrist\"
      - emb_wrist:   f\"{series}_emb_wrist\"

    Returns (high_map, wrist_map) with timestep->embedding.
    """
    z = _load_npz_dict(cache_path)
    if z is None:
        return {}, {}
    try:
        if int(np.asarray(z.get("trim_start")).item()) != int(expected_trim_start):
            return {}, {}
        if int(np.asarray(z.get("trim_end")).item()) != int(expected_trim_end):
            return {}, {}
        if int(np.asarray(z.get("t_total")).item()) != int(expected_t_total):
            return {}, {}
    except Exception:
        return {}, {}

    high: dict[int, np.ndarray] = {}
    wrist: dict[int, np.ndarray] = {}
    k_th = f"{series}_t_high"
    k_eh = f"{series}_emb_high"
    k_tw = f"{series}_t_wrist"
    k_ew = f"{series}_emb_wrist"
    if k_th in z and k_eh in z:
        t_arr = np.asarray(z[k_th], dtype=np.int32).reshape(-1)
        emb_arr = np.asarray(z[k_eh], dtype=np.float32)
        for t, e in zip(t_arr.tolist(), emb_arr, strict=False):
            high[int(t)] = np.asarray(e, dtype=np.float32).reshape(-1)
    if k_tw in z and k_ew in z:
        t_arr = np.asarray(z[k_tw], dtype=np.int32).reshape(-1)
        emb_arr = np.asarray(z[k_ew], dtype=np.float32)
        for t, e in zip(t_arr.tolist(), emb_arr, strict=False):
            wrist[int(t)] = np.asarray(e, dtype=np.float32).reshape(-1)
    return high, wrist


def _merge_t_emb(
    old_t: np.ndarray | None,
    old_emb: np.ndarray | None,
    new_map: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    merged: dict[int, np.ndarray] = {}
    if old_t is not None and old_emb is not None:
        old_t = np.asarray(old_t, dtype=np.int32).reshape(-1)
        old_emb = np.asarray(old_emb, dtype=np.float32)
        for t, e in zip(old_t.tolist(), old_emb, strict=False):
            merged[int(t)] = np.asarray(e, dtype=np.float32).reshape(-1)
    for t, e in new_map.items():
        merged[int(t)] = np.asarray(e, dtype=np.float32).reshape(-1)
    t_all = np.asarray(sorted(merged.keys()), dtype=np.int32)
    emb_all = np.stack([merged[int(t)] for t in t_all.tolist()], axis=0).astype(np.float32)
    return t_all, emb_all


def _save_episode_embed_cache(
    cache_path: Path,
    *,
    hdf5_path: Path,
    trim_start: int,
    trim_end: int,
    t_total: int,
    vjepa_ckpt: Path,
    vjepa_img_size: int,
    vjepa_num_frames: int,
    new_high: dict[int, np.ndarray] | None,
    new_wrist: dict[int, np.ndarray] | None,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    old = _load_npz_dict(cache_path)
    if old is not None:
        # If trim metadata mismatched, discard old cache (likely different epsilon or corrupted file).
        try:
            if int(np.asarray(old.get("trim_start")).item()) != int(trim_start):
                old = None
            elif int(np.asarray(old.get("trim_end")).item()) != int(trim_end):
                old = None
            elif int(np.asarray(old.get("t_total")).item()) != int(t_total):
                old = None
        except Exception:
            old = None

    save: dict[str, Any] = {} if old is None else {k: old[k] for k in old.keys()}

    # Merge camera-specific timelines.
    if new_high:
        t_old = save.get("t_high")
        emb_old = save.get("emb_high")
        t_all, emb_all = _merge_t_emb(t_old, emb_old, new_high)
        save["t_high"] = t_all
        save["emb_high"] = emb_all
    if new_wrist:
        t_old = save.get("t_wrist")
        emb_old = save.get("emb_wrist")
        t_all, emb_all = _merge_t_emb(t_old, emb_old, new_wrist)
        save["t_wrist"] = t_all
        save["emb_wrist"] = emb_all

    # Write/refresh metadata (overrides old).
    save["schema_version"] = np.asarray(1, dtype=np.int32)
    save["hdf5_path"] = np.asarray(str(hdf5_path))
    save["trim_start"] = np.asarray(int(trim_start), dtype=np.int32)
    save["trim_end"] = np.asarray(int(trim_end), dtype=np.int32)
    save["t_total"] = np.asarray(int(t_total), dtype=np.int32)
    save["vjepa_ckpt"] = np.asarray(str(vjepa_ckpt))
    save["vjepa_img_size"] = np.asarray(int(vjepa_img_size), dtype=np.int32)
    save["vjepa_num_frames"] = np.asarray(int(vjepa_num_frames), dtype=np.int32)

    np.savez_compressed(str(cache_path), **save)


def _save_episode_series_cache(
    cache_path: Path,
    *,
    series: str,
    hdf5_path: Path,
    trim_start: int,
    trim_end: int,
    t_total: int,
    vjepa_ckpt: Path,
    vjepa_img_size: int,
    vjepa_num_frames: int,
    new_high: dict[int, np.ndarray] | None,
    new_wrist: dict[int, np.ndarray] | None,
) -> None:
    """Merge a named series of embeddings into the shared episode cache file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    old = _load_npz_dict(cache_path)
    if old is not None:
        try:
            if int(np.asarray(old.get("trim_start")).item()) != int(trim_start):
                old = None
            elif int(np.asarray(old.get("trim_end")).item()) != int(trim_end):
                old = None
            elif int(np.asarray(old.get("t_total")).item()) != int(t_total):
                old = None
        except Exception:
            old = None

    save: dict[str, Any] = {} if old is None else {k: old[k] for k in old.keys()}

    # Merge camera-specific timelines for this series.
    if new_high:
        t_old = save.get(f"{series}_t_high")
        emb_old = save.get(f"{series}_emb_high")
        t_all, emb_all = _merge_t_emb(t_old, emb_old, new_high)
        save[f"{series}_t_high"] = t_all
        save[f"{series}_emb_high"] = emb_all
    if new_wrist:
        t_old = save.get(f"{series}_t_wrist")
        emb_old = save.get(f"{series}_emb_wrist")
        t_all, emb_all = _merge_t_emb(t_old, emb_old, new_wrist)
        save[f"{series}_t_wrist"] = t_all
        save[f"{series}_emb_wrist"] = emb_all

    # Refresh global metadata.
    save["schema_version"] = np.asarray(1, dtype=np.int32)
    save["hdf5_path"] = np.asarray(str(hdf5_path))
    save["trim_start"] = np.asarray(int(trim_start), dtype=np.int32)
    save["trim_end"] = np.asarray(int(trim_end), dtype=np.int32)
    save["t_total"] = np.asarray(int(t_total), dtype=np.int32)
    save["vjepa_ckpt"] = np.asarray(str(vjepa_ckpt))
    save["vjepa_img_size"] = np.asarray(int(vjepa_img_size), dtype=np.int32)
    save["vjepa_num_frames"] = np.asarray(int(vjepa_num_frames), dtype=np.int32)

    np.savez_compressed(str(cache_path), **save)


class VJepaEmbedder:
    """Minimal V-JEPA embedder matching siiRL's VideoEmbeddingModel behavior."""

    def __init__(
        self,
        *,
        ckpt_path: Path,
        img_size: int,
        num_frames: int,
        device: torch.device,
        enable_fp16: bool,
        use_sdpa: bool,
    ) -> None:
        # Make vjepa2's `src.*` importable (and siiRL optional imports safe).
        repo_root = Path(__file__).resolve().parents[1]
        siirl_root = repo_root / "siiRL"
        vjepa2_root = siirl_root / "vjepa2"
        for p in (str(siirl_root), str(vjepa2_root)):
            if p not in sys.path:
                sys.path.insert(0, p)

        import src.datasets.utils.video.transforms as video_transforms  # type: ignore
        import src.datasets.utils.video.volume_transforms as volume_transforms  # type: ignore
        from src.models.vision_transformer import vit_giant_xformers_rope  # type: ignore

        self.device = device
        self.num_frames = int(num_frames)
        self.img_size = int(img_size)
        self.auto_cast_dtype = (
            torch.float16 if (bool(enable_fp16) and device.type == "cuda") else torch.float32
        )

        model = vit_giant_xformers_rope(img_size=(self.img_size, self.img_size), num_frames=self.num_frames, use_sdpa=bool(use_sdpa))

        # Match siiRL's safe load behavior: the V-JEPA checkpoint is expected to contain a plain
        # state dict (or a dict with key 'encoder').
        state = torch.load(str(ckpt_path), weights_only=True, map_location=device)
        if isinstance(state, dict) and "encoder" in state:
            state = state["encoder"]
        if not isinstance(state, dict):
            raise ValueError(f"Unexpected checkpoint format at {ckpt_path}")
        # Normalize key prefixes to match model definition.
        state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
        msg = model.load_state_dict(state, strict=False)
        logging.info(f"[vjepa] loaded weights from {ckpt_path} (load_state_dict: {msg})")

        model.eval().to(device)
        self.model = model
        self.embedding_dim = int(model.norm.bias.shape[0])

        short_side_size = int(256.0 / 224.0 * float(self.img_size))
        self.pt_video_transform = video_transforms.Compose(
            [
                video_transforms.Resize(short_side_size, interpolation="bilinear"),
                video_transforms.CenterCrop(size=(self.img_size, self.img_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @torch.inference_mode()
    def embed_batch(self, videos_tchw: list[torch.Tensor]) -> np.ndarray:
        if not videos_tchw:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        xs: list[torch.Tensor] = []
        for v in videos_tchw:
            v = v.to(self.device, non_blocking=True)
            x = self.pt_video_transform(v).to(self.device, non_blocking=True)  # (C, T, H, W)
            xs.append(x)
        x_bcthw = torch.stack(xs, dim=0)  # (B, C, T, H, W)
        use_amp = self.device.type == "cuda" and self.auto_cast_dtype == torch.float16
        with torch.amp.autocast("cuda", dtype=self.auto_cast_dtype, enabled=use_amp):
            y = self.model(x_bcthw)  # (B, N, D)
        y = y.mean(dim=1).to(torch.float32).cpu().numpy()
        return np.asarray(y, dtype=np.float32)


def _iter_hdf5_files(dir_path: Path) -> list[Path]:
    return sorted(Path(dir_path).glob("episode_*.hdf5"))


def _iter_rollouts_files(rollouts_dir: Path) -> tuple[list[Path], list[Path]]:
    succ = sorted((Path(rollouts_dir) / "success").glob("episode_*.hdf5"))
    fail = sorted((Path(rollouts_dir) / "failure").glob("episode_*.hdf5"))
    return succ, fail


@dataclasses.dataclass(frozen=True)
class Args:
    # Inputs
    teleop_hdf5_dir: Path = Path("/home/ztlab/project/ELM/openpi/datasets/GraspAnything/hdf5")
    rollouts_hdf5_dir: Path = Path("/home/ztlab/project/ELM/openpi/datasets/GraspAnything/rollouts_hdf5")
    # Optional: merged dataset directory that contains:
    #   <dataset_hdf5_dir>/{success,failure,intervention}/episode_*.hdf5
    #
    # If set, rollouts_hdf5_dir is effectively treated as dataset_hdf5_dir.
    # intervention_hdf5_dir defaults to "<dataset_hdf5_dir>/intervention" unless explicitly provided.
    dataset_hdf5_dir: Path | None = None
    intervention_hdf5_dir: Path | None = None

    # Output cache directory (will contain cluster centers + per-episode npz).
    out_cache_dir: Path = Path("/home/ztlab/project/ELM/openpi/datasets/GraspAnything/vjepa_value_cache/vjepa_value_v1")
    overwrite: bool = False
    # Embedding cache root directory. If not set, defaults to "<out_cache_dir>/embedding_cache".
    # This cache is shared by success-reference embedding and prefix embeddings.
    embed_cache_dir: Path | None = None

    # Success reference definition
    success_mode: Literal["teleop_all", "rollouts_shortest_p"] = "teleop_all"
    shortest_p: float = 0.3  # used when success_mode=rollouts_shortest_p

    # Which episodes to score (per-frame value). Default: score rollouts only.
    # - rollouts: <rollouts_hdf5_dir>/{success,failure}
    # - teleop: <teleop_hdf5_dir>
    # - dataset: rollouts + intervention (from dataset_hdf5_dir or intervention_hdf5_dir)
    # - all: rollouts + teleop + intervention
    score_sources: Literal["rollouts", "teleop", "dataset", "all"] = "rollouts"

    # Value curve method
    value_method: Literal["progress_warp", "segment_match50"] = "progress_warp"

    # Camera
    camera_mode: Literal["high", "wrist", "both"] = "high"
    camera_fusion: Literal["dist_min", "dist_mean", "emb_mean", "emb_concat"] = "dist_min"

    # Trimming
    epsilon: float = 1e-4
    trim_signal: Literal["action", "qpos"] = "qpos"
    max_episode_len: int | None = None  # if set, episodes longer than this (after trim) are skipped

    # Embedding model
    vjepa_ckpt: Path = Path("/home/ztlab/project/ELM/openpi/siiRL/vjepa2/models/vjepa2/vitg-384.pt")
    vjepa_img_size: int = 384
    vjepa_num_frames: int = 64
    device: str = "auto"  # "auto" | "cuda:0" | "cpu"
    enable_fp16: bool = True

    # Performance
    embed_batch_size: int = 8
    timestep_stride: int = 1
    max_success_episodes: int | None = None
    max_score_episodes: int | None = None
    max_score_rollouts_failure_episodes: int | None = None
    max_score_rollouts_success_episodes: int | None = None

    # Clustering (DBSCAN)
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 2
    # Optional: reduce standardized embeddings with PCA before DBSCAN.
    # 0 disables PCA (cluster in full standardized D-dim space).
    dbscan_pca_dim: int = 0

    # Segment-Match50 method parameters
    segment_match_num_segments: int = 50
    segment_match_window_stride: int = 6
    segment_match_radius: int = 5
    segment_match_first_fixed: bool = True
    # How to score candidate segments during matching.
    # - dist: pick the segment with minimum standardized Euclidean distance to its centers.
    # - gaussian_nll (recommended): pick by Gaussian negative log-likelihood-like cost:
    #   cost = d^2 + 2*sum(log(std)), which penalizes high-variance segments that otherwise
    #   become "attractors" under per-segment StandardScaler.
    # - diffusion: use conditional diffusion denoising error as the observation cost (trained on success_ref).
    segment_match_score: Literal["dist", "gaussian_nll", "diffusion"] = "dist"

    # Diffusion score (used when segment_match_score=diffusion)
    # Reduce standardized embeddings to a smaller latent for diffusion training/inference.
    # 0 disables PCA (train diffusion in full standardized space; not recommended).
    segment_match_diffusion_pca_dim: int = 32
    # Diffusion training hyperparameters.
    segment_match_diffusion_timesteps: int = 100
    segment_match_diffusion_hidden_dim: int = 256
    segment_match_diffusion_num_layers: int = 4
    segment_match_diffusion_lr: float = 1e-4
    segment_match_diffusion_batch_size: int = 256
    segment_match_diffusion_epochs: int = 500
    segment_match_diffusion_seed: int = 0
    # Diffusion inference hyperparameters.
    # Evaluate denoising error at a fixed timestep (smaller means less noise / more local detail).
    segment_match_diffusion_t_eval: int = 20
    # Average denoising error over multiple random noise draws per window to reduce variance.
    segment_match_diffusion_noise_samples: int = 1
    # How to convert matched segments into per-frame value.
    # - hard: value at window end is (seg_idx+1)/num_segments (original spec).
    # - soft_next (recommended): still uses the matched seg_idx, but adds a fractional
    #   progress inside the segment based on distances to seg_idx and seg_idx+1, which
    #   greatly reduces long plateaus where seg_idx stays constant while the scene moves.
    segment_match_value_mode: Literal["hard", "soft_next"] = "soft_next"
    # Segment matching path algorithm.
    # - greedy: previous local search (may suffer from local traps / late rollback)
    # - viterbi: dynamic programming for a globally optimal path under transition constraints
    segment_match_path: Literal["greedy", "viterbi"] = "viterbi"
    # Success-only: encourage segment index to follow time progress to avoid early jumps and late rollback.
    # Penalty is added to observation cost: time_prior * |seg_idx - expected_by_time|.
    segment_match_time_prior_success: float = 0.5
    # Success-only: penalize backward transitions in Viterbi (in addition to radius constraint).
    # Penalty is: back_penalty_success * max(0, prev_seg - seg).
    segment_match_back_penalty_success: float = 0.0

    # Failure-only (non-success episodes in segment_match50): ensure the whole value curve stays below this cap.
    # We only scale DOWN if needed (never scale up).
    segment_match_failure_value_cap: float = 0.8

    # SRPO mapping
    norm_scope: Literal["fail_only", "all_scored"] = "fail_only"
    # When computing d_min/d_max, ignore the first K frames of each trajectory.
    # Those prefix frames will also be forced to value_pred=0 in Stage2.
    minmax_skip_prefix_k: int = 0
    # Whether to enforce a non-decreasing value curve (via cummax).
    # If False, value curves may decrease to reflect real progress regression / mistakes.
    enforce_monotonic: bool = False
    # Distance->value mapping. Default is the new linear mapping (no sigmoid).
    value_mapping: Literal["linear_offset_scale", "sigmoid"] = "linear_offset_scale"
    # For non-success episodes (e.g., rollouts_failure), after monotonicization we scale the
    # curve so that the last value equals this number (before/after warp depending on config).
    failure_end_value: float = 0.5

    # Global monotonic warp f (isotonic regression) fitted from success_ref curves.
    warp_enable: bool = True
    warp_method: Literal["isotonic"] = "isotonic"
    # Time-proportion sampling step for fitting f on success_ref (e.g., 0.05 => every 5%).
    success_fit_step_ratio: float = 0.05
    reward_scale: float = 0.6
    sigmoid_steepness: float = 10.0
    sigmoid_offset: float = 0.5
    norm_eps: float = 1e-6
    # Legacy: kept for backward compatibility; not recommended with the new mapping.
    set_success_terminal_reward: bool = False
    success_terminal_reward: float = 1.0


def _run_segment_match50(
    *,
    args: Args,
    device: torch.device,
    out_dir: Path,
    embed_cache_root: Path,
    embed_cache_dir: Path,
    embed_cfg: dict[str, Any],
    embed_cfg_id: str,
    map_path: Path,
    meta_path: Path,
    success_ref: list[Path],
    scoring: list[tuple[Path, str, bool]],
    success_ref_sig: str,
) -> None:
    """Segment-Match50 value curve method.

    This entrypoint implements:
    - Build/load per-segment success centers (segment_centers.npz)
    - (Further steps: rollouts matching + value_pred) are implemented in later todos.
    """
    num_segments = int(args.segment_match_num_segments)
    if num_segments <= 0:
        raise ValueError("--segment-match-num-segments must be > 0")

    segment_centers_path = out_dir / "segment_centers.npz"

    # Build config id for segment centers.
    try:
        success_ref_ids = sorted(_hdf5_sig_id(p) for p in success_ref)
    except Exception:
        success_ref_ids = sorted(_sha1_short(str(p)) for p in success_ref)
    seg_centers_cfg = {
        "value_method": "segment_match50",
        "success_ref_sig": str(success_ref_sig),
        "success_ref_count": int(len(success_ref)),
        "success_mode": str(args.success_mode),
        "shortest_p": float(args.shortest_p),
        "max_success_episodes": None if args.max_success_episodes is None else int(args.max_success_episodes),
        "camera_mode": str(args.camera_mode),
        "camera_fusion": str(args.camera_fusion),
        "dbscan_eps": float(args.dbscan_eps),
        "dbscan_min_samples": int(args.dbscan_min_samples),
        "dbscan_pca_dim": int(args.dbscan_pca_dim),
        "num_segments": int(num_segments),
        "vjepa_num_frames": int(args.vjepa_num_frames),
        "epsilon": float(args.epsilon),
        "embed_cfg_id": str(embed_cfg_id),
    }
    seg_centers_cfg_id = _sha1_short(json.dumps(seg_centers_cfg, sort_keys=True))

    centers_blob: dict[str, dict[str, list[np.ndarray]]] | None = None
    scaler_blob: dict[str, dict[str, list[dict[str, Any]]]] | None = None
    centers_info: dict[str, dict[str, list[dict[str, Any]]]] | None = None
    # Lazy-initialized V-JEPA embedder. We only create it if we actually need to compute missing embeddings.
    # This is important for clustering hyperparam sweeps where all embeddings are already cached.
    embedder: VJepaEmbedder | None = None

    if segment_centers_path.exists() and not args.overwrite:
        blob = _load_npz_dict(segment_centers_path)
        if blob is not None and "centers" in blob and "segment_centers_cfg_id" in blob:
            stored_id = _decode_bytes(blob.get("segment_centers_cfg_id"))
            stored_embed = _decode_bytes(blob.get("embed_cfg_id")) if "embed_cfg_id" in blob else None
            if stored_id == seg_centers_cfg_id and (stored_embed is None or stored_embed == str(embed_cfg_id)):
                centers_blob = blob["centers"].item()
                scaler_blob = blob["scaler"].item() if "scaler" in blob else None
                centers_info = blob["info"].item() if "info" in blob else None
                if centers_blob is not None and scaler_blob is not None and centers_info is not None:
                    logging.info(
                        f"Loaded segment centers from {segment_centers_path} (cfg_id={seg_centers_cfg_id})"
                    )

    if centers_blob is None or scaler_blob is None or centers_info is None:
        logging.info("Building segment centers for Segment-Match50 (this may take a while on first run)...")

        seg_series = f"seg{num_segments}"
        need_high = args.camera_mode in ("high", "both")
        need_wrist = args.camera_mode in ("wrist", "both")

        # Collect per-task, per-key, per-segment embedding lists for clustering.
        # succ_embs[task][key] -> list-of-lists length S: seg_idx -> list[emb(D,)]
        succ_embs: dict[str, dict[str, list[list[np.ndarray]]]] = {}

        cache_hit = 0
        cache_partial = 0
        cache_miss = 0

        for p in tqdm.tqdm(success_ref, desc="Segment-Match50: embed success_ref segments"):
            try:
                with h5py.File(p, "r") as ep:
                    sig = _get_trim_signal(ep, trim_signal=str(args.trim_signal))
                    span = _trim_static_head_tail(sig, epsilon=float(args.epsilon))
                    if span is None:
                        _append_jsonl(map_path, {"hdf5": str(p), "stage": "seg_success_ref", "status": "skipped", "reason": "trim_none"})
                        continue
                    s, e = span
                    t_total = int(e - s + 1)
                    if args.max_episode_len is not None and t_total > int(args.max_episode_len):
                        _append_jsonl(
                            map_path,
                            {"hdf5": str(p), "stage": "seg_success_ref", "status": "skipped", "reason": "too_long", "t_total": t_total},
                        )
                        continue
                    task = _task_from_hdf5(ep, fallback_name=p.name)
                    bounds = _segment_bounds(t_total, num_segments=num_segments)
                    if len(bounds) != num_segments:
                        raise RuntimeError("segment bounds mismatch")

                    cache_path = _episode_embed_cache_path(embed_cache_dir, p)
                    high_map, wrist_map = _load_episode_series_maps(
                        cache_path,
                        series=seg_series,
                        expected_trim_start=int(s),
                        expected_trim_end=int(e),
                        expected_t_total=int(t_total),
                    )

                    missing_high = [i for i in range(num_segments) if need_high and int(i) not in high_map]
                    missing_wrist = [i for i in range(num_segments) if need_wrist and int(i) not in wrist_map]

                    if (not missing_high) and (not missing_wrist):
                        cache_hit += 1
                    elif (len(missing_high) == num_segments if need_high else True) and (
                        len(missing_wrist) == num_segments if need_wrist else True
                    ):
                        cache_miss += 1
                    else:
                        cache_partial += 1

                    # Compute missing segment embeddings (only load sampled frames).
                    new_high: dict[int, np.ndarray] = {}
                    new_wrist: dict[int, np.ndarray] = {}
                    bs = int(max(1, args.embed_batch_size))
                    to_compute: list[tuple[int, str]] = []
                    for seg_idx in range(num_segments):
                        if seg_idx in missing_high:
                            to_compute.append((seg_idx, "high"))
                        if seg_idx in missing_wrist:
                            to_compute.append((seg_idx, "wrist"))

                    if to_compute:
                        if embedder is None:
                            embedder = VJepaEmbedder(
                                ckpt_path=args.vjepa_ckpt,
                                img_size=int(args.vjepa_img_size),
                                num_frames=int(args.vjepa_num_frames),
                                device=device,
                                enable_fp16=bool(args.enable_fp16),
                                use_sdpa=(device.type == "cuda"),
                            )
                        for i0 in range(0, len(to_compute), bs):
                            i1 = min(len(to_compute), i0 + bs)
                            chunk = to_compute[i0:i1]
                            videos: list[torch.Tensor] = []
                            tags: list[tuple[int, str]] = []
                            for seg_idx, cam_key in chunk:
                                seg_s, seg_e = bounds[int(seg_idx)]
                                abs_idx = _sample_indices_in_range(
                                    int(s) + int(seg_s),
                                    int(s) + int(seg_e),
                                    n=int(args.vjepa_num_frames),
                                    min_index=int(s),
                                )
                                cam_name = "cam_high" if cam_key == "high" else "cam_left_wrist"
                                frames = _load_images_sample(ep, cam_name, abs_idx)
                                videos.append(torch.from_numpy(frames))
                                tags.append((int(seg_idx), cam_key))
                            if not videos:
                                continue
                            assert embedder is not None
                            embs = embedder.embed_batch(videos)
                            for emb, (seg_idx, cam_key) in zip(embs, tags, strict=True):
                                emb32 = np.asarray(emb, dtype=np.float32).reshape(-1)
                                if cam_key == "high":
                                    high_map[int(seg_idx)] = emb32
                                    new_high[int(seg_idx)] = emb32
                                else:
                                    wrist_map[int(seg_idx)] = emb32
                                    new_wrist[int(seg_idx)] = emb32

                    if new_high or new_wrist:
                        _save_episode_series_cache(
                            cache_path,
                            series=seg_series,
                            hdf5_path=p,
                            trim_start=int(s),
                            trim_end=int(e),
                            t_total=int(t_total),
                            vjepa_ckpt=args.vjepa_ckpt,
                            vjepa_img_size=int(args.vjepa_img_size),
                            vjepa_num_frames=int(args.vjepa_num_frames),
                            new_high=(new_high if new_high else None),
                            new_wrist=(new_wrist if new_wrist else None),
                        )

                    # Append to clustering lists.
                    if args.camera_mode == "high":
                        per = succ_embs.setdefault(task, {}).setdefault("single", [[] for _ in range(num_segments)])
                        for seg_idx in range(num_segments):
                            e_seg = high_map.get(int(seg_idx))
                            if e_seg is not None:
                                per[int(seg_idx)].append(np.asarray(e_seg, dtype=np.float32))
                    elif args.camera_mode == "wrist":
                        per = succ_embs.setdefault(task, {}).setdefault("single", [[] for _ in range(num_segments)])
                        for seg_idx in range(num_segments):
                            e_seg = wrist_map.get(int(seg_idx))
                            if e_seg is not None:
                                per[int(seg_idx)].append(np.asarray(e_seg, dtype=np.float32))
                    else:
                        # both
                        if args.camera_fusion in ("dist_min", "dist_mean"):
                            per_h = succ_embs.setdefault(task, {}).setdefault("high", [[] for _ in range(num_segments)])
                            per_w = succ_embs.setdefault(task, {}).setdefault("wrist", [[] for _ in range(num_segments)])
                            for seg_idx in range(num_segments):
                                eh = high_map.get(int(seg_idx))
                                ew = wrist_map.get(int(seg_idx))
                                if eh is not None:
                                    per_h[int(seg_idx)].append(np.asarray(eh, dtype=np.float32))
                                if ew is not None:
                                    per_w[int(seg_idx)].append(np.asarray(ew, dtype=np.float32))
                        else:
                            per_f = succ_embs.setdefault(task, {}).setdefault("fused", [[] for _ in range(num_segments)])
                            for seg_idx in range(num_segments):
                                eh = high_map.get(int(seg_idx))
                                ew = wrist_map.get(int(seg_idx))
                                if eh is None or ew is None:
                                    continue
                                if args.camera_fusion == "emb_mean":
                                    ef = 0.5 * (np.asarray(eh, dtype=np.float32) + np.asarray(ew, dtype=np.float32))
                                else:
                                    ef = np.concatenate([np.asarray(eh, dtype=np.float32), np.asarray(ew, dtype=np.float32)], axis=0)
                                per_f[int(seg_idx)].append(np.asarray(ef, dtype=np.float32))

            except Exception as ex:  # noqa: BLE001
                _append_jsonl(map_path, {"hdf5": str(p), "stage": "seg_success_ref", "status": "error", "error": str(ex)})
                continue

        logging.info(
            f"Segment-Match50 success_ref segment embedding cache: hit={cache_hit} partial={cache_partial} miss={cache_miss} "
            f"(total={len(success_ref)})"
        )

        # Cluster per task/per key/per segment.
        centers_blob = {}
        scaler_blob = {}
        centers_info = {}
        for task, per_key in succ_embs.items():
            centers_blob[task] = {}
            scaler_blob[task] = {}
            centers_info[task] = {}
            for key, embs_by_seg in per_key.items():
                centers_list, scaler_list, info_list = _cluster_centers_per_segment(
                    embs_by_seg,
                    dbscan_eps=float(args.dbscan_eps),
                    dbscan_min_samples=int(args.dbscan_min_samples),
                    dbscan_pca_dim=int(args.dbscan_pca_dim),
                )
                centers_blob[task][key] = centers_list
                scaler_blob[task][key] = scaler_list
                centers_info[task][key] = info_list

        np.savez_compressed(
            str(segment_centers_path),
            schema_version=np.asarray(1, dtype=np.int32),
            embed_cfg_id=np.asarray(str(embed_cfg_id)),
            segment_centers_cfg_id=np.asarray(str(seg_centers_cfg_id)),
            segment_centers_cfg=np.asarray(json.dumps(seg_centers_cfg, sort_keys=True)),
            success_ref_sig=np.asarray(str(success_ref_sig)),
            success_ref_ids=np.asarray(success_ref_ids),
            centers=centers_blob,
            scaler=scaler_blob,
            info=centers_info,
        )
        logging.info(f"Saved segment centers to {segment_centers_path}")

    assert centers_blob is not None and scaler_blob is not None and centers_info is not None

    # Save a human-readable clustering report under out_dir for debugging/diffing.
    try:
        report = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "value_method": "segment_match50",
            "segment_centers_path": str(segment_centers_path),
            "segment_centers_cfg_id": str(seg_centers_cfg_id),
            "embed_cfg_id": str(embed_cfg_id),
            "success_ref_sig": str(success_ref_sig),
            "success_ref_count": int(len(success_ref)),
            "camera_mode": str(args.camera_mode),
            "camera_fusion": str(args.camera_fusion),
            "trim_signal": str(args.trim_signal),
            "epsilon": float(args.epsilon),
            "dbscan_eps": float(args.dbscan_eps),
            "dbscan_min_samples": int(args.dbscan_min_samples),
            "dbscan_pca_dim": int(args.dbscan_pca_dim),
            "num_segments": int(num_segments),
            "tasks": {},
            "summary": {"segments_total": 0, "segments_fallback_mean": 0, "segments_all_noise": 0},
        }
        seg_total = seg_fb = seg_all_noise = 0
        for task, per_key in centers_info.items():
            report["tasks"][task] = {}
            for key, info_list in per_key.items():
                seg_summ: list[dict[str, Any]] = []
                for seg_idx, inf in enumerate(info_list):
                    s = _summarize_dbscan_info(dict(inf))
                    s["seg_idx"] = int(seg_idx)
                    seg_summ.append(s)
                    seg_total += 1
                    if str(s.get("method")) == "dbscan_fallback_mean":
                        seg_fb += 1
                    if s.get("noise_frac") == 1.0:
                        seg_all_noise += 1
                report["tasks"][task][key] = {"segments": seg_summ}
        report["summary"]["segments_total"] = int(seg_total)
        report["summary"]["segments_fallback_mean"] = int(seg_fb)
        report["summary"]["segments_all_noise"] = int(seg_all_noise)
        _write_json(out_dir / "segment_centers_report.json", report)
        logging.info(f"Wrote segment clustering report to {out_dir / 'segment_centers_report.json'}")
    except Exception as ex:  # noqa: BLE001
        logging.warning(f"Failed to write segment_centers_report.json: {ex}")

    # Precompute standardized centers per (task,key,seg) for fast distance computation.
    centers_scaled: dict[str, dict[str, list[np.ndarray]]] = {}
    # scaler_stats[task][key][seg] = (mean, inv_std, use_l2, logdet_std)
    scaler_stats: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray, bool, float]]]] = {}
    for task, per_key in centers_blob.items():
        centers_scaled[task] = {}
        scaler_stats[task] = {}
        for key, centers_list in per_key.items():
            centers_scaled[task][key] = []
            scaler_stats[task][key] = []
            scalers_list = scaler_blob[task][key]
            for seg_idx, (centers_arr, sc) in enumerate(zip(centers_list, scalers_list, strict=True)):
                c = np.asarray(centers_arr, dtype=np.float32)
                mean = np.asarray(sc["mean"], dtype=np.float32).reshape(-1)
                std = np.asarray(sc["std"], dtype=np.float32).reshape(-1)
                inv_std = (1.0 / np.maximum(std, 1e-6)).astype(np.float32)
                use_l2 = bool(sc.get("l2_normalize", False))
                logdet = float(np.sum(np.log(np.maximum(std, 1e-6))).astype(np.float32))
                centers_scaled[task][key].append((c - mean) * inv_std)
                scaler_stats[task][key].append((mean, inv_std, use_l2, logdet))
                _ = seg_idx

    # Rollouts / scoring pass: sliding windows + local matching + interpolate to per-frame curves.
    # NOTE: embedder is lazy-initialized only if we need to compute missing window embeddings.

    num_segments = int(num_segments)
    win_stride = int(max(1, args.segment_match_window_stride))
    radius = int(max(0, args.segment_match_radius))
    seg_series = f"seg{num_segments}"
    win_series = f"win{num_segments}s{win_stride}"

    skip_k = int(max(0, args.minmax_skip_prefix_k))

    # ---------------------------------------------------------------------
    # Diffusion score: train/load conditional diffusion models on success_ref
    # ---------------------------------------------------------------------
    diffusion_scaler_blob: dict[str, dict[str, dict[str, Any]]] | None = None
    diffusion_pca_blob: dict[str, dict[str, dict[str, Any] | None]] | None = None
    diffusion_models_blob: dict[str, dict[str, str]] | None = None  # relpath under out_dir
    diffusion_cfg: dict[str, Any] | None = None
    diffusion_cfg_id: str | None = None

    if str(args.segment_match_score) == "diffusion":
        diff_scaler_path = out_dir / "segmatch50_diffusion_scaler.npz"
        diff_pca_path = out_dir / "segmatch50_diffusion_pca.npz"
        diff_models_path = out_dir / "segmatch50_diffusion_models.npz"
        diff_train_path = out_dir / "segmatch50_diffusion_train.npz"
        diff_report_path = out_dir / "segmatch50_diffusion_report.json"

        diffusion_cfg = {
            "value_method": "segment_match50",
            "segment_centers_cfg_id": str(seg_centers_cfg_id),
            "success_ref_sig": str(success_ref_sig),
            "success_ref_count": int(len(success_ref)),
            "camera_mode": str(args.camera_mode),
            "camera_fusion": str(args.camera_fusion),
            "epsilon": float(args.epsilon),
            "trim_signal": str(args.trim_signal),
            "max_episode_len": None if args.max_episode_len is None else int(args.max_episode_len),
            "max_success_episodes": None if args.max_success_episodes is None else int(args.max_success_episodes),
            "num_segments": int(num_segments),
            "vjepa_num_frames": int(args.vjepa_num_frames),
            "embed_cfg_id": str(embed_cfg_id),
            "diffusion_pca_dim": int(args.segment_match_diffusion_pca_dim),
            "timesteps": int(args.segment_match_diffusion_timesteps),
            "hidden_dim": int(args.segment_match_diffusion_hidden_dim),
            "num_layers": int(args.segment_match_diffusion_num_layers),
            "lr": float(args.segment_match_diffusion_lr),
            "batch_size": int(args.segment_match_diffusion_batch_size),
            "epochs": int(args.segment_match_diffusion_epochs),
            "seed": int(args.segment_match_diffusion_seed),
            "beta_start": 1e-4,
            "beta_end": 0.02,
        }
        diffusion_cfg_id = _sha1_short(json.dumps(diffusion_cfg, sort_keys=True))

        loaded = False
        if diff_scaler_path.exists() and diff_pca_path.exists() and diff_models_path.exists() and not args.overwrite:
            b_sc = _load_npz_dict(diff_scaler_path)
            b_pc = _load_npz_dict(diff_pca_path)
            b_md = _load_npz_dict(diff_models_path)
            if (
                b_sc is not None
                and b_pc is not None
                and b_md is not None
                and _decode_bytes(b_sc.get("diffusion_cfg_id", "")) == str(diffusion_cfg_id)
                and _decode_bytes(b_pc.get("diffusion_cfg_id", "")) == str(diffusion_cfg_id)
                and _decode_bytes(b_md.get("diffusion_cfg_id", "")) == str(diffusion_cfg_id)
                and "scaler" in b_sc
                and "pca" in b_pc
                and "models" in b_md
            ):
                diffusion_scaler_blob = b_sc["scaler"].item()
                diffusion_pca_blob = b_pc["pca"].item()
                diffusion_models_blob = b_md["models"].item()
                ok = True
                try:
                    for _task, per_key in diffusion_models_blob.items():
                        for _key, rel in per_key.items():
                            mp = (out_dir / str(rel)).resolve()
                            if not mp.exists():
                                ok = False
                                break
                        if not ok:
                            break
                except Exception:
                    ok = False
                if ok:
                    loaded = True
                    logging.info(
                        f"Loaded diffusion models from cache (cfg_id={diffusion_cfg_id}) "
                        f"scaler={diff_scaler_path.name} pca={diff_pca_path.name} models={diff_models_path.name}"
                    )

        if not loaded:
            logging.info(
                f"Training diffusion models for Segment-Match50 (cfg_id={diffusion_cfg_id}). "
                f"This uses success_ref segment embeddings and may take a few minutes on first run."
            )

            # Collect raw per-segment embeddings from success_ref (from embed cache; compute missing if needed).
            # train_raw[task][key] -> {"X": list[np.ndarray], "seg": list[int]}
            train_raw: dict[str, dict[str, dict[str, Any]]] = {}

            need_high = args.camera_mode in ("high", "both")
            need_wrist = args.camera_mode in ("wrist", "both")

            for p in tqdm.tqdm(success_ref, desc="Diffusion: collect success_ref seg embeddings"):
                try:
                    with h5py.File(p, "r") as ep:
                        sig = _get_trim_signal(ep, trim_signal=str(args.trim_signal))
                        span = _trim_static_head_tail(sig, epsilon=float(args.epsilon))
                        if span is None:
                            continue
                        s, e = span
                        t_total = int(e - s + 1)
                        if args.max_episode_len is not None and t_total > int(args.max_episode_len):
                            continue
                        task = _task_from_hdf5(ep, fallback_name=p.name)
                        bounds = _segment_bounds(t_total, num_segments=num_segments)
                        if len(bounds) != num_segments:
                            raise RuntimeError("segment bounds mismatch")

                        cache_path = _episode_embed_cache_path(embed_cache_dir, p)
                        high_map, wrist_map = _load_episode_series_maps(
                            cache_path,
                            series=seg_series,
                            expected_trim_start=int(s),
                            expected_trim_end=int(e),
                            expected_t_total=int(t_total),
                        )

                        missing_high = [i for i in range(num_segments) if need_high and int(i) not in high_map]
                        missing_wrist = [i for i in range(num_segments) if need_wrist and int(i) not in wrist_map]

                        # Compute missing segment embeddings (only load sampled frames).
                        new_high: dict[int, np.ndarray] = {}
                        new_wrist: dict[int, np.ndarray] = {}
                        bs = int(max(1, args.embed_batch_size))
                        to_compute: list[tuple[int, str]] = []
                        for seg_idx in range(num_segments):
                            if seg_idx in missing_high:
                                to_compute.append((seg_idx, "high"))
                            if seg_idx in missing_wrist:
                                to_compute.append((seg_idx, "wrist"))
                        if to_compute:
                            if embedder is None:
                                embedder = VJepaEmbedder(
                                    ckpt_path=args.vjepa_ckpt,
                                    img_size=int(args.vjepa_img_size),
                                    num_frames=int(args.vjepa_num_frames),
                                    device=device,
                                    enable_fp16=bool(args.enable_fp16),
                                    use_sdpa=(device.type == "cuda"),
                                )
                            for i0 in range(0, len(to_compute), bs):
                                i1 = min(len(to_compute), i0 + bs)
                                chunk = to_compute[i0:i1]
                                videos: list[torch.Tensor] = []
                                tags: list[tuple[int, str]] = []
                                for seg_idx, cam_key in chunk:
                                    seg_s, seg_e = bounds[int(seg_idx)]
                                    abs_idx = _sample_indices_in_range(
                                        int(s) + int(seg_s),
                                        int(s) + int(seg_e),
                                        n=int(args.vjepa_num_frames),
                                        min_index=int(s),
                                    )
                                    cam_name = "cam_high" if cam_key == "high" else "cam_left_wrist"
                                    frames = _load_images_sample(ep, cam_name, abs_idx)
                                    videos.append(torch.from_numpy(frames))
                                    tags.append((int(seg_idx), cam_key))
                                if not videos:
                                    continue
                                assert embedder is not None
                                embs = embedder.embed_batch(videos)
                                for emb, (seg_idx, cam_key) in zip(embs, tags, strict=True):
                                    emb32 = np.asarray(emb, dtype=np.float32).reshape(-1)
                                    if cam_key == "high":
                                        high_map[int(seg_idx)] = emb32
                                        new_high[int(seg_idx)] = emb32
                                    else:
                                        wrist_map[int(seg_idx)] = emb32
                                        new_wrist[int(seg_idx)] = emb32

                        if new_high or new_wrist:
                            _save_episode_series_cache(
                                cache_path,
                                series=seg_series,
                                hdf5_path=p,
                                trim_start=int(s),
                                trim_end=int(e),
                                t_total=int(t_total),
                                vjepa_ckpt=args.vjepa_ckpt,
                                vjepa_img_size=int(args.vjepa_img_size),
                                vjepa_num_frames=int(args.vjepa_num_frames),
                                new_high=(new_high if new_high else None),
                                new_wrist=(new_wrist if new_wrist else None),
                            )

                        # Append samples
                        if args.camera_mode == "high":
                            per = train_raw.setdefault(task, {}).setdefault("single", {"X": [], "seg": []})
                            for seg_idx in range(num_segments):
                                ee = high_map.get(int(seg_idx))
                                if ee is None:
                                    continue
                                per["X"].append(np.asarray(ee, dtype=np.float32).reshape(-1))
                                per["seg"].append(int(seg_idx))
                        elif args.camera_mode == "wrist":
                            per = train_raw.setdefault(task, {}).setdefault("single", {"X": [], "seg": []})
                            for seg_idx in range(num_segments):
                                ee = wrist_map.get(int(seg_idx))
                                if ee is None:
                                    continue
                                per["X"].append(np.asarray(ee, dtype=np.float32).reshape(-1))
                                per["seg"].append(int(seg_idx))
                        else:
                            # both
                            if args.camera_fusion in ("dist_min", "dist_mean"):
                                per_h = train_raw.setdefault(task, {}).setdefault("high", {"X": [], "seg": []})
                                per_w = train_raw.setdefault(task, {}).setdefault("wrist", {"X": [], "seg": []})
                                for seg_idx in range(num_segments):
                                    eh = high_map.get(int(seg_idx))
                                    ew = wrist_map.get(int(seg_idx))
                                    if eh is not None:
                                        per_h["X"].append(np.asarray(eh, dtype=np.float32).reshape(-1))
                                        per_h["seg"].append(int(seg_idx))
                                    if ew is not None:
                                        per_w["X"].append(np.asarray(ew, dtype=np.float32).reshape(-1))
                                        per_w["seg"].append(int(seg_idx))
                            else:
                                per_f = train_raw.setdefault(task, {}).setdefault("fused", {"X": [], "seg": []})
                                for seg_idx in range(num_segments):
                                    eh = high_map.get(int(seg_idx))
                                    ew = wrist_map.get(int(seg_idx))
                                    if eh is None or ew is None:
                                        continue
                                    if args.camera_fusion == "emb_mean":
                                        ef = 0.5 * (np.asarray(eh, dtype=np.float32) + np.asarray(ew, dtype=np.float32))
                                    else:
                                        ef = np.concatenate(
                                            [np.asarray(eh, dtype=np.float32), np.asarray(ew, dtype=np.float32)], axis=0
                                        )
                                    per_f["X"].append(np.asarray(ef, dtype=np.float32).reshape(-1))
                                    per_f["seg"].append(int(seg_idx))

                except Exception as ex:  # noqa: BLE001
                    logging.warning(f"[diffusion] failed to read {p}: {ex}")
                    continue

            # Fit scaler/PCA and train per task+key model.
            diffusion_scaler_blob = {}
            diffusion_pca_blob = {}
            diffusion_models_blob = {}
            train_blob: dict[str, dict[str, dict[str, Any]]] = {}
            report: dict[str, Any] = {"cfg_id": str(diffusion_cfg_id), "cfg": diffusion_cfg, "tasks": {}}

            for task, per_key in train_raw.items():
                diffusion_scaler_blob[task] = {}
                diffusion_pca_blob[task] = {}
                diffusion_models_blob[task] = {}
                train_blob[task] = {}
                report["tasks"][task] = {}

                for key, data in per_key.items():
                    X_list = list(data.get("X", []))
                    seg_list = list(data.get("seg", []))
                    if not X_list or not seg_list:
                        continue
                    X_raw = np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in X_list], axis=0).astype(np.float32)
                    seg_id = np.asarray(seg_list, dtype=np.int64).reshape(-1)

                    mean, std, use_l2, sc_info = _fit_diffusion_scaler(X_raw)
                    X_proc = _l2_normalize(X_raw) if bool(use_l2) else X_raw
                    X_std = ((X_proc - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)
                    pca = _fit_pca_for_diffusion(X_std, pca_dim=int(args.segment_match_diffusion_pca_dim))
                    X_latent = _pca_transform_for_diffusion(pca, X_std).astype(np.float32)

                    model, tr_info = _train_cond_diffusion_model(
                        X_latent,
                        seg_id,
                        num_segments=int(num_segments),
                        timesteps=int(args.segment_match_diffusion_timesteps),
                        hidden_dim=int(args.segment_match_diffusion_hidden_dim),
                        num_layers=int(args.segment_match_diffusion_num_layers),
                        lr=float(args.segment_match_diffusion_lr),
                        batch_size=int(args.segment_match_diffusion_batch_size),
                        epochs=int(args.segment_match_diffusion_epochs),
                        seed=int(args.segment_match_diffusion_seed),
                        device=device,
                    )

                    task_id = _sha1_short(str(task))
                    model_name = f"segmatch50_diffusion_model__{diffusion_cfg_id}__{task_id}__{key}.pt"
                    model_path = out_dir / model_name
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "dim_in": int(X_latent.shape[1]),
                            "num_segments": int(num_segments),
                            "hidden_dim": int(args.segment_match_diffusion_hidden_dim),
                            "num_layers": int(args.segment_match_diffusion_num_layers),
                            "timesteps": int(args.segment_match_diffusion_timesteps),
                            "beta_start": 1e-4,
                            "beta_end": 0.02,
                            "cfg_id": str(diffusion_cfg_id),
                            "task": str(task),
                            "key": str(key),
                        },
                        str(model_path),
                    )

                    diffusion_scaler_blob[task][key] = {
                        "mean": np.asarray(mean, dtype=np.float32),
                        "std": np.asarray(std, dtype=np.float32),
                        "l2_normalize": bool(use_l2),
                        "info": sc_info,
                    }
                    diffusion_pca_blob[task][key] = pca
                    diffusion_models_blob[task][key] = str(Path(model_name))

                    # Store training latent data for diagnostics (best-effort; may be large for big datasets).
                    train_blob[task][key] = {
                        "X": np.asarray(X_latent, dtype=np.float32),
                        "seg_id": np.asarray(seg_id, dtype=np.int32),
                    }

                    seg_counts = np.bincount(seg_id.astype(np.int32), minlength=int(num_segments)).astype(np.int32).tolist()
                    report["tasks"][task][key] = {
                        "n_samples": int(X_latent.shape[0]),
                        "dim_in": int(X_latent.shape[1]),
                        "seg_counts": seg_counts,
                        "scaler_use_l2": bool(use_l2),
                        "scaler_norm_cv": sc_info.get("norm_cv", None),
                        "scaler_norm_ratio_p95_p5": sc_info.get("norm_ratio_p95_p5", None),
                        "train": tr_info,
                        "model_path": str(model_name),
                    }

            np.savez_compressed(
                str(diff_scaler_path),
                diffusion_cfg_id=np.asarray(str(diffusion_cfg_id)),
                diffusion_cfg=np.asarray(json.dumps(diffusion_cfg, ensure_ascii=False, sort_keys=True)),
                scaler=np.asarray(diffusion_scaler_blob, dtype=object),
            )
            np.savez_compressed(
                str(diff_pca_path),
                diffusion_cfg_id=np.asarray(str(diffusion_cfg_id)),
                pca=np.asarray(diffusion_pca_blob, dtype=object),
            )
            np.savez_compressed(
                str(diff_models_path),
                diffusion_cfg_id=np.asarray(str(diffusion_cfg_id)),
                models=np.asarray(diffusion_models_blob, dtype=object),
            )
            # Best-effort: store training latent data for quick diagnostics.
            try:
                np.savez_compressed(
                    str(diff_train_path),
                    diffusion_cfg_id=np.asarray(str(diffusion_cfg_id)),
                    train=np.asarray(train_blob, dtype=object),
                )
            except Exception as ex:  # noqa: BLE001
                logging.warning(f"Failed to write segmatch50_diffusion_train.npz: {ex}")
            try:
                _write_json(diff_report_path, report)
            except Exception as ex:  # noqa: BLE001
                logging.warning(f"Failed to write segmatch50_diffusion_report.json: {ex}")

            logging.info(
                f"Diffusion training done (cfg_id={diffusion_cfg_id}). "
                f"Wrote: {diff_scaler_path.name}, {diff_pca_path.name}, {diff_models_path.name}"
            )

    diffusion_model_cache: dict[tuple[str, str], _CondDiffusionDenoiser] = {}
    diffusion_sched: dict[str, torch.Tensor] | None = None
    if str(args.segment_match_score) == "diffusion":
        diffusion_sched = _diffusion_precompute(
            timesteps=int(args.segment_match_diffusion_timesteps),
            device=device,
            beta_start=1e-4,
            beta_end=0.02,
        )

    written = 0
    reused = 0
    skipped = 0

    for p, source, is_success in tqdm.tqdm(scoring, desc="Segment-Match50: score"):
        out_npz = out_dir / "episode_values" / f"{p.name}.npz"
        if out_npz.exists() and not args.overwrite:
            try:
                z = np.load(str(out_npz), allow_pickle=True)
                if ("value_method" not in z.files) or (_decode_bytes(z["value_method"]) != "segment_match50"):
                    raise RuntimeError("value_method mismatch")
                if ("segment_centers_cfg_id" not in z.files) or (_decode_bytes(z["segment_centers_cfg_id"]) != str(seg_centers_cfg_id)):
                    raise RuntimeError("segment_centers_cfg_id mismatch")
                if ("segment_match_num_segments" not in z.files) or (int(np.asarray(z["segment_match_num_segments"]).item()) != int(num_segments)):
                    raise RuntimeError("num_segments mismatch")
                if ("segment_match_window_stride" not in z.files) or (int(np.asarray(z["segment_match_window_stride"]).item()) != int(win_stride)):
                    raise RuntimeError("window_stride mismatch")
                if ("segment_match_radius" not in z.files) or (int(np.asarray(z["segment_match_radius"]).item()) != int(radius)):
                    raise RuntimeError("radius mismatch")
                if ("segment_match_first_fixed" not in z.files) or (
                    bool(int(np.asarray(z["segment_match_first_fixed"]).item())) != bool(args.segment_match_first_fixed)
                ):
                    raise RuntimeError("first_fixed mismatch")
                if ("segment_match_score" not in z.files) or (_decode_bytes(z["segment_match_score"]) != str(args.segment_match_score)):
                    raise RuntimeError("segment_match_score mismatch")
                if str(args.segment_match_score) == "diffusion":
                    if ("segment_match_diffusion_pca_dim" not in z.files) or (
                        int(np.asarray(z["segment_match_diffusion_pca_dim"]).item()) != int(args.segment_match_diffusion_pca_dim)
                    ):
                        raise RuntimeError("segment_match_diffusion_pca_dim mismatch")
                    if ("segment_match_diffusion_timesteps" not in z.files) or (
                        int(np.asarray(z["segment_match_diffusion_timesteps"]).item()) != int(args.segment_match_diffusion_timesteps)
                    ):
                        raise RuntimeError("segment_match_diffusion_timesteps mismatch")
                    if ("segment_match_diffusion_hidden_dim" not in z.files) or (
                        int(np.asarray(z["segment_match_diffusion_hidden_dim"]).item()) != int(args.segment_match_diffusion_hidden_dim)
                    ):
                        raise RuntimeError("segment_match_diffusion_hidden_dim mismatch")
                    if ("segment_match_diffusion_num_layers" not in z.files) or (
                        int(np.asarray(z["segment_match_diffusion_num_layers"]).item()) != int(args.segment_match_diffusion_num_layers)
                    ):
                        raise RuntimeError("segment_match_diffusion_num_layers mismatch")
                    if ("segment_match_diffusion_lr" not in z.files) or (
                        abs(float(np.asarray(z["segment_match_diffusion_lr"]).item()) - float(args.segment_match_diffusion_lr)) > 1e-12
                    ):
                        raise RuntimeError("segment_match_diffusion_lr mismatch")
                    if ("segment_match_diffusion_batch_size" not in z.files) or (
                        int(np.asarray(z["segment_match_diffusion_batch_size"]).item()) != int(args.segment_match_diffusion_batch_size)
                    ):
                        raise RuntimeError("segment_match_diffusion_batch_size mismatch")
                    if ("segment_match_diffusion_epochs" not in z.files) or (
                        int(np.asarray(z["segment_match_diffusion_epochs"]).item()) != int(args.segment_match_diffusion_epochs)
                    ):
                        raise RuntimeError("segment_match_diffusion_epochs mismatch")
                    if ("segment_match_diffusion_seed" not in z.files) or (
                        int(np.asarray(z["segment_match_diffusion_seed"]).item()) != int(args.segment_match_diffusion_seed)
                    ):
                        raise RuntimeError("segment_match_diffusion_seed mismatch")
                    if ("segment_match_diffusion_t_eval" not in z.files) or (
                        int(np.asarray(z["segment_match_diffusion_t_eval"]).item()) != int(args.segment_match_diffusion_t_eval)
                    ):
                        raise RuntimeError("segment_match_diffusion_t_eval mismatch")
                    if ("segment_match_diffusion_noise_samples" not in z.files) or (
                        int(np.asarray(z["segment_match_diffusion_noise_samples"]).item()) != int(args.segment_match_diffusion_noise_samples)
                    ):
                        raise RuntimeError("segment_match_diffusion_noise_samples mismatch")
                if ("segment_match_value_mode" not in z.files) or (_decode_bytes(z["segment_match_value_mode"]) != str(args.segment_match_value_mode)):
                    raise RuntimeError("segment_match_value_mode mismatch")
                if ("enforce_monotonic" not in z.files) or (bool(int(np.asarray(z["enforce_monotonic"]).item())) != bool(args.enforce_monotonic)):
                    raise RuntimeError("enforce_monotonic mismatch")
                if ("segment_match_path" not in z.files) or (_decode_bytes(z["segment_match_path"]) != str(args.segment_match_path)):
                    raise RuntimeError("segment_match_path mismatch")
                if ("segment_match_time_prior_success" not in z.files) or (
                    abs(float(np.asarray(z["segment_match_time_prior_success"]).item()) - float(args.segment_match_time_prior_success)) > 1e-6
                ):
                    raise RuntimeError("segment_match_time_prior_success mismatch")
                if ("segment_match_back_penalty_success" not in z.files) or (
                    abs(float(np.asarray(z["segment_match_back_penalty_success"]).item()) - float(args.segment_match_back_penalty_success)) > 1e-6
                ):
                    raise RuntimeError("segment_match_back_penalty_success mismatch")
                if not bool(is_success):
                    if ("segment_match_failure_value_cap" not in z.files) or (
                        abs(float(np.asarray(z["segment_match_failure_value_cap"]).item()) - float(args.segment_match_failure_value_cap)) > 1e-6
                    ):
                        raise RuntimeError("segment_match_failure_value_cap mismatch")
                if ("minmax_skip_prefix_k" not in z.files) or (int(np.asarray(z["minmax_skip_prefix_k"]).item()) != int(skip_k)):
                    raise RuntimeError("skip_k mismatch")
                if "value_pred" in z.files and "distance" in z.files:
                    reused += 1
                    continue
            except Exception:
                pass

        try:
            with h5py.File(p, "r") as ep:
                sig = _get_trim_signal(ep, trim_signal=str(args.trim_signal))
                span = _trim_static_head_tail(sig, epsilon=float(args.epsilon))
                if span is None:
                    skipped += 1
                    _append_jsonl(map_path, {"hdf5": str(p), "stage": "segmatch_score", "status": "skipped", "reason": "trim_none"})
                    continue
                s, e = span
                t_total = int(e - s + 1)
                if args.max_episode_len is not None and t_total > int(args.max_episode_len):
                    skipped += 1
                    _append_jsonl(
                        map_path,
                        {"hdf5": str(p), "stage": "segmatch_score", "status": "skipped", "reason": "too_long", "t_total": t_total},
                    )
                    continue
                task = _task_from_hdf5(ep, fallback_name=p.name)
                if task not in centers_scaled:
                    raise RuntimeError(f"Task {task!r} missing in segment centers. Did success_ref contain this task?")

                # Sliding windows
                win_len = int(max(1, int(t_total) // int(num_segments)))
                windows: list[tuple[int, int, int]] = []  # (k, start, end) in [0, t_total)
                k = 0
                for start in range(0, t_total, win_stride):
                    end = min(t_total - 1, start + win_len - 1)
                    windows.append((k, int(start), int(end)))
                    k += 1
                    if end >= t_total - 1:
                        break
                if not windows:
                    skipped += 1
                    _append_jsonl(map_path, {"hdf5": str(p), "stage": "segmatch_score", "status": "skipped", "reason": "no_windows"})
                    continue

                # Load cached window embeddings (series) and compute missing.
                cache_path = _episode_embed_cache_path(embed_cache_dir, p)
                high_map, wrist_map = _load_episode_series_maps(
                    cache_path,
                    series=win_series,
                    expected_trim_start=int(s),
                    expected_trim_end=int(e),
                    expected_t_total=int(t_total),
                )

                need_high = args.camera_mode in ("high", "both")
                need_wrist = args.camera_mode in ("wrist", "both")
                want_t = [int(end) for _, _, end in windows]

                missing_high = [t for t in want_t if need_high and int(t) not in high_map]
                missing_wrist = [t for t in want_t if need_wrist and int(t) not in wrist_map]

                # Compute missing in batches.
                bs = int(max(1, args.embed_batch_size))
                new_high: dict[int, np.ndarray] = {}
                new_wrist: dict[int, np.ndarray] = {}

                to_compute: list[tuple[int, str, int, int]] = []  # (t_end, cam_key, start, end)
                # Map from t_end -> (start,end) for sampling.
                se_map = {int(end): (int(start), int(end)) for _, start, end in windows}
                for t_end in missing_high:
                    st, ed = se_map[int(t_end)]
                    to_compute.append((int(t_end), "high", int(st), int(ed)))
                for t_end in missing_wrist:
                    st, ed = se_map[int(t_end)]
                    to_compute.append((int(t_end), "wrist", int(st), int(ed)))

                if to_compute:
                    if embedder is None:
                        embedder = VJepaEmbedder(
                            ckpt_path=args.vjepa_ckpt,
                            img_size=int(args.vjepa_img_size),
                            num_frames=int(args.vjepa_num_frames),
                            device=device,
                            enable_fp16=bool(args.enable_fp16),
                            use_sdpa=(device.type == "cuda"),
                        )
                    for i0 in range(0, len(to_compute), bs):
                        i1 = min(len(to_compute), i0 + bs)
                        chunk = to_compute[i0:i1]
                        videos: list[torch.Tensor] = []
                        tags: list[tuple[int, str]] = []
                        for t_end, cam_key, st, ed in chunk:
                            abs_idx = _sample_indices_in_range(
                                int(s) + int(st), int(s) + int(ed), n=int(args.vjepa_num_frames), min_index=int(s)
                            )
                            cam_name = "cam_high" if cam_key == "high" else "cam_left_wrist"
                            frames = _load_images_sample(ep, cam_name, abs_idx)
                            videos.append(torch.from_numpy(frames))
                            tags.append((int(t_end), cam_key))
                        if not videos:
                            continue
                        assert embedder is not None
                        embs = embedder.embed_batch(videos)
                        for emb, (t_end, cam_key) in zip(embs, tags, strict=True):
                            emb32 = np.asarray(emb, dtype=np.float32).reshape(-1)
                            if cam_key == "high":
                                high_map[int(t_end)] = emb32
                                new_high[int(t_end)] = emb32
                            else:
                                wrist_map[int(t_end)] = emb32
                                new_wrist[int(t_end)] = emb32

                if new_high or new_wrist:
                    _save_episode_series_cache(
                        cache_path,
                        series=win_series,
                        hdf5_path=p,
                        trim_start=int(s),
                        trim_end=int(e),
                        t_total=int(t_total),
                        vjepa_ckpt=args.vjepa_ckpt,
                        vjepa_img_size=int(args.vjepa_img_size),
                        vjepa_num_frames=int(args.vjepa_num_frames),
                        new_high=(new_high if new_high else None),
                        new_wrist=(new_wrist if new_wrist else None),
                    )

        except Exception as ex:  # noqa: BLE001
            _append_jsonl(map_path, {"hdf5": str(p), "stage": "segmatch_score", "status": "error", "error": str(ex)})
            continue

        # Matching (outside h5 context; only uses cached embeddings and centers)
        window_end_t_sparse = np.asarray([int(end) for _, _, end in windows], dtype=np.int32)
        K = int(window_end_t_sparse.size)
        N = int(num_segments)

        # Precompute observation costs for all candidate segments (K, N).
        # This avoids greedy local traps and enables Viterbi global optimization.
        cost_mat = np.full((K, N), np.inf, dtype=np.float32)
        dist_mat = np.full((K, N), np.inf, dtype=np.float32)
        if str(args.segment_match_score) == "diffusion":
            if diffusion_cfg_id is None or diffusion_scaler_blob is None or diffusion_pca_blob is None or diffusion_models_blob is None:
                raise RuntimeError("Diffusion cache missing. Re-run with --segment-match-score diffusion (and --overwrite if needed).")
            if diffusion_sched is None:
                raise RuntimeError("Diffusion schedule missing (internal error).")

            def _get_diffusion_model(task_name: str, key_name: str) -> _CondDiffusionDenoiser:
                ck = (str(task_name), str(key_name))
                m = diffusion_model_cache.get(ck)
                if m is not None:
                    return m
                rel = diffusion_models_blob.get(str(task_name), {}).get(str(key_name))
                if not rel:
                    raise RuntimeError(f"Missing diffusion model for task={task_name!r} key={key_name!r}")
                ckpt = torch.load(str((out_dir / str(rel)).resolve()), map_location=device)
                if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
                    raise RuntimeError(f"Bad diffusion checkpoint at {out_dir / str(rel)}")
                dim_in = int(ckpt.get("dim_in", 0))
                nseg_ckpt = int(ckpt.get("num_segments", int(num_segments)))
                if dim_in <= 0 or nseg_ckpt != int(num_segments):
                    raise RuntimeError("Diffusion checkpoint dimension mismatch")
                hd = int(ckpt.get("hidden_dim", int(args.segment_match_diffusion_hidden_dim)))
                nl = int(ckpt.get("num_layers", int(args.segment_match_diffusion_num_layers)))
                model = _CondDiffusionDenoiser(dim_in=int(dim_in), num_segments=int(num_segments), hidden_dim=int(hd), num_layers=int(nl)).to(device)
                model.load_state_dict(ckpt["state_dict"], strict=True)
                model.eval()
                diffusion_model_cache[ck] = model
                return model

            def _diffusion_preprocess(task_name: str, key_name: str, X_raw: np.ndarray) -> np.ndarray:
                sc = diffusion_scaler_blob.get(str(task_name), {}).get(str(key_name))
                if sc is None:
                    raise RuntimeError(f"Missing diffusion scaler for task={task_name!r} key={key_name!r}")
                mean = np.asarray(sc["mean"], dtype=np.float32).reshape(-1)
                std = np.asarray(sc["std"], dtype=np.float32).reshape(-1)
                use_l2 = bool(sc.get("l2_normalize", False))
                X = np.asarray(X_raw, dtype=np.float32)
                if bool(use_l2):
                    X = _l2_normalize(X)
                X_std = ((X - mean.reshape(1, -1)) / np.maximum(std.reshape(1, -1), 1e-6)).astype(np.float32)
                pca = diffusion_pca_blob.get(str(task_name), {}).get(str(key_name))
                return _pca_transform_for_diffusion(pca, X_std).astype(np.float32)

            def _diffusion_cost_matrix(task_name: str, key_name: str, X_raw: np.ndarray) -> np.ndarray:
                X_lat = _diffusion_preprocess(str(task_name), str(key_name), X_raw)
                model = _get_diffusion_model(str(task_name), str(key_name))

                K0 = int(X_lat.shape[0])
                N0 = int(num_segments)
                if K0 <= 0:
                    return np.zeros((0, N0), dtype=np.float32)

                t_eval = int(args.segment_match_diffusion_t_eval)
                T = int(args.segment_match_diffusion_timesteps)
                t_eval = int(max(0, min(t_eval, max(0, T - 1))))
                s1 = diffusion_sched["sqrt_alpha_cumprod"][t_eval]
                s2 = diffusion_sched["sqrt_one_minus_alpha_cumprod"][t_eval]
                n_noise = int(max(1, args.segment_match_diffusion_noise_samples))

                X0 = torch.from_numpy(X_lat).to(device=device, dtype=torch.float32)
                cost_acc = torch.zeros((K0, N0), device=device, dtype=torch.float32)

                # Chunk segments to avoid large (K*N) batches on smaller GPUs.
                max_eval_batch = 8192
                seg_block = int(max(1, max_eval_batch // max(1, K0)))

                with torch.inference_mode():
                    for _ in range(n_noise):
                        noise = torch.randn_like(X0)
                        xt = s1 * X0 + s2 * noise
                        for n0 in range(0, N0, seg_block):
                            n1 = min(N0, n0 + seg_block)
                            nb = int(n1 - n0)
                            xt_rep = xt.repeat(nb, 1)
                            noise_rep = noise.repeat(nb, 1)
                            seg_ids = torch.arange(n0, n1, device=device, dtype=torch.int64).repeat_interleave(K0)
                            t_rep = torch.full((int(K0 * nb),), int(t_eval), device=device, dtype=torch.int64)
                            pred = model(xt_rep, t_rep, seg_ids)
                            mse = ((pred - noise_rep) ** 2).mean(dim=1).reshape(nb, K0).transpose(0, 1)
                            cost_acc[:, n0:n1] += mse

                cost = (cost_acc / float(n_noise)).detach().cpu().numpy().astype(np.float32)
                return cost

            # Build per-window raw embedding matrices and compute fused costs.
            if args.camera_mode == "both" and args.camera_fusion in ("dist_min", "dist_mean"):
                Xh = np.stack([np.asarray(high_map[int(t)], dtype=np.float32).reshape(-1) for t in window_end_t_sparse.tolist()], axis=0)
                Xw = np.stack([np.asarray(wrist_map[int(t)], dtype=np.float32).reshape(-1) for t in window_end_t_sparse.tolist()], axis=0)
                ch = _diffusion_cost_matrix(task, "high", Xh)
                cw = _diffusion_cost_matrix(task, "wrist", Xw)
                cost_mat = np.minimum(ch, cw).astype(np.float32) if args.camera_fusion == "dist_min" else (0.5 * (ch + cw)).astype(np.float32)
                dist_mat = cost_mat.copy()
            elif args.camera_mode == "both" and args.camera_fusion in ("emb_mean", "emb_concat"):
                Xf_list: list[np.ndarray] = []
                for t_end_i in window_end_t_sparse.tolist():
                    eh0 = high_map.get(int(t_end_i))
                    ew0 = wrist_map.get(int(t_end_i))
                    if eh0 is None or ew0 is None:
                        raise RuntimeError("Missing cached window embeddings (high/wrist).")
                    if args.camera_fusion == "emb_mean":
                        ef = 0.5 * (np.asarray(eh0, dtype=np.float32) + np.asarray(ew0, dtype=np.float32))
                    else:
                        ef = np.concatenate([np.asarray(eh0, dtype=np.float32), np.asarray(ew0, dtype=np.float32)], axis=0)
                    Xf_list.append(np.asarray(ef, dtype=np.float32).reshape(-1))
                Xf = np.stack(Xf_list, axis=0).astype(np.float32)
                cf = _diffusion_cost_matrix(task, "fused", Xf)
                cost_mat = cf.astype(np.float32)
                dist_mat = cost_mat.copy()
            else:
                # single camera
                Xs = np.stack(
                    [
                        np.asarray((high_map if args.camera_mode == "high" else wrist_map).get(int(t)), dtype=np.float32).reshape(-1)
                        for t in window_end_t_sparse.tolist()
                    ],
                    axis=0,
                ).astype(np.float32)
                cs = _diffusion_cost_matrix(task, "single", Xs)
                cost_mat = cs.astype(np.float32)
                dist_mat = cost_mat.copy()
        else:
            for j, t_end in enumerate(window_end_t_sparse.tolist()):
                t_end_i = int(t_end)
                if args.camera_mode == "both" and args.camera_fusion in ("dist_min", "dist_mean"):
                    eh0 = high_map.get(t_end_i)
                    ew0 = wrist_map.get(t_end_i)
                    if eh0 is None or ew0 is None:
                        raise RuntimeError("Missing cached window embeddings (high/wrist).")
                    eh_raw = np.asarray(eh0, dtype=np.float32).reshape(-1)
                    ew_raw = np.asarray(ew0, dtype=np.float32).reshape(-1)
                    eh_l2 = _l2_normalize(eh_raw)
                    ew_l2 = _l2_normalize(ew_raw)
                    for cand in range(N):
                        c_h = centers_scaled[task]["high"][cand]
                        c_w = centers_scaled[task]["wrist"][cand]
                        mean_h, invstd_h, l2_h, logdet_h = scaler_stats[task]["high"][cand]
                        mean_w, invstd_w, l2_w, logdet_w = scaler_stats[task]["wrist"][cand]
                        eh = eh_l2 if l2_h else eh_raw
                        ew = ew_l2 if l2_w else ew_raw
                        eh_s = (eh - mean_h) * invstd_h
                        ew_s = (ew - mean_w) * invstd_w
                        dh2 = _min_dist2_to_centers(eh_s, c_h)
                        dw2 = _min_dist2_to_centers(ew_s, c_w)
                        dh = float(np.sqrt(dh2))
                        dw = float(np.sqrt(dw2))
                        dist = float(min(dh, dw)) if args.camera_fusion == "dist_min" else float(0.5 * (dh + dw))
                        if args.segment_match_score == "gaussian_nll":
                            ch = float(dh2 + 2.0 * float(logdet_h))
                            cw = float(dw2 + 2.0 * float(logdet_w))
                            cost = float(min(ch, cw)) if args.camera_fusion == "dist_min" else float(0.5 * (ch + cw))
                        else:
                            cost = dist
                        dist_mat[j, cand] = float(dist)
                        cost_mat[j, cand] = float(cost)
                elif args.camera_mode == "both" and args.camera_fusion in ("emb_mean", "emb_concat"):
                    eh0 = high_map.get(t_end_i)
                    ew0 = wrist_map.get(t_end_i)
                    if eh0 is None or ew0 is None:
                        raise RuntimeError("Missing cached window embeddings (high/wrist).")
                    if args.camera_fusion == "emb_mean":
                        ef_raw = 0.5 * (np.asarray(eh0, dtype=np.float32) + np.asarray(ew0, dtype=np.float32))
                    else:
                        ef_raw = np.concatenate(
                            [np.asarray(eh0, dtype=np.float32), np.asarray(ew0, dtype=np.float32)], axis=0
                        )
                    ef_raw = np.asarray(ef_raw, dtype=np.float32).reshape(-1)
                    ef_l2 = _l2_normalize(ef_raw)
                    for cand in range(N):
                        c_f = centers_scaled[task]["fused"][cand]
                        mean_f, invstd_f, l2_f, logdet_f = scaler_stats[task]["fused"][cand]
                        ef = ef_l2 if l2_f else ef_raw
                        ef_s = (ef - mean_f) * invstd_f
                        d2 = _min_dist2_to_centers(ef_s, c_f)
                        dist = float(np.sqrt(d2))
                        cost = float(d2 + 2.0 * float(logdet_f)) if args.segment_match_score == "gaussian_nll" else float(dist)
                        dist_mat[j, cand] = float(dist)
                        cost_mat[j, cand] = float(cost)
                else:
                    # single camera
                    e0 = high_map.get(t_end_i) if args.camera_mode == "high" else wrist_map.get(t_end_i)
                    if e0 is None:
                        raise RuntimeError("Missing cached window embeddings (single camera).")
                    e_raw = np.asarray(e0, dtype=np.float32).reshape(-1)
                    e_l2 = _l2_normalize(e_raw)
                    for cand in range(N):
                        c_s = centers_scaled[task]["single"][cand]
                        mean_s, invstd_s, l2_s, logdet_s = scaler_stats[task]["single"][cand]
                        ee = e_l2 if l2_s else e_raw
                        ee_s = (ee - mean_s) * invstd_s
                        d2 = _min_dist2_to_centers(ee_s, c_s)
                        dist = float(np.sqrt(d2))
                        cost = float(d2 + 2.0 * float(logdet_s)) if args.segment_match_score == "gaussian_nll" else float(dist)
                        dist_mat[j, cand] = float(dist)
                        cost_mat[j, cand] = float(cost)

        # Choose a segment-index path.
        if args.segment_match_path == "greedy":
            path = np.zeros((K,), dtype=np.int32)
            prev = 0
            for j in range(K):
                if j == 0 and bool(args.segment_match_first_fixed):
                    n = 0
                else:
                    lo = max(0, int(prev) - int(radius))
                    hi = min(N - 1, int(prev) + int(radius))
                    n = int(lo + int(np.argmin(cost_mat[j, lo : hi + 1])))
                path[j] = int(n)
                prev = int(n)
            cost_eff = cost_mat  # for debug only
        elif args.segment_match_path == "viterbi":
            cost_eff = cost_mat.copy()
            if bool(is_success) and float(args.segment_match_time_prior_success) > 0.0 and int(t_total) > 1:
                expected = np.rint((window_end_t_sparse.astype(np.float32) / float(int(t_total) - 1)) * float(N - 1)).astype(np.int32)
                grid = np.arange(N, dtype=np.int32)[None, :]
                cost_eff = cost_eff + float(args.segment_match_time_prior_success) * np.abs(grid - expected[:, None]).astype(np.float32)
            back_pen = float(args.segment_match_back_penalty_success) if bool(is_success) else 0.0
            path = _viterbi_path(
                cost_eff,
                radius=int(radius),
                first_fixed=bool(args.segment_match_first_fixed),
                back_penalty=float(back_pen),
            )
        else:
            raise ValueError(f"Unknown segment_match_path: {args.segment_match_path}")

        match_seg_idx_sparse = np.asarray(path, dtype=np.int32).reshape(-1)
        distance_sparse = dist_mat[np.arange(K, dtype=np.int32), match_seg_idx_sparse].astype(np.float32)
        match_cost_sparse = cost_mat[np.arange(K, dtype=np.int32), match_seg_idx_sparse].astype(np.float32)
        match_cost_eff_sparse = cost_eff[np.arange(K, dtype=np.int32), match_seg_idx_sparse].astype(np.float32)

        # Interpolate sparse distance to per-frame distance.
        x = np.arange(int(t_total), dtype=np.float32)
        xp = window_end_t_sparse.astype(np.float32)
        dist_full = np.interp(x, xp, distance_sparse.astype(np.float32), left=float(distance_sparse[0]), right=float(distance_sparse[-1])).astype(np.float32)

        # Build value curve from matched segment indices.
        # We first compute sparse window-end values, then interpolate to per-frame.
        value_end_sparse = np.zeros((len(windows),), dtype=np.float32)

        def _fused_dist_to_seg(t_end: int, seg_idx: int) -> float:
            seg_idx = int(seg_idx)
            if args.camera_mode == "both" and args.camera_fusion in ("dist_min", "dist_mean"):
                eh_raw = high_map.get(int(t_end))
                ew_raw = wrist_map.get(int(t_end))
                if eh_raw is None or ew_raw is None:
                    raise RuntimeError("Missing cached window embeddings (high/wrist).")
                c_h = centers_scaled[task]["high"][seg_idx]
                c_w = centers_scaled[task]["wrist"][seg_idx]
                mean_h, invstd_h, l2_h, _logdet_h = scaler_stats[task]["high"][seg_idx]
                mean_w, invstd_w, l2_w, _logdet_w = scaler_stats[task]["wrist"][seg_idx]
                eh = np.asarray(eh_raw, dtype=np.float32).reshape(-1)
                ew = np.asarray(ew_raw, dtype=np.float32).reshape(-1)
                if l2_h:
                    eh = _l2_normalize(eh)
                if l2_w:
                    ew = _l2_normalize(ew)
                eh_s = (eh - mean_h) * invstd_h
                ew_s = (ew - mean_w) * invstd_w
                dh = float(np.sqrt(_min_dist2_to_centers(eh_s, c_h)))
                dw = float(np.sqrt(_min_dist2_to_centers(ew_s, c_w)))
                return float(min(dh, dw)) if args.camera_fusion == "dist_min" else float(0.5 * (dh + dw))
            if args.camera_mode == "both" and args.camera_fusion in ("emb_mean", "emb_concat"):
                eh_raw = high_map.get(int(t_end))
                ew_raw = wrist_map.get(int(t_end))
                if eh_raw is None or ew_raw is None:
                    raise RuntimeError("Missing cached window embeddings (high/wrist).")
                if args.camera_fusion == "emb_mean":
                    ef0 = 0.5 * (np.asarray(eh_raw, dtype=np.float32) + np.asarray(ew_raw, dtype=np.float32))
                else:
                    ef0 = np.concatenate([np.asarray(eh_raw, dtype=np.float32), np.asarray(ew_raw, dtype=np.float32)], axis=0)
                c_f = centers_scaled[task]["fused"][seg_idx]
                mean_f, invstd_f, l2_f, _logdet_f = scaler_stats[task]["fused"][seg_idx]
                ef = np.asarray(ef0, dtype=np.float32).reshape(-1)
                if l2_f:
                    ef = _l2_normalize(ef)
                ef_s = (ef - mean_f) * invstd_f
                return float(np.sqrt(_min_dist2_to_centers(ef_s, c_f)))
            # single camera
            if args.camera_mode == "high":
                e_raw = high_map.get(int(t_end))
            else:
                e_raw = wrist_map.get(int(t_end))
            if e_raw is None:
                raise RuntimeError("Missing cached window embeddings (single camera).")
            c_s = centers_scaled[task]["single"][seg_idx]
            mean_s, invstd_s, l2_s, _logdet_s = scaler_stats[task]["single"][seg_idx]
            ee = np.asarray(e_raw, dtype=np.float32).reshape(-1)
            if l2_s:
                ee = _l2_normalize(ee)
            ee_s = (ee - mean_s) * invstd_s
            return float(np.sqrt(_min_dist2_to_centers(ee_s, c_s)))

        if args.segment_match_value_mode == "hard":
            value_end_sparse = ((match_seg_idx_sparse.astype(np.float32) + 1.0) / float(num_segments)).astype(np.float32)
        else:
            # soft_next: add within-segment fractional progress based on distances (or costs) to (n) and (n+1).
            if str(args.segment_match_score) == "diffusion":
                for j in range(K):
                    n = int(match_seg_idx_sparse[j])
                    c_cur = float(cost_mat[int(j), int(n)])
                    if n >= int(num_segments) - 1:
                        frac = 0.0
                    else:
                        c_next = float(cost_mat[int(j), int(n + 1)])
                        frac = float(c_cur / (c_cur + c_next + 1e-6))
                        frac = float(max(0.0, min(1.0, frac)))
                    value_end_sparse[int(j)] = float((n + 1) + frac) / float(num_segments)
            else:
                for j, t_end in enumerate(window_end_t_sparse.tolist()):
                    n = int(match_seg_idx_sparse[j])
                    d_cur = float(_fused_dist_to_seg(int(t_end), n))
                    if n >= int(num_segments) - 1:
                        frac = 0.0
                    else:
                        d_next = float(_fused_dist_to_seg(int(t_end), n + 1))
                        frac = float(d_cur / (d_cur + d_next + 1e-6))
                        frac = float(max(0.0, min(1.0, frac)))
                    value_end_sparse[j] = float((n + 1) + frac) / float(num_segments)

        xp_v = np.concatenate([np.asarray([0.0], dtype=np.float32), window_end_t_sparse.astype(np.float32)], axis=0)
        fp_v = np.concatenate([np.asarray([0.0], dtype=np.float32), value_end_sparse.astype(np.float32)], axis=0)
        value_full = np.interp(x, xp_v, fp_v, left=0.0, right=float(fp_v[-1])).astype(np.float32)

        if bool(is_success):
            # Success: keep legacy behavior (scaled to end=1).
            if bool(args.enforce_monotonic):
                value_full = _cummax_and_scale(value_full, target_end=1.0)
            else:
                value_full = _scale_to_end(value_full, target_end=1.0)
        else:
            # Failure (non-success): only ensure the curve never gets too close to success.
            # We do NOT scale up; we only scale down if max exceeds the cap.
            if bool(args.enforce_monotonic):
                value_full = np.maximum.accumulate(value_full).astype(np.float32)

            cap = float(min(float(args.segment_match_failure_value_cap), 0.8))
            cap = float(max(0.0, cap))
            cap_eps = float(max(0.0, cap - 1e-6))
            if value_full.size > 0:
                vmax = float(np.max(value_full))
                if vmax > float(cap_eps) and vmax > 1e-9:
                    value_full = (value_full * (cap_eps / vmax)).astype(np.float32)
            value_full = np.clip(value_full, 0.0, cap_eps).astype(np.float32)

        if skip_k > 0 and value_full.size > 0:
            value_full[: min(skip_k, int(value_full.size))] = 0.0

        save_dict: dict[str, Any] = {
            "distance": dist_full.astype(np.float32),
            "value_pred": value_full.astype(np.float32),
            "trim_start": np.asarray(int(s), dtype=np.int32),
            "trim_end": np.asarray(int(e), dtype=np.int32),
            "t_total": np.asarray(int(t_total), dtype=np.int32),
            "task": np.asarray(task),
            "source": np.asarray(source),
            "is_success": np.asarray(int(bool(is_success)), dtype=np.int32),
            "camera_mode": np.asarray(str(args.camera_mode)),
            "camera_fusion": np.asarray(str(args.camera_fusion)),
            "value_method": np.asarray("segment_match50"),
            "segment_centers_path": np.asarray(str(segment_centers_path)),
            "segment_centers_cfg_id": np.asarray(str(seg_centers_cfg_id)),
            "segment_match_num_segments": np.asarray(int(num_segments), dtype=np.int32),
            "segment_match_window_stride": np.asarray(int(win_stride), dtype=np.int32),
            "segment_match_radius": np.asarray(int(radius), dtype=np.int32),
            "segment_match_first_fixed": np.asarray(int(bool(args.segment_match_first_fixed)), dtype=np.int32),
            "segment_match_score": np.asarray(str(args.segment_match_score)),
            "segment_match_diffusion_pca_dim": np.asarray(int(args.segment_match_diffusion_pca_dim), dtype=np.int32),
            "segment_match_diffusion_timesteps": np.asarray(int(args.segment_match_diffusion_timesteps), dtype=np.int32),
            "segment_match_diffusion_hidden_dim": np.asarray(int(args.segment_match_diffusion_hidden_dim), dtype=np.int32),
            "segment_match_diffusion_num_layers": np.asarray(int(args.segment_match_diffusion_num_layers), dtype=np.int32),
            "segment_match_diffusion_lr": np.asarray(float(args.segment_match_diffusion_lr), dtype=np.float32),
            "segment_match_diffusion_batch_size": np.asarray(int(args.segment_match_diffusion_batch_size), dtype=np.int32),
            "segment_match_diffusion_epochs": np.asarray(int(args.segment_match_diffusion_epochs), dtype=np.int32),
            "segment_match_diffusion_seed": np.asarray(int(args.segment_match_diffusion_seed), dtype=np.int32),
            "segment_match_diffusion_t_eval": np.asarray(int(args.segment_match_diffusion_t_eval), dtype=np.int32),
            "segment_match_diffusion_noise_samples": np.asarray(int(args.segment_match_diffusion_noise_samples), dtype=np.int32),
            "segment_match_value_mode": np.asarray(str(args.segment_match_value_mode)),
            "segment_match_path": np.asarray(str(args.segment_match_path)),
            "segment_match_time_prior_success": np.asarray(float(args.segment_match_time_prior_success), dtype=np.float32),
            "segment_match_back_penalty_success": np.asarray(float(args.segment_match_back_penalty_success), dtype=np.float32),
            "segment_match_failure_value_cap": np.asarray(float(args.segment_match_failure_value_cap), dtype=np.float32),
            "enforce_monotonic": np.asarray(int(bool(args.enforce_monotonic)), dtype=np.int32),
            "minmax_skip_prefix_k": np.asarray(int(skip_k), dtype=np.int32),
            "window_end_t_sparse": window_end_t_sparse.astype(np.int32),
            "match_seg_idx_sparse": match_seg_idx_sparse.astype(np.int32),
            "distance_sparse": distance_sparse.astype(np.float32),
            "match_cost_sparse": match_cost_sparse.astype(np.float32),
            "match_cost_eff_sparse": match_cost_eff_sparse.astype(np.float32),
            "value_end_sparse": value_end_sparse.astype(np.float32),
        }
        np.savez_compressed(str(out_npz), **save_dict)
        written += 1
        _append_jsonl(map_path, {"hdf5": str(p), "stage": "segmatch_score", "status": "written", "out_npz": str(out_npz)})

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "out_cache_dir": str(out_dir),
        "value_method": "segment_match50",
        "teleop_hdf5_dir": str(args.teleop_hdf5_dir),
        "dataset_hdf5_dir": None if args.dataset_hdf5_dir is None else str(args.dataset_hdf5_dir),
        "rollouts_hdf5_dir": str(Path(args.dataset_hdf5_dir) if args.dataset_hdf5_dir is not None else Path(args.rollouts_hdf5_dir)),
        "intervention_hdf5_dir": (
            None
            if (args.dataset_hdf5_dir is None and args.intervention_hdf5_dir is None)
            else str(
                Path(args.intervention_hdf5_dir)
                if args.intervention_hdf5_dir is not None
                else (Path(args.dataset_hdf5_dir) / "intervention")
            )
        ),
        "success_mode": str(args.success_mode),
        "shortest_p": float(args.shortest_p),
        "score_sources": str(args.score_sources),
        "camera_mode": str(args.camera_mode),
        "camera_fusion": str(args.camera_fusion),
        "epsilon": float(args.epsilon),
        "dbscan_eps": float(args.dbscan_eps),
        "dbscan_min_samples": int(args.dbscan_min_samples),
        "dbscan_pca_dim": int(args.dbscan_pca_dim),
        "max_episode_len": None if args.max_episode_len is None else int(args.max_episode_len),
        "device": str(device),
        "enable_fp16": bool(args.enable_fp16),
        "embed_batch_size": int(args.embed_batch_size),
        "embed_cache_root": str(embed_cache_root),
        "embed_cache_dir": str(embed_cache_dir),
        "embed_cache_cfg_id": str(embed_cfg_id),
        "embed_cache_cfg": embed_cfg,
        "segment_centers_path": str(segment_centers_path),
        "segment_centers_cfg_id": str(seg_centers_cfg_id),
        "segment_centers_cfg": seg_centers_cfg,
        "num_scoring_episodes": int(len(scoring)),
        "num_success_ref_episodes": int(len(success_ref)),
        "segment_match_num_segments": int(num_segments),
        "segment_match_window_stride": int(win_stride),
        "segment_match_radius": int(radius),
        "segment_match_first_fixed": bool(args.segment_match_first_fixed),
        "segment_match_score": str(args.segment_match_score),
        "segment_match_diffusion_pca_dim": int(args.segment_match_diffusion_pca_dim),
        "segment_match_diffusion_timesteps": int(args.segment_match_diffusion_timesteps),
        "segment_match_diffusion_hidden_dim": int(args.segment_match_diffusion_hidden_dim),
        "segment_match_diffusion_num_layers": int(args.segment_match_diffusion_num_layers),
        "segment_match_diffusion_lr": float(args.segment_match_diffusion_lr),
        "segment_match_diffusion_batch_size": int(args.segment_match_diffusion_batch_size),
        "segment_match_diffusion_epochs": int(args.segment_match_diffusion_epochs),
        "segment_match_diffusion_seed": int(args.segment_match_diffusion_seed),
        "segment_match_diffusion_t_eval": int(args.segment_match_diffusion_t_eval),
        "segment_match_diffusion_noise_samples": int(args.segment_match_diffusion_noise_samples),
        "segment_match_value_mode": str(args.segment_match_value_mode),
        "segment_match_path": str(args.segment_match_path),
        "segment_match_time_prior_success": float(args.segment_match_time_prior_success),
        "segment_match_back_penalty_success": float(args.segment_match_back_penalty_success),
        "segment_match_failure_value_cap": float(args.segment_match_failure_value_cap),
        "enforce_monotonic": bool(args.enforce_monotonic),
        "minmax_skip_prefix_k": int(skip_k),
        "written": int(written),
        "reused": int(reused),
        "skipped": int(skipped),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info(f"Segment-Match50 done: written={written}, reused={reused}, skipped={skipped}. Wrote metadata to {meta_path}")

    # Best-effort: free the huge V-JEPA model weights early.
    if embedder is not None:
        try:
            del embedder
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(str(args.device))
    logging.info(f"device={device}")

    out_dir = Path(args.out_cache_dir)
    if out_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episode_values").mkdir(parents=True, exist_ok=True)

    # Also write logs to a file under out_dir for reproducibility/debugging.
    # This captures DBSCAN fallback warnings and other runtime details.
    try:
        log_path = out_dir / "estimate_vjepa_value.log"
        root = logging.getLogger()
        if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(log_path) for h in root.handlers):
            fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            root.addHandler(fh)
            logging.info(f"Logging to file: {log_path}")
    except Exception as ex:  # noqa: BLE001
        logging.warning(f"Failed to set up file logging under out_dir: {ex}")

    # Public embedding cache: shared by success-reference embedding and prefix embeddings.
    embed_cache_root = Path(args.embed_cache_dir) if args.embed_cache_dir is not None else (out_dir / "embedding_cache")
    embed_cache_root.mkdir(parents=True, exist_ok=True)
    try:
        ckpt_st = args.vjepa_ckpt.stat()
        embed_cfg = {
            "vjepa_ckpt": str(args.vjepa_ckpt.resolve()),
            "vjepa_ckpt_size": int(ckpt_st.st_size),
            "vjepa_ckpt_mtime_ns": int(ckpt_st.st_mtime_ns),
            "vjepa_img_size": int(args.vjepa_img_size),
            "vjepa_num_frames": int(args.vjepa_num_frames),
            "epsilon": float(args.epsilon),
            # Bump this when frame sampling semantics change, to avoid silently reusing
            # old embedding caches computed with a different sampler.
            "frame_sampling_policy": "linspace_backfill_prev_v2",
        }
    except Exception:
        embed_cfg = {
            "vjepa_ckpt": str(args.vjepa_ckpt),
            "vjepa_img_size": int(args.vjepa_img_size),
            "vjepa_num_frames": int(args.vjepa_num_frames),
            "epsilon": float(args.epsilon),
            "frame_sampling_policy": "linspace_backfill_prev_v2",
        }
    embed_cfg_id = _sha1_short(json.dumps(embed_cfg, sort_keys=True))
    embed_cache_dir = embed_cache_root / f"vjepa_embed_{embed_cfg_id}"
    embed_cache_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"embed_cache_dir={embed_cache_dir}")

    map_path = out_dir / "hdf5_to_vjepa_value_map.jsonl"
    centers_path = out_dir / "cluster_centers.npz"
    meta_path = out_dir / "vjepa_value_metadata.json"

    # Resolve merged dataset layout (optional).
    rollouts_dir = Path(args.dataset_hdf5_dir) if args.dataset_hdf5_dir is not None else Path(args.rollouts_hdf5_dir)
    if args.dataset_hdf5_dir is not None:
        intervention_dir = (
            Path(args.intervention_hdf5_dir)
            if args.intervention_hdf5_dir is not None
            else (Path(args.dataset_hdf5_dir) / "intervention")
        )
    else:
        intervention_dir = Path(args.intervention_hdf5_dir) if args.intervention_hdf5_dir is not None else None

    teleop_files = _iter_hdf5_files(args.teleop_hdf5_dir)
    roll_succ, roll_fail = _iter_rollouts_files(rollouts_dir)
    interv_files = _iter_hdf5_files(intervention_dir) if intervention_dir is not None else []
    if not teleop_files:
        logging.warning(f"No teleop HDF5 found under {args.teleop_hdf5_dir}")
    if not roll_succ and not roll_fail:
        logging.warning(f"No rollouts HDF5 found under {rollouts_dir}/{{success,failure}}")
    if args.score_sources in ("dataset", "all") and not interv_files:
        logging.warning(f"No intervention HDF5 found under {intervention_dir or '<unset>'}")

    # 1) Select success reference episodes.
    success_ref: list[Path] = []
    rollouts_success_ref_set: set[str] = set()
    rollouts_scoring_succ: list[Path] = []

    if args.success_mode == "teleop_all":
        success_ref = teleop_files
        if args.max_success_episodes is not None:
            success_ref = success_ref[: int(args.max_success_episodes)]
        logging.info(f"success_mode=teleop_all: success_ref={len(success_ref)} teleop episodes")
    elif args.success_mode == "rollouts_shortest_p":
        # Compute trimmed lengths for rollouts success and pick shortest p%.
        lengths: list[tuple[int, Path]] = []
        for p in tqdm.tqdm(roll_succ, desc="Scan rollouts_success lengths"):
            try:
                with h5py.File(p, "r") as ep:
                    action = np.asarray(ep["/action"][:], dtype=np.float32)
                span = _trim_static_head_tail(action, epsilon=float(args.epsilon))
                if span is None:
                    continue
                s, e = span
                t_total = int(e - s + 1)
                if args.max_episode_len is not None and t_total > int(args.max_episode_len):
                    continue
                lengths.append((t_total, p))
            except Exception as e:  # noqa: BLE001
                _append_jsonl(map_path, {"hdf5": str(p), "stage": "scan_success_len", "status": "error", "error": str(e)})

        if not lengths:
            raise RuntimeError("No valid rollouts_success episodes found to build success references.")
        lengths.sort(key=lambda x: x[0])
        k = int(max(1, math.ceil(float(args.shortest_p) * float(len(lengths)))))
        chosen = [p for _, p in lengths[:k]]
        success_ref = chosen
        rollouts_success_ref_set = {p.name for p in chosen}
        # Remaining rollouts_success are scored.
        rollouts_scoring_succ = [p for p in roll_succ if p.name not in rollouts_success_ref_set]
        if args.max_success_episodes is not None:
            success_ref = success_ref[: int(args.max_success_episodes)]
            rollouts_success_ref_set = {p.name for p in success_ref}
            rollouts_scoring_succ = [p for p in roll_succ if p.name not in rollouts_success_ref_set]
        logging.info(
            f"success_mode=rollouts_shortest_p: success_ref={len(success_ref)}/{len(roll_succ)} "
            f"(p={args.shortest_p})"
        )
    else:
        raise ValueError(f"Unknown success_mode: {args.success_mode}")

    # 2) Prepare scoring episode list.
    scoring: list[tuple[Path, str, bool]] = []
    if args.score_sources in ("rollouts", "dataset", "all"):
        # failure are always scored
        roll_fail_scoring = roll_fail
        if args.max_score_rollouts_failure_episodes is not None:
            roll_fail_scoring = roll_fail_scoring[: int(args.max_score_rollouts_failure_episodes)]
        for p in roll_fail_scoring:
            scoring.append((p, "rollouts_failure", False))
        # success may be scored (unless they are part of success_ref in rollouts_shortest_p mode)
        if args.success_mode == "rollouts_shortest_p":
            succ_to_score = rollouts_scoring_succ
        else:
            succ_to_score = roll_succ
        if args.max_score_rollouts_success_episodes is not None:
            succ_to_score = succ_to_score[: int(args.max_score_rollouts_success_episodes)]
        for p in succ_to_score:
            # trust /success if present; else directory naming
            is_success = True
            try:
                with h5py.File(p, "r") as ep:
                    if "/success" in ep:
                        is_success = bool(int(np.array(ep["/success"][()]).item()) != 0)
            except Exception:
                pass
            scoring.append((p, "rollouts_success", bool(is_success)))
    if args.score_sources in ("teleop", "all"):
        for p in teleop_files:
            scoring.append((p, "teleop", True))
    if args.score_sources in ("dataset", "all"):
        for p in interv_files:
            # default: treat intervention as success unless /success exists
            is_success = True
            try:
                with h5py.File(p, "r") as ep:
                    if "/success" in ep:
                        v = np.asarray(ep["/success"][()])
                        if v.ndim == 0:
                            is_success = bool(int(v.item()) != 0)
                        else:
                            vv = v.reshape(-1)
                            is_success = bool(int(vv[-1].item()) != 0) if vv.size > 0 else True
            except Exception:
                pass
            scoring.append((p, "intervention", bool(is_success)))

    if not scoring:
        raise RuntimeError("No scoring episodes selected. Check --score-sources.")
    if args.max_score_episodes is not None:
        scoring = scoring[: int(args.max_score_episodes)]
    logging.info(f"scoring episodes: {len(scoring)}")

    # Build a signature for success_ref selection so cluster centers are not accidentally reused
    # across different success reference sets / camera settings / DBSCAN params.
    try:
        success_ref_ids = sorted(_hdf5_sig_id(p) for p in success_ref)
    except Exception:
        success_ref_ids = sorted(_sha1_short(str(p)) for p in success_ref)
    success_ref_sig = _sha1_short("|".join(success_ref_ids)) if success_ref_ids else "empty"

    # Branch by value method early to avoid mixing caches (cluster_centers/warp) with segment_match50.
    if args.value_method == "segment_match50":
        _run_segment_match50(
            args=args,
            device=device,
            out_dir=out_dir,
            embed_cache_root=embed_cache_root,
            embed_cache_dir=embed_cache_dir,
            embed_cfg=embed_cfg,
            embed_cfg_id=embed_cfg_id,
            map_path=map_path,
            meta_path=meta_path,
            success_ref=success_ref,
            scoring=scoring,
            success_ref_sig=success_ref_sig,
        )
        return

    if args.value_method != "progress_warp":
        raise ValueError(f"Unknown value_method: {args.value_method}")

    centers_cfg = {
        "success_mode": str(args.success_mode),
        "shortest_p": float(args.shortest_p),
        "max_success_episodes": None if args.max_success_episodes is None else int(args.max_success_episodes),
        "success_ref_sig": str(success_ref_sig),
        "success_ref_count": int(len(success_ref)),
        "camera_mode": str(args.camera_mode),
        "camera_fusion": str(args.camera_fusion),
        "dbscan_eps": float(args.dbscan_eps),
        "dbscan_min_samples": int(args.dbscan_min_samples),
        "dbscan_pca_dim": int(args.dbscan_pca_dim),
    }
    centers_cfg_id = _sha1_short(json.dumps(centers_cfg, sort_keys=True))

    # 3) Build/load success cluster centers.
    # We store a dict with keys:
    # - schema_version
    # - camera_mode/camera_fusion
    # - centers: dict[str_task][str_key] -> np.ndarray(K, D)
    #   where str_key is: 'high'/'wrist' for dist fusion; 'fused' for emb fusion; or 'single' for single-camera.
    centers_blob: dict[str, dict[str, np.ndarray]] | None = None
    centers_info: dict[str, dict[str, Any]] | None = None
    scaler_blob: dict[str, dict[str, Any]] | None = None

    need_build_centers = True
    if centers_path.exists() and not args.overwrite:
        blob = _load_npz_dict(centers_path)
        if blob is not None and "centers" in blob:
            centers_blob = blob["centers"].item()
            centers_info = blob["info"].item() if "info" in blob else {}
            scaler_blob = blob["scaler"].item() if "scaler" in blob else None
            try:
                schema_version = int(np.asarray(blob.get("schema_version", 1)).item())
            except Exception:
                schema_version = 1
            stored_embed_cfg_id = None
            stored_centers_cfg_id = None
            try:
                if "embed_cfg_id" in blob:
                    stored_embed_cfg_id = str(_decode_bytes(blob["embed_cfg_id"]))
                if "centers_cfg_id" in blob:
                    stored_centers_cfg_id = str(_decode_bytes(blob["centers_cfg_id"]))
            except Exception:
                stored_embed_cfg_id = None
                stored_centers_cfg_id = None

            if schema_version < 3 or scaler_blob is None or stored_centers_cfg_id is None:
                logging.warning(
                    f"Cluster centers cache {centers_path} is old schema (schema_version={schema_version}) or missing scaler/cfg; rebuilding."
                )
                need_build_centers = True
            elif stored_embed_cfg_id is not None and stored_embed_cfg_id != str(embed_cfg_id):
                logging.warning(
                    f"Cluster centers cache {centers_path} embed_cfg_id mismatch (stored={stored_embed_cfg_id}, current={embed_cfg_id}); rebuilding."
                )
                need_build_centers = True
            elif stored_centers_cfg_id != str(centers_cfg_id):
                logging.warning(
                    f"Cluster centers cache {centers_path} centers_cfg_id mismatch (stored={stored_centers_cfg_id}, current={centers_cfg_id}); rebuilding."
                )
                need_build_centers = True
            else:
                need_build_centers = False
                logging.info(f"Loaded cluster centers from {centers_path}")

    if need_build_centers:
        logging.info("Building success cluster centers (this may take a while on first run)...")
        embedder = VJepaEmbedder(
            ckpt_path=args.vjepa_ckpt,
            img_size=int(args.vjepa_img_size),
            num_frames=int(args.vjepa_num_frames),
            device=device,
            enable_fp16=bool(args.enable_fp16),
            use_sdpa=(device.type == "cuda"),
        )

        # Collect success embeddings per task (with a shared embedding cache).
        need_high = args.camera_mode in ("high", "both")
        need_wrist = args.camera_mode in ("wrist", "both")

        succ_records: list[dict[str, Any]] = []
        pending: dict[str, dict[str, Any]] = {}  # ep_key -> record
        batch_videos: list[torch.Tensor] = []
        batch_tags: list[tuple[str, str]] = []  # (ep_key, cam_key: "high"|"wrist")

        succ_cache_hit = 0
        succ_cache_partial = 0
        succ_cache_miss = 0

        def _is_ready(r: dict[str, Any]) -> bool:
            if need_high and r.get("emb_high") is None:
                return False
            if need_wrist and r.get("emb_wrist") is None:
                return False
            return True

        def _finalize_ready() -> None:
            done = [k for k, r in pending.items() if _is_ready(r)]
            for k in done:
                succ_records.append(pending.pop(k))

        def flush_batch() -> None:
            nonlocal batch_videos, batch_tags
            if not batch_videos:
                return
            embs = embedder.embed_batch(batch_videos)

            # Aggregate updates per episode so we only write each cache file once per flush.
            updates: dict[str, dict[str, dict[int, np.ndarray]]] = {}
            for emb, (ep_key, cam_key) in zip(embs, batch_tags, strict=True):
                rec = pending.get(ep_key)
                if rec is None:
                    continue
                t_end = int(rec["t_end"])
                emb32 = np.asarray(emb, dtype=np.float32).reshape(-1)
                if cam_key == "high":
                    rec["emb_high"] = emb32
                    updates.setdefault(ep_key, {}).setdefault("high", {})[t_end] = emb32
                elif cam_key == "wrist":
                    rec["emb_wrist"] = emb32
                    updates.setdefault(ep_key, {}).setdefault("wrist", {})[t_end] = emb32
                else:
                    raise ValueError(cam_key)

            for ep_key, per_cam in updates.items():
                rec = pending.get(ep_key)
                if rec is None:
                    continue
                _save_episode_embed_cache(
                    rec["embed_cache_path"],
                    hdf5_path=rec["hdf5_path"],
                    trim_start=int(rec["trim_start"]),
                    trim_end=int(rec["trim_end"]),
                    t_total=int(rec["t_total"]),
                    vjepa_ckpt=args.vjepa_ckpt,
                    vjepa_img_size=int(args.vjepa_img_size),
                    vjepa_num_frames=int(args.vjepa_num_frames),
                    new_high=per_cam.get("high"),
                    new_wrist=per_cam.get("wrist"),
                )

            batch_videos = []
            batch_tags = []
            _finalize_ready()

        for p in tqdm.tqdm(success_ref, desc="Embed success references"):
            try:
                with h5py.File(p, "r") as ep:
                    sig = _get_trim_signal(ep, trim_signal=str(args.trim_signal))
                    span = _trim_static_head_tail(sig, epsilon=float(args.epsilon))
                    if span is None:
                        _append_jsonl(
                            map_path,
                            {"hdf5": str(p), "stage": "success_ref", "status": "skipped", "reason": "trim_none"},
                        )
                        continue
                    s, e = span
                    t_total = int(e - s + 1)
                    if args.max_episode_len is not None and t_total > int(args.max_episode_len):
                        _append_jsonl(
                            map_path,
                            {"hdf5": str(p), "stage": "success_ref", "status": "skipped", "reason": "too_long", "t_total": t_total},
                        )
                        continue
                    task = _task_from_hdf5(ep, fallback_name=p.name)
                    t_end = int(t_total - 1)

                    embed_cache_path = _episode_embed_cache_path(embed_cache_dir, p)
                    high_map, wrist_map = _load_episode_embed_maps(
                        embed_cache_path,
                        expected_trim_start=int(s),
                        expected_trim_end=int(e),
                        expected_t_total=int(t_total),
                    )
                    emb_high = high_map.get(t_end) if need_high else None
                    emb_wrist = wrist_map.get(t_end) if need_wrist else None

                    missing_high = bool(need_high and emb_high is None)
                    missing_wrist = bool(need_wrist and emb_wrist is None)

                    if not missing_high and not missing_wrist:
                        succ_cache_hit += 1
                        succ_records.append(
                            {
                                "hdf5_path": p,
                                "task": task,
                                "trim_start": int(s),
                                "trim_end": int(e),
                                "t_total": int(t_total),
                                "t_end": int(t_end),
                                "emb_high": emb_high,
                                "emb_wrist": emb_wrist,
                            }
                        )
                        continue

                    if (emb_high is not None) or (emb_wrist is not None):
                        succ_cache_partial += 1
                    else:
                        succ_cache_miss += 1

                    ep_key = str(p)
                    pending[ep_key] = {
                        "hdf5_path": p,
                        "embed_cache_path": embed_cache_path,
                        "task": task,
                        "trim_start": int(s),
                        "trim_end": int(e),
                        "t_total": int(t_total),
                        "t_end": int(t_end),
                        "emb_high": emb_high,
                        "emb_wrist": emb_wrist,
                    }

                    # Sample indices for the full clip (relative to trimmed span).
                    rel = _prefix_sample_indices(t_total, n=int(args.vjepa_num_frames))
                    abs_idx = (int(s) + rel).astype(np.int64)

                    if missing_high:
                        frames_h = _load_images_sample(ep, "cam_high", abs_idx)
                        batch_videos.append(torch.from_numpy(frames_h))
                        batch_tags.append((ep_key, "high"))
                    if missing_wrist:
                        frames_w = _load_images_sample(ep, "cam_left_wrist", abs_idx)
                        batch_videos.append(torch.from_numpy(frames_w))
                        batch_tags.append((ep_key, "wrist"))
            except Exception as e:  # noqa: BLE001
                _append_jsonl(map_path, {"hdf5": str(p), "stage": "success_ref", "status": "error", "error": str(e)})
                continue

            if len(batch_videos) >= int(max(1, args.embed_batch_size)):
                flush_batch()

        flush_batch()
        _finalize_ready()

        if pending:
            logging.warning(f"Some success_ref episodes did not finish embedding (skipping): {len(pending)}")
            pending.clear()

        logging.info(
            f"success_ref embedding cache: hit={succ_cache_hit} partial={succ_cache_partial} miss={succ_cache_miss} "
            f"(total={len(success_ref)})"
        )

        # Build per-task embedding lists for clustering.
        succ_embs: dict[str, dict[str, list[np.ndarray]]] = {}
        for r in succ_records:
            task = str(r["task"])
            if args.camera_mode == "high":
                succ_embs.setdefault(task, {}).setdefault("single", []).append(np.asarray(r["emb_high"], dtype=np.float32))
            elif args.camera_mode == "wrist":
                succ_embs.setdefault(task, {}).setdefault("single", []).append(np.asarray(r["emb_wrist"], dtype=np.float32))
            else:
                eh = np.asarray(r["emb_high"], dtype=np.float32)
                ew = np.asarray(r["emb_wrist"], dtype=np.float32)
                if args.camera_fusion in ("dist_min", "dist_mean"):
                    succ_embs.setdefault(task, {}).setdefault("high", []).append(eh)
                    succ_embs.setdefault(task, {}).setdefault("wrist", []).append(ew)
                elif args.camera_fusion == "emb_mean":
                    succ_embs.setdefault(task, {}).setdefault("fused", []).append(0.5 * (eh + ew))
                elif args.camera_fusion == "emb_concat":
                    succ_embs.setdefault(task, {}).setdefault("fused", []).append(np.concatenate([eh, ew], axis=0))
                else:
                    raise ValueError(f"Unknown camera_fusion: {args.camera_fusion}")

        # Cluster per task.
        centers_blob = {}
        scaler_blob = {}
        centers_info = {}
        for task, per_key in succ_embs.items():
            centers_blob[task] = {}
            scaler_blob[task] = {}
            centers_info[task] = {}
            for key, emb_list in per_key.items():
                embs = np.stack([np.asarray(x, dtype=np.float32) for x in emb_list], axis=0)
                centers, scaler, info = _try_cluster_centers(
                    embs,
                    dbscan_eps=float(args.dbscan_eps),
                    dbscan_min_samples=int(args.dbscan_min_samples),
                    dbscan_pca_dim=int(args.dbscan_pca_dim),
                )
                centers_blob[task][key] = centers
                scaler_blob[task][key] = scaler
                centers_info[task][key] = {**info}

        np.savez_compressed(
            str(centers_path),
            schema_version=np.asarray(3, dtype=np.int32),
            embed_cfg_id=np.asarray(str(embed_cfg_id)),
            centers_cfg_id=np.asarray(str(centers_cfg_id)),
            centers_cfg=np.asarray(json.dumps(centers_cfg, sort_keys=True)),
            success_ref_sig=np.asarray(str(success_ref_sig)),
            success_ref_ids=np.asarray(success_ref_ids),
            centers=centers_blob,
            scaler=scaler_blob,
            info=centers_info,
        )
        logging.info(f"Saved cluster centers to {centers_path}")
    else:
        assert centers_blob is not None and centers_info is not None and scaler_blob is not None

    assert centers_blob is not None and centers_info is not None and scaler_blob is not None

    # Save a human-readable clustering report under out_dir.
    try:
        report = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "value_method": "progress_warp",
            "cluster_centers_path": str(centers_path),
            "centers_cfg_id": str(centers_cfg_id),
            "embed_cfg_id": str(embed_cfg_id),
            "success_ref_sig": str(success_ref_sig),
            "success_ref_count": int(len(success_ref)),
            "camera_mode": str(args.camera_mode),
            "camera_fusion": str(args.camera_fusion),
            "trim_signal": str(args.trim_signal),
            "epsilon": float(args.epsilon),
            "dbscan_eps": float(args.dbscan_eps),
            "dbscan_min_samples": int(args.dbscan_min_samples),
            "dbscan_pca_dim": int(args.dbscan_pca_dim),
            "tasks": {},
            "summary": {"groups_total": 0, "groups_fallback_mean": 0, "groups_all_noise": 0},
        }
        g_total = g_fb = g_all_noise = 0
        for task, per_key in centers_info.items():
            report["tasks"][task] = {}
            for key, inf in per_key.items():
                s = _summarize_dbscan_info(dict(inf) if isinstance(inf, dict) else {})
                report["tasks"][task][key] = s
                g_total += 1
                if str(s.get("method")) == "dbscan_fallback_mean":
                    g_fb += 1
                if s.get("noise_frac") == 1.0:
                    g_all_noise += 1
        report["summary"]["groups_total"] = int(g_total)
        report["summary"]["groups_fallback_mean"] = int(g_fb)
        report["summary"]["groups_all_noise"] = int(g_all_noise)
        _write_json(out_dir / "cluster_centers_report.json", report)
        logging.info(f"Wrote clustering report to {out_dir / 'cluster_centers_report.json'}")
    except Exception as ex:  # noqa: BLE001
        logging.warning(f"Failed to write cluster_centers_report.json: {ex}")

    # Precompute standardized centers + scaler stats (mean, inv_std, l2_normalize) for fast distance computation.
    centers_scaled_blob: dict[str, dict[str, np.ndarray]] = {}
    scaler_stats: dict[str, dict[str, tuple[np.ndarray, np.ndarray, bool]]] = {}
    n_l2 = 0
    n_total = 0
    for task, per_key in centers_blob.items():
        centers_scaled_blob[task] = {}
        scaler_stats[task] = {}
        for key, centers in per_key.items():
            sc = scaler_blob[task][key]
            mean = np.asarray(sc["mean"], dtype=np.float32).reshape(-1)
            std = np.asarray(sc["std"], dtype=np.float32).reshape(-1)
            inv_std = (1.0 / np.maximum(std, 1e-6)).astype(np.float32)
            use_l2 = bool(sc.get("l2_normalize", False))
            centers_arr = np.asarray(centers, dtype=np.float32)
            centers_scaled_blob[task][key] = (centers_arr - mean) * inv_std
            scaler_stats[task][key] = (mean, inv_std, use_l2)
            n_total += 1
            n_l2 += int(use_l2)
            try:
                info = centers_info.get(task, {}).get(key, {})
                logging.info(
                    f"[centers] task={task} key={key} n_success={info.get('n_success', info.get('n_success', 'NA'))} "
                    f"use_l2={use_l2} norm_cv={info.get('norm_cv', 'NA')} ratio_p95_p5={info.get('norm_ratio_p95_p5', 'NA')}"
                )
            except Exception:
                pass
    logging.info(f"[centers] L2-normalize enabled for {n_l2}/{n_total} (task,key) groups.")

    # Build embedder (after centers ready).
    embedder = VJepaEmbedder(
        ckpt_path=args.vjepa_ckpt,
        img_size=int(args.vjepa_img_size),
        num_frames=int(args.vjepa_num_frames),
        device=device,
        enable_fp16=bool(args.enable_fp16),
        use_sdpa=(device.type == "cuda"),
    )

    # 4) Stage 1: compute distances per episode (if missing) + gather per-task min/max for normalization.
    dist_minmax: dict[str, dict[str, float]] = {}  # task -> {"min":..., "max":...} (maybe from failures only)
    dist_minmax_all: dict[str, dict[str, float]] = {}  # task -> min/max over all scored (fallback)
    skip_k = int(max(0, args.minmax_skip_prefix_k))
    if skip_k > 0:
        logging.info(f"value_pred prefix: forcing first {skip_k} frames to 0 (does not affect baseline d0 at t=0)")

    def update_minmax(store: dict[str, dict[str, float]], task: str, d: np.ndarray) -> None:
        if d.size == 0:
            return
        mn = float(np.min(d))
        mx = float(np.max(d))
        cur = store.get(task)
        if cur is None:
            store[task] = {"min": mn, "max": mx}
        else:
            store[task] = {"min": min(cur["min"], mn), "max": max(cur["max"], mx)}

    written = 0
    reused = 0
    skipped = 0
    stage1_embed_hit = 0
    stage1_embed_partial = 0
    stage1_embed_miss = 0

    for p, source, is_success in tqdm.tqdm(scoring, desc="Stage1: distance"):
        out_npz = out_dir / "episode_values" / f"{p.name}.npz"

        if out_npz.exists():
            try:
                z = np.load(str(out_npz), allow_pickle=True)
                # Cache compatibility checks: if this file was produced by an older version
                # (raw-space distance) or different embedding config / stride, recompute.
                dist_space = _decode_bytes(z["distance_space"]) if "distance_space" in z else "raw"
                stored_cfg_id = _decode_bytes(z["embed_cfg_id"]) if "embed_cfg_id" in z else None
                stored_centers_cfg_id = _decode_bytes(z["centers_cfg_id"]) if "centers_cfg_id" in z else None
                stored_stride = int(np.asarray(z["timestep_stride"]).item()) if "timestep_stride" in z else None
                stored_cam_mode = _decode_bytes(z["camera_mode"]) if "camera_mode" in z else None
                stored_cam_fusion = _decode_bytes(z["camera_fusion"]) if "camera_fusion" in z else None
                if dist_space != "standardized":
                    raise RuntimeError(f"distance_space={dist_space} (need standardized)")
                if stored_cfg_id is not None and str(stored_cfg_id) != str(embed_cfg_id):
                    raise RuntimeError(f"embed_cfg_id mismatch (stored={stored_cfg_id}, current={embed_cfg_id})")
                if stored_centers_cfg_id is not None and str(stored_centers_cfg_id) != str(centers_cfg_id):
                    raise RuntimeError(
                        f"centers_cfg_id mismatch (stored={stored_centers_cfg_id}, current={centers_cfg_id})"
                    )
                if stored_stride is not None and int(stored_stride) != int(max(1, args.timestep_stride)):
                    raise RuntimeError(f"timestep_stride mismatch (stored={stored_stride}, current={args.timestep_stride})")
                if stored_cam_mode is not None and str(stored_cam_mode) != str(args.camera_mode):
                    raise RuntimeError(f"camera_mode mismatch (stored={stored_cam_mode}, current={args.camera_mode})")
                if stored_cam_fusion is not None and str(stored_cam_fusion) != str(args.camera_fusion):
                    raise RuntimeError(f"camera_fusion mismatch (stored={stored_cam_fusion}, current={args.camera_fusion})")

                dist = np.asarray(z["distance"], dtype=np.float32)
                task = _decode_bytes(z["task"]) if "task" in z else p.stem
                dist_stats = dist[skip_k:] if dist.size > skip_k else dist[:0]
                update_minmax(dist_minmax_all, task, dist_stats)
                if args.norm_scope == "all_scored" or (not bool(is_success)):
                    update_minmax(dist_minmax, task, dist_stats)
                reused += 1
                continue
            except Exception:
                # fallthrough to recompute
                pass

        # Load/compute prefix embeddings (shared cache) first, then compute sparse distances.
        try:
            with h5py.File(p, "r") as ep:
                sig = _get_trim_signal(ep, trim_signal=str(args.trim_signal))
                span = _trim_static_head_tail(sig, epsilon=float(args.epsilon))
                if span is None:
                    skipped += 1
                    _append_jsonl(map_path, {"hdf5": str(p), "stage": "distance", "status": "skipped", "reason": "trim_none"})
                    continue
                s, e = span
                t_total = int(e - s + 1)
                if args.max_episode_len is not None and t_total > int(args.max_episode_len):
                    skipped += 1
                    _append_jsonl(
                        map_path,
                        {"hdf5": str(p), "stage": "distance", "status": "skipped", "reason": "too_long", "t_total": t_total},
                    )
                    continue

                task = _task_from_hdf5(ep, fallback_name=p.name)

                # Build time indices to compute (sparse), always include last frame.
                stride = int(max(1, args.timestep_stride))
                t_sparse = list(range(0, t_total, stride))
                if (t_total - 1) not in t_sparse:
                    t_sparse.append(t_total - 1)
                t_sparse = sorted(set(t_sparse))

                need_high = args.camera_mode in ("high", "both")
                need_wrist = args.camera_mode in ("wrist", "both")

                embed_cache_path = _episode_embed_cache_path(embed_cache_dir, p)
                high_map, wrist_map = _load_episode_embed_maps(
                    embed_cache_path,
                    expected_trim_start=int(s),
                    expected_trim_end=int(e),
                    expected_t_total=int(t_total),
                )

                missing_high = [int(t) for t in t_sparse if need_high and int(t) not in high_map]
                missing_wrist = [int(t) for t in t_sparse if need_wrist and int(t) not in wrist_map]

                # Cache stats (embeddings) for visibility, especially for dual-camera mode.
                if (not missing_high) and (not missing_wrist):
                    stage1_embed_hit += 1
                else:
                    miss_high_all = (len(missing_high) == len(t_sparse)) if need_high else True
                    miss_wrist_all = (len(missing_wrist) == len(t_sparse)) if need_wrist else True
                    if miss_high_all and miss_wrist_all:
                        stage1_embed_miss += 1
                    else:
                        stage1_embed_partial += 1

                imgs_high = None
                imgs_wrist = None
                # Only load frames if we actually need to compute missing embeddings.
                if missing_high:
                    imgs_high = _load_images_full(ep, "cam_high", s=s, e=e)
                if missing_wrist:
                    imgs_wrist = _load_images_full(ep, "cam_left_wrist", s=s, e=e)
        except Exception as ex:  # noqa: BLE001
            _append_jsonl(map_path, {"hdf5": str(p), "stage": "distance", "status": "error", "error": str(ex)})
            continue

        # Get centers for this task.
        if task not in centers_blob:
            raise RuntimeError(f"Task {task!r} missing in cluster centers. Did success_ref contain this task?")

        # Compute missing prefix embeddings and update the shared cache.
        new_high: dict[int, np.ndarray] = {}
        new_wrist: dict[int, np.ndarray] = {}
        if missing_high or missing_wrist:
            missing_high_set = set(missing_high)
            missing_wrist_set = set(missing_wrist)
            bs = int(max(1, args.embed_batch_size))
            for i0 in range(0, len(t_sparse), bs):
                i1 = min(len(t_sparse), i0 + bs)
                chunk_ts = t_sparse[i0:i1]
                videos: list[torch.Tensor] = []
                tags: list[tuple[int, str]] = []  # (timestep, cam_key)
                for t in chunk_ts:
                    prefix_len = int(t + 1)
                    rel_idx = _prefix_sample_indices(prefix_len, n=int(args.vjepa_num_frames))
                    if int(t) in missing_high_set:
                        assert imgs_high is not None
                        frames = imgs_high[rel_idx]
                        videos.append(torch.from_numpy(frames))
                        tags.append((int(t), "high"))
                    if int(t) in missing_wrist_set:
                        assert imgs_wrist is not None
                        frames = imgs_wrist[rel_idx]
                        videos.append(torch.from_numpy(frames))
                        tags.append((int(t), "wrist"))
                if not videos:
                    continue
                embs = embedder.embed_batch(videos)
                for emb, (t, cam_key) in zip(embs, tags, strict=True):
                    emb32 = np.asarray(emb, dtype=np.float32).reshape(-1)
                    if cam_key == "high":
                        high_map[int(t)] = emb32
                        new_high[int(t)] = emb32
                    elif cam_key == "wrist":
                        wrist_map[int(t)] = emb32
                        new_wrist[int(t)] = emb32
                    else:
                        raise ValueError(cam_key)

            if new_high or new_wrist:
                _save_episode_embed_cache(
                    embed_cache_path,
                    hdf5_path=p,
                    trim_start=int(s),
                    trim_end=int(e),
                    t_total=int(t_total),
                    vjepa_ckpt=args.vjepa_ckpt,
                    vjepa_img_size=int(args.vjepa_img_size),
                    vjepa_num_frames=int(args.vjepa_num_frames),
                    new_high=(new_high if new_high else None),
                    new_wrist=(new_wrist if new_wrist else None),
                )

        # Compute sparse distances from embeddings.
        dist_high_sparse: np.ndarray | None = None
        dist_wrist_sparse: np.ndarray | None = None
        dist_single_or_fused_sparse: np.ndarray | None = None

        if args.camera_mode == "both" and args.camera_fusion in ("dist_min", "dist_mean"):
            dist_high_sparse = np.zeros((len(t_sparse),), dtype=np.float32)
            dist_wrist_sparse = np.zeros((len(t_sparse),), dtype=np.float32)
            c_high = centers_scaled_blob[task]["high"]
            c_wrist = centers_scaled_blob[task]["wrist"]
            mean_h, invstd_h, l2_h = scaler_stats[task]["high"]
            mean_w, invstd_w, l2_w = scaler_stats[task]["wrist"]
            for j, t in enumerate(t_sparse):
                eh_raw = high_map.get(int(t))
                ew_raw = wrist_map.get(int(t))
                if eh_raw is None or ew_raw is None:
                    raise RuntimeError(f"Missing cached embeddings for t={t} (high or wrist).")
                eh = np.asarray(eh_raw, dtype=np.float32).reshape(-1)
                ew = np.asarray(ew_raw, dtype=np.float32).reshape(-1)
                if l2_h:
                    eh = _l2_normalize(eh)
                if l2_w:
                    ew = _l2_normalize(ew)
                eh_s = (eh - mean_h) * invstd_h
                ew_s = (ew - mean_w) * invstd_w
                dist_high_sparse[j] = _min_dist_to_centers(eh_s, c_high)
                dist_wrist_sparse[j] = _min_dist_to_centers(ew_s, c_wrist)
        elif args.camera_mode == "both" and args.camera_fusion in ("emb_mean", "emb_concat"):
            c_fused = centers_scaled_blob[task]["fused"]
            mean_f, invstd_f, l2_f = scaler_stats[task]["fused"]
            dist_single_or_fused_sparse = np.zeros((len(t_sparse),), dtype=np.float32)
            for j, t in enumerate(t_sparse):
                eh_raw = high_map.get(int(t))
                ew_raw = wrist_map.get(int(t))
                if eh_raw is None or ew_raw is None:
                    raise RuntimeError(f"Missing cached embeddings for t={t} (high or wrist).")
                if args.camera_fusion == "emb_mean":
                    ef = 0.5 * (np.asarray(eh_raw, dtype=np.float32) + np.asarray(ew_raw, dtype=np.float32))
                else:
                    ef = np.concatenate([np.asarray(eh_raw, dtype=np.float32), np.asarray(ew_raw, dtype=np.float32)], axis=0)
                ef = np.asarray(ef, dtype=np.float32).reshape(-1)
                if l2_f:
                    ef = _l2_normalize(ef)
                ef_s = (ef - mean_f) * invstd_f
                dist_single_or_fused_sparse[j] = _min_dist_to_centers(ef_s, c_fused)
        else:
            # single camera
            c_single = centers_scaled_blob[task]["single"]
            mean_s, invstd_s, l2_s = scaler_stats[task]["single"]
            dist_single_or_fused_sparse = np.zeros((len(t_sparse),), dtype=np.float32)
            if args.camera_mode == "high":
                for j, t in enumerate(t_sparse):
                    eh_raw = high_map.get(int(t))
                    if eh_raw is None:
                        raise RuntimeError(f"Missing cached embeddings for t={t} (high).")
                    eh = np.asarray(eh_raw, dtype=np.float32).reshape(-1)
                    if l2_s:
                        eh = _l2_normalize(eh)
                    eh_s = (eh - mean_s) * invstd_s
                    dist_single_or_fused_sparse[j] = _min_dist_to_centers(eh_s, c_single)
            elif args.camera_mode == "wrist":
                for j, t in enumerate(t_sparse):
                    ew_raw = wrist_map.get(int(t))
                    if ew_raw is None:
                        raise RuntimeError(f"Missing cached embeddings for t={t} (wrist).")
                    ew = np.asarray(ew_raw, dtype=np.float32).reshape(-1)
                    if l2_s:
                        ew = _l2_normalize(ew)
                    ew_s = (ew - mean_s) * invstd_s
                    dist_single_or_fused_sparse[j] = _min_dist_to_centers(ew_s, c_single)
            else:
                raise ValueError(f"Unexpected camera_mode for single-camera branch: {args.camera_mode}")

        # Interpolate sparse distances to full length.
        x = np.arange(t_total, dtype=np.float32)
        xp = np.asarray(t_sparse, dtype=np.float32)
        if args.camera_mode == "both" and args.camera_fusion in ("dist_min", "dist_mean"):
            assert dist_high_sparse is not None and dist_wrist_sparse is not None
            dist_high = np.interp(x, xp, dist_high_sparse).astype(np.float32)
            dist_wrist = np.interp(x, xp, dist_wrist_sparse).astype(np.float32)
            if args.camera_fusion == "dist_min":
                dist = np.minimum(dist_high, dist_wrist)
            else:
                dist = 0.5 * (dist_high + dist_wrist)
        else:
            assert dist_single_or_fused_sparse is not None
            dist = np.interp(x, xp, dist_single_or_fused_sparse.astype(np.float32)).astype(np.float32)
            dist_high = None
            dist_wrist = None

        # Update min/max stores.
        dist_stats = dist[skip_k:] if dist.size > skip_k else dist[:0]
        update_minmax(dist_minmax_all, task, dist_stats)
        if args.norm_scope == "all_scored" or (not bool(is_success)):
            update_minmax(dist_minmax, task, dist_stats)

        # Save (distance only for now; value_pred computed in Stage2).
        save_dict: dict[str, Any] = {
            "distance": dist.astype(np.float32),
            "trim_start": np.asarray(int(s), dtype=np.int32),
            "trim_end": np.asarray(int(e), dtype=np.int32),
            "t_total": np.asarray(int(t_total), dtype=np.int32),
            "task": np.asarray(task),
            "source": np.asarray(source),
            "is_success": np.asarray(int(bool(is_success)), dtype=np.int32),
            "camera_mode": np.asarray(str(args.camera_mode)),
            "camera_fusion": np.asarray(str(args.camera_fusion)),
            "timestep_stride": np.asarray(int(stride), dtype=np.int32),
            "distance_space": np.asarray("standardized"),
            "embed_cfg_id": np.asarray(str(embed_cfg_id)),
            "centers_cfg_id": np.asarray(str(centers_cfg_id)),
            "minmax_skip_prefix_k": np.asarray(int(skip_k), dtype=np.int32),
        }
        if dist_high is not None and dist_wrist is not None:
            save_dict["distance_high"] = dist_high.astype(np.float32)
            save_dict["distance_wrist"] = dist_wrist.astype(np.float32)
        np.savez_compressed(str(out_npz), **save_dict)
        written += 1
        _append_jsonl(map_path, {"hdf5": str(p), "stage": "distance", "status": "written", "out_npz": str(out_npz)})

    logging.info(
        f"Stage1 done: written={written}, reused={reused}, skipped={skipped}; "
        f"embed_cache(hit={stage1_embed_hit}, partial={stage1_embed_partial}, miss={stage1_embed_miss})"
    )

    # 5) Resolve normalization min/max per task (only used for legacy sigmoid mapping).
    # If fail_only and a task has no failures, fallback to all_scored stats.
    norm_stats: dict[str, dict[str, float]] = {}
    if args.value_mapping == "sigmoid":
        for task, mm_all in dist_minmax_all.items():
            if args.norm_scope == "all_scored":
                norm_stats[task] = {"min": float(mm_all["min"]), "max": float(mm_all["max"]), "scope": "all_scored"}
            else:
                mm_fail = dist_minmax.get(task)
                if mm_fail is None:
                    norm_stats[task] = {
                        "min": float(mm_all["min"]),
                        "max": float(mm_all["max"]),
                        "scope": "fallback_all_scored",
                    }
                else:
                    norm_stats[task] = {"min": float(mm_fail["min"]), "max": float(mm_fail["max"]), "scope": "fail_only"}

    # 6) Fit/load global monotonic warp f from success_ref (optional).
    warp_path = out_dir / "warp_f.npz"
    warp_cfg = {
        "warp_method": str(args.warp_method),
        "success_fit_step_ratio": float(args.success_fit_step_ratio),
        "success_ref_sig": str(success_ref_sig),
        "success_ref_count": int(len(success_ref)),
        "embed_cfg_id": str(embed_cfg_id),
        "centers_cfg_id": str(centers_cfg_id),
        "value_mapping": str(args.value_mapping),
        "failure_end_value": float(args.failure_end_value),
        "minmax_skip_prefix_k": int(skip_k),
    }
    warp_cfg_id = _sha1_short(json.dumps(warp_cfg, sort_keys=True))

    x_breaks: np.ndarray | None = None
    y_breaks: np.ndarray | None = None
    warp_info: dict[str, Any] | None = None

    if bool(args.warp_enable):
        blob = None if args.overwrite else _load_npz_dict(warp_path)
        if blob is not None:
            try:
                stored_id = _decode_bytes(blob.get("warp_cfg_id"))
            except Exception:
                stored_id = None
            if stored_id == warp_cfg_id and ("x_breaks" in blob) and ("y_breaks" in blob):
                x_breaks = np.asarray(blob["x_breaks"], dtype=np.float32).reshape(-1)
                y_breaks = np.asarray(blob["y_breaks"], dtype=np.float32).reshape(-1)
                try:
                    if "info" in blob:
                        warp_info = blob["info"].item()
                except Exception:
                    warp_info = None
                logging.info(f"Loaded warp f from {warp_path} (warp_cfg_id={warp_cfg_id}, breaks={len(x_breaks)})")

        if x_breaks is None or y_breaks is None:
            if args.warp_method != "isotonic":
                raise ValueError(f"Unknown warp_method: {args.warp_method}")
            logging.info("Fitting global warp f (isotonic) from success_ref samples...")

            xs_all: list[float] = []
            ys_all: list[float] = []
            used_eps = 0
            skipped_eps = 0

            need_high = args.camera_mode in ("high", "both")
            need_wrist = args.camera_mode in ("wrist", "both")
            bs = int(max(1, args.embed_batch_size))

            for p in tqdm.tqdm(success_ref, desc="Fit warp: sample success_ref"):
                try:
                    with h5py.File(p, "r") as ep:
                        sig = _get_trim_signal(ep, trim_signal=str(args.trim_signal))
                        span = _trim_static_head_tail(sig, epsilon=float(args.epsilon))
                        if span is None:
                            skipped_eps += 1
                            continue
                        s, e = span
                        t_total = int(e - s + 1)
                        if t_total <= 1:
                            skipped_eps += 1
                            continue
                        if args.max_episode_len is not None and t_total > int(args.max_episode_len):
                            skipped_eps += 1
                            continue
                        task = _task_from_hdf5(ep, fallback_name=p.name)

                        if task not in centers_blob:
                            skipped_eps += 1
                            continue

                        t_list = _sample_timesteps_by_ratio(t_total, step_ratio=float(args.success_fit_step_ratio))
                        if not t_list or t_list[0] != 0 or t_list[-1] != (t_total - 1):
                            skipped_eps += 1
                            continue

                        embed_cache_path = _episode_embed_cache_path(embed_cache_dir, p)
                        high_map, wrist_map = _load_episode_embed_maps(
                            embed_cache_path,
                            expected_trim_start=int(s),
                            expected_trim_end=int(e),
                            expected_t_total=int(t_total),
                        )

                        missing_high = [int(t) for t in t_list if need_high and int(t) not in high_map]
                        missing_wrist = [int(t) for t in t_list if need_wrist and int(t) not in wrist_map]

                        imgs_high = None
                        imgs_wrist = None
                        if missing_high:
                            imgs_high = _load_images_full(ep, "cam_high", s=s, e=e)
                        if missing_wrist:
                            imgs_wrist = _load_images_full(ep, "cam_left_wrist", s=s, e=e)

                except Exception:
                    skipped_eps += 1
                    continue

                # Compute missing prefix embeddings and update shared cache.
                new_high: dict[int, np.ndarray] = {}
                new_wrist: dict[int, np.ndarray] = {}
                missing_high_set = set(missing_high)
                missing_wrist_set = set(missing_wrist)
                if missing_high_set or missing_wrist_set:
                    for i0 in range(0, len(t_list), bs):
                        i1 = min(len(t_list), i0 + bs)
                        chunk_ts = t_list[i0:i1]
                        videos: list[torch.Tensor] = []
                        tags: list[tuple[int, str]] = []
                        for t in chunk_ts:
                            prefix_len = int(t + 1)
                            rel_idx = _prefix_sample_indices(prefix_len, n=int(args.vjepa_num_frames))
                            if int(t) in missing_high_set:
                                assert imgs_high is not None
                                frames = imgs_high[rel_idx]
                                videos.append(torch.from_numpy(frames))
                                tags.append((int(t), "high"))
                            if int(t) in missing_wrist_set:
                                assert imgs_wrist is not None
                                frames = imgs_wrist[rel_idx]
                                videos.append(torch.from_numpy(frames))
                                tags.append((int(t), "wrist"))
                        if not videos:
                            continue
                        embs = embedder.embed_batch(videos)
                        for emb, (t, cam_key) in zip(embs, tags, strict=True):
                            emb32 = np.asarray(emb, dtype=np.float32).reshape(-1)
                            if cam_key == "high":
                                high_map[int(t)] = emb32
                                new_high[int(t)] = emb32
                            elif cam_key == "wrist":
                                wrist_map[int(t)] = emb32
                                new_wrist[int(t)] = emb32
                            else:
                                raise ValueError(cam_key)

                    if new_high or new_wrist:
                        _save_episode_embed_cache(
                            embed_cache_path,
                            hdf5_path=p,
                            trim_start=int(s),
                            trim_end=int(e),
                            t_total=int(t_total),
                            vjepa_ckpt=args.vjepa_ckpt,
                            vjepa_img_size=int(args.vjepa_img_size),
                            vjepa_num_frames=int(args.vjepa_num_frames),
                            new_high=(new_high if new_high else None),
                            new_wrist=(new_wrist if new_wrist else None),
                        )

                # Compute distances at sampled timesteps.
                dist_samp = np.zeros((len(t_list),), dtype=np.float32)
                ok = True
                if args.camera_mode == "both" and args.camera_fusion in ("dist_min", "dist_mean"):
                    c_high = centers_scaled_blob[task]["high"]
                    c_wrist = centers_scaled_blob[task]["wrist"]
                    mean_h, invstd_h, l2_h = scaler_stats[task]["high"]
                    mean_w, invstd_w, l2_w = scaler_stats[task]["wrist"]
                    for j, t in enumerate(t_list):
                        eh_raw = high_map.get(int(t))
                        ew_raw = wrist_map.get(int(t))
                        if eh_raw is None or ew_raw is None:
                            ok = False
                            break
                        eh = np.asarray(eh_raw, dtype=np.float32).reshape(-1)
                        ew = np.asarray(ew_raw, dtype=np.float32).reshape(-1)
                        if l2_h:
                            eh = _l2_normalize(eh)
                        if l2_w:
                            ew = _l2_normalize(ew)
                        eh_s = (eh - mean_h) * invstd_h
                        ew_s = (ew - mean_w) * invstd_w
                        dh = _min_dist_to_centers(eh_s, c_high)
                        dw = _min_dist_to_centers(ew_s, c_wrist)
                        dist_samp[j] = float(min(dh, dw)) if args.camera_fusion == "dist_min" else float(0.5 * (dh + dw))
                elif args.camera_mode == "both" and args.camera_fusion in ("emb_mean", "emb_concat"):
                    c_fused = centers_scaled_blob[task]["fused"]
                    mean_f, invstd_f, l2_f = scaler_stats[task]["fused"]
                    for j, t in enumerate(t_list):
                        eh_raw = high_map.get(int(t))
                        ew_raw = wrist_map.get(int(t))
                        if eh_raw is None or ew_raw is None:
                            ok = False
                            break
                        if args.camera_fusion == "emb_mean":
                            ef = 0.5 * (np.asarray(eh_raw, dtype=np.float32) + np.asarray(ew_raw, dtype=np.float32))
                        else:
                            ef = np.concatenate(
                                [np.asarray(eh_raw, dtype=np.float32), np.asarray(ew_raw, dtype=np.float32)], axis=0
                            )
                        ef = np.asarray(ef, dtype=np.float32).reshape(-1)
                        if l2_f:
                            ef = _l2_normalize(ef)
                        ef_s = (ef - mean_f) * invstd_f
                        dist_samp[j] = _min_dist_to_centers(ef_s, c_fused)
                else:
                    c_single = centers_scaled_blob[task]["single"]
                    mean_s, invstd_s, l2_s = scaler_stats[task]["single"]
                    for j, t in enumerate(t_list):
                        if args.camera_mode == "high":
                            e_raw = high_map.get(int(t))
                        else:
                            e_raw = wrist_map.get(int(t))
                        if e_raw is None:
                            ok = False
                            break
                        e = np.asarray(e_raw, dtype=np.float32).reshape(-1)
                        if l2_s:
                            e = _l2_normalize(e)
                        e_s = (e - mean_s) * invstd_s
                        dist_samp[j] = _min_dist_to_centers(e_s, c_single)

                if not ok:
                    skipped_eps += 1
                    continue

                # dist -> v_base -> v1 (cummax+scale to 1)
                v_base_s = _distance_to_value_linear_offset_scale(dist_samp, eps=float(args.norm_eps))
                v1_s = _cummax_and_scale(v_base_s, target_end=1.0, eps=float(args.norm_eps))
                if float(v1_s[-1]) <= float(args.norm_eps):
                    skipped_eps += 1
                    continue

                for t, xv in zip(t_list, v1_s.tolist(), strict=True):
                    xs_all.append(float(xv))
                    ys_all.append(float(t) / float(t_total - 1))
                used_eps += 1

            # Anchor points for stability.
            xs_all.extend([0.0, 1.0])
            ys_all.extend([0.0, 1.0])

            x_breaks, y_breaks, warp_info = _fit_isotonic_warp(np.asarray(xs_all), np.asarray(ys_all))
            np.savez_compressed(
                str(warp_path),
                schema_version=np.asarray(1, dtype=np.int32),
                warp_cfg_id=np.asarray(str(warp_cfg_id)),
                warp_cfg=np.asarray(json.dumps(warp_cfg, sort_keys=True)),
                x_breaks=np.asarray(x_breaks, dtype=np.float32),
                y_breaks=np.asarray(y_breaks, dtype=np.float32),
                info=np.asarray(warp_info, dtype=object),
                used_success_episodes=np.asarray(int(used_eps), dtype=np.int32),
                skipped_success_episodes=np.asarray(int(skipped_eps), dtype=np.int32),
            )
            logging.info(
                f"Saved warp f to {warp_path} (warp_cfg_id={warp_cfg_id}, used={used_eps}, skipped={skipped_eps}, breaks={len(x_breaks)})"
            )
    else:
        warp_cfg_id = ""

    # 7) Stage 2: write value_pred for each episode_values npz.
    updated = 0
    reused_value = 0
    for p, source, is_success in tqdm.tqdm(scoring, desc="Stage2: value_pred"):
        out_npz = out_dir / "episode_values" / f"{p.name}.npz"
        if not out_npz.exists():
            continue
        z = np.load(str(out_npz), allow_pickle=True)

        # If value_pred already computed with the same config, skip rewriting.
        if "value_pred" in z.files and not args.overwrite:
            try:
                stored_mapping = _decode_bytes(z["value_mapping"]) if "value_mapping" in z.files else ""
                stored_failure_end = float(np.asarray(z["failure_end_value"]).item()) if "failure_end_value" in z.files else None
                stored_skip_k = int(np.asarray(z["minmax_skip_prefix_k"]).item()) if "minmax_skip_prefix_k" in z.files else None
                stored_enforce_mono = bool(int(np.asarray(z["enforce_monotonic"]).item())) if "enforce_monotonic" in z.files else None
                stored_warp_enable = bool(int(np.asarray(z["warp_enable"]).item())) if "warp_enable" in z.files else False
                stored_warp_method = _decode_bytes(z["warp_method"]) if "warp_method" in z.files else ""
                stored_fit_ratio = float(np.asarray(z["success_fit_step_ratio"]).item()) if "success_fit_step_ratio" in z.files else None
                stored_warp_cfg_id = _decode_bytes(z["warp_cfg_id"]) if "warp_cfg_id" in z.files else ""
                stored_set_term = bool(int(np.asarray(z["set_success_terminal_reward"]).item())) if "set_success_terminal_reward" in z.files else False
                stored_term = float(np.asarray(z["success_terminal_reward"]).item()) if "success_terminal_reward" in z.files else None

                ok = True
                ok &= (stored_mapping == str(args.value_mapping))
                ok &= (stored_failure_end is not None and abs(stored_failure_end - float(args.failure_end_value)) < 1e-8)
                ok &= (stored_skip_k is not None and int(stored_skip_k) == int(skip_k))
                ok &= (stored_enforce_mono is not None and stored_enforce_mono == bool(args.enforce_monotonic))
                ok &= (stored_set_term == bool(args.set_success_terminal_reward))
                ok &= (stored_term is not None and abs(stored_term - float(args.success_terminal_reward)) < 1e-8)
                ok &= (stored_warp_enable == bool(args.warp_enable))
                if bool(args.warp_enable):
                    ok &= (stored_warp_method == str(args.warp_method))
                    ok &= (stored_fit_ratio is not None and abs(stored_fit_ratio - float(args.success_fit_step_ratio)) < 1e-8)
                    ok &= (stored_warp_cfg_id == str(warp_cfg_id))
                if ok:
                    reused_value += 1
                    continue
            except Exception:
                pass

        dist = np.asarray(z["distance"], dtype=np.float32)
        task = _decode_bytes(z["task"]) if "task" in z else p.stem
        if args.value_mapping == "sigmoid":
            mm = norm_stats.get(task)
            if mm is None:
                # No stats for this task (e.g., everything skipped). Define as zeros.
                value = np.zeros_like(dist, dtype=np.float32)
                mm = {"min": float("nan"), "max": float("nan"), "scope": "no_stats"}
            else:
                value = _srpo_map_distance(
                    dist,
                    d_min=float(mm["min"]),
                    d_max=float(mm["max"]),
                    reward_scale=float(args.reward_scale),
                    sigmoid_steepness=float(args.sigmoid_steepness),
                    sigmoid_offset=float(args.sigmoid_offset),
                    eps=float(args.norm_eps),
                ).astype(np.float32)
        elif args.value_mapping == "linear_offset_scale":
            value = _distance_to_value_linear_offset_scale(dist, eps=float(args.norm_eps)).astype(np.float32)
            mm = {"min": float("nan"), "max": float("nan"), "scope": "not_used_linear"}
        else:
            raise ValueError(f"Unknown value_mapping: {args.value_mapping}")

        # (Modification 1) Make curve monotonic and scale by end value.
        # - success: end==1
        # - non-success (e.g., rollouts_failure): end==failure_end_value
        target_end = 1.0 if bool(is_success) else float(args.failure_end_value)
        if bool(args.enforce_monotonic):
            value = _cummax_and_scale(value, target_end=float(target_end), eps=float(args.norm_eps)).astype(np.float32)
        else:
            value = _scale_to_end(value, target_end=float(target_end), eps=float(args.norm_eps)).astype(np.float32)

        # Apply global warp f (fit from success_ref) and re-enforce monotonic+end-scaling.
        if bool(args.warp_enable) and x_breaks is not None and y_breaks is not None:
            value = _apply_warp(value, x_breaks=x_breaks, y_breaks=y_breaks).astype(np.float32)
            if bool(args.enforce_monotonic):
                value = _cummax_and_scale(value, target_end=float(target_end), eps=float(args.norm_eps)).astype(np.float32)
            else:
                value = _scale_to_end(value, target_end=float(target_end), eps=float(args.norm_eps)).astype(np.float32)

        # Force prefix frames to zero (for shaping stability).
        if skip_k > 0 and value.size > 0:
            value[: min(skip_k, int(value.size))] = 0.0

        # Legacy option: if user insists, allow overriding terminal reward (not recommended).
        # Note this may break the end-scaling invariants above if success_terminal_reward != target_end.
        if bool(args.set_success_terminal_reward) and bool(is_success) and value.size > 0 and int(value.size) - 1 >= int(skip_k):
            value[-1] = float(args.success_terminal_reward)

        # Overwrite file with added value_pred + mapping params.
        save_dict = {k: z[k] for k in z.files}
        save_dict["value_pred"] = value
        save_dict["value_mapping"] = np.asarray(str(args.value_mapping))
        save_dict["enforce_monotonic"] = np.asarray(int(bool(args.enforce_monotonic)), dtype=np.int32)
        save_dict["warp_enable"] = np.asarray(int(bool(args.warp_enable)), dtype=np.int32)
        save_dict["warp_method"] = np.asarray(str(args.warp_method))
        save_dict["success_fit_step_ratio"] = np.asarray(float(args.success_fit_step_ratio), dtype=np.float32)
        save_dict["warp_cfg_id"] = np.asarray(str(warp_cfg_id))
        save_dict["norm_min"] = np.asarray(float(mm["min"]), dtype=np.float32)
        save_dict["norm_max"] = np.asarray(float(mm["max"]), dtype=np.float32)
        save_dict["norm_scope_used"] = np.asarray(str(mm["scope"]))
        save_dict["reward_scale"] = np.asarray(float(args.reward_scale), dtype=np.float32)
        save_dict["sigmoid_steepness"] = np.asarray(float(args.sigmoid_steepness), dtype=np.float32)
        save_dict["sigmoid_offset"] = np.asarray(float(args.sigmoid_offset), dtype=np.float32)
        save_dict["set_success_terminal_reward"] = np.asarray(int(bool(args.set_success_terminal_reward)), dtype=np.int32)
        save_dict["success_terminal_reward"] = np.asarray(float(args.success_terminal_reward), dtype=np.float32)
        save_dict["failure_end_value"] = np.asarray(float(args.failure_end_value), dtype=np.float32)
        save_dict["minmax_skip_prefix_k"] = np.asarray(int(skip_k), dtype=np.int32)
        np.savez_compressed(str(out_npz), **save_dict)
        updated += 1

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "teleop_hdf5_dir": str(args.teleop_hdf5_dir),
        "dataset_hdf5_dir": None if args.dataset_hdf5_dir is None else str(args.dataset_hdf5_dir),
        "rollouts_hdf5_dir": str(Path(args.dataset_hdf5_dir) if args.dataset_hdf5_dir is not None else Path(args.rollouts_hdf5_dir)),
        "intervention_hdf5_dir": (
            None
            if (args.dataset_hdf5_dir is None and args.intervention_hdf5_dir is None)
            else str(
                Path(args.intervention_hdf5_dir)
                if args.intervention_hdf5_dir is not None
                else (Path(args.dataset_hdf5_dir) / "intervention")
            )
        ),
        "out_cache_dir": str(out_dir),
        "success_mode": str(args.success_mode),
        "shortest_p": float(args.shortest_p),
        "score_sources": str(args.score_sources),
        "camera_mode": str(args.camera_mode),
        "camera_fusion": str(args.camera_fusion),
        "epsilon": float(args.epsilon),
        "max_episode_len": None if args.max_episode_len is None else int(args.max_episode_len),
        "vjepa_ckpt": str(args.vjepa_ckpt),
        "vjepa_img_size": int(args.vjepa_img_size),
        "vjepa_num_frames": int(args.vjepa_num_frames),
        "device": str(device),
        "enable_fp16": bool(args.enable_fp16),
        "embed_batch_size": int(args.embed_batch_size),
        "embed_cache_root": str(embed_cache_root),
        "embed_cache_dir": str(embed_cache_dir),
        "embed_cache_cfg_id": str(embed_cfg_id),
        "embed_cache_cfg": embed_cfg,
        "distance_space": "standardized",
        "cluster_centers_schema_version": 3,
        "cluster_centers_embed_cfg_id": str(embed_cfg_id),
        "l2_normalize_groups": {"enabled": int(n_l2), "total": int(n_total)},
        "timestep_stride": int(args.timestep_stride),
        "dbscan_eps": float(args.dbscan_eps),
        "dbscan_min_samples": int(args.dbscan_min_samples),
        "norm_scope": str(args.norm_scope),
        "minmax_skip_prefix_k": int(skip_k),
        "enforce_monotonic": bool(args.enforce_monotonic),
        "value_mapping": str(args.value_mapping),
        "failure_end_value": float(args.failure_end_value),
        "warp_enable": bool(args.warp_enable),
        "warp_method": str(args.warp_method),
        "success_fit_step_ratio": float(args.success_fit_step_ratio),
        "warp_path": str(warp_path),
        "warp_cfg_id": str(warp_cfg_id),
        "warp_cfg": warp_cfg,
        "warp_info": warp_info,
        "reward_scale": float(args.reward_scale),
        "sigmoid_steepness": float(args.sigmoid_steepness),
        "sigmoid_offset": float(args.sigmoid_offset),
        "norm_eps": float(args.norm_eps),
        "set_success_terminal_reward": bool(args.set_success_terminal_reward),
        "success_terminal_reward": float(args.success_terminal_reward),
        "cluster_centers_path": str(centers_path),
        "cluster_centers_cfg_id": str(centers_cfg_id),
        "cluster_centers_cfg": centers_cfg,
        "hdf5_to_vjepa_value_map": str(map_path),
        "num_scoring_episodes": int(len(scoring)),
        "num_success_ref_episodes": int(len(success_ref)),
        "norm_stats": norm_stats,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info(f"Stage2 done: updated={updated}, reused={reused_value}. Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))


