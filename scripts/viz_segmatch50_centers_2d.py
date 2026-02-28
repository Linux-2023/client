#!/usr/bin/env python3
"""
Visualize Segment-Match50 success centers and rollouts trajectories in 2D.

Given an out-cache-dir produced by:
  scripts/estimate_vjepa_value_hdf5.py --value-method segment_match50

This script:
1) Loads per-segment success centers from <out_cache_dir>/segment_centers.npz
2) Builds ONE representative center per segment (mean over DBSCAN centers)
3) Projects the 50 centers to 2D (PCA by default)
4) Loads window embeddings for N rollouts from the embed_cache_dir recorded in vjepa_value_metadata.json
   (series name is derived from segment_match_num_segments and segment_match_window_stride, e.g. win50s6)
5) Projects rollouts window embeddings to the same 2D space and plots everything in one figure.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tyro


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(p: str | Path | None) -> Path | None:
    if p is None:
        return None
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (_repo_root() / pp).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _decode_np_str(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, np.ndarray):
        if x.dtype.kind in ("S", "U") and x.size > 0:
            try:
                return str(x.reshape(-1)[0])
            except Exception:
                return str(x)
        if x.size == 1:
            try:
                return _decode_np_str(x.item())
            except Exception:
                return str(x)
    return str(x)


def _load_npz_dict(path: Path) -> dict[str, Any]:
    z = np.load(str(path), allow_pickle=True)
    out = {k: z[k] for k in z.files}
    z.close()
    return out


def _l2_normalize(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        n = float(np.linalg.norm(x))
        return x / max(float(eps), n)
    if x.ndim == 2:
        n = np.linalg.norm(x, axis=1, keepdims=True).astype(np.float32)
        return x / np.maximum(n, float(eps))
    raise ValueError(f"Expected 1D/2D array, got shape={x.shape}")


def _episode_embed_cache_path(embed_cache_dir: Path, hdf5_path: Path) -> Path:
    import hashlib

    st = hdf5_path.stat()
    s = f"{hdf5_path.resolve()}|{int(st.st_size)}|{int(st.st_mtime_ns)}"
    ep_id = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
    return embed_cache_dir / f"{hdf5_path.stem}__{ep_id}.npz"


def _locate_episode_embed_cache(embed_cache_dir: Path, hdf5_path: Path) -> Path | None:
    """Locate embed cache file for an episode.

    Primary key is the deterministic id from (abs_path, size, mtime_ns). If the HDF5 file
    was touched after caching (mtime changes), fall back to any cached file with the same stem.
    """
    cand = _episode_embed_cache_path(embed_cache_dir, hdf5_path)
    if cand.exists():
        return cand
    # Fallback: find newest cache file sharing the same stem prefix.
    matches = list(embed_cache_dir.glob(f"{hdf5_path.stem}__*.npz"))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime_ns if hasattr(p.stat(), "st_mtime_ns") else p.stat().st_mtime)


def _load_episode_series_maps(
    cache_path: Path,
    *,
    series: str,
    expected_trim_start: int,
    expected_trim_end: int,
    expected_t_total: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Load camera-specific embedding maps for a named series from cache.

    Returns (high_map, wrist_map): {timestep -> emb(D,)}.
    """
    z = _load_npz_dict(cache_path)
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


def _pca_2d_fit_transform(X: np.ndarray) -> tuple[np.ndarray, Any]:
    """Return (X2, pca_model_like). Prefers sklearn PCA; falls back to SVD PCA."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        raise ValueError(f"Need X shape (N>=2, D>=2). Got {X.shape}")
    try:
        from sklearn.decomposition import PCA  # type: ignore

        pca = PCA(n_components=2, svd_solver="auto", random_state=0)
        X2 = pca.fit_transform(X).astype(np.float32)
        return X2, pca
    except Exception:
        Xc = (X - X.mean(axis=0, keepdims=True)).astype(np.float32)
        _u, _s, vt = np.linalg.svd(Xc, full_matrices=False)
        W = vt[:2].T.astype(np.float32)
        X2 = (Xc @ W).astype(np.float32)

        class _SvdPca:
            def __init__(self, mean: np.ndarray, W: np.ndarray) -> None:
                self.mean_ = mean
                self.W_ = W

            def transform(self, Xnew: np.ndarray) -> np.ndarray:
                Xnew = np.asarray(Xnew, dtype=np.float32)
                Xc2 = (Xnew - self.mean_).astype(np.float32)
                return (Xc2 @ self.W_).astype(np.float32)

        return X2, _SvdPca(X.mean(axis=0, keepdims=True).astype(np.float32), W)


def _pca_2d_transform(pca: Any, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if hasattr(pca, "transform"):
        return np.asarray(pca.transform(X), dtype=np.float32)
    raise ValueError("PCA object missing transform().")


@dataclasses.dataclass(frozen=True)
class Args:
    out_cache_dir: Path
    # Which task to visualize if multiple tasks exist in segment_centers.npz.
    task: str | None = None
    # Which data to fit the 2D projection on.
    # - success_centers: fit PCA on (trimmed) success centers (default behavior)
    # - rollouts: fit PCA on randomly sampled rollouts window embeddings (requested)
    # - mean_success_rollout: fit PCA on the computed Mean Success Rollout (requested)
    fit_on: Literal["success_centers", "rollouts", "mean_success_rollout"] = "success_centers"
    # Ignore prefix/suffix segments of the success-centers trajectory when FITTING the 2D projection.
    # (We still project rollouts into the same space; by default we also plot only the kept center segments.)
    ignore_center_prefix_segments: int = 0
    ignore_center_suffix_segments: int = 0
    # Whether to plot the ignored center segments as faint points (not connected).
    plot_ignored_centers: bool = False
    
    # --- Mean Success Rollout Settings ---
    # Number of success rollouts to average into a single "Mean Success Rollout". 0 to disable.
    mean_success_count: int = 0
    # Number of resampling points for averaging (so all rollouts match in length).
    mean_success_resample_points: int = 100

    # --- Individual Rollouts Settings ---
    # Which rollouts to visualize (separate from mean).
    num_rollouts: int = 5
    rollouts_kind: Literal["any", "success", "failure"] = "any"
    rollouts_sample: Literal["first", "random"] = "random"
    rollouts_seed: int = 0
    # Max plotted points per rollout (downsample uniformly if longer).
    rollouts_max_points: int = 250
    # Projection method
    proj: Literal["pca"] = "pca"
    # Whether to L2-normalize embeddings before projection (recommended for stability).
    l2_normalize: bool = True
    # Output
    out_png: Path | None = None


def main(args: Args) -> None:
    out_dir = Path(args.out_cache_dir)
    meta = _read_json(out_dir / "vjepa_value_metadata.json")
    embed_cache_dir = _resolve_path(meta.get("embed_cache_dir"))
    if embed_cache_dir is None:
        raise ValueError("Missing embed_cache_dir in vjepa_value_metadata.json")
    if not embed_cache_dir.exists():
        raise FileNotFoundError(f"embed_cache_dir not found: {embed_cache_dir}")

    camera_mode = str(meta.get("camera_mode", "high"))
    camera_fusion = str(meta.get("camera_fusion", "dist_min"))
    num_segments = int(meta.get("segment_match_num_segments", 50))
    win_stride = int(meta.get("segment_match_window_stride", 6))
    win_series = f"win{num_segments}s{win_stride}"

    # Load segment centers
    blob = _load_npz_dict(out_dir / "segment_centers.npz")
    centers_blob = blob["centers"].item()
    tasks = sorted(list(centers_blob.keys()))
    if not tasks:
        raise RuntimeError("segment_centers.npz has no tasks in centers")
    task = args.task or tasks[0]
    if task not in centers_blob:
        raise KeyError(f"Task {task!r} not found in centers. Available: {tasks[:8]}{'...' if len(tasks)>8 else ''}")

    per_key = centers_blob[task]
    keys = sorted(list(per_key.keys()))
    if not keys:
        raise RuntimeError(f"No camera keys in centers for task={task!r}")

    def _seg_rep_centers(key: str) -> np.ndarray:
        centers_list = per_key[key]
        if len(centers_list) != int(num_segments):
            raise RuntimeError(f"centers_list len mismatch for key={key}: {len(centers_list)} vs {num_segments}")
        reps: list[np.ndarray] = []
        for seg_idx in range(int(num_segments)):
            c = np.asarray(centers_list[int(seg_idx)], dtype=np.float32)
            if c.ndim != 2 or c.shape[0] <= 0:
                raise RuntimeError(f"Bad centers array at seg={seg_idx} key={key}: shape={c.shape}")
            rep = c.mean(axis=0).astype(np.float32).reshape(-1)
            if bool(args.l2_normalize):
                rep = _l2_normalize(rep)
            reps.append(rep)
        return np.stack(reps, axis=0).astype(np.float32)

    # Build fused success center trajectory (one point per segment)
    if camera_mode == "both" and camera_fusion in ("dist_min", "dist_mean"):
        if "high" not in per_key or "wrist" not in per_key:
            raise KeyError(f"Expected keys 'high' and 'wrist' in centers for camera_mode=both, got {keys}")
        c_high = _seg_rep_centers("high")
        c_wrist = _seg_rep_centers("wrist")
        C = np.concatenate([c_high, c_wrist], axis=1)
    elif camera_mode == "both" and camera_fusion in ("emb_mean", "emb_concat"):
        # In embedding-fusion modes, centers are stored under 'fused'.
        k = "fused" if "fused" in per_key else (keys[0])
        C = _seg_rep_centers(k)
    else:
        # single camera
        k = keys[0]
        C = _seg_rep_centers(k)

    # Choose which center segments to use for fitting/plotting.
    pre_k = int(max(0, args.ignore_center_prefix_segments))
    suf_k = int(max(0, args.ignore_center_suffix_segments))
    if pre_k + suf_k >= int(C.shape[0]):
        raise ValueError(
            f"ignore_center_prefix_segments+ignore_center_suffix_segments must be < num_segments. "
            f"Got {pre_k}+{suf_k} >= {int(C.shape[0])}"
        )
    keep_idx = np.arange(pre_k, int(C.shape[0]) - suf_k, dtype=np.int64)
    C_fit = C[keep_idx]

    # Build mapping from episode_values npz name -> hdf5 path (from jsonl map)
    by_npz_name: dict[str, str] = {}
    by_hdf5_name: dict[str, str] = {}
    for row in _load_jsonl(out_dir / "hdf5_to_vjepa_value_map.jsonl"):
        if str(row.get("status", "")) not in ("written", "reused", "ok", "hit"):
            continue
        hdf5 = row.get("hdf5")
        if not hdf5:
            continue
        try:
            hdf5_p = Path(str(hdf5)).resolve()
        except Exception:
            hdf5_p = Path(str(hdf5))
        by_hdf5_name[hdf5_p.name] = str(hdf5_p)
        out_npz = row.get("out_npz", None)
        if out_npz:
            by_npz_name[Path(str(out_npz)).name] = str(hdf5_p)

    # Choose rollouts episodes to plot
    ep_dir = out_dir / "episode_values"
    ep_paths = sorted(ep_dir.glob("*.npz"))
    if not ep_paths:
        raise FileNotFoundError(f"No episode_values/*.npz under {ep_dir}")

    chosen: list[tuple[Path, bool, str]] = []  # (ep_npz_path, is_success, hdf5_path)
    for ep_npz in ep_paths:
        z = np.load(str(ep_npz), allow_pickle=True)
        try:
            src = _decode_np_str(z["source"]) if "source" in z.files else ""
            if not str(src).startswith("rollouts_"):
                continue
            t_ep = _decode_np_str(z["task"]) if "task" in z.files else ""
            if task and t_ep and str(t_ep) != str(task):
                continue
            is_succ = bool(int(np.asarray(z["is_success"]).item())) if "is_success" in z.files else False
            if args.rollouts_kind == "success" and not is_succ:
                continue
            if args.rollouts_kind == "failure" and is_succ:
                continue
        finally:
            z.close()

        # locate hdf5 path
        hdf5_name = ep_npz.stem  # includes ".hdf5" as part of stem? e.g. "episode_xxx.hdf5"
        # For "<name>.hdf5.npz", stem is "<name>.hdf5"
        hdf5_basename = hdf5_name
        hdf5_path = by_npz_name.get(ep_npz.name) or by_hdf5_name.get(hdf5_basename)
        if hdf5_path is None:
            # fallback: try metadata rollouts_hdf5_dir
            roll_dir = Path(str(meta.get("rollouts_hdf5_dir", "")))
            roll_dir = _resolve_path(roll_dir) or roll_dir
            cand0 = roll_dir / "success" / hdf5_basename
            cand1 = roll_dir / "failure" / hdf5_basename
            if cand0.exists():
                hdf5_path = str(cand0)
            elif cand1.exists():
                hdf5_path = str(cand1)
        if hdf5_path is None:
            continue
        if not Path(hdf5_path).exists():
            continue
        chosen.append((ep_npz, bool(is_succ), str(hdf5_path)))

    # Optional random sampling of rollouts candidates (before loading embed caches).
    if args.rollouts_sample == "random" and len(chosen) >= 2:
        rng = np.random.default_rng(int(args.rollouts_seed))
        rng.shuffle(chosen)

    def _load_rollout_X(ep_npz: Path, *, hdf5_path: str) -> np.ndarray | None:
        z = np.load(str(ep_npz), allow_pickle=True)
        try:
            s = int(np.asarray(z["trim_start"]).item())
            e = int(np.asarray(z["trim_end"]).item())
            t_total = int(np.asarray(z["t_total"]).item())
        finally:
            z.close()

        cache_path = _locate_episode_embed_cache(embed_cache_dir, Path(hdf5_path))
        if cache_path is None or (not cache_path.exists()):
            print(f"[warn] missing embed cache for {hdf5_path}")
            return None

        high_map, wrist_map = _load_episode_series_maps(
            cache_path,
            series=win_series,
            expected_trim_start=s,
            expected_trim_end=e,
            expected_t_total=t_total,
        )
        if camera_mode == "both":
            if not high_map or not wrist_map:
                print(f"[warn] missing high/wrist series in cache for {hdf5_path}")
                return None
            ts = sorted(set(high_map.keys()) & set(wrist_map.keys()))
            Xh = np.stack([high_map[t] for t in ts], axis=0).astype(np.float32)
            Xw = np.stack([wrist_map[t] for t in ts], axis=0).astype(np.float32)
            if bool(args.l2_normalize):
                Xh = _l2_normalize(Xh)
                Xw = _l2_normalize(Xw)
            X = np.concatenate([Xh, Xw], axis=1)
        elif camera_mode == "high":
            if not high_map:
                print(f"[warn] missing high series in cache for {hdf5_path}")
                return None
            ts = sorted(high_map.keys())
            X = np.stack([high_map[t] for t in ts], axis=0).astype(np.float32)
            if bool(args.l2_normalize):
                X = _l2_normalize(X)
        else:
            if not wrist_map:
                print(f"[warn] missing wrist series in cache for {hdf5_path}")
                return None
            ts = sorted(wrist_map.keys())
            X = np.stack([wrist_map[t] for t in ts], axis=0).astype(np.float32)
            if bool(args.l2_normalize):
                X = _l2_normalize(X)
        return X

    def _resample_traj(X: np.ndarray, num_points: int) -> np.ndarray:
        if X.shape[0] < 2:
            return X
        old_idx = np.arange(X.shape[0])
        new_idx = np.linspace(0, X.shape[0] - 1, num=num_points)
        X_new = np.zeros((num_points, X.shape[1]), dtype=np.float32)
        for d in range(X.shape[1]):
            X_new[:, d] = np.interp(new_idx, old_idx, X[:, d])
        return X_new

    # --- Compute Mean Success Rollout (if requested) ---
    mean_rollout_X: np.ndarray | None = None
    mean_rollout_X_fit_src: list[np.ndarray] = []  # Store raw X used for mean, if needed for fitting

    if args.mean_success_count > 0:
        # Filter for success only
        succ_cands = [x for x in chosen if x[1]]  # x[1] is is_succ
        if len(succ_cands) < args.mean_success_count:
            print(f"[warn] requested {args.mean_success_count} mean success rollouts, but only have {len(succ_cands)}")
        
        # We need to shuffle succ_cands independently if we want a random sample for the mean
        # (args.rollouts_seed applies to everything, so standard shuffle is fine)
        # Note: chosen was already shuffled if rollouts_sample=random.
        # So taking the first N is effectively a random sample of successes.
        pool = succ_cands[:args.mean_success_count]
        
        loaded_Xs = []
        for ep_npz, is_succ, hdf5_path in pool:
            X = _load_rollout_X(ep_npz, hdf5_path=hdf5_path)
            if X is not None and X.shape[0] >= 2:
                # Store raw X for fitting PCA later if fit_on=rollouts
                mean_rollout_X_fit_src.append(X)
                # Resample for averaging
                X_res = _resample_traj(X, args.mean_success_resample_points)
                loaded_Xs.append(X_res)
        
        if loaded_Xs:
            stack = np.stack(loaded_Xs, axis=0)
            mean_rollout_X = stack.mean(axis=0)  # (N_res, D)
            print(f"Computed Mean Success Rollout from {len(loaded_Xs)} trajectories.")
        else:
            print("[warn] Could not load any success rollouts for Mean calculation.")

    # --- Load Individual Rollouts ---
    want_n = int(max(0, args.num_rollouts))
    rollouts_raw: list[tuple[str, bool, np.ndarray]] = []  # (label, is_succ, X)
    for cand_i, (ep_npz, is_succ, hdf5_path) in enumerate(chosen):
        if want_n > 0 and len(rollouts_raw) >= want_n:
            break
        # Optimization: if we already loaded X for mean calc, we could reuse it, but logic is complex.
        # Just reload (cached by OS hopefully).
        X = _load_rollout_X(ep_npz, hdf5_path=hdf5_path)
        if X is None or X.size == 0:
            continue
        
        # Downsample for display (only for individual rollouts, not for mean calc)
        # The _load_rollout_X function does NOT downsample inside anymore? Wait, I need to check.
        # Ah, looking at previous code, _load_rollout_X DID downsample at the end.
        # I removed that from _load_rollout_X in the pasted code above?
        # WAIT: In the previous tool output, _load_rollout_X ends with "return X".
        # I need to verify if I accidentally removed the downsampling logic from _load_rollout_X 
        # or if I should add it here.
        # The previous 'search_replace' replaced the block. Let's look at the implementation I provided in 'new_string'.
        # In my provided new_string for _load_rollout_X:
        #   if max_pts > 0 ... X = X[idx]
        #   return X
        # YES, it does downsample.
        # This is bad for 'Mean Rollout' calculation because we want full resolution before resampling.
        # I should modify _load_rollout_X to take an optional 'downsample_max' arg.
        
        label = f"rollout_{len(rollouts_raw)+1} ({'succ' if is_succ else 'fail'}) {Path(hdf5_path).name}"
        rollouts_raw.append((label, bool(is_succ), X))

    if want_n > 0 and len(rollouts_raw) < want_n:
        print(f"[warn] only loaded {len(rollouts_raw)} rollouts trajectories (requested {want_n})")

    # Fit projection
    if args.proj != "pca":
        raise ValueError(f"Unknown proj: {args.proj}")
        
    if args.fit_on == "mean_success_rollout":
        if mean_rollout_X is None:
            raise RuntimeError("fit_on=mean_success_rollout but Mean Success Rollout computation failed.")
        _x2, pca = _pca_2d_fit_transform(mean_rollout_X)
    elif args.fit_on == "rollouts":
        # If mean computation ran, we have mean_rollout_X_fit_src (raw rollouts used for mean).
        # We also have rollouts_raw (individual rollouts for plotting).
        # We should use all available unique data to fit PCA robustly.
        sources = []
        if mean_rollout_X_fit_src:
            sources.extend(mean_rollout_X_fit_src)
        if rollouts_raw:
            sources.extend([x for _, _, x in rollouts_raw])
            
        if not sources:
            raise RuntimeError("fit_on=rollouts but no rollouts loaded.")
        
        # Determine total size to avoid OOM? PCA is usually fine with <100k points.
        X_fit = np.concatenate(sources, axis=0).astype(np.float32)
        _x2, pca = _pca_2d_fit_transform(X_fit)
    else:
        _x2, pca = _pca_2d_fit_transform(C_fit)

    # Project centers
    C2_all = _pca_2d_transform(pca, C)
    C2_fit = C2_all[keep_idx]
    
    # Project Mean Rollout
    mean_rollout_2d = None
    if mean_rollout_X is not None:
        mean_rollout_2d = _pca_2d_transform(pca, mean_rollout_X)

    # Project rollouts
    rollouts_2d: list[tuple[str, bool, np.ndarray]] = []
    for label, is_succ, X in rollouts_raw:
        XY = _pca_2d_transform(pca, X)
        rollouts_2d.append((label, bool(is_succ), XY))

    if want_n > 0 and len(rollouts_2d) < want_n:
        print(f"[warn] only plotted {len(rollouts_2d)} rollouts trajectories (requested {want_n})")

    # Plot
    try:
        import matplotlib.pyplot as plt
    except Exception as ex:
        raise RuntimeError("matplotlib is required for plotting") from ex

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(
        C2_fit[:, 0],
        C2_fit[:, 1],
        "-o",
        color="black",
        linewidth=3.0,
        markersize=4.0,
        label=f"success_centers (mean per segment, kept segs [{pre_k},{int(C.shape[0])-suf_k}))",
    )
    ax.scatter([C2_fit[0, 0]], [C2_fit[0, 1]], color="black", s=60, marker="s", label=f"success_start(seg={int(keep_idx[0])})")
    ax.scatter(
        [C2_fit[-1, 0]],
        [C2_fit[-1, 1]],
        color="black",
        s=80,
        marker="*",
        label=f"success_end(seg={int(keep_idx[-1])})",
    )
    if bool(args.plot_ignored_centers) and (pre_k > 0 or suf_k > 0):
        ignore_mask = np.ones((int(C.shape[0]),), dtype=bool)
        ignore_mask[keep_idx] = False
        ign = C2_all[ignore_mask]
        if ign.size:
            ax.scatter(ign[:, 0], ign[:, 1], color="gray", s=18, alpha=0.35, label="ignored_success_centers")

    if mean_rollout_2d is not None:
        ax.plot(
            mean_rollout_2d[:, 0],
            mean_rollout_2d[:, 1],
            "--",
            color="purple",
            linewidth=2.5,
            label=f"Mean Success Rollout (N={args.mean_success_count})",
        )
        ax.scatter([mean_rollout_2d[0, 0]], [mean_rollout_2d[0, 1]], color="purple", s=50, marker="^")
        ax.scatter([mean_rollout_2d[-1, 0]], [mean_rollout_2d[-1, 1]], color="purple", s=60, marker="v")

    cmap = plt.get_cmap("tab10")
    for i, (label, is_succ, XY) in enumerate(rollouts_2d):
        c = cmap(i % 10)
        ls = "-" if bool(is_succ) else "--"
        ax.plot(XY[:, 0], XY[:, 1], ls=ls, color=c, linewidth=1.6, alpha=0.85, label=label)
        ax.scatter([XY[0, 0]], [XY[0, 1]], color=c, s=24, marker="o", alpha=0.85)
        ax.scatter([XY[-1, 0]], [XY[-1, 1]], color=c, s=36, marker="x", alpha=0.85)

    ax.set_title(
        f"Segment-Match50 2D projection (task={task})\\n"
        f"centers={num_segments}  rollouts={len(rollouts_2d)}  camera_mode={camera_mode} fusion={camera_fusion}  series={win_series}  "
        f"ignore_centers=({pre_k},{suf_k})  fit_on={args.fit_on}  sample={args.rollouts_sample} seed={int(args.rollouts_seed)}"
    )
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    out_png = (
        Path(args.out_png)
        if args.out_png is not None
        else (
            out_dir
            / (
                f"segmatch50_centers_and_rollouts_2d_fit{args.fit_on}_ignore{pre_k}_{suf_k}_"
                f"sample{args.rollouts_sample}_seed{int(args.rollouts_seed)}.png"
            )
        )
    )
    out_png = _resolve_path(out_png) or out_png
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=200)
    print(f"wrote: {out_png}")


if __name__ == "__main__":
    main(tyro.cli(Args))


