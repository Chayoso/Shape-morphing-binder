"""Main runtime surface synthesis function."""

import torch
from typing import Dict, Optional, Tuple
from contextlib import nullcontext

from ..utils.config import default_cfg
from ..utils.utils import ensure_torch
from ..analysis.knn import HybridFAISSKNN, FAISS_AVAILABLE
from ..analysis.detection import compute_surface_mask_diff
from ..analysis.pca import batched_pca_surface_optimized
from ..core.point_sampling import sample_surface_points_diff
from ..analysis.density import density_equalize_diff
from ..processing.smoother import surface_smoother_diff
from ..processing.deformation import smooth_F_diff_optimized

def extract_config_params(cfg: Dict) -> Dict:
    """Extract and validate config parameters."""
    return {
        'use_hybrid': bool(cfg.get("use_hybrid_faiss", True)),
        'k_surface': int(cfg.get("k_surface", 36)),
        'thr_pct': float(cfg.get("thr_percentile", 8.0)),
        'ema_beta': float(cfg.get("ema_beta", 0.95)),
        'hys': float(cfg.get("hysteresis", 0.03)),
        'tau': float(cfg.get("soft_tau", 0.08)),
        'M': int(cfg.get("M", 50_000)),
        'alpha': float(cfg.get("surf_jitter_alpha", 0.6)),
        'thickness': float(cfg.get("thickness", 0.00)),
        'density_g': float(cfg.get("density_gamma", 2.5)),
        'sampling_tau': float(cfg.get("sampling_tau", 0.2)),
        'surface_power': float(cfg.get("surface_power", 4.0)),
        'knn_tau': float(cfg.get("knn_tau", 0.15)),
        'use_ivf': bool(cfg.get("use_faiss_ivf", True)),
        'use_amp': bool(cfg.get("use_amp", True)),
    }


def run_fast_cov_only_path(cfg: Dict, cache: Dict, x_low: torch.Tensor, F_low: torch.Tensor, device: torch.device) -> Dict:
    """Fast path: reuse μ, recompute Σ only."""
    sigma0 = float(cfg.get("sigma0", 0.08))
    
    pts = cache["pts"].to(device)
    normals = cache.get("normals")
    if normals is not None:
        normals = normals.to(device)
    idxF = cache["idxF"].to(device)
    wF = cache["wF"].to(device)
    
    if cfg.get("fast_use_f_smooth", False):
        ed = cfg.get("ed", {})
        if ed.get("enabled", True):
            ed_fast = dict(ed)
            ed_fast["num_nodes"] = max(ed.get("num_nodes", 180) // 2, 50)
            F_smooth = smooth_F_diff_optimized(x_low, F_low, ed_fast)
        else:
            F_smooth = F_low
    else:
        F_smooth = F_low
    
    F_neighbors = F_smooth[idxF]
    F_loc = torch.einsum('mk,mkrc->mrc', wF, F_neighbors)
    cov = (sigma0 ** 2) * torch.matmul(F_loc, F_loc.transpose(-2, -1))
    
    debug = {
        "fast_cov_only": True,
        "cache_reused": True,
        "M": pts.shape[0],
        "sigma0": sigma0,
        "thr_low": 0.0,
        "thr_high": 0.0,
        "ema_thr": 0.0,
        "mean_prob": 0.0,
    }
    
    return {
        "points": pts,
        "normals": normals,
        "F_smooth": F_loc,
        "cov": cov,
        "debug": debug,
        "state": {"cache": cache},
    }


def prepare_inputs(x_low, F_low, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare and validate inputs."""
    x_requires_grad = x_low.requires_grad if torch.is_tensor(x_low) else False
    F_requires_grad = F_low.requires_grad if torch.is_tensor(F_low) else False
    
    x_low = ensure_torch(x_low, device=device)
    F_low = ensure_torch(F_low, device=device).reshape(-1, 3, 3)
    
    if x_requires_grad and not x_low.requires_grad:
        x_low.requires_grad_(True)
    if F_requires_grad and not F_low.requires_grad:
        F_low.requires_grad_(True)
    
    return x_low, F_low


def synthesize_runtime_surface(
    x_low: torch.Tensor,
    F_low: torch.Tensor,
    cfg: Dict,
    ema_state: Optional[Dict] = None,
    seed: int = 1234,
    differentiable: bool = True,
    return_torch: bool = True
) -> Dict:
    """
    Full runtime surface synthesis with hybrid FAISS and differentiable pipeline.
    
    Supports FAST path for μ reuse + Σ-only recomputation.
    """
    if ema_state is None:
        ema_state = {}
    
    device = x_low.device if torch.is_tensor(x_low) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    x_low, F_low = prepare_inputs(x_low, F_low, device)
    
    fast_only = bool(cfg.get("fast_cov_only", False))
    cache = ema_state.get("cache", {})
    
    if fast_only and all(k in cache for k in ["pts", "idxF", "wF"]):
        return run_fast_cov_only_path(cfg, cache, x_low, F_low, device)
    
    params = extract_config_params(cfg)
    
    knn = HybridFAISSKNN(
        use_faiss=params['use_hybrid'] and FAISS_AVAILABLE,
        use_ivf=params['use_ivf'],
        tau=params['knn_tau'],
        nlist=int(cfg.get("ivf_nlist", 100)),
        nprobe=int(cfg.get("ivf_nprobe", 10)),
        use_soft_radius=bool(cfg.get("use_soft_radius", False)),
        soft_radius_candidates=int(cfg.get("soft_radius_candidates", 128))
    )
    
    generator = torch.Generator(device=device).manual_seed(int(seed))
    
    use_amp = params['use_amp'] and device.type == 'cuda'
    amp_ctx = torch.amp.autocast('cuda') if use_amp else nullcontext()
    
    with amp_ctx:
        if cfg.get("use_surface_detection", True):
            surf_prob, normals, spacing, thr_low, thr_high = compute_surface_mask_diff(
                x_low, knn, params['k_surface'], params['thr_pct'],
                ema_state.get("ema_thr"), params['ema_beta'], params['hys'], 
                params['tau'], params['surface_power'], ema_state
            )
        else:
            N = x_low.shape[0]
            surf_prob = torch.full((N,), 1.0 / N, device=device)
            idx, w = knn(x_low, x_low, params['k_surface'])
            normals, _, spacing = batched_pca_surface_optimized(x_low, idx, w)
            thr_low = thr_high = 0.0
            ema_state.update({"ema_thr": 0.0, "thr_low": 0.0, "thr_high": 0.0})
        
        pts, nrm, anchors = sample_surface_points_diff(
            x_low, normals, spacing, surf_prob,
            params['M'], params['alpha'], params['thickness'], params['density_g'], 
            seed, tau=params['sampling_tau'], generator=generator
        )
        
        eq_cfg = dict(cfg.get("post_equalize", {}))
        eq_cfg['knn_tau'] = params['knn_tau']
        if eq_cfg.get("enabled", True):
            pts, nrm = density_equalize_diff(
                pts, nrm, anchors, x_low, normals, params['thickness'], knn, eq_cfg
            )
        
        smooth_cfg = cfg.get("smoother", {})
        if smooth_cfg.get("enabled", False):
            pts = surface_smoother_diff(
                pts, x_low, normals, knn,
                iters=int(smooth_cfg.get("iters", 3)),
                k=int(smooth_cfg.get("k", 24)),
                step=float(smooth_cfg.get("step", 0.12)),
                lambda_normal=float(smooth_cfg.get("lambda_normal", 0.15)),
                mls_every=int(smooth_cfg.get("mls_every", 2)),
            )
        
        # ✅ NEW: Normal Smoothing (after position smoothing)
        if cfg.get("smooth_normals", False) and nrm is not None:
            from ..processing.smoother import smooth_normals_diff
            
            nrm = smooth_normals_diff(
                nrm,
                pts,  # Use upsampled positions for spatial neighbors
                knn,
                iters=int(cfg.get("normal_smooth_iters", 2)),
                k=int(cfg.get("normal_smooth_k", 16)),
                lambda_smooth=float(cfg.get("normal_smooth_lambda", 0.8))
            )
            
            if cfg.get("verbose", False):
                print(f"[Normal] Smoothed normals ({cfg.get('normal_smooth_iters', 2)} iters)")
        
        F_smooth = F_low
        if cfg.get("use_F_kernel", True):
            ed = cfg.get("ed", {})
            if ed.get("enabled", True):
                F_smooth = smooth_F_diff_optimized(x_low, F_low, ed)
        
        k_F = int(cfg.get("k_F", 32))
        sigma0 = float(cfg.get("sigma0", 0.08))
        idxF, wF = knn(pts, x_low, k_F)
        F_neighbors = F_smooth[idxF]
        F_loc = torch.einsum('mk,mkrc->mrc', wF, F_neighbors)
        cov = (sigma0 ** 2) * torch.matmul(F_loc, F_loc.transpose(-2, -1))
    
    if use_amp:
        pts = pts.float()
        nrm = nrm.float()
        F_loc = F_loc.float()
        cov = cov.float()
    
    if cfg.get("cache_for_fast", True):
        ema_state.setdefault("cache", {})
        ema_state["cache"].update({
            "pts": pts.detach().clone(),
            "normals": nrm.detach().clone() if nrm is not None else None,
            "idxF": idxF.detach().clone(),
            "wF": wF.detach().clone(),
        })
    
    if cfg.get("clear_cache_each_call", True):
        knn.clear_cache()
    
    debug = {
        "thr_low": float(thr_low),
        "thr_high": float(thr_high),
        "ema_thr": float(ema_state.get("ema_thr", 0.0)),
        "mean_prob": float(surf_prob.mean().item()),
        "use_hybrid_faiss": params['use_hybrid'] and FAISS_AVAILABLE,
        "use_faiss_ivf": params['use_ivf'],
        "use_soft_radius": bool(cfg.get("use_soft_radius", False)),
        "sampling_tau": params['sampling_tau'],
        "surface_power": params['surface_power'],
        "knn_tau": params['knn_tau'],
        "fast_cov_only": False,
        "normal_smoothed": cfg.get("smooth_normals", False),  # ✅ 추가
    }
    
    if return_torch:
        return {
            "points": pts,
            "normals": nrm,
            "F_smooth": F_loc,
            "cov": cov,
            "debug": debug,
            "state": ema_state,
            "anchors": anchors,
            "surf_prob": surf_prob,
        }
    else:
        return {
            "points": pts.detach().cpu().numpy(),
            "normals": nrm.detach().cpu().numpy(),
            "F_smooth": F_loc.detach().cpu().numpy(),
            "cov": cov.detach().cpu().numpy(),
            "debug": debug,
            "state": ema_state,
        }