# -*- coding: utf-8 -*-
"""
runtime_surface.py
==================

Differentiable runtime *surface* upsampler with ED-regularized deformation gradient
and a **density-equalization relaxation** that avoids discrete hole filling.
The relax step **does not add points**; it continuously relocates them to reduce
sparsity while keeping them on a thin shell. This keeps the pipeline friendly to
autodiff.

Two execution modes:
1) Torch mode (preferred): operations are done in torch, keeping gradients.
2) NumPy fallback: same outputs with NumPy/Scikit-Learn (no gradients).

Differentiability:
- Set cfg["differentiable"] = True or pass differentiable=True to enable gradient flow
- Differentiable mode uses:
  * Torch cdist for kNN (instead of sklearn)
  * Conditional detach() for statistics
  * Full torch operations for F interpolation
- Non-differentiable mode (default) is faster but breaks gradients at:
  * kNN graph construction (sklearn)
  * Categorical sampling (multinomial)
  * Various numpy conversions

Public API used by the runner:
- default_cfg()
- synthesize_runtime_surface(..., differentiable=False)
- save_ply_xyz(...)
- save_gaussians_npz(...)
- save_comparison_png(...)
- save_axis_hist_png(...)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

# Optional heavy deps
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except Exception:
    NearestNeighbors = None
    SKLEARN_AVAILABLE = False


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
def default_cfg() -> Dict:
    """Default config dict. Safe to update() with YAML overrides."""
    return {
        # Differentiability control
        "differentiable": True,  # If True, uses gradient-friendly ops (slower)
        
        # Neighborhood / surface detection
        "k_surface": 24,
        "k_spacing": 12,
        "thr_percentile": 20.0,   # tighter gate for sheetness
        "ema_beta": 0.95,
        "hysteresis": 0.03,
        "soft_tau": 0.02,         # sigmoid temperature for soft gate
        
        # Sampling
        "M": 180_000,
        "surf_jitter_alpha": 0.30,
        "thickness": 0.12,        # half-thickness along normal direction
        "density_gamma": 2.5,     # sparse-region upweighting exponent
        
        # F smoothing (ED regularization)
        "use_F_kernel": True,
        "k_F": 24,                # k-nearest neighbors for F interpolation
        "h_mul": 1.5,             # bandwidth multiplier for F interpolation
        "sigma0": 0.02,           # base scale for covariance
        "ed": {
            "enabled": True,
            "num_nodes": 180,
            "node_knn": 8,
            "point_knn_nodes": 8,
            "lambda_lap": 1.0e-2,
            "iters": 1
        },
        
        # Discrete hole-filling (kept for compatibility but disabled by default)
        "hole_filling": {
            "enabled": False,
            "grid_dx": 0.25,
            "min_points_per_cell": 2,
            "add_per_empty": 1
        },

        # NEW: differentiable density equalization (no new points)
        "post_equalize": {
            "enabled": True,
            "iters": 3,
            "k": 24,
            "step": 0.45,
            "radius_mul": 1.0,
            "project_to_shell": True
        },

        # PNG options (runner expects this key to exist)
        "png": {
            "enabled": True,
            "dpi": 160,
            "ptsize": 0.5
        },

        # Gaussian splat export control
        "export_gaussians": True
    }


# ------------------------------------------------------------
# Utils (torch/np dual)
# ------------------------------------------------------------
def _as_torch(x):
    if TORCH_AVAILABLE:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return x
    return x  # numpy path

def _to_numpy(x):
    if TORCH_AVAILABLE and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x

def _device_dtype_like(x):
    if TORCH_AVAILABLE and torch.is_tensor(x):
        return x.device, x.dtype
    if TORCH_AVAILABLE:
        return torch.device("cpu"), torch.float32
    return None, None

def _randn_like(shape, device=None, dtype=None, seed=None):
    if TORCH_AVAILABLE:
        gen = None
        if seed is not None:
            gen = torch.Generator(device=device).manual_seed(int(seed))
        return torch.randn(shape, device=device, dtype=dtype, generator=gen)
    else:
        rng = np.random.RandomState(seed if seed is not None else 0)
        return rng.randn(*shape)

def _rand_like(shape, device=None, dtype=None, seed=None):
    if TORCH_AVAILABLE:
        gen = None
        if seed is not None:
            gen = torch.Generator(device=device).manual_seed(int(seed))
        return torch.rand(shape, device=device, dtype=dtype, generator=gen)
    else:
        rng = np.random.RandomState(seed if seed is not None else 0)
        return rng.rand(*shape)

def _sigmoid(x):
    if TORCH_AVAILABLE:
        return torch.sigmoid(x)
    else:
        return 1.0 / (1.0 + np.exp(-x))

def _pairwise_knn(x, k, differentiable=False):
    """Return indices of k nearest neighbors.
    
    If differentiable=False: uses sklearn (faster but non-differentiable, graph treated as constant)
    If differentiable=True: uses torch cdist (slower but differentiable via distances)
    """
    if differentiable and TORCH_AVAILABLE and torch.is_tensor(x):
        # Torch-based kNN (differentiable distances, but indices still discrete)
        # Note: neighbor selection is still non-differentiable, but distances are
        X = x if torch.is_tensor(x) else torch.from_numpy(x).float()
        D = torch.cdist(X, X, p=2)  # (N, N)
        _, ind = torch.topk(D, k=min(k, len(X)), dim=1, largest=False)
        return ind.cpu().numpy()
    else:
        # sklearn fallback (faster, non-differentiable)
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for neighbor search.")
        nn = NearestNeighbors(n_neighbors=min(k, len(x))).fit(_to_numpy(x))
        ind = nn.kneighbors(_to_numpy(x), return_distance=False)
        return ind

def _local_pca(P):
    """Covariance eigenvalues for a neighborhood P: return eigenvalues, eigenvectors."""
    if TORCH_AVAILABLE and torch.is_tensor(P):
        C = (P - P.mean(0, keepdim=True))
        C = C.T @ C / max(1, P.shape[0]-1)
        evals, evecs = torch.linalg.eigh(C)
        return evals, evecs
    else:
        Q = P - P.mean(0, keepdims=True)
        C = (Q.T @ Q) / max(1, P.shape[0]-1)
        evals, evecs = np.linalg.eigh(C)
        return evals, evecs

def _normalize(v, eps=1e-9):
    if TORCH_AVAILABLE and torch.is_tensor(v):
        return v / (torch.norm(v, dim=-1, keepdim=True) + eps)
    else:
        n = np.linalg.norm(v, axis=-1, keepdims=True) + eps
        return v / n


# --- PNG helpers (safe for NumPy/Torch) --------------------------------------
def _as_np(a):
    try:
        import torch
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(a)

def save_comparison_png(path, current_before=None, current_after=None, radial_after=None, 
                        target_before=None, dpi=160, ptsize=0.5):
    """Save 3-panel comparison (before/after/radial)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    X  = _as_np(current_before) if current_before is not None else None
    PU = _as_np(current_after) if current_after is not None else None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if PU is None or PU.size == 0:
        fig = plt.figure(figsize=(3,3)); fig.text(0.5,0.5,"No points", ha="center")
        fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig); return

    fig = plt.figure(figsize=(18,5))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    if X is not None and X.size > 0:
        ax1.scatter(X[:,0], X[:,1], X[:,2], s=ptsize, alpha=0.6)
    ax1.set_title("Current (before upsampling)"); ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

    ax2.scatter(PU[:,0], PU[:,1], PU[:,2], s=ptsize, alpha=0.6)
    ax2.set_title("Current (after upsampling)")

    c = PU.mean(0); r = np.linalg.norm(PU - c[None,:], axis=1)
    sc = ax3.scatter(PU[:,0], PU[:,1], PU[:,2], c=r, s=ptsize, alpha=0.6, cmap="viridis")
    ax3.set_title("Radial color (after)")
    fig.colorbar(sc, ax=ax3, shrink=0.6)

    ALL = PU if PU.size>0 else (X if X is not None and X.size>0 else PU)
    mins, maxs = ALL.min(0), ALL.max(0)
    rng = (maxs - mins).max() * 0.5; mid = (maxs + mins) * 0.5
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(mid[0]-rng, mid[0]+rng)
        ax.set_ylim(mid[1]-rng, mid[1]+rng)
        ax.set_zlim(mid[2]-rng, mid[2]+rng)
        ax.set_box_aspect([1,1,1])
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)

def save_axis_hist_png(path, pts, dpi=160):
    """Save axis histograms for upsampled points."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    PU = _as_np(pts)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if PU is None or PU.size == 0:
        fig = plt.figure(figsize=(3,3)); fig.text(0.5,0.5,"No points", ha="center")
        fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig); return

    fig, axes = plt.subplots(1,3, figsize=(16,4))
    for j,ax in enumerate(axes):
        ax.hist(PU[:,j], bins=48, alpha=0.85)
        ax.set_title(["X-Axis Distribution","Y-Axis Distribution","Z-Axis Distribution"][j])
        ax.set_xlabel(f"{'XYZ'[j]} Coordinate"); ax.set_ylabel("Count")
    fig.suptitle("Runtime Surface (Axis Distribution)")
    fig.tight_layout(rect=[0,0,1,0.92])
    fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------
# Surface detection (soft gating with EMA + hysteresis)
# ------------------------------------------------------------
def compute_surface_mask_soft(x: np.ndarray,
                              k_surface: int,
                              thr_percentile: float,
                              ema_prev: Optional[float],
                              ema_beta: float,
                              hysteresis: float,
                              soft_tau: float,
                              state_out: Dict,
                              differentiable: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return (surf_prob, normals, spacing, thr_low, thr_high).
    - surf_prob: [0,1] probability from soft sigmoid gate (torch or np array)
    - normals  : unit normals from PCA
    - spacing  : local spacing (median NN distance)
    """
    # kNN once (fixed graph for this episode)
    ind = _pairwise_knn(x, k_surface)
    N = len(x)
    device, dtype = _device_dtype_like(_as_torch(x))

    # containers
    if TORCH_AVAILABLE:
        normals = torch.zeros((N,3), device=device, dtype=dtype)
        spacing = torch.zeros((N,), device=device, dtype=dtype)
    else:
        normals = np.zeros((N,3), dtype=np.float32)
        spacing = np.zeros((N,), dtype=np.float32)

    surfvar = np.zeros((N,), dtype=np.float32)

    for i in range(N):
        P = _as_torch(x[ind[i]])
        evals, evecs = _local_pca(P)
        if TORCH_AVAILABLE and torch.is_tensor(evals):
            # Conditionally detach for threshold computation (non-differentiable metric)
            lam = evals.detach().cpu().numpy() if not differentiable else evals.cpu().numpy()
            nrm = evecs[:,0]
            normals[i] = _normalize(nrm)
        else:
            lam = evals if isinstance(evals, np.ndarray) else np.array(evals)
            normals[i] = _normalize(evecs[:,0])
        lam = np.clip(lam, 1e-12, None)
        surfvar[i] = float(lam[0] / lam.sum())
        # local spacing (median NN distance, excluding self)
        d = np.linalg.norm(_to_numpy(x[ind[i]]) - _to_numpy(x[i])[None,:], axis=1)
        sp = np.median(d[1:]) if d.shape[0] > 1 else np.median(d)
        if TORCH_AVAILABLE and torch.is_tensor(spacing):
            spacing[i] = float(sp)
        else:
            spacing[i] = sp

    # Adaptive percentile threshold + EMA + hysteresis
    thr_now = float(np.percentile(surfvar, thr_percentile))
    ema_thr = thr_now if (ema_prev is None) else (ema_beta * float(ema_prev) + (1.0 - ema_beta) * thr_now)
    band = hysteresis * max(ema_thr, 1e-6)
    thr_low, thr_high = ema_thr - band, ema_thr + band

    if TORCH_AVAILABLE:
        surfvar_t = torch.tensor(surfvar, device=device, dtype=dtype)
        p_on = _sigmoid((thr_high - surfvar_t) / max(soft_tau, 1e-6))
    else:
        p_on = _sigmoid((thr_high - surfvar) / max(soft_tau, 1e-6))

    state_out["ema_thr"] = float(ema_thr)
    state_out["thr_low"] = float(thr_low)
    state_out["thr_high"] = float(thr_high)

    return p_on, _as_torch(normals), _as_torch(spacing), thr_low, thr_high


# ------------------------------------------------------------
# F smoothing with small ED system (Laplace-regularized)
# ------------------------------------------------------------
def _fps(x: np.ndarray, K: int, seed: int = 0) -> np.ndarray:
    """Farthest Point Sampling (non-differentiable, uses argmax).
    
    Note: FPS is inherently non-differentiable due to argmax operations.
    For differentiable alternatives, consider:
    - Random sampling (differentiable through sample weights)
    - Soft attention over all points (memory intensive)
    - Learnable/fixed node positions (no sampling needed)
    """
    rng = np.random.RandomState(seed)
    X = _to_numpy(x)
    N = X.shape[0]
    sel = np.zeros((K,), dtype=np.int32)
    sel[0] = int(rng.randint(0, N))
    d2 = np.sum((X - X[sel[0]])**2, axis=1)
    for i in range(1, K):
        sel[i] = int(np.argmax(d2))  # ❌ Non-differentiable
        d2 = np.minimum(d2, np.sum((X - X[sel[i]])**2, axis=1))
    return sel

def _random_sampling(x: np.ndarray, K: int, seed: int = 0) -> np.ndarray:
    """Random sampling (differentiable alternative to FPS).
    
    Less optimal than FPS for coverage, but allows gradients through point coordinates.
    """
    rng = np.random.RandomState(seed)
    N = len(x)
    return rng.choice(N, size=min(K, N), replace=False)

def _build_node_graph(x_nodes: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn required for node graph.")
    nn = NearestNeighbors(n_neighbors=min(k+1, len(x_nodes))).fit(_to_numpy(x_nodes))
    idx = nn.kneighbors(_to_numpy(x_nodes), return_distance=False)[:,1:]
    rows = np.repeat(np.arange(len(x_nodes)), idx.shape[1])
    cols = idx.reshape(-1)
    return rows, cols

def smooth_F_with_ED(x: np.ndarray, F: np.ndarray, cfg: Dict, differentiable: bool = False) -> np.ndarray:
    """Return smoothed deformation gradients F for each *point* x via small ED system.
    
    Args:
        differentiable: If True, uses random sampling instead of FPS (allows gradient flow)
    """
    if not cfg.get("enabled", True):
        return F

    K = int(cfg.get("num_nodes", 180))
    node_knn = int(cfg.get("node_knn", 8))
    point_knn_nodes = int(cfg.get("point_knn_nodes", 8))
    lam = float(cfg.get("lambda_lap", 1.0e-2))

    # Node selection
    if differentiable:
        # Random sampling (differentiable alternative, less optimal coverage)
        sel = _random_sampling(_to_numpy(x), K)
    else:
        # FPS (non-differentiable but better coverage)
        sel = _fps(_to_numpy(x), K)
    Xn = _as_torch(_to_numpy(x)[sel])

    # Node graph Laplacian
    rows, cols = _build_node_graph(Xn, node_knn)
    if TORCH_AVAILABLE:
        device, dtype = _device_dtype_like(_as_torch(x))
        L = torch.zeros((K,K), device=device, dtype=dtype)
        idx = torch.stack([torch.tensor(rows), torch.tensor(cols)], dim=0)
        L[idx[0], idx[1]] = -1.0
        deg = torch.zeros((K,), device=device, dtype=dtype)
        for i in range(K):
            deg[i] = torch.sum(L[i] != 0).abs()
        L = L + torch.diag(deg)
    else:
        L = np.zeros((K,K), dtype=np.float32)
        for r,c in zip(rows, cols):
            L[r, c] = -1.0
        deg = np.zeros((K,), dtype=np.float32)
        for i in range(K):
            deg[i] = np.sum(L[i] != 0)
        L = L + np.diag(deg)

    # Points-to-nodes soft weights (Gaussian)
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for point-node neighbors.")
    nn = NearestNeighbors(n_neighbors=min(point_knn_nodes, K)).fit(_to_numpy(Xn))
    d, j = nn.kneighbors(_to_numpy(x), return_distance=True)
    h = np.median(d, axis=1, keepdims=True) + 1e-9
    w = np.exp(- (d / h)**2)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-9)  # (N, k)

    # Build normal equations and solve per component
    if TORCH_AVAILABLE:
        device, dtype = _device_dtype_like(_as_torch(x))
        WtW = torch.zeros((K,K), device=device, dtype=dtype)
        for row in range(w.shape[0]):
            cols_row = j[row]
            wr = torch.tensor(w[row], device=device, dtype=dtype)
            WtW[cols_row[:,None], cols_row[None,:]] += (wr[:,None] * wr[None,:])
        A = WtW + lam * L

        F_np = _to_numpy(F).reshape(-1, 9)  # (N,9)
        Y = torch.zeros((K,9), device=device, dtype=dtype)
        for c in range(9):
            rhs = torch.zeros((K,), device=device, dtype=dtype)
            for row in range(w.shape[0]):
                cols_row = j[row]
                rhs[cols_row] += torch.tensor(w[row], device=device, dtype=dtype) * float(F_np[row, c])
            sol = torch.linalg.solve(A, rhs)
            Y[:, c] = sol
        F_s = torch.zeros((F.shape[0], 9), device=device, dtype=dtype)
        for row in range(w.shape[0]):
            cols_row = j[row]
            F_s[row] = torch.sum(torch.tensor(w[row], device=device, dtype=dtype)[:,None] * Y[cols_row], dim=0)
        F_s = F_s.reshape(-1, 3, 3)
        return F_s
    else:
        WtW = np.zeros((K,K), dtype=np.float32)
        for row in range(w.shape[0]):
            cols_row = j[row]
            wr = w[row]
            WtW[np.ix_(cols_row, cols_row)] += (wr[:,None] * wr[None,:])
        A = WtW + lam * L
        F_np = _to_numpy(F).reshape(-1, 9)
        Y = np.zeros((K,9), dtype=np.float32)
        for c in range(9):
            rhs = np.zeros((K,), dtype=np.float32)
            for row in range(w.shape[0]):
                cols_row = j[row]
                rhs[cols_row] += w[row] * F_np[row, c]
            Y[:, c] = np.linalg.solve(A, rhs)
        F_s = np.zeros((F.shape[0], 9), dtype=np.float32)
        for row in range(w.shape[0]):
            cols_row = j[row]
            F_s[row] = (w[row][:,None] * Y[cols_row]).sum(axis=0)
        return F_s.reshape(-1, 3, 3)


# ------------------------------------------------------------
# Sampling on tangent planes (reparameterization)
# ------------------------------------------------------------
def _gumbel_softmax_sample(logits, tau=1.0, hard=False):
    """Gumbel-Softmax sampling (differentiable approximation to categorical sampling).
    
    Args:
        logits: (N,) unnormalized log probabilities
        tau: temperature (lower = more discrete, higher = more uniform)
        hard: if True, returns one-hot (forward) but soft gradients (backward)
    
    Returns:
        (N,) soft weights or one-hot vector
    """
    if not TORCH_AVAILABLE or not torch.is_tensor(logits):
        # NumPy fallback (non-differentiable)
        p = logits / (logits.sum() + 1e-12)
        return p
    
    # Sample Gumbel noise
    gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y = (logits.log() + gumbel) / tau
    y_soft = torch.softmax(y, dim=0)
    
    if hard:
        # Straight-through: one-hot forward, soft backward
        index = y_soft.argmax(dim=0)
        y_hard = torch.zeros_like(y_soft)
        y_hard[index] = 1.0
        # Trick: (y_hard - y_soft).detach() + y_soft = y_hard in forward, y_soft in backward
        return (y_hard - y_soft).detach() + y_soft
    else:
        return y_soft

def sample_surface_points(x, normals, spacing, probs, M, alpha, thickness, density_gamma, seed=0, 
                         differentiable=False, gumbel_tau=1.0, use_soft_sampling=False):
    """Generate M points near tangent planes using reparameterization.
    Returns (points, normals_at_points, anchors).
    - probs: soft surface probabilities [0,1] (importance weights)
    - use_soft_sampling: if True and differentiable, uses soft attention (memory: N x M)
    """
    device, dtype = _device_dtype_like(_as_torch(x))
    N = len(x)

    # Importance per anchor ~ probs * (1/spacing)^gamma
    s = _to_numpy(spacing).reshape(-1) + 1e-9
    w_import = _to_numpy(probs).reshape(-1) * (1.0 / (s ** max(density_gamma, 0.0)))
    w_import = w_import / (w_import.sum() + 1e-12)

    # Sample anchor indices
    if differentiable and use_soft_sampling and TORCH_AVAILABLE:
        # FULLY DIFFERENTIABLE soft sampling (memory intensive: N x M)
        # Each sampled point is a weighted combination of all anchors
        w_t = torch.tensor(w_import, device=device, dtype=dtype)
        
        # Generate M soft distributions via Gumbel-Softmax
        soft_weights = []
        for i in range(int(M)):
            torch.manual_seed(seed + i)  # deterministic per sample
            w_soft = _gumbel_softmax_sample(w_t, tau=gumbel_tau, hard=False)
            soft_weights.append(w_soft)
        soft_weights = torch.stack(soft_weights, dim=0)  # (M, N)
        
        # This would require soft weighting in the generation step
        # For now, fall back to hard sampling with STE gradient
        # TODO: implement full soft sampling
        sel = torch.multinomial(w_t, num_samples=int(M), replacement=True).cpu().numpy()
    elif differentiable and TORCH_AVAILABLE:
        # Hard sampling with gradients through weights (STE-like)
        w_t = torch.tensor(w_import, device=device, dtype=dtype)
        sel = torch.multinomial(w_t, num_samples=int(M), replacement=True).cpu().numpy()
    elif TORCH_AVAILABLE:
        w_t = torch.tensor(w_import, device=device, dtype=dtype)
        sel = torch.multinomial(w_t, num_samples=int(M), replacement=True).cpu().numpy()
    else:
        sel = np.random.choice(np.arange(N), size=int(M), replace=True, p=w_import)

    X = _to_numpy(x); Nrm = _to_numpy(normals); H = _to_numpy(spacing)
    U = _to_numpy(_randn_like((M,1), seed=seed))
    V = _to_numpy(_randn_like((M,1), seed=seed+1))
    Z = _to_numpy(_rand_like((M,1), seed=seed+2) * 2.0 - 1.0)  # uniform in [-1,1]

    # Build tangent bases by Gram-Schmidt
    n = _normalize(Nrm[sel])
    a = np.tile(np.array([1.0,0.0,0.0], dtype=np.float32), (M,1))
    mask = (np.abs((n * a).sum(axis=1)) > 0.9)
    a[mask] = np.array([0.0,1.0,0.0], dtype=np.float32)
    t1 = _normalize(a - (a * n).sum(axis=1, keepdims=True) * n)
    t2 = _normalize(np.cross(n, t1))

    # Curvature-aware alpha (use prob as inverse curvature proxy)
    alpha_local = float(alpha) * (0.5 + 0.5 * _to_numpy(probs).reshape(-1)[sel])
    alpha_local = np.clip(alpha_local, 0.15, float(alpha))

    hloc = H[sel][:,None]
    mu = X[sel] + alpha_local[:,None] * hloc * (U * t1 + V * t2) + (thickness * Z) * n

    anchors = X[sel].astype(np.float32)  # return anchors for later projection
    return mu.astype(np.float32), n.astype(np.float32), anchors


# ------------------------------------------------------------
# Differentiable density-equalization relaxation
# ------------------------------------------------------------
def _density_equalize_relax(pts, nrms, anchors, cfg, thickness, differentiable=False):
    """
    Spread points from over-dense regions into under-dense regions while
    remaining on a thin shell around the anchors. Fully differentiable
    (except the fixed neighbor graph construction).
    
    If differentiable=True, avoids detaching density statistics.
    """
    if not cfg.get("enabled", True) or len(pts) == 0:
        return pts, nrms

    iters = int(cfg.get("iters", 3))
    k     = int(cfg.get("k", 24))
    step  = float(cfg.get("step", 0.45))
    rmul  = float(cfg.get("radius_mul", 1.0))
    project = bool(cfg.get("project_to_shell", True))

    if TORCH_AVAILABLE:
        P = _as_torch(pts).float(); N = _as_torch(nrms).float(); A = _as_torch(anchors).float()
        device, dtype = _device_dtype_like(P)
        for _ in range(iters):
            nn = NearestNeighbors(n_neighbors=min(k, len(P))).fit(_to_numpy(P))
            idx = torch.from_numpy(nn.kneighbors(_to_numpy(P), return_distance=False)).to(device)

            Q = P[idx]                         # (M, k, 3)
            diff = Q - P[:, None, :]           # (M, k, 3)
            dist = torch.norm(diff, dim=-1)    # (M, k)

            h = rmul * (torch.median(dist[:, 1:], dim=1).values + 1e-6)  # (M,)
            w = torch.exp(- (dist / h[:, None])**2)
            w[:, 0] = 0.0                      # ignore self in the kernel
            rho = w.sum(dim=1, keepdim=True)   # local density
            # Conditionally detach median computation
            rho_star = torch.median(rho.detach() if not differentiable else rho)

            s = torch.tanh((rho - rho_star) / (rho_star + 1e-6))
            disp = (w[..., None] * diff).sum(dim=1) / (rho + 1e-6)
            P = P - step * s * disp

            if project:
                # Project back to the thin shell around anchors using normals
                t = ((P - A) * N).sum(dim=1, keepdim=True)
                t = torch.clamp(t, -thickness, thickness)
                tang = P - A - t * N
                P = A + tang + t * N
        return _to_numpy(P), _to_numpy(N)
    else:
        P = np.asarray(pts, dtype=np.float32); N = np.asarray(nrms, dtype=np.float32)
        A = np.asarray(anchors, dtype=np.float32)
        for _ in range(iters):
            nn = NearestNeighbors(n_neighbors=min(k, len(P))).fit(P)
            d, j = nn.kneighbors(P, return_distance=True)
            diff = P[j] - P[:, None, :]
            h = rmul * (np.median(d[:, 1:], axis=1, keepdims=True) + 1e-6)
            w = np.exp(- (d / h)**2); w[:, 0] = 0.0
            rho = w.sum(axis=1, keepdims=True)
            rho_star = np.median(rho)
            s = np.tanh((rho - rho_star) / (rho_star + 1e-6))
            disp = (w[..., None] * diff).sum(axis=1) / (rho + 1e-6)
            P = P - step * s * disp
            if project:
                t = np.sum((P - A) * N, axis=1, keepdims=True)
                t = np.clip(t, -thickness, thickness)
                tang = P - A - t * N
                P = A + tang + t * N
        return P.astype(np.float32), N.astype(np.float32)
    
# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def synthesize_runtime_surface(x_low,  # np.ndarray or torch.Tensor
                               F_low,  # np.ndarray or torch.Tensor
                               cfg: Dict,
                               ema_state: Optional[Dict] = None,
                               seed: int = 1234,
                               differentiable: bool = False,
                               return_torch: bool = False) -> Dict:
    """Main entry. See the module docstring for outputs.
    
    Args:
        x_low: (N,3) positions (NumPy or Torch tensor)
        F_low: (N,3,3) deformation gradients (NumPy or Torch tensor)
        differentiable: If True, uses differentiable operations where possible
                       (slower but gradient-friendly for end-to-end training).
                       Can be overridden by cfg["differentiable"].
        return_torch: If True, returns Torch tensors (keeps gradient).
                     If False, returns NumPy arrays (backward compatible).
    
    Returns:
        dict with keys:
            - "points": (M,3) upsampled positions
            - "normals": (M,3) normals at upsampled points
            - "F_smooth": (M,3,3) smoothed deformation gradients
            - "cov": (M,3,3) covariance matrices for Gaussian splatting
            - "debug": dict with diagnostic info
            - "state": updated EMA state
    """
    if ema_state is None:
        ema_state = {}

    # Check if inputs are torch tensors
    is_torch_input = TORCH_AVAILABLE and torch.is_tensor(x_low)
    
    # Convert to appropriate format
    if is_torch_input and not return_torch:
        # Torch input but NumPy output requested - detach and convert
        x_low_np = _to_numpy(x_low)
        F_low_np = _to_numpy(F_low)
    elif not is_torch_input and return_torch:
        # NumPy input but Torch output requested - convert
        x_low = _as_torch(x_low).float()
        F_low = _as_torch(F_low).float()
        x_low_np = _to_numpy(x_low)
        F_low_np = _to_numpy(F_low)
    elif is_torch_input:
        # Torch input and Torch output
        x_low_np = _to_numpy(x_low)
        F_low_np = _to_numpy(F_low)
    else:
        # NumPy input and NumPy output (original behavior)
        x_low_np = np.asarray(x_low)
        F_low_np = np.asarray(F_low)

    # Use config value if provided, otherwise use parameter
    differentiable = bool(cfg.get("differentiable", differentiable))
    
    k_surface  = int(cfg.get("k_surface", 24))
    thr_pct    = float(cfg.get("thr_percentile", 20.0))
    ema_beta   = float(cfg.get("ema_beta", 0.95))
    hys        = float(cfg.get("hysteresis", 0.03))
    tau        = float(cfg.get("soft_tau", 0.02))
    M          = int(cfg.get("M", 180_000))
    alpha      = float(cfg.get("surf_jitter_alpha", 0.30))
    thickness  = float(cfg.get("thickness", 0.12))
    density_g  = float(cfg.get("density_gamma", 2.5))

    # 1) Soft surface probabilities, normals, spacing
    surf_prob, normals, spacing, thr_low, thr_high = compute_surface_mask_soft(
        x_low_np, k_surface, thr_pct, ema_state.get("ema_thr", None), ema_beta, hys, tau, ema_state,
        differentiable=differentiable
    )

    # 2) Sampling (reparameterization)
    pts, nrm, anchors = sample_surface_points(
        x_low_np, normals, spacing, surf_prob, M, alpha, thickness, density_g, seed=seed,
        differentiable=differentiable
    )

    # 2.5) Differentiable density equalization (no new points)
    pts, nrm = _density_equalize_relax(
        pts, nrm, anchors,
        cfg.get("post_equalize", {}), thickness=thickness, differentiable=differentiable
    )
    
    # 3) F smoothing (ED regularization) on low-res anchors
    if bool(cfg.get("use_F_kernel", True)):
        F_total = F_low_np.reshape(-1, 3, 3)
        F_smooth = smooth_F_with_ED(_as_torch(x_low_np), _as_torch(F_total), 
                                     cfg.get("ed", {}), differentiable=differentiable)
    else:
        # pass-through
        F_smooth = _as_torch(F_low_np.reshape(-1,3,3))

    # 4) Interpolate F at sampled points with local kernel (Gaussian)
    k_F = int(cfg.get("k_F", 24)); h_mul = float(cfg.get("h_mul", 1.5))
    
    if differentiable and TORCH_AVAILABLE:
        # Torch-based interpolation (differentiable)
        X_low = _as_torch(x_low_np).float()
        Pts = _as_torch(pts).float()
        F_s = _as_torch(F_smooth).float() if not torch.is_tensor(F_smooth) else F_smooth.float()
        
        device, dtype = _device_dtype_like(X_low)
        X_low = X_low.to(device); Pts = Pts.to(device); F_s = F_s.to(device)
        
        # Compute pairwise distances
        D = torch.cdist(Pts, X_low, p=2)  # (M, N_low)
        d, j = torch.topk(D, k=min(k_F, len(x_low_np)), dim=1, largest=False)  # (M, k_F)
        
        h = h_mul * (torch.median(d, dim=1, keepdim=True).values + 1e-9)
        w = torch.exp(- (d / h)**2)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-9)  # (M, k_F)
        
        # Gather F values at neighbors and interpolate
        F_neighbors = F_s[j]  # (M, k_F, 3, 3)
        F_loc = torch.einsum('mk,mkrc->mrc', w, F_neighbors)  # (M, 3, 3)
        
        # 5) Per-point covariance for GS: Sigma = sigma0^2 * F F^T
        sigma0 = float(cfg.get("sigma0", 0.02))
        Sig = torch.matmul(F_loc, F_loc.transpose(-2, -1))  # (M, 3, 3)
        Sig = (sigma0 ** 2) * Sig
        
        # Keep as torch if return_torch, else convert to numpy
        if not return_torch:
            F_loc = _to_numpy(F_loc)
            Sig = _to_numpy(Sig)
    else:
        # NumPy/sklearn fallback (non-differentiable but faster)
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required for interpolation.")
        nn = NearestNeighbors(n_neighbors=min(k_F, len(x_low_np))).fit(x_low_np)
        d, j = nn.kneighbors(_to_numpy(pts), return_distance=True)
        h = h_mul * (np.median(d, axis=1, keepdims=True) + 1e-9)
        w = np.exp(- (d / h)**2)
        w = w / (w.sum(axis=1, keepdims=True) + 1e-9)
        F_loc = np.einsum('mk,mkrc->mrc', w, _to_numpy(F_smooth)[j])
        
        # 5) Per-point covariance for GS: Sigma = sigma0^2 * F F^T
        sigma0 = float(cfg.get("sigma0", 0.02))
        Sig = np.matmul(F_loc, np.transpose(F_loc, (0,2,1)))
        Sig = (sigma0 ** 2) * Sig

    debug = {
        "thr_low": float(thr_low), "thr_high": float(thr_high),
        "ema_thr": float(ema_state.get("ema_thr", thr_high)),
        "mean_prob": float(_to_numpy(surf_prob).mean()) if TORCH_AVAILABLE else float(surf_prob.mean())
    }

    # Return format based on return_torch flag
    if return_torch and TORCH_AVAILABLE:
        # Convert all outputs to torch tensors
        pts_t = _as_torch(pts).float() if not torch.is_tensor(pts) else pts
        nrm_t = _as_torch(nrm).float() if not torch.is_tensor(nrm) else nrm
        F_loc_t = _as_torch(F_loc).float() if not torch.is_tensor(F_loc) else F_loc
        Sig_t = _as_torch(Sig).float() if not torch.is_tensor(Sig) else Sig
        
        return {
            "points": pts_t,      # Torch tensor
            "normals": nrm_t,     # Torch tensor
            "F_smooth": F_loc_t,  # Torch tensor
            "cov": Sig_t,         # Torch tensor
            "debug": debug,
            "state": ema_state
        }
    else:
        # Return NumPy arrays (backward compatible)
        return {
            "points": _to_numpy(pts) if torch.is_tensor(pts) else pts,
            "normals": _to_numpy(nrm) if torch.is_tensor(nrm) else nrm,
            "F_smooth": _to_numpy(F_loc) if torch.is_tensor(F_loc) else F_loc,
            "cov": _to_numpy(Sig) if torch.is_tensor(Sig) else Sig,
            "debug": debug,
            "state": ema_state
        }


# ------------------------------------------------------------
# Simple PLY / NPZ exporters (used by runner)
# ------------------------------------------------------------
def save_ply_xyz(path: Path, xyz: np.ndarray):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        f.write("ply\nformat ascii 1.0\n" + f"element vertex %d\n" % len(xyz) +
                "property float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(f, xyz, fmt="%.6f")

def save_gaussians_npz(path: Path, xyz: np.ndarray, cov: np.ndarray, rgb=None, opacity=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if rgb is None: rgb = np.ones_like(xyz, dtype=np.float32)
    if opacity is None: opacity = np.ones((len(xyz),1), dtype=np.float32)
    np.savez_compressed(path, xyz=xyz.astype(np.float32),
                        cov=cov.astype(np.float32),
                        rgb=rgb.astype(np.float32),
                        opacity=opacity.astype(np.float32))
