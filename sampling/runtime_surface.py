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

Public API used by the runner:
- default_cfg()
- synthesize_runtime_surface(...)
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

def _pairwise_knn(x, k):
    """Return indices of k nearest neighbors using sklearn.
    The neighbor graph is treated as constant for backprop (standard in GCN/ED)."""
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
                              state_out: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
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
            lam = evals.detach().cpu().numpy()
            nrm = evecs[:,0]
            normals[i] = _normalize(nrm)
        else:
            lam = evals
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
    """Farthest Point Sampling (NumPy)."""
    rng = np.random.RandomState(seed)
    X = _to_numpy(x)
    N = X.shape[0]
    sel = np.zeros((K,), dtype=np.int32)
    sel[0] = int(rng.randint(0, N))
    d2 = np.sum((X - X[sel[0]])**2, axis=1)
    for i in range(1, K):
        sel[i] = int(np.argmax(d2))
        d2 = np.minimum(d2, np.sum((X - X[sel[i]])**2, axis=1))
    return sel

def _build_node_graph(x_nodes: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn required for node graph.")
    nn = NearestNeighbors(n_neighbors=min(k+1, len(x_nodes))).fit(_to_numpy(x_nodes))
    idx = nn.kneighbors(_to_numpy(x_nodes), return_distance=False)[:,1:]
    rows = np.repeat(np.arange(len(x_nodes)), idx.shape[1])
    cols = idx.reshape(-1)
    return rows, cols

def smooth_F_with_ED(x: np.ndarray, F: np.ndarray, cfg: Dict) -> np.ndarray:
    """Return smoothed deformation gradients F for each *point* x via small ED system."""
    if not cfg.get("enabled", True):
        return F

    K = int(cfg.get("num_nodes", 180))
    node_knn = int(cfg.get("node_knn", 8))
    point_knn_nodes = int(cfg.get("point_knn_nodes", 8))
    lam = float(cfg.get("lambda_lap", 1.0e-2))

    # Node selection by FPS (non-differentiable selection; typical for ED)
    sel = _fps(x, K)
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
def sample_surface_points(x, normals, spacing, probs, M, alpha, thickness, density_gamma, seed=0):
    """Generate M points near tangent planes using reparameterization.
    Returns (points, normals_at_points, anchors).
    - probs: soft surface probabilities [0,1] (importance weights)
    """
    device, dtype = _device_dtype_like(_as_torch(x))
    N = len(x)

    # Importance per anchor ~ probs * (1/spacing)^gamma
    s = _to_numpy(spacing).reshape(-1) + 1e-9
    w_import = _to_numpy(probs).reshape(-1) * (1.0 / (s ** max(density_gamma, 0.0)))
    w_import = w_import / (w_import.sum() + 1e-12)

    # Sample anchor indices
    if TORCH_AVAILABLE:
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
def _density_equalize_relax(pts, nrms, anchors, cfg, thickness):
    """
    Spread points from over-dense regions into under-dense regions while
    remaining on a thin shell around the anchors. Fully differentiable
    (except the fixed neighbor graph construction).
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
            rho_star = torch.median(rho.detach())

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
# Optional discrete hole fill (kept for completeness)
# ------------------------------------------------------------
def _hole_fill_voxel(pts: np.ndarray, nrms: np.ndarray, cfg: Dict, seed: int = 0):
    """Voxel-based completion (DISCRETE; disabled by default)."""
    if not cfg.get("enabled", False):
        return pts, nrms
    dx = float(cfg.get("grid_dx", 0.25))
    min_pts = int(cfg.get("min_points_per_cell", 2))
    add_per = int(cfg.get("add_per_empty", 1))
    if len(pts) == 0:
        return pts, nrms

    lo = pts.min(0) - 1e-3
    hi = pts.max(0) + 1e-3
    dims = np.maximum(1, np.ceil((hi - lo) / dx).astype(int))

    g = np.floor((pts - lo[None,:]) / dx).astype(int)
    g = np.clip(g, 0, dims-1)
    hashv = g[:,0] + dims[0]*(g[:,1] + dims[1]*g[:,2])
    unique, counts = np.unique(hashv, return_counts=True)

    target_cells = unique[counts < min_pts]
    if target_cells.size == 0:
        return pts, nrms

    rng = np.random.RandomState(seed)
    add_pts, add_nrms = [], []
    if SKLEARN_AVAILABLE:
        nn = NearestNeighbors(n_neighbors=1).fit(pts)
    for h in target_cells:
        z = h // (dims[0]*dims[1])
        y = (h - z*dims[0]*dims[1]) // dims[0]
        x = h - z*dims[0]*dims[1] - y*dims[0]
        center = lo + dx*(np.array([x+0.5, y+0.5, z+0.5]))
        for _ in range(add_per):
            jitter = (rng.rand(3)-0.5) * (0.25*dx)
            p = center + jitter
            add_pts.append(p)
            if SKLEARN_AVAILABLE:
                _, j = nn.kneighbors(p.reshape(1,3), return_distance=True)
                add_nrms.append(nrms[j[0][0]])
            else:
                add_nrms.append(np.array([0,0,1.0], dtype=np.float32))
    if len(add_pts) > 0:
        pts = np.vstack([pts, np.array(add_pts, dtype=np.float32)])
        nrms = np.vstack([nrms, np.array(add_nrms, dtype=np.float32)])
    return pts, nrms


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def synthesize_runtime_surface(x_low: np.ndarray,
                               F_low: np.ndarray,
                               cfg: Dict,
                               ema_state: Optional[Dict] = None,
                               seed: int = 1234) -> Dict:
    """Main entry. See the module docstring for outputs."""
    if ema_state is None:
        ema_state = {}

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
        x_low, k_surface, thr_pct, ema_state.get("ema_thr", None), ema_beta, hys, tau, ema_state
    )

    # 2) Sampling (reparameterization)
    pts, nrm, anchors = sample_surface_points(
        x_low, normals, spacing, surf_prob, M, alpha, thickness, density_g, seed=seed
    )

    # 2.5) Differentiable density equalization (no new points)
    pts, nrm = _density_equalize_relax(
        pts, nrm, anchors,
        cfg.get("post_equalize", {}), thickness=thickness
    )

    # Optional discrete hole filling (disabled by default; breaks differentiability)
    pts, nrm = _hole_fill_voxel(pts, nrm, cfg.get('hole_filling', {}), seed=seed)

    # 3) F smoothing (ED regularization) on low-res anchors
    if bool(cfg.get("use_F_kernel", True)):
        F_total = F_low.reshape(-1, 3, 3)
        F_smooth = smooth_F_with_ED(_as_torch(x_low), _as_torch(F_total), cfg.get("ed", {}))
    else:
        # pass-through
        F_smooth = _as_torch(F_low.reshape(-1,3,3))

    # 4) Interpolate F at sampled points with local kernel (Gaussian)
    k_F = int(cfg.get("k_F", 24)); h_mul = float(cfg.get("h_mul", 1.5))
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn required for interpolation.")
    nn = NearestNeighbors(n_neighbors=min(k_F, len(x_low))).fit(_to_numpy(x_low))
    d, j = nn.kneighbors(pts, return_distance=True)
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

    return {
        "points": pts, "normals": nrm,
        "F_smooth": F_loc, "cov": Sig,
        "debug": debug, "state": ema_state
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
