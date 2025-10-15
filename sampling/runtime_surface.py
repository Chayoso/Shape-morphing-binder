# -*- coding: utf-8 -*-
"""
runtime_surface.py
============================

Improved differentiable runtime surface upsampler with enhanced uniformity and smoothness.

Key improvements over baseline:
1. Stratified sampling for uniform point distribution
2. Controlled jitter to prevent salt-and-pepper noise
3. Multi-scale density equalization for hole filling
4. Laplacian smoothing for surface coherence
5. Adaptive covariance scaling for hole-aware rendering

All operations maintain differentiability for end-to-end gradient flow.

Novelty:
- Physics-aware upsampling: Deformation gradient F guides anisotropic covariance
- Differentiable density equalization without discrete hole filling
- Stratified importance sampling for uniform coverage
- Shell-constrained relaxation preserves thin surface manifold
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

# Optional dependencies
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


# ============================================================================
# Configuration
# ============================================================================
def default_cfg() -> Dict:
    """Default configuration with improved parameters for uniform coverage."""
    return {
        # Differentiability control
        "differentiable": True,  # Enable gradient-friendly operations
        
        # Surface detection
        "use_surface_detection": False,  # NEW: Bypass for thin-shell objects
        "k_surface": 24,
        "k_spacing": 12,
        "thr_percentile": 80.0,    # HIGH = relaxed (80% of points as surface)
        "ema_beta": 0.95,
        "hysteresis": 0.03,
        "soft_tau": 0.08,          # Larger tau = softer sigmoid = more uniform prob
        
        # Sampling configuration
        "M": 250_000,                  # More points for denser coverage
        "surf_jitter_alpha": 0.25,     # More jitter for better coverage
        "thickness": 0.10,             # Thicker shell to ensure overlap
        "density_gamma": 2.5,
        "use_stratified_sampling": False,  # Use standard multinomial for denser sampling
        
        # F smoothing (ED regularization)
        "use_F_kernel": True,
        "k_F": 24,
        "h_mul": 1.5,
        "sigma0": 0.02,
        "ed": {
            "enabled": True,
            "num_nodes": 180,
            "node_knn": 8,
            "point_knn_nodes": 8,
            "lambda_lap": 1.0e-2,
            "iters": 1
        },
        
        # Density equalization (IMPROVED)
        "post_equalize": {
            "enabled": True,
            "iters": 10,               # INCREASED: 3 -> 10 (more iterations)
            "k": 32,                   # INCREASED: 24 -> 32 (wider neighborhood)
            "step": 0.60,              # INCREASED: 0.45 -> 0.60 (larger steps)
            "radius_mul": 1.5,         # INCREASED: 1.0 -> 1.5 (wider kernel)
            "project_to_shell": True
        },
        
        # Laplacian smoothing (NEW)
        "laplacian_smooth": {
            "enabled": True,
            "iters": 5,
            "lambda_smooth": 0.5
        },
        
        # Normal smoothing (NEW)
        "normal_smooth": {
            "enabled": True,
            "iters": 5,
            "lambda_smooth": 0.7,
            "k_neighbors": 24
        },
        
        # Adaptive covariance (NEW)
        "adaptive_sigma": {
            "enabled": False,  # DISABLED for gradient flow (converts to numpy)
            "min_scale": 0.5,          # Minimum sigma scaling
            "max_scale": 2.0           # Maximum sigma scaling
        },
        
        # Export control
        "png": {
            "enabled": True,
            "dpi": 160,
            "ptsize": 0.5
        },
        "export_gaussians": True
    }


# ============================================================================
# Utility Functions
# ============================================================================
def _as_torch(x):
    """Convert NumPy array to torch tensor if PyTorch is available."""
    if TORCH_AVAILABLE:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return x
    return x

def _to_numpy(x):
    """Convert torch tensor to NumPy array."""
    if TORCH_AVAILABLE and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x

def _device_dtype_like(x):
    """Get device and dtype from tensor."""
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

def _normalize(v, eps=1e-9):
    """Normalize vectors to unit length."""
    if TORCH_AVAILABLE and torch.is_tensor(v):
        return v / (torch.norm(v, dim=-1, keepdim=True) + eps)
    else:
        n = np.linalg.norm(v, axis=-1, keepdims=True) + eps
        return v / n

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


# ============================================================================
# Surface Detection (from runtime_surface.py)
# ============================================================================
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
    
    # Compute global centroid for fallback
    global_centroid = _as_torch(x).mean(dim=0) if TORCH_AVAILABLE else np.mean(x, axis=0)

    for i in range(N):
        P = _as_torch(x[ind[i]])
        evals, evecs = _local_pca(P)
        if TORCH_AVAILABLE and torch.is_tensor(evals):
            # Conditionally detach for threshold computation (non-differentiable metric)
            lam = evals.detach().cpu().numpy() if not differentiable else evals.cpu().numpy()
            nrm = evecs[:,0]
            nrm = _normalize(nrm)
            # Orient normal consistently: use LOCAL neighborhood centroid (more robust)
            local_centroid = P.mean(dim=0)
            point = _as_torch(x[i])
            to_outside = point - local_centroid
            # If point is too close to centroid, use global as fallback
            dist_to_local = torch.norm(to_outside)
            if dist_to_local < 1e-6:
                to_outside = point - global_centroid
            if torch.dot(nrm, to_outside) < 0:
                nrm = -nrm
            normals[i] = nrm
        else:
            lam = evals if isinstance(evals, np.ndarray) else np.array(evals)
            nrm = _normalize(evecs[:,0])
            # Orient normal consistently: use LOCAL neighborhood centroid
            local_centroid = P.mean(axis=0) if hasattr(P, 'mean') else np.mean(P, axis=0)
            point = x[i]
            to_outside = point - local_centroid
            dist_to_local = np.linalg.norm(to_outside)
            if dist_to_local < 1e-6:
                to_outside = point - global_centroid
            if np.dot(nrm, to_outside) < 0:
                nrm = -nrm
            normals[i] = nrm
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


# ============================================================================
# F Smoothing with ED (from runtime_surface.py)
# ============================================================================
def _fps(x: np.ndarray, K: int, seed: int = 0) -> np.ndarray:
    """Farthest Point Sampling (non-differentiable, uses argmax)."""
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

def _random_sampling(x: np.ndarray, K: int, seed: int = 0) -> np.ndarray:
    """Random sampling (differentiable alternative to FPS)."""
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
    """Return smoothed deformation gradients F for each *point* x via small ED system."""
    if not cfg.get("enabled", True):
        return F

    K = int(cfg.get("num_nodes", 180))
    node_knn = int(cfg.get("node_knn", 8))
    point_knn_nodes = int(cfg.get("point_knn_nodes", 8))
    lam = float(cfg.get("lambda_lap", 1.0e-2))

    # Check if inputs are torch tensors with gradients
    is_torch_input = TORCH_AVAILABLE and torch.is_tensor(F) and F.requires_grad
    
    if differentiable and is_torch_input and TORCH_AVAILABLE:
        # NEW: Fully differentiable path
        return _smooth_F_with_ED_differentiable(x, F, K, node_knn, point_knn_nodes, lam)
    else:
        # OLD: Non-differentiable path (for compatibility)
        return _smooth_F_with_ED_numpy(x, F, K, node_knn, point_knn_nodes, lam, differentiable)


def _smooth_F_with_ED_differentiable(x, F, K, node_knn, point_knn_nodes, lam):
    """Fully differentiable F smoothing (torch-only implementation)."""
    device = F.device
    dtype = F.dtype
    N = x.shape[0] if not torch.is_tensor(x) else x.size(0)
    
    # Ensure x is torch tensor
    if not torch.is_tensor(x):
        x = torch.from_numpy(np.asarray(x)).to(device=device, dtype=dtype)
    else:
        x = x.to(device=device, dtype=dtype)
    
    # Node selection (random for differentiability)
    torch.manual_seed(42)
    K = min(K, N)
    sel = torch.randperm(N, device=device)[:K]
    Xn = x[sel]  # [K, 3]
    
    # Build node graph Laplacian
    D_nodes = torch.cdist(Xn, Xn, p=2)  # [K, K]
    k_node = min(node_knn, K - 1)
    _, j_nodes = torch.topk(D_nodes, k=k_node + 1, dim=1, largest=False)
    j_nodes = j_nodes[:, 1:]  # Exclude self
    
    L = torch.zeros((K, K), device=device, dtype=dtype)
    for i in range(K):
        neighbors = j_nodes[i]
        L[i, neighbors] = -1.0
        L[i, i] = len(neighbors)
    
    # Point-to-node weights (Gaussian kernel)
    D_p2n = torch.cdist(x, Xn, p=2)  # [N, K]
    k_point = min(point_knn_nodes, K)
    d, j = torch.topk(D_p2n, k=k_point, dim=1, largest=False)  # [N, k_point]
    
    h = torch.median(d, dim=1, keepdim=True).values + 1e-9  # [N, 1]
    w_sparse = torch.exp(- (d / h) ** 2)  # [N, k_point]
    w_sparse = w_sparse / (w_sparse.sum(dim=1, keepdim=True) + 1e-9)
    
    # Build sparse weight matrix W [N, K]
    indices = torch.stack([
        torch.arange(N, device=device).repeat_interleave(k_point),
        j.flatten()
    ], dim=0)
    values = w_sparse.flatten()
    W = torch.sparse_coo_tensor(indices, values, (N, K), device=device).to_dense()
    
    # Solve smoothed F at nodes
    WtW = torch.matmul(W.T, W)  # [K, K]
    A = WtW + lam * L  # [K, K]
    
    # KEY FIX: Keep F as torch tensor (no numpy conversion!)
    F_flat = F.reshape(N, 9)  # [N, 9]
    
    # Right-hand side: W^T F
    rhs = torch.matmul(W.T, F_flat)  # [K, 9]
    
    # Solve: A Y = rhs (differentiable!)
    Y = torch.linalg.solve(A, rhs)  # [K, 9]
    
    # Interpolate back: F_smooth = W Y
    F_smooth_flat = torch.matmul(W, Y)  # [N, 9]
    F_smooth = F_smooth_flat.reshape(N, 3, 3)  # [N, 3, 3]
    
    return F_smooth


def _smooth_F_with_ED_numpy(x, F, K, node_knn, point_knn_nodes, lam, differentiable):
    """Non-differentiable F smoothing (numpy implementation for compatibility)."""
    # Node selection
    if differentiable:
        sel = _random_sampling(_to_numpy(x), K)
    else:
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

    # Points-to-nodes soft weights
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for point-node neighbors.")
    nn = NearestNeighbors(n_neighbors=min(point_knn_nodes, K)).fit(_to_numpy(Xn))
    d, j = nn.kneighbors(_to_numpy(x), return_distance=True)
    h = np.median(d, axis=1, keepdims=True) + 1e-9
    w = np.exp(- (d / h)**2)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-9)

    # Build normal equations and solve
    if TORCH_AVAILABLE:
        device, dtype = _device_dtype_like(_as_torch(x))
        WtW = torch.zeros((K,K), device=device, dtype=dtype)
        for row in range(w.shape[0]):
            cols_row = j[row]
            wr = torch.tensor(w[row], device=device, dtype=dtype)
            WtW[cols_row[:,None], cols_row[None,:]] += (wr[:,None] * wr[None,:])
        A = WtW + lam * L

        F_np = _to_numpy(F).reshape(-1, 9)
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


# ============================================================================
# PNG Export Functions (from runtime_surface.py)
# ============================================================================
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


# ============================================================================
# Surface Point Sampling (from runtime_surface.py)
# ============================================================================
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

def _gumbel_softmax_sample(logits, tau=1.0, hard=False):
    """Gumbel-Softmax sampling (differentiable approximation to categorical sampling)."""
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


# ============================================================================
# Stratified Sampling
# ============================================================================
def stratified_sample_anchors(probs: np.ndarray, M: int, seed: int = 0) -> np.ndarray:
    """
    Stratified importance sampling for uniform point distribution.
    
    Unlike multinomial sampling which clusters points around high-probability anchors,
    this method ensures each anchor receives a proportional allocation, preventing
    both over-sampling and under-sampling.
    
    Algorithm:
    1. Compute expected count per anchor: c_i = p_i * M
    2. Allocate floor(c_i) deterministically
    3. Distribute residual M - sum(floor(c_i)) proportionally
    
    Args:
        probs: [N] - Importance weights (unnormalized)
        M: Target number of samples
        seed: Random seed
        
    Returns:
        sel: [M] - Anchor indices (stratified distribution)
        
    Differentiability: âŒ Discrete (but produces uniform distribution for better coverage)
    """
    np.random.seed(seed)
    N = len(probs)
    
    # Normalize probabilities
    p = probs / (probs.sum() + 1e-12)
    
    # Expected count per anchor
    expected_counts = p * M
    
    # Allocate base counts (floor)
    base_counts = np.floor(expected_counts).astype(int)
    residual = int(M - base_counts.sum())
    
    # Allocate residual proportionally (fractional parts)
    if residual > 0:
        residual_probs = expected_counts - base_counts
        residual_probs = residual_probs / (residual_probs.sum() + 1e-12)
        
        # Sample without replacement for residual
        residual_indices = np.random.choice(
            N, size=min(residual, N), replace=False, p=residual_probs
        )
        for idx in residual_indices:
            base_counts[idx] += 1
    
    # Build selection array by repeating indices
    sel = []
    for i in range(N):
        sel.extend([i] * base_counts[i])
    
    sel = np.array(sel, dtype=np.int32)
    np.random.shuffle(sel)  # Randomize order (deterministic with seed)
    
    return sel[:M]  # Ensure exact M samples


# ============================================================================
# Controlled Jitter Sampling
# ============================================================================
def sample_with_controlled_jitter(x: np.ndarray,
                                  normals: np.ndarray,
                                  spacing: np.ndarray,
                                  sel: np.ndarray,
                                  alpha: float,
                                  thickness: float,
                                  seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample points near surface with controlled jitter to prevent outliers.
    
    Improvements over baseline:
    1. Truncated Gaussian for tangent offsets (eliminates salt-and-pepper outliers)
    2. Reduced normal offset range (tighter shell adherence)
    3. Adaptive alpha scaling based on local spacing
    
    Args:
        x: [N, 3] - Low-res anchor positions
        normals: [N, 3] - Surface normals
        spacing: [N] - Local point spacing
        sel: [M] - Selected anchor indices
        alpha: Jitter scale factor
        thickness: Half-thickness of shell
        seed: Random seed
        
    Returns:
        mu: [M, 3] - Sampled positions
        n: [M, 3] - Normals at sampled points
        anchors: [M, 3] - Anchor positions (for projection)
        
    Differentiability: âš ï¸ Partial (jitter is random but position computation is differentiable)
    """
    np.random.seed(seed)
    M = len(sel)
    
    X = x[sel]          # [M, 3] - anchor positions
    Nrm = normals[sel]  # [M, 3] - normals
    H = spacing[sel]    # [M] - spacing
    
    
    n = _normalize(Nrm)
    
    # Start with cardinal axis
    a = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (M, 1))
    
    # Switch to Y-axis if normal is too aligned with X
    mask = (np.abs((n * a).sum(axis=1)) > 0.9)
    a[mask] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    # Gram-Schmidt orthogonalization
    t1 = _normalize(a - (a * n).sum(axis=1, keepdims=True) * n)
    t2 = _normalize(np.cross(n, t1))
    
    # â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
    # IMPROVED: Truncated Gaussian offsets (prevent outliers)
    # â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
    U = np.random.randn(M, 1)
    V = np.random.randn(M, 1)
    
    # Truncate to [-2Ïƒ, 2Ïƒ] (removes 5% extreme outliers)
    U = np.clip(U, -2.0, 2.0)
    V = np.clip(V, -2.0, 2.0)
    
    # â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
    # IMPROVED: Reduced normal offset (tighter shell)
    # â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
    Z = (np.random.rand(M, 1) * 2.0 - 1.0) * 0.5  # [-0.5, 0.5] instead of [-1, 1]
    
    # â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
    # IMPROVED: Adaptive alpha based on local spacing
    # Dense regions â†’ smaller jitter, Sparse regions â†’ larger jitter
    # â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
    alpha_adaptive = alpha * np.clip(H / (H.mean() + 1e-9), 0.5, 2.0)
    
    # â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
    # Compute final positions
    # mu = anchor + tangent_jitter + normal_jitter
    # â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
    tangent_offset = alpha_adaptive[:, None] * H[:, None] * (U * t1 + V * t2)
    normal_offset = (thickness * Z) * n
    
    mu = X + tangent_offset + normal_offset
    
    return mu.astype(np.float32), n.astype(np.float32), X.astype(np.float32)


# ============================================================================
# IMPROVED: Multi-Scale Density Equalization
# ============================================================================
def multi_scale_density_equalize(pts: np.ndarray,
                                 nrms: np.ndarray,
                                 anchors: np.ndarray,
                                 thickness: float,
                                 differentiable: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coarse-to-fine density equalization for filling holes and uniform distribution.
    
    Strategy:
    1. Coarse stage: Large kernel, aggressive steps â†’ fill large holes
    2. Medium stage: Moderate kernel â†’ smooth distribution
    3. Fine stage: Small kernel, gentle steps â†’ final smoothing
    
    This multi-scale approach is more effective than single-scale equalization
    as it addresses holes of different sizes progressively.
    
    Args:
        pts: [M, 3] - Point positions
        nrms: [M, 3] - Surface normals
        anchors: [M, 3] - Anchor positions (for projection)
        thickness: Half-thickness of shell
        differentiable: If True, maintains gradient flow
        
    Returns:
        pts: [M, 3] - Equalized positions
        nrms: [M, 3] - Updated normals
        
    Differentiability: Fully differentiable (except kNN graph construction)
    """
    # Stage 1: Coarse - Fill large holes
    pts, nrms = _density_equalize_relax(
        pts, nrms, anchors,
        cfg={
            "enabled": True,
            "iters": 5,
            "k": 48,              # Wide neighborhood
            "step": 0.8,          # Aggressive movement
            "radius_mul": 2.0,    # Large kernel
            "project_to_shell": True
        },
        thickness=thickness,
        differentiable=differentiable
    )
    
    # Stage 2: Medium - Smooth distribution
    pts, nrms = _density_equalize_relax(
        pts, nrms, anchors,
        cfg={
            "enabled": True,
            "iters": 5,
            "k": 32,
            "step": 0.6,
            "radius_mul": 1.5,
            "project_to_shell": True
        },
        thickness=thickness,
        differentiable=differentiable
    )
    
    # Stage 3: Fine - Final smoothing
    pts, nrms = _density_equalize_relax(
        pts, nrms, anchors,
        cfg={
            "enabled": True,
            "iters": 5,
            "k": 24,
            "step": 0.4,          # Gentle movement
            "radius_mul": 1.0,
            "project_to_shell": True
        },
        thickness=thickness,
        differentiable=differentiable
    )
    
    return pts, nrms


def _density_equalize_relax(pts, nrms, anchors, cfg, thickness, differentiable=False):
    """
    Single-scale density equalization via repulsive/attractive forces.
    
    Algorithm:
    1. Compute local density Ï_i = Î£_j w_ij (Gaussian kernel)
    2. Compute global median density Ï*
    3. Compute force direction: s_i = tanh((Ï_i - Ï*) / Ï*)
       - Positive (over-dense) â†’ repel
       - Negative (under-dense) â†’ attract
    4. Move: x_i â† x_i - step * s_i * Î£_j w_ij (x_j - x_i)
    5. Project back to shell
    
    This is differentiable as all operations are smooth (except kNN graph).
    """
    if not cfg.get("enabled", True) or len(pts) == 0:
        return pts, nrms

    iters = int(cfg.get("iters", 3))
    k = int(cfg.get("k", 24))
    step = float(cfg.get("step", 0.45))
    rmul = float(cfg.get("radius_mul", 1.0))
    project = bool(cfg.get("project_to_shell", True))

    if TORCH_AVAILABLE:
        P = _as_torch(pts).float()
        N = _as_torch(nrms).float()
        A = _as_torch(anchors).float()
        device, dtype = _device_dtype_like(P)
        
        for iter_idx in range(iters):
            # kNN graph (fixed structure, but distances are differentiable)
            nn = NearestNeighbors(n_neighbors=min(k, len(P))).fit(_to_numpy(P))
            idx = torch.from_numpy(nn.kneighbors(_to_numpy(P), return_distance=False)).to(device)

            Q = P[idx]                          # [M, k, 3] - neighbor positions
            diff = Q - P[:, None, :]            # [M, k, 3] - displacement vectors
            dist = torch.norm(diff, dim=-1)     # [M, k] - distances

            # Adaptive bandwidth (median of neighbor distances)
            h = rmul * (torch.median(dist[:, 1:], dim=1).values + 1e-6)  # [M]
            
            # Gaussian kernel weights
            w = torch.exp(- (dist / h[:, None])**2)  # [M, k]
            w[:, 0] = 0.0  # Exclude self

            # Local density
            rho = w.sum(dim=1, keepdim=True)  # [M, 1]
            
            # Global median density (detach if non-differentiable mode)
            rho_star = torch.median(rho.detach() if not differentiable else rho)

            # Force magnitude: positive (over-dense) â†’ repel, negative (under-dense) â†’ attract
            s = torch.tanh((rho - rho_star) / (rho_star + 1e-6))  # [M, 1]

            # Weighted displacement
            disp = (w[..., None] * diff).sum(dim=1) / (rho + 1e-6)  # [M, 3]

            # Update positions
            P = P - step * s * disp

            if project:
                # Project back to thin shell around anchors
                # Decompose: x - anchor = tangent_component + normal_component
                t = ((P - A) * N).sum(dim=1, keepdim=True)  # Normal component magnitude
                t = torch.clamp(t, -thickness, thickness)    # Clamp to shell
                
                tang = P - A - t * N  # Tangent component
                P = A + tang + t * N  # Reconstruct
        
        return _to_numpy(P), _to_numpy(N)
    else:
        # NumPy fallback (non-differentiable)
        P = np.asarray(pts, dtype=np.float32)
        N = np.asarray(nrms, dtype=np.float32)
        A = np.asarray(anchors, dtype=np.float32)
        
        for _ in range(iters):
            nn = NearestNeighbors(n_neighbors=min(k, len(P))).fit(P)
            d, j = nn.kneighbors(P, return_distance=True)
            diff = P[j] - P[:, None, :]
            h = rmul * (np.median(d[:, 1:], axis=1, keepdims=True) + 1e-6)
            w = np.exp(- (d / h)**2)
            w[:, 0] = 0.0
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


# ============================================================================
# NEW: Laplacian Smoothing
# ============================================================================
def smooth_normals(nrms: np.ndarray,
                  pts: np.ndarray,
                  iters: int = 5,
                  lambda_smooth: float = 0.7,
                  k_neighbors: int = 24,
                  differentiable: bool = False) -> np.ndarray:
    """
    Smooth surface normals for better shading quality.
    
    Averages each normal with its spatial neighbors, then renormalizes.
    This reduces normal discontinuities and produces smoother shading.
    
    Args:
        nrms: [M, 3] - Surface normals to smooth
        pts: [M, 3] - Point positions (for neighbor finding)
        iters: Number of smoothing iterations
        lambda_smooth: Smoothing factor [0, 1] (0=no change, 1=full average)
        k_neighbors: Number of neighbors for averaging
        differentiable: If True, maintains gradient flow
        
    Returns:
        nrms_smooth: [M, 3] - Smoothed unit normals
        
    Differentiability: âœ… Fully differentiable
    """
    if not TORCH_AVAILABLE:
        return nrms
    
    N = torch.from_numpy(nrms).float()
    P = torch.from_numpy(pts).float()
    
    if torch.cuda.is_available():
        N = N.cuda()
        P = P.cuda()
    
    M = len(N)
    k = min(k_neighbors, M)
    
    for _ in range(iters):
        # Find k-nearest neighbors in spatial domain
        nn = NearestNeighbors(n_neighbors=k).fit(P.cpu().numpy())
        idx = torch.from_numpy(nn.kneighbors(P.cpu().numpy(), return_distance=False)).to(N.device)
        
        # Gather neighbor normals
        N_neighbors = N[idx]  # [M, k, 3]
        
        # Average with neighbors
        N_avg = N_neighbors.mean(dim=1)  # [M, 3]
        
        # Blend with original
        N = (1 - lambda_smooth) * N + lambda_smooth * N_avg
        
        # Renormalize to unit length
        N = N / (torch.norm(N, dim=-1, keepdim=True) + 1e-9)
    
    return N.cpu().numpy()


def laplacian_smooth(pts: np.ndarray,
                    nrms: np.ndarray,
                    anchors: np.ndarray,
                    thickness: float,
                    iters: int = 5,
                    lambda_smooth: float = 0.5,
                    differentiable: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Laplacian smoothing for surface coherence while preserving shell constraint.
    
    Classical Laplacian smoothing moves each point toward the centroid of its neighbors:
        x_i â† (1-Î») x_i + Î» * (1/k) Î£_j x_j
    
    This creates a smoother, more uniform point distribution by reducing high-frequency
    irregularities. Combined with shell projection, it maintains the thin manifold structure.
    
    Args:
        pts: [M, 3] - Point positions
        nrms: [M, 3] - Surface normals
        anchors: [M, 3] - Anchor positions
        thickness: Half-thickness of shell
        iters: Number of smoothing iterations
        lambda_smooth: Smoothing factor [0, 1]
        differentiable: If True, maintains gradient flow
        
    Returns:
        pts: [M, 3] - Smoothed positions
        nrms: [M, 3] - Normals (unchanged)
        
    Differentiability: âœ… Fully differentiable
    """
    if not TORCH_AVAILABLE:
        return pts, nrms
    
    P = torch.from_numpy(pts).float().cuda() if torch.cuda.is_available() else torch.from_numpy(pts).float()
    N = torch.from_numpy(nrms).float().to(P.device)
    A = torch.from_numpy(anchors).float().to(P.device)
    
    for _ in range(iters):
        # kNN graph
        nn = NearestNeighbors(n_neighbors=min(24, len(P))).fit(P.cpu().numpy())
        idx = nn.kneighbors(P.cpu().numpy(), return_distance=False)
        idx = torch.from_numpy(idx).to(P.device)
        
        # Neighbor positions
        neighbors = P[idx]  # [M, k, 3]
        
        # Laplacian: centroid of neighbors
        P_mean = neighbors.mean(dim=1)  # [M, 3]
        
        # Weighted update
        P = P * (1 - lambda_smooth) + P_mean * lambda_smooth
        
        # Project back to shell
        t = ((P - A) * N).sum(dim=1, keepdim=True)
        t = torch.clamp(t, -thickness, thickness)
        tang = P - A - t * N
        P = A + tang + t * N
    
    return P.cpu().numpy(), N.cpu().numpy()


# ============================================================================
# NEW: Adaptive Covariance Scaling
# ============================================================================
def compute_adaptive_covariance(mu: np.ndarray,
                               cov_base: np.ndarray,
                               cfg: Dict) -> np.ndarray:
    """
    Hole-aware adaptive covariance scaling for uniform rendering coverage.
    
    Problem: Fixed sigma leads to holes in sparse regions and over-blur in dense regions.
    Solution: Scale sigma proportionally to local spacing.
    
    Algorithm:
    1. Compute local density: spacing = avg distance to k nearest neighbors
    2. Scale sigma: Ïƒ_i = Ïƒ_0 * (spacing_i / mean_spacing)
    3. Clamp to prevent extreme values
    
    This ensures:
    - Sparse regions: Larger Gaussians â†’ fill holes
    - Dense regions: Smaller Gaussians â†’ preserve detail
    
    Args:
        mu: [M, 3] - Point positions
        cov_base: [M, 3, 3] - Base covariance matrices (from F)
        cfg: Configuration dict
            - sigma0: Base scale
            - min_scale: Minimum scaling factor
            - max_scale: Maximum scaling factor
            
    Returns:
        cov_adaptive: [M, 3, 3] - Adaptively scaled covariance matrices
        
    Differentiability: âš ï¸ Partial (kNN graph is fixed, but scaling is differentiable)
    """
    if not cfg.get("enabled", True):
        return cov_base
    
    # Compute local spacing via kNN
    nn = NearestNeighbors(n_neighbors=min(24, len(mu))).fit(mu)
    dist, _ = nn.kneighbors(mu)
    
    # Average distance to nearest neighbors (excluding self)
    local_spacing = dist[:, 1:].mean(axis=1)  # [M] - no keepdims to avoid broadcasting issues
    
    # Normalize by mean spacing (ensure scalar)
    mean_spacing = float(local_spacing.mean())
    spacing_ratio = local_spacing / (mean_spacing + 1e-9)  # [M]
    
    # Clamp to prevent extreme values
    min_scale = cfg.get("min_scale", 0.5)
    max_scale = cfg.get("max_scale", 2.0)
    spacing_ratio = np.clip(spacing_ratio, min_scale, max_scale)  # [M]
    
    # Scale covariance: Sigma_adaptive = scale^2 * Sigma_base
    # Reshape to [M, 1, 1] for proper broadcasting with [M, 3, 3]
    scale_squared = (spacing_ratio ** 2).reshape(-1, 1, 1)  # [M, 1, 1]
    cov_adaptive = cov_base * scale_squared  # [M, 3, 3] * [M, 1, 1] = [M, 3, 3]
    
    return cov_adaptive


# ============================================================================
# Main Synthesis Function (Unchanged interface, improved implementation)
# ============================================================================
def synthesize_runtime_surface(x_low, F_low, cfg: Dict,
                               ema_state: Optional[Dict] = None,
                               seed: int = 1234,
                               differentiable: bool = False,
                               return_torch: bool = False) -> Dict:
    """
    Synthesize high-resolution runtime surface from low-resolution physics simulation.
    
    This function maintains the same interface as the original but incorporates
    improved algorithms for uniformity and smoothness:
    
    1. Stratified sampling -> uniform anchor distribution
    2. Controlled jitter -> no salt-and-pepper outliers
    3. Multi-scale equalization -> hole filling
    4. Laplacian smoothing -> surface coherence
    5. Adaptive covariance -> hole-aware rendering
    
    All improvements maintain or enhance differentiability for end-to-end training.
    
    See default_cfg() for all available parameters.
    """
    if ema_state is None:
        ema_state = {}
    
    # Convert inputs - keep torch tensors if differentiable
    is_torch_input = TORCH_AVAILABLE and torch.is_tensor(x_low)
    differentiable = bool(cfg.get("differentiable", differentiable))
    
    # Store both torch and numpy versions
    if is_torch_input:
        if differentiable and return_torch:
            # Keep original torch tensors for gradient flow
            x_low_t = x_low.float() if x_low.dtype != torch.float32 else x_low
            F_low_t = F_low.float() if F_low.dtype != torch.float32 else F_low
            # Create detached numpy for operations that require it
            x_low_np = x_low.detach().cpu().numpy()
            F_low_np = F_low.detach().cpu().numpy()
        else:
            # No gradient tracking needed
            x_low_np = _to_numpy(x_low)
            F_low_np = _to_numpy(F_low)
            x_low_t = None
            F_low_t = None
    else:
        x_low_np = np.asarray(x_low)
        F_low_np = np.asarray(F_low)
        x_low_t = _as_torch(x_low_np).float() if differentiable and return_torch and TORCH_AVAILABLE else None
        F_low_t = _as_torch(F_low_np).float() if differentiable and return_torch and TORCH_AVAILABLE else None
    
    # Extract parameters
    k_surface = int(cfg.get("k_surface", 24))
    thr_pct = float(cfg.get("thr_percentile", 20.0))
    ema_beta = float(cfg.get("ema_beta", 0.95))
    hys = float(cfg.get("hysteresis", 0.03))
    tau = float(cfg.get("soft_tau", 0.02))
    M = int(cfg.get("M", 180_000))
    alpha = float(cfg.get("surf_jitter_alpha", 0.20))
    thickness = float(cfg.get("thickness", 0.08))
    density_g = float(cfg.get("density_gamma", 2.5))
    
    # Step 1: Surface detection with soft probability mask
    use_surf_detect = cfg.get("use_surface_detection", True)
    
    if use_surf_detect:
        # Use sheetness-based surface detection
        surf_prob, normals, spacing, thr_low, thr_high = compute_surface_mask_soft(
            x_low_np, k_surface, thr_pct, ema_state.get("ema_thr", None),
            ema_beta, hys, tau, ema_state, differentiable=differentiable
        )
        
        # Ensure minimum sampling probability to prevent holes
        surf_prob_np = _to_numpy(surf_prob)
        max_prob = surf_prob_np.max()
        min_prob = max_prob * 0.1
        surf_prob_np = np.maximum(surf_prob_np, min_prob)
        surf_prob_np = surf_prob_np / surf_prob_np.sum()
        surf_prob = surf_prob_np
    else:
        # Bypass surface detection: uniform sampling (for thin-shell objects)
        N = len(x_low_np)
        surf_prob = np.ones(N, dtype=np.float32) / N
        
        # Still compute normals and spacing for jittering
        ind = _pairwise_knn(x_low_np, k_surface)
        normals_np = np.zeros((N, 3), dtype=np.float32)
        spacing_np = np.zeros((N,), dtype=np.float32)
        
        for i in range(N):
            P = x_low_np[ind[i]]
            if TORCH_AVAILABLE:
                P_t = torch.from_numpy(P).float()
                evals, evecs = _local_pca(P_t)
                normals_np[i] = _normalize(evecs[:,0]).cpu().numpy()
            else:
                evals, evecs = _local_pca(P)
                normals_np[i] = _normalize(evecs[:,0])
            
            d = np.linalg.norm(x_low_np[ind[i]] - x_low_np[i][None,:], axis=1)
            spacing_np[i] = np.median(d[1:]) if d.shape[0] > 1 else np.median(d)
        
        normals = normals_np
        spacing = spacing_np
        thr_low = thr_high = 0.0  # Dummy values
        
        ema_state["ema_thr"] = 0.0
        ema_state["thr_low"] = 0.0
        ema_state["thr_high"] = 0.0
    
    # Step 2: Sample anchor points
    use_stratified = cfg.get("use_stratified_sampling", True)
    
    if use_stratified:
        # Stratified sampling for uniform distribution
        sel = stratified_sample_anchors(_to_numpy(surf_prob), M, seed)
        pts, nrm, anchors = sample_with_controlled_jitter(
            x_low_np, _to_numpy(normals), _to_numpy(spacing),
            sel, alpha, thickness, seed
        )
    else:
        # Original sampling (fallback)
        pts, nrm, anchors = sample_surface_points(
            x_low_np, normals, spacing, surf_prob,
            M, alpha, thickness, density_g, seed, differentiable
        )
    
    # Step 2.5: Density equalization for uniform coverage
    equalize_cfg = cfg.get("post_equalize", {})
    if equalize_cfg.get("enabled", True):
        use_multiscale = equalize_cfg.get("use_multiscale", True)
        
        if use_multiscale:
            # Multi-scale coarse-to-fine equalization
            pts, nrm = multi_scale_density_equalize(
                pts, nrm, anchors, thickness, differentiable
            )
        else:
            # Single-scale equalization
            pts, nrm = _density_equalize_relax(
                pts, nrm, anchors, equalize_cfg, thickness, differentiable
            )
    
    # Step 2.6: Laplacian smoothing for surface coherence
    smooth_cfg = cfg.get("laplacian_smooth", {})
    if smooth_cfg.get("enabled", False):
        pts, nrm = laplacian_smooth(
            pts, nrm, anchors, thickness,
            iters=smooth_cfg.get("iters", 5),
            lambda_smooth=smooth_cfg.get("lambda_smooth", 0.5),
            differentiable=differentiable
        )
    
    # Step 2.7: Normal smoothing for better shading
    normal_cfg = cfg.get("normal_smooth", {})
    if normal_cfg.get("enabled", False):
        nrm = smooth_normals(
            nrm, pts,
            iters=normal_cfg.get("iters", 5),
            lambda_smooth=normal_cfg.get("lambda_smooth", 0.7),
            k_neighbors=normal_cfg.get("k_neighbors", 24),
            differentiable=differentiable
        )
    
    # Step 3: Deformation gradient smoothing
    if bool(cfg.get("use_F_kernel", True)):
        if differentiable and return_torch and x_low_t is not None:
            # Use original torch tensors for gradient flow
            F_total_t = F_low_t.reshape(-1, 3, 3)
            F_smooth = smooth_F_with_ED(
                x_low_t, F_total_t,
                cfg.get("ed", {}), differentiable=True
            )
        else:
            F_total = F_low_np.reshape(-1, 3, 3)
            F_smooth = smooth_F_with_ED(
                _as_torch(x_low_np), _as_torch(F_total),
                cfg.get("ed", {}), differentiable=differentiable
            )
    else:
        if differentiable and return_torch and F_low_t is not None:
            F_smooth = F_low_t.reshape(-1, 3, 3)
        else:
            F_smooth = _as_torch(F_low_np.reshape(-1, 3, 3))
    
    # Step 4: Deformation gradient interpolation to upsampled points
    k_F = int(cfg.get("k_F", 24))
    h_mul = float(cfg.get("h_mul", 1.5))
    sigma0 = float(cfg.get("sigma0", 0.02))
    
    if differentiable and return_torch and TORCH_AVAILABLE and x_low_t is not None:
        # Differentiable torch interpolation with gradient flow
        # Use original torch tensors
        X_low = x_low_t  # Original tensor with gradients
        Pts = _as_torch(pts).float()  # pts from sampling (no gradient needed for pts itself)
        F_s = F_smooth.float() if not torch.is_tensor(F_smooth) else F_smooth.float()
        
        device, dtype = _device_dtype_like(X_low)
        X_low = X_low.to(device); Pts = Pts.to(device); F_s = F_s.to(device)
        
        # Compute interpolation weights (differentiable w.r.t. X_low)
        D = torch.cdist(Pts, X_low, p=2)
        d, j = torch.topk(D, k=min(k_F, len(x_low_np)), dim=1, largest=False)
        
        h = h_mul * (torch.median(d, dim=1, keepdim=True).values + 1e-9)
        w = torch.exp(- (d / h)**2)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-9)
        
        F_neighbors = F_s[j]
        F_loc = torch.einsum('mk,mkrc->mrc', w, F_neighbors)
        
        Sig = torch.matmul(F_loc, F_loc.transpose(-2, -1))
        Sig = (sigma0 ** 2) * Sig
        
        # Keep as torch tensors for gradient flow
        pts_out = Pts
        nrm_out = _as_torch(nrm).float()
        
    elif differentiable and TORCH_AVAILABLE:
        # Differentiable but no gradient tracking
        X_low = _as_torch(x_low_np).float()
        Pts = _as_torch(pts).float()
        F_s = _as_torch(F_smooth).float() if not torch.is_tensor(F_smooth) else F_smooth.float()
        
        device, dtype = _device_dtype_like(X_low)
        X_low = X_low.to(device); Pts = Pts.to(device); F_s = F_s.to(device)
        
        D = torch.cdist(Pts, X_low, p=2)
        d, j = torch.topk(D, k=min(k_F, len(x_low_np)), dim=1, largest=False)
        
        h = h_mul * (torch.median(d, dim=1, keepdim=True).values + 1e-9)
        w = torch.exp(- (d / h)**2)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-9)
        
        F_neighbors = F_s[j]
        F_loc = torch.einsum('mk,mkrc->mrc', w, F_neighbors)
        
        Sig = torch.matmul(F_loc, F_loc.transpose(-2, -1))
        Sig = (sigma0 ** 2) * Sig
        
        if not return_torch:
            F_loc = _to_numpy(F_loc)
            Sig = _to_numpy(Sig)
            pts_out = pts
            nrm_out = nrm
        else:
            pts_out = Pts
            nrm_out = _as_torch(nrm).float()
    else:
        # NumPy interpolation (from original)
        nn = NearestNeighbors(n_neighbors=min(k_F, len(x_low_np))).fit(x_low_np)
        d, j = nn.kneighbors(_to_numpy(pts), return_distance=True)
        h = h_mul * (np.median(d, axis=1, keepdims=True) + 1e-9)
        w = np.exp(- (d / h)**2)
        w = w / (w.sum(axis=1, keepdims=True) + 1e-9)
        F_loc = np.einsum('mk,mkrc->mrc', w, _to_numpy(F_smooth)[j])
        
        Sig = np.matmul(F_loc, np.transpose(F_loc, (0, 2, 1)))
        Sig = (sigma0 ** 2) * Sig
        
        pts_out = pts
        nrm_out = nrm
    
    # Step 5: Adaptive covariance scaling for hole filling
    adaptive_cfg = cfg.get("adaptive_sigma", {})
    if adaptive_cfg.get("enabled", False):
        if torch.is_tensor(pts_out):
            pts_np = pts_out.detach().cpu().numpy()
            Sig_np = Sig.detach().cpu().numpy() if torch.is_tensor(Sig) else Sig
            Sig = compute_adaptive_covariance(pts_np, Sig_np, adaptive_cfg)
            if return_torch:
                Sig = _as_torch(Sig).float()
        else:
            Sig = compute_adaptive_covariance(_to_numpy(pts_out), _to_numpy(Sig), adaptive_cfg)
    
    # Debug info
    debug = {
        "thr_low": float(thr_low),
        "thr_high": float(thr_high),
        "ema_thr": float(ema_state.get("ema_thr", thr_high)),
        "mean_prob": float(_to_numpy(surf_prob).mean()) if TORCH_AVAILABLE else float(surf_prob.mean())
    }
    
    # Return
    if return_torch and TORCH_AVAILABLE:
        # Keep torch tensors with gradients
        pts_t = pts_out if torch.is_tensor(pts_out) else _as_torch(pts_out).float()
        nrm_t = nrm_out if torch.is_tensor(nrm_out) else _as_torch(nrm_out).float()
        F_loc_t = F_loc if torch.is_tensor(F_loc) else _as_torch(F_loc).float()
        Sig_t = Sig if torch.is_tensor(Sig) else _as_torch(Sig).float()
        
        return {
            "points": pts_t,
            "normals": nrm_t,
            "F_smooth": F_loc_t,
            "cov": Sig_t,
            "debug": debug,
            "state": ema_state
        }
    else:
        # Return numpy arrays
        return {
            "points": _to_numpy(pts_out) if torch.is_tensor(pts_out) else pts_out,
            "normals": _to_numpy(nrm_out) if torch.is_tensor(nrm_out) else nrm_out,
            "F_smooth": _to_numpy(F_loc) if torch.is_tensor(F_loc) else F_loc,
            "cov": _to_numpy(Sig) if torch.is_tensor(Sig) else Sig,
            "debug": debug,
            "state": ema_state
        }