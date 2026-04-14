"""
Chamfer-driven plasticity: use 3D nearest-neighbor displacement to target
surface as the plasticity driving signal, instead of 2D render gradients.

Key insight: updating F_plastic shifts the elastic rest configuration toward
the target, so physics elastic forces become allies (driving particles toward
target) rather than adversaries (pulling them back).
"""

import numpy as np
from scipy.spatial import cKDTree


def compute_chamfer_plasticity(
    x: np.ndarray,
    target_pts: np.ndarray,
    Fp: np.ndarray,
    eta: float = 0.02,
    adaptive_eta_max: float = 5.0,
    smooth_k: int = 64,
    diffusion_iters: int = 3,
    clip_pct: float = 95.0,
    damping: float = 0.1,
    J_min: float = 0.5,
    J_max: float = 1.5,
) -> tuple[np.ndarray, dict]:
    """
    Compute Fp update from Chamfer displacement field.

    Args:
        x: (N, 3) current particle positions
        target_pts: (M, 3) target surface points
        Fp: (N, 3, 3) current plastic deformation
        eta: base plasticity rate
        smooth_k: KNN neighbors for diffusion
        diffusion_iters: spatial smoothing iterations

    Returns:
        (dFp, metrics): (N, 3, 3) plasticity increment and diagnostics
    """
    N = len(x)
    metrics = {}

    # 1. Nearest neighbor: particle → target
    tree = cKDTree(target_pts)
    nn_dist, nn_idx = tree.query(x, k=1)
    nn_pts = target_pts[nn_idx]  # (N, 3)
    disp = nn_pts - x  # displacement toward target

    metrics['chamfer_mean'] = float(nn_dist.mean())
    metrics['chamfer_max'] = float(nn_dist.max())
    metrics['chamfer_median'] = float(np.median(nn_dist))

    # 2. Clip outliers
    norms = np.linalg.norm(disp, axis=1)
    if clip_pct < 100:
        active = norms > 1e-8
        if active.any():
            thresh = np.percentile(norms[active], clip_pct)
            scale = np.minimum(1.0, thresh / (norms + 1e-8))
            disp *= scale[:, None]

    # 3. Spatial diffusion (smooth displacement field)
    if diffusion_iters > 0 and smooth_k > 0:
        ptree = cKDTree(x)
        _, knn_idx = ptree.query(x, k=smooth_k + 1)
        knn_idx = knn_idx[:, 1:]  # exclude self
        for _ in range(diffusion_iters):
            neighbor_mean = disp[knn_idx].mean(axis=1)
            disp = 0.5 * disp + 0.5 * neighbor_mean

    # 4. Adaptive eta: scale by local displacement magnitude
    norms_smooth = np.linalg.norm(disp, axis=1)
    median_norm = max(np.median(norms_smooth[norms_smooth > 1e-8]), 1e-8)
    eta_local = eta * np.minimum(norms_smooth / median_norm, adaptive_eta_max)

    # 5. Compute dFp from displacement Jacobian
    #    J_ij = d(disp_i)/dx_j estimated from neighbor differences
    if smooth_k > 0:
        ptree_local = cKDTree(x)
        _, knn_idx_j = ptree_local.query(x, k=min(smooth_k, 16) + 1)
        knn_idx_j = knn_idx_j[:, 1:]

        dx = x[knn_idx_j] - x[:, None, :]  # (N, K, 3)
        dd = disp[knn_idx_j] - disp[:, None, :]  # (N, K, 3)
        dx_norm_sq = np.sum(dx ** 2, axis=2, keepdims=True).clip(1e-12)

        # Outer product: dd_i * dx_j / |dx|^2, averaged over neighbors
        J = np.einsum('nki,nkj->nij', dd, dx / dx_norm_sq) / knn_idx_j.shape[1]
        # Symmetrize
        dFp = 0.5 * (J + np.swapaxes(J, 1, 2))
    else:
        dFp = np.zeros((N, 3, 3), dtype=np.float32)

    # 6. Scale by adaptive eta
    dFp *= eta_local[:, None, None]

    metrics['dFp_chamfer_norm'] = float(np.linalg.norm(dFp.reshape(N, -1), axis=1).mean())
    metrics['chamfer_eta_mean'] = float(eta_local.mean())
    metrics['chamfer_active'] = int((norms_smooth > 1e-6).sum())

    return dFp.astype(np.float32), metrics
