"""
Point Cloud Covariance Pipeline.

Direct pipeline: F-field → anisotropic covariance for Gaussian rendering.
No upsampling/subdivision — learnable per-particle σ + opacity handles coverage.

Author: CHAYO
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional

from .utils.config import default_cfg
from .utils.utils import ensure_torch, as_numpy
from .analysis.knn import HybridFAISSKNN, FAISS_AVAILABLE
from .geometry.deformation_covariance import build_deformation_covariance
from .geometry.curvature_covariance import create_curvature_based_covariance_star


# ============================================================================
# PCA-based Normal Computation
# ============================================================================
@torch.no_grad()
def compute_normals_pca(points: torch.Tensor, knn, k: int = 32, prefer_outward: bool = True) -> torch.Tensor:
    """
    PCA로 local surface normal 계산 + orientation consistency.

    Args:
        points: (N, 3) point positions
        knn: KNN function
        k: neighbor count
        prefer_outward: orient normals away from centroid

    Returns:
        normals: (N, 3) unit normals
    """
    k = min(k, points.shape[0])

    idx, _ = knn(points, points, k)
    neighbors = points[idx]  # (N, k, 3)

    centroid = neighbors.mean(dim=1, keepdim=True)  # (N, 1, 3)
    centered = neighbors - centroid  # (N, k, 3)

    C = torch.einsum('nki,nkj->nij', centered, centered) / k  # (N, 3, 3)

    from sampling.geometry.curvature_covariance import _safe_eigh_cuda_cpu
    eigenvalues, eigenvectors = _safe_eigh_cuda_cpu(C, eps=1e-6)

    # Normal = eigenvector with smallest eigenvalue
    normals = eigenvectors[:, :, 0]  # (N, 3)
    normals = F.normalize(normals, dim=-1, eps=1e-6)

    if prefer_outward:
        global_centroid = points.mean(dim=0)
        to_point = points - global_centroid
        dot = (normals * to_point).sum(dim=-1)
        flip = dot < 0
        normals[flip] = -normals[flip]

    return normals


def _build_curvature_covariance(
    points: torch.Tensor,
    knn,
    cov_cfg: Dict,
    device: torch.device,
    verbose: bool
) -> Optional[torch.Tensor]:
    """Compute curvature-based covariance (Σ★) for target meshes."""
    if points.shape[0] < 8:
        return None

    from sampling.analysis.pca import batched_pca_surface_optimized

    k_curv = min(32, points.shape[0] - 1)
    idx_nn, w_nn = knn(points, points, k_curv)

    normals_pca, surfvar, spacing_pca, curvature, anisotropy_t, planarity_t, \
    principal_dir1, principal_dir2, principal_curv = batched_pca_surface_optimized(
        x=points, indices=idx_nn, weights=w_nn, return_principal_dirs=True
    )

    sigma_params = dict(cov_cfg.get("curvature_sigma", {}))

    cov_np = create_curvature_based_covariance_star(
        points=points.detach().cpu().numpy(),
        normals=normals_pca.detach().cpu().numpy(),
        planarity=planarity_t.detach().cpu().numpy(),
        anisotropy=anisotropy_t.detach().cpu().numpy(),
        sigma_params=sigma_params,
        principal_curv=principal_curv.detach().cpu().numpy(),
        principal_dirs=(principal_dir1.detach().cpu().numpy(),
                        principal_dir2.detach().cpu().numpy())
    )

    cov_torch = torch.from_numpy(cov_np).to(device)

    if verbose:
        print(f"  Curvature covariance computed (sigma_n0={sigma_params.get('sigma_n0', 0.03):.3f})")

    return cov_torch


def upsample(
    x_low: torch.Tensor,
    F_low: torch.Tensor,
    cfg: Optional[Dict] = None,
    state: Optional[Dict] = None,
    seed: int = 1234,
    return_torch: bool = True,
    export_stages: bool = False,
    learnable_cov_module=None,
    current_episode: int = -1,
    external_levelset=None,
    per_particle_sigma: Optional[torch.Tensor] = None,
    sigma_aniso: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Direct F → Covariance pipeline for Gaussian rendering.

    No upsampling or subdivision — uses input points as-is.
    Per-particle σ and opacity (learnable) handle rendering coverage.

    Args:
        x_low: (N, 3) point positions
        F_low: (N, 3, 3) deformation gradients
        cfg: pipeline config dict
        state: EMA state dict (updated in-place)
        seed: random seed
        return_torch: True for torch tensors, False for numpy
        per_particle_sigma: (N,) learnable per-particle sigma for covariance

    Returns:
        dict with keys: points, normals, cov, F_interp, anchors, state, debug
    """
    if cfg is None:
        cfg = default_cfg()

    if state is None:
        state = {}

    device = x_low.device if torch.is_tensor(x_low) else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    x_low = ensure_torch(x_low, device=device)
    F_low = ensure_torch(F_low, device=device).reshape(-1, 3, 3)

    upsample_cfg = cfg.get("upsample", {})

    # KNN setup
    knn_cfg = upsample_cfg.get("knn", {})
    knn = HybridFAISSKNN(
        use_faiss=knn_cfg.get("use_faiss", True) and FAISS_AVAILABLE,
        use_ivf=knn_cfg.get("use_ivf", True),
        tau=knn_cfg.get("tau", 0.15),
        nlist=knn_cfg.get("nlist", 100),
        nprobe=knn_cfg.get("nprobe", 10),
    )

    verbose = export_stages
    cov_cfg = upsample_cfg.get("covariance", {})

    # ========================================================================
    # 1. Compute normals (PCA)
    # ========================================================================
    from sampling.analysis.pca import batched_pca_surface_optimized

    k_pca = min(32, x_low.shape[0] - 1)
    idx_nn, w_nn = knn(x_low, x_low, k_pca)

    normals_pca, surfvar, spacing_pca, curvature, anisotropy, planarity, \
    principal_dir1, principal_dir2, principal_curv = batched_pca_surface_optimized(
        x=x_low, indices=idx_nn, weights=w_nn, return_principal_dirs=True
    )

    normals = compute_normals_pca(x_low, knn, k=k_pca, prefer_outward=True)

    # ========================================================================
    # 2. Build covariance
    # ========================================================================
    is_target = (current_episode < 0)
    cov_target = None

    if is_target:
        sigma_iso = float(cov_cfg.get("sigma_isotropic", 0.01))
        cov = torch.eye(3, device=device, dtype=x_low.dtype).unsqueeze(0).expand(
            x_low.shape[0], 3, 3
        ) * (sigma_iso ** 2)
        F_interp = torch.eye(3, device=device, dtype=x_low.dtype).unsqueeze(0).expand(
            x_low.shape[0], 3, 3
        )
        cov_target = _build_curvature_covariance(x_low, knn, cov_cfg, device, verbose)
    else:
        cov, F_interp, _ = build_deformation_covariance(
            points=x_low,
            x_low=x_low,
            F_low=F_low,
            knn=knn,
            cfg=cov_cfg,
            learnable_cov_module=learnable_cov_module,
            x_low_normals=normals,
            x_low_curvature=None,
            per_particle_sigma=per_particle_sigma,
            sigma_aniso=sigma_aniso,
        )

    # ========================================================================
    # 3. Return results
    # ========================================================================
    result = {
        "points": x_low if return_torch else as_numpy(x_low),
        "normals": normals if return_torch else as_numpy(normals),
        "cov": cov if return_torch else as_numpy(cov),
        "F_interp": F_interp if return_torch else as_numpy(F_interp),
        "anchors": x_low if return_torch else as_numpy(x_low),
        "state": state,
        "debug": {
            "pipeline_mode": "direct",
            "num_points": x_low.shape[0],
        }
    }

    if cov_target is not None:
        result["cov_target"] = cov_target if return_torch else as_numpy(cov_target)

    return result


__all__ = [
    "upsample",
]
