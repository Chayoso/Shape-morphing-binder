"""Density equalization with MLS reprojection."""

import torch
from typing import Tuple, Dict
from ..utils.config import EPS_SAFE, CLAMP_KERNEL_EXP
from ..geometry.mls import project_to_mls_surface_diff


def compute_local_density(P: torch.Tensor, Q: torch.Tensor, w: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Compute local density using RBF kernel."""
    diff = Q - P[:, None, :]
    dist = torch.norm(diff.float(), dim=-1)
    
    kernel = -((dist / h) ** 2).clamp(min=CLAMP_KERNEL_EXP)
    W = torch.exp(kernel) * w
    
    return W


def mask_self_neighbors(W: torch.Tensor, indices: torch.Tensor, M: int) -> torch.Tensor:
    """Mask self-weight to avoid singular attraction."""
    self_mask = (indices == torch.arange(M, device=indices.device).unsqueeze(1))
    W = W.masked_fill(self_mask, 0.0)
    W = W / (W.sum(dim=1, keepdim=True) + EPS_SAFE)
    return W


def compute_density_displacement(P: torch.Tensor, Q: torch.Tensor, W: torch.Tensor, rho: torch.Tensor, step: float) -> torch.Tensor:
    """Compute displacement to equalize density."""
    rho_star = rho.mean()
    s = torch.tanh((rho - rho_star) / (rho_star + 1e-6))
    
    diff = Q - P[:, None, :]
    disp = torch.einsum('nk,nkd->nd', W, diff) / (rho + 1e-6)
    
    return -step * s * disp


def density_equalize_diff(
    pts: torch.Tensor,
    nrms: torch.Tensor,
    anchors: torch.Tensor,
    x_low: torch.Tensor,
    normals_low: torch.Tensor,
    thickness: float,
    knn,
    cfg: Dict
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adjust sampling density while staying close to the MLS surface."""
    if not cfg.get("enabled", True):
        return pts, nrms
    
    iters = int(cfg.get("iters", 8))
    k = int(cfg.get("k", 32))
    step0 = float(cfg.get("step", 0.45))
    annealing = float(cfg.get("annealing", 0.9))
    rmul = float(cfg.get("radius_mul", 1.2))
    use_mls = bool(cfg.get("use_mls_projection", True))
    mls_iters = int(cfg.get("mls_iters", 2))
    mls_step = float(cfg.get("mls_step", 1.0))
    knn_tau = float(cfg.get("knn_tau", 0.15))
    
    P, N = pts, nrms
    
    for it in range(iters):
        if it > 0 and it % 3 == 0:
            knn.invalidate_cache()
        
        idx, w = knn(P, P, k)
        Q = P[idx]
        
        diff = Q - P[:, None, :]
        dist = torch.norm(diff.float(), dim=-1)
        
        h = rmul * ((dist[:, 1:].mean(dim=1) + 1e-6)[:, None])
        
        W = compute_local_density(P, Q, w, h)
        W = mask_self_neighbors(W, idx, P.shape[0])
        
        rho = W.sum(dim=1, keepdim=True)
        
        step = step0 * (annealing ** it)
        disp = compute_density_displacement(P, Q, W, rho, step)
        P = P + disp
        
        if use_mls:
            knn.invalidate_cache()
            P = project_to_mls_surface_diff(
                P, x_low, normals_low,
                iters=mls_iters, k=k, step=mls_step,
                knn=knn, knn_tau=knn_tau
            )
    
    return P, N