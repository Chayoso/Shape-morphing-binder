"""Surface smoothing with tangent Laplacian."""

import torch
from typing import Tuple
from ..utils.config import EPS_SAFE
from ..utils.utils import normalize
from ..geometry.mls import project_to_mls_surface_diff


def interpolate_normals_at_points(P: torch.Tensor, X: torch.Tensor, N: torch.Tensor, knn, k_norm: int) -> torch.Tensor:
    """Interpolate normals from X to P."""
    idx_N, w_N = knn(P, X, k_norm)
    N_neighbors = N[idx_N]
    n = normalize((w_N.unsqueeze(-1) * N_neighbors).sum(1), eps=EPS_SAFE)
    return n


def compute_laplacian_displacement(P: torch.Tensor, Q: torch.Tensor, w: torch.Tensor, dist: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Compute Laplacian smoothing displacement."""
    diff = Q - P[:, None, :]
    
    ww = torch.exp(- (dist / h) ** 2) * w
    ww[:, 0] = 0.0
    ww = ww / (ww.sum(dim=1, keepdim=True) + EPS_SAFE)
    
    lap = torch.einsum('nk,nkd->nd', ww, diff)
    
    return lap


def split_tangent_normal_components(lap: torch.Tensor, n: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split Laplacian into tangent and normal components."""
    lap_n = (lap * n).sum(-1, keepdim=True) * n
    lap_t = lap - lap_n
    return lap_t, lap_n


def surface_smoother_diff(
    P: torch.Tensor,
    X: torch.Tensor,
    N: torch.Tensor,
    knn,
    iters: int = 3,
    k: int = 24,
    step: float = 0.12,
    lambda_normal: float = 0.15,
    mls_every: int = 2
) -> torch.Tensor:
    """Gentle surface smoothing using tangent Laplacian."""
    k_norm = min(8, X.shape[0])
    n = interpolate_normals_at_points(P, X, N, knn, k_norm)
    
    for t in range(int(iters)):
        idx, w = knn(P, P, min(k, P.shape[0]-1))
        Q = P[idx]
        diff = Q - P[:, None, :]
        dist = torch.norm(diff, dim=-1)
        
        h = (dist[:, 1:].mean(dim=1) + 1e-6).unsqueeze(-1)
        
        lap = compute_laplacian_displacement(P, Q, w, dist, h)
        lap_t, lap_n = split_tangent_normal_components(lap, n)
        
        update = step * (lap_t + lambda_normal * lap_n)
        P = P + update
        
        if mls_every > 0 and (t + 1) % mls_every == 0:
            knn.invalidate_cache()
            P = project_to_mls_surface_diff(P, X, N, iters=1, k=k, step=1.0, knn=knn)
    
    return P

def smooth_normals_diff(
    normals: torch.Tensor,
    positions: torch.Tensor,
    knn,
    iters: int = 2,
    k: int = 16,
    lambda_smooth: float = 0.8
) -> torch.Tensor:
    """
    Smooth normals using spatial neighbors with distance-weighted averaging.
    
    Args:
        normals: (N, 3) surface normals
        positions: (N, 3) surface positions (for finding spatial neighbors)
        knn: KNN function
        iters: number of smoothing iterations
        k: number of neighbors
        lambda_smooth: smoothing strength [0, 1]
    
    Returns:
        smoothed_normals: (N, 3) smoothed normals
    """
    N = normals.shape[0]
    device = normals.device
    
    normals_smooth = normals.clone()
    
    for t in range(iters):
        # Find spatial neighbors
        idx, w = knn(positions, positions, min(k, N-1))
        
        # Get neighbor normals
        neighbor_normals = normals_smooth[idx]  # (N, k, 3)
        
        # Compute spatial distances for additional weighting
        neighbor_positions = positions[idx]  # (N, k, 3)
        diff = neighbor_positions - positions.unsqueeze(1)  # (N, k, 3)
        dist = torch.norm(diff, dim=-1)  # (N, k)
        
        # Compute adaptive bandwidth (median distance to neighbors)
        h = dist[:, 1:].median(dim=1, keepdim=True).values + EPS_SAFE  # (N, 1)
        
        # Distance-weighted smoothing (Gaussian kernel)
        spatial_weights = torch.exp(-(dist / h) ** 2)  # (N, k)
        
        # Mask self (first neighbor is always self with distance ~0)
        spatial_weights[:, 0] = 0.0
        
        # Combine with KNN weights
        combined_weights = spatial_weights * w  # (N, k)
        combined_weights = combined_weights / (combined_weights.sum(dim=1, keepdim=True) + EPS_SAFE)
        
        # Weighted average
        avg_normal = (combined_weights.unsqueeze(-1) * neighbor_normals).sum(dim=1)  # (N, 3)
        
        # Normalize
        avg_normal = normalize(avg_normal, eps=EPS_SAFE)
        
        # Blend with original (EMA-style update)
        normals_smooth = lambda_smooth * avg_normal + (1 - lambda_smooth) * normals_smooth
        
        # Re-normalize to ensure unit length
        normals_smooth = normalize(normals_smooth, eps=EPS_SAFE)
    
    return normals_smooth