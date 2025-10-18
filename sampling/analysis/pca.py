"""PCA-based surface analysis."""

import torch
from typing import Tuple
from ..utils.config import EPS_SAFE, EPS_PCA, TANH_SCALE
from ..utils.utils import normalize


def compute_weighted_centroid(neighbors: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute weighted centroid of neighbors."""
    return torch.einsum('nk,nkd->nd', weights, neighbors)


def compute_weighted_covariance(centered: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute weighted covariance matrix."""
    sqrt_w = torch.sqrt(weights).unsqueeze(-1)
    weighted = centered * sqrt_w
    cov = torch.einsum('nki,nkj->nij', weighted, weighted)
    cov = cov / (weights.sum(dim=1, keepdim=True).unsqueeze(-1) + EPS_SAFE)
    return cov


def extract_normal_from_pca(evecs: torch.Tensor, x: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
    """Extract and orient normal vector from PCA."""
    n_raw = evecs[:, :, 0]
    
    global_c = x.mean(dim=0)
    to_out = x - centroid
    
    mask = torch.norm(to_out, dim=1) < 1e-6
    to_out[mask] = x[mask] - global_c
    
    dot = torch.einsum('nd,nd->n', n_raw, to_out)
    sign = torch.tanh(TANH_SCALE * dot).unsqueeze(-1)
    
    normals = normalize(n_raw * sign)
    return normals


def compute_local_spacing(neighbors: torch.Tensor, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute local spacing for adaptive jitter."""
    dists = torch.norm(neighbors - x.unsqueeze(1), dim=-1)
    spacing = torch.einsum('nk,nk->n', dists, weights) / (weights.sum(dim=1) + EPS_SAFE)
    return spacing


def batched_pca_surface_optimized(
    x: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Weighted PCA per point.
    
    Returns:
        normals: (N,3) surface normals
        surface_variance: (N,) surface quality metric
        local_spacing: (N,) local point spacing
    """
    neighbors = x[indices]
    
    centroid = compute_weighted_centroid(neighbors, weights)
    centered = neighbors - centroid.unsqueeze(1)
    
    cov = compute_weighted_covariance(centered, weights)
    
    evals, evecs = torch.linalg.eigh(cov)
    evals = torch.clamp(evals, min=EPS_PCA)
    
    surfvar = evals[:, 0] / (evals.sum(dim=1) + EPS_PCA)
    normals = extract_normal_from_pca(evecs, x, centroid)
    spacing = compute_local_spacing(neighbors, x, weights)
    
    return normals, surfvar, spacing