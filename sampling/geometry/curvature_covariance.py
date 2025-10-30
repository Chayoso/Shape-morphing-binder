"""
🔥 Curvature-based Covariance Initialization (TARGET ONLY!)

Implements the planarity (s) and anisotropy (ρ) based covariance
initialization as per the advanced connection strategy.

This is ONLY used for target mesh rendering, NOT for source particles!
Source uses F-field based covariance.

Author: CHAYO (2025-10)
"""

import torch
import numpy as np
from typing import Tuple, Optional


def create_curvature_based_covariance_star(
    points: np.ndarray,
    normals: np.ndarray,
    planarity: np.ndarray,
    anisotropy: np.ndarray,
    sigma_params: dict = None
) -> np.ndarray:
    """
    🔥 Create target covariance Σ★ based on planarity and anisotropy (TARGET ONLY!)
    
    Formulation:
        Σ₀ = R · diag(σ²_t1, σ²_t2, σ²_n) · R^T
        
        where:
            σ_n = σ_n0 / (1 + a·κ̂(s))              [normal direction]
            σ_t1 = σ_t0 · (1 + b·κ̂(s))             [tangent 1]
            σ_t2 = σ_t0 · (1 + b·κ̂(s)) · (1 + u·ρ̂)  [tangent 2, anisotropic]
            
            κ̂(s) = (s - μ_s)₊ / (σ_s + ε)         [normalized curvature proxy]
            ρ̂ = (ρ - P90(ρ)) / (P99 - P90)        [percentile-normalized anisotropy]
    
    Args:
        points: (N, 3) positions
        normals: (N, 3) unit normals
        planarity: (N,) planarity values [0,1]
        anisotropy: (N,) anisotropy values [0,1]
        sigma_params: {
            'sigma_n0': 0.02,   # Base normal scale
            'sigma_t0': 0.03,   # Base tangent scale
            'a': 3.0,           # Normal curvature sensitivity
            'b': 0.5,           # Tangent curvature sensitivity
            'u': 0.4            # Tangent anisotropy factor
        }
    
    Returns:
        cov: (N, 3, 3) covariance matrices (numpy)
    """
    # Convert to torch for computation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    points_t = torch.from_numpy(points).float().to(device)
    normals_t = torch.from_numpy(normals).float().to(device)
    s = torch.from_numpy(planarity).float().to(device)
    rho = torch.from_numpy(anisotropy).float().to(device)
    
    N = points.shape[0]
    eps = 1e-6
    
    # Default parameters (conservative)
    if sigma_params is None:
        sigma_params = {}
    
    sigma_n0 = float(sigma_params.get('sigma_n0', 0.020))
    sigma_t0 = float(sigma_params.get('sigma_t0', 0.030))
    a = float(sigma_params.get('a', 3.0))
    b = float(sigma_params.get('b', 0.5))
    u = float(sigma_params.get('u', 0.4))
    
    # ═══════════════════════════════════════════════════════════════════
    # Step 1: Normalized curvature proxy κ̂(s)
    # ═══════════════════════════════════════════════════════════════════
    s_mean = s.mean()
    s_std = torch.clamp(s.std(), min=eps)
    kappa_hat = torch.clamp((s - s_mean) / s_std, min=0.0)  # (s - μ)₊ / σ
    
    # ═══════════════════════════════════════════════════════════════════
    # Step 2: Percentile-normalized anisotropy ρ̂
    # ═══════════════════════════════════════════════════════════════════
    p90 = torch.quantile(rho, 0.90)
    p99 = torch.quantile(rho, 0.99)
    rho_hat = torch.clamp((rho - p90) / (p99 - p90 + eps), 0.0, 1.0)
    
    # ═══════════════════════════════════════════════════════════════════
    # Step 3: Curvature-dependent scales
    # ═══════════════════════════════════════════════════════════════════
    # σ_n = σ_n0 / (1 + a·κ̂)  →  HIGH curvature → SMALL normal scale
    sigma_n = sigma_n0 / (1.0 + a * kappa_hat)
    
    # σ_t1 = σ_t0 · (1 + b·κ̂)  →  HIGH curvature → LARGE tangent scale
    sigma_t1 = sigma_t0 * (1.0 + b * kappa_hat)
    
    # σ_t2 = σ_t0 · (1 + b·κ̂) · (1 + u·ρ̂)  →  ANISOTROPIC tangent
    sigma_t2 = sigma_t0 * (1.0 + b * kappa_hat) * (1.0 + u * rho_hat)
    
    # ═══════════════════════════════════════════════════════════════════
    # Step 4: Build rotation matrix R = [t1, t2, n]
    # ═══════════════════════════════════════════════════════════════════
    # Normalize normals (should already be normalized)
    n = normals_t / (normals_t.norm(dim=1, keepdim=True) + eps)  # (N, 3)
    
    # Build tangent frame (Gram-Schmidt)
    # Choose arbitrary reference (prefer [0,0,1] unless parallel to n)
    ref = torch.zeros_like(n)
    ref[:, 2] = 1.0
    
    # If n is nearly parallel to [0,0,1], use [1,0,0] instead
    parallel_mask = (torch.abs(n[:, 2]) > 0.9)
    ref[parallel_mask, 2] = 0.0
    ref[parallel_mask, 0] = 1.0
    
    # t1 = ref - (ref·n)n  (Gram-Schmidt orthogonalization)
    t1 = ref - (ref * n).sum(dim=1, keepdim=True) * n
    t1 = t1 / (t1.norm(dim=1, keepdim=True) + eps)  # (N, 3)
    
    # t2 = n × t1  (right-handed frame)
    t2 = torch.cross(n, t1, dim=1)  # (N, 3)
    
    # R = [t1, t2, n] as column vectors → (N, 3, 3)
    R = torch.stack([t1, t2, n], dim=2)  # (N, 3, 3)
    
    # ═══════════════════════════════════════════════════════════════════
    # Step 5: Build covariance Σ = R · diag(σ²_t1, σ²_t2, σ²_n) · R^T
    # ═══════════════════════════════════════════════════════════════════
    # Diagonal matrix of squared scales
    scales_sq = torch.stack([
        sigma_t1 ** 2,
        sigma_t2 ** 2,
        sigma_n ** 2
    ], dim=1)  # (N, 3)
    
    # S = diag(σ²_t1, σ²_t2, σ²_n)  → (N, 3, 3)
    S = torch.diag_embed(scales_sq)  # (N, 3, 3)
    
    # Σ = R · S · R^T
    cov = torch.bmm(torch.bmm(R, S), R.transpose(1, 2))  # (N, 3, 3)
    
    # Symmetrize (numerical stability)
    cov = 0.5 * (cov + cov.transpose(1, 2))
    
    # Add tiny regularization for PSD guarantee
    eye = torch.eye(3, device=device).unsqueeze(0)  # (1, 3, 3)
    cov = cov + 1e-6 * eye
    
    # ═══════════════════════════════════════════════════════════════════
    # Step 6: Convert to numpy and return
    # ═══════════════════════════════════════════════════════════════════
    cov_np = cov.cpu().numpy()
    
    return cov_np