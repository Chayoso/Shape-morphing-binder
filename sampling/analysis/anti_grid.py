"""
🔥 Anti-Grid De-biasing for Surface Detection

Corrects planarity and anisotropy metrics when points are arranged in regular
grid patterns, which can bias PCA-based surface detection.

Algorithm:
1. Compute gridness: measure of neighbor directional regularity (spherical histogram)
2. De-bias planarity: s' = s - α_grid·gridness
3. De-bias anisotropy: ρ' = relu(ρ - β_grid·gridness)
4. Confidence gating: conf = exp(-γ·gridness)

Author: CHAYO (2025-10)
"""

import torch
import math


def compute_gridness(
    neighbor_vec: torch.Tensor,
    num_bins: int = 48
) -> torch.Tensor:
    """
    Compute gridness score from neighbor direction distribution.
    
    Gridness measures how aligned neighbors are to principal grid axes.
    High gridness indicates regular lattice patterns (artifacts).
    
    Args:
        neighbor_vec: (N, K, 3) neighbor vectors (x[neighbors] - x[center])
        num_bins: Number of spherical bins (default: 48)
    
    Returns:
        gridness: (N,) gridness score [0,1], HIGH = grid-like pattern
    
    Algorithm:
        1. Normalize neighbor vectors to unit sphere
        2. Bin into spherical histogram (θ, φ)
        3. Gridness = max_bin_count / K - uniform_baseline
        
    Notes:
        - Uniform distribution → gridness ≈ 0
        - Grid-aligned neighbors → high gridness (peaks in histogram)
    """
    N, K, _ = neighbor_vec.shape
    device = neighbor_vec.device
    eps = 1e-8
    
    # Normalize to unit vectors
    V = neighbor_vec / (neighbor_vec.norm(dim=2, keepdim=True) + eps)  # (N, K, 3)
    
    # Convert to spherical coordinates (θ, φ)
    # θ = arccos(z) ∈ [0, π], φ = atan2(y, x) ∈ [-π, π]
    z = V[:, :, 2]  # (N, K)
    theta = torch.acos(torch.clamp(z, -1.0, 1.0))  # (N, K)
    phi = torch.atan2(V[:, :, 1], V[:, :, 0])  # (N, K)
    
    # Discretize into bins
    # θ: [0, π] → [0, num_bins_theta)
    # φ: [-π, π] → [0, num_bins_phi)
    num_bins_theta = int(math.sqrt(num_bins))  # e.g., 48 → 7x7 ≈ 49
    num_bins_phi = num_bins_theta
    
    theta_bin = (theta / math.pi * num_bins_theta).long()  # (N, K)
    theta_bin = torch.clamp(theta_bin, 0, num_bins_theta - 1)
    
    phi_bin = ((phi + math.pi) / (2 * math.pi) * num_bins_phi).long()  # (N, K)
    phi_bin = torch.clamp(phi_bin, 0, num_bins_phi - 1)
    
    # Combined bin ID: [0, num_bins)
    bin_id = theta_bin * num_bins_phi + phi_bin  # (N, K)
    
    # Build histogram per point (N, num_bins)
    total_bins = num_bins_theta * num_bins_phi
    hist = torch.zeros(N, total_bins, device=device)  # (N, total_bins)
    
    # Count bins (vectorized)
    # Use one_hot encoding for differentiability
    one_hot = torch.nn.functional.one_hot(bin_id, num_classes=total_bins).float()  # (N, K, total_bins)
    hist = one_hot.sum(dim=1)  # (N, total_bins)
    
    # Normalize histogram
    hist = hist / (K + eps)
    
    # Gridness: deviation from uniform distribution
    # Uniform baseline: 1 / total_bins
    # Gridness = max(hist) - uniform_baseline
    uniform_baseline = 1.0 / total_bins
    gridness = hist.max(dim=1).values - uniform_baseline  # (N,)
    
    # Normalize to [0, 1]
    # Max possible gridness = 1.0 - uniform_baseline (all neighbors in one bin)
    max_gridness = 1.0 - uniform_baseline
    gridness = gridness / (max_gridness + eps)
    gridness = torch.clamp(gridness, 0.0, 1.0)
    
    return gridness


def debias_planarity_anisotropy(
    planarity: torch.Tensor,
    anisotropy: torch.Tensor,
    gridness: torch.Tensor,
    alpha_grid: float = 0.5,
    beta_grid: float = 0.5
) -> tuple:
    """
    De-bias planarity and anisotropy using gridness correction.
    
    Grid-aligned points artificially inflate planarity (appear flat)
    and suppress anisotropy (edges are smoothed out).
    
    Args:
        planarity: (N,) planarity metric s ∈ [0,1]
        anisotropy: (N,) anisotropy metric ρ ∈ [0,1]
        gridness: (N,) gridness score ∈ [0,1]
        alpha_grid: De-biasing strength for planarity (0.3~0.6)
        beta_grid: De-biasing strength for anisotropy (0.3~0.6)
    
    Returns:
        planarity_debias: (N,) de-biased planarity
        anisotropy_debias: (N,) de-biased anisotropy
    
    Algorithm:
        s' = sigmoid((z_s - τ)/γ) - α_grid·gridness
        ρ' = relu(ρ - β_grid·gridness)
        
    Notes:
        - High gridness → reduce planarity (less "fake flat")
        - High gridness → reduce anisotropy (less "fake edge")
    """
    eps = 1e-6
    
    # De-bias planarity: reduce inflated planarity in grid regions
    # Use z-score for stable de-biasing
    s_mean = planarity.mean()
    s_std = torch.clamp(planarity.std(), min=eps)
    z_s = (planarity - s_mean) / s_std
    
    # Apply de-biasing via sigmoid adjustment
    tau = 0.0  # Neutral threshold
    gamma = 1.0  # Moderate steepness
    planarity_norm = torch.sigmoid((z_s - tau) / gamma)
    planarity_debias = planarity_norm - alpha_grid * gridness
    planarity_debias = torch.clamp(planarity_debias, 0.0, 1.0)
    
    # De-bias anisotropy: reduce suppressed anisotropy in grid regions
    anisotropy_debias = torch.clamp(anisotropy - beta_grid * gridness, 0.0, 1.0)
    
    return planarity_debias, anisotropy_debias


def compute_confidence_from_gridness(
    gridness: torch.Tensor,
    gamma_conf: float = 1.2,
    min_conf: float = 0.2
) -> torch.Tensor:
    """
    Compute confidence score from gridness for gating.
    
    Low confidence in high-gridness regions (suppress artifacts).
    
    Args:
        gridness: (N,) gridness score ∈ [0,1]
        gamma_conf: Confidence decay rate (1.0~2.0)
        min_conf: Minimum confidence floor (0.1~0.3)
    
    Returns:
        conf: (N,) confidence ∈ [min_conf, 1.0]
    
    Algorithm:
        conf = exp(-γ·gridness), clamped to [min_conf, 1.0]
    
    Notes:
        - gridness=0 → conf=1.0 (full confidence)
        - gridness=1 → conf=exp(-γ) ≈ min_conf (low confidence)
    """
    conf = torch.exp(-gamma_conf * gridness)
    conf = torch.clamp(conf, min_conf, 1.0)
    
    return conf


__all__ = [
    'compute_gridness',
    'debias_planarity_anisotropy',
    'compute_confidence_from_gridness',
]

