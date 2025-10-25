"""
Anchor Redistribution Algorithm for Point Cloud Densification.

This module implements an anchor-based redistribution strategy that identifies
sparse regions in a point cloud and redistributes anchors to achieve more uniform
surface coverage. Unlike volume filtering, this approach actively moves or replicates
anchors to fill gaps in the surface.

═══════════════════════════════════════════════════════════════════════════════
ALGORITHM OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

Input:  {x_low, normals, surf_prob, spacing} - Sparse anchors with geometry
Output: {anchors_new, probs_new, normals_new} - Redistributed anchors

Strategy:
1. Identify sparse regions using local density analysis
2. Generate candidate positions in sparse areas
3. Project candidates onto nearest surface patches
4. Blend with original anchors using importance weights

Key Features:
- Surface-aware: respects local geometry and normals
- Differentiable: maintains gradient flow for end-to-end learning
- Adaptive: density analysis scales with local point spacing

═══════════════════════════════════════════════════════════════════════════════
MATHEMATICAL FORMULATION
═══════════════════════════════════════════════════════════════════════════════

Step 1: Density Estimation
────────────────────────
    ρᵢ = k / V(rₖ)                          [Local density]
    V(r) = (4/3)πr³                         [Volume of k-NN ball]
    ρ_norm = ρ / median(ρ)                  [Normalized density]

Step 2: Sparsity Score
──────────────────────
    s = sigmoid(α·(1 - ρ_norm))             [Sparsity weight]
    where s ≈ 1 in sparse regions, s ≈ 0 in dense regions

Step 3: Candidate Generation
─────────────────────────────
    For each sparse point i with sᵢ > threshold:
        Generate n_new candidates around xᵢ
        cⱼ = xᵢ + r·(cos(θⱼ)·t₁ + sin(θⱼ)·t₂)
        where r ∼ [h, 2h] and θⱼ ∈ [0, 2π]

Step 4: Surface Projection
───────────────────────────
    For each candidate cⱼ:
        Find k nearest original points
        Compute local plane: π = (n̄, p̄)
        Project: c'ⱼ = cⱼ - ((cⱼ - p̄)·n̄)·n̄

Step 5: Anchor Merging
──────────────────────
    x_new = [x_original; x_candidates]
    p_new = [p_original; p_candidates]
    where p_candidates ∝ sparsity of local region

═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


def estimate_local_density(
    points: torch.Tensor,
    knn,
    k: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate local point density using k-NN volume.
    
    Args:
        points: (N, 3) point positions
        knn: FAISS wrapper for nearest neighbor search
        k: Number of neighbors to consider
        
    Returns:
        density: (N,) normalized density values
        radius: (N,) radius of k-NN ball for each point
    """
    N = len(points)
    device = points.device
    
    # Find k nearest neighbors
    dists, _ = knn.query(points, k=k+1)  # +1 because point itself is included
    
    # Use k-th nearest distance as radius (exclude self at index 0)
    radius = dists[:, -1]  # (N,)
    
    # Compute volume of k-NN ball
    volume = (4.0 / 3.0) * math.pi * (radius ** 3 + 1e-8)  # Add epsilon for stability
    
    # Density = points per unit volume
    density = k / volume  # (N,)
    
    # Normalize by median density
    median_density = torch.median(density)
    density_norm = density / (median_density + 1e-8)
    
    return density_norm, radius


def compute_sparsity_scores(
    density: torch.Tensor,
    alpha: float = 5.0,
    threshold: float = 0.7,
) -> torch.Tensor:
    """
    Convert density to sparsity scores using sigmoid gating.
    
    Args:
        density: (N,) normalized density values
        alpha: Sigmoid sharpness
        threshold: Density threshold (regions below this are considered sparse)
        
    Returns:
        sparsity: (N,) sparsity scores in [0, 1]
    """
    # Invert density and apply sigmoid
    # High density -> low sparsity, low density -> high sparsity
    sparsity = torch.sigmoid(alpha * (threshold - density))
    return sparsity


def generate_surface_candidates(
    points: torch.Tensor,
    normals: torch.Tensor,
    spacing: torch.Tensor,
    sparsity: torch.Tensor,
    n_candidates_per_point: int = 4,
    radius_min_factor: float = 1.0,
    radius_max_factor: float = 2.0,
    sparsity_threshold: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate candidate anchor positions in sparse regions.
    
    Args:
        points: (N, 3) original anchor positions
        normals: (N, 3) surface normals at anchors
        spacing: (N,) local point spacing
        sparsity: (N,) sparsity scores
        n_candidates_per_point: Number of candidates per sparse point
        radius_min_factor: Minimum radius as multiple of spacing
        radius_max_factor: Maximum radius as multiple of spacing
        sparsity_threshold: Only generate candidates for points with sparsity > threshold
        
    Returns:
        candidates: (M, 3) candidate positions
        parent_indices: (M,) indices of parent points in original cloud
    """
    device = points.device
    N = len(points)
    
    # Identify sparse points
    sparse_mask = sparsity > sparsity_threshold
    sparse_indices = torch.nonzero(sparse_mask, as_tuple=True)[0]
    n_sparse = len(sparse_indices)
    
    if n_sparse == 0:
        # No sparse regions, return empty
        return torch.zeros((0, 3), device=device), torch.zeros((0,), dtype=torch.long, device=device)
    
    # Get sparse points data
    sparse_points = points[sparse_indices]  # (n_sparse, 3)
    sparse_normals = normals[sparse_indices]  # (n_sparse, 3)
    sparse_spacing = spacing[sparse_indices]  # (n_sparse,)
    
    # Build tangent frames for each sparse point
    # Use Gram-Schmidt to get orthonormal basis {t1, t2, n}
    arbitrary = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n_sparse, 3)
    parallel_mask = (torch.abs(torch.sum(sparse_normals * arbitrary, dim=1)) > 0.99)
    arbitrary[parallel_mask] = torch.tensor([0.0, 1.0, 0.0], device=device)
    
    t1 = arbitrary - sparse_normals * (arbitrary * sparse_normals).sum(dim=1, keepdim=True)
    t1 = F.normalize(t1, dim=1)
    t2 = torch.cross(sparse_normals, t1, dim=1)
    t2 = F.normalize(t2, dim=1)
    
    # Generate candidates in circular pattern around each sparse point
    angles = torch.linspace(0, 2 * math.pi, n_candidates_per_point + 1, device=device)[:-1]  # (n_cand,)
    
    # Expand for all sparse points
    angles = angles.unsqueeze(0).expand(n_sparse, -1)  # (n_sparse, n_cand)
    
    # Random radii in [radius_min, radius_max] * spacing
    radii = torch.rand(n_sparse, n_candidates_per_point, device=device)
    radii = radii * (radius_max_factor - radius_min_factor) + radius_min_factor
    radii = radii * sparse_spacing.unsqueeze(1)  # (n_sparse, n_cand)
    
    # Convert to Cartesian offsets in tangent plane
    cos_angles = torch.cos(angles)  # (n_sparse, n_cand)
    sin_angles = torch.sin(angles)  # (n_sparse, n_cand)
    
    # Tangent offsets
    offset_t1 = (radii * cos_angles).unsqueeze(2) * t1.unsqueeze(1)  # (n_sparse, n_cand, 3)
    offset_t2 = (radii * sin_angles).unsqueeze(2) * t2.unsqueeze(1)  # (n_sparse, n_cand, 3)
    
    # Total offset in tangent plane
    offsets = offset_t1 + offset_t2  # (n_sparse, n_cand, 3)
    
    # Generate candidates
    candidates = sparse_points.unsqueeze(1) + offsets  # (n_sparse, n_cand, 3)
    candidates = candidates.reshape(-1, 3)  # (n_sparse * n_cand, 3)
    
    # Parent indices for tracking
    parent_indices = sparse_indices.unsqueeze(1).expand(-1, n_candidates_per_point).reshape(-1)  # (n_sparse * n_cand,)
    
    return candidates, parent_indices


def project_to_surface(
    candidates: torch.Tensor,
    points: torch.Tensor,
    normals: torch.Tensor,
    knn,
    k: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project candidate points onto nearest surface patch.
    
    Args:
        candidates: (M, 3) candidate positions
        points: (N, 3) original surface points
        normals: (N, 3) surface normals
        knn: FAISS wrapper for nearest neighbor search
        k: Number of neighbors for local plane estimation
        
    Returns:
        projected: (M, 3) projected candidate positions
        projected_normals: (M, 3) interpolated normals at projected positions
    """
    if len(candidates) == 0:
        return candidates, torch.zeros((0, 3), device=candidates.device)
    
    device = candidates.device
    M = len(candidates)
    
    # Find k nearest original points for each candidate
    _, indices = knn.query(candidates, k=k)  # (M, k)
    
    # Get neighbor positions and normals
    neighbor_points = points[indices]  # (M, k, 3)
    neighbor_normals = normals[indices]  # (M, k, 3)
    
    # Compute local plane center and normal
    plane_center = neighbor_points.mean(dim=1)  # (M, 3)
    plane_normal = F.normalize(neighbor_normals.mean(dim=1), dim=1)  # (M, 3)
    
    # Project candidates onto local planes
    # projection = candidate - dot(candidate - plane_center, normal) * normal
    to_center = candidates - plane_center  # (M, 3)
    dist_to_plane = torch.sum(to_center * plane_normal, dim=1, keepdim=True)  # (M, 1)
    projected = candidates - dist_to_plane * plane_normal  # (M, 3)
    
    # Interpolate normals (already computed as plane_normal)
    projected_normals = plane_normal
    
    return projected, projected_normals


def merge_anchors(
    original_points: torch.Tensor,
    original_normals: torch.Tensor,
    original_probs: torch.Tensor,
    candidate_points: torch.Tensor,
    candidate_normals: torch.Tensor,
    parent_sparsity: torch.Tensor,
    sparsity_weight: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Merge original and candidate anchors with importance weights.
    
    Args:
        original_points: (N, 3) original anchor positions
        original_normals: (N, 3) original normals
        original_probs: (N,) original importance probabilities
        candidate_points: (M, 3) candidate positions
        candidate_normals: (M, 3) candidate normals
        parent_sparsity: (M,) sparsity scores of parent points
        sparsity_weight: Weight for candidate importance (relative to parent sparsity)
        
    Returns:
        merged_points: (N+M, 3) all anchor positions
        merged_normals: (N+M, 3) all normals
        merged_probs: (N+M,) all importance probabilities
    """
    if len(candidate_points) == 0:
        # No candidates, return original
        return original_points, original_normals, original_probs
    
    # Candidate probabilities based on parent sparsity
    # Higher sparsity -> higher importance for candidates
    candidate_probs = parent_sparsity * sparsity_weight
    
    # Concatenate
    merged_points = torch.cat([original_points, candidate_points], dim=0)
    merged_normals = torch.cat([original_normals, candidate_normals], dim=0)
    merged_probs = torch.cat([original_probs, candidate_probs], dim=0)
    
    return merged_points, merged_normals, merged_probs


def redistribute_anchors(
    points: torch.Tensor,
    normals: torch.Tensor,
    surf_prob: torch.Tensor,
    spacing: torch.Tensor,
    knn,
    cfg: Optional[Dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Main anchor redistribution algorithm.
    
    Identifies sparse regions and generates new anchors to improve surface coverage.
    
    Args:
        points: (N, 3) original anchor positions
        normals: (N, 3) surface normals
        surf_prob: (N,) surface importance probabilities
        spacing: (N,) local point spacing
        knn: FAISS wrapper for nearest neighbor search
        cfg: Configuration dictionary with parameters:
            - enabled: bool, enable/disable redistribution (default: True)
            - k_density: int, neighbors for density estimation (default: 16)
            - sparsity_alpha: float, sigmoid sharpness (default: 5.0)
            - sparsity_threshold_detect: float, density threshold for sparse detection (default: 0.7)
            - sparsity_threshold_generate: float, minimum sparsity to generate candidates (default: 0.3)
            - n_candidates: int, candidates per sparse point (default: 4)
            - radius_min: float, min radius as multiple of spacing (default: 1.0)
            - radius_max: float, max radius as multiple of spacing (default: 2.0)
            - k_project: int, neighbors for surface projection (default: 8)
            - candidate_weight: float, importance weight for candidates (default: 0.5)
            
    Returns:
        points_new: (N', 3) redistributed anchor positions
        normals_new: (N', 3) normals at new positions
        probs_new: (N',) importance probabilities for new anchors
    """
    if cfg is None:
        cfg = {}
    
    # Check if enabled
    if not cfg.get("enabled", True):
        return points, normals, surf_prob
    
    # Extract parameters
    k_density = cfg.get("k_density", 16)
    sparsity_alpha = cfg.get("sparsity_alpha", 5.0)
    sparsity_threshold_detect = cfg.get("sparsity_threshold_detect", 0.7)
    sparsity_threshold_generate = cfg.get("sparsity_threshold_generate", 0.3)
    n_candidates = cfg.get("n_candidates", 4)
    radius_min = cfg.get("radius_min", 1.0)
    radius_max = cfg.get("radius_max", 2.0)
    k_project = cfg.get("k_project", 8)
    candidate_weight = cfg.get("candidate_weight", 0.5)
    
    # Step 1: Estimate local density
    density, _ = estimate_local_density(points, knn, k=k_density)
    
    # Step 2: Compute sparsity scores
    sparsity = compute_sparsity_scores(
        density, 
        alpha=sparsity_alpha, 
        threshold=sparsity_threshold_detect
    )
    
    # Step 3: Generate candidates in sparse regions
    candidates, parent_indices = generate_surface_candidates(
        points=points,
        normals=normals,
        spacing=spacing,
        sparsity=sparsity,
        n_candidates_per_point=n_candidates,
        radius_min_factor=radius_min,
        radius_max_factor=radius_max,
        sparsity_threshold=sparsity_threshold_generate,
    )
    
    # Step 4: Project candidates onto surface
    if len(candidates) > 0:
        projected_candidates, projected_normals = project_to_surface(
            candidates=candidates,
            points=points,
            normals=normals,
            knn=knn,
            k=k_project,
        )
        
        # Get parent sparsity for weighting
        parent_sparsity = sparsity[parent_indices]
    else:
        projected_candidates = candidates
        projected_normals = torch.zeros((0, 3), device=points.device)
        parent_sparsity = torch.zeros((0,), device=points.device)
    
    # Step 5: Merge with original anchors
    points_new, normals_new, probs_new = merge_anchors(
        original_points=points,
        original_normals=normals,
        original_probs=surf_prob,
        candidate_points=projected_candidates,
        candidate_normals=projected_normals,
        parent_sparsity=parent_sparsity,
        sparsity_weight=candidate_weight,
    )
    
    return points_new, normals_new, probs_new


__all__ = [
    "redistribute_anchors",
    "estimate_local_density",
    "compute_sparsity_scores",
    "generate_surface_candidates",
    "project_to_surface",
    "merge_anchors",
]