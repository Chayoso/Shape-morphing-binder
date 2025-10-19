"""
Soft volume filtering using normal consistency.

Key idea:
- Surface points: normals align with neighbors
- Interior points: normals are chaotic (random orientation)

We use sigmoid(avg_alignment - threshold) for differentiable filtering.
"""

import torch
from typing import Dict, Tuple


def compute_normal_consistency(
    normals: torch.Tensor,
    knn,
    k: int
) -> torch.Tensor:
    """
    Compute average normal alignment with neighbors.
    
    Args:
        normals: (N, 3) surface normals
        knn: KNN function
        k: Number of neighbors
    
    Returns:
        consistency: (N,) average cosine similarity
    """
    idx, w = knn(normals, normals, k)  # Note: KNN on normals themselves
    neighbor_normals = normals[idx]  # (N, k, 3)
    
    # Cosine similarity
    dots = torch.einsum('nd,nkd->nk', normals, neighbor_normals)  # (N, k)
    
    # Weighted average
    avg_alignment = (w * dots).sum(dim=1)  # (N,)
    
    return avg_alignment


def apply_volume_filter(
    surf_prob: torch.Tensor,
    normals: torch.Tensor,
    x: torch.Tensor,
    knn,
    cfg: Dict
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply soft volume filtering to surface probability.
    
    Args:
        surf_prob: (N,) surface probabilities
        normals: (N, 3) surface normals
        x: (N, 3) point positions
        knn: KNN function
        cfg: Configuration dict
    
    Returns:
        filtered_prob: (N,) filtered probabilities
        volume_weight: (N,) volume weights (for debugging)
    """
    k = int(cfg.get("k", 48))
    threshold = float(cfg.get("consistency_threshold", 0.25))
    temperature = float(cfg.get("temperature", 15.0))
    positive_only = bool(cfg.get("positive_only", True))
    use_distance_weight = bool(cfg.get("use_distance_weight", False))
    
    # Compute consistency using spatial neighbors (not normal-space)
    idx_result = knn(x, x, k)
    
    # Handle different return types from KNN
    if isinstance(idx_result, tuple) and len(idx_result) == 2:
        idx, w_or_dist = idx_result
        
        # Check if second return is distance or weight
        if use_distance_weight:
            # Assume it's distance, convert to weight
            dist = w_or_dist
            bandwidth = float(cfg.get("distance_bandwidth", 0.1))
            w = torch.exp(-dist / bandwidth)
            w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
        else:
            # Use as-is (assume it's weight)
            w = w_or_dist
    else:
        raise ValueError(f"Unexpected KNN return type: {type(idx_result)}")
    
    neighbor_normals = normals[idx]  # (N, k, 3)
    
    # Cosine similarity
    dots = torch.einsum('nd,nkd->nk', normals, neighbor_normals)  # (N, k)
    
    # 🔥 NEW: Ignore opposite-facing normals (thin structures)
    if positive_only:
        dots = torch.clamp(dots, min=0)
        # Now back-facing neighbors (dots < 0) become 0
    
    # Weighted average alignment
    avg_alignment = (w * dots).sum(dim=1)  # (N,)
    
    # 🔥 DEBUG
    verbose = cfg.get("debug", {}).get("verbose", False)
    if verbose:
        print(f"  === Volume Filter Debug ===")
        print(f"  k={k}, threshold={threshold:.3f}, temp={temperature:.1f}")
        print(f"  positive_only={positive_only}, use_distance={use_distance_weight}")
        print(f"  Avg alignment: mean={avg_alignment.mean():.3f}, "
              f"min={avg_alignment.min():.3f}, max={avg_alignment.max():.3f}")
        
        if positive_only:
            dots_raw = torch.einsum('nd,nkd->nk', normals, neighbor_normals)
            negative_ratio = (dots_raw < 0).float().mean()
            print(f"  Negative dot products: {negative_ratio*100:.1f}% (clamped to 0)")
    
    # Soft threshold (differentiable)
    # avg_alignment > threshold → weight ≈ 1 (surface)
    # avg_alignment < threshold → weight ≈ 0 (interior)
    volume_weight = torch.sigmoid((avg_alignment - threshold) * temperature)
    
    if verbose:
        print(f"  Volume weight: mean={volume_weight.mean():.3f}, "
              f"min={volume_weight.min():.3f}, max={volume_weight.max():.3f}")
        
        # Surface points after filtering
        surface_mask = volume_weight > 0.5
        print(f"  Surface points (weight>0.5): {surface_mask.sum()}/{len(volume_weight)} "
              f"({100.0 * surface_mask.sum() / len(volume_weight):.1f}%)")
        print(f"  ===========================\n")
    
    # Modulate probability
    filtered_prob = surf_prob * volume_weight
    filtered_prob = filtered_prob / (filtered_prob.sum() + 1e-12)
    
    return filtered_prob, volume_weight


__all__ = [
    "apply_volume_filter",
    "compute_normal_consistency",
]