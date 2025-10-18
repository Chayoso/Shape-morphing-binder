"""
Point cloud analysis and feature detection.

Includes:
- Density estimation and equalization
- PCA and normal estimation
- K-nearest neighbors search (Hybrid FAISS)
- Surface detection and probability computation
"""

# ============================================================================
# KNN (Hybrid FAISS)
# ============================================================================
from .knn import (
    HybridFAISSKNN,
)

# ============================================================================
# PCA
# ============================================================================
from .pca import (
    compute_weighted_centroid,
    compute_weighted_covariance,
    extract_normal_from_pca,
    compute_local_spacing,
    batched_pca_surface_optimized,
)

# ============================================================================
# Detection (Surface Detection)
# ============================================================================
from .detection import (
    soft_quantile,
    compute_surface_threshold,
    compute_surface_probability,
    compute_surface_mask_diff,
)

# ============================================================================
# Density (Density Equalization)
# ============================================================================
from .density import (
    compute_local_density,
    mask_self_neighbors,
    compute_density_displacement,
    density_equalize_diff,
)


__all__ = [
    # KNN
    "HybridFAISSKNN",
    
    # PCA
    "compute_weighted_centroid",
    "compute_weighted_covariance",
    "extract_normal_from_pca",
    "compute_local_spacing",
    "batched_pca_surface_optimized",
    
    # Detection
    "soft_quantile",
    "compute_surface_threshold",
    "compute_surface_probability",
    "compute_surface_mask_diff",
    
    # Density
    "compute_local_density",
    "mask_self_neighbors",
    "compute_density_displacement",
    "density_equalize_diff",
]