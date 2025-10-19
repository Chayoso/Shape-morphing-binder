"""
Point cloud analysis and feature detection.

Includes:
- K-nearest neighbors search (Hybrid FAISS)
- PCA and normal estimation
"""

# ============================================================================
# KNN (Hybrid FAISS)
# ============================================================================
from .knn import (
    HybridFAISSKNN,
    FAISS_AVAILABLE,
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


__all__ = [
    # KNN
    "HybridFAISSKNN",
    "FAISS_AVAILABLE",
    
    # PCA
    "compute_weighted_centroid",
    "compute_weighted_covariance",
    "extract_normal_from_pca",
    "compute_local_spacing",
    "batched_pca_surface_optimized",
]