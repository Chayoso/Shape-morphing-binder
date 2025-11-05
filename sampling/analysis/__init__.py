"""
Point cloud analysis and feature detection.

Includes:
- K-nearest neighbors search (Hybrid FAISS)
- Level Set φ for differentiable surface representation
"""

# ============================================================================
# KNN (Hybrid FAISS)
# ============================================================================
from .knn import (
    HybridFAISSKNN,
    FAISS_AVAILABLE,
)

# ============================================================================
# Level Set (Differentiable Surface Representation)
# ============================================================================
from .levelset import (
    LevelSetGrid,
    extract_surface_anchors,
)


__all__ = [
    # KNN
    "HybridFAISSKNN",
    "FAISS_AVAILABLE",
    
    # Level Set
    "LevelSetGrid",
    "extract_surface_anchors",
]