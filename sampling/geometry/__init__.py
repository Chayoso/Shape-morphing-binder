"""
Geometric operations and interpolation.

Includes:
- Moving Least Squares (MLS) surface projection
- Oriented-point implicit surface reconstruction
"""

from .mls import (
    compute_mls_signed_distance,
    compute_mls_normal,
    project_to_mls_surface_diff,
)

__all__ = [
    "compute_mls_signed_distance",
    "compute_mls_normal",
    "project_to_mls_surface_diff",
]