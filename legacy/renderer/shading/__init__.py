"""Shading system for surface illumination."""

from .config import LightConfig
from .compute import compute_shading
from .models import (
    compute_lambert_shading,
    compute_phong_shading,
    compute_diffuse_term,
    compute_specular_term,
)
from .normals import (
    orient_normals,
    orient_normals_toward_reference,
)

__all__ = [
    # Config
    "LightConfig",
    
    # Main API
    "compute_shading",
    
    # Models
    "compute_lambert_shading",
    "compute_phong_shading",
    "compute_diffuse_term",
    "compute_specular_term",
    
    # Normals
    "orient_normals",
    "orient_normals_toward_reference",
]