"""Surface treatment (§3.9): differentiable density-based soft surface weight."""
from .weight import particle_density, surface_weight, outlier_mask

__all__ = ["particle_density", "surface_weight", "outlier_mask"]
