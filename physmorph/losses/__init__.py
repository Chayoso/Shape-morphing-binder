"""Differentiable losses (torch, on the device): mass rasterisation and grid terms,
the grid Sinkhorn transport, the surface proximity, the silhouette rasteriser."""
from .volumetric import rasterize_mass, d_vol, target_mass_grid

__all__ = ["rasterize_mass", "d_vol", "target_mass_grid"]
