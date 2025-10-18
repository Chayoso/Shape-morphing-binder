"""Main shading computation."""

from __future__ import annotations
from typing import Optional, Dict, Any, Union
import numpy as np

from .config import LightConfig, DEFAULT_LIGHT_COLOR
from .lights import compute_light_and_view_vectors
from .normals import orient_normals
from .models import compute_lambert_shading, compute_phong_shading


DEFAULT_ALBEDO_COLOR = np.array([0.7, 0.7, 0.7], dtype=np.float32)


def compute_shading(
    xyz: np.ndarray,
    normals: np.ndarray,
    camera_pos: np.ndarray,
    light_cfg: Optional[Dict[str, Any]] = None,
    albedo_color: Union[tuple, list, np.ndarray] = DEFAULT_ALBEDO_COLOR,
    model: str = 'phong'
) -> np.ndarray:
    """
    Compute per-point RGB colors with local illumination.
    
    Args:
        xyz: Surface point positions (N, 3)
        normals: Surface normals (N, 3) - will be normalized and oriented
        camera_pos: Camera position in world space (3,)
        light_cfg: Light configuration dict (or None for defaults)
        albedo_color: Base surface color (3,) or (N, 3)
        model: Shading model - 'lambert' or 'phong'
    
    Returns:
        RGB colors (N, 3) in [0, 1]
    
    Raises:
        ValueError: If model is unknown
    
    Examples:
        >>> # Simple directional light
        >>> rgb = compute_shading(
        ...     xyz=points,
        ...     normals=point_normals,
        ...     camera_pos=[0, 2, -5],
        ...     light_cfg={'type': 'directional', 'direction': [0, -1, 0]},
        ...     model='phong'
        ... )
    """
    # Validate inputs
    xyz = np.asarray(xyz, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    camera_pos = np.asarray(camera_pos, dtype=np.float32)
    
    if xyz.shape[0] != normals.shape[0]:
        raise ValueError(f"xyz and normals must have same length: {xyz.shape[0]} != {normals.shape[0]}")
    
    # Parse light configuration
    if light_cfg is None:
        light_cfg = {}
    
    light_config = LightConfig.from_dict(light_cfg)
    
    # Prepare albedo
    albedo = np.asarray(albedo_color, dtype=np.float32).reshape(-1)
    if albedo.shape[0] == 1:
        albedo = np.tile(albedo, 3)
    elif albedo.shape[0] != 3:
        raise ValueError(f"albedo_color must be (3,) or scalar, got {albedo.shape}")
    
    albedo = albedo.reshape(1, 3).astype(np.float32)
    
    # Compute light and view vectors
    L, V, attenuation = compute_light_and_view_vectors(xyz, camera_pos, light_config)
    
    # Orient normals
    N = orient_normals(normals, L, V, mode=light_config.orient)
    
    # Compute shading based on model
    model = model.lower().strip()
    
    if model == 'lambert':
        light_term = compute_lambert_shading(N, L, attenuation, light_config)
    elif model == 'phong':
        light_term = compute_phong_shading(N, L, V, attenuation, light_config)
    else:
        raise ValueError(f"Unknown shading model: {model}")
    
    # Combine with albedo and light color
    light_color = light_config.color.reshape(1, 3)
    rgb = albedo * light_color * light_term
    
    # Clamp to valid range
    rgb = np.clip(rgb, 0.0, 1.0)
    
    # Broadcast to match number of points
    if rgb.shape[0] == 1 and xyz.shape[0] > 1:
        rgb = np.repeat(rgb, xyz.shape[0], axis=0)
    
    return rgb.astype(np.float32)