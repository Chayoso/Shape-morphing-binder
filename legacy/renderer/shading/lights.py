"""Light vector computation."""

from __future__ import annotations
from typing import Tuple
import numpy as np

from .config import LightConfig


EPSILON_NORMALIZE = 1e-9
EPSILON_DISTANCE = 1e-6


def safe_normalize(
    vectors: np.ndarray, 
    axis: int = -1, 
    eps: float = EPSILON_NORMALIZE
) -> np.ndarray:
    """
    Safely normalize vectors along specified axis.
    
    Args:
        vectors: Input vectors (..., D)
        axis: Axis to normalize along
        eps: Small constant to prevent division by zero
    
    Returns:
        Normalized vectors with same shape
    """
    norm = np.linalg.norm(vectors, axis=axis, keepdims=True) + eps
    return vectors / norm


def compute_directional_light_vectors(
    xyz: np.ndarray,
    light_direction: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute light vectors for directional light.
    
    Args:
        xyz: Point positions (N, 3)
        light_direction: Direction from light to scene (3,)
    
    Returns:
        L: Incident light direction (N, 3) - from point toward light
        attenuation: Constant attenuation (N, 1) - all ones
    """
    N = xyz.shape[0]
    
    # Incident direction: opposite of light direction
    L = safe_normalize(-light_direction.reshape(1, 3))
    L = np.repeat(L, N, axis=0)
    
    # No attenuation for directional lights
    attenuation = np.ones((N, 1), dtype=np.float32)
    
    return L, attenuation


def compute_point_light_vectors(
    xyz: np.ndarray,
    light_position: np.ndarray,
    attenuation_coeffs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute light vectors for point light.
    
    Args:
        xyz: Point positions (N, 3)
        light_position: Light position in world space (3,)
        attenuation_coeffs: [c0, c1, c2] for 1/(c0 + c1*d + c2*d²)
    
    Returns:
        L: Incident light direction (N, 3) - from point toward light
        attenuation: Distance-based attenuation (N, 1)
    """
    # Direction from point to light
    light_pos = light_position.reshape(1, 3)
    vec = light_pos - xyz
    
    # Distance with safety epsilon
    distance = np.linalg.norm(vec, axis=1, keepdims=True) + EPSILON_DISTANCE
    
    # Normalized direction
    L = vec / distance
    
    # Attenuation: 1 / (c0 + c1*d + c2*d²)
    c0, c1, c2 = attenuation_coeffs
    attenuation = 1.0 / (c0 + c1 * distance + c2 * (distance ** 2))
    
    return L.astype(np.float32), attenuation.astype(np.float32)


def compute_light_and_view_vectors(
    xyz: np.ndarray,
    camera_pos: np.ndarray,
    light_cfg: LightConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute light and view direction vectors.
    
    Args:
        xyz: Surface point positions (N, 3)
        camera_pos: Camera position in world space (3,)
        light_cfg: Light configuration
    
    Returns:
        L: Light direction vectors (N, 3) - from point toward light
        V: View direction vectors (N, 3) - from point toward camera
        attenuation: Per-point attenuation factors (N, 1)
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    camera_pos = np.asarray(camera_pos, dtype=np.float32).reshape(1, 3)
    
    # Compute light vectors based on type
    light_type = light_cfg.type.lower().strip()
    
    if light_type == 'directional':
        L, attenuation = compute_directional_light_vectors(xyz, light_cfg.direction)
    elif light_type == 'point':
        L, attenuation = compute_point_light_vectors(
            xyz, light_cfg.position, light_cfg.attenuation
        )
    else:
        raise ValueError(f"Unknown light type: {light_type}")
    
    # Compute view vectors
    V = safe_normalize(camera_pos - xyz)
    
    return L, V, attenuation