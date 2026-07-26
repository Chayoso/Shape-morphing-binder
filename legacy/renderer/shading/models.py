"""Shading models (Lambert, Phong)."""

from __future__ import annotations
import numpy as np

from .config import LightConfig
from .lights import safe_normalize


def compute_diffuse_term(
    N: np.ndarray,
    L: np.ndarray,
    kd: float,
    two_sided: bool = True
) -> np.ndarray:
    """
    Compute Lambertian diffuse term.
    
    Formula:
        diffuse = kd * max(0, N·L)  (one-sided)
        diffuse = kd * |N·L|        (two-sided)
    
    Args:
        N: Surface normals (N, 3)
        L: Light directions (N, 3)
        kd: Diffuse coefficient [0, 1]
        two_sided: Use absolute value for thin surfaces
    
    Returns:
        Diffuse term (N, 1)
    """
    ndotl = (N * L).sum(axis=1, keepdims=True)
    
    if two_sided:
        ndotl = np.abs(ndotl)
    else:
        ndotl = np.clip(ndotl, 0.0, 1.0)
    
    diffuse = kd * ndotl
    return diffuse.astype(np.float32)


def compute_specular_term(
    N: np.ndarray,
    L: np.ndarray,
    V: np.ndarray,
    ks: float,
    shininess: float,
    two_sided: bool = True
) -> np.ndarray:
    """
    Compute Blinn-Phong specular term.
    
    Formula:
        H = normalize(L + V)
        specular = ks * (max(0, N·H))^shininess
    
    Args:
        N: Surface normals (N, 3)
        L: Light directions (N, 3)
        V: View directions (N, 3)
        ks: Specular coefficient [0, 1]
        shininess: Specular exponent (controls tightness)
        two_sided: Use absolute value for thin surfaces
    
    Returns:
        Specular term (N, 1)
    """
    if ks <= 0.0:
        return np.zeros((N.shape[0], 1), dtype=np.float32)
    
    # Half-vector (Blinn-Phong)
    H = safe_normalize(L + V)
    
    # N·H
    ndoth = (N * H).sum(axis=1, keepdims=True)
    
    if two_sided:
        ndoth = np.abs(ndoth)
    else:
        ndoth = np.clip(ndoth, 0.0, 1.0)
    
    # Specular power
    specular = ks * (ndoth ** shininess)
    
    return specular.astype(np.float32)


def compute_lambert_shading(
    N: np.ndarray,
    L: np.ndarray,
    attenuation: np.ndarray,
    light_cfg: LightConfig
) -> np.ndarray:
    """
    Compute Lambert (diffuse-only) shading.
    
    Args:
        N: Oriented normals (N, 3)
        L: Light directions (N, 3)
        attenuation: Per-point attenuation (N, 1)
        light_cfg: Light configuration
    
    Returns:
        Light contribution (N, 1)
    """
    ka = light_cfg.ambient
    kd = light_cfg.diffuse
    intensity = light_cfg.intensity
    two_sided = light_cfg.two_sided
    min_ambient = light_cfg.min_ambient
    
    # Diffuse term
    diffuse = compute_diffuse_term(N, L, kd, two_sided)
    
    # Combine: ambient + attenuated diffuse
    light_term = ka + attenuation * intensity * diffuse
    
    # Apply minimum ambient floor
    light_term = np.maximum(light_term, min_ambient)
    
    return light_term.astype(np.float32)


def compute_phong_shading(
    N: np.ndarray,
    L: np.ndarray,
    V: np.ndarray,
    attenuation: np.ndarray,
    light_cfg: LightConfig
) -> np.ndarray:
    """
    Compute Blinn-Phong shading.
    
    Args:
        N: Oriented normals (N, 3)
        L: Light directions (N, 3)
        V: View directions (N, 3)
        attenuation: Per-point attenuation (N, 1)
        light_cfg: Light configuration
    
    Returns:
        Light contribution (N, 1)
    """
    ka = light_cfg.ambient
    kd = light_cfg.diffuse
    ks = light_cfg.specular
    shininess = light_cfg.shininess
    intensity = light_cfg.intensity
    two_sided = light_cfg.two_sided
    min_ambient = light_cfg.min_ambient
    
    # Diffuse term
    diffuse = compute_diffuse_term(N, L, kd, two_sided)
    
    # Specular term
    specular = compute_specular_term(N, L, V, ks, shininess, two_sided)
    
    # Combine: ambient + attenuated (diffuse + specular)
    light_term = ka + attenuation * intensity * (diffuse + specular)
    
    # Apply minimum ambient floor
    light_term = np.maximum(light_term, min_ambient)
    
    return light_term.astype(np.float32)