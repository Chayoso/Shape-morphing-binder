"""Normal orientation utilities."""

from __future__ import annotations
import numpy as np

from .lights import safe_normalize, EPSILON_NORMALIZE


def orient_normals_toward_reference(
    normals: np.ndarray,
    reference: np.ndarray,
    eps: float = EPSILON_NORMALIZE
) -> np.ndarray:
    """
    Flip normals to face reference direction.
    
    Args:
        normals: Input normals (N, 3)
        reference: Reference direction (N, 3)
        eps: Small constant for numerical stability
    
    Returns:
        Oriented normals (N, 3)
    """
    N = safe_normalize(normals)
    
    # Compute dot product with reference
    dot = (N * reference).sum(axis=1, keepdims=True)
    
    # Flip if facing away (dot < 0)
    flip_sign = np.sign(dot + eps)
    
    return (N * flip_sign).astype(np.float32)


def orient_normals(
    normals: np.ndarray,
    L: np.ndarray,
    V: np.ndarray,
    mode: str = 'view'
) -> np.ndarray:
    """
    Orient normals consistently based on view or light direction.
    
    Args:
        normals: Raw normals (N, 3) - sign ambiguous from PCA
        L: Light direction (N, 3)
        V: View direction (N, 3)
        mode: 'view' (face camera) or 'light' (face light)
    
    Returns:
        Oriented normals (N, 3)
    """
    mode = (mode or 'view').lower().strip()
    
    if mode == 'light':
        reference = L
    else:  # 'view' (default)
        reference = V
    
    return orient_normals_toward_reference(normals, reference)