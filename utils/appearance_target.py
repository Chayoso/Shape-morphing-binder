"""
Per-particle target covariance from bunny surface normals.

Design: det-preserving anisotropy (shape only, no scale change)
  - sigma_t / sigma_n  controls ASPECT RATIO only (e.g. 0.4/0.08 = 5:1)
  - Σ_target is normalized so det(Σ_target) = 1
  - Absolute scale is fully controlled by sigma0 in the upsample config

Normalized surface-aligned SVs for aspect ratio r = sigma_t/sigma_n:
  a = r^{1/3}   (tangent, >1 → expand)
  b = r^{-2/3}  (normal,  <1 → compress)
  a²·b = 1 ✓   (det-preserving)

Example: sigma_t=0.4, sigma_n=0.08 → r=5 → a≈1.71, b≈0.34
"""

import numpy as np
from scipy.spatial import KDTree


def _build_tangent_basis(n: np.ndarray) -> tuple:
    """
    Build orthonormal {t1, t2} for each unit normal n.

    Args:
        n: (N, 3) unit normals

    Returns:
        t1, t2: (N, 3) each, orthonormal to n and to each other
    """
    N = n.shape[0]
    helper = np.zeros((N, 3), dtype=np.float32)
    helper[:, 0] = 1.0
    # Where n is close to x-axis, switch to y-axis
    parallel = np.abs(n[:, 0]) > 0.9
    helper[parallel, 0] = 0.0
    helper[parallel, 1] = 1.0

    # t1 = normalize(helper - (helper·n)n)
    proj = np.einsum('ni,ni->n', helper, n)[:, None] * n   # (N, 3)
    t1 = helper - proj
    t1 = t1 / (np.linalg.norm(t1, axis=1, keepdims=True) + 1e-8)

    # t2 = n × t1
    t2 = np.cross(n, t1)
    t2 = t2 / (np.linalg.norm(t2, axis=1, keepdims=True) + 1e-8)

    return t1.astype(np.float32), t2.astype(np.float32)


def compute_sigma_target(
    positions: np.ndarray,        # (N, 3) particle positions
    surface_points: np.ndarray,   # (M, 3) bunny surface samples
    surface_normals: np.ndarray,  # (M, 3) unit normals
    sigma_t: float = 0.4,         # tangent scale (only ratio sigma_t/sigma_n matters)
    sigma_n: float = 0.08,        # normal scale  (only ratio sigma_t/sigma_n matters)
    dist_falloff: float = 2.0,    # blend distance: surface→interior
    interior_sigma: float = 0.3,  # unused (interior → isotropic I, det=1)
) -> np.ndarray:                  # (N, 3, 3) SPD covariance matrices, det=1
    """
    Compute per-particle target covariance Σ_target (det-normalized to 1).

    Surface particles → anisotropic disk: aspect ratio = sigma_t / sigma_n
                        tangent SVs = r^{1/3} > 1  (expand)
                        normal SV   = r^{-2/3} < 1 (compress)
                        det(Σ) = 1  → pure shape change, no scale change

    Interior particles → isotropic: Σ = I  (det=1, scale via sigma0)

    Blend based on distance to surface.
    """
    tree = KDTree(surface_points)
    dists, idxs = tree.query(positions, k=1)    # (N,), (N,)

    # Nearest surface normals
    n = surface_normals[idxs].astype(np.float32)   # (N, 3)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)

    t1, t2 = _build_tangent_basis(n)   # (N, 3) each

    # R_i = [t1, t2, n] as columns → shape (N, 3, 3)
    R = np.stack([t1, t2, n], axis=-1)   # (N, 3, 3)

    # Anisotropy ratio r = sigma_t / sigma_n
    # det-preserving SVs: a = r^{1/3} (tangent), b = r^{-2/3} (normal)
    r = float(sigma_t) / float(sigma_n)
    a = r ** (1.0 / 3.0)    # tangent SV (>1 for r>1)
    b = r ** (-2.0 / 3.0)   # normal  SV (<1 for r>1)

    # Σ_surf = R diag(a², a², b²) R^T  →  det(Σ_surf) = a^4 · b^2 = (r^{2/3})^2 · r^{-4/3} = 1 ✓
    D = np.array([a**2, a**2, b**2], dtype=np.float32)
    Sigma_surf = np.einsum('nij,j,nkj->nik', R, D, R)   # (N, 3, 3)

    # Isotropic interior: I (det=1)
    Sigma_iso = np.eye(3, dtype=np.float32)

    # Blend: w=1 on surface, w=0 at dist_falloff
    w = np.clip(1.0 - dists / dist_falloff, 0.0, 1.0).astype(np.float32)  # (N,)
    w3 = w[:, None, None]

    Sigma = w3 * Sigma_surf + (1.0 - w3) * Sigma_iso   # (N, 3, 3)

    return Sigma.astype(np.float32)
