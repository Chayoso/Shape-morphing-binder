"""
Analytical F_plastic solver for surface-aligned Gaussian appearance.

Design:
  - F from physics = F_elastic  (C++ returns this as get_def_grads_total())
  - F_plastic = Python-side (N,3,3) array, initialized to I
  - F_total = F_elastic @ F_plastic  (for rendering)

Goal: shape-only alignment, no scale collapse.
  Sigma_target has det=1 (pure anisotropy); absolute scale via sigma0.

  F_target = Sigma_target^{1/2}          (SVs: tangent >1, normal <1)
  F_plastic* = F_elastic^{-1} @ F_target
  → F_total = F_target  →  Σ = sigma0^2 · Sigma_target  ✓

Update rule (relaxed blend):
  F_plastic^{k+1} = (1 - η) · F_plastic^k + η · F_plastic*

Stability:
  - F_elastic SVD-clamped before inversion
  - F_plastic SVs clamped to [sv_min, sv_max]
  - det(F_plastic) clamped to [det_min, det_max] via uniform rescale
"""

import numpy as np


def _matrix_sqrt_spd(A: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    SPD matrix square root via eigendecomposition.
    A = V D V^T  →  sqrt(A) = V sqrt(D) V^T

    Args:
        A: (..., 3, 3) symmetric positive definite
    Returns:
        (..., 3, 3) square root
    """
    vals, vecs = np.linalg.eigh(A)          # ascending eigenvalues
    vals = np.maximum(vals, eps)
    sqrt_vals = np.sqrt(vals)
    return np.einsum('...ij,...j,...kj->...ik', vecs, sqrt_vals, vecs)


def _invert_F_clamped(F: np.ndarray, sv_min: float = 0.2) -> np.ndarray:
    """
    Invert F with SVD clamping for numerical stability.

    Args:
        F:      (N, 3, 3) matrices
        sv_min: minimum singular value (prevents near-singular inversion)
    Returns:
        (N, 3, 3) inverses
    """
    U, S, Vh = np.linalg.svd(F)
    S_clamped = np.maximum(S, sv_min)          # (N, 3)
    S_inv = 1.0 / S_clamped                    # (N, 3)
    # F^{-1} = Vh^T @ diag(S_inv) @ U^T
    return np.einsum('nij,nj,njk->nik',
                     Vh.transpose(0, 2, 1), S_inv, U.transpose(0, 2, 1))


def solve_F_plastic(
    F_elastic: np.ndarray,    # (N, 3, 3) physics F
    Sigma_target: np.ndarray, # (N, 3, 3) desired covariance (SPD, det=1)
    F_plastic: np.ndarray,    # (N, 3, 3) current plastic state
    eta: float = 0.1,         # blend rate  0 < η ≤ 1
    sv_min: float = 0.3,      # F_plastic SV lower clamp
    sv_max: float = 3.0,      # F_plastic SV upper clamp
    sv_elastic_min: float = 0.2,  # F_elastic inversion stability
    det_min: float = 0.5,     # det(F_plastic) lower bound
    det_max: float = 2.0,     # det(F_plastic) upper bound
) -> np.ndarray:
    """
    Analytical F_plastic update.

    Sigma_target must have det=1 (pure anisotropy).
    Absolute scale is controlled by sigma0 in the upsample config.

    Returns:
        F_plastic_new: (N, 3, 3) updated plastic deformation
    """
    # Step 1: F_target = Sigma_target^{1/2}  (SVs: tangent>1, normal<1 for r>1)
    F_target = _matrix_sqrt_spd(Sigma_target)    # (N, 3, 3)

    # Step 2: F_elastic^{-1}  (SVD-clamped)
    F_elastic_inv = _invert_F_clamped(F_elastic, sv_min=sv_elastic_min)

    # Step 3: F_plastic* = F_elastic^{-1} @ F_target
    F_plastic_star = np.einsum('nij,njk->nik', F_elastic_inv, F_target)

    # Step 4: Relaxed blend
    F_plastic_new = (1.0 - eta) * F_plastic + eta * F_plastic_star

    # Step 5: SV clamp
    U, S, Vh = np.linalg.svd(F_plastic_new)
    S = np.clip(S, sv_min, sv_max)

    # Step 6: det clamp — uniform rescale to keep volume in [det_min, det_max]
    det_fp = np.prod(S, axis=-1)                              # (N,)
    det_clipped = np.clip(det_fp, det_min, det_max)           # (N,)
    scale = (det_clipped / np.maximum(det_fp, 1e-8)) ** (1.0 / 3.0)  # (N,)
    S = S * scale[:, None]

    F_plastic_new = np.einsum('nij,nj,njk->nik', U, S, Vh)

    return F_plastic_new.astype(np.float32)


def apply_appearance_update(
    x: np.ndarray,                  # (N, 3) particle positions
    F_elastic: np.ndarray,          # (N, 3, 3) physics F from C++
    F_plastic: np.ndarray,          # (N, 3, 3) current plastic state
    surface_points: np.ndarray,     # (M, 3) bunny surface samples
    surface_normals: np.ndarray,    # (M, 3) unit normals
    appearance_cfg: dict,
) -> tuple:                         # (F_plastic_new, diagnostics)
    """
    Full analytical F_plastic update step.

    Returns:
        F_plastic_new: (N, 3, 3)
        diagnostics:   dict with dF_mean, sv_min, sv_max, det_mean, det_min, det_max
    """
    from utils.appearance_target import compute_sigma_target

    sigma_t        = float(appearance_cfg.get('sigma_t',        0.4))
    sigma_n        = float(appearance_cfg.get('sigma_n',        0.08))
    eta            = float(appearance_cfg.get('eta',            0.1))
    sv_min         = float(appearance_cfg.get('sv_min',         0.3))
    sv_max         = float(appearance_cfg.get('sv_max',         3.0))
    sv_elastic_min = float(appearance_cfg.get('sv_elastic_min', 0.2))
    dist_falloff   = float(appearance_cfg.get('dist_falloff',   2.0))
    interior_sigma = float(appearance_cfg.get('interior_sigma', 0.3))
    det_min        = float(appearance_cfg.get('det_min',        0.5))
    det_max        = float(appearance_cfg.get('det_max',        2.0))

    # Compute per-particle target covariance (det-normalized to 1)
    Sigma_target = compute_sigma_target(
        x, surface_points, surface_normals,
        sigma_t=sigma_t, sigma_n=sigma_n,
        dist_falloff=dist_falloff, interior_sigma=interior_sigma,
    )

    # Analytical solve
    F_plastic_new = solve_F_plastic(
        F_elastic, Sigma_target, F_plastic,
        eta=eta, sv_min=sv_min, sv_max=sv_max,
        sv_elastic_min=sv_elastic_min,
        det_min=det_min, det_max=det_max,
    )

    # Diagnostics
    N = x.shape[0]
    dF = np.linalg.norm((F_plastic_new - F_plastic).reshape(N, -1), axis=1)
    _, S_new, _ = np.linalg.svd(F_plastic_new)
    det_new = np.prod(S_new, axis=-1)

    diagnostics = {
        'dF_mean':  float(dF.mean()),
        'dF_max':   float(dF.max()),
        'sv_min':   float(S_new.min()),
        'sv_max':   float(S_new.max()),
        'det_mean': float(det_new.mean()),
        'det_min':  float(det_new.min()),
        'det_max':  float(det_new.max()),
    }

    return F_plastic_new, diagnostics
