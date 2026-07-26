"""Gaussian covariance from F + decomposition to scale/quaternion. eq (11).

Sigma = sigma0^2 F F^T. Decompose via eigh -> (scales, rotation) for 3DGS .ply.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def sigma0_from_nn(x: np.ndarray, scale: float = 0.7) -> float:
    """Rest Gaussian size from mean nearest-neighbour spacing. sigma0 = scale*d_nn."""
    d, _ = cKDTree(x).query(x, k=2)
    return float(scale * np.median(d[:, 1]))


def cov_from_F(F: np.ndarray, sigma0: float) -> np.ndarray:
    """Sigma = sigma0^2 F F^T (+ tiny jitter). F: (N,3,3) -> (N,3,3)."""
    s = sigma0 * sigma0
    cov = s * np.einsum("nij,nkj->nik", F, F).astype(np.float32)
    cov += 1e-8 * np.eye(3, dtype=np.float32)[None]
    return cov


def decompose_cov(cov: np.ndarray):
    """Sigma -> (scales (N,3) float32, quats WXYZ (N,4) float32).

    Sigma = R diag(scales^2) R^T with R proper rotation (det=+1).
    """
    C = 0.5 * (cov + np.transpose(cov, (0, 2, 1)))
    w, V = np.linalg.eigh(C.astype(np.float64))       # ascending eigenvalues
    scales = np.sqrt(np.clip(w, 1e-12, None)).astype(np.float32)
    # ensure proper rotation: flip a column where det < 0
    det = np.linalg.det(V)
    V[det < 0, :, 0] *= -1.0
    q_xyzw = Rotation.from_matrix(V).as_quat()         # (N,4) XYZW
    q_wxyz = q_xyzw[:, [3, 0, 1, 2]].astype(np.float32)
    return scales, q_wxyz
