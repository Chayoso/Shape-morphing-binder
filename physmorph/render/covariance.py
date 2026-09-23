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


def cov_from_F(F: np.ndarray, sigma0: float, sat: float = 0.0) -> np.ndarray:
    """Sigma = sigma0^2 F F^T (+ tiny jitter). F: (N,3,3) -> (N,3,3).
    sat > 0: stretch saturation M -> M (I + M/sat^2)^-1 (the same forward model as
    pipeline.gauss_loss.saturate_stretch), so viewer/export/photoreal show what the
    objective rendered."""
    s = sigma0 * sigma0
    M = np.einsum("nij,nkj->nik", F, F).astype(np.float64)
    if sat and sat > 0:
        eye = np.eye(3)[None]
        Ms = np.linalg.solve(eye + M / (float(sat) ** 2), M)
        M = 0.5 * (Ms + np.transpose(Ms, (0, 2, 1)))
    cov = (s * M).astype(np.float32)
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


def select_archive_F(d, fi: int, prefer_geom: bool = False):
    """Pick the deformation gradient to render at frame ``fi`` of a pipeline_run archive.

    ``F_samples`` (physics F, sampled at ``F_sample_idx`` frame indices) is the default;
    with ``prefer_geom`` and an archive that carries ``Fg_commits`` (geometric F_g at
    accepted commits; ``Fg_commit_idx[k]`` = len(frames) after that commit, i.e. the
    state's frame index + 1) the latest F_g at or before the frame is used instead —
    the PhysGaussian kinematics the render_F_geom arms optimised against. Returns
    (F (N,3,3) float32, kind) with kind in {"physics", "geom"}."""
    import numpy as np
    files = set(getattr(d, "files", d.keys()))
    if prefer_geom and "Fg_commits" in files and "Fg_commit_idx" in files:
        idx = np.asarray(d["Fg_commit_idx"]) - 1           # frame index of each state
        Fg = d["Fg_commits"]
        if len(idx) and Fg.ndim == 4 and Fg.shape[0] == len(idx):
            k = int(np.searchsorted(idx, fi, side="right") - 1)
            if k >= 0:
                return np.ascontiguousarray(Fg[k], np.float32), "geom"
    if "F_sample_idx" in files:
        sidx = np.asarray(d["F_sample_idx"])
        k = int(np.searchsorted(sidx, fi, side="right") - 1)
        return np.ascontiguousarray(d["F_samples"][max(k, 0)], np.float32), "physics"
    return np.ascontiguousarray(d["F_samples"][-1], np.float32), "physics"
