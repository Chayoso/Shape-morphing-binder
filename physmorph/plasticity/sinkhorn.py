"""Entropic-OT (Sinkhorn) rest-state migration. eq (16)-(19).

Replaces Chamfer-NN plasticity (song2026 shows Chamfer fails in 3D); set
hard=True for the Chamfer/NN ablation. Target is anchor-subsampled for scale.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree


def _ot_disp(x, y, eps_frac, iters, n_anchor, device, seed):
    """Core entropic-OT barycentric displacement on (possibly small) x."""
    xt = torch.as_tensor(x, device=device)
    yt = torch.as_tensor(np.ascontiguousarray(y, np.float32), device=device)
    if yt.shape[0] > n_anchor:
        gcpu = torch.Generator().manual_seed(seed)
        yt = yt[torch.randperm(yt.shape[0], generator=gcpu)[:n_anchor]]
    C = torch.cdist(xt, yt) ** 2
    eps = float(eps_frac) * C.median().clamp_min(1e-6)
    N, M = C.shape
    loga = torch.full((N,), -np.log(N), device=device)
    logb = torch.full((M,), -np.log(M), device=device)
    f = torch.zeros(N, device=device)
    g = torch.zeros(M, device=device)
    for _ in range(iters):
        f = -eps * torch.logsumexp((g[None, :] - C) / eps + logb[None, :], dim=1)
        g = -eps * torch.logsumexp((f[:, None] - C) / eps + loga[:, None], dim=0)
    pi = torch.exp((f[:, None] + g[None, :] - C) / eps + loga[:, None] + logb[None, :])
    bary = (pi @ yt) / pi.sum(1, keepdim=True).clamp_min(1e-12)
    return (bary - xt).cpu().numpy().astype(np.float32)


def sinkhorn_displacement(x, y, eps_frac=0.05, iters=60, n_anchor=2048,
                          hard=False, n_src=8000, device="cuda", seed=0) -> np.ndarray:
    """Barycentric OT displacement d_i = (sum_j pi_ij y_j)/(sum_j pi_ij) - x_i.

    For large N the OT is computed on a random source subsample (n_src) and the
    smooth displacement field is interpolated to all particles (fast + smoother).
    """
    x = np.ascontiguousarray(x, np.float32)
    if hard:
        _, idx = cKDTree(y).query(x, k=1)
        return (y[idx] - x).astype(np.float32)
    N = x.shape[0]
    if N <= n_src:
        return _ot_disp(x, y, eps_frac, iters, n_anchor, device, seed)
    rng = np.random.default_rng(seed)
    sub = rng.choice(N, n_src, replace=False)
    d_sub = _ot_disp(x[sub], y, eps_frac, iters, n_anchor, device, seed)
    _, nn = cKDTree(x[sub]).query(x, k=4)              # interpolate to all (smooth)
    return d_sub[nn].mean(1).astype(np.float32)


def displacement_jacobian(x, d, k=16, diffusion_iters=2, ridge=1e-4, symmetrize=True) -> np.ndarray:
    """Smoothed least-squares Jacobian of the displacement field -> dFp.

    J = (sum dd dx^T)(sum dx dx^T)^-1 recovers an affine field exactly. eq (18).
    symmetrize=True (v1 default) drops the rotation part — fine for the designed OT fields the
    v1 callers feed in, but NOT objective for realised motion (sym(R-I) != 0 for a rigid
    rotation); commit-time assimilation must use symmetrize=False (see plasticity/assimilation).
    """
    x = np.ascontiguousarray(x, np.float32)
    d = np.ascontiguousarray(d, np.float32)
    _, idx = cKDTree(x).query(x, k=k + 1)
    idx = idx[:, 1:]
    for _ in range(diffusion_iters):
        d = 0.5 * d + 0.5 * d[idx].mean(1)
    dx = x[idx] - x[:, None, :]
    dd = d[idx] - d[:, None, :]
    M = np.einsum("nki,nkj->nij", dx, dx) / idx.shape[1] + ridge * np.eye(3, dtype=np.float32)
    B = np.einsum("nki,nkj->nij", dd, dx) / idx.shape[1]
    J = np.einsum("nij,njk->nik", B, np.linalg.inv(M))
    if symmetrize:
        J = 0.5 * (J + np.transpose(J, (0, 2, 1)))
    return J.astype(np.float32)


def update_fp(Fp, dFp, eta=0.2, smin=0.5, smax=2.0, isochoric=True) -> np.ndarray:
    """Fp <- (I + eta dFp) Fp, then SVD-clamp + isochoric projection. eq (19)."""
    N = Fp.shape[0]
    Fp_new = np.einsum("nij,njk->nik", np.eye(3, dtype=np.float32)[None] + eta * dFp, Fp)
    U, S, Vt = np.linalg.svd(Fp_new.astype(np.float32))
    S = np.clip(S, smin, smax)
    if isochoric:
        S = S / np.cbrt(np.prod(S, axis=1, keepdims=True).clip(1e-6))
    return np.einsum("nij,nj,njk->nik", U, S, Vt).astype(np.float32)
