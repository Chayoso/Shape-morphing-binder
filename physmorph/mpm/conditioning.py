"""F conditioning: SVD singular-value clamp with reflection REPAIR.

The v1 morph loops rebuilt U diag(clip(S)) Vᵀ directly — but numpy's SVD may return an
improper (U, Vᵀ) pair (det(U)det(Vᵀ) = −1), and clamping the positive singular values keeps
the reflection, so an inverted F stayed inverted (det −27 → det −8). Flipping the last
column of U when the pair is improper (exactly what corotated_R does in constitutive.py)
makes the output a genuine orientation-preserving deformation. Non-finite rows are reset to
identity first (numpy's SVD raises on NaN input).
"""
from __future__ import annotations

import numpy as np


def batched_svd(F):
    """SVD of an (N,3,3) float array — torch on the GPU when available (150k: 0.47 s in
    LAPACK per call, ~20 ms batched on the GPU), numpy otherwise. Same decomposition up to
    float precision and the usual sign freedom of U/V (products are invariant)."""
    F = np.asarray(F)
    try:
        import torch
        if torch.cuda.is_available() and F.shape[0] >= 20000:
            Ft = torch.as_tensor(np.ascontiguousarray(F, np.float32), device="cuda")
            U, S, Vh = torch.linalg.svd(Ft)
            return U.cpu().numpy(), S.cpu().numpy(), Vh.cpu().numpy()
    except Exception:
        pass
    return np.linalg.svd(F)


def batched_det(F):
    """det of an (..., 3, 3) array — torch on the GPU for large batches, numpy otherwise."""
    F = np.asarray(F)
    try:
        import torch
        if torch.cuda.is_available() and F.size >= 20000 * 9:
            return torch.linalg.det(torch.as_tensor(np.ascontiguousarray(F, np.float32), device="cuda")).cpu().numpy()
    except Exception:
        pass
    return np.linalg.det(F)



def condition_F(F, smin=0.5, smax=2.0, clamp=True):
    """Return (F_repaired, n_nonfinite_reset, n_reflection_flips, n_sv_clamped).

    clamp=False (the v2 blessed path) repairs only the numerical pathologies — non-finite
    rows and reflections, both COUNTED — and leaves singular values untouched: a silent SV
    projection rewrites the state every commit without any counter, which gate G2 exists to
    forbid (adversarial finding). Legacy callers keep clamp=True."""
    F = np.ascontiguousarray(F, np.float32).reshape(-1, 3, 3)
    if not clamp and F.shape[0] >= 20000:
        # 2026-09-23 (speed): the v2 path on the device end to end — the non-finite test and the
        # reflection test (det F < 0 <=> det U det V^T < 0 for F = U S V^T, S >= 0) need no
        # SVD, and the SVD runs only for the rows that need a repair. The unrepaired F is
        # returned bit-exact as before; the counts are the same.
        try:
            import torch
            if torch.cuda.is_available():
                Ft = torch.as_tensor(F, device="cuda")
                bad_t = ~torch.isfinite(Ft).all(dim=(1, 2))
                Fw = Ft.clone() if bool(bad_t.any()) else Ft
                if bool(bad_t.any()):
                    Fw[bad_t] = torch.eye(3, device="cuda")
                flip_t = torch.linalg.det(Fw) < 0
                n_bad = int(bad_t.sum().item()); n_flip = int(flip_t.sum().item())
                if n_bad == 0 and n_flip == 0:
                    return F, 0, 0, 0
                out = Fw.clone()
                if n_flip:
                    U, S, Vh = torch.linalg.svd(Fw[flip_t])
                    U = U.clone(); U[:, :, -1] *= -1.0
                    out[flip_t] = U @ torch.diag_embed(S) @ Vh
                return out.cpu().numpy().astype(np.float32), n_bad, n_flip, 0
        except Exception:
            pass
    bad = ~np.isfinite(F).all(axis=(1, 2))
    if bad.any():
        F = F.copy()
        F[bad] = np.eye(3, dtype=np.float32)
    U, S, Vt = batched_svd(F)
    flip = np.linalg.det(U) * np.linalg.det(Vt) < 0        # improper pair -> reflection
    n_flip = int(flip.sum())
    if n_flip:
        U = U.copy()
        U[flip, :, -1] *= -1.0
    n_clamp = 0
    if clamp:
        n_clamp = int(((S < smin - 1e-4) | (S > smax + 1e-4)).any(1).sum())
        S = np.clip(S, smin, smax)
    if not (bad.any() or n_flip or n_clamp):
        return F, 0, 0, 0                                  # nothing to repair: bit-exact F
    out = np.einsum("nij,nj,njk->nik", U, S, Vt).astype(np.float32)
    return out, int(bad.sum()), n_flip, n_clamp
