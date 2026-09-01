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


def condition_F(F, smin=0.5, smax=2.0, clamp=True):
    """Return (F_repaired, n_nonfinite_reset, n_reflection_flips, n_sv_clamped).

    clamp=False (the v2 blessed path) repairs only the numerical pathologies — non-finite
    rows and reflections, both COUNTED — and leaves singular values untouched: a silent SV
    projection rewrites the state every commit without any counter, which gate G2 exists to
    forbid (adversarial finding). Legacy callers keep clamp=True."""
    F = np.ascontiguousarray(F, np.float32).reshape(-1, 3, 3)
    bad = ~np.isfinite(F).all(axis=(1, 2))
    if bad.any():
        F = F.copy()
        F[bad] = np.eye(3, dtype=np.float32)
    U, S, Vt = np.linalg.svd(F)
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
