"""Commit-time plastic assimilation of the REALISED motion (docs/pipeline_v2.md §3.5).

Why not update_fp: the v1 primitive symmetrises the displacement Jacobian and projects
isochorically — designed for OT *guidance* fields. Fed the realised commit motion it
(a) fabricates strain from rigid rotation (sym(R−I) ≠ 0: a 90° rotation invents plastic
stretch and hence elastic energy from a stress-free state — adversarial finding, verified
numerically), and (b) is an exact no-op for dilation (det Fp ≡ 1), so the one mode D_vol
drives springs back in full every commit.

This module is the objective version. Local incremental map A = I + ∇d (UNsymmetrised LSQ),
proper polar A = R·S (right stretch S = VΣVᵀ from the SVD): a rigid rotation gives S = I and
leaves Fp exactly unchanged; a dilation is assimilated in full. Partial assimilation uses the
stretch power S^η = VΣ^ηVᵀ. The cumulative singular-value band clamp is applied LAST (the v1
order — clamp then renormalise — leaves Fp outside the requested band). Rows whose local map
is inverted (det A ≤ 0) are skipped: pathological increments must surface in the F guards,
not be assimilated.
"""
from __future__ import annotations

import numpy as np

from .sinkhorn import displacement_jacobian


def assimilate_elastic(F, Fp, eta=0.5, smin=0.2, smax=5.0) -> np.ndarray:
    """Fp <- S_e^eta Fp with R_e S_e = polar(F_e), F_e = F Fp^-1. Per particle, EXACT.

    Because S_e is symmetric it commutes with its own powers, so
        F_e_new = F (S_e^eta Fp)^-1 = R_e S_e^{1-eta}
    exactly: an eta-fraction of the ELASTIC stretch is relaxed each commit, the rotation is
    untouched, and the fixed-corotated energy decreases monotonically. No displacement-field
    estimation is involved — the displacement-based variant (assimilate_fp below) proved
    unstable in this engine because dFc is injected straight into F (F <- (I+dtC)(F+dFc)),
    so F carries deformation the realised motion never had; migrating Fp toward the
    motion-stretch then MISMATCHES F and every commit boundary becomes a stress spike
    (measured locally: window 3 line-search stall, |v|max 60-290; exact version: 8/8
    commits, monotone D_vol). Rows with det(F_e) <= 0 are skipped (the F guards own them).
    """
    F = np.ascontiguousarray(F, np.float32).reshape(-1, 3, 3)
    Fp = np.ascontiguousarray(Fp, np.float32).reshape(-1, 3, 3)
    if eta <= 0:
        return Fp
    Fe = np.einsum("nij,njk->nik", F, np.linalg.inv(Fp))
    ok = np.linalg.det(Fe) > 1e-6
    _, S, Vt = np.linalg.svd(Fe)
    V = np.transpose(Vt, (0, 2, 1))
    Se = np.clip(S, 1e-3, None) ** eta
    Sa = np.einsum("nij,nj,nkj->nik", V, Se, V)      # V diag(S^eta) V^T = S_e^eta
    Sa[~ok] = np.eye(3, dtype=np.float32)
    Fp_new = np.einsum("nij,njk->nik", Sa, Fp)
    U2, S2, Vt2 = np.linalg.svd(Fp_new)              # cumulative band clamp LAST
    S2 = np.clip(S2, smin, smax)
    return np.einsum("nij,nj,njk->nik", U2, S2, Vt2).astype(np.float32)


def assimilate_fp(Fp, x, d, eta=0.5, k=12, diffusion_iters=0, smin=0.2, smax=5.0) -> np.ndarray:
    """Fp <- S_A^eta Fp with A = I + grad(d), S_A the right polar stretch of A.

    x (N,3) positions at the start of the commit; d (N,3) realised displacement.
    Returns float32 (N,3,3); Fp is not modified in place.

    diffusion_iters defaults to 0: the value-space pre-smoothing distorts LARGE displacement
    fields (a 90° rotation came back with |J - J_true| = 0.90 after two smoothing passes vs
    0.03 without — measured), which re-introduces exactly the fabricated strain this module
    exists to avoid. The kNN least-squares fit is already an averaging operator; objectivity
    needs the affine-exactness that smoothing destroys."""
    Fp = np.ascontiguousarray(Fp, np.float32)
    N = x.shape[0]
    k = int(min(k, N - 2))
    if eta <= 0 or k < 3:
        return Fp
    J = displacement_jacobian(x, d, k=k, diffusion_iters=diffusion_iters, symmetrize=False)
    A = np.eye(3, dtype=np.float32)[None] + J
    ok = np.linalg.det(A) > 1e-6                     # skip inverted local maps
    _, S, Vt = np.linalg.svd(A)                      # rotation part discarded (objective)
    V = np.transpose(Vt, (0, 2, 1))
    Se = np.clip(S, 1e-3, None) ** eta               # stretch part only, to the power eta
    Sa = np.einsum("nij,nj,nkj->nik", V, Se, V)      # V diag(S^eta) V^T (symmetric PD)
    Sa[~ok] = np.eye(3, dtype=np.float32)
    Fp_new = np.einsum("nij,njk->nik", Sa, Fp)
    # cumulative band clamp LAST so the returned Fp honours [smin, smax]
    U2, S2, Vt2 = np.linalg.svd(Fp_new)
    S2 = np.clip(S2, smin, smax)
    return np.einsum("nij,nj,njk->nik", U2, S2, Vt2).astype(np.float32)
