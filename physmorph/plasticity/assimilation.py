"""Commit-time plastic assimilation of the ELASTIC stretch (docs/pipeline_v2.md §3.5).

The one surviving plasticity primitive. Two earlier variants were measured and removed
(git history): the OT/Jacobian `update_fp` fabricated strain from rigid rotation and was
volumetrically blind; the displacement-field polar variant mismatched the dFc-inflated F
(this engine injects dFc straight into F) and spiked stress at every commit boundary.
"""
from __future__ import annotations

import numpy as np


def assimilate_elastic(F, Fp, eta=0.5, smin=0.2, smax=5.0,
                       isochoric=False) -> np.ndarray:
    """Fp <- S_e^eta Fp with R_e S_e = polar(F_e), F_e = F Fp^-1. Per particle, EXACT.

    Because S_e is symmetric it commutes with its own powers, so
        F_e_new = F (S_e^eta Fp)^-1 = R_e S_e^{1-eta}
    exactly: an eta-fraction of the ELASTIC stretch is relaxed each commit, the rotation is
    untouched (a rigid motion is a strict no-op), and the fixed-corotated energy decreases
    monotonically (tested). Rows with det(F_e) <= 0 are skipped — the F guards own them.
    The cumulative singular-value band clamp is applied LAST so the returned Fp honours
    [smin, smax].

    isochoric=True: assimilate only the DEVIATORIC part — the assimilated increment is
    normalised to det=1 (S_e^eta / det^(1/3)), so ALL volumetric strain stays elastic and
    lambda keeps resisting it forever. Standard multiplicative plasticity (det Fp = 1, as
    in MPM sand/snow lineages); the unnormalised form is a measured volume RATCHET: each
    commit bakes compression permanently, |J-1|>0.3 grew 0->34% across hero6 with detF
    driven to ~0 by 120 commits, and the between-ears floaters are its squeeze-ejecta
    (J 0.52-0.84, forensics 2026-09-01)."""
    F = np.ascontiguousarray(F, np.float32).reshape(-1, 3, 3)
    Fp = np.ascontiguousarray(Fp, np.float32).reshape(-1, 3, 3)
    if eta <= 0:
        return Fp
    Fe = np.einsum("nij,njk->nik", F, np.linalg.inv(Fp))
    ok = np.linalg.det(Fe) > 1e-6
    _, S, Vt = np.linalg.svd(Fe)
    V = np.transpose(Vt, (0, 2, 1))
    Se = np.clip(S, 1e-3, None) ** eta
    if isochoric:                                    # det-free increment: J_p stays 1
        Se = Se / np.prod(Se, axis=1, keepdims=True) ** (1.0 / 3.0)
    Sa = np.einsum("nij,nj,nkj->nik", V, Se, V)      # V diag(S^eta) V^T = S_e^eta
    Sa[~ok] = np.eye(3, dtype=np.float32)
    Fp_new = np.einsum("nij,njk->nik", Sa, Fp)
    U2, S2, Vt2 = np.linalg.svd(Fp_new)              # cumulative band clamp LAST
    S2 = np.clip(S2, smin, smax)
    if isochoric:
        # Codex stack-review f16: a single det renormalisation VIOLATES the band
        # (probe: 792/1000 rows left [0.2,5], reaching [0.136,8.01]). Project onto
        # {sum log s = 0} INTERSECT the log-band by alternating projections (both
        # sets convex in log space; 4 rounds converge well under float32 noise).
        l = np.log(S2)
        lo, hi = np.log(smin), np.log(smax)
        for _ in range(4):
            l = l - l.mean(axis=1, keepdims=True)    # det = 1 plane
            l = np.clip(l, lo, hi)                   # band box
        l = l - l.mean(axis=1, keepdims=True)        # end on the invariant
        S2 = np.exp(l)
    return np.einsum("nij,nj,njk->nik", U2, S2, Vt2).astype(np.float32)
