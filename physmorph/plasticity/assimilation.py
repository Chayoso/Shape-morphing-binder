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
    return _assimilate(F, Fp, Fe, eta, smin, smax, isochoric, None, 1.0)


def assimilate_growth(F, Fp, eta=0.5, smin=0.2, smax=5.0, isochoric=True,
                      grow=None, grow_band=1.5) -> np.ndarray:
    """assimilate_elastic + a COMMANDED per-particle volumetric growth g (morphoelastic
    F = F_e·G): Fp gains an isotropic factor grow^(1/3) per commit, so the rest volume
    expands exactly where the demand field says coverage is missing and elastic
    pressure fills the space. This is NOT the falsified ratchet: the ratchet absorbed
    whatever volume the control produced (uncontrolled, monotone); growth is
    DEMAND-DRIVEN (zero where covered — it stops by construction), CAPPED per commit
    by the caller, and GOVERNED cumulatively (det(Fp) clamped to [1/grow_band,
    grow_band] — the Stomakhin-snow lesson: plastic volume change is admissible only
    with a governor). The uncommanded remainder stays isochoric. It lives at commit
    time, outside the optimizer's gradient balance — the fill-v3 verdict showed any
    loss-side pull dies with the physics gradient before finishing thin features."""
    F = np.ascontiguousarray(F, np.float32).reshape(-1, 3, 3)
    Fp = np.ascontiguousarray(Fp, np.float32).reshape(-1, 3, 3)
    Fe = np.einsum("nij,njk->nik", F, np.linalg.inv(Fp))
    return _assimilate(F, Fp, Fe, eta, smin, smax, isochoric, grow, grow_band)


def _assimilate(F, Fp, Fe, eta, smin, smax, isochoric, grow, grow_band) -> np.ndarray:
    ok = np.linalg.det(Fe) > 1e-6
    _, S, Vt = np.linalg.svd(Fe)
    V = np.transpose(Vt, (0, 2, 1))
    Se = np.clip(S, 1e-3, None) ** eta
    if isochoric:                                    # det-free increment: J_p stays 1
        Se = Se / np.prod(Se, axis=1, keepdims=True) ** (1.0 / 3.0)
    Sa = np.einsum("nij,nj,nkj->nik", V, Se, V)      # V diag(S^eta) V^T = S_e^eta
    Sa[~ok] = np.eye(3, dtype=np.float32)
    if grow is not None:                             # commanded volumetric growth:
        g = np.clip(np.asarray(grow, np.float32), 0.5, 2.0) ** (1.0 / 3.0)
        Sa = Sa * g[:, None, None]                   # isotropic factor on the increment
    Fp_new = np.einsum("nij,njk->nik", Sa, Fp)
    U2, S2, Vt2 = np.linalg.svd(Fp_new)              # cumulative band clamp LAST
    S2 = np.clip(S2, smin, smax)
    if grow is not None:                             # growth governor: cumulative det
        det = np.prod(S2, axis=1)                    # band, NOT det=1 (growth is the
        lo_d, hi_d = 1.0 / grow_band, grow_band      # one channel allowed to command
        S2 = _project_logsv(np.log(S2), np.log(smin), np.log(smax),
                            np.log(np.clip(det, lo_d, hi_d)))
    elif isochoric:
        # EXACT projection onto {sum log s = target} INTERSECT the log-band.
        # History: a single det renorm violated the band (f16 probe: 792/1000 rows
        # out); 4 alternating projections then ENDED on the det plane, so the box
        # could still be exited (REFUTE 2026-09-02: [0.2,5,0.2] -> smax*1.002).
        # The KKT solution is clip(l - nu, lo, hi) with nu from scalar bisection.
        S2 = _project_logsv(np.log(S2), np.log(smin), np.log(smax),
                            np.zeros(len(S2), np.float32))
    return np.einsum("nij,nj,njk->nik", U2, S2, Vt2).astype(np.float32)


def _project_logsv(l0, lo, hi, target) -> np.ndarray:
    """Exact Euclidean projection of log-singular-values onto
    {sum(l) = target} INTERSECT {lo <= l_i <= hi} (KKT: l = clip(l0 - nu, lo, hi),
    nu found by bisection on the monotone sum). target is clipped to the feasible
    range [3 lo, 3 hi] so the intersection is never empty. Returns exp(l)."""
    target = np.clip(target, 3 * lo + 1e-6, 3 * hi - 1e-6)
    nu_lo = (l0.min(axis=1) - hi) - 1e-3             # sum == 3*hi >= target
    nu_hi = (l0.max(axis=1) - lo) + 1e-3             # sum == 3*lo <= target
    for _ in range(50):
        nu = 0.5 * (nu_lo + nu_hi)
        s = np.clip(l0 - nu[:, None], lo, hi).sum(axis=1)
        high = s > target                            # sum decreases as nu grows
        nu_lo = np.where(high, nu, nu_lo)
        nu_hi = np.where(high, nu_hi, nu)
    return np.exp(np.clip(l0 - (0.5 * (nu_lo + nu_hi))[:, None], lo, hi))
