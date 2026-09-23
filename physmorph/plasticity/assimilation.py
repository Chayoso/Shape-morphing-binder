"""Commit-time plastic assimilation of the ELASTIC stretch (docs/pipeline_v2.md §3.5).

The one surviving plasticity primitive. Two earlier variants were measured and removed
(git history): the OT/Jacobian `update_fp` fabricated strain from rigid rotation and was
volumetrically blind; the displacement-field polar variant mismatched the dFc-inflated F
(this engine injects dFc straight into F) and spiked stress at every commit boundary.
"""
from __future__ import annotations

import numpy as np
import torch


def assimilate_elastic(F, Fp, eta=0.5, smin=0.2, smax=5.0,
                       isochoric=False, Fe=None) -> np.ndarray:
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
    if Fe is None:                                   # per-particle elastic deformation
        Fe = np.einsum("nij,njk->nik", F, np.linalg.inv(Fp))
    else:                                            # a consensus F_e (consensus_elastic)
        Fe = np.ascontiguousarray(Fe, np.float32).reshape(-1, 3, 3)
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
    if _torch_cuda() and F.shape[0] >= 20000:
        return _assimilate_torch(F, Fp, eta, smin, smax, isochoric, grow, grow_band)
    from ..mpm.conditioning import batched_det
    ok = batched_det(Fe) > 1e-6
    from ..mpm.conditioning import batched_svd, batched_det
    _, S, Vt = batched_svd(Fe)
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
    U2, S2, Vt2 = batched_svd(Fp_new)                # cumulative band clamp LAST
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


def _bspline_1d(r):
    """Cubic B-spline (the MPM transfer kernel) at |r| (in cells), torch."""
    a = r.abs()
    w = torch.where(a < 1.0, 0.5 * a ** 3 - a ** 2 + 2.0 / 3.0,
                    torch.where(a < 2.0, (2.0 - a) ** 3 / 6.0, torch.zeros_like(a)))
    return w


def consensus_elastic(x, F, Fp, grid_min, dx, dims, m=None, device=None) -> np.ndarray:
    """Neighbourhood-consensus elastic deformation F̄_e per particle: the mass-weighted
    cubic-B-spline grid average of F_e = F Fp^-1 over the particle's 4^3 stencil with the
    particle's OWN contribution removed at every node; nodes with no other mass do not
    vote; a particle with no voting node gets its own F_e (nothing to compare with).
    Same transfer as P2G/G2P (physmorph.mpm.kernels), so 'neighbourhood' is the
    discretisation's own interaction range. Returns (N,3,3) float32."""
    F = np.ascontiguousarray(F, np.float32).reshape(-1, 3, 3)
    Fp = np.ascontiguousarray(Fp, np.float32).reshape(-1, 3, 3)
    N = F.shape[0]
    dev = device or ("cuda" if _torch_cuda() else "cpu")
    xt = torch.as_tensor(np.ascontiguousarray(x, np.float32), device=dev)
    Ft = torch.as_tensor(F, device=dev)
    Fpt = torch.as_tensor(Fp, device=dev)
    Fe = (Ft @ torch.linalg.inv(Fpt)).reshape(N, 9)
    mt = (torch.ones(N, device=dev) if m is None
          else torch.as_tensor(np.broadcast_to(np.asarray(m, np.float32), (N,)).copy(), device=dev))
    gmin = torch.as_tensor(np.asarray(grid_min, np.float32), device=dev)
    nx, ny, nz = (int(d) for d in dims)
    inv_dx = 1.0 / float(dx)
    rel = (xt - gmin) * inv_dx                        # position in cells
    base = torch.floor(rel).long() - 1                # base_node (kernels.base_node)
    ncell = nx * ny * nz
    # accumulate S_i = Σ_p w m Fe and M_i = Σ_p w m over the 64 stencil nodes
    S = torch.zeros(ncell, 9, device=dev)
    M = torch.zeros(ncell, device=dev)
    offs, wts, gids, valid = [], [], [], []
    for oi in range(4):
        for oj in range(4):
            for ok in range(4):
                node = base + torch.tensor([oi, oj, ok], device=dev)
                d = node.float() - rel                # (x_g - x_p) / dx
                w = _bspline_1d(d[:, 0]) * _bspline_1d(d[:, 1]) * _bspline_1d(d[:, 2])
                ok_ = ((node >= 0) & (node < torch.tensor([nx, ny, nz], device=dev))).all(1)
                g = (node[:, 0] * ny + node[:, 1]) * nz + node[:, 2]
                g = torch.where(ok_, g, torch.zeros_like(g))
                w = torch.where(ok_, w, torch.zeros_like(w))
                S.index_add_(0, g, (w * mt)[:, None] * Fe)
                M.index_add_(0, g, w * mt)
                wts.append(w); gids.append(g)
    # gather back, removing the particle's own contribution at each node
    num = torch.zeros(N, 9, device=dev)
    den = torch.zeros(N, device=dev)
    for w, g in zip(wts, gids):
        own = (w * mt)
        M_other = M[g] - own                          # other mass at the node
        S_other = S[g] - own[:, None] * Fe
        vote = (M_other > 1e-9 * (mt + 1e-30)) & (w > 0)
        contrib = torch.where(vote[:, None], S_other / M_other.clamp_min(1e-30)[:, None], torch.zeros_like(S_other))
        num = num + w[:, None] * contrib
        den = den + torch.where(vote, w, torch.zeros_like(w))
    has = den > 1e-12
    Fbar = torch.where(has[:, None], num / den.clamp_min(1e-30)[:, None], Fe)
    return Fbar.reshape(N, 3, 3).float().cpu().numpy()


def _torch_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _assimilate_torch(F, Fp, eta, smin, smax, isochoric, grow, grow_band) -> np.ndarray:
    """_assimilate in torch on the GPU (150k: the numpy inv/einsum/bisection chain was
    ~0.25 s per commit). Same operations in the same order, float32."""
    import torch
    dev = "cuda"
    Ft = torch.as_tensor(np.ascontiguousarray(F, np.float32), device=dev)
    Fpt = torch.as_tensor(np.ascontiguousarray(Fp, np.float32), device=dev)
    Fe = Ft @ torch.linalg.inv(Fpt)
    ok = torch.linalg.det(Fe) > 1e-6
    _, S, Vh = torch.linalg.svd(Fe)
    Se = S.clamp_min(1e-3) ** eta
    if isochoric:                                    # det-free increment: J_p stays 1
        Se = Se / Se.prod(1, keepdim=True) ** (1.0 / 3.0)
    Sa = Vh.transpose(1, 2) @ torch.diag_embed(Se) @ Vh      # V diag(S^eta) V^T
    Sa[~ok] = torch.eye(3, device=dev)
    if grow is not None:
        g = np.clip(np.asarray(grow, np.float32), 0.5, 2.0) ** (1.0 / 3.0)
        Sa = Sa * torch.as_tensor(np.ascontiguousarray(g, np.float32), device=dev)[:, None, None]
    Fp_new = Sa @ Fpt
    U2, S2, Vh2 = torch.linalg.svd(Fp_new)           # cumulative band clamp LAST
    S2 = S2.clamp(smin, smax)
    if grow is not None:
        det = S2.prod(1)
        S2 = _project_logsv_torch(S2.log(), float(np.log(smin)), float(np.log(smax)),
                                  det.clamp(1.0 / grow_band, grow_band).log())
    elif isochoric:
        S2 = _project_logsv_torch(S2.log(), float(np.log(smin)), float(np.log(smax)),
                                  torch.zeros_like(S2[:, 0]))
    return (U2 @ torch.diag_embed(S2) @ Vh2).float().cpu().numpy()


def _project_logsv_torch(l0, lo, hi, target):
    """_project_logsv in torch (same bisection)."""
    import torch
    target = target.clamp(3 * lo + 1e-6, 3 * hi - 1e-6)
    nu_lo = (l0.min(1).values - hi) - 1e-3
    nu_hi = (l0.max(1).values - lo) + 1e-3
    for _ in range(50):
        nu = 0.5 * (nu_lo + nu_hi)
        s = (l0 - nu[:, None]).clamp(lo, hi).sum(1)
        high = s > target
        nu_lo = torch.where(high, nu, nu_lo)
        nu_hi = torch.where(high, nu_hi, nu)
    return (l0 - (0.5 * (nu_lo + nu_hi))[:, None]).clamp(lo, hi).exp()


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


def relax_stretch(F, eta=0.5) -> np.ndarray:
    """EXACT R S^(1-eta) for F = R S (per-particle polar via SVD, proper rotation).

    Offline QA helper (photoreal comparisons), NOT used by the pipeline: a commit-time
    edit of the render deformation changes the image with no particle motion (REFUTE-2
    F11), so the production path saturates the stretch inside the render forward model
    instead (gauss_loss.saturate_stretch). REFUTE-2 F10: the first version routed through
    _assimilate, whose isochoric branch returned R S^(1-eta) J^(eta/3) — not R S^(1-eta).
    Rows with det(F) <= 1e-6 are returned unchanged."""
    F = np.ascontiguousarray(F, np.float32).reshape(-1, 3, 3)
    if eta <= 0:
        return F
    U, S, Vt = np.linalg.svd(F.astype(np.float64))
    flip = np.linalg.det(U) * np.linalg.det(Vt) < 0
    U[flip, :, -1] *= -1.0
    S = S.copy()
    S[flip, -1] *= -1.0
    S = np.abs(S)
    out = np.einsum("nij,nj,njk->nik", U, S ** (1.0 - eta), Vt).astype(np.float32)
    bad = ~np.isfinite(out).all(axis=(1, 2)) | (np.linalg.det(F) <= 1e-6)
    out[bad] = F[bad]
    return out
