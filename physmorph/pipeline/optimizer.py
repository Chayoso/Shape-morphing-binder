"""optimize_window — one window of multi-leaf trajectory optimisation (docs/pipeline_v2.md §3).

Evolves the C++ CompGraph parity port (v1 `trajectory_opt.py`, removed 2026-09-01, in git
history; the C++ oracle itself lives in legacy/DiffMPMLib3D) with:
  * leaves = [dFc sequence] + optional [s material field]  (render→material channel),
  * terminal kinetic loss mean|v_T|^2                      (arrive at rest),
  * asymmetric multi-elevation D_render                    (holes/spray see gradient),
  * a box leash relu(|x|−extent)^2                         (gradient exists BEYOND the render
    viewport and the D_vol grid — the far-field term the adversarial round showed D_render
    cannot provide once a particle leaves every view),
  * λ_R estimated ONCE per window from gradient norms (EMA across windows) — within a window
    the line search therefore decreases a single fixed objective; a per-iteration λ made
    "monotone acceptance" meaningless (adversarial finding),
  * acceptance requires a FINITE ROLLOUT STATE (x,F,v), not just finite scalars — the kernels'
    valid_pos guard silently drops NaN particles from the splats, so a NaN rollout can LOWER
    D_vol by deleting mass and would otherwise be accepted (adversarial finding),
  * history reuses the accepted line-search evaluation     (no wasted extra rollout).

Step control is the C++ recipe: persistent Adam moments, backtracking line search that
rejects any non-improving step (restoring leaves AND moments), adaptive initial step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import warp as wp

from ..losses.volumetric import (d_jdens, d_kde, d_nn_band, d_vol, d_w1, deficit_field,
                                 isolation_gate, kde_assign, nn_band_assign, w1_budget)
from ..mpm.constitutive import lame
from ..mpm.function import RolloutSpec, warp_mpm_full
from ..mpm.state import MPMParams
from ..mpm.traj import Trajectory
from .config import PipelineConfig
from .grid_smooth import smooth_particle_field
from .render_loss import LambdaBalancer, d_pbr, d_render


@dataclass
class TargetPack:
    """Precomputed target quantities shared by every window (built once in runner)."""
    grid: torch.Tensor          # D_vol target mass grid
    lgmin: torch.Tensor
    ldx: float
    ldims: tuple
    m: torch.Tensor             # unit masses (N,)
    views: list                 # [(theta, phi)]
    sils: list | None           # target alpha images (None when render channel off)
    extent: float
    shade: list | None = None   # target shaded images (PBR-lite channel, w_pbr>0)
    dt3: torch.Tensor | None = None   # fine target-fitted 3D outside-DT (W1 cleanup)
    dtgmin: torch.Tensor | None = None  # its own grid: NOT the loss grid (Opus finding 2:
    dtdx: float = 0.0                   # coarse cells left a dead radius covering the
    dtdims: tuple = ()                  # entire production fringe band)
    tmass3: torch.Tensor | None = None  # fine target mass raster (hole-side W1, w_fill>0)
    pts: torch.Tensor | None = None     # raw target particles (grid-free near-band, w_nn>0)
    nn_spacing: float = 0.0             # target median NN spacing (the honest metric's unit)
    gauss: object | None = None         # GaussViews bundle (use_gauss_loss)
    gauss_scale: float | None = None    # one calibration per target build, not per window
    jd_gmin: torch.Tensor | None = None  # density-J prior raster (w_jdens>0)
    jd_dx: float = 0.0
    jd_dims: tuple = ()
    jd_rho0: torch.Tensor | None = None  # per-particle rest density (source, same estimator)
    jd_scale: float | None = None        # one-shot equal-norm calibration vs D_vol
    kde_h: float = 0.0                  # particle-scale density term (w_kde>0)
    kde_rho_ref: float = 1.0
    kde_scale: float | None = None      # one-shot equal-norm calibration vs D_vol


def _norm(gs) -> float:
    """Joint L2 norm over a list of per-leaf gradients."""
    return float(torch.sqrt(sum(g.pow(2).sum() for g in gs)).item())


def _finite(tensors) -> bool:
    return all(bool(torch.isfinite(t).all()) for t in tensors)


def _pcgrad(gp, gr):
    """One-sided PCGrad: strip from gr its component conflicting with gp (joint over
    leaves). Returns (gr', conflicted). Descent for the PHYSICS term is preserved; the
    composite is additionally guarded by the line search — the projection alone does NOT
    guarantee composite descent for arbitrary λ (adversarial finding: only for
    |cos| < 0.894 or a λ band)."""
    dot = sum((a * b).sum() for a, b in zip(gp, gr))
    if float(dot) >= 0:
        return gr, False
    gp2 = sum(a.pow(2).sum() for a in gp).clamp_min(1e-30)
    return [b - (dot / gp2) * a for a, b in zip(gp, gr)], True


def _control_h1(g: torch.Tensor, knn: torch.Tensor, iters: int,
                kappa: float) -> torch.Tensor:
    """Screened neighbour solve on a (T,N,3,3) control gradient.

    This is a search-direction preconditioner only.  It propagates a surface render
    covector into coherent dFc controls; candidate x/F/v still come exclusively from
    the original MPM forward rollout and line search.
    """
    if iters <= 0:
        return g
    rhs = g
    u = g
    for _ in range(iters):
        u = (rhs + float(kappa) * u[:, knn].mean(2)) / (1.0 + float(kappa))
    return u * (rhs.norm() / u.norm().clamp_min(1e-30))


def _linearized_work(grads, deltas) -> tuple[float, list[float]]:
    """Return total and per-state ``-grad dot delta`` endpoint work for telemetry.
    NOTE (transfer-function probe 2026-09-02): raw work is dominated by LOSS SCALE
    (||g_phys||/||g_render|| ~ 6500 in x-space), so a work SHARE says nothing about
    steering; use _steer_cos for that."""
    parts = [(-float((g.detach() * d.detach()).sum()) if g is not None else 0.0)
             for g, d in zip(grads, deltas)]
    return sum(parts), parts


def _steer_cos(grads, deltas) -> float | None:
    """Scale-free steering metric: cosine between the accepted state step and the
    channel's descent direction (-grad), joint over the given state slots."""
    num = den_g = den_d = 0.0
    for g, d in zip(grads, deltas):
        if g is None:
            continue
        g, d = g.detach(), d.detach()
        num += -float((g * d).sum())
        den_g += float((g * g).sum())
        den_d += float((d * d).sum())
    if den_g <= 0 or den_d <= 0:
        return None
    return num / (den_g ** 0.5 * den_d ** 0.5)


def _state_ok(state) -> bool:
    """Finite AND orientation-preserving. det(F_T) <= 0 is invisible to every loss term
    (the data terms see positions only), so without this check the line search happily
    accepts an inverting control — measured: warm-started windows committed 9 inversions
    with 8/8 'accepted' steps. state[3] (when present) is the WHOLE-TRAJECTORY min det
    from the candidate rollout: terminal-only checking let single-step inversions slip
    into committed trajectories (hero7/hero9: F_invert_steps=1 under the W1 pull —
    a pulled particle compresses through zero for one step and recovers by T)."""
    if not _finite(state[:3]):
        return False
    FT = state[1]
    if not bool((torch.linalg.det(FT.view(-1, 3, 3)) > 0).all()):
        return False
    if len(state) > 3 and state[3] is not None:
        return state[3] > 1e-4      # margin above the float32 det noise floor
    return True                     # dets like 1e-12 with exploding F^-T


def optimize_window(x0, prm: MPMParams, cfg: PipelineConfig, tgt: TargetPack,
                    balancer: LambdaBalancer, F0=None, Fp=None, v0=None, C0=None,
                    s_init=None, dfc_init=None, on_iter=None, log=print,
                    fill_bal: LambdaBalancer | None = None, alpha_scale: float = 1.0,
                    mom_init=None, vol0=None, surface_w=None):
    """Optimise dFc[0..T-1] (+ material s) over one horizon. Returns
    (frames, F_seq, end_state, s_out, hist, stats)."""
    dev = cfg.device
    x0 = np.ascontiguousarray(x0, np.float32)
    N, T = x0.shape[0], cfg.T
    lam0, mu0 = lame(cfg.young, cfg.poisson)
    spec = RolloutSpec(x0=x0, m=1.0, lam=lam0, mu=mu0, prm=prm, T=T,
                       F0=F0, Fp=Fp, v0=v0, C0=C0, device=dev, vol0=vol0)

    dFc = torch.zeros(T, N, 3, 3, device=dev, requires_grad=True)
    leaves = [dFc]
    s = None
    if cfg.opt_material:
        s0 = np.zeros((2, N), np.float32) if s_init is None else np.asarray(s_init, np.float32)
        s = torch.tensor(s0, device=dev, requires_grad=True)
        leaves.append(s)
    mom = [torch.zeros_like(p) for p in leaves]
    vel = [torch.zeros_like(p) for p in leaves]
    lr_scale = [1.0] + ([cfg.mat_lr_scale] if s is not None else [])
    adam_t = 0
    if cfg.mom_carry > 0 and mom_init is not None:
        m_in, v_in, t_in = mom_init
        if len(m_in) == len(mom) and all(a.shape == b.shape for a, b in zip(m_in, mom)):
            for m_, mi in zip(mom, m_in):
                m_.copy_(mi * cfg.mom_carry)   # first moment: directional, decayed
            for v_, vi in zip(vel, v_in):
                v_.copy_(vi)                   # second moment: curvature scale, verbatim
            adam_t = int(t_in)

    # control-field spatial regularisation: frozen kNN topology at window start; the
    # penalty lives purely in control space (no rollout needed for its gradient)
    knn_t = None
    if cfg.w_creg > 0 or cfg.control_h1_iters > 0:
        from scipy.spatial import cKDTree
        knn = cKDTree(x0).query(x0, k=cfg.creg_k + 1)[1][:, 1:]
        knn_t = torch.as_tensor(np.ascontiguousarray(knn), device=dev)
    surface_w_t = (torch.as_tensor(surface_w, device=dev).view(N, 1)
                   if surface_w is not None and cfg.surface_mask_objective else None)
    gauss_mask_t = (torch.as_tensor(np.asarray(surface_w) > 0.5, device=dev)
                    if surface_w is not None and cfg.render_surface_only else None)

    # W1 transport budget, frozen at window start: one scalar caps the total pull mass
    # (per-particle gates were falsified twice — grid density AND kNN isolation; the
    # budget form has no per-particle classification to get wrong, rationale §7.5)
    m_dt = None
    x0_t = torch.as_tensor(x0, device=dev)
    if tgt.dt3 is not None:
        if cfg.dt_gate not in ("knn", "budget"):    # f18: a typo must not silently
            raise ValueError(f"unknown dt_gate {cfg.dt_gate!r}")   # select the default
        if cfg.dt_gate == "budget":       # kept for A/B; measured no-op at 0.01 (§7.6)
            m_dt = tgt.m * w1_budget(x0_t, tgt.dt3, tgt.dtgmin, tgt.dtdx, tgt.dtdims,
                                     cfg.dt_budget)
        else:                             # "knn" — the honest-metric winner (§7.6)
            m_dt = tgt.m * isolation_gate(x0_t, cfg.dt_iso_lo, cfg.dt_iso_hi)

    # grid-free near-band cleanup, frozen per window (fork-halo forensic §7.10)
    if cfg.w_jdens > 0 and tgt.jd_rho0 is not None and tgt.jd_scale is None:
        xg = x0_t.detach().clone().requires_grad_(True)     # one-shot equal-norm vs D_vol
        gv = torch.autograd.grad(d_vol(xg, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx,
                                       tgt.ldims), xg)[0].norm()
        gj = torch.autograd.grad(d_jdens(xg, tgt.m, tgt.jd_rho0, tgt.jd_gmin, tgt.jd_dx,
                                         tgt.jd_dims), xg)[0].norm()
        tgt.jd_scale = float(gv / gj.clamp_min(1e-30))
        log(f"[win] jdens calibration: |g_vol|={float(gv):.3g} |g_jd|={float(gj):.3g} "
            f"scale={tgt.jd_scale:.3g}")
    kde_nbr = None
    if cfg.w_kde > 0 and tgt.pts is not None:
        kde_nbr = kde_assign(x0_t, tgt.pts, cfg.kde_k)
        if tgt.kde_scale is None:            # one-shot equal-norm calibration vs D_vol
            xg = x0_t.detach().clone().requires_grad_(True)
            gv = torch.autograd.grad(d_vol(xg, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx,
                                           tgt.ldims), xg)[0].norm()
            gk = torch.autograd.grad(d_kde(xg, tgt.pts, kde_nbr,
                                           tgt.kde_h, tgt.kde_rho_ref), xg)[0].norm()
            tgt.kde_scale = float(gv / gk.clamp_min(1e-30))
            log(f"[win] kde calibration: |g_vol|={float(gv):.3g} |g_kde|={float(gk):.3g} "
                f"scale={tgt.kde_scale:.3g}")
    nn_idx = nn_elig = None
    if cfg.w_nn > 0 and tgt.pts is not None:
        nn_idx, nn_elig = nn_band_assign(x0_t, tgt.pts, tgt.nn_spacing,
                                         cfg.nn_berth_k, cfg.nn_far_k,
                                         cfg.nn_tail_frac)

    # HOLE-side W1 (deficit fill v2), frozen per window: support-ANDed mask (F1),
    # budget on DEFICIT MASS not in-range particles (F2), SMOOTH locality ramp (F12 —
    # the hard 0/1 indicator was single-particle actuation, what w_creg exists to stop)
    fill_pairs = None
    if cfg.w_fill > 0 and tgt.tmass3 is not None:
        from ..losses.volumetric import deficit_assign
        fill_pairs = deficit_assign(x0_t, tgt.m, tgt.tmass3, tgt.dtgmin, tgt.dtdx,
                                    tgt.dtdims, cfg.fill_thresh, cfg.fill_sigma,
                                    cap_frac=cfg.fill_cap_frac,
                                    range_wu=cfg.fill_range_frac * tgt.extent)

    def material():
        if s is None:
            return None, None
        return lam0 * torch.exp(s[0]), mu0 * torch.exp(s[1])

    sil_gauss = {"sil": None, "gauss": None}
    # REFUTE B1 (Opus) / finding 9 (Codex): when the gauss term rides in the render
    # scalar, every gate that reads d_render — freeze track, outer merit, latch —
    # can be moved by an observation-side change (dressing) with zero state change.
    # The channels are therefore logged separately and every gate consumes the pure
    # silhouette d_sil only.

    def losses_of(xT, FT, vT):
        lv = d_vol(xT, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims)
        lk = vT.pow(2).sum(1).mean()
        lr = lpbr = None
        sil_gauss["sil"] = sil_gauss["gauss"] = None
        if balancer.active and tgt.gauss is not None and not cfg.gauss_in_objective:
            # gauss for dressing/telemetry only: log d_gauss no-grad, then fall
            # through to the plain silhouette channel below
            with torch.no_grad():
                sil_gauss["gauss"] = float(tgt.gauss.loss(
                    xT.detach(), FT.detach() if cfg.gauss_covariance else None,
                    mask=gauss_mask_t))
        if balancer.active and tgt.gauss is not None and cfg.gauss_in_objective:
            gauss_mask = gauss_mask_t
            lg_ = tgt.gauss.loss(xT, FT if cfg.gauss_covariance else None,
                                 mask=gauss_mask)
            sil_gauss["gauss"] = float(lg_.detach())
            if cfg.gauss_mix > 0:                 # hybrid: silhouette keeps fine geometry
                lsil = d_render(xT, tgt.sils, tgt.views, cfg.render_res, tgt.extent,
                                cfg.sil_k, cfg.w_hole, cfg.w_spray)
                sil_gauss["sil"] = float(lsil.detach())
                if tgt.gauss_scale is None:
                    tgt.gauss_scale = float(lsil.detach() / lg_.detach().clamp_min(1e-12))
                lr = lsil + cfg.gauss_mix * tgt.gauss_scale * lg_
            else:
                lr = lg_                          # pure replacement (falsified for geometry, g1)
                if tgt.sils is not None:          # gate telemetry still needs d_sil
                    with torch.no_grad():
                        sil_gauss["sil"] = float(d_render(
                            xT.detach(), tgt.sils, tgt.views, cfg.render_res,
                            tgt.extent, cfg.sil_k, cfg.w_hole, cfg.w_spray))
        elif balancer.active:
            lsil = d_render(xT, tgt.sils, tgt.views, cfg.render_res, tgt.extent,
                            cfg.sil_k, cfg.w_hole, cfg.w_spray)
            sil_gauss["sil"] = float(lsil.detach())
            lr = lsil
            if cfg.w_pbr > 0 and tgt.shade is not None:     # shading channel (PBR-lite)
                lpbr = d_pbr(xT, tgt.shade, tgt.views, cfg.render_res, tgt.extent,
                             tgt.lgmin, tgt.ldx, tgt.ldims, cfg.sil_k, cfg.pbr_ambient)
                lr = lsil + cfg.w_pbr * lpbr
        return lv, lk, lr, lpbr

    def terms(dfc):
        """Differentiable path: torch Function + wp.Tape (gradient phase only)."""
        lam_t, mu_t = material()
        xT, FT, vT = warp_mpm_full(dfc, spec, lam_t, mu_t)
        lv, lk, lr, lpbr = losses_of(xT, FT, vT)
        return (xT, FT, vT), lv, lk, lr, lpbr

    def eval_terms(dfc):
        """No-grad path for line-search candidates: plain rollout, NO tape, NO adjoint
        buffers — the graph path allocates ~2x memory and tape bookkeeping that a
        candidate evaluation (up to max_ls_iters per iteration) never uses."""
        with torch.no_grad():
            lam_t, mu_t = material()
            lam_e = lam_t.detach().cpu().numpy() if lam_t is not None else lam0
            mu_e = mu_t.detach().cpu().numpy() if mu_t is not None else mu0
            dc = dfc.detach().contiguous()
            seq = [wp.from_torch(dc[t].view(N, 3, 3), dtype=wp.mat33) for t in range(T)]
            tr = Trajectory(x0, 1.0, lam_e, mu_e, prm, T, F0=F0, Fp=Fp, v0=v0, C0=C0,
                            dFc=seq, device=dev, requires_grad=False, vol0=vol0)
            tr.rollout()
            xT = wp.to_torch(tr.x[T])
            FT = wp.to_torch(tr.F[T]).reshape(N, 9)
            vT = wp.to_torch(tr.v[T])
            lv, lk, lr, lpbr = losses_of(xT, FT, vT)
            # whole-trajectory orientation check for _state_ok. Stack-review fixes:
            # f1 — the stored F is SMOOTHED, so a constitutive inversion in the
            # EFFECTIVE deformation (F+dFc, whose det sign equals det(F_e) since
            # det(Fp)>0) could hide; both are checked. f17 — one stacked reduction,
            # one device sync. NaN propagates through .min() -> rejected by _state_ok.
            F_post = torch.stack([wp.to_torch(tr.F[t]).reshape(N, 3, 3)
                                  for t in range(1, T + 1)])
            F_pre = torch.stack([wp.to_torch(tr.F[t]).reshape(N, 3, 3)
                                 for t in range(T)])
            j_eff = torch.linalg.det(F_pre + dc.view(T, N, 3, 3)).min()
            jt = float(torch.minimum(torch.linalg.det(F_post).min(), j_eff))
        return (xT, FT, vT, jt), lv, lk, lr, lpbr

    def phys_core(lv, lk, dfc, xT, fT=None):
        """Physics objective WITHOUT the W1 term — the lambda balancer's numerator and
        PCGrad's reference direction (Codex finding 9: folding the W1 term into gp let
        w_dt inflate the balanced silhouette weight and project render components off a
        direction that was never 'the physics')."""
        L = lv + cfg.w_kin * lk + cfg.w_ctrl * dfc.pow(2).sum() / (T * N)
        if cfg.w_tctrl > 0 and T > 1:
            L = L + cfg.w_tctrl * (dfc[1:] - dfc[:-1]).pow(2).mean()
        if cfg.w_box > 0:      # far-field leash: differentiable everywhere, zero inside box
            L = L + cfg.w_box * torch.clamp(xT.abs() - tgt.extent, min=0).pow(2).sum(1).mean()
        if knn_t is not None:  # control smoothness: a lone particle cannot be actuated
            L = L + cfg.w_creg * (dfc - dfc[:, knn_t].mean(2)).pow(2).mean()
        if cfg.w_jdens > 0 and tgt.jd_rho0 is not None:
            # density-measured volume prior (REVISION 3): J from the particle mass
            # field, not from the lagging stored F
            L = L + cfg.w_jdens * tgt.jd_scale * d_jdens(xT, tgt.m, tgt.jd_rho0,
                                                         tgt.jd_gmin, tgt.jd_dx, tgt.jd_dims)
        if cfg.w_jvol > 0 and fT is not None:
            # sKL volume prior (J-1)·log J (F5 verdict): counters the permanent
            # volumetric spring + control-injected drift; soft barrier as J->0+
            J = torch.linalg.det(fT.view(-1, 3, 3))
            L = L + cfg.w_jvol * ((J - 1.0) * torch.log(J.clamp_min(1e-6))).mean()
        if cfg.w_cov > 0 and fT is not None:
            # The viewer renders Sigma=sigma0^2 F F^T.  Penalise only excessive
            # anisotropy/scale; rotations remain free and normal elastic motion inside
            # the band is untouched.
            sv = torch.linalg.svdvals(fT.view(-1, 3, 3)).clamp_min(1e-6)
            lo = torch.relu(torch.log(cfg.cov_smin / sv))
            hi = torch.relu(torch.log(sv / cfg.cov_smax))
            L = L + cfg.w_cov * (lo.square() + hi.square()).mean()
        if s is not None:
            L = L + cfg.w_mat * s.pow(2).mean()
        return L

    def dt_term(xT):
        """Fixed-weight one-signed cleanup terms (spray DT-W1 + near-band) — NOT
        lambda-scaled (lambda->cap x constant gradient = documented mass-ejection
        mode) and NOT part of phys_core (finding 9). The fill term is separate:
        it carries its own norm-balanced weight (fill v3)."""
        L = None
        if tgt.dt3 is not None:
            L = cfg.w_dt * d_w1(xT, m_dt, tgt.dt3, tgt.dtgmin, tgt.dtdx, tgt.dtdims)
        if nn_idx is not None:
            Ln = cfg.w_nn * d_nn_band(xT, tgt.m, tgt.pts, nn_idx, nn_elig,
                                      cfg.nn_berth_k * tgt.nn_spacing)
            L = Ln if L is None else L + Ln
        if kde_nbr is not None:
            Lk = cfg.w_kde * tgt.kde_scale * d_kde(xT, tgt.pts, kde_nbr,
                                                   tgt.kde_h, tgt.kde_rho_ref)
            L = Lk if L is None else L + Lk
        return L

    fill_on = fill_pairs is not None
    fill_lam = cfg.w_fill if fill_on else 0.0     # v4: FIXED weight — dominance is
    # bounded by the capacity-limited MATCHING (<= fill_cap_frac*N pairs), not by a
    # scalar that dies with the physics gradient (the v3 failure)

    def fill_raw(xT):
        from ..losses.volumetric import d_fill_pairs
        return d_fill_pairs(xT, fill_pairs[0], fill_pairs[1], 0.5 * tgt.dtdx)

    def phys_total(lv, lk, dfc, xT, fT=None):
        L = phys_core(lv, lk, dfc, xT, fT)
        ldt = dt_term(xT)
        if ldt is not None:
            L = L + ldt
        if fill_on:
            L = L + fill_lam * fill_raw(xT)
        return L

    def scalars(lv, lk, lr, lam_r, dfc, xT, fT=None):
        with torch.no_grad():    # scalar only — never build a second autograd graph
            L = float(phys_total(lv, lk, dfc.detach(), xT.detach(),
                                 fT.detach() if fT is not None else None))
        return L if lr is None else L + lam_r * float(lr.detach())

    hist, accepted, rejected = [], 0, 0
    pace_bound = False               # window exited via the pace floor (on schedule)
    lk_start = None                  # kinetic term at window start (quasi-static rule)
    g_cos = g_raw_cos = g_share = g_phys_norm = g_rend_norm = None
    render_work = render_work_x = render_work_F = None
    phys_work = phys_work_x = phys_work_F = phys_work_v = None
    step_norm = predicted_decrease = None
    render_cos = phys_cos = None
    alpha, g0_norm, L_start = cfg.alpha * alpha_scale, None, None
    lam_r = (balancer.lam or 0.0) if balancer.active else 0.0
    grad_converged = False

    # v3 SAFEGUARDED warm start: decayed previous solution, kept only if it (a) yields a
    # finite, orientation-preserving state and (b) actually beats the zero start. dFc is an
    # absolute control — verbatim reuse double-applies it (measured cascade to inversion).
    if dfc_init is not None and cfg.warm_decay > 0:
        st0, lv0, lk0, lr0, _ = eval_terms(dFc)
        # stack-review f5: an INVALID cold baseline must not be the comparator — its
        # position-only loss can be artificially low and block every valid warm start
        E0 = (scalars(lv0, lk0, lr0, lam_r, dFc, st0[0], st0[1])
              if _state_ok(st0) else np.inf)
        with torch.no_grad():
            dFc.copy_(torch.tensor(np.ascontiguousarray(dfc_init, np.float32),
                                   device=dev) * cfg.warm_decay)
        stw, lvw, lkw, lrw, _ = eval_terms(dFc)
        Ew = scalars(lvw, lkw, lrw, lam_r, dFc, stw[0], stw[1])
        if not (_state_ok(stw) and np.isfinite(Ew) and Ew < E0):
            with torch.no_grad():
                dFc.zero_()                          # stale controls: fall back to cold start
            for m_ in mom:                           # stale moments go with them
                m_.zero_()
            for v_ in vel:
                v_.zero_()
            adam_t = 0

    # Replay-noise calibration (sobolev_precond probe, side finding: the fixed
    # ls_noise_rel=1e-7 replay tolerance discarded ~10% of fully-accepted windows -
    # CUDA atomics make two rollouts of the SAME control differ by more than that).
    # Measure the discrepancy on this window's start control and use 10x it as the
    # floor of the commit-rollout tolerance: a rule from a measurement, not a knob.
    replay_rel = 0.0
    if cfg.replay_calibrate:
        stA, lvA, lkA, lrA, _ = eval_terms(dFc)
        stB, lvB, lkB, lrB, _ = eval_terms(dFc)
        EA = scalars(lvA, lkA, lrA, lam_r, dFc, stA[0], stA[1])
        EB = scalars(lvB, lkB, lrB, lam_r, dFc, stB[0], stB[1])
        if np.isfinite(EA) and np.isfinite(EB):
            replay_rel = abs(EA - EB) / max(abs(EA), 1.0)

    for it in range(cfg.iters):
        # ---- gradients. λ_R is fixed for the WHOLE window (estimated from the first
        # iteration's per-term norms), so every accepted step decreases one objective. ----
        state, lv, lk, lr, lpbr = terms(dFc)
        if lk_start is None:
            lk_start = float(lk.detach())
        Lp_core = phys_core(lv, lk, dFc, state[0], state[1])
        Lfill = fill_raw(state[0]) if fill_on else None
        smooth = balancer.active and cfg.render_gs_iters > 0
        special_render = smooth or surface_w_t is not None or cfg.control_h1_iters > 0
        gp = None
        Ldt = dt_term(state[0])
        if Lfill is not None:
            Ldt = (fill_lam * Lfill) if Ldt is None else (Ldt + fill_lam * Lfill)
        gx_phys_diag = gF_phys_diag = gv_phys_diag = None
        gx_rend_diag = gF_rend_diag = None
        if on_iter is not None or cfg.work_telemetry:
            # Endpoint position-space gradients are the interpretable vector fields
            # shown by the viewer.  They are diagnostics only; the control update still
            # uses the full MPM adjoint through x, F and v.
            Lp_diag = Lp_core if Ldt is None else Lp_core + Ldt
            gx_phys_diag, gF_phys_diag, gv_phys_diag = torch.autograd.grad(
                Lp_diag, state, retain_graph=True, allow_unused=True)
            if lr is not None:
                gx_rend_diag, gF_rend_diag = torch.autograd.grad(
                    lr, (state[0], state[1]), retain_graph=True, allow_unused=True)
                if surface_w_t is not None:
                    gx_rend_diag = gx_rend_diag * surface_w_t
                    if gF_rend_diag is not None:
                        gF_rend_diag = gF_rend_diag * surface_w_t.repeat(1, 9)
        if balancer.active and (special_render or cfg.grad_project or it == 0):
            if gp is None:
                gp = torch.autograd.grad(Lp_core, leaves, retain_graph=True)
            gdt = (torch.autograd.grad(Ldt, leaves, retain_graph=True)
                   if Ldt is not None else None)
            if smooth:
                # v3 grid-GS preconditioning: smooth the IMAGE-SPACE pull on the grid,
                # then pull the smoothed direction back through the SAME MPM adjoint
                # (seeded backward) — physics-exact, docs/method.md §6.
                gx = torch.autograd.grad(lr, state[0], retain_graph=True)[0]
                if surface_w_t is not None:
                    gx = gx * surface_w_t
                gxs = smooth_particle_field(state[0].detach(), gx, tgt.lgmin, tgt.ldx,
                                             tgt.ldims, cfg.render_gs_iters,
                                             cfg.render_gs_kappa)
                gr = torch.autograd.grad(state[0], leaves, grad_outputs=gxs)
            elif surface_w_t is not None:
                gx, gF = torch.autograd.grad(lr, (state[0], state[1]), retain_graph=True,
                                             allow_unused=True)
                if gF is None:
                    gF = torch.zeros_like(state[1])
                gr = torch.autograd.grad((state[0], state[1]), leaves,
                                         grad_outputs=(gx * surface_w_t,
                                                       gF * surface_w_t.repeat(1, 9)))
            else:
                gr = torch.autograd.grad(lr, leaves)
            if cfg.control_h1_iters > 0:
                gr = list(gr)
                gr[0] = _control_h1(gr[0], knn_t, cfg.control_h1_iters,
                                    cfg.control_h1_kappa)
            gr_raw = [r.detach().clone() for r in gr]
            np_raw, nr_raw = _norm(gp), _norm(gr_raw)
            dot_raw = float(sum((a * b).sum() for a, b in zip(gp, gr_raw)))
            if cfg.grad_project:
                gr, _conf = _pcgrad(gp, gr)
            if it == 0:
                # λ from the PROJECTED render grad — estimating it from the raw one and
                # then projecting silently de-weighted the channel by ~33% at cos=-0.74
                # (adversarial finding), breaking the balancer contract.
                lam_r = balancer.update(_norm(gp), _norm(gr))
                # render-influence telemetry (standing request): how much does the
                # render channel actually steer the update this window?
                np_, nr_ = _norm(gp), _norm(gr)
                dot_ = float(sum((a * b).sum() for a, b in zip(gp, gr)))
                g_cos = dot_ / max(np_ * nr_, 1e-30)
                g_raw_cos = dot_raw / max(np_raw * nr_raw, 1e-30)
                g_share = lam_r * nr_ / max(np_ + lam_r * nr_, 1e-30)
                g_phys_norm, g_rend_norm = np_, nr_
            g = [a + lam_r * b for a, b in zip(gp, gr)]
            if gdt is not None:      # W1 joins the composite AFTER lambda/PCGrad (find. 9)
                g = [gi + di for gi, di in zip(g, gdt)]
        elif balancer.active:
            total = Lp_core + lam_r * lr if Ldt is None else Lp_core + Ldt + lam_r * lr
            g = list(torch.autograd.grad(total, leaves))
        else:
            total = Lp_core if Ldt is None else Lp_core + Ldt
            g = list(torch.autograd.grad(total, leaves))
        cur = scalars(lv, lk, lr, lam_r, dFc, state[0], state[1])
        if not np.isfinite(cur):
            log(f"[win] iter {it}: non-finite loss, aborting window")
            break
        if L_start is None:
            L_start = cur
        gn = _norm(g)
        if g0_norm is None:
            g0_norm = max(gn, 1e-12)
        if gn < cfg.gd_tol * g0_norm:
            grad_converged = True
            log(f"[win] converged at iter {it} (||g||={gn:.4g})")
            break

        # ---- adaptive alpha (C++) ----
        a_try = alpha
        if cfg.adaptive_alpha:
            a_try *= max(cfg.min_alpha_scale, min(1.0, cfg.target_norm / max(gn, 1e-6)))

        # ---- backtracking line search over the Adam step ----
        bak = [p.detach().clone() for p in leaves]
        bak_m = [m_.clone() for m_ in mom]
        bak_v = [v_.clone() for v_ in vel]
        step_ok = False
        new = cur
        lv_n = lk_n = lr_n = lpbr_n = None
        # trajectory pacing floor: a candidate may not take the window's loss below
        # (1-pace)·L_start — an OVERSHOOTING step is rejected and halved like any other
        # bad step, making pace a true UPPER bound on per-window progress (the previous
        # break-after-accept form only enforced "at least pace, then stop" — adversarial
        # finding: one big accepted step could still snap the morph).
        floor = (1.0 - cfg.pace) * L_start if cfg.pace > 0 else -np.inf
        for _ls in range(cfg.max_ls_iters):
            with torch.no_grad():
                t_ = adam_t + 1
                for p, gi, m_, v_, sc in zip(leaves, g, mom, vel, lr_scale):
                    m_.mul_(cfg.beta1).add_(gi, alpha=1 - cfg.beta1)
                    v_.mul_(cfg.beta2).addcmul_(gi, gi, value=1 - cfg.beta2)
                    mh = m_ / (1 - cfg.beta1 ** t_)
                    vh = v_ / (1 - cfg.beta2 ** t_)
                    p -= (a_try * sc) * mh / (vh.sqrt() + cfg.eps)
                if cfg.dfc_clip > 0:
                    n = dFc.flatten(2).norm(dim=2, keepdim=True).unsqueeze(-1)
                    dFc *= (cfg.dfc_clip / n.clamp_min(1e-8)).clamp(max=1.0)
                if s is not None:
                    s.clamp_(-cfg.mat_clamp, cfg.mat_clamp)
            state_n, lv_n, lk_n, lr_n, lpbr_n = eval_terms(dFc)
            with torch.no_grad():
                new = scalars(lv_n, lk_n, lr_n, lam_r, dFc, state_n[0], state_n[1])
                predicted_decrease = -float(sum((gi.detach() * (p - b)).sum()
                                                for gi, p, b in zip(g, leaves, bak)))
                # The Armijo slope is only meaningful when the model predicts descent.
                # A carried/stale Adam moment can point against the fresh gradient
                # (predicted_decrease <= 0) while the step still lowers the true
                # objective; auto-rejecting there exhausts the line search on the
                # first windows after every plateau commit (observed: null commits
                # at anim 12/18 in final_hires20k_child4). Fall back to the
                # noise-floor sufficient decrease instead.
                noise_floor = cfg.ls_noise_rel * max(abs(cur), 1.0)
                required = (max(cfg.armijo_c1 * predicted_decrease, noise_floor)
                            if predicted_decrease > 0.0 else noise_floor)
            # acceptance requires a FINITE, ORIENTATION-PRESERVING state: NaN particles
            # vanish from the splats and det(F)<=0 is invisible to the data terms — both
            # can fake a lower loss (adversarial finding + v3 warm-start cascade).
            if (np.isfinite(new) and floor <= new <= cur - required
                    and _state_ok(state_n)):
                adam_t = t_
                alpha = min(a_try * 1.1, cfg.alpha * alpha_scale)  # C++ grows alpha on acceptance
                step_ok = True
                accepted += 1
                break
            with torch.no_grad():                            # reject: restore and halve
                for p, b in zip(leaves, bak):
                    p.copy_(b)
                for m_, b in zip(mom, bak_m):
                    m_.copy_(b)
                for v_, b in zip(vel, bak_v):
                    v_.copy_(b)
            a_try *= 0.5
        if not step_ok:
            rejected += 1
            alpha *= 0.5
            if alpha < 1e-8:
                log("[win] alpha underflow, stopping window")
                break
            continue

        if gx_phys_diag is not None:                 # work telemetry (REFUTE M18: the
            # P-render headline metric must exist in HEADLESS runs, not only when
            # the live viewer is attached)
            dx_diag = state_n[0] - state[0].detach()
            dF_diag = state_n[1] - state[1].detach()
            dv_diag = state_n[2] - state[2].detach()
            phys_work, (phys_work_x, phys_work_F, phys_work_v) = _linearized_work(
                (gx_phys_diag, gF_phys_diag, gv_phys_diag),
                (dx_diag, dF_diag, dv_diag))
            phys_cos = _steer_cos((gx_phys_diag, gF_phys_diag, gv_phys_diag),
                                  (dx_diag, dF_diag, dv_diag))
            if gx_rend_diag is not None:
                render_work, (render_work_x, render_work_F) = _linearized_work(
                    (gx_rend_diag, gF_rend_diag), (dx_diag, dF_diag))
                render_cos = _steer_cos((gx_rend_diag, gF_rend_diag), (dx_diag, dF_diag))
            else:
                render_work = render_work_x = render_work_F = None
                render_cos = None
            step_norm = float((dFc.detach() - bak[0]).norm())
        if on_iter is not None:                      # live viewer: stream the window's
            on_iter(it, state_n[0].detach().cpu().numpy().astype(np.float32),  # optimisation
                     state_n[1].detach().reshape(N, 3, 3).cpu().numpy().astype(np.float32),
                     {"loss": new, "d_vol": float(lv_n), "kin": float(lk_n),
                      "d_render": float(lr_n) if lr_n is not None else None,
                      "lambda": lam_r if balancer.active else None, "grad_norm": gn,
                      "g_raw_cos": g_raw_cos, "g_cos": g_cos, "g_share": g_share,
                      "g_phys_norm": g_phys_norm, "g_rend_norm": g_rend_norm,
                       "render_work": render_work, "render_work_x": render_work_x,
                       "render_work_F": render_work_F, "phys_work": phys_work,
                       "phys_work_x": phys_work_x, "phys_work_F": phys_work_F,
                       "phys_work_v": phys_work_v,
                      "step_norm": step_norm, "predicted_decrease": predicted_decrease,
              "render_cos": render_cos, "phys_cos": phys_cos,
                      "_grad_phys": gx_phys_diag.detach().cpu().numpy().astype(np.float32),
                      "_grad_render": (gx_rend_diag.detach().cpu().numpy().astype(np.float32)
                                       if gx_rend_diag is not None else None)})
        # history from the ACCEPTED evaluation. NOTE "d_render" is the pure silhouette
        # scalar; the shading channel is logged separately (they were conflated before).
        hist.append({"iter": it, "loss": new,
                     "d_vol": float(lv_n), "kin": float(lk_n),
                     "d_sil": sil_gauss["sil"], "d_gauss": sil_gauss["gauss"],
                     "d_render": (float(lr_n) - cfg.w_pbr * float(lpbr_n)
                                  if lpbr_n is not None else
                                  (float(lr_n) if lr_n is not None else None)),
                     "d_pbr": float(lpbr_n) if lpbr_n is not None else None,
                      "lambda": lam_r if balancer.active else None,
                      "grad_norm": gn, "alpha": a_try,
                      "predicted_decrease": predicted_decrease,
                       "render_work": render_work, "render_work_x": render_work_x,
                       "render_work_F": render_work_F, "phys_work": phys_work,
                       "phys_work_x": phys_work_x, "phys_work_F": phys_work_F,
                       "phys_work_v": phys_work_v,
                      "step_norm": step_norm,
                     "render_cos": render_cos, "phys_cos": phys_cos,
                     "dfc_absmax": float(dFc.detach().abs().max()),
                     "s_absmax": float(s.detach().abs().max()) if s is not None else None})
        # pacing: budget reached (within one halving) — this window's share is done.
        # QUASI-STATIC commit rule (b8 forensic): a paced window that exits on its
        # shape budget after 1-2 iterations commits a body still in flight; the
        # next window's free rollout drifts on that momentum, every candidate then
        # regresses the fixed merit and the brake pins the run (354/450 rejects).
        # A window may stop early only if it did not INCREASE the kinetic term.
        if cfg.pace > 0 and new <= floor * 1.0001 + 1e-12:
            if (cfg.pace_quasistatic and lk_start is not None and lk_n is not None
                    and float(lk_n) > lk_start * (1.0 + 1e-3)):
                pass                                   # keep iterating: damp first
            else:
                pace_bound = True
                break

    # ---- final rollout: every intermediate state + FULL end state ----
    with torch.no_grad():
        lam_t, mu_t = material()
        lam_np = lam_t.detach().cpu().numpy() if lam_t is not None else lam0
        mu_np = mu_t.detach().cpu().numpy() if mu_t is not None else mu0
        dc = dFc.detach().contiguous()
        seq = [wp.from_torch(dc[t].view(N, 3, 3), dtype=wp.mat33) for t in range(T)]
        tr = Trajectory(x0, 1.0, lam_np, mu_np, prm, T, F0=F0, Fp=Fp, v0=v0, C0=C0,
                        dFc=seq, device=dev, requires_grad=False, vol0=vol0)
        tr.rollout()
        # stack-review f2: this rollout — not the accepted candidate — is what gets
        # COMMITTED, and CUDA atomics make replay non-bit-identical: validate it with
        # the same trajectory checks; a failed replay hands the runner an empty window
        # (null commit) instead of an unchecked trajectory.
        F_post = torch.stack([wp.to_torch(tr.F[t]).reshape(N, 3, 3)
                              for t in range(1, T + 1)])
        F_pre = torch.stack([wp.to_torch(tr.F[t]).reshape(N, 3, 3) for t in range(T)])
        j_eff = torch.linalg.det(F_pre + dc.view(T, N, 3, 3)).min()
        jt_final = float(torch.minimum(torch.linalg.det(F_post).min(), j_eff))
        x_final = wp.to_torch(tr.x[T])
        F_final = wp.to_torch(tr.F[T]).reshape(N, 9)
        v_final = wp.to_torch(tr.v[T])
        lv_f, lk_f, lr_f, _ = losses_of(x_final, F_final, v_final)
        E_final = scalars(lv_f, lk_f, lr_f, lam_r, dFc, x_final, F_final)
        E_accept = hist[-1]["loss"] if hist else None
        replay_tol = max(cfg.ls_noise_rel, 10.0 * replay_rel) * max(abs(E_accept or 0.0), 1.0)
        replay_bad = E_accept is not None and E_final > E_accept + replay_tol
        if ((not np.isfinite(jt_final) or jt_final <= 1e-4 or replay_bad)
                and accepted > 0):
            log(f"[win] commit rollout failed trajectory check (jt={jt_final:.3g}) — "
                "discarding window (replay/accepted-candidate mismatch)")
            hist, accepted = [], 0
        frames = [tr.x[t].numpy().copy() for t in range(T + 1)]
        F_seq = [tr.F[t].numpy().copy() for t in range(T + 1)]
        end = {"F": tr.F[T].numpy().copy(), "v": tr.v[T].numpy().copy(),
               "C": tr.C[T].numpy().copy()}
    s_out = s.detach().cpu().numpy() if s is not None else None
    if cfg.mom_carry > 0:
        mom_out = ([m.detach() for m in mom], [v.detach() for v in vel], adam_t)
    stats = {"pace_bound": pace_bound, "replay_rel": replay_rel,
             "mom_out": mom_out if cfg.mom_carry > 0 else None,
              "accepted": accepted, "rejected": rejected, "grad_converged": grad_converged,
              "L_start": L_start, "g_cos": g_cos, "g_raw_cos": g_raw_cos,
              "g_share": g_share,
              "g_phys_norm": g_phys_norm, "g_rend_norm": g_rend_norm,
              "render_work": render_work, "render_work_x": render_work_x,
              "render_work_F": render_work_F, "phys_work": phys_work,
              "phys_work_x": phys_work_x, "phys_work_F": phys_work_F,
              "phys_work_v": phys_work_v,
              "step_norm": step_norm, "predicted_decrease": predicted_decrease,
              "render_cos": render_cos, "phys_cos": phys_cos,
             "fill_lam": fill_lam if fill_on else None,
             "dfc": dc.cpu().numpy() if cfg.warm_start else None}
    return frames, F_seq, end, s_out, hist, stats
