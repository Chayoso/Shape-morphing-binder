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

import os
import time

import numpy as np
import torch
import warp as wp

from ..losses.volumetric import (d_h1, d_jdens, d_kde, d_nn_band, d_vol, d_vol_density, d_w1,
                                 deficit_field, gather_cic, isolation_gate, kde_assign,
                                 nn_band_assign, rasterize_mass, w1_budget)
from ..mpm.constitutive import lame
from ..mpm.function import PersistentAdjoint, RolloutSpec, warp_mpm_ext
from ..mpm.state import MPMParams
from ..mpm.traj import Trajectory
from .config import PipelineConfig, disc_ref_factor
from .control_basis import ControlBasis
from .grad_combine import combine as combine_grads, pcgrad as _pcgrad_impl
from .grid_smooth import chebyshev_rho, smooth_particle_field
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
    pgmin: object = None                # G1 (cfg.pbr_denoised): the morph's normal grid at the render
    pdx: float = 0.0                    #   pixel, blurred by pblur cells (the renderer's density)
    pdims: tuple = ()
    pblur: float = 0.0
    gauss: object | None = None         # GaussViews bundle (use_gauss_loss)
    gauss_scale: float | None = None    # one calibration per target build, not per window
    jd_gmin: torch.Tensor | None = None  # density-J prior raster (w_jdens>0)
    jd_dx: float = 0.0
    jd_dims: tuple = ()
    jd_rho0: torch.Tensor | None = None  # per-particle rest density (source, same estimator)
    jd_scale: float | None = None        # one-shot equal-norm calibration vs D_vol
    points: torch.Tensor | None = None   # the target sample cloud (N_t,3) on the device
    ot_pull: object = None               # losses/ot.SinkhornPull (phys_loss = "ot")
    ot_scale: float | None = None        # one-shot equal-norm calibration of the OT loss vs D_vol
    h1_scale: float | None = None        # H^-1 mass-balance term: equal-norm vs D_vol
    corr_scale: float | None = None      # 10.35 neighbourhood correspondence: equal-norm vs D_vol
    kde_h: float = 0.0                  # particle-scale density term (w_kde>0)
    kde_rho_ref: float = 1.0
    kde_scale: float | None = None      # one-shot equal-norm calibration vs D_vol
    m_ref: float = 1.0                  # loss_units="density": mean target mass per
    n_support: int = 1                  #   occupied cell, and the number of such cells
    unit_ratio: float = 1.0             # loss_units="density": MEASURED at the source,
                                        #   D_vol(legacy)/D_vol(density) — converts every
                                        #   fixed weight (REFUTE F1: the analytic per-cell
                                        #   constant was 4-45x off and one scalar cannot
                                        #   serve both uses)
    unit_grad_ratio: float = 1.0        #   |grad D_vol(legacy)|/|grad D_vol(density)| at
                                        #   the source — converts the gradient-magnitude
                                        #   constants (Adam eps, target_norm, noise floor)


# ---- per-window timing (PHYSMORPH_TIMING=1): device syncs at the section boundaries ----
_TIMING = os.environ.get("PHYSMORPH_TIMING", "") == "1"
_NO_ADJ_GRAPH = os.environ.get("PHYSMORPH_NO_ADJ_GRAPH", "") == "1"   # bisection switches
_NO_LS_BREAK = os.environ.get("PHYSMORPH_NO_LS_BREAK", "") == "1"
_TM: dict = {}


def _tick():
    if not _TIMING:
        return None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _tm_add(key, t0):
    if t0 is None:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _TM[key] = _TM.get(key, 0.0) + (time.perf_counter() - t0)
    _TM["n_" + key] = _TM.get("n_" + key, 0) + 1


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
    # Implementation: grad_combine.pcgrad (the other modes — "phys", "cagrad", "blend" —
    # live there too, docs/render_controls_physics.md §5).
    return _pcgrad_impl(gp, gr)


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


def _sobolev_direction(g: torch.Tensor, knn: torch.Tensor, kappa: float,
                       tol: float = 1e-4, max_iters: int = 200) -> torch.Tensor:
    """(I + kappa (I - A)) u = g on the material kNN graph (A = neighbour mean), Jacobi
    iterated to convergence (relative change < tol), then rescaled to |g|. Unlike
    _control_h1 (a fixed number of sweeps on one channel) this is the H1 gradient of the
    whole objective: the descent direction cannot vary between material neighbours at
    sub-stencil scale, so a lone surface particle cannot be pushed away from its
    neighbourhood by the per-particle rasterised gradient."""
    u = g
    k = float(kappa)
    for _ in range(max_iters):
        u_new = (g + k * u[:, knn].mean(2)) / (1.0 + k)
        delta = float((u_new - u).norm() / u_new.norm().clamp_min(1e-30))
        u = u_new
        if delta < tol:
            break
    return u * (g.norm() / u.norm().clamp_min(1e-30))


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


def _cic_gather(g: torch.Tensor, x: torch.Tensor, grid_min: torch.Tensor, dx: float, dims) -> torch.Tensor:
    """Trilinear gather of a per-node field g (cells, C) at the positions x — the mirror of rasterize_mass."""
    nx, ny, nz = dims
    rel = (x - grid_min) / dx
    base = torch.floor(rel).long()
    frac = rel - base.float()
    acc = torch.zeros(len(x), g.shape[1], device=x.device, dtype=g.dtype)
    for ox in (0, 1):
        wx = frac[:, 0] if ox else 1.0 - frac[:, 0]
        for oy in (0, 1):
            wy = frac[:, 1] if oy else 1.0 - frac[:, 1]
            for oz in (0, 1):
                wz = frac[:, 2] if oz else 1.0 - frac[:, 2]
                ii, jj, kk = base[:, 0] + ox, base[:, 1] + oy, base[:, 2] + oz
                valid = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny) & (kk >= 0) & (kk < nz)
                idx = ((ii * ny + jj) * nz + kk).clamp(0, nx * ny * nz - 1)
                acc = acc + (wx * wy * wz * valid.float())[:, None] * g[idx]
    return acc


def _grid_map(field: torch.Tensor, x_src: torch.Tensor, x_dst: torch.Tensor, grid_min, dx: float, dims) -> torch.Tensor:
    """P278b replay (diagnostic, 2026-09-26): a per-particle control field (T, Ns, C) at the source cloud x_src
    projected onto the loss grid (CIC node average) and sampled back at x_dst (trilinear) -> (T, Nd, C). With
    x_dst = x_src it measures the projection's own smoothing; with another cloud of the same body it hands the
    SAME grid-level control to a different sampling."""
    ones = torch.ones(len(x_src), device=x_src.device)
    wsum = rasterize_mass(x_src, ones, grid_min, dx, dims).clamp_min(1e-12)
    out = []
    for t in range(field.shape[0]):
        g = torch.stack([rasterize_mass(x_src, field[t, :, c], grid_min, dx, dims) / wsum
                         for c in range(field.shape[2])], 1)
        out.append(_cic_gather(g, x_dst, grid_min, dx, dims))
    return torch.stack(out)


def optimize_window(x0, prm: MPMParams, cfg: PipelineConfig, tgt: TargetPack,
                    balancer: LambdaBalancer, F0=None, Fp=None, v0=None, C0=None,
                    s_init=None, dfc_init=None, on_iter=None, log=print,
                    fill_bal: LambdaBalancer | None = None, alpha_scale: float = 1.0,
                    mom_init=None, vol0=None, surface_w=None, Fg0=None, coh_nbr=None,
                    coh_nbr_src=None, frontier=None, bond_rest=None, bond_frag=None,
                    u_scale_init=None, ctrl_scale_init=None, eta_init=None, pin_init=None, stick_init=None):
    """Optimise dFc[0..T-1] (+ material s) over one horizon. Returns
    (frames, F_seq, end_state, s_out, hist, stats).

    u_scale_init (N,) in [0, 1] or None: the per-particle multiplier of the u channel's one-spacing
    bound (config.u_rprop, docs/method.md 10.19; the runner updates it from the sign history of the
    accepted windows' u); the accepted u is returned in stats["u_final"].

    2026-09-14 (docs/render_controls_physics.md): the control LEAF may live on a coarse
    basis (cfg.control_grid / control_tknots) — `expand` maps it to the per-particle,
    per-step field the kernels consume; the render covariance may ride the geometric
    F_g (cfg.render_F_geom); a running kinetic term (cfg.w_kin_running) reads every
    step's velocity; grid smoothing may be Chebyshev-accelerated and covers the F
    covector; the physics/render composite follows cfg.grad_project_mode."""
    dev = cfg.device
    x0 = np.ascontiguousarray(x0, np.float32)
    N, T = x0.shape[0], cfg.T
    lam0, mu0 = lame(cfg.young, cfg.poisson)
    bond_nbr = None
    if cfg.bonds and coh_nbr is not None:
        # material re-coupling: frozen source neighbours; rest lengths are STATE carried by
        # the runner (refreshed only for coupled particles at the window start)
        bond_nbr = np.ascontiguousarray(coh_nbr, np.int32)
        if bond_rest is None:
            x0n = np.asarray(x0, np.float32)
            bond_rest = np.linalg.norm(x0n[bond_nbr] - x0n[:, None, :], axis=2).astype(np.float32)
        if bond_frag is None:
            bond_frag = np.zeros(len(x0), np.float32)
    else:
        bond_rest = None
        bond_frag = None
    m_np = (tgt.m.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(tgt.m) else 1.0)
    _mref = int(getattr(cfg, "mass_ref_n", 0) or 0)
    if _mref > 0 and int(len(x0)) != _mref:
        # the dynamics mass of the discretisation (config.mass_ref_n): the body's mass is N-invariant,
        # so a unit control moves the 300k body as it moves the 40k one; the loss-side tgt.m is untouched
        m_np = np.asarray(m_np, np.float32) * np.float32(_mref / float(len(x0)))
    if isinstance(m_np, np.ndarray) and np.allclose(m_np, 1.0):
        m_np = 1.0                                    # unit masses: keep the scalar path
    layer = None
    sp0 = None
    W_apply = None
    u_gate_frac = None
    if cfg.layer_relax or cfg.layer_ctrl:
        # outer-layer relaxation (docs/surface_gradient.md §6) and/or the position-mode control
        # channel (§7): the layer, its normals and its same-side neighbourhoods are frozen at the
        # window start; the relaxation fraction is 1/T (over one window), 0 when only the channel is on
        from scipy.spatial import cKDTree as _KD
        from ..render.surface_recon import layer_relax_data
        sub = x0[np.random.default_rng(0).choice(N, min(N, 20000), replace=False)]
        sp0 = float(np.median(_KD(sub).query(sub, k=9, workers=-1)[0][:, -1])) * (min(N, 20000) / N) ** (1.0 / 3.0)
        # the reference discretisation (config.disc_ref): the layer depth, the relaxation width and the u clip
        # at the reference spacing, the layer / asymmetry neighbour counts at the reference MASS
        _f = disc_ref_factor(N, cfg)
        sp0 *= _f
        lmask, lnrm, lnbr, lw = layer_relax_data(x0, sp0, k=int(round(cfg.layer_k * _f ** 3)), h_sp=cfg.layer_h_sp,
                                                 k_asym=int(round(32 * _f ** 3)))
        if cfg.layer_ctrl and cfg.layer_ctrl_smooth:
            # W of the relaxation as a search-direction transform on the u step (§7): rows of the
            # normalised same-side neighbour weights; zero rows off the layer
            _lnbr_t = torch.as_tensor(lnbr, device=dev, dtype=torch.long)
            _lw_t = torch.as_tensor(lw, device=dev)

            def W_apply(v):
                return (_lw_t * v[_lnbr_t]).sum(1)
        else:
            W_apply = None
        lfrac = (cfg.layer_frac if cfg.layer_frac > 0 else 1.0 / float(T)) if cfg.layer_relax else 0.0
        layer = (lmask, lnrm, lnbr, lw, float(lfrac))
        if cfg.layer_ctrl and cfg.layer_F:
            # P3 (docs/final_plan.md 2): the u channel through F — the least-squares tangential gradient
            # weights of the frozen layer neighbourhood, and the layer depth (one spacing)
            from ..render.surface_recon import layer_grad_weights
            layer = layer + (layer_grad_weights(x0, lmask, lnrm, lnbr, lw, sp0), float(cfg.layer_F_depth * sp0))
        if cfg.layer_ctrl and cfg.layer_gate and tgt.pts is not None:
            # P2 (docs/surface_gradient.md 12): u may act only where the particle-scale density residual
            # at the window start is above the sampling floor (surface_recon.layer_u_gate)
            from ..render.surface_recon import layer_u_gate
            if len(layer) == 5:
                layer = layer + (None, 0.0)
            ug, u_gate_frac = layer_u_gate(x0, tgt.pts.detach().cpu().numpy(), lmask, sp0,
                                           sigma_sp=cfg.layer_gate_sigma_sp, nsig=cfg.layer_gate_nsig)
            layer = layer + (ug,)
            print(f"[layer] u gate: {u_gate_frac * 100:.1f} % of the layer above the sampling floor", flush=True)
        if cfg.layer_ctrl and cfg.layer_gate_geom and tgt.pts is not None:
            # the geometric gate (docs/surface_gradient.md 15): u may act only where the target's outer
            # layer lies within layer_gate_geom_cells MPM cells of the particle — the sub-grid residual
            # the stress path cannot resolve; farther off, the outline is the transport's job and u's
            # per-particle step is noise on a moving surface (the mid-morph lumps)
            from ..render.surface_recon import layer_u_gate_geom
            ug_g, g_frac = layer_u_gate_geom(x0, tgt.pts.detach().cpu().numpy(), lmask, sp0,
                                             float(cfg.layer_gate_geom_cells) * float(prm.dx))
            if len(layer) == 5:
                layer = layer + (None, 0.0)
            layer = (layer[:7] + (layer[7] * ug_g,)) if len(layer) > 7 else layer + (ug_g,)
            u_gate_frac = g_frac if u_gate_frac is None else float(u_gate_frac) * g_frac
            print(f"[layer] u geometric gate: {g_frac * 100:.1f} % of the layer within "
                  f"{cfg.layer_gate_geom_cells:g} cell(s) of the target surface", flush=True)
    spec = RolloutSpec(x0=x0, m=m_np, lam=lam0, mu=mu0, prm=prm, T=T,
                       F0=F0, Fp=Fp, v0=v0, C0=C0, device=dev, vol0=vol0, Fg0=Fg0,
                       bond_nbr=bond_nbr, bond_rest=bond_rest, bond_frag=bond_frag, layer=layer,
                       eta=(np.ascontiguousarray(eta_init, np.float32) if eta_init is not None else None),
                       pin=(np.ascontiguousarray(pin_init, np.float32) if pin_init is not None else None),
                       pin_slip=bool(getattr(cfg, "settle_pin_slip", False)))

    basis = ControlBasis(x0, T, cfg.control_grid, cfg.control_tknots, device=dev)
    expand = basis.expand                       # leaf -> (T,N,3,3) control field
    dFc = basis.zeros()                         # the LEAF (per-particle when grid=0)
    leaves = [dFc]
    use_geom = bool(cfg.render_F_geom)
    # PERSISTENT no-grad trajectory (2026-09-16 speed pass) for every line-search candidate,
    # the warm-start comparison and the commit rollout: allocated once per window, rolled
    # out as a CUDA graph (Trajectory.capture/run). The control is copied into dc_buf,
    # which its dFc sequence views. It carries the SAME bonds as the RolloutSpec — before
    # this pass the no-grad rollouts had none, so the line search evaluated plain physics
    # while the adjoint described the re-coupled system.
    dc_buf = torch.zeros(T, N, 3, 3, device=dev)
    seq_eval = [wp.from_torch(dc_buf[t], dtype=wp.mat33) for t in range(T)]
    tr_eval = Trajectory(x0, m_np, lam0, mu0, prm, T, F0=F0, Fp=Fp, v0=v0, C0=C0,
                         dFc=seq_eval, device=dev, requires_grad=False, vol0=vol0,
                         Fg0=Fg0, track_geom=use_geom, persistent=True,
                         bonds=((bond_nbr, bond_rest, bond_frag, disc_ref_factor(N, cfg) ** 3)   # decoupling count = 1 reference particle
                                if bond_nbr is not None else None),
                         layer=layer, eta=spec.eta, pin=spec.pin, pin_slip=spec.pin_slip)   # the SAME viscosity / pin as the adjoint rollout
                                                                    # (2026-09-24 night: the commit rollout is this one)
    tr_eval.capture()
    adj_box = [None]                 # PersistentAdjoint, built at the first gradient rollout

    def _set_material(lam_t, mu_t):
        if lam_t is None:
            return
        tr_eval.lam.assign(np.ascontiguousarray(np.broadcast_to(lam_t.detach().cpu().numpy(), (N,)), np.float32))
        tr_eval.mu.assign(np.ascontiguousarray(np.broadcast_to(mu_t.detach().cpu().numpy(), (N,)), np.float32))
    if cfg.control_h1_iters > 0 and not basis.per_particle:
        # the kNN control preconditioner indexes PARTICLES; the leaf is nodes x knots
        # (measured: IndexError at the first window). The basis already propagates a
        # node update to every particle in its support, so the two are exclusive.
        raise ValueError("control_h1_iters > 0 requires the per-particle control leaf "
                         "(control_grid=0 and control_tknots in {0, T})")
    # unit-aware optimizer constants: Adam's eps and the adaptive-alpha target norm are
    # gradient-magnitude numbers tuned in legacy D_vol units (the C++ oracle scale);
    # in density units the same Adam trajectory needs them divided by the unit ratio
    unit_ratio = float(tgt.unit_ratio) if cfg.loss_units == "density" else 1.0
    unit_grad_ratio = float(tgt.unit_grad_ratio) if cfg.loss_units == "density" else 1.0
    eps_eff = cfg.eps / unit_grad_ratio
    target_norm_eff = cfg.target_norm / unit_grad_ratio
    # every FIXED weight that is added to D_vol is a legacy-unit number: in density mode
    # it is converted by the same ratio, so the objective's relative weighting at the
    # calibration cell is unchanged (the calibrated terms h1/jdens/kde follow dvol()
    # automatically). Measured without this: the kinetic penalty out-weighed the
    # rescaled D_vol 50:1 and no line-search step was accepted.
    wu = 1.0 / unit_ratio
    # (the line-search noise floor's absolute "1.0" and the replay tolerance are
    # rescaled the same way below)

    # frontier-restricted target (vol_frontier): target cells within one loss cell of
    # the window's start occupancy; recomputed per window by the runner
    grid_eff = tgt.grid if frontier is None else tgt.grid * frontier

    def dvol_density(xT):
        if cfg.loss_units == "density":
            return d_vol_density(xT, tgt.m, grid_eff, tgt.lgmin, tgt.ldx, tgt.ldims,
                                 tgt.m_ref, tgt.n_support, form=getattr(cfg, "dvol_form", "log"))
        if cfg.loss_units != "legacy":
            raise ValueError(f"unknown loss_units {cfg.loss_units!r}")
        return d_vol(xT, tgt.m, grid_eff, tgt.lgmin, tgt.ldx, tgt.ldims)

    pace_grid = None
    if getattr(cfg, "phys_loss", "density") == "ot_resid":
        # RESIDUAL TRANSPORT PACING. The cell sum's own residual at the window start —
        # excess = (cloud − target)+ and deficit = (target − cloud)+ per loss cell, equal
        # totals — is what has to move; a Sinkhorn plan is solved between the EXCESS mass
        # (particles drawn ∝ their excess fraction e_i = excess/cloud at their cell) and
        # the DEFICIT mass (target points drawn ∝ the deficit at their cell), and each
        # particle's paced position is x0 + e_i · min(1, h/|d|) · d with d its debiased map
        # displacement (material-smoothed, excess-weighted) and h the plan blur radius.
        # Particles in satisfied cells (e_i = 0) contribute their own position: the paced
        # target differs from the current occupancy only where the cell sum itself wants
        # mass to move, and there it asks for at most one blur radius along a coherent
        # flow — no far cell rewards a lone particle (the ejection mechanism), and no
        # interior redistribution the cell sum does not ask for (the ot_pace roughness:
        # a plan between the full cloud and the uniform target equalised the interior).
        from ..losses.ot import SinkhornPull
        x0_r = torch.as_tensor(x0, device=dev)
        _t_ot = time.perf_counter()
        m_t = torch.as_tensor(tgt.m, device=dev) if not torch.is_tensor(tgt.m) else tgt.m
        cur = rasterize_mass(x0_r, m_t, tgt.lgmin, tgt.ldx, tgt.ldims)
        ex_c = torch.clamp(cur - grid_eff, min=0.0)
        df_c = torch.clamp(grid_eff - cur, min=0.0)
        e_i = gather_cic(ex_c / cur.clamp_min(1e-12), x0_r, tgt.lgmin, tgt.ldx, tgt.ldims).clamp(0.0, 1.0)
        w_t = gather_cic(df_c, tgt.points, tgt.lgmin, tgt.ldx, tgt.ldims).clamp_min(0.0)
        n_p = x0_r.shape[0]
        p_sp = (float(tgt.nn_spacing) if tgt.nn_spacing > 0 else 0.5 * float(tgt.ldx)) / (disc_ref_factor(int(x0_r.shape[0]), cfg) if getattr(cfg, "plan_native", False) else 1.0)   # config.plan_native: the plan blur from the sample spacing, never the reference one
        h_r = p_sp * max(1.0, n_p / float(cfg.ot_samples)) ** (1.0 / 3.0)
        ex_frac = float((e_i * m_t).sum() / (m_t.sum()))
        if float(w_t.sum()) > 0 and float(e_i.sum()) > 0:
            n_s = int(min(cfg.ot_samples, int((w_t > 0).sum()), int((e_i > 0).sum())))
            gen = torch.Generator(device=dev).manual_seed(int(getattr(cfg, "seed", 0)))
            idx_t = torch.multinomial(w_t / w_t.sum(), n_s, replacement=False, generator=gen)
            idx_p = torch.multinomial((e_i * m_t) / (e_i * m_t).sum(), n_s, replacement=False, generator=gen)
            pull = SinkhornPull(tgt.points[idx_t].detach(), eps=h_r ** 2, iters=cfg.ot_iters,
                                tol=getattr(cfg, "ot_tol", 1e-2))
            pull._sub_idx = idx_p
            disp = pull.debiased_map_displacement(x0_r, n_s)
            if getattr(tgt, "ot_kd", None) is None:
                from scipy.spatial import cKDTree
                tgt.ot_kd = cKDTree(np.ascontiguousarray(tgt.points.detach().cpu().numpy(), np.float32))
                k_nb = int(max(4, min(64, round(4.0 / 3.0 * np.pi * (h_r / p_sp) ** 3))))
                x0_np = np.ascontiguousarray(np.asarray(x0, np.float32))
                _, knn = cKDTree(x0_np).query(x0_np, k=k_nb, workers=-1)
                tgt.ot_knn = torch.as_tensor(knn, device=dev)
                print(f"[win] OT resid: plan blur {h_r:.4g} wu, displacement denoised over k={k_nb} "
                      f"material neighbours (excess-weighted)", flush=True)
            wn = e_i[tgt.ot_knn]                                        # (N, k) excess weights
            disp = (disp[tgt.ot_knn] * wn[..., None]).sum(1) / wn.sum(1, keepdim=True).clamp_min(1e-12)
            dn = disp.norm(dim=1, keepdim=True)
            step = e_i[:, None] * torch.clamp(h_r / dn.clamp_min(1e-9), max=1.0)
            x_int = (x0_r + step * disp).detach()
            arrived = (e_i > 0) & (dn.squeeze(1) <= h_r)
            if bool(arrived.any()):
                _, nn_a = tgt.ot_kd.query(x_int[arrived].cpu().numpy(), workers=-1)
                x_int[arrived] = tgt.points[torch.as_tensor(nn_a, device=dev)].to(x_int.dtype)
            pace_grid = rasterize_mass(x_int, m_t, tgt.lgmin, tgt.ldx, tgt.ldims).detach()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"[win] OT resid: excess mass {ex_frac * 100:.1f}%, {n_s} samples, plan {pull.last_sweeps} sweeps "
                  f"err={pull.last_err:.3g}, arrived {float(arrived.float().mean()) * 100:.1f}%, "
                  f"{time.perf_counter() - _t_ot:.2f}s", flush=True)
        else:
            pace_grid = grid_eff                                          # nothing to move
            print(f"[win] OT resid: excess mass {ex_frac * 100:.2f}% — target grid used", flush=True)

        def dvol(xT):
            return d_vol_density(xT, tgt.m, pace_grid, tgt.lgmin, tgt.ldx, tgt.ldims,
                                 tgt.m_ref, tgt.n_support, form=getattr(cfg, "dvol_form", "log"))
    elif getattr(cfg, "phys_loss", "density") in ("ot", "ot_leash", "ot_pace", "ot_shape"):
        # H3: entropic optimal transport replaces the cell sum. The plan is solved once per
        # evaluation (warm-started), the differentiated value is the transport cost under
        # the detached plan; rescaled ONCE by gradient-norm parity with D_vol at the source.
        from ..losses.ot import SinkhornPull, target_samples
        if getattr(tgt, "ot_pull", None) is None:
            # blur radius of the plan: the spacing of the SAMPLE sets the plan is computed on
            # (ot_samples particles / ot_samples target points): particle spacing x
            # (N / ot_samples)^(1/3) — the resolution of the estimator, derived from the
            # discretisation — unless a loss-cell multiple is asked for explicitly
            n_part = int(np.asarray(x0).shape[0])
            p_sp = (float(tgt.nn_spacing) if tgt.nn_spacing > 0 else 0.5 * float(tgt.ldx)) / (disc_ref_factor(int(np.asarray(x0).shape[0]), cfg) if getattr(cfg, "plan_native", False) else 1.0)   # config.plan_native: the plan blur from the sample spacing, never the reference one
            eps_len = (cfg.ot_eps_cells * float(tgt.ldx) if cfg.ot_eps_cells > 0 else
                       p_sp * max(1.0, n_part / float(cfg.ot_samples)) ** (1.0 / 3.0))
            tgt.ot_pull = SinkhornPull(target_samples(tgt.points, cfg.ot_samples),
                                       eps=eps_len ** 2, iters=cfg.ot_iters,
                                       tol=getattr(cfg, "ot_tol", 1e-2))
            print(f"[win] OT plan: sqrt(eps)={eps_len:.4g} wu = {eps_len / float(tgt.ldx):.3g} loss cells, "
                  f"tol {getattr(cfg, 'ot_tol', 1e-2):.3g} (cap {cfg.ot_iters} sweeps), "
                  f"{cfg.ot_samples} target samples", flush=True)
            tgt.ot_scale = None
        # one plan per window: the barycentric targets T of the window's start positions;
        # with cfg.ot_debias the self-term of the debiased divergence cancels the entropic
        # shrinkage (target = x0 + (T - T_self))
        x0_ot = torch.as_tensor(x0, device=dev)
        _t_ot = time.perf_counter()
        # the dual is solved on an ot_samples-sized subsample of the particles and the
        # entropic map evaluated for all N (cost independent of N per sweep)
        # (SinkhornPull.debiased_map_full — potentials c-transformed to every target point
        # and every particle — was measured on the real 150k bunny end state: |d| p50 2.0
        # blur radii with either map, arrived 26 vs 28 %: the large plan displacements are
        # interior density redistribution, not sample noise, so the 3x dearer map is not used)
        if cfg.phys_loss == "ot_shape":
            # SHAPE transport: the particle subsample is drawn with probability inverse to
            # the cloud's cell mass at the particle, so the plan's source measure is uniform
            # over the OCCUPIED cells (the support), like the uniformly sampled target —
            # the plan transports the shape, not the mass distribution, and asks for no
            # interior redistribution (which the cell sum tolerates and ot_pace paid for
            # with rough surfaces). Redrawn every window; the solve is cold each time.
            m_t = torch.as_tensor(tgt.m, device=dev) if not torch.is_tensor(tgt.m) else tgt.m
            cur = rasterize_mass(x0_ot, m_t, tgt.lgmin, tgt.ldx, tgt.ldims)
            w_p = 1.0 / gather_cic(cur, x0_ot, tgt.lgmin, tgt.ldx, tgt.ldims).clamp_min(1e-12)
            gen = torch.Generator(device=dev).manual_seed(int(getattr(cfg, "seed", 0)))
            n_s = int(min(cfg.ot_samples, x0_ot.shape[0]))
            tgt.ot_pull._sub_idx = torch.multinomial(w_p / w_p.sum(), n_s, replacement=False, generator=gen)
            tgt.ot_pull.f = None
            tgt.ot_pull.f_cold = True
            tgt.ot_pull._self_pull = None
        if getattr(cfg, "ot_debias", False):
            ot_T = x0_ot + tgt.ot_pull.debiased_map_displacement(x0_ot, cfg.ot_samples)
        else:
            ot_T = tgt.ot_pull.entropic_map(x0_ot, cfg.ot_samples)
        leash_r = float(tgt.ot_pull.eps) ** 0.5           # the plan's own resolution
        # The hole regime (`ot`, e.g. C) uses the debiased map image as it is. The
        # material-kNN smoothing below belongs to the cell-sum regimes: on a target with a
        # hole the map is genuinely discontinuous where the material splits between the
        # arms, and averaging across that surface sends the seam into the hole (40k C with
        # the smoothing: gate stop at 26 windows, silIoU 0.79; raw image 0.96).
        if cfg.phys_loss in ("ot_leash", "ot_pace", "ot_shape"):
            # the leash anchors are the map images PROJECTED onto the target point set: the
            # entropic image sits ~0.9 spacings inside the target (blur), which made the
            # leash and the cell sum pull surface particles to different places (v1: merit
            # oscillation, gate stop at ~25 windows). Projected, both terms want the same
            # support; the plan still decides WHICH region a particle belongs to.
            from scipy.spatial import cKDTree
            if getattr(tgt, "ot_kd", None) is None:
                tgt.ot_kd = cKDTree(np.ascontiguousarray(tgt.points.detach().cpu().numpy(), np.float32))
                # material neighbourhood for denoising the sampled map: the k particles
                # inside one blur radius of a particle at the source (k from the blur
                # volume and the particle spacing), fixed for the run (material graph)
                p_sp = (float(tgt.nn_spacing) if tgt.nn_spacing > 0 else 0.5 * float(tgt.ldx)) / (disc_ref_factor(int(np.asarray(x0).shape[0]), cfg) if getattr(cfg, "plan_native", False) else 1.0)   # config.plan_native: the plan blur from the sample spacing, never the reference one
                k_nb = int(max(4, min(64, round(4.0 / 3.0 * np.pi * (leash_r / p_sp) ** 3))))
                x0_np = np.ascontiguousarray(np.asarray(x0, np.float32))
                from ..render.knn_gpu import knn_self
                _, tgt.ot_knn = knn_self(x0_np, k_nb)     # GPU hash grid (2026-09-23); scipy rows
                tgt.ot_knn = torch.as_tensor(tgt.ot_knn, device=dev)
                print(f"[win] OT leash: displacement denoised over k={k_nb} material neighbours "
                      f"(blur radius {leash_r:.4g} wu / spacing {p_sp:.4g})", flush=True)
            # the per-particle sampled map is noisy at the sample scale (~0.9 spacings on
            # the real bunny); the map of the continuum is smooth, so the displacement is
            # averaged over the material neighbourhood before projection
            disp = (ot_T - x0_ot)
            disp = disp[tgt.ot_knn].mean(dim=1)
            ot_T = x0_ot + disp
            _stk = None
            if getattr(cfg, "plan_sticky", False) and stick_init is not None:
                # config.plan_sticky (2026-09-25 13:30 CDT, method.md 10.33): an arrived particle KEEPS the target point it
                # arrived at. The plan is re-solved every window and, on a filled surface, the assignment of arrived
                # material is free up to a permutation: each re-solve hands an arrived particle a slightly different
                # endpoint, it chases it along the surface (the cell sum is flat there), it never reverses twice and is
                # never pinned — 40-50 % of the body free at the end of every 300k run (bm300 49 %), drifting through the
                # delivered tail. With its endpoint fixed the arrived material settles, reverses, pins. No constant.
                _stk = torch.as_tensor(np.asarray(stick_init, np.int64), device=dev)
                _has = _stk >= 0
                if bool(_has.any()):
                    ot_T = ot_T.clone(); ot_T[_has] = tgt.points[_stk[_has]].to(ot_T.dtype)
                    disp = ot_T - x0_ot
            # Per-particle target variants tried on the 150k C and FALSIFIED (2026-09-17
            # night): (i) a PACED target — each window's target bounded to one pace =
            # max(plan blur, loss cell) along the smoothed map — cut the 40k C's
            # re-attachments 114 → 7 but at 150k gave silIoU 0.88 / 0.55 (unpaced 0.95): a
            # target that walks with the particle makes the window loss quasi-stationary,
            # so the inner line search and the merit gate see no descent and stop the run
            # on a half-formed body; (ii) resolving the displacement on the loss grid (CIC
            # deposit/gather) before the pace, 0.59; (iii) the material-kNN smoothing alone
            # on the hole regime, 0.79 at 40k. The chunks those variants were meant to stop
            # were particles frozen in the domain's boundary band (mpm/kernels.py
            # k_grid_op, the separating walls) — not a property of the target.
            if cfg.phys_loss == "ot_leash":
                _, nn = tgt.ot_kd.query(ot_T.detach().cpu().numpy(), workers=-1)
                ot_T = tgt.points[torch.as_tensor(nn, device=dev)].detach().to(ot_T.dtype)
        corr_chat = None                    # config.w_corr (10.35): the centroid of each neighbourhood's paced images
        pace_grid = None
        pace_proj_stats = None
        arrived_mask_np = None
        arrive_idx_np = None                # the plan_sticky arrival index; set only in the ot_pace block (the "ot" branch —
                                            # the C target, 64 % of its source in target-empty cells — crashed on it, g41ld 02:13)
        pace_r_np = None                    # the paced target's arrival radius (config.settle_pin_clear reads it)
        plan_img_np = None                  # the plan image per particle (config.settle_pin_ray: the transit rays)
        arrive_cap_frac = None              # config.arrive_cap: the fraction of arrivals over capacity (kept on the plan image)
        pace_front_frac = None              # config.pace_front: the fraction of particles whose image was clamped at the front
        pace_front_fill_frac = None         # config.pace_front_fill: the fraction of particles assigned to a front vacancy
        sils_eff, shade_eff, pbr_grid_eff = tgt.sils, tgt.shade, True   # config.render_paced replaces them per window
        if cfg.phys_loss in ("ot_pace", "ot_shape"):
            # DISPLACEMENT-INTERPOLATED TARGET (McCann interpolation along the transport
            # plan): this window's target density is the current cloud advected toward its
            # map images by at most one blur radius per particle, rasterised with the same
            # CIC splat the loss uses. The cell sum then only ever asks for local moves along
            # the plan — no cell far from a particle rewards it for leaving the body (the
            # H3 mechanism) — and once every particle is within a blur radius of its image
            # the target is the image cloud itself (the transport's end state).
            dn = disp.norm(dim=1, keepdim=True)
            # the pace is the resolution the LOSS can see: one loss cell when that is
            # coarser than the plan's blur radius (C forensic, cd9: with a 0.18 wu pace on a
            # 0.31 wu cell the paced grid of a blob sliding along an arm equals its current
            # grid except at the ends, the cell sum is blind to the shift, and the transport
            # divergence stalls at 5 % arrived)
            pace_r = max(leash_r, float(tgt.ldx))
            arr_r = pace_r                       # the ARRIVAL radius: config.pace_lead leaves it here
            if float(getattr(cfg, "pace_lead", 0.0)) > 0:
                # reviewer item 3 (2026-09-26 01:50 CDT): the LEAD distance of the paced target decoupled from the
                # arrival radius — a small fixed pace at the existing grid (P280: the 36^3 cell sum sees a plan-shaped
                # 0.5-spacing move), while "arrived", the snap, Rprop-arrived and the pin keep the previous radius.
                pace_r = float(cfg.pace_lead)
            if float(getattr(cfg, "pace_lead_sp", 0.0)) > 0:
                # the same lead as a RULE (2026-09-26 03:20 CDT, P284: a lead of one native particle spacing fed the
                # slab 2-3x denser at 300k with the fit up 0.005 and no stall): lead = pace_lead_sp x the source
                # cloud's native spacing (the median nearest-neighbour distance, measured once) — the smallest step
                # the body delivers coherently (P278: ~1 spacing per early window at both N). No world-unit constant.
                if getattr(tgt, "native_sp", None) is None:
                    # measured on the SOURCE cloud (coh_nbr_src), not the window's start positions: the target pack
                    # is rebuilt in some runs (bp306d, the 300k dragon: a second measurement on the stretched cloud
                    # gave 0.0527 instead of 0.0350 wu, 2026-09-26 02:10 CDT)
                    _xs_np = np.ascontiguousarray(np.asarray(coh_nbr_src if coh_nbr_src is not None else x0, np.float32))
                    from scipy.spatial import cKDTree as _KDn
                    tgt.native_sp = float(np.median(_KDn(_xs_np).query(_xs_np, k=2, workers=-1)[0][:, 1]))
                    print(f"[win] pace lead: {cfg.pace_lead_sp:g} x native spacing {tgt.native_sp:.4f} wu "
                          f"({'source' if coh_nbr_src is not None else 'window start'}) = "
                          f"{cfg.pace_lead_sp * tgt.native_sp:.4f} wu (arrival radius {arr_r:.4f})", flush=True)
                pace_r = float(cfg.pace_lead_sp) * float(tgt.native_sp)
            step = torch.clamp(pace_r / dn.clamp_min(1e-9), max=1.0)
            if getattr(cfg, "pace_coherent", False):
                # config.pace_coherent (2026-09-26 23:00, method.md 10.30): the ear's material is one column of the head
                # that the plan translates by one length, and the physics STRETCHES it in transit (its rear inside the
                # head is slow, its front in the ear free): 68-100 % of the tip-bound material stands above the neck
                # before 30 % of the neck-bound has arrived, on every form, with or without the render channel — the
                # stretched column is the spike, its lead bunching at the top the knob. The paced target keeps the
                # material neighbourhood together: a particle more than one blur radius AHEAD of its plan neighbours'
                # centroid along its own ray (the neighbourhood is a ball of that radius at the source, fixed for the
                # run) has its step shortened by the excess and waits for the rear; particles behind advance at the
                # pace. A smooth map moves a neighbourhood together, so the bulk is untouched. No new constant.
                _cen = x0_ot[tgt.ot_knn].mean(dim=1)                                       # (N, 3) neighbourhood centroid
                _u = disp / dn.clamp_min(1e-9)                                              # the ray direction
                _lead = ((x0_ot - _cen) * _u).sum(dim=1, keepdim=True)                      # offset ahead along the ray
                _excess = (_lead - float(leash_r)).clamp_min(0.0)
                step = torch.clamp((pace_r - _excess).clamp_min(0.0) / dn.clamp_min(1e-9), max=1.0)
                pace_coh_frac = float((_excess > 0.0).float().mean())
                print(f"[win] coherent pace: {100.0 * pace_coh_frac:.1f} % of the particles ahead of their neighbourhood "
                      f"(held share of the step {float((_excess.clamp_max(pace_r) / pace_r).mean()):.3f})", flush=True)
            if getattr(cfg, "pace_support", False):
                # config.pace_support (2026-09-25 17:30 CDT, method.md 10.34): the empty-looking transit measured — above the
                # head at t = 0.10-0.14 the material present is 5-8 % of the target's density (bm300, bq300 alike): the
                # front's few leads run at the pace through free space while the bulk drags through the body, and
                # density = flux / speed falls. The stream pace (10.32) read the material BEHIND a lead — the body, dense
                # but slow — and let it run (bq300: 0.06 / 0.08 at t = 0.10 / 0.14, unchanged). The step is instead
                # scaled by the density AT the particle: the particles within the target's shell radius around it over
                # half the target's own count at its nearest target point (the body convention). Under-dense material
                # waits for its bulk; the bulk moves at the pace; the front is a plug. No constant.
                from scipy.spatial import cKDTree as _KDu
                if getattr(tgt, "supp_tree", None) is None:
                    _Pu = tgt.pts.detach().cpu().numpy().astype(np.float32)
                    tgt.supp_tree = _KDu(_Pu)
                    _d8 = tgt.supp_tree.query(_Pu, k=9, workers=-1)[0]
                    tgt.supp_rcov = float(np.median(_d8[:, 8]))
                    tgt.supp_cnt = np.asarray(tgt.supp_tree.query_ball_point(_Pu, r=tgt.supp_rcov, return_length=True), np.float32) - 1.0
                    print(f"[win] support pace: target shell radius {tgt.supp_rcov:.4f} wu, median count {float(np.median(tgt.supp_cnt)):.1f}", flush=True)
                _x0u = x0_ot.detach().cpu().numpy().astype(np.float32)
                _n_i = np.asarray(_KDu(_x0u).query_ball_point(_x0u, r=tgt.supp_rcov, return_length=True), np.float32) - 1.0
                _, _q = tgt.supp_tree.query(_x0u, k=1, workers=-1)
                _need = np.maximum(1.0, 0.5 * tgt.supp_cnt[_q])
                _supp = np.clip(_n_i / _need, 0.0, 1.0)
                if getattr(cfg, "pace_support_hard", False):
                    _supp = (_supp >= 1.0).astype(np.float32)                    # the body convention as a gate: below half, no step
                _supp_t = torch.as_tensor(_supp, device=dev, dtype=dn.dtype).unsqueeze(1)
                step = torch.clamp(_supp_t * pace_r / dn.clamp_min(1e-9), max=1.0)
                print(f"[win] support pace: {100.0 * float((_supp < 1.0).mean()):.1f} % under-dense (mean support there "
                      f"{float(_supp[_supp < 1.0].mean()) if (_supp < 1.0).any() else 1.0:.2f})", flush=True)
            if getattr(cfg, "pace_stream", False):
                # config.pace_stream (2026-09-27 01:10, method.md 10.32): the coherent pace of 10.30 compares a particle with
                # its source ball (one blur radius, k neighbours) and the column's layers — tip-, neck-, base-bound — lie
                # 0.2-0.5 wu apart at the source, outside each other's balls: the rule bound on 0.2 % of the particles
                # and the ear still rose as a thin jet by eye (the column's centre outruns its periphery and the layers
                # behind). The stream rule reads the material BEHIND a particle on its own ray: the ball of one blur
                # radius around the point one pace step behind it, against half the count that ball holds at the source
                # density (k, the plan's own neighbourhood count; half = the body convention). A particle's step scales
                # with that fill: a lead with a sparse stream behind it waits, a particle inside a continuous stream
                # advances at the pace, the bulk (a moving body) is untouched. No new constant.
                _u_s = disp / dn.clamp_min(1e-9)
                _rear = (x0_ot - float(pace_r) * _u_s).detach().cpu().numpy().astype(np.float32)
                from scipy.spatial import cKDTree as _KDs
                _x0np = x0_ot.detach().cpu().numpy().astype(np.float32)
                _cnt_r = np.asarray(_KDs(_x0np).query_ball_point(_rear, r=float(leash_r), return_length=True, workers=-1), np.float32)
                _k_nb = int(tgt.ot_knn.shape[1]) if getattr(tgt, "ot_knn", None) is not None else 64
                _fill_r = torch.as_tensor(np.minimum(1.0, _cnt_r / max(1.0, 0.5 * _k_nb)), device=dev, dtype=dn.dtype).unsqueeze(1)
                step = torch.clamp(_fill_r * pace_r / dn.clamp_min(1e-9), max=1.0)
                pace_stream_frac = float((_fill_r < 1.0).float().mean())
                print(f"[win] stream pace: {100.0 * pace_stream_frac:.1f} % of the particles with a sparse stream behind "
                      f"(mean fill there {float(_fill_r[_fill_r < 1.0].mean()) if pace_stream_frac > 0 else 1.0:.2f}; k {_k_nb})", flush=True)
            x_int = (x0_ot + step * disp).detach()
            if getattr(cfg, "pace_project", False):
                # the SUPPORT-PRESERVING paced target (docs/method.md 10.22): the straight-ray
                # interpolant of an anisotropic map is not volume preserving in transit (a fan
                # converging into a thin feature is over-dense at its root and under-dense in the
                # stream; at 300k the cloud realises that as a base bulge feeding a sub-cell
                # filament). The paced step's divergent part on the body is removed (Chorin
                # projection on the loss grid, pressure zero on the free surface) so the advected
                # cloud keeps bulk density and thin features are extruded from the body.
                from ..losses.projection import bulk_mode, project_step
                m_pj = torch.as_tensor(tgt.m, device=dev) if not torch.is_tensor(tgt.m) else tgt.m
                if getattr(tgt, "bulk_node", None) is None:
                    # the body's reference density is the TARGET's bulk node mass, read once from
                    # the fixed target grid (a cloud in transit gives no reliable estimate of it)
                    tgt.bulk_node = bulk_mode(grid_eff.detach())
                    print(f"[win] pace project: bulk node mass from the target grid {tgt.bulk_node:.1f} "
                          f"(the body = nodes at half of it or more)", flush=True)
                d_pj, pace_proj_stats = project_step(x0_ot, (step * disp).detach(), m_pj,
                                                     tgt.lgmin, tgt.ldx, tgt.ldims, bulk=tgt.bulk_node)
                x_int = (x0_ot + d_pj).detach()
                _pp = pace_proj_stats["peak_pos"]
                print(f"[win] pace project: body {pace_proj_stats['body']} nodes (densest node "
                      f"{pace_proj_stats['peak']:.1f}x bulk at ({_pp[0]:.2f}, {_pp[1]:.2f}, {_pp[2]:.2f}), "
                      f"{100 * pace_proj_stats['dense_frac']:.1f} % of the mass in nodes >= 2x bulk), step divergence rms "
                      f"{pace_proj_stats['div0']:.3f} -> {pace_proj_stats['div1']:.4f} per window "
                      f"(p95 {pace_proj_stats['div0_p95']:.3f}), correction median "
                      f"{pace_proj_stats['corr_med']:.3f} / p95 {pace_proj_stats['corr_p95']:.3f} cells, "
                      f"grid view of the step off by {100 * pace_proj_stats['pic_change']:.0f} %, "
                      f"CG {pace_proj_stats['cg_iters']} it (res {pace_proj_stats['cg_res']:.1e})", flush=True)
            # an ARRIVED particle (within one blur radius of its image) contributes its
            # image projected onto the target point set: the entropic image sits inside
            # the target (blur), which left the end state fuzzy (150k bunny chamfer 0.098
            # vs 0.076); on the target support the end target is the target. (Snapping
            # every paced position that lies on the support instead — d4db68a — killed the
            # tangential transport where the source overlaps the target: 150k cow silIoU
            # 0.920 vs 0.944, 104 re-attachments vs 83.)
            if getattr(cfg, "pace_front", False):
                # config.pace_front (2026-09-26, method.md 10.29): the target is REVEALED as a front. A straight-ray
                # displacement interpolation into a thin feature is a filament (the ear rises as a spike, the tip bound
                # particles move in parallel with the base-bound ones and arrive by ray length); the density terms
                # then thicken the spike after the fact, and the silhouette term is satisfied by the spike as much as
                # by a tongue. Here a particle may only be sent to target cells that are filled or within one pace
                # step of a filled one: its image is clamped along its own ray at the revealed region's boundary, so
                # the feature fills from its base at the target's cross-section (the front advances one pace step per
                # window). "Filled" = the cell holds at least half of the target's mass in it (the target's own
                # occupancy); the reveal step = the pace radius (the arrival scale). No new constant.
                nxg, nyg, nzg = tgt.ldims
                _tg = tgt.grid.reshape(nxg, nyg, nzg)
                _cur = rasterize_mass(x0_ot, tgt.m, tgt.lgmin, tgt.ldx, tgt.ldims).reshape(nxg, nyg, nzg)
                _filled = (_tg > 0) & (_cur >= 0.5 * _tg)
                _reach = max(1, int(round(float(pace_r) / float(tgt.ldx))))
                _rev = torch.nn.functional.max_pool3d(_filled.float()[None, None], kernel_size=2 * _reach + 1,
                                                      stride=1, padding=_reach)[0, 0] > 0.5
                _rev = _rev & (_tg > 0)                                  # revealed = target cells near filled ones
                _rev = _rev | (_tg > 0) & (_cur > 0)                      # every occupied target cell counts
                def _in_rev(p):
                    ijk = torch.floor((p - tgt.lgmin) / tgt.ldx).long()
                    ok = ((ijk >= 0) & (ijk < torch.tensor([nxg, nyg, nzg], device=p.device))).all(1)
                    ijk = ijk.clamp(min=0); ijk[:, 0].clamp_(max=nxg - 1); ijk[:, 1].clamp_(max=nyg - 1); ijk[:, 2].clamp_(max=nzg - 1)
                    return ok & _rev[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
                _out = ~_in_rev(x_int)
                if bool(_out.any()):
                    _K = 24
                    _t = torch.linspace(0.0, 1.0, _K, device=dev)                      # samples along the ray x0 -> image
                    _seg = x0_ot[_out][:, None, :] + _t[None, :, None] * (x_int[_out] - x0_ot[_out])[:, None, :]
                    _inside = _in_rev(_seg.reshape(-1, 3)).reshape(-1, _K)
                    _inside[:, 0] = True                                                # the particle's own position
                    _last = (_inside.float() * torch.arange(_K, device=dev).float()[None, :]).max(1).values.long()
                    x_int = x_int.clone()
                    x_int[_out] = _seg[torch.arange(_seg.shape[0], device=dev), _last]
                    pace_front_frac = float(_out.float().mean())
                else:
                    pace_front_frac = 0.0
            if (getattr(cfg, "pace_front_pts", False) or getattr(cfg, "pace_front_geo", False)) and tgt.pts is not None:
                # config.pace_front_pts (2026-09-26, method.md 10.29 at the particle scale): the grid front orders the
                # growth at the loss-cell scale (the head's hump) and cannot order it below — the ear's spike is a
                # filament thinner than the cell. Here the front is read on the target's own points: a target point is
                # FILLED when a particle lies within one spacing of it, REVEALED when within one pace step of a filled
                # one, and a sample on a particle's ray is inside when a revealed point lies within one spacing of it
                # (the coverage probe's definition). A tip-bound particle's image is clamped along its ray at the last
                # inside sample. No new constant: the spacing is the target's, the reveal step the arrival radius.
                from scipy.spatial import cKDTree as _KDf
                _P = tgt.pts.detach().cpu().numpy().astype(np.float32)
                _sp = float(getattr(tgt, "nn_spacing", 0.0) or 0.0)
                if _sp <= 0.0:
                    _sp = float(np.median(_KDf(_P).query(_P, k=2, workers=-1)[0][:, 1]))
                if getattr(tgt, "front_tree", None) is None:
                    # the inside / filled radius is the sample's COVERAGE radius, not its median spacing (2026-09-26 17:40):
                    # 24-27 % of a target-filling particle cloud lies farther than one spacing from every target point (the
                    # point cloud's own gaps) and 0-1 % farther than the 8-neighbour shell radius (1.98 spacings) — the fronts
                    # of g41fp / g41fq / g41fg held a standing 26-32 % of the images for that reason (the serialisation of the
                    # bulk targets and the dragon's stall were mostly this). Per target point, its shell radius.
                    tgt.front_tree = _KDf(_P)
                    tgt.front_rcov = tgt.front_tree.query(_P, k=9, workers=-1)[0][:, 8].astype(np.float32)
                    print(f"[win] front: coverage radius median {float(np.median(tgt.front_rcov)):.4f} wu = "
                          f"{float(np.median(tgt.front_rcov)) / max(_sp, 1e-9):.2f} spacings", flush=True)
                _treeP = tgt.front_tree; _rcov = tgt.front_rcov
                _X0 = x0_ot.detach().cpu().numpy().astype(np.float32)
                if getattr(cfg, "pace_front_dense", False):
                    # config.pace_front_dense (2026-09-26 20:40): the ear's material is one column below the ear that the plan
                    # translates by one length (tip-, neck- and base-bound groups all travel 1.34 wu on ar300), and the
                    # physics STRETCHES the column in transit (its rear inside the head is slow, its front in the ear is
                    # free): at t = 0.4 68 % of the tip-bound material is above the neck slab while 27 % of the neck-bound
                    # has arrived — the stretched column is the spike, the tip group bunching at the top is the knob. A
                    # point counted as filled by ONE particle lets a sparse lead run the front at the pace (a chain); here
                    # a point is filled when the particles within its shell radius reach HALF its own count there (the
                    # target holds 8 by the shell's definition; half = the body convention of 10.22, scaled by n / |target|),
                    # so the front advances only as a plug at the target's density — the material accumulates at the
                    # boundary until the rear catches up: the tongue. No new constant.
                    _need = max(1, int(round(0.5 * 8.0 * len(x_int) / max(1, len(_P)))))
                    _cntP = _KDf(_X0).query_ball_point(_P, r=_rcov, return_length=True, workers=-1)
                    _filled = np.asarray(_cntP) >= _need
                else:
                    _dP, _ = _KDf(_X0).query(_P, k=1, workers=-1)
                    _filled = _dP <= _rcov                              # a target point with a particle within its shell radius
                if getattr(cfg, "pace_front_geo", False):
                    # config.pace_front_geo (2026-09-26, method.md 10.29 eq. 52b): the fill-based front below holds every
                    # particle whose ray does not touch a filled point — the part of the source outside the target waits
                    # until the fill reaches its surface, and a bulk target is transported as a wave at half the pace
                    # (g41fp nefertiti: 115 windows for g41pw's 57, -0.008). Here the front is the TARGET grown along its
                    # own geodesics from the region the source occupies at the first window, one pace step per window:
                    # material moving along the target (the ear's base-to-tip order) is never held, only material that
                    # would shortcut through the air to a region the growth has not reached. A piece the graph cannot
                    # reach takes its Euclidean distance from the origin. No new constant (the pace, the spacing).
                    if getattr(tgt, "front_geo_d", None) is None:
                        from scipy.sparse import coo_matrix as _coo
                        from scipy.sparse.csgraph import dijkstra as _dijk
                        _dk, _ik = _KDf(_P).query(_P, k=9, workers=-1)
                        _src = np.repeat(np.arange(len(_P)), 8); _dst = _ik[:, 1:].reshape(-1); _w = _dk[:, 1:].reshape(-1)
                        _ok = _dst < len(_P)
                        _G = _coo((_w[_ok].astype(np.float64), (_src[_ok], _dst[_ok])), shape=(len(_P), len(_P))).tocsr()
                        _orig = np.nonzero(_filled)[0]
                        if len(_orig) == 0:
                            _orig = np.array([int(np.argmin(_KDf(_X0).query(_P, k=1, workers=-1)[0]))])
                        _dg = _dijk(_G, directed=False, indices=_orig, min_only=True)
                        _isl = int((~np.isfinite(_dg)).sum())
                        _de, _ = _KDf(_P[_orig]).query(_P, k=1, workers=-1)
                        _dg = np.where(np.isfinite(_dg), _dg, _de).astype(np.float32)
                        tgt.front_geo_d = _dg; tgt.front_geo_k = 0
                        print(f"[win] geodesic front: origin {len(_orig)} of {len(_P)} target points, reach "
                              f"{float(_dg.max()):.3f} wu = {float(_dg.max()) / float(pace_r):.1f} pace steps, "
                              f"{_isl} island points", flush=True)
                    else:
                        tgt.front_geo_k += 1
                    _revealed = tgt.front_geo_d <= float(pace_r) * float(tgt.front_geo_k + 1)
                else:
                    if _filled.any() and (~_filled).any():
                        if getattr(cfg, "pace_front_dense", False):
                            # (20:55) the reveal is one SHELL beyond the filled region — the fill's own resolution — not one
                            # pace (0.3 wu at 40k: a lead a third of the ear ahead of the filled boundary was never held)
                            _dR, _ = _KDf(_P[_filled]).query(_P[~_filled], k=1, workers=-1)
                            _revealed = _filled.copy(); _revealed[np.nonzero(~_filled)[0][_dR <= _rcov[~_filled]]] = True
                        else:
                            _dR, _ = _KDf(_P[_filled]).query(_P[~_filled], k=1, distance_upper_bound=float(pace_r), workers=-1)
                            _revealed = _filled.copy(); _revealed[np.nonzero(~_filled)[0][np.isfinite(_dR)]] = True
                    else:
                        _revealed = _filled | (~_filled)                 # nothing filled yet (the first window) or all filled
                if getattr(cfg, "pace_front_thin", False):
                    # config.pace_front_thin (2026-09-26 19:40): the front orders only the part of the target the cell sum
                    # cannot resolve — points whose CIC nodes are all below half the target's bulk node mass (the body's
                    # definition of 10.22, read once from the target grid). The bulk is transported by the pace as before:
                    # its arrangement is the loss's to resolve, and every front form so far scrambled it (nefertiti under
                    # 52c+52e: 22 % pinned at 45 windows with the target 99.7 % covered — the plan's endpoints no longer
                    # match where the accreted material sits). The thin features still grow from their base.
                    if getattr(tgt, "front_bulk_pt", None) is None:
                        from ..losses.projection import bulk_mode as _bm
                        _g3 = tgt.grid.detach().reshape(tuple(int(v) for v in tgt.ldims)).cpu().numpy()
                        _bn = float(_bm(tgt.grid.detach()))
                        _lg = tgt.lgmin.detach().cpu().numpy() if torch.is_tensor(tgt.lgmin) else np.asarray(tgt.lgmin)
                        _b0 = np.floor((_P - np.asarray(_lg, np.float32)[None, :]) / float(tgt.ldx)).astype(np.int64)
                        _mx = np.zeros(len(_P), np.float32)
                        for _ox in (0, 1):
                            for _oy in (0, 1):
                                for _oz in (0, 1):
                                    _ix = np.clip(_b0[:, 0] + _ox, 0, _g3.shape[0] - 1)
                                    _iy = np.clip(_b0[:, 1] + _oy, 0, _g3.shape[1] - 1)
                                    _iz = np.clip(_b0[:, 2] + _oz, 0, _g3.shape[2] - 1)
                                    _mx = np.maximum(_mx, _g3[_ix, _iy, _iz].astype(np.float32))
                        tgt.front_bulk_pt = _mx >= 0.5 * _bn
                        tgt.front_bulk_node = (tgt.grid.detach() >= 0.5 * _bn)
                        print(f"[win] front: thin part of the target = {100.0 * float((~tgt.front_bulk_pt).mean()):.1f} % of its "
                              f"points (bulk node mass {_bn:.1f}; a point is bulk when one of its CIC nodes holds half of it)", flush=True)
                    _revealed = _revealed | tgt.front_bulk_pt
                _XI = x_int.detach().cpu().numpy().astype(np.float32)
                # (19:55) a ray is held only where it passes through UNREVEALED TARGET: a sample in the air is passable (the
                # source's part outside the target flies to the target as the pace says — holding it was the air-side hold
                # that serialised the bulk targets and, under 52e, re-routed 30 % of the material to vacancies), a sample
                # inside the target is passable when its nearest point is revealed; the image is clamped at the last
                # passable sample before the first unpassable one (contiguous from the particle).
                _K = 24
                _tt = np.linspace(0.0, 1.0, _K, dtype=np.float32)
                _segA = _X0[:, None, :] + _tt[None, :, None] * (_XI - _X0)[:, None, :]
                _dSA, _qSA = _treeP.query(_segA.reshape(-1, 3), k=1, workers=-1)
                _badA = ((_dSA <= _rcov[_qSA]) & ~_revealed[_qSA]).reshape(-1, _K); _badA[:, 0] = False
                _out = _badA.any(1)
                if _out.any():
                    _first = np.argmax(_badA[_out], axis=1)             # the first unpassable sample (one exists)
                    _last = np.maximum(_first - 1, 0)
                    _seg = _segA[_out]
                    _new = _seg[np.arange(_seg.shape[0]), _last]
                    if getattr(cfg, "pace_front_fill", False):
                        # config.pace_front_fill (2026-09-26, method.md 10.29 eq. 52c): the clamp piles every held image at one
                        # place on its ray — without the cap the pile over-fills the front cell (g41fp dragon: 44 below det F
                        # 0.5), with it the front CELL is full before the unfilled target POINTS behind it are reached and a
                        # sub-cell spike deadlocks (g41fq dragon: 35 % pinned at 86 windows). A held image is assigned instead
                        # to a revealed VACANCY (a revealed point without a particle within one spacing) within one pace of its
                        # clamp, one particle per point (the capacity ratio of 10.28), closest first; no vacancy in reach =
                        # keep the clamp. The front then holds exactly the target's mass and the fill advances at the point scale.
                        # (52e, 18:10) and a held particle with NO vacancy within one pace does not wait either: the part of
                        # the source outside the target (dragon 39 %, nefertiti 56 % of the particles at the first window)
                        # has nothing revealed on its ray and would sit until the fill walked to it — the bulk targets'
                        # serialisation. Its image is instead its nearest vacancy, wherever the front is (one per point
                        # among its 8 nearest, closest first), approached at the pace: material outside the target accretes
                        # at the growing front, which is the "volume first" growth the user described, and nothing seeds an
                        # unrevealed thin feature from the air.
                        _vac = np.nonzero(_revealed & ~_filled)[0]
                        _nfill = 0; _nappr = 0
                        if len(_vac) > 0:
                            _capv = max(1, int(round(len(x_int) / max(1, len(_P)))))
                            _goal = np.full(len(_new), -1, np.int64)
                            _cnt = np.zeros(len(_vac), np.int32)
                            _free = np.ones(len(_new), bool); _open = np.ones(len(_vac), bool)
                            for _round in range(64):                         # each round: nearest OPEN vacancy, closest first
                                _pi = np.nonzero(_free)[0]; _vi = np.nonzero(_open)[0]
                                if len(_pi) == 0 or len(_vi) == 0:
                                    break
                                _dv, _iv = _KDf(_P[_vac[_vi]]).query(_new[_pi], k=1, workers=-1)
                                _ordv = np.lexsort((_dv, _iv)); _ivs = _iv[_ordv]
                                _firstv = np.r_[True, _ivs[1:] != _ivs[:-1]]
                                _gs = np.maximum.accumulate(np.where(_firstv, np.arange(len(_ivs)), 0))
                                _rank = np.arange(len(_ivs)) - _gs
                                _take = np.zeros(len(_pi), bool); _take[_ordv] = _rank < (_capv - _cnt[_vi[_iv]][_ordv])
                                _sel = np.nonzero(_take)[0]
                                if len(_sel) == 0:
                                    break
                                _goal[_pi[_sel]] = _vi[_iv[_sel]]
                                np.add.at(_cnt, _vi[_iv[_sel]], 1)
                                _free[_pi[_sel]] = False
                                _open[_vi] = _cnt[_vi] < _capv
                            _has = _goal >= 0
                            if _has.any():
                                _gp = _P[_vac[_goal[_has]]]; _c0 = _new[_has]
                                _dvec = _gp - _c0; _dl = np.linalg.norm(_dvec, axis=1, keepdims=True)
                                _near = (_dl[:, 0] <= float(pace_r))
                                _stp = np.minimum(1.0, float(pace_r) / np.maximum(_dl, 1e-9))
                                _new[_has] = np.where(_near[:, None], _gp, _c0 + _stp * _dvec).astype(np.float32)
                                _nfill = int(_near.sum()); _nappr = int((~_near).sum())
                        pace_front_fill_frac = float(_nfill) / float(len(x_int))
                        print(f"[win] front: held {float(_out.mean()):.3f}, vacancies {len(_vac)}, assigned {pace_front_fill_frac:.3f}, "
                              f"approaching {float(_nappr) / float(len(x_int)):.3f}", flush=True)
                    x_int = x_int.clone()
                    x_int[torch.as_tensor(np.nonzero(_out)[0], device=dev)] = torch.as_tensor(_new, device=dev, dtype=x_int.dtype)
                pace_front_frac = float(_out.mean())
            arrived = dn.squeeze(1) <= arr_r
            arrive_idx_np = None
            if bool(arrived.any()):
                _, nn_a = tgt.ot_kd.query(x_int[arrived].cpu().numpy(), workers=-1)
                if _stk is not None:
                    # (2026-09-25 19:05 CDT) the sticky assignment with CAPACITY: a stuck particle snaps to its own point; an
                    # arriving particle takes the nearest target point not reserved by another (closest arrivals first, one
                    # per point, reserved for the run); none free among its 8 nearest -> it keeps the plan image, not stuck.
                    # The first form froze several particles onto one point and they competed for it forever (bp300: 7 %
                    # pinned at window 40).
                    _stk_np = _stk.cpu().numpy(); _stk_a = _stk_np[arrived.detach().cpu().numpy()]
                    _reserved = np.zeros(len(tgt.points), bool); _reserved[_stk_np[_stk_np >= 0]] = True
                    _arr_idx = torch.nonzero(arrived).squeeze(1).cpu().numpy()
                    _free_arr = np.nonzero(_stk_a < 0)[0]                             # arrived, not yet stuck
                    nn_a = np.asarray(nn_a, np.int64).copy()
                    nn_a[_stk_a >= 0] = _stk_a[_stk_a >= 0]
                    _taken_now = np.zeros(len(_free_arr), np.int64) - 1
                    if len(_free_arr) > 0:
                        _xa = x_int[torch.as_tensor(_arr_idx[_free_arr], device=dev)].cpu().numpy()
                        _dk, _ik = tgt.ot_kd.query(_xa, k=8, workers=-1)
                        for _o in np.argsort(_dk[:, 0]):                              # closest arrivals first
                            for _c in range(_ik.shape[1]):
                                _p = int(_ik[_o, _c])
                                if not _reserved[_p]:
                                    _reserved[_p] = True; _taken_now[_o] = _p; break
                        nn_a[_free_arr] = np.where(_taken_now >= 0, _taken_now, nn_a[_free_arr])
                    arrive_idx_np = np.full(len(x_int), -1, np.int64)
                    arrive_idx_np[_arr_idx[_stk_a >= 0]] = _stk_a[_stk_a >= 0]
                    if len(_free_arr) > 0:
                        _ok = _taken_now >= 0
                        arrive_idx_np[_arr_idx[_free_arr[_ok]]] = _taken_now[_ok]
                else:
                    arrive_idx_np = np.full(len(x_int), -1, np.int64)
                    arrive_idx_np[arrived.detach().cpu().numpy()] = np.asarray(nn_a, np.int64)
                if getattr(cfg, "arrive_cap", False):
                    # config.arrive_cap (method.md 10.28): the snap respects the target's CAPACITY. Without it every
                    # arrived particle near a thin feature snaps to the same few target points and the cell sum packs
                    # them in (the dragon's spikes at 300k: 50-270 particles below det F 0.3; the last arrivals wedged
                    # against a pinned neighbour). Each target point takes at most cap = N / |target points| arrivals
                    # (the mass ratio, no constant), the closest first; the surplus keeps its plan image.
                    _idx_arr = torch.nonzero(arrived).squeeze(1).cpu().numpy()
                    _d_a = np.linalg.norm(x_int[arrived].cpu().numpy() - tgt.points[torch.as_tensor(nn_a, device=dev)].cpu().numpy(), axis=1)
                    _cap = max(1, int(round(len(x_int) / max(1, len(tgt.points)))))
                    _ord = np.lexsort((_d_a, nn_a))                       # by target point, then by distance
                    _nn_s = np.asarray(nn_a)[_ord]
                    _first = np.r_[True, _nn_s[1:] != _nn_s[:-1]]
                    _grp_start = np.maximum.accumulate(np.where(_first, np.arange(len(_nn_s)), 0))
                    _rank = np.arange(len(_nn_s)) - _grp_start
                    _keep = np.zeros(len(_nn_s), bool); _keep[_ord] = _rank < _cap
                    arrive_cap_frac = float(1.0 - _keep.mean())
                    _sel = torch.as_tensor(_idx_arr[_keep], device=dev)
                    x_int[_sel] = tgt.points[torch.as_tensor(np.asarray(nn_a)[_keep], device=dev)].to(x_int.dtype)
                else:
                    x_int[arrived] = tgt.points[torch.as_tensor(nn_a, device=dev)].to(x_int.dtype)
            if getattr(cfg, "render_paced", False) and tgt.sils is not None and not getattr(tgt, "render_paced_off", False):
                # config.render_paced (2026-09-26 23:40, method.md 10.31): the render channel's target is the PACED target's
                # own images. With the target's final silhouettes as the reference, the silhouette term pulls the first
                # material up the ear's outline (a thin lead satisfies it) — at 300k the early knob's material is there
                # only with the render channel on (ar300 21-67 particles in the top region at t = 0.2-0.3, bb300 none) —
                # while the physics channel is driven to the paced cloud one step ahead. Here the silhouettes (and the
                # shading, on the loss grid's normals for both sides) are those of the paced cloud x_int, re-rendered each
                # window without gradient: the two channels agree on the growth order, and the render's fit arrives as
                # the paced cloud converges to the target (the last windows are unchanged, x_int = target). No constant.
                from .render_loss import target_silhouettes as _tsil   # the (theta, phi) views variant the runner uses
                with torch.no_grad():
                    sils_eff = _tsil(x_int.detach(), tgt.views, cfg.render_res, tgt.extent, cfg.sil_k)
                    if cfg.w_pbr > 0 and tgt.shade is not None:
                        from .render_loss import shade_targets as _shd
                        shade_eff = _shd(x_int.detach(), tgt.views, cfg.render_res, tgt.extent,
                                         tgt.lgmin, tgt.ldx, tgt.ldims, cfg.sil_k, cfg.pbr_ambient)
                        pbr_grid_eff = False
                    if getattr(cfg, "render_paced_conv", False):
                        # config.render_paced_conv (2026-09-27 03:30): the switch to the target's own images is read in the
                        # render's own metric — when the paced cloud's silhouettes are closer to the target's than the morph
                        # is to the paced cloud's, the paced target is no longer what limits the fit (the pin's onset came
                        # too early for the ear on bj300 / bl300: the neck-bound overshoot returned; the full paced target
                        # cost -0.004 on bf300). Permanent once met. No constant.
                        _d_pt = float(d_render(x_int.detach(), tgt.sils, tgt.views, cfg.render_res, tgt.extent,
                                               cfg.sil_k, cfg.w_hole, cfg.w_spray))
                        _d_m = float(d_render(x0_ot.detach(), sils_eff, tgt.views, cfg.render_res, tgt.extent,
                                              cfg.sil_k, cfg.w_hole, cfg.w_spray))
                        print(f"[win] paced render target: paced-vs-target {_d_pt:.3e}, morph-vs-paced {_d_m:.3e}", flush=True)
                        if _d_pt < _d_m:
                            tgt.render_paced_off = True
                            sils_eff, shade_eff, pbr_grid_eff = tgt.sils, tgt.shade, True
                            print("[win] paced render target: converged in the render's metric — the target's own images from here", flush=True)
            if getattr(cfg, "w_corr", 0.0) > 0 and getattr(tgt, "ot_knn", None) is not None:
                # config.w_corr (2026-09-25 22:15 CDT, method.md 10.35): correspondence at the plan's own resolution. The
                # paced cell sum is a density comparison: a filled region translating has equal cell sums inside, so
                # only its density-jump layer is pulled, the body behind lags, and the front's skin strips off as the
                # vapour (strip_probe: the head top's own material makes no progress until t = 0.14 while the ear's
                # lead passes). A per-particle pull to the plan images tears the bulk and kills the detail (the leash,
                # falsified three times). Here each material NEIGHBOURHOOD (the plan's k-NN set, one blur radius at the
                # source) is pulled as a whole: its centroid at the window's end toward the centroid of its paced
                # images; inside the neighbourhood the arrangement stays the density and render terms'.
                corr_chat = x_int[tgt.ot_knn].mean(dim=1).detach()
            _dump = os.environ.get("PHYSMORPH_DUMP_PLAN", "")
            if _dump and not os.path.exists(_dump):
                # diagnostic only (2026-09-25 23:30 CDT): the first window's plan — the source, the denoised full
                # displacement and the paced images — for the plan-vs-delivery probe (scratch/plan_probe.py)
                np.savez(_dump, x0=np.ascontiguousarray(np.asarray(x0, np.float32)),
                         x_int=x_int.detach().cpu().numpy().astype(np.float32),
                         disp=disp.detach().cpu().numpy().astype(np.float32), pace_r=float(pace_r), leash_r=float(leash_r))
            pace_grid = rasterize_mass(x_int, tgt.m, tgt.lgmin, tgt.ldx, tgt.ldims).detach()
            if getattr(cfg, "pace_cap", False):
                # config.pace_cap (2026-09-26, with the fronts of 10.29): every image lies inside the target, so the paced
                # grid may never ask a cell for more than the target holds there — the images queued at the front no
                # longer over-fill the front cell (g41fp dragon: 44 particles below det F 0.5 at the spikes' fronts)
                if getattr(cfg, "pace_front_thin", False) and getattr(tgt, "front_bulk_node", None) is not None:
                    _bnode = tgt.front_bulk_node.reshape(pace_grid.shape)
                    pace_grid = torch.where(_bnode, pace_grid, torch.minimum(pace_grid, tgt.grid))   # the cap at the thin nodes only
                else:
                    pace_grid = torch.minimum(pace_grid, tgt.grid)
            frac_arrived = float(arrived.float().mean())
            arrived_mask_np = arrived.detach().cpu().numpy().astype(bool)   # per-particle arrival (config.ctrl_rprop_arrived)
            pace_r_np = float(arr_r)                  # the arrival radius (config.pace_lead: not the lead)
            plan_img_np = (x0_ot + disp).detach().cpu().numpy().astype(np.float32)
            frac_sup = float("nan")
            # HAND-OFF to the fixed target: when every target cell that still lacks mass is
            # within one cell of an occupied cell, the cell sum's gradient (CIC reach: one
            # cell) already touches every deficit from the body — no far cell can reward a
            # lone particle — and the fixed target fills the thin features at full strength
            # (the paced target left them sparse at 150k). Re-evaluated every window.
            # (FALSIFIED 2026-09-17 oh40_bunny: after the switch the gate rejected 14
            # candidates and the run stopped at 136 windows with chamfer 0.1279 — changing
            # the inner objective mid-run reads as a merit regression. Kept opt-in.)
            if getattr(cfg, "ot_handoff", False):
                # CELL-WISE hand-off (v2, 2026-09-17 evening): every target cell within one cell
                # of the occupied set carries the FIXED target mass — the CIC gradient reaches
                # it from the body, so it is ordinary fill, never a far-cell reward — and every
                # cell beyond that carries the paced mass (coherent transport toward it). The
                # global switch (v1: fixed target only once NO deficit cell is far) never
                # fired on C, whose arm tips stay beyond reach (19 cells) while the arms
                # under-fill (hole 8 %).
                m_t = torch.as_tensor(tgt.m, device=dev) if not torch.is_tensor(tgt.m) else tgt.m
                occ = (rasterize_mass(x0_ot, m_t, tgt.lgmin, tgt.ldx, tgt.ldims) > 0).float()
                nx_, ny_, nz_ = tgt.ldims
                occ3 = occ.reshape(1, 1, nx_, ny_, nz_)
                near = torch.nn.functional.max_pool3d(occ3, 3, stride=1, padding=1).reshape(-1) > 0
                far_deficit = int(((grid_eff > 0) & ~near).sum())
                # only DEFICIT cells hand off (target mass above the current occupancy: fill
                # from the fixed target); EXCESS cells keep the paced mass — on C the cells
                # around the body are the hole (m_t = 0) and the fixed target there is the
                # outward runaway push (cd7: silIoU 0.72 again); the pace evacuates them
                cur_grid = rasterize_mass(x0_ot, m_t, tgt.lgmin, tgt.ldx, tgt.ldims)
                near_def = near & (grid_eff > cur_grid)
                pace_grid = torch.where(near_def, grid_eff, pace_grid)
                print(f"[win] OT pace: target cells with mass beyond one cell of the body: {far_deficit} "
                      f"(cell-wise hand-off: {int(near_def.sum())} deficit cells on the fixed target)", flush=True)
            print(f"[win] OT pace: {frac_arrived * 100:.1f}% of particles within one blur radius "
                  f"of their image, mean |d|={float(dn.mean()):.3g} wu, max |d|={float(dn.max()):.3g} wu", flush=True)
            if cfg.layer_ctrl and cfg.layer_gate_ot and layer is not None:
                # the transport gate (docs/surface_gradient.md 15c): u may act only on layer particles
                # whose remaining transport |x - T(x)| (the plan's image, material-kNN averaged like the
                # pace) is at most layer_gate_ot_cells MPM cells — the sub-grid residual is u's regime;
                # while a particle is still in transit, u's per-particle step rides a moving surface and
                # leaves the mid-morph lumps (15b). Recomputed at every window start with the plan.
                res_ot = dn.squeeze(1)
                if getattr(cfg, "layer_gate_ot_normal", False):
                    # the NORMAL part of the remaining transport (15e): u acts along the layer normal,
                    # so tangential transport (material sliding along the outline, nefertiti's arriving
                    # front) does not disqualify it — only transport through space does (the lumps)
                    res_ot = (disp * torch.as_tensor(np.asarray(lnrm, np.float32), device=disp.device)).sum(1).abs()
                ug_ot = (res_ot <= float(cfg.layer_gate_ot_cells) * float(prm.dx)).float().cpu().numpy()
                ug_ot = (ug_ot * (np.asarray(lmask, np.float32) > 0.5)).astype(np.float32)
                lay = spec.layer if len(spec.layer) >= 7 else spec.layer + (None, 0.0)
                base = lay[7] if len(lay) > 7 and lay[7] is not None else None
                ug_new = np.ascontiguousarray(ug_ot if base is None else ug_ot * np.asarray(base, np.float32), np.float32)
                spec.layer = lay[:7] + (ug_new,)
                tr_eval.layer_ug.assign(ug_new)
                _lm = np.asarray(lmask) > 0.5
                u_gate_frac = float(ug_new[_lm].mean()) if bool(_lm.any()) else 0.0
                print(f"[layer] u transport gate: {u_gate_frac * 100:.1f} % of the layer within "
                      f"{cfg.layer_gate_ot_cells:g} cell(s) of its OT image", flush=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _sp = getattr(tgt.ot_pull, "_self_pull", None)
        print(f"[win] OT plan solved: {tgt.ot_pull.last_sweeps} sweeps err={tgt.ot_pull.last_err:.3g}"
              + (f", self {_sp.last_sweeps} sweeps err={_sp.last_err:.3g}" if _sp is not None else "")
              + f", {time.perf_counter() - _t_ot:.2f}s", flush=True)

        def ot_loss(xT):
            if cfg.phys_loss == "ot_leash":
                # transport-plan LEASH (Huber hinge): zero within the plan's blur radius r of
                # the particle's anchor (the cell sum refines freely there), quadratic from
                # r to 2r, linear beyond (a constant pull — no stray is pulled harder than
                # the density loss pulls its most-pulled particle, see the calibration).
                d = (xT - ot_T).norm(dim=1)
                e = torch.clamp(d - leash_r, min=0.0)
                quad = e.pow(2)
                lin = leash_r ** 2 + 2.0 * leash_r * (e - leash_r)
                return torch.where(e <= leash_r, quad, lin).mean()
            return (xT - ot_T).pow(2).sum(1).mean()
        if tgt.ot_scale is None and cfg.phys_loss in ("ot_pace", "ot_shape"):
            tgt.ot_scale = 1.0                                # no extra term to calibrate
        if tgt.ot_scale is None:
            xg = torch.as_tensor(x0, device=dev).clone().requires_grad_(True)
            g_vol = torch.autograd.grad(dvol_density(xg), xg)[0]
            gv = g_vol.norm()
            if cfg.phys_loss == "ot_leash":
                # per-particle parity: a particle at 2r (and beyond) feels the pull the
                # density loss exerts on its most-pulled particle at the source
                n_p = float(xg.shape[0])
                g_max = float(g_vol.norm(dim=1).max())
                tgt.ot_scale = g_max * n_p / (2.0 * leash_r)
                print(f"[win] OT leash calibration: max|g_vol_i|={g_max:.3g} r={leash_r:.4g} wu "
                      f"scale={tgt.ot_scale:.3g}", flush=True)
            else:
                xg2 = torch.as_tensor(x0, device=dev).clone().requires_grad_(True)
                go = torch.autograd.grad(ot_loss(xg2), xg2)[0].norm()
                tgt.ot_scale = float(gv / go.clamp_min(1e-30))
                print(f"[win] OT calibration: |g_vol|={float(gv):.3g} |g_ot|={float(go):.3g} scale={tgt.ot_scale:.3g}", flush=True)

        if cfg.phys_loss == "ot_leash":
            def dvol(xT):
                return dvol_density(xT) + tgt.ot_scale * ot_loss(xT)
        elif cfg.phys_loss in ("ot_pace", "ot_shape"):
            def dvol(xT):
                return d_vol_density(xT, tgt.m, pace_grid, tgt.lgmin, tgt.ldx, tgt.ldims,
                                     tgt.m_ref, tgt.n_support, form=getattr(cfg, "dvol_form", "log"))
        else:
            def dvol(xT):
                return tgt.ot_scale * ot_loss(xT)
    else:
        dvol = dvol_density
    s = None
    if cfg.opt_material:
        s0 = np.zeros((2, N), np.float32) if s_init is None else np.asarray(s_init, np.float32)
        s = torch.tensor(s0, device=dev, requires_grad=True)
        leaves.append(s)
    u = None
    if cfg.layer_ctrl:
        # position-mode control leaf (§7): a normal displacement per outer-layer particle for THIS
        # window (applied 1/T per step, consumed by the window, never warm-started); bounded by one
        # spacing per window — beyond that it is transport, the stress channel's job
        u = torch.zeros(N, device=dev, requires_grad=True)
        leaves.append(u)
    # the u bound per particle: one spacing, times the Rprop scale the runner carries (config.u_rprop)
    u_bound = None
    if u is not None:
        _us = np.ones(N, np.float32) if u_scale_init is None else np.asarray(u_scale_init, np.float32)
        u_bound = torch.as_tensor(float(sp0) * _us, device=dev)
    mom = [torch.zeros_like(p) for p in leaves]
    vel = [torch.zeros_like(p) for p in leaves]
    lr_scale = [1.0] + ([cfg.mat_lr_scale] if s is not None else []) + ([1.0] if u is not None else [])
    # per-particle Rprop scale of the control step (config.ctrl_rprop; docs/method.md 10.24): the runner
    # halves a particle's scale when its window displacement reversed the previous accepted one and
    # raises it x1.2 (to 1) when it kept its direction; no floor — the arrived body's step decays to
    # zero while a part still in transport keeps its step. Broadcast along the leaf's particle axis
    # (per-particle leaves only: a coarse node basis has no particle axis).
    ctrl_scale_v = None
    if ctrl_scale_init is not None and int(getattr(cfg, "control_grid", 0) or 0) == 0:
        _cs = torch.as_tensor(np.asarray(ctrl_scale_init, np.float32), device=dev)
        _ax = [i for i, sz in enumerate(dFc.shape) if sz == N]
        if _ax:
            ctrl_scale_v = _cs.view([N if i == _ax[0] else 1 for i in range(dFc.dim())])
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
    if cfg.w_creg > 0 or cfg.control_h1_iters > 0 or cfg.grad_h1:   # (creg penalty below is gated on w_creg)
        from scipy.spatial import cKDTree
        knn = cKDTree(x0).query(x0, k=cfg.creg_k + 1)[1][:, 1:]
        knn_t = torch.as_tensor(np.ascontiguousarray(knn), device=dev)
    surface_w_t = (torch.as_tensor(surface_w, device=dev).view(N, 1)
                   if surface_w is not None and cfg.surface_mask_objective else None)
    # material-coherence prior (w_coh): frozen SOURCE kNN, the displacement reference is
    # this window's start x0 (so the prior sees the window's increment, not the morph)
    coh_t = None
    sp_ref = float(tgt.nn_spacing) if tgt.nn_spacing > 0 else 0.0
    if sp_ref <= 0 and (cfg.w_coh > 0 or cfg.w_bond > 0):
        # nn spacing is only built for the nn/kde terms; derive it from the source here
        from scipy.spatial import cKDTree
        sp_ref = float(np.median(cKDTree(x0).query(x0, k=2, workers=-1)[0][:, 1]))
    coh_sp2 = max(sp_ref, 1e-6) ** 2
    if (cfg.w_coh > 0 or cfg.w_bond > 0 or cfg.w_esc > 0 or cfg.continuity or cfg.bonds) and coh_nbr is not None:
        coh_t = torch.as_tensor(np.ascontiguousarray(coh_nbr), device=dev)
    bond_t = None
    if cfg.w_bond > 0 and coh_nbr is not None:
        # frozen source neighbours; weights from SOURCE distances (Luiten et al.), rest
        # lengths from THIS window's start positions (one-sided: only new separation)
        src_np = np.ascontiguousarray(coh_nbr_src, np.float32) if coh_nbr_src is not None else x0
        d_src = np.linalg.norm(src_np[coh_nbr] - src_np[:, None, :], axis=2)
        w_b = np.exp(-d_src ** 2 / (2.0 * (2.0 * sp_ref) ** 2)).astype(np.float32)
        x0_b = torch.as_tensor(x0, device=dev)
        bond_t = (torch.as_tensor(w_b, device=dev),
                  (x0_b[coh_t] - x0_b[:, None, :]).norm(dim=2).detach() * (1.0 + float(cfg.bond_s0)))
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
            # k = 8 reference particles' mass (config.disc_ref: x N / mass_ref_n — a clump of a few native
            # particles alone in that neighbourhood is isolated at every N)
            m_dt = tgt.m * isolation_gate(x0_t, cfg.dt_iso_lo, cfg.dt_iso_hi,
                                          k=int(round(8 * disc_ref_factor(len(x0), cfg) ** 3)))
    # the gate's support (2026-09-23, speed): the DT sum runs on these particles only — the same
    # sum and gradient, a small fraction of N (the isolated particles)
    dt_idx = torch.nonzero(m_dt > 0).squeeze(1) if m_dt is not None else None

    # grid-free near-band cleanup, frozen per window (fork-halo forensic §7.10)
    if cfg.w_jdens > 0 and tgt.jd_rho0 is not None and tgt.jd_scale is None:
        # one-shot equal-norm vs D_vol - DEFERRED to the first window where the prior
        # has a gradient (v5: calibrating at the source, where J == 1 and the sKL
        # gradient vanishes, produced an astronomical scale and froze the run at a9)
        xg = x0_t.detach().clone().requires_grad_(True)
        gv = torch.autograd.grad(dvol(xg), xg)[0].norm()
        gj = torch.autograd.grad(d_jdens(xg, tgt.m, tgt.jd_rho0, tgt.jd_gmin, tgt.jd_dx,
                                         tgt.jd_dims), xg)[0].norm()
        if float(gj) > 1e-4 * float(gv):
            tgt.jd_scale = float(min(gv / gj, 1e3))
            log(f"[win] jdens calibration: |g_vol|={float(gv):.3g} |g_jd|={float(gj):.3g} "
                f"scale={tgt.jd_scale:.3g}")
        else:
            log(f"[win] jdens calibration deferred (|g_jd|={float(gj):.3g} vs |g_vol|={float(gv):.3g})")
    h1_ratio = None
    if cfg.w_h1 > 0:
        # one-shot equal-norm vs D_vol at the FIRST window, in position space (the
        # control-space norms after the MPM Jacobian differ - documented caveat).
        # Legitimate here, unlike the density prior (v5/v5b): the H^-1 residual shares
        # D_vol's minimiser and its gradient at the source is non-zero. Every window
        # also records the CURRENT ratio s*|g_h1|/|g_vol| (REFUTE Opus F2: the local
        # equal-norm scale drifted x5 over a synthetic morph, so parity holds only at
        # the source - the drift is measured per window, not assumed away).
        xg = x0_t.detach().clone().requires_grad_(True)
        gv = torch.autograd.grad(dvol(xg), xg)[0].norm()
        gh = torch.autograd.grad(d_h1(xg, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx,
                                      tgt.ldims), xg)[0].norm()
        if tgt.h1_scale is None:
            tgt.h1_scale = float(min(gv / gh.clamp_min(1e-30), 1e3))   # v5 cap
            log(f"[win] h1 calibration: |g_vol|={float(gv):.3g} |g_h1|={float(gh):.3g} "
                f"scale={tgt.h1_scale:.3g}")
        h1_ratio = float(tgt.h1_scale * gh / gv.clamp_min(1e-30))
    def corr_loss(xc):
        return ((xc[tgt.ot_knn].mean(dim=1) - corr_chat) ** 2).sum(dim=1).mean()
    if getattr(cfg, "w_corr", 0.0) > 0 and corr_chat is not None and tgt.corr_scale is None:
        # one-shot equal-norm calibration vs D_vol at the first window (the H^-1 precedent): at the source the
        # neighbourhood centroids are one pace from their images, so the gradient is non-zero and shares D_vol's
        # direction of descent
        xg = x0_t.detach().clone().requires_grad_(True)
        gv = torch.autograd.grad(dvol(xg), xg)[0].norm()
        gc = torch.autograd.grad(corr_loss(xg), xg)[0].norm()
        tgt.corr_scale = float(min(gv / gc.clamp_min(1e-30), 1e3))
        # print, not log: the runner hands optimize_window log=lambda *_: None, so the h1/jdens calibration lines
        # never reach the run log (found 2026-09-25 23:10 CDT while checking whether bp304 was active)
        print(f"[win] corr calibration: |g_vol|={float(gv):.3g} |g_corr|={float(gc):.3g} scale={tgt.corr_scale:.3g}",
              flush=True)
    kde_nbr = None
    if cfg.w_kde > 0 and tgt.pts is not None:
        kde_nbr = kde_assign(x0_t, tgt.pts, cfg.kde_k)
        if tgt.kde_scale is None:            # one-shot equal-norm calibration vs D_vol
            xg = x0_t.detach().clone().requires_grad_(True)
            gv = torch.autograd.grad(dvol(xg), xg)[0].norm()
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

    def losses_of(xT, FT, vT, FgT=None):
        """FgT: geometric deformation (render kinematics). With cfg.render_F_geom the
        Gaussian covariance rides it instead of the controlled, smoothed FT."""
        lv = dvol(xT)
        lk = vT.pow(2).sum(1).mean()
        lr = lpbr = None
        FR = FgT if (use_geom and FgT is not None) else FT
        sil_gauss["sil"] = sil_gauss["gauss"] = None
        if balancer.active and tgt.gauss is not None and not cfg.gauss_in_objective:
            # gauss for dressing/telemetry only: log d_gauss no-grad, then fall
            # through to the plain silhouette channel below
            with torch.no_grad():
                sil_gauss["gauss"] = float(tgt.gauss.loss(
                    xT.detach(), FR.detach() if cfg.gauss_covariance else None,
                    mask=gauss_mask_t))
        if balancer.active and tgt.gauss is not None and cfg.gauss_in_objective:
            gauss_mask = gauss_mask_t
            lg_ = tgt.gauss.loss(xT, FR if cfg.gauss_covariance else None,
                                 mask=gauss_mask)
            sil_gauss["gauss"] = float(lg_.detach())
            if cfg.gauss_mix > 0:                 # hybrid: silhouette keeps fine geometry
                lsil = d_render(xT, sils_eff, tgt.views, cfg.render_res, tgt.extent,
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
            lsil = d_render(xT, sils_eff, tgt.views, cfg.render_res, tgt.extent,
                            cfg.sil_k, cfg.w_hole, cfg.w_spray)
            sil_gauss["sil"] = float(lsil.detach())
            lr = lsil
            if cfg.w_pbr > 0 and tgt.shade is not None:     # shading channel (PBR-lite)
                if cfg.pbr_denoised and tgt.pdims and pbr_grid_eff:
                    # G1: the morph's normals on the render-pixel grid, blurred by the renderer's
                    # 1.5 spacings — the normals of the drawn surface, against a denoised target
                    lpbr = d_pbr(xT, shade_eff, tgt.views, cfg.render_res, tgt.extent,
                                 tgt.pgmin, tgt.pdx, tgt.pdims, cfg.sil_k, cfg.pbr_ambient, tgt.pblur)
                else:
                    lpbr = d_pbr(xT, shade_eff, tgt.views, cfg.render_res, tgt.extent,
                                 tgt.lgmin, tgt.ldx, tgt.ldims, cfg.sil_k, cfg.pbr_ambient)
                lr = lsil + cfg.w_pbr * lpbr
        return lv, lk, lr, lpbr

    gx_box = [None]                  # config.settle_pin_kkt: dL/dx_T of the last gradient evaluation (every particle)

    def terms(leaf):
        """Differentiable path: torch Function + wp.Tape (gradient phase only).
        Returns (state=(xT,FT,vT), lv, lk, lr, lpbr, extra) with extra = {"dfc": the
        expanded control field, "Fg": F_g^T, "V": all velocities, "lk_run": running
        kinetic}."""
        lam_t, mu_t = material()
        dfc = expand(leaf)
        if lam_t is None and str(dev).startswith("cuda") and not _NO_ADJ_GRAPH:
            # persistent tape trajectory (forward + adjoint as CUDA graphs), one per window
            if adj_box[0] is None:
                adj_box[0] = PersistentAdjoint(spec)
            xT, FT, vT, FgT, V = adj_box[0].apply(dfc, u)
        else:
            xT, FT, vT, FgT, V = warp_mpm_ext(dfc, spec, lam_t, mu_t, u_t=u)
        if xT.requires_grad and getattr(cfg, "settle_pin_kkt", False):
            # the objective's gradient at the end-of-window positions of EVERY particle, pinned ones included (a pinned
            # particle's control has no effect, but its position still carries the render and density residuals)
            xT.register_hook(lambda g: gx_box.__setitem__(0, g.detach()))
        lv, lk, lr, lpbr = losses_of(xT, FT, vT, FgT)
        extra = {"dfc": dfc, "Fg": FgT, "V": V, "lk_run": V.pow(2).sum(2).mean(),
                 "lk_var": (V.pow(2).sum(2).mean(0) - V.mean(0).pow(2).sum(1)).mean()}
        return (xT, FT, vT), lv, lk, lr, lpbr, extra

    def eval_terms(leaf):
        """No-grad path for line-search candidates: plain rollout, NO tape, NO adjoint
        buffers — the graph path allocates ~2x memory and tape bookkeeping that a
        candidate evaluation (up to max_ls_iters per iteration) never uses."""
        with torch.no_grad():
            lam_t, mu_t = material()
            _set_material(lam_t, mu_t)
            dc = expand(leaf.detach()).detach().contiguous()
            dc_buf.copy_(dc.view(T, N, 3, 3))
            if u is not None:
                wp.to_torch(tr_eval.layer_u).copy_(u.detach())
            t0 = _tick()
            tr = tr_eval
            tr.run()
            # the buffers are rewritten by the next candidate: the outputs are copies
            xT = wp.to_torch(tr.x[T]).clone()
            FT = wp.to_torch(tr.F[T]).reshape(N, 9).clone()
            vT = wp.to_torch(tr.v[T]).clone()
            FgT = wp.to_torch(tr.Fg[T]).reshape(N, 9).clone() if use_geom else None
            V = torch.stack([wp.to_torch(tr.v[t]) for t in range(1, T + 1)])
            _tm_add("eval_roll", t0)
            t0 = _tick()
            lv, lk, lr, lpbr = losses_of(xT, FT, vT, FgT)
            extra = {"dfc": dc, "Fg": FgT, "V": V, "lk_run": V.pow(2).sum(2).mean(),
                     "lk_var": (V.pow(2).sum(2).mean(0) - V.mean(0).pow(2).sum(1)).mean()}
            _tm_add("eval_loss", t0)
            t0 = _tick()
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
            _tm_add("eval_det", t0)
        return (xT, FT, vT, jt), lv, lk, lr, lpbr, extra

    def _vT(extra):
        v = extra.get("V") if isinstance(extra, dict) else None
        return v[-1] if v is not None else None

    # DISCRETE CONTINUITY (cfg.continuity): per-particle limit sp_i / (T dt) on the window-end
    # velocity relative to the frozen material neighbours; sp_i = mean distance to those
    # neighbours at the window start. A launched particle exceeds it by 5-20x (40k archives:
    # ejecta 5-6 wu/s vs a limit of 1.3 wu/s); coherent motion, including thin-feature
    # stretching, stays far below it (body p95 0.28 wu/s).
    cont_lim = None
    if cfg.continuity and coh_t is not None:
        x0_c = torch.as_tensor(np.ascontiguousarray(x0, np.float32), device=dev)
        cont_lim = ((x0_c[coh_t] - x0_c[:, None, :]).norm(dim=2).mean(1)
                    / float(max(cfg.T * prm.dt, 1e-9))).detach()
    cont_state = {"ref": None, "viol": 0, "ratio": 0.0, "rejects": 0, "ref_ratio": 0.0}

    def cont_rel(V):
        """max over the window's steps of |v_i - mean_j v_j| (V: (T, N, 3))."""
        with torch.no_grad():
            rel = (V - V[:, coh_t].mean(2)).norm(dim=2)          # (T, N)
            return rel.max(0).values

    def cont_check(extra_cand):
        """True if the candidate violates continuity nowhere the FREE rollout did not."""
        if cont_lim is None:
            return True
        V = extra_cand.get("V") if isinstance(extra_cand, dict) else None
        if V is None:
            return True
        rel = cont_rel(V)
        ref = cont_state["ref"] if cont_state["ref"] is not None else torch.zeros_like(rel)
        allowed = torch.maximum(cont_lim, ref)
        viol = int((rel > allowed).sum())
        cont_state["viol"] = viol
        cont_state["ratio"] = float((rel / cont_lim).max())
        cont_state["excess"] = float((rel / allowed.clamp_min(1e-12)).max())   # worst rel/allowed
        if viol:
            cont_state["rejects"] += 1
        return viol == 0

    def phys_core(lv, lk, dfc, xT, fT=None, lk_run=None, fR=None, lk_var=None, vT=None):
        """Physics objective WITHOUT the W1 term — the lambda balancer's numerator and
        PCGrad's reference direction (Codex finding 9: folding the W1 term into gp let
        w_dt inflate the balanced silhouette weight and project render components off a
        direction that was never 'the physics')."""
        L = lv + wu * cfg.w_kin * lk + wu * cfg.w_ctrl * dfc.pow(2).sum() / (T * N)
        if cfg.w_kin_running > 0 and lk_run is not None:
            # RUNNING kinetic (docs/oscillation_triage.md driver C): penalise motion at
            # every step, not only the endpoint, so a window cannot sprint-then-brake
            L = L + wu * cfg.w_kin_running * lk_run
        if cfg.w_kin_var > 0 and lk_var is not None:
            # velocity VARIANCE over the window: the measured limit cycle (speed V-shape,
            # period T) is a reversal inside the window; constant-velocity progress is free
            L = L + wu * cfg.w_kin_var * lk_var
        if coh_t is not None and cfg.w_esc > 0 and vT is not None:
            # escape-velocity hinge: a particle whose window-end velocity differs from its
            # source neighbours' mean by more than esc_k x the median speed is being
            # launched; quadratic above the threshold, zero for coherent motion
            speed = vT.norm(dim=1)
            thr = (float(cfg.esc_k) * speed.median()).detach().clamp(min=1e-6)
            rel = (vT - vT[coh_t].mean(1)).norm(dim=1)
            L = L + wu * cfg.w_esc * torch.relu(rel - thr).pow(2).mean() / (thr * thr)
        if coh_t is not None and cfg.w_coh > 0:
            # material coherence: a particle may not leave its source neighbours' motion
            u = xT - x0_t
            L = L + wu * cfg.w_coh * (u - u[coh_t].mean(1)).pow(2).sum(1).mean() / coh_sp2
        if bond_t is not None:
            # one-sided bond stretch beyond (1+s0) x this window's start length
            w_b, lmax = bond_t
            d = (xT[coh_t] - xT[:, None, :]).norm(dim=2)
            L = L + wu * cfg.w_bond * (w_b * torch.relu(d - lmax).pow(2)).sum(1).mean() / coh_sp2
        if cfg.w_tctrl > 0 and T > 1:
            L = L + wu * cfg.w_tctrl * (dfc[1:] - dfc[:-1]).pow(2).mean()
        if cfg.w_box > 0:      # far-field leash: differentiable everywhere, zero inside box
            L = L + wu * cfg.w_box * torch.clamp(xT.abs() - tgt.extent, min=0).pow(2).sum(1).mean()
        if knn_t is not None and cfg.w_creg > 0:  # control smoothness: a lone particle
            # cannot be actuated (REFUTE F9: the (T,N,k,9) gather was built and multiplied
            # by zero when only control_h1 needed the neighbour list)
            L = L + wu * cfg.w_creg * (dfc - dfc[:, knn_t].mean(2)).pow(2).mean()
        if cfg.w_h1 > 0 and tgt.h1_scale is not None and not getattr(cfg, "h1_outside", False):
            # non-local mass balance (REVISION 3 amendment): H^-1 norm of the density
            # residual, self-energy corrected. INSIDE phys_core: it is a data term on
            # the same residual, so it belongs in the physics direction the lambda
            # balancer scales the render channel against and PCGrad projects it off
            # (REFUTE Opus 2026-09-04 F3: outside the core it was the one large
            # unprojected, unbalanced gradient, cos(g_vol, g_h1) ~ 0 to -0.1 along a
            # morph). The attribution concern (Codex F3: it also inflates lambda_render)
            # is answered by the v7c isolation arm and the paired-at-equal-d_vol census,
            # not by weakening the control.
            L = L + cfg.w_h1 * tgt.h1_scale * d_h1(xT, tgt.m, tgt.grid, tgt.lgmin,
                                                   tgt.ldx, tgt.ldims)
        if cfg.w_jdens > 0 and tgt.jd_rho0 is not None and tgt.jd_scale is not None:
            # density-measured volume prior (REVISION 3): J from the particle mass
            # field, not from the lagging stored F (inactive until calibrated)
            L = L + cfg.w_jdens * tgt.jd_scale * d_jdens(xT, tgt.m, tgt.jd_rho0,
                                                         tgt.jd_gmin, tgt.jd_dx, tgt.jd_dims)
        if cfg.w_jvol > 0 and fT is not None:
            # sKL volume prior (J-1)·log J (F5 verdict): counters the permanent
            # volumetric spring + control-injected drift; soft barrier as J->0+
            J = torch.linalg.det(fT.view(-1, 3, 3))
            L = L + wu * cfg.w_jvol * ((J - 1.0) * torch.log(J.clamp_min(1e-6))).mean()
        fV = fR if fR is not None else fT            # the F the viewer renders (F_g
        if cfg.w_cov > 0 and fV is not None:          # under render_F_geom; REFUTE F7)
            # The viewer renders Sigma=sigma0^2 F F^T.  Penalise only excessive
            # anisotropy/scale; rotations remain free and normal elastic motion inside
            # the band is untouched.
            sv = torch.linalg.svdvals(fV.view(-1, 3, 3)).clamp_min(1e-6)
            lo = torch.relu(torch.log(cfg.cov_smin / sv))
            hi = torch.relu(torch.log(sv / cfg.cov_smax))
            L = L + wu * cfg.w_cov * (lo.square() + hi.square()).mean()
        if s is not None:
            L = L + wu * cfg.w_mat * s.pow(2).mean()
        return L

    def dt_term(xT):
        """Fixed-weight one-signed cleanup terms (spray DT-W1 + near-band) — NOT
        lambda-scaled (lambda->cap x constant gradient = documented mass-ejection
        mode) and NOT part of phys_core (finding 9). The fill term is separate:
        it carries its own norm-balanced weight (fill v3)."""
        L = None
        if tgt.dt3 is not None:
            # 2026-09-23 (speed): the DT term is a SUM of m_p * DT(x_p) and the gate m_p is zero on
            # all but the isolated particles; the gather and its backward ran over every particle
            # (1.0 s of a 6 s window at 300k). Evaluating it on the gate's support is the same
            # sum and the same gradient (a zero-weight particle contributes nothing to either).
            if dt_idx is not None and dt_idx.numel() > 0:
                L = wu * cfg.w_dt * d_w1(xT.index_select(0, dt_idx), m_dt.index_select(0, dt_idx),
                                         tgt.dt3, tgt.dtgmin, tgt.dtdx, tgt.dtdims)
            elif dt_idx is not None:
                L = xT.sum() * 0.0                          # an empty gate: zero, still on the graph
            else:
                L = wu * cfg.w_dt * d_w1(xT, m_dt, tgt.dt3, tgt.dtgmin, tgt.dtdx, tgt.dtdims)
        if nn_idx is not None:
            Ln = wu * cfg.w_nn * d_nn_band(xT, tgt.m, tgt.pts, nn_idx, nn_elig,
                                      cfg.nn_berth_k * tgt.nn_spacing)
            L = Ln if L is None else L + Ln
        if kde_nbr is not None:
            Lk = cfg.w_kde * tgt.kde_scale * d_kde(xT, tgt.pts, kde_nbr,
                                                   tgt.kde_h, tgt.kde_rho_ref)
            L = Lk if L is None else L + Lk
        return L

    fill_on = fill_pairs is not None
    fill_lam = wu * cfg.w_fill if fill_on else 0.0     # v4: FIXED weight — dominance is
    # bounded by the capacity-limited MATCHING (<= fill_cap_frac*N pairs), not by a
    # scalar that dies with the physics gradient (the v3 failure)

    def fill_raw(xT):
        from ..losses.volumetric import d_fill_pairs
        return d_fill_pairs(xT, fill_pairs[0], fill_pairs[1], 0.5 * tgt.dtdx)

    def phys_total(lv, lk, dfc, xT, fT=None, lk_run=None, fR=None, lk_var=None, vT=None):
        L = phys_core(lv, lk, dfc, xT, fT, lk_run, fR, lk_var, vT)
        if cfg.w_h1 > 0 and tgt.h1_scale is not None and getattr(cfg, "h1_outside", False):
            # config.h1_outside (2026-09-26): the H^-1 term OUTSIDE the core, the W1 precedent — inside it, its
            # gradient inflates the physics norm the lambda balancer scales the render channel against (the
            # h1 ratio 0.3-1.4 along a morph), and on bimba the render channel then out-pulled the transport at
            # 88 % arrival (the divergence +7 % a window, the brake stopped the run at 15). Outside, the balancer
            # and PCGrad see the cell sum alone; the H^-1 pull is added unscaled after them.
            L = L + cfg.w_h1 * tgt.h1_scale * d_h1(xT, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims)
        if getattr(cfg, "w_corr", 0.0) > 0 and corr_chat is not None and tgt.corr_scale is not None:
            L = L + cfg.w_corr * tgt.corr_scale * corr_loss(xT)     # 10.35, outside the core as H^-1
        ldt = dt_term(xT)
        if ldt is not None:
            L = L + ldt
        if fill_on:
            L = L + fill_lam * fill_raw(xT)
        return L

    def scalars(lv, lk, lr, lam_r, dfc, xT, fT=None, lk_run=None, fR=None, lk_var=None, vT=None):
        with torch.no_grad():    # scalar only — never build a second autograd graph
            L = float(phys_total(lv, lk, dfc.detach(), xT.detach(),
                                 fT.detach() if fT is not None else None,
                                 lk_run.detach() if lk_run is not None else None,
                                 fR.detach() if fR is not None else None,
                                 lk_var.detach() if lk_var is not None else None,
                                 vT.detach() if vT is not None else None))
        return L if lr is None else L + lam_r * float(lr.detach())

    hist, accepted, rejected = [], 0, 0
    ls_exhausted = False
    _TM.clear()
    _TM["t_win"] = time.perf_counter()
    pace_bound = False               # window exited via the pace floor (on schedule)
    lk_start = None                  # kinetic term at window start (quasi-static rule)
    g_cos = g_raw_cos = g_share = g_phys_norm = g_rend_norm = None
    lam_capped = None                # REFUTE F12: read at the update, not at window end
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
        st0, lv0, lk0, lr0, _, ex0 = eval_terms(dFc)
        # stack-review f5: an INVALID cold baseline must not be the comparator — its
        # position-only loss can be artificially low and block every valid warm start
        E0 = (scalars(lv0, lk0, lr0, lam_r, ex0["dfc"], st0[0], st0[1], ex0["lk_run"],
                      ex0["Fg"], ex0["lk_var"], _vT(ex0))
              if _state_ok(st0) else np.inf)
        with torch.no_grad():
            init = torch.tensor(np.ascontiguousarray(dfc_init, np.float32), device=dev)
            # the previous window's EXPANDED field is restricted onto THIS window's basis
            # (the node grid follows the window start positions)
            dFc.copy_((init if basis.per_particle else basis.project(init)) * cfg.warm_decay)
        stw, lvw, lkw, lrw, _, exw = eval_terms(dFc)
        Ew = scalars(lvw, lkw, lrw, lam_r, exw["dfc"], stw[0], stw[1], exw["lk_run"],
                     exw["Fg"], exw["lk_var"], _vT(exw))
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
        stA, lvA, lkA, lrA, _, exA = eval_terms(dFc)
        stB, lvB, lkB, lrB, _, exB = eval_terms(dFc)
        EA = scalars(lvA, lkA, lrA, lam_r, exA["dfc"], stA[0], stA[1], exA["lk_run"], exA["Fg"],
                     exA["lk_var"], _vT(exA))
        EB = scalars(lvB, lkB, lrB, lam_r, exB["dfc"], stB[0], stB[1], exB["lk_run"], exB["Fg"],
                     exB["lk_var"], _vT(exB))
        if np.isfinite(EA) and np.isfinite(EB):
            replay_rel = abs(EA - EB) / max(abs(EA), 1.0)

    grad_dump_leaf0 = dFc.detach().clone() if cfg.grad_dump else None   # the window's start control
    grad_dump_state = {}
    _rp_load = os.environ.get("PHYSMORPH_REPLAY_LOAD", "")
    _rp_save = os.environ.get("PHYSMORPH_REPLAY_SAVE", "")
    n_iters = cfg.iters
    if _rp_load:
        # P278b (diagnostic only, 2026-09-26 00:50 CDT): REPLAY a saved window control instead of optimising. The
        # control is the saved run's accepted expanded field (T, Ns, 3, 3); PHYSMORPH_REPLAY_MAP=direct copies it
        # (same cloud), "grid" projects it onto the loss grid and samples it at THIS cloud (same or another N).
        # u is zero unless PHYSMORPH_REPLAY_U=1 (then mapped the same way and applied with this cloud's own
        # layer mask, normals and gate). No iteration runs; the commit rollout below delivers the window.
        _z = np.load(_rp_load)
        _xs = torch.as_tensor(np.asarray(_z["x0"], np.float32), device=dev)
        _dc_src = torch.as_tensor(np.asarray(_z["dfc"], np.float32), device=dev)
        if _dc_src.shape[0] > T:         # a shorter horizon (e.g. --T 1 for the first-step response): the first T slices
            _dc_src = _dc_src[:T].contiguous()
        _mode = os.environ.get("PHYSMORPH_REPLAY_MAP", "grid")
        _with_u = os.environ.get("PHYSMORPH_REPLAY_U", "0") == "1" and "u" in _z.files
        with torch.no_grad():
            if _mode == "direct":
                assert _dc_src.shape[1] == N, "direct replay needs the same cloud"
                _init = _dc_src
                _u_new = torch.as_tensor(np.asarray(_z["u"], np.float32), device=dev) if _with_u else None
            else:
                # the projection grid: this run's loss grid, or PHYSMORPH_REPLAY_GRID cells per axis over the same
                # domain (so a run at another MPM cell size still receives the SAME grid-level control)
                _g = int(os.environ.get("PHYSMORPH_REPLAY_GRID", "0") or 0)
                _dims_m = (_g, _g, _g) if _g > 0 else tuple(tgt.ldims)
                _ldx_m = float(tgt.ldx) * float(tgt.ldims[0]) / float(_dims_m[0])
                _init = _grid_map(_dc_src.reshape(_dc_src.shape[0], -1, 9), _xs, x0_t, tgt.lgmin, _ldx_m,
                                  _dims_m).reshape(_dc_src.shape[0], N, 3, 3)
                _u_new = (_grid_map(torch.as_tensor(np.asarray(_z["u"], np.float32), device=dev).reshape(1, -1, 1),
                                    _xs, x0_t, tgt.lgmin, _ldx_m, _dims_m).reshape(N) if _with_u else None)
                print(f"[replay] projection grid {_dims_m[0]}^3, cell {_ldx_m:.4f} wu", flush=True)
            dFc.copy_(_init if basis.per_particle else basis.project(_init))
            if u is not None:
                if _u_new is not None:
                    u.copy_(_u_new)
                else:
                    u.zero_()
        print(f"[replay] control loaded from {_rp_load} (map {_mode}, u {'on' if _u_new is not None else 'off'}): "
              f"|dFc| mean {float(dFc.detach().abs().mean()):.3e} (source {float(_dc_src.abs().mean()):.3e})", flush=True)
        n_iters = 0
    for it in range(n_iters):
        # ---- gradients. λ_R is fixed for the WHOLE window (estimated from the first
        # iteration's per-term norms), so every accepted step decreases one objective. ----
        t0 = _tick()
        state, lv, lk, lr, lpbr, extra = terms(dFc)
        _tm_add("terms", t0)
        t0 = _tick()
        dfc_x = extra["dfc"]                           # expanded control field (graph)
        render_F = extra["Fg"] if use_geom else state[1]  # what the image differentiates
        if lk_start is None:
            lk_start = float(lk.detach())
        Lp_core = phys_core(lv, lk, dfc_x, state[0], state[1], extra["lk_run"],
                            extra["Fg"] if use_geom else None, extra["lk_var"], _vT(extra))
        Lfill = fill_raw(state[0]) if fill_on else None
        smooth = balancer.active and cfg.render_gs_iters > 0
        special_render = smooth or surface_w_t is not None or cfg.control_h1_iters > 0
        gp = None
        Ldt = dt_term(state[0])
        if Lfill is not None:
            Ldt = (fill_lam * Lfill) if Ldt is None else (Ldt + fill_lam * Lfill)
        gx_phys_diag = gF_phys_diag = gv_phys_diag = None
        gx_rend_diag = gF_rend_diag = None
        if (on_iter is not None or cfg.work_telemetry) and (it == 0 or it == cfg.iters - 1):
            # Endpoint position-space gradients are the interpretable vector fields
            # shown by the viewer.  They are diagnostics only; the control update still
            # uses the full MPM adjoint through x, F and v. Computed on the first and
            # last iteration of the window (2026-09-16: two extra backward passes per
            # iteration at 150k were pure telemetry cost).
            Lp_diag = Lp_core if Ldt is None else Lp_core + Ldt
            gx_phys_diag, gF_phys_diag, gv_phys_diag = torch.autograd.grad(
                Lp_diag, state, retain_graph=True, allow_unused=True)
            if lr is not None:
                gx_rend_diag, gF_rend_diag = torch.autograd.grad(
                    lr, (state[0], render_F), retain_graph=True, allow_unused=True)
                if surface_w_t is not None:
                    gx_rend_diag = gx_rend_diag * surface_w_t
                    if gF_rend_diag is not None:
                        gF_rend_diag = gF_rend_diag * surface_w_t.repeat(1, 9)
        if balancer.active and (special_render or cfg.grad_project or it == 0 or cfg.layer_u_render_only):
            if gp is None:
                t1 = _tick()
                gp = torch.autograd.grad(Lp_core, leaves, retain_graph=True)
                _tm_add("g_phys", t1)
            t1 = _tick()
            gdt = (torch.autograd.grad(Ldt, leaves, retain_graph=True)
                   if Ldt is not None else None)
            _tm_add("g_dt", t1)
            if cfg.grad_dump and it == 0 and lr is not None:
                # gradient-stage dump (docs/surface_gradient.md): the terminal covectors of each
                # channel on the particles and their pull-back to the control leaf, taken here
                # while the graph is still retained. Five extra backward passes, first iteration
                # only; a diagnostic.
                lsil_t = lr - cfg.w_pbr * lpbr if lpbr is not None else lr
                _gx_sil, = torch.autograd.grad(lsil_t, state[0], retain_graph=True, allow_unused=True)
                _gx_pbr = (torch.autograd.grad(lpbr, state[0], retain_graph=True, allow_unused=True)[0]
                           if lpbr is not None else None)
                _gx_phys, = torch.autograd.grad(Lp_core, state[0], retain_graph=True, allow_unused=True)
                _gl_sil = torch.autograd.grad(lsil_t, leaves, retain_graph=True, allow_unused=True)[0]
                _gl_pbr = (torch.autograd.grad(lpbr, leaves, retain_graph=True, allow_unused=True)[0]
                           if lpbr is not None else None)
                _gl_rend = torch.autograd.grad(lr, leaves, retain_graph=True, allow_unused=True)[0]
                grad_dump_state.update(gx_phys=_gx_phys, gx_sil=_gx_sil, gx_pbr=_gx_pbr,
                                       gl_phys=gp[0].detach().clone(), gl_sil=_gl_sil, gl_pbr=_gl_pbr,
                                       gl_rend=_gl_rend, xT0=state[0].detach().clone())
                if u is not None:
                    # the position-mode channel (§7): each term's gradient on the u leaf (u is the
                    # last leaf; gp is over all leaves)
                    def _gu(term):
                        if term is None:
                            return None
                        g_ = torch.autograd.grad(term, u, retain_graph=True, allow_unused=True)[0]
                        return torch.zeros_like(u) if g_ is None else g_.detach().clone()
                    grad_dump_state.update(gu_phys=gp[-1].detach().clone(), gu_sil=_gu(lsil_t),
                                           gu_pbr=_gu(lpbr), gu_rend=_gu(lr))
            t1 = _tick()
            if smooth:
                # v3 grid-GS preconditioning: smooth the IMAGE-SPACE pull on the grid,
                # then pull the smoothed direction back through the SAME MPM adjoint
                # (seeded backward) — physics-exact, docs/method.md §6. BOTH covectors
                # (x AND F: the audit found the F part was dropped here) are smoothed
                # with the same operator; Chebyshev acceleration is optional (§6).
                gx, gF = torch.autograd.grad(lr, (state[0], render_F), retain_graph=True,
                                             allow_unused=True)
                if gF is None:
                    gF = torch.zeros_like(render_F)
                if surface_w_t is not None:
                    gx = gx * surface_w_t
                    gF = gF * surface_w_t.repeat(1, 9)
                rho = chebyshev_rho(cfg.render_gs_kappa) if cfg.render_gs_cheb else 0.0
                both = smooth_particle_field(state[0].detach(), torch.cat([gx, gF], 1),
                                             tgt.lgmin, tgt.ldx, tgt.ldims,
                                             cfg.render_gs_iters, cfg.render_gs_kappa,
                                             cheb_rho=rho, rescale=False)
                gxs, gFs = both[:, :3], both[:, 3:]
                gxs = gxs * (gx.norm() / gxs.norm().clamp_min(1e-30))   # per-channel
                gFs = gFs * (gF.norm() / gFs.norm().clamp_min(1e-30))   # norm preserved
                gr = torch.autograd.grad((state[0], render_F), leaves,
                                         grad_outputs=(gxs, gFs))
            elif surface_w_t is not None:
                gx, gF = torch.autograd.grad(lr, (state[0], render_F), retain_graph=True,
                                             allow_unused=True)
                if gF is None:
                    gF = torch.zeros_like(render_F)
                gr = torch.autograd.grad((state[0], render_F), leaves,
                                         grad_outputs=(gx * surface_w_t,
                                                       gF * surface_w_t.repeat(1, 9)))
            else:
                gr = torch.autograd.grad(lr, leaves)
            _tm_add("g_rend", t1)
            if cfg.control_h1_iters > 0:
                gr = list(gr)
                gr[0] = _control_h1(gr[0], knn_t, cfg.control_h1_iters,
                                    cfg.control_h1_kappa)
            gr_raw = [r.detach().clone() for r in gr]
            np_raw, nr_raw = _norm(gp), _norm(gr_raw)
            dot_raw = float(sum((a * b).sum() for a, b in zip(gp, gr_raw)))
            mode = cfg.grad_project_mode if cfg.grad_project else "off"
            if mode == "render":                      # legacy one-sided PCGrad: the
                gr, _conf = _pcgrad(gp, gr)           # balancer sees the PROJECTED grad
            if it == 0:
                # λ from the PROJECTED render grad — estimating it from the raw one and
                # then projecting silently de-weighted the channel by ~33% at cos=-0.74
                # (adversarial finding), breaking the balancer contract.
                lam_r = balancer.update(_norm(gp), _norm(gr))
                lam_capped = int(bool(getattr(balancer, "capped", False)))
                # render-influence telemetry (standing request): how much does the
                # render channel actually steer the update this window?
                np_, nr_ = _norm(gp), _norm(gr)
                dot_ = float(sum((a * b).sum() for a, b in zip(gp, gr)))
                g_cos = dot_ / max(np_ * nr_, 1e-30)
                g_raw_cos = dot_raw / max(np_raw * nr_raw, 1e-30)
                g_share = lam_r * nr_ / max(np_ + lam_r * nr_, 1e-30)
                g_phys_norm, g_rend_norm = np_, nr_
                if cfg.grad_dump:
                    grad_dump_state.update(lam_r=float(lam_r), g_share=float(g_share))
            if mode in ("off", "render"):
                g = [a + lam_r * b for a, b in zip(gp, gr)]
            else:                    # "phys" / "cagrad" / "blend": grad_combine.combine
                g, _info = combine_grads(mode, gp, gr, lam_r, cfg.cagrad_c,
                                         blend_beta=balancer.alpha_lam)
            if gdt is not None:      # W1 joins the composite AFTER lambda/PCGrad (find. 9)
                g = [gi + di for gi, di in zip(g, gdt)]
            if cfg.layer_u_render_only and cfg.layer_ctrl:
                # P1 (docs/surface_gradient.md 13): the u channel driven by the RENDER channel only — the
                # cell-sum loss lives on the cell grid and has no legitimate sub-cell content, so its
                # u-gradient is the granularity signal; the W1 term is physics-side too
                iu = [i for i, l in enumerate(leaves) if l is u][0]
                g[iu] = lam_r * gr[iu] if gr[iu] is not None else torch.zeros_like(u)
        elif balancer.active:
            total = Lp_core + lam_r * lr if Ldt is None else Lp_core + Ldt + lam_r * lr
            g = list(torch.autograd.grad(total, leaves))
        else:
            total = Lp_core if Ldt is None else Lp_core + Ldt
            g = list(torch.autograd.grad(total, leaves))
        if cfg.grad_h1 and knn_t is not None and basis.per_particle:
            # Sobolev descent direction: the total control gradient projected onto directions
            # that are smooth on the material kNN graph (screened Poisson, solved to
            # convergence; norm preserved). Every channel, after PCGrad/lambda.
            g[0] = _sobolev_direction(g[0], knn_t, cfg.control_h1_kappa)
        _tm_add("grad", t0)
        cur = scalars(lv, lk, lr, lam_r, dfc_x, state[0], state[1], extra["lk_run"],
                      extra["Fg"] if use_geom else None, extra["lk_var"], _vT(extra))
        if not np.isfinite(cur):
            log(f"[win] iter {it}: non-finite loss, aborting window")
            break
        if L_start is None:
            L_start = cur
            if cont_lim is not None:
                # reference = the window's FREE rollout (zero control): what physics does
                # without this window's actuation; computed once per window
                with torch.no_grad():
                    keep = dFc.detach().clone()
                    dFc.zero_()
                    _, _, _, _, _, ex_free = eval_terms(dFc)
                    dFc.copy_(keep)
                Vf = ex_free.get("V") if isinstance(ex_free, dict) else None
                cont_state["ref"] = cont_rel(Vf) if Vf is not None else None
                if cont_state["ref"] is not None:
                    cont_state["ref_ratio"] = float((cont_state["ref"] / cont_lim).max())
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
            a_try *= max(cfg.min_alpha_scale, min(1.0, target_norm_eff / max(gn, 1e-30)))

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
                    d_ = mh / (vh.sqrt() + eps_eff)
                    if ctrl_scale_v is not None and p is dFc:
                        d_ = d_ * ctrl_scale_v          # the per-particle Rprop scale (config.ctrl_rprop)
                    if W_apply is not None and u is not None and p is u:
                        d_ = W_apply(d_)             # the u step on the layer's smooth subspace (§7)
                    p -= (a_try * sc) * d_
                if cfg.dfc_clip > 0:
                    n = dFc.flatten(2).norm(dim=2, keepdim=True).unsqueeze(-1)
                    dFc *= (cfg.dfc_clip / n.clamp_min(1e-8)).clamp(max=1.0)
                if s is not None:
                    s.clamp_(-cfg.mat_clamp, cfg.mat_clamp)
                if u is not None:
                    u.copy_(torch.maximum(torch.minimum(u, u_bound), -u_bound))   # one spacing per window, x the Rprop scale
            state_n, lv_n, lk_n, lr_n, lpbr_n, extra_n = eval_terms(dFc)
            with torch.no_grad():
                new = scalars(lv_n, lk_n, lr_n, lam_r, extra_n["dfc"], state_n[0],
                              state_n[1], extra_n["lk_run"], extra_n["Fg"], extra_n["lk_var"], _vT(extra_n))
                predicted_decrease = -float(sum((gi.detach() * (p - b)).sum()
                                                for gi, p, b in zip(g, leaves, bak)))
                # The Armijo slope is only meaningful when the model predicts descent.
                # A carried/stale Adam moment can point against the fresh gradient
                # (predicted_decrease <= 0) while the step still lowers the true
                # objective; auto-rejecting there exhausts the line search on the
                # first windows after every plateau commit (observed: null commits
                # at anim 12/18 in final_hires20k_child4). Fall back to the
                # noise-floor sufficient decrease instead.
                noise_floor = cfg.ls_noise_rel * max(abs(cur), 1.0 / unit_ratio)  # loss units
                required = (max(cfg.armijo_c1 * predicted_decrease, noise_floor)
                            if predicted_decrease > 0.0 else noise_floor)
            # acceptance requires a FINITE, ORIENTATION-PRESERVING state: NaN particles
            # vanish from the splats and det(F)<=0 is invisible to the data terms — both
            # can fake a lower loss (adversarial finding + v3 warm-start cascade).
            merit_ok = bool(np.isfinite(new) and floor <= new <= cur - required and _state_ok(state_n))
            cont_ok = cont_check(extra_n) if merit_ok else True
            if merit_ok and cont_ok:
                adam_t = t_
                # config.ctrl_rprop_hold: under the per-particle Rprop the global step does not grow on
                # acceptance (Rprop has no global rate: the per-particle scale is the only step control) —
                # ag300 read the global step re-inflating 5-8x and cancelling the per-particle decay
                alpha = a_try if (ctrl_scale_v is not None and getattr(cfg, "ctrl_rprop_hold", False)) else min(a_try * 1.1, cfg.alpha * alpha_scale)  # C++ grows alpha on acceptance
                step_ok = True
                accepted += 1
                break
            with torch.no_grad():                            # reject: restore and shrink
                for p, b in zip(leaves, bak):
                    p.copy_(b)
                for m_, b in zip(mom, bak_m):
                    m_.copy_(b)
                for v_, b in zip(vel, bak_v):
                    v_.copy_(b)
            if merit_ok and not cont_ok:
                # only continuity failed: the control-induced relative velocity is ~linear
                # in the step, so shrink by the measured excess (never less than halving)
                # (measured at 150k: a gentler shrink raised the rejections 18 -> 71; at
                # least halve, shrink more when the excess is large)
                a_try *= max(0.05, min(0.5, 0.8 / max(cont_state.get("excess", 2.0), 1.0 + 1e-6)))
            else:
                a_try *= 0.5
        if not step_ok:
            rejected += 1
            log(f"[win] iter {it}: line search exhausted (cur={cur:.6g} last_new={new:.6g} "
                f"required={required:.3g} pred={predicted_decrease:.3g} ||g||={gn:.3g} "
                f"a_try={a_try:.3g} state_ok={_state_ok(state_n)}; last-attempt deltas "
                f"d_vol={float(lv_n - lv.detach()):.3g} kin={float(lk_n - lk.detach()):.3g} "
                f"render={(float(lr_n - lr.detach()) if lr is not None else 0.0):.3g} "
                f"lam={lam_r:.3g}"
                f"{'; continuity violators=' + str(cont_state['viol']) + ' max rel/lim=' + format(cont_state['ratio'], '.2f') if cont_lim is not None else ''})")
            alpha *= 0.5
            if alpha < 1e-8:
                log("[win] alpha underflow, stopping window")
                break
            # An exhausted line search leaves the point, the Adam state and therefore the
            # next gradient UNCHANGED (everything a rejected trial touched is restored), so
            # the following iteration would recompute the same gradient and re-test step
            # sizes the search just rejected (alpha/2 ... alpha/2^10 after alpha ... alpha/2^9)
            # — 10 rollouts and an adjoint per iteration for one new trial at alpha/2^10,
            # which cannot clear the noise floor. The window ends here instead (measured
            # at 150k: 3 exhausted iterations per window = 30 of 37 rollouts).
            ls_exhausted = True
            if _NO_LS_BREAK:                 # bisection switch: the pre-37e6629 behaviour
                continue
            log(f"[win] line search exhausted at iter {it}: ending the window")
            break

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
            F_view = extra_n["Fg"] if use_geom else state_n[1]     # REFUTE F8: same F
            on_iter(it, state_n[0].detach().cpu().numpy().astype(np.float32),  # as on_commit
                     F_view.detach().reshape(N, 3, 3).cpu().numpy().astype(np.float32),
                     {"loss": new, "d_vol": float(lv_n), "kin": float(lk_n),
                      "kin_run": float(extra_n["lk_run"]), "kin_var": float(extra_n["lk_var"]),
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
                      "_grad_phys": (gx_phys_diag.detach().cpu().numpy().astype(np.float32)
                                     if gx_phys_diag is not None else None),   # first/last iter only
                      "_grad_render": (gx_rend_diag.detach().cpu().numpy().astype(np.float32)
                                       if gx_rend_diag is not None else None)})
        # history from the ACCEPTED evaluation. NOTE "d_render" is the pure silhouette
        # scalar; the shading channel is logged separately (they were conflated before).
        hist.append({"iter": it, "loss": new,
                     "d_vol": float(lv_n), "kin": float(lk_n),
                     "kin_run": float(extra_n["lk_run"]), "kin_var": float(extra_n["lk_var"]),
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
                     "dfc_absmax": float(extra_n["dfc"].abs().max()),
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
        _set_material(lam_t, mu_t)
        dc = expand(dFc.detach()).detach().contiguous()
        dc_buf.copy_(dc.view(T, N, 3, 3))
        if u is not None:                # the ACCEPTED u (a rejected candidate's may sit in the buffer)
            wp.to_torch(tr_eval.layer_u).copy_(u.detach())
        t0 = _tick()
        tr = tr_eval                     # the same bonds and buffers as every candidate
        tr.run()
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
        Fg_final = wp.to_torch(tr.Fg[T]).reshape(N, 9) if use_geom else None
        V_final = torch.stack([wp.to_torch(tr.v[t]) for t in range(1, T + 1)])
        lv_f, lk_f, lr_f, _ = losses_of(x_final, F_final, v_final, Fg_final)
        E_final = scalars(lv_f, lk_f, lr_f, lam_r, dc, x_final, F_final,
                          V_final.pow(2).sum(2).mean(), Fg_final,
                          (V_final.pow(2).sum(2).mean(0) - V_final.mean(0).pow(2).sum(1)).mean(), V_final[-1])
        E_accept = hist[-1]["loss"] if hist else None
        replay_tol = (max(cfg.ls_noise_rel, 10.0 * replay_rel)
                      * max(abs(E_accept or 0.0), 1.0 / unit_ratio))
        replay_bad = E_accept is not None and E_final > E_accept + replay_tol
        if ((not np.isfinite(jt_final) or jt_final <= 1e-4 or replay_bad)
                and accepted > 0):
            log(f"[win] commit rollout failed trajectory check (jt={jt_final:.3g}) — "
                "discarding window (replay/accepted-candidate mismatch)")
            hist, accepted = [], 0
        # 2026-09-23 (speed): wp.array.numpy() already returns a fresh host copy — the extra
        # .copy() doubled 300 MB of traffic a window; and the whole-window F health (any step
        # with det F <= 0, per particle) is counted on the device instead of stacking T x N
        # matrices on the host for a numpy determinant (4 s a window at 300k)
        frames = [tr.x[t].numpy() for t in range(T + 1)]
        F_seq = [tr.F[t].numpy() for t in range(T + 1)]
        if _rp_save:                     # P278b: the window's accepted control and what it delivered
            np.savez(_rp_save, x0=np.asarray(frames[0], np.float32), x1=np.asarray(frames[-1], np.float32),
                     dfc=dc.view(T, N, 3, 3).cpu().numpy().astype(np.float32),
                     u=(u.detach().cpu().numpy().astype(np.float32) if u is not None else np.zeros(N, np.float32)))
            print(f"[replay] window control saved to {_rp_save} (T={T}, N={N})", flush=True)
        if _rp_load:
            _rp_out = os.environ.get("PHYSMORPH_REPLAY_OUT", _rp_load.replace(".npz", "") + f"_replayed_N{N}.npz")
            np.savez(_rp_out, x0=np.asarray(frames[0], np.float32), x1=np.asarray(frames[-1], np.float32))
            print(f"[replay] delivered end positions saved to {_rp_out}", flush=True)
        with torch.no_grad():
            inv_any = None
            jmin_traj = float("inf")
            for t in range(1, T + 1):
                Ft = wp.to_torch(tr.F[t]).reshape(-1, 3, 3).float()
                det_t = torch.linalg.det(Ft)
                bad = det_t <= 0.0                            # NaN rows compare False, as the numpy path did
                inv_any = bad if inv_any is None else (inv_any | bad)
                jmin_traj = min(jmin_traj, float(det_t.min().item()))   # numpy min propagates NaN; torch min too
            n_inv_steps = int(inv_any.sum().item()) if inv_any is not None else 0
        end = {"F": tr.F[T].numpy(), "v": tr.v[T].numpy(),
               "C": tr.C[T].numpy(),
               "Fg": tr.Fg[T].numpy() if use_geom else None,
               "n_inv_steps": n_inv_steps, "Jmin_traj": jmin_traj}
        _tm_add("final", t0)
        if cfg.grad_dump and grad_dump_state.get("gx_phys") is not None:
            # linear-response rollouts: each channel's control gradient alone, scaled to the
            # SAME control norm as the accepted change of this window (leaf_final - leaf0), so
            # the end states are comparable: what would the window have done had it followed
            # only the physics / silhouette / shading channel?
            leaf_final = dFc.detach().clone()
            step_norm_c = float((leaf_final - grad_dump_leaf0).norm())
            resp = {}
            for name in ("gl_phys", "gl_sil", "gl_pbr", "gl_rend"):
                gl = grad_dump_state.get(name)
                if gl is None:
                    continue
                gn = float(gl.norm())
                if gn <= 0 or step_norm_c <= 0:
                    continue
                cand = grad_dump_leaf0 - (step_norm_c / gn) * gl
                dc_c = expand(cand).detach().contiguous()
                dc_buf.copy_(dc_c.view(T, N, 3, 3))
                tr.run()
                resp["xT_" + name[3:]] = wp.to_torch(tr.x[T]).clone().cpu().numpy()
            dc_b = expand(grad_dump_leaf0).detach().contiguous()
            dc_buf.copy_(dc_b.view(T, N, 3, 3))
            tr.run()
            resp["xT_base"] = wp.to_torch(tr.x[T]).clone().cpu().numpy()
            u_red = {}
            if u is not None:
                # the u channel alone: each term's u-gradient from u0 = 0, scaled to the accepted
                # u's norm, with the START control (dFc = leaf0); base = leaf0 with u = 0
                u_final = u.detach().clone()
                u_step = float(u_final.norm())
                for name in ("gu_phys", "gu_sil", "gu_pbr", "gu_rend"):
                    gu = grad_dump_state.get(name)
                    if gu is None:
                        continue
                    u_red[name] = gu.cpu().numpy()
                    gn = float(gu.norm())
                    if gn <= 0 or u_step <= 0:
                        continue
                    u_cand = (-(u_step / gn) * gu).clamp(-sp0, sp0)
                    wp.to_torch(tr_eval.layer_u).copy_(u_cand)
                    tr.run()
                    resp["xT_" + name[3:] + "_u"] = wp.to_torch(tr.x[T]).clone().cpu().numpy()
                wp.to_torch(tr_eval.layer_u).zero_()
                tr.run()
                resp["xT_base_u0"] = wp.to_torch(tr.x[T]).clone().cpu().numpy()
                u_red["u_final"] = u_final.cpu().numpy()
                u_red["u_step"] = u_step
                u_red["layer_mask"] = np.asarray(layer[0], np.float32)
                wp.to_torch(tr_eval.layer_u).copy_(u_final)
            # restore the committed rollout in the buffers (the runner reads frames/end above,
            # already copied; the buffers themselves are rewritten by the next window)
            dc_buf.copy_(dc.view(T, N, 3, 3)); tr.run()
            os.makedirs(cfg.grad_dump, exist_ok=True)
            k_win = len([f for f in os.listdir(cfg.grad_dump) if f.startswith("win_")])
            gd = grad_dump_state
            red = {}
            for name in ("gl_phys", "gl_sil", "gl_pbr", "gl_rend"):
                gl = gd.get(name)
                if gl is not None:
                    g4 = expand(gl).detach().view(T, N, 9)
                    red[name + "_pnorm"] = g4.norm(dim=(0, 2)).cpu().numpy()      # (N,)
                    red[name + "_tmean"] = g4.mean(0).cpu().numpy()               # (N,9)
            np.savez_compressed(os.path.join(cfg.grad_dump, f"win_{k_win:04d}.npz"),
                                x0=x0, xT0=gd["xT0"].cpu().numpy(), xT_final=frames[-1],
                                gx_phys=gd["gx_phys"].cpu().numpy(), gx_sil=gd["gx_sil"].cpu().numpy(),
                                gx_pbr=(gd["gx_pbr"].cpu().numpy() if gd.get("gx_pbr") is not None else np.zeros(0)),
                                lam_r=gd["lam_r"], g_share=gd["g_share"], step_norm=step_norm_c,
                                leaf0_norm=float(grad_dump_leaf0.norm()), leaf_final_norm=float(leaf_final.norm()),
                                **red, **resp, **u_red)
    if _TIMING:
        tot = time.perf_counter() - _TM.get("t_win", time.perf_counter())
        keys = ("eval_roll", "eval_loss", "eval_det", "terms", "grad", "g_phys", "g_dt", "g_rend", "final")
        print("[time] " + " ".join(f"{k} {_TM.get(k, 0.0):.2f}s/{_TM.get('n_' + k, 0)}x" for k in keys)
              + f" other {tot - sum(_TM.get(k, 0.0) for k in keys[:6]):.2f}s window {tot:.2f}s", flush=True)
    s_out = s.detach().cpu().numpy() if s is not None else None
    if cfg.mom_carry > 0:
        mom_out = ([m.detach() for m in mom], [v.detach() for v in vel], adam_t)
    stats = {"pace_bound": pace_bound, "replay_rel": replay_rel, "h1_ratio": h1_ratio,
             "mom_out": mom_out if cfg.mom_carry > 0 else None,
              "accepted": accepted, "rejected": rejected, "grad_converged": grad_converged,
              "ls_exhausted": ls_exhausted,
              "L_start": L_start, "g_cos": g_cos, "g_raw_cos": g_raw_cos,
              "g_share": g_share, "u_gate": u_gate_frac, "pace_proj": pace_proj_stats, "arrived_mask": arrived_mask_np, "arrive_idx": arrive_idx_np, "pace_r": pace_r_np, "plan_img": plan_img_np, "arrive_cap_frac": arrive_cap_frac, "pace_front_frac": pace_front_frac, "pace_front_fill_frac": pace_front_fill_frac,
              "gx": (gx_box[0].cpu().numpy().astype(np.float32) if gx_box[0] is not None else None),
              "u_final": (u.detach().cpu().numpy() if u is not None else None),
              "g_phys_norm": g_phys_norm, "g_rend_norm": g_rend_norm,
              "render_work": render_work, "render_work_x": render_work_x,
              "render_work_F": render_work_F, "phys_work": phys_work,
              "phys_work_x": phys_work_x, "phys_work_F": phys_work_F,
              "phys_work_v": phys_work_v,
              "step_norm": step_norm, "predicted_decrease": predicted_decrease,
              "render_cos": render_cos, "phys_cos": phys_cos,
             "fill_lam": fill_lam if fill_on else None,
             "basis": basis.describe(),
             "lambda_capped": lam_capped,
             "dfc": dc.cpu().numpy() if cfg.warm_start else None,
             "cont_ratio": cont_state["ratio"] if cont_lim is not None else None,
             "cont_rejects": cont_state["rejects"] if cont_lim is not None else None,
             "cont_ref_ratio": cont_state["ref_ratio"] if cont_lim is not None else None}
    return frames, F_seq, end, s_out, hist, stats
