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
from .config import PipelineConfig
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


def optimize_window(x0, prm: MPMParams, cfg: PipelineConfig, tgt: TargetPack,
                    balancer: LambdaBalancer, F0=None, Fp=None, v0=None, C0=None,
                    s_init=None, dfc_init=None, on_iter=None, log=print,
                    fill_bal: LambdaBalancer | None = None, alpha_scale: float = 1.0,
                    mom_init=None, vol0=None, surface_w=None, Fg0=None, coh_nbr=None,
                    coh_nbr_src=None, frontier=None, bond_rest=None, bond_frag=None):
    """Optimise dFc[0..T-1] (+ material s) over one horizon. Returns
    (frames, F_seq, end_state, s_out, hist, stats).

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
    if isinstance(m_np, np.ndarray) and np.allclose(m_np, 1.0):
        m_np = 1.0                                    # unit masses: keep the scalar path
    spec = RolloutSpec(x0=x0, m=m_np, lam=lam0, mu=mu0, prm=prm, T=T,
                       F0=F0, Fp=Fp, v0=v0, C0=C0, device=dev, vol0=vol0, Fg0=Fg0,
                       bond_nbr=bond_nbr, bond_rest=bond_rest, bond_frag=bond_frag)

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
                         bonds=((bond_nbr, bond_rest, bond_frag) if bond_nbr is not None else None))
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
        p_sp = float(tgt.nn_spacing) if tgt.nn_spacing > 0 else 0.5 * float(tgt.ldx)
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
            p_sp = float(tgt.nn_spacing) if tgt.nn_spacing > 0 else 0.5 * float(tgt.ldx)
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
        if cfg.phys_loss in ("ot", "ot_leash", "ot_pace", "ot_shape"):
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
                p_sp = float(tgt.nn_spacing) if tgt.nn_spacing > 0 else 0.5 * float(tgt.ldx)
                k_nb = int(max(4, min(64, round(4.0 / 3.0 * np.pi * (leash_r / p_sp) ** 3))))
                x0_np = np.ascontiguousarray(np.asarray(x0, np.float32))
                _, tgt.ot_knn = cKDTree(x0_np).query(x0_np, k=k_nb, workers=-1)
                tgt.ot_knn = torch.as_tensor(tgt.ot_knn, device=dev)
                print(f"[win] OT leash: displacement denoised over k={k_nb} material neighbours "
                      f"(blur radius {leash_r:.4g} wu / spacing {p_sp:.4g})", flush=True)
            # the per-particle sampled map is noisy at the sample scale (~0.9 spacings on
            # the real bunny); the map of the continuum is smooth, so the displacement is
            # averaged over the material neighbourhood before projection
            disp = (ot_T - x0_ot)
            disp = disp[tgt.ot_knn].mean(dim=1)
            ot_T = x0_ot + disp
            # The hole regime (`ot`, e.g. C) uses this material-smoothed map image directly.
            # Two per-particle target variants were tried on the 150k C and FALSIFIED
            # (2026-09-17 night): (i) a PACED target — each window's target bounded to one
            # pace = max(plan blur, loss cell) along the smoothed map — cut the 40k C's
            # re-attachments 114 → 7 but at 150k gave silIoU 0.88 / 0.55 (unpaced 0.95): a
            # target that walks with the particle makes the window loss quasi-stationary,
            # so the inner line search and the merit gate see no descent and stop the run
            # on a half-formed body; (ii) resolving the displacement on the loss grid (CIC
            # deposit/gather) before the pace, 0.59. The chunks those variants were meant
            # to stop were particles frozen in the domain's boundary band (mpm/kernels.py
            # k_grid_op, the separating walls) — not a property of the target.
            if cfg.phys_loss == "ot_leash":
                _, nn = tgt.ot_kd.query(ot_T.detach().cpu().numpy(), workers=-1)
                ot_T = tgt.points[torch.as_tensor(nn, device=dev)].detach().to(ot_T.dtype)
        pace_grid = None
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
            step = torch.clamp(pace_r / dn.clamp_min(1e-9), max=1.0)
            x_int = (x0_ot + step * disp).detach()
            # an ARRIVED particle (within one blur radius of its image) contributes its
            # image projected onto the target point set: the entropic image sits inside
            # the target (blur), which left the end state fuzzy (150k bunny chamfer 0.098
            # vs 0.076); on the target support the end target is the target. (Snapping
            # every paced position that lies on the support instead — d4db68a — killed the
            # tangential transport where the source overlaps the target: 150k cow silIoU
            # 0.920 vs 0.944, 104 re-attachments vs 83.)
            arrived = dn.squeeze(1) <= pace_r
            if bool(arrived.any()):
                _, nn_a = tgt.ot_kd.query(x_int[arrived].cpu().numpy(), workers=-1)
                x_int[arrived] = tgt.points[torch.as_tensor(nn_a, device=dev)].to(x_int.dtype)
            pace_grid = rasterize_mass(x_int, tgt.m, tgt.lgmin, tgt.ldx, tgt.ldims).detach()
            frac_arrived = float(arrived.float().mean())
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
            m_dt = tgt.m * isolation_gate(x0_t, cfg.dt_iso_lo, cfg.dt_iso_hi)

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
            xT, FT, vT, FgT, V = adj_box[0].apply(dfc)
        else:
            xT, FT, vT, FgT, V = warp_mpm_ext(dfc, spec, lam_t, mu_t)
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
        if cfg.w_h1 > 0 and tgt.h1_scale is not None:
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

    for it in range(cfg.iters):
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
        if balancer.active and (special_render or cfg.grad_project or it == 0):
            if gp is None:
                t1 = _tick()
                gp = torch.autograd.grad(Lp_core, leaves, retain_graph=True)
                _tm_add("g_phys", t1)
            t1 = _tick()
            gdt = (torch.autograd.grad(Ldt, leaves, retain_graph=True)
                   if Ldt is not None else None)
            _tm_add("g_dt", t1)
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
            if mode in ("off", "render"):
                g = [a + lam_r * b for a, b in zip(gp, gr)]
            else:                    # "phys" / "cagrad" / "blend": grad_combine.combine
                g, _info = combine_grads(mode, gp, gr, lam_r, cfg.cagrad_c,
                                         blend_beta=balancer.alpha_lam)
            if gdt is not None:      # W1 joins the composite AFTER lambda/PCGrad (find. 9)
                g = [gi + di for gi, di in zip(g, gdt)]
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
                    p -= (a_try * sc) * mh / (vh.sqrt() + eps_eff)
                if cfg.dfc_clip > 0:
                    n = dFc.flatten(2).norm(dim=2, keepdim=True).unsqueeze(-1)
                    dFc *= (cfg.dfc_clip / n.clamp_min(1e-8)).clamp(max=1.0)
                if s is not None:
                    s.clamp_(-cfg.mat_clamp, cfg.mat_clamp)
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
                alpha = min(a_try * 1.1, cfg.alpha * alpha_scale)  # C++ grows alpha on acceptance
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
        frames = [tr.x[t].numpy().copy() for t in range(T + 1)]
        F_seq = [tr.F[t].numpy().copy() for t in range(T + 1)]
        end = {"F": tr.F[T].numpy().copy(), "v": tr.v[T].numpy().copy(),
               "C": tr.C[T].numpy().copy(),
               "Fg": tr.Fg[T].numpy().copy() if use_geom else None}
        _tm_add("final", t0)
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
              "g_share": g_share,
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
