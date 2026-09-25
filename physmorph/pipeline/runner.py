"""run_pipeline — outer commit loop of the v2 blessed path (docs/pipeline_v2.md §3.5, §3.7).

Per commit: optimise one window, promote the FULL state (x, F, v, C — partial promotion was
the v1 energy-re-injection bug), assimilate an eta-fraction of the elastic stretch into the
plastic rest state Fp (exact polar relaxation — render→Fp channel), track convergence on the
RAW loss components (λ-free; a drifting λ must not decide the freeze) and freeze on plateau.

Guards are containment + telemetry: a fired guard means the run is INVALID (gate G2 requires
every counter to read zero); the sanitisation only prevents one poisoned state from cascading
into meaningless downstream telemetry. F is repaired for numerical pathologies only
(non-finite rows, reflections — both counted); there is NO silent singular-value projection
in this path. The archived frames are the PROMOTED states, so metrics, plasticity and the
next window all describe the same trajectory.
"""
from __future__ import annotations

import dataclasses
import numpy as np
import torch

from ..losses.volumetric import (coverage_shortfall, d_h1, d_vol, d_w1, density_units,
                                 target_dt_grid, target_mass_grid)
from ..mpm.conditioning import condition_F
from ..mpm.state import MPMParams
from ..mpm.traj import compute_rest_volumes
from ..plasticity import assimilate_elastic
from .config import PipelineConfig, disc_ref_factor
from .optimizer import TargetPack, optimize_window
from .render_loss import (LambdaBalancer, d_render, make_views, shade_targets,
                          target_silhouettes)
from .surface_local import surface_local_pass


def _id(N):
    from ..mpm.traj import _id as _traj_id      # the cached identity (2026-09-23 speed pass)
    return _traj_id(N)


def _surface_weights(x: np.ndarray, k: int, fraction: float, floor: float) -> np.ndarray:
    """Persistent soft surface score from one-sided local neighbour directions.

    A volume-interior particle sees approximately cancelling unit directions; a boundary
    particle does not.  The score is computed once in source/material coordinates so the
    active set cannot flicker between optimisation windows.
    """
    from scipy.spatial import cKDTree
    n = len(x)
    kk = min(n, max(2, 3 * int(k) + 1))  # tolerate duplicate voxel samples
    _, idx = cKDTree(x).query(x, k=kk, workers=-1)
    d = x[idx[:, 1:]] - x[:, None, :]
    dn = np.linalg.norm(d, axis=2, keepdims=True)
    valid = dn[..., 0] > 1e-8
    unit = d / np.maximum(dn, 1e-8)
    score = np.linalg.norm((unit * valid[..., None]).sum(1) /
                           np.maximum(valid.sum(1, keepdims=True), 1), axis=1)
    threshold = float(np.quantile(score, 1.0 - float(fraction)))
    width = max(float(np.std(score)) * 0.15, 1e-4)
    soft = 1.0 / (1.0 + np.exp(np.clip(-(score - threshold) / width, -40.0, 40.0)))
    return np.ascontiguousarray(floor + (1.0 - floor) * soft, np.float32)


def reattach_fragments(x, v, C, F, Fp, Fg, prm: MPMParams, spacing: float, seed: int = 0,
                       tgt_points: np.ndarray | None = None) -> int:
    """Conservative particle resampling: every particle the fragment mask flags (its grid
    cell is not connected to the body on the occupancy dilated by one cell — it shares no
    grid node with any other material point, so it is a stray mass, not a continuum element)
    is merged IN PLACE onto the nearest body particle: position + half a spacing of jitter,
    v, C, F, Fp, Fg copied. No particle is deleted (mass conserved); returns the count.
    tgt_points: when given, a flagged particle that lies ON the target support (within one
    MPM cell of a target point) is left alone — a part of the body that sits where the
    target is, separated from the rest by a thin neck, is not ejecta (150k bob at ppc 27:
    2343 particles merged in one commit, a whole part teleported)."""
    frag = fragment_mask(x, prm)
    if tgt_points is not None and frag.any():
        from scipy.spatial import cKDTree
        d_t, _ = cKDTree(tgt_points).query(x[frag], k=1, workers=-1)
        keep = np.zeros_like(frag)
        keep[np.where(frag)[0][d_t > float(prm.dx)]] = True
        frag = keep
    n = int(frag.sum())
    if n == 0:
        return 0
    body_idx = np.where(~frag)[0]
    if len(body_idx) == 0:
        return 0
    from scipy.spatial import cKDTree
    _, jn = cKDTree(x[body_idx]).query(x[frag], k=1, workers=-1)
    jb = body_idx[jn]
    rng = np.random.default_rng(seed)
    jit = rng.normal(size=(n, 3)).astype(np.float32)
    jit *= (0.5 * float(spacing)) / np.maximum(np.linalg.norm(jit, axis=1, keepdims=True), 1e-9)
    x[frag] = x[jb] + jit
    for arr in (v, C, F, Fp, Fg):
        if arr is not None:
            arr[frag] = arr[jb]
    return n


def fragment_mask(x: np.ndarray, prm: MPMParams) -> np.ndarray:
    """True where the particle's occupied grid cell belongs to a connected component
    (26-connectivity) of occupied cells that is NOT the largest one: material that has
    broken off the body (numerical fracture debris). Thin features stay connected to
    the body through occupied cells and are never flagged."""
    from scipy import ndimage
    ijk = np.floor((x - np.asarray(prm.grid_min, np.float32)) / prm.dx).astype(np.int64)
    dims = np.array([prm.nx, prm.ny, prm.nz])
    ok = ((ijk >= 0) & (ijk < dims)).all(1)
    occ = np.zeros(dims, bool)
    occ[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]] = True
    # STENCIL connectivity: two particles couple through shared grid nodes when their cells
    # are within the 4^3 B-spline support of each other, so components are taken on the
    # occupancy dilated by one cell (a thin feature with a one-cell occupancy gap is still one
    # body; v4 on the raw occupancy flagged a 732-particle dragon spine as a fragment)
    occ_d = ndimage.binary_dilation(occ, structure=np.ones((3, 3, 3), bool))
    lab, n = ndimage.label(occ_d, structure=np.ones((3, 3, 3), int))
    if n <= 1:
        return np.zeros(len(x), bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    body = int(sizes.argmax())
    frag = np.ones(len(x), bool)
    frag[ok] = lab[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]] != body
    return frag


def _coupled_mask(x: np.ndarray, prm: MPMParams) -> np.ndarray:
    """True where the particle shares its 3^3 grid cells with at least one other particle
    (the same test the kernels use for material re-coupling)."""
    ijk = np.floor((x - np.asarray(prm.grid_min, np.float32)) / prm.dx).astype(np.int64)
    dims = np.array([prm.nx, prm.ny, prm.nz])
    ok = ((ijk >= 0) & (ijk < dims)).all(1)
    grid = np.zeros(dims, np.int32)
    np.add.at(grid, (ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]), 1)
    pad = np.pad(grid, 1)
    n = np.zeros(len(x), np.int64)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                n[ok] += pad[ijk[ok, 0] + a, ijk[ok, 1] + b, ijk[ok, 2] + c]
    return n > 1


def _iso_count(x: np.ndarray, radius: float) -> int:
    """Number of particles with no other particle within `radius` (the ejection signature:
    a lone particle cannot be re-coupled by the grid)."""
    from scipy.spatial import cKDTree
    d = cKDTree(x).query(x, k=2, workers=-1)[0][:, 1]
    return int((d > radius).sum())


def build_target(target_x, prm: MPMParams, cfg: PipelineConfig, w_tgt=None, w_src=None) -> TargetPack:
    dev = cfg.device
    from ..losses.silhouette import set_kernel
    set_kernel(getattr(cfg, "sil_kernel", "cic"))     # every rasteriser (targets and morph) alike
    N = target_x.shape[0]
    # per-particle masses: unit by default; with shell-biased sampling the relative rest
    # volumes (mean 1) of the TARGET particles build the target grid and those of the
    # SOURCE particles (tgt.m, the moving cloud's masses) enter every particle-side term
    m_t = torch.ones(N, device=dev) if w_tgt is None else torch.as_tensor(np.asarray(w_tgt, np.float32), device=dev)
    m = torch.ones(N, device=dev) if w_src is None else torch.as_tensor(np.asarray(w_src, np.float32), device=dev)
    dmin = np.asarray(prm.grid_min, np.float32)
    dmax = dmin + prm.dx * np.array([prm.nx, prm.ny, prm.nz], np.float32)
    ldx = float((dmax - dmin).max() / cfg.loss_res)
    ldims = (cfg.loss_res,) * 3
    lgmin = torch.tensor(dmin, device=dev)
    tgt_t = torch.tensor(np.ascontiguousarray(target_x, np.float32), device=dev)
    grid = target_mass_grid(tgt_t, m_t, lgmin, ldx, ldims)
    views = make_views(cfg.render_views, cfg.render_elevs)
    extent = float(np.abs(target_x).max()) * 1.25
    sils = shade = dt3 = None
    pgmin, pdx, pdims, pblur = None, 0.0, (), 0.0
    if cfg.lambda_auto > 0:
        sils = target_silhouettes(tgt_t, views, cfg.render_res, extent, cfg.sil_k)
        if cfg.w_pbr > 0 and cfg.pbr_denoised:
            # G1 (docs/surface_gradient.md §4): the shading reference from the TARGET's reconstructed
            # surface (no shot noise), the morph's normals on a render-pixel grid over the loss
            # box, blurred by the renderer's 1.5 spacings
            from scipy.spatial import cKDTree as _KD
            from ..render.surface_recon import target_surface_normals
            sub = target_x[np.random.default_rng(0).choice(N, min(N, 20000), replace=False)]
            sp_t = float(np.median(_KD(sub).query(sub, k=9, workers=-1)[0][:, -1])) * (min(N, 20000) / N) ** (1.0 / 3.0)
            sp_t *= disc_ref_factor(N, cfg)                       # the reference spacing (config.disc_ref)
            n_t, sw_t = target_surface_normals(np.asarray(target_x, np.float32), sp_t)
            shade = shade_targets(tgt_t, views, cfg.render_res, extent, lgmin, ldx, ldims,
                                  cfg.sil_k, cfg.pbr_ambient,
                                  normals=(torch.as_tensor(n_t, device=dev), torch.as_tensor(sw_t, device=dev)))
            pdx = 2.0 * extent / cfg.render_res                       # the render pixel
            pdims = tuple(int(np.ceil((dmax - dmin).max() / pdx)) for _ in range(3))
            pgmin = lgmin
            pblur = 1.5 * sp_t / pdx
            print(f"[target] denoised shading target: spacing {sp_t:.4f}, normal grid {pdims[0]}^3 at {pdx:.4f} wu "
                  f"({pdx / sp_t:.2f} spacings), blur {pblur:.2f} cells", flush=True)
        elif cfg.w_pbr > 0:
            shade = shade_targets(tgt_t, views, cfg.render_res, extent,
                                  lgmin, ldx, ldims, cfg.sil_k, cfg.pbr_ambient)
    dtgmin, dtdx, dtdims, tmass3 = None, 0.0, (), None
    if cfg.w_dt > 0 or cfg.w_fill > 0 or cfg.w_grow > 0:
        # fine target-fitted grid shared by both one-signed W1 terms — independent of
        # the render channel (Opus finding 2: the loss grid's ~1-unit cells made a dead
        # radius covering the whole fringe band). Cube spans 1.5x extent: everything the
        # box leash allows stays on a live DT slope; the EDT is build-time.
        dtdims = (cfg.dt_res,) * 3
        dtdx = 3.0 * extent / cfg.dt_res
        dtgmin = torch.tensor([-1.5 * extent] * 3, device=dev)
        dt_mass = target_mass_grid(tgt_t, m_t, dtgmin, dtdx, dtdims)
        if cfg.w_dt > 0:
            dt3 = target_dt_grid(dt_mass, dtdx, dtdims,
                                 clamp=cfg.dt_clamp_frac * extent)
        if cfg.w_fill > 0 or cfg.w_grow > 0:
            tmass3 = dt_mass
    gauss = None
    if cfg.use_gauss_loss and cfg.lambda_auto > 0:
        from ..render.covariance import sigma0_from_nn
        from .gauss_loss import GaussViews
        target_mask = None
        gaussian_points = target_x
        if cfg.render_surface_only:
            tw = _surface_weights(target_x, cfg.surface_grad_k,
                                  cfg.surface_grad_frac, cfg.surface_grad_floor)
            target_mask = torch.as_tensor(tw > 0.5, device=dev)
            gaussian_points = target_x[tw > 0.5]
        gauss = GaussViews(views, extent,
                           sigma0_from_nn(gaussian_points, cfg.gauss_sigma_scale) * disc_ref_factor(len(target_x), cfg),
                           cfg.gauss_res, dev, child_count=cfg.gauss_children,
                           child_sigma_scale=cfg.gauss_child_sigma_scale,
                           child_offset_scale=cfg.gauss_child_offset_scale,
                           child_k=cfg.gauss_child_k,
                           robust_eps=cfg.gauss_robust_eps,
                           cov_sat=cfg.gauss_cov_sat)
        gauss.bake_targets(tgt_t, mask=target_mask)
    pts, nn_sp = None, 0.0
    kde_h, kde_rho = 0.0, 1.0
    if cfg.w_nn > 0 or cfg.w_kde > 0 or cfg.layer_gate:     # layer_gate: the u gate reads the target cloud
        from scipy.spatial import cKDTree
        nn_sp = float(np.median(cKDTree(target_x).query(target_x, k=2,
                                                        workers=-1)[0][:, 1]))
        nn_sp *= disc_ref_factor(len(target_x), cfg)      # the reference spacing (config.disc_ref)
        pts = tgt_t
        if cfg.w_kde > 0:
            from ..losses.volumetric import kde_self_density
            kde_h = cfg.kde_h_k * nn_sp
            kde_rho = kde_self_density(tgt_t, kde_h, cfg.kde_k)
    jd = {}
    if cfg.w_jdens > 0:
        jd_dims = (cfg.jdens_res,) * 3
        jd_dx = float((dmax - dmin).max() / cfg.jdens_res)
        jd = dict(jd_gmin=lgmin, jd_dx=jd_dx, jd_dims=jd_dims)
    m_ref, n_support = density_units(grid)      # loss_units="density" constants
    return TargetPack(**jd, points=tgt_t, grid=grid, lgmin=lgmin, ldx=ldx, ldims=ldims, m=m,
                      views=views, sils=sils, extent=extent, shade=shade,
                      dt3=dt3, dtgmin=dtgmin, dtdx=dtdx, dtdims=dtdims, tmass3=tmass3,
                      pts=pts, nn_spacing=nn_sp, gauss=gauss,
                      kde_h=kde_h, kde_rho_ref=kde_rho,
                      m_ref=m_ref, n_support=n_support,
                      pgmin=pgmin, pdx=pdx, pdims=pdims, pblur=pblur)


def calibrate_units(tgt: TargetPack, source_x, target_x, cfg: PipelineConfig) -> None:
    """loss_units="density": MEASURE the two legacy/density ratios at the source state
    (REFUTE F1 2026-09-15: the analytic per-cell constant n*2m/(1+m) was 4-45x off the
    measured loss ratio and 5-44x off the gradient ratio, and the two ratios differ by
    1.2-1.3x, so one scalar cannot serve both). unit_ratio converts every fixed weight so
    the objective's relative weighting equals the legacy one AT THE SOURCE (it drifts
    along the morph, like the h1 calibration — logged, not assumed away);
    unit_grad_ratio converts the gradient-magnitude constants."""
    from ..losses.volumetric import d_vol_density
    xs = torch.as_tensor(np.ascontiguousarray(source_x, np.float32), device=cfg.device)
    # REFUTE-2 F6: the LEGACY side of the ratio is evaluated on a FIXED reference loss
    # grid (cfg.unit_ref_res, the grid every legacy weight was tuned on), not on the
    # run's own grid — otherwise the converted weights inherit the cell-sum's resolution
    # dependence (measured 3.2x weaker at --ppc 8 / 149^3 than at 64^3 for equal flags)
    ref = int(getattr(cfg, "unit_ref_res", 0) or 0)
    if ref > 0 and ref != int(tgt.ldims[0]):
        dims_ref = (ref,) * 3
        ldx_ref = float(tgt.ldx * tgt.ldims[0] / ref)
        tgt_t = torch.as_tensor(np.ascontiguousarray(target_x, np.float32), device=cfg.device)
        grid_ref = target_mass_grid(tgt_t, tgt.m, tgt.lgmin, ldx_ref, dims_ref)
    else:
        grid_ref, ldx_ref, dims_ref = tgt.grid, tgt.ldx, tgt.ldims
    xg = xs.clone().requires_grad_(True)
    L_leg = d_vol(xg, tgt.m, grid_ref, tgt.lgmin, ldx_ref, dims_ref)
    g_leg = torch.autograd.grad(L_leg, xg)[0].norm()
    xg2 = xs.clone().requires_grad_(True)
    L_den = d_vol_density(xg2, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims,
                          tgt.m_ref, tgt.n_support)
    g_den = torch.autograd.grad(L_den, xg2)[0].norm()
    if float(L_den) <= 0 or float(g_den) <= 0:
        raise ValueError("density-unit calibration needs a source that differs from the "
                         "target (zero residual at the source)")
    tgt.unit_ratio = float(L_leg / L_den)
    tgt.unit_grad_ratio = float(g_leg / g_den)


def run_pipeline(source_x, target_x, prm: MPMParams, cfg: PipelineConfig, log=print,
                 on_commit=None, on_iter=None, w_src=None, w_tgt=None):
    """Morph source -> target. Returns a result dict (frames, F_frames, history, guards, s,
    n_held, converged). frames/F_frames archive the PROMOTED per-step states.
    on_commit(a, x, F, v, rec) fires after each promoted commit; on_iter(it, xT, FT, tele)
    streams each accepted optimisation iteration (live viewer hooks)."""
    src = np.ascontiguousarray(source_x, np.float32)
    N = src.shape[0]
    assert target_x.shape[0] == N, ("D_vol compares unit-mass clouds: source and target need "
                                    f"the same particle count (got {N} vs {target_x.shape[0]})")
    if cfg.lg_sweeps > 0 and (cfg.w_dt > 0 or cfg.w_fill > 0):
        # Codex finding 7 (+stack-review f11): the local pass's exact-quadratic energy
        # excludes BOTH one-signed W1 terms, so it can undo an accepted W1/fill step and
        # assimilation then ratchets the regression
        raise ValueError("lg_sweeps>0 with w_dt>0 or w_fill>0 is unsupported (local "
                         "energy has no W1/fill term; a non-quadratic term breaks its "
                         "exact line search)")
    if cfg.pace_budget > 0:          # budget-derived glidepath (see config.pace_budget)
        cfg.pace = 1.0 - cfg.pace_budget ** (1.0 / max(cfg.animations, 1))
        log(f"[v2] pace_budget={cfg.pace_budget:g} over {cfg.animations} anims -> "
            f"per-window cap {cfg.pace:.4f}")
    tgt = build_target(target_x, prm, cfg, w_tgt=w_tgt, w_src=w_src)
    if getattr(cfg, "phys_loss", "density") == "auto":
        # regime of the discretised problem (2026-09-17, C forensic): when the source's
        # mass sits in cells the target leaves empty (the sphere inside the C's hole) the
        # cell sum has nothing but an outward push there and only a transport plan says
        # where the mass goes; when source and target overlap, the cell sum's local fill
        # is the better objective. Measured once at the start, no per-shape choice.
        from ..losses.volumetric import gather_cic
        with torch.no_grad():
            xs = torch.as_tensor(np.asarray(src, np.float32), device=cfg.device)
            m_at = gather_cic(tgt.grid, xs, tgt.lgmin, tgt.ldx, tgt.ldims)
            empty = float((m_at <= 0).float().mean())
        if empty > 0.5:
            cfg.phys_loss = "ot"
        else:
            # overlap regime: the transport-PACED cell sum with the cell-wise hand-off —
            # the expansion is a coherent flow (no far-cell reward, strays 17x fewer on the
            # dragon) and the deficit cells next to the body fill at full strength
            cfg.phys_loss = "ot_pace"
            cfg.ot_handoff = True
            cfg.ot_debias = True
        log(f"[v2] phys_loss auto: {empty * 100:.1f}% of the source particles sit in target-empty "
            f"cells -> {cfg.phys_loss}" + (" + cell-wise hand-off" if cfg.phys_loss == "ot_pace" else ""))
    if cfg.loss_units == "density":
        calibrate_units(tgt, src, target_x, cfg)
        log(f"[v2] density units: D_vol legacy({cfg.unit_ref_res}^3)/density = "
            f"{tgt.unit_ratio:.4g} (weights), gradient ratio = {tgt.unit_grad_ratio:.4g} "
            f"(eps/target_norm), n_support={tgt.n_support} m_ref={tgt.m_ref:.3g}")
    if cfg.w_jdens > 0:                          # per-particle REST density at the source
        from ..losses.volumetric import density_at
        with torch.no_grad():
            tgt.jd_rho0 = density_at(torch.tensor(src, device=cfg.device), tgt.m,
                                     tgt.jd_gmin, tgt.jd_dx, tgt.jd_dims).detach()
    if cfg.loss_units == "density":              # the absolute cap is a legacy-unit
        balancer = LambdaBalancer(cfg.lambda_auto, cfg.lambda_ema, None,   # number:
                                  cap_rel=20.0)  # relative guard (20x the first ratio)
    else:
        balancer = LambdaBalancer(cfg.lambda_auto, cfg.lambda_ema, cfg.lambda_cap)
    # the local-global pass calibrates λ in ITS OWN variable space (u, joules) — sharing
    # the global balancer both mis-scales the pass and poisons the global EMA
    lg_balancer = LambdaBalancer(cfg.lambda_auto, cfg.lambda_ema, cfg.lambda_cap)
    # fill v3: its own norm balancer (alpha = w_fill) — dominance structurally bounded
    fill_balancer = (LambdaBalancer(cfg.w_fill, cfg.lambda_ema, cap=100.0)
                     if cfg.w_fill > 0 else None)

    dmin = np.asarray(prm.grid_min, np.float32)
    dmax = dmin + prm.dx * np.array([prm.nx, prm.ny, prm.nz], np.float32)
    lo, hi = dmin + 2 * prm.dx, dmax - 2 * prm.dx

    x = src.copy()
    if prm.gate_r_hi > prm.gate_r_lo and prm.gate_n0 <= 0:
        # support-gated APIC: n0 is fixed ONCE from the source so every window gates alike
        from ..mpm.step import nominal_support
        prm = dataclasses.replace(prm, gate_n0=nominal_support(src, prm, cfg.device))
        log(f"[v2] support gate: n0={prm.gate_n0:.1f} (median 3^3-cell count of the source), "
            f"r_lo={prm.gate_r_lo} r_hi={prm.gate_r_hi}")
    vol0 = (compute_rest_volumes(src, (1.0 if w_src is None else np.asarray(w_src, np.float32)), prm, cfg.device)
            if cfg.persistent_rest_volume else None)
    surface_w = (_surface_weights(src, cfg.surface_grad_k, cfg.surface_grad_frac,
                                  cfg.surface_grad_floor)
                 if cfg.surface_grad_frac > 0 else None)
    coh_nbr = None
    bond_rest = None                     # material re-coupling: rest lengths carried across windows
    bond_frag = None
    if cfg.w_coh > 0 or cfg.w_bond > 0 or cfg.w_esc > 0 or cfg.continuity or cfg.bonds:   # frozen source-material neighbours
        from scipy.spatial import cKDTree
        # the bond neighbourhood holds coh_k REFERENCE particles' mass (config.disc_ref: x N / mass_ref_n)
        coh_nbr = cKDTree(src).query(src, k=int(round(cfg.coh_k * disc_ref_factor(len(src), cfg) ** 3)) + 1,
                                     workers=-1)[1][:, 1:]
    if cfg.render_surface_only:
        if surface_w is None:
            raise ValueError("render_surface_only requires surface_grad_frac > 0")
        surface_w = np.ascontiguousarray(surface_w > 0.5, np.float32)
    if tgt.gauss is not None and cfg.gauss_children > 1:
        if not cfg.render_surface_only:
            raise ValueError("gauss_children>1 requires render_surface_only material parents")
        tgt.gauss.configure_source(src, surface_w > 0.5)
    dress = None
    if cfg.local_dress_iters > 0:                # Tier D (design v2 §4): observation-only
        if tgt.gauss is None or cfg.gauss_children <= 1:
            raise ValueError("local_dress_iters>0 requires use_gauss_loss and gauss_children>1")
        from .dressing import DressState
        dress = DressState(tgt.gauss, src, surface_w > 0.5, cfg.dress_cap_frac,
                           cfg.device)
    st = {"F": None, "v": None, "C": None, "Fg": None}
    Fp = _id(N)
    s, dfc_prev = None, None
    frames, F_frames, hist = [x.copy()], [_id(N)], []
    n_reattach_total = 0                 # cfg.reattach: merged grid-disconnected particles (all commits)
    sp_native = None                     # cfg.shift_sub: the cloud's native spacing (measured at the first commit)
    last_shift = None                    # cfg.shift_sub: this window's shift statistics (for the record)
    cyc_sub, cyc_hist = None, []         # net / summed displacement test (config.stop_on_cycle): a fixed subsample,
                                         # its commit positions over the last `patience` windows
    cyc_stale = 0                        # consecutive windows at or below the random-walk bound
    u_scale, u_prev = None, None         # config.u_rprop: the per-particle u bound scale and the last accepted u
    ctrl_scale, ctrl_prev_disp = None, None   # config.ctrl_rprop: the per-particle control step scale and the last accepted displacement
    ctrl_scale_apply = None              # the scale handed to the optimiser (neighbourhood-smoothed under ctrl_rprop_smooth)
    ctrl_rev_count, frozen_p = None, None  # config.freeze_arrived: per-particle reversal count and the frozen set
    settled_p, settle_eta_arr = None, None  # config.settle_eta: the settled set and the per-particle viscosity handed to the rollout
    settle_pin_arr = None                # config.settle_pin: the (N,) pin array handed to the rollout
    settled_at = None                    # config.settle_pin: the window (1-based) at which each particle was pinned, -1 = never
    rest_latched = False                 # config.rest_commit: windows from rest once the transport has arrived
    rev_prev_neg = False                 # config.rest_commit_reversal: the previous accepted commit reversed its predecessor
    rev_prev_neg_acc = False             # config.outer_latch_reversal: the same reading, kept at every accepted commit
    rev_prev_neg_arr = False             # config.ctrl_rprop_hold_onset: the reversal on the arrived particles, previous commit
    outer_latched_rev = False            # config.outer_latch_reversal: the gate armed at the alternation's onset (kept)
    # geometric (render) deformation at every ACCEPTED commit, aligned to frame_end —
    # the covariance the viewer/deliverable renders when cfg.render_F_geom (F_frames
    # keeps the PHYSICS F for metrics and assimilation)
    Fg_commits = []
    guards = {"clamped": 0, "nan_x": 0, "nan_state": 0, "F_reset": 0, "F_flip": 0,
              "F_invert_steps": 0}
    # freeze tracks the RAW components (λ-free): physics, render, and W1 tracks. The W1
    # track is SEPARATE (Opus finding 4: folded into phys_track it sat at the tolerance
    # noise floor, so fringe-only progress could still stale out)
    best_phys, best_rend, best_dt, best_fill, best_kde = None, None, None, None, None
    best_jd = None
    best_h1 = None
    reject_streak, last_reject_score = 0, None
    stale, frozen, n_held = 0, False, 0
    anneal = 1.0                     # plateau-scheduled step scale (zigzag forensic)
    prev_tracks = None               # last ACCEPTED commit's lambda-free tracks
    mom_prev = None                  # cross-window Adam moments (mom_carry)
    outer_scales = outer_prev = outer_prev_phys = prev_disp = None
    # Once the trajectory first reaches the small-motion regime, keep the outer
    # trust gate active.  Without this latch, one accepted large-motion candidate
    # disables the gate again and the optimizer can re-enter a long limit cycle.
    outer_gate_latched = False

    log(f"[v2] N={N} T={cfg.T} iters={cfg.iters} animations={cfg.animations} "
        f"render={'on(a=%g)' % cfg.lambda_auto if cfg.lambda_auto > 0 else 'OFF'} "
        f"material={'on' if cfg.opt_material else 'off'} assim={cfg.assim} "
        f"w_kin={cfg.w_kin} w_box={cfg.w_box}")
    if cfg.control_grid > 0 or cfg.control_tknots > 0 or cfg.render_F_geom \
            or cfg.w_kin_running > 0 or cfg.loss_units != "legacy":
        log(f"[v2] render-controls-physics contract: control_grid={cfg.control_grid} "
            f"tknots={cfg.control_tknots} render_F_geom={cfg.render_F_geom} "
            f"w_kin_running={cfg.w_kin_running} grad_mode="
            f"{cfg.grad_project_mode if cfg.grad_project else 'off'} "
            f"gs_cheb={cfg.render_gs_cheb} loss_units={cfg.loss_units}")

    for a in range(cfg.animations):
        if cfg.render_until > 0 and a == cfg.render_until and balancer.active:
            # INTERVENTION (2026-09-17): the render channel is switched off from here on;
            # everything else (merit, gate, targets) is unchanged, so any divergence of the
            # trajectory from the render-on twin after this window is the render gradient's
            # effect on the physics
            balancer.alpha_lam = 0.0
            balancer.lam = 0.0
            log(f"[v2] anim {a + 1}: render channel OFF from here (render_until={cfg.render_until})")
        if frozen:
            if cfg.hold_after_converge:
                frames.append(x.copy()); F_frames.append(F_frames[-1].copy())
            if dress is not None:
                dress.cover_frames(len(frames))
                hist.append({"animation": a, "held": 1})
                n_held += 1
                continue
            break
        # coarse-to-fine: sharpen the render targets late in the run (thin features)
        if (cfg.c2f_at > 0 and cfg.lambda_auto > 0
                and a == int(cfg.c2f_at * cfg.animations)):
            cfg.render_res = cfg.render_res_hi
            keep = (tgt.h1_scale, tgt.jd_scale, tgt.jd_rho0, tgt.gauss_scale,
                    tgt.kde_scale, tgt.unit_ratio, tgt.unit_grad_ratio)
            tgt = build_target(target_x, prm, cfg, w_tgt=w_tgt, w_src=w_src)
            # EVERY one-shot calibration survives the rebuild (REFUTE 2026-09-04 F1: a
            # fresh TargetPack has h1_scale=None, so the next window silently RE-
            # calibrated the H^-1 term at a mid-run state - a hidden weight schedule;
            # REFUTE 2026-09-15 F5: gauss_scale and kde_scale had the same hole, so the
            # hybrid render weight stepped at the c2f boundary of every gauss arm)
            (tgt.h1_scale, tgt.jd_scale, tgt.jd_rho0, tgt.gauss_scale,
             tgt.kde_scale, tgt.unit_ratio, tgt.unit_grad_ratio) = keep
            best_rend, stale = None, 0              # rescaled track must not inherit a
            outer_scales = outer_prev = outer_prev_phys = prev_disp = None
            outer_gate_latched = False              # render track rescaled: re-earn the latch
            if tgt.gauss is not None and cfg.gauss_children > 1:
                # REFUTE B5: the rebuild constructs a fresh GaussViews whose
                # source_offsets is None; the next child-render loss would raise.
                tgt.gauss.configure_source(src, surface_w > 0.5)
            hist.append({"animation": a, "c2f_render_res": cfg.render_res})   # near-full
            log(f"[v2] c2f at anim {a + 1}: render targets rebuilt at {cfg.render_res}px")
            # plateau counter (adversarial finding: freeze could fire one commit later)
        x_start = x.copy()
        rollback = {
            "st": {k: (None if q is None else q.copy()) for k, q in st.items()},
            "Fp": Fp.copy(), "s": None if s is None else s.copy(),
            "dfc": dfc_prev, "mom": mom_prev, "lam": balancer.lam,
            "frames": len(frames), "F_frames": len(F_frames), "guards": dict(guards),
            "Fg_commits": len(Fg_commits),
        }
        frontier = None
        if cfg.vol_frontier:
            # target cells within ONE loss cell of the current occupancy (3^3 dilation)
            from ..losses.volumetric import rasterize_mass
            with torch.no_grad():
                occ = rasterize_mass(torch.as_tensor(x_start, device=cfg.device), tgt.m,
                                     tgt.lgmin, tgt.ldx, tgt.ldims) > 1e-6
                occ = occ.reshape(1, 1, *tgt.ldims).float()
                frontier = torch.nn.functional.max_pool3d(occ, 3, stride=1, padding=1)
                frontier = frontier.reshape(-1)
        if cfg.bonds and coh_nbr is not None:
            # rest lengths: refresh only for particles coupled at this window's start; a
            # decoupled particle keeps the lengths from its last coupled state, so the
            # projection pulls it back into the body (v2 re-based everything and accepted
            # the separation — the census showed no return)
            d_now = np.linalg.norm(x_start[coh_nbr] - x_start[:, None, :], axis=2).astype(np.float32)
            frag_np = fragment_mask(x_start, prm)          # broken-off material this window
            if bond_rest is None:
                bond_rest = d_now
            else:
                bond_rest = np.where(frag_np[:, None], bond_rest, d_now).astype(np.float32)
            bond_frag = frag_np.astype(np.float32)
            if a % 10 == 0 or frag_np.any():
                log(f"[v2] anim {a + 1}: fragments {int(frag_np.sum())} particles")
        fr, F_seq, end, s, whist, stats = optimize_window(
            x_start, prm, cfg, tgt, balancer, F0=st["F"], Fp=Fp, v0=st["v"], C0=st["C"],
            s_init=s, dfc_init=dfc_prev, on_iter=on_iter, log=lambda *_: None,
            fill_bal=fill_balancer, alpha_scale=anneal, mom_init=mom_prev, vol0=vol0,
            surface_w=surface_w, Fg0=st.get("Fg"), coh_nbr=coh_nbr, coh_nbr_src=src,
            frontier=frontier, bond_rest=bond_rest, bond_frag=bond_frag,
            u_scale_init=(u_scale if getattr(cfg, "u_rprop", False) else None),
            ctrl_scale_init=(ctrl_scale_apply if (getattr(cfg, "ctrl_rprop", False) or getattr(cfg, "freeze_arrived", False)) else None),
            eta_init=settle_eta_arr, pin_init=settle_pin_arr)
        if a == 0 and stats.get("basis"):
            log(f"[v2] control basis: {stats['basis']}")
        if stats.get("cont_ratio") is not None and (stats.get("cont_rejects") or stats["cont_ratio"] > 1.0):
            log(f"[v2] anim {a + 1}: continuity rejects={stats['cont_rejects']} "
                f"max rel/lim={stats['cont_ratio']:.2f} (free rollout {stats['cont_ref_ratio']:.2f})")
        if cfg.warm_start:
            dfc_prev = stats.get("dfc")
        if not whist:
            if stats.get("grad_converged"):
                frozen = True                       # zero gradient at the start: at the optimum
                hist.append({"animation": a, "grad_converged": 1})
                log(f"[v2] anim {a + 1}: gradient converged at window start; holding still")
                continue
            # line-search exhaustion is NOT convergence (Codex stack-review f6: a hard
            # stop here bypassed patience — hero7_base truncated at anim 106). Null
            # commit: hold the state, let the patience counter decide the freeze.
            log(f"[v2] anim {a + 1}: no accepted step — null commit (stale {stale + 1})")
            frames.append(x.copy()); F_frames.append(F_frames[-1].copy())
            if dress is not None:
                dress.cover_frames(len(frames))
            hist.append({"animation": a, "null_commit": 1})
            stale += 1
            mom_prev = None
            reject_streak, last_reject_score = 0, None   # lineage changed: no replay across
                                                         # a null commit (REFUTE Opus F4)
            if cfg.anneal_stale > 0:
                # Retrying the identical state/moments at the identical alpha reproduces
                # the same line-search exhaustion forever.  Null commits must participate
                # in the same plateau schedule as rejected outer candidates.
                anneal = max(0.05, anneal * cfg.anneal_stale)
            if stale >= cfg.patience:
                frozen = True
                log(f"[v2] frozen after {cfg.patience} stale/null commits")
            continue

        if cfg.mom_carry > 0:            # only a committed window donates its moments
            mom_prev = stats.get("mom_out")
        # ---- FULL state promotion + guard counters (must stay zero, gate G2) ----
        x_new = np.ascontiguousarray(fr[-1], np.float32)
        n_out = int(((x_new < lo) | (x_new > hi)).any(1).sum())
        n_nan = int((~np.isfinite(x_new).all(1)).sum())
        x = np.clip(np.nan_to_num(x_new), lo, hi).astype(np.float32)
        # repair numerical pathologies ONLY (counted); no singular-value projection here
        Fc, n_bad, n_flip, _ = condition_F(end["F"], clamp=False)
        n_ns = int((~np.isfinite(end["v"]).all(1)).sum()
                   + (~np.isfinite(end["C"]).all(axis=(1, 2))).sum())
        v_p = np.nan_to_num(end["v"]).astype(np.float32)
        C_p = np.nan_to_num(end["C"]).astype(np.float32)
        # geometric F: a pure kinematic product of the (already guarded) C history; a
        # non-finite row is reset to the physics F and counted with the state guard
        Fg_p = None
        if end.get("Fg") is not None:
            Fg_p = np.ascontiguousarray(end["Fg"], np.float32)
            bad_g = ~np.isfinite(Fg_p).all(axis=(1, 2))
            if bad_g.any():
                Fg_p = Fg_p.copy()
                Fg_p[bad_g] = Fc[bad_g]
                n_ns += int(bad_g.sum())
        st = {"F": Fc, "v": v_p, "C": C_p, "Fg": Fg_p}
        # whole-window F health, not just the endpoint (an inversion mid-window that
        # recovers by T would otherwise be invisible)
        from ..mpm.conditioning import batched_det
        if end.get("n_inv_steps") is not None:
            n_inv = int(end["n_inv_steps"])            # counted on the device (2026-09-23 speed pass)
            jmin_traj = float(end["Jmin_traj"])
        else:
            dets = batched_det(np.stack(F_seq[1:]))    # one batched det over the window
            n_inv = int((dets <= 0.0).any(0).sum())
            jmin_traj = float(dets.min())
        guards["clamped"] += n_out; guards["nan_x"] += n_nan; guards["nan_state"] += n_ns
        guards["F_reset"] += n_bad; guards["F_flip"] += n_flip
        guards["F_invert_steps"] += n_inv

        # ---- LOCAL phase (local-global): band-limited surface GS pass on the render
        # residual, interior pinned to the global solution just promoted. Runs BEFORE
        # assimilation so one ratchet covers global+local. λ is calibrated by its OWN
        # balancer in the local energy's variable space (adversarial blocker). ----
        lg_tele = None
        if cfg.lg_sweeps > 0 and balancer.active:
            out = surface_local_pass(x, Fc, Fp, tgt, cfg, lg_balancer, prm)
            if out is not None:
                x_lg, F_lg, lg_tele = out
                n_lg_nan = int((~np.isfinite(x_lg).all(1)).sum())
                n_lg_out = int(((x_lg < lo) | (x_lg > hi)).any(1).sum())   # COUNT the clip
                n_inv2 = int((np.linalg.det(np.nan_to_num(F_lg)) <= 0).sum())  # pre-repair
                Fc2, n_b2, n_f2, _ = condition_F(F_lg, clamp=False)
                guards["nan_x"] += n_lg_nan; guards["clamped"] += n_lg_out
                guards["F_reset"] += n_b2; guards["F_flip"] += n_f2
                guards["F_invert_steps"] += n_inv2
                x = np.clip(np.nan_to_num(x_lg), lo, hi).astype(np.float32)
                Fc = Fc2
                st["F"] = Fc
                n_out += n_lg_out; n_nan += n_lg_nan
                n_bad += n_b2; n_flip += n_f2; n_inv += n_inv2

        # ---- plastic assimilation of the ELASTIC stretch (§3.5): F_e -> R_e S_e^{1-eta},
        # exact and per-particle — displacement-field estimation mismatched the
        # dFc-inflated F and spiked stress at every commit boundary (measured) ----
        if cfg.assim > 0:
            if cfg.w_grow > 0 and tgt.tmass3 is not None:
                from ..losses.volumetric import growth_demand
                from ..plasticity.assimilation import assimilate_growth
                dem = growth_demand(torch.as_tensor(x, device=cfg.device), tgt.m,
                                    tgt.tmass3, tgt.dtgmin, tgt.dtdx, tgt.dtdims,
                                    cfg.fill_sigma)
                Fp = assimilate_growth(Fc, Fp, eta=cfg.assim,
                                       smin=cfg.assim_smin, smax=cfg.assim_smax,
                                       isochoric=cfg.assim_iso,
                                       grow=1.0 + cfg.w_grow * dem,
                                       grow_band=cfg.grow_band)
            else:
                Fe_bar = None
                if cfg.assim_consensus:
                    # plasticity is a continuum property: the increment follows the STENCIL
                    # NEIGHBOURHOOD's elastic stretch (self excluded); a particle stretching
                    # away from material that is not stretching keeps that excess elastic
                    from ..plasticity.assimilation import consensus_elastic
                    Fe_bar = consensus_elastic(x, Fc, Fp, prm.grid_min, prm.dx,
                                               (prm.nx, prm.ny, prm.nz), device=cfg.device)
                Fp = assimilate_elastic(Fc, Fp, eta=cfg.assim,
                                        smin=cfg.assim_smin, smax=cfg.assim_smax,
                                        isochoric=cfg.assim_iso, Fe=Fe_bar)

        # F_g is NOT edited at commits (REFUTE-2 F11: a commit-time relaxation changed
        # the image with no particle motion and saturated the rendered anisotropy at
        # ~1.5 %). The needle-splat problem is handled in the render FORWARD MODEL
        # instead (cfg.gauss_cov_sat, gauss_loss.saturate_stretch), which is a fixed
        # function of the state at every step.
        # ---- conservative particle resampling (cfg.reattach): a particle whose grid cell is
        # not connected to the body (fragment_mask: components of the occupancy dilated by
        # one cell, i.e. the stencil's own interaction range) shares no node with any other
        # material point — it is not a continuum element any more, only a stray mass. It is
        # merged back onto the nearest body particle: position (plus half a spacing of
        # jitter), velocity, C, F, Fp, Fg copied. Mass is conserved (no particle is
        # deleted), every commit ends with zero fragments by construction, and the window
        # dynamics are untouched (this is resampling of a degenerate sampling, the same
        # remedy the MPM/PIC literature applies to under-sampled regions). ----
        n_reattached = 0
        if cfg.reattach:
            if getattr(tgt, "points_np", None) is None:
                tgt.points_np = np.ascontiguousarray(tgt.points.detach().cpu().numpy(), np.float32)
            n_reattached = reattach_fragments(x, v_p, C_p, Fc, Fp, Fg_p, prm,
                                              float(tgt.nn_spacing) if tgt.nn_spacing > 0 else 0.5 * prm.dx,
                                              seed=a + 1, tgt_points=tgt.points_np)
            if n_reattached:
                n_reattach_total += n_reattached
                log(f"[v2] anim {a + 1}: re-attached {n_reattached} grid-disconnected particles")
        # ---- DIAGNOSTIC (cfg.rebound_probe): the elastic rebound. One window of zero-control dynamics from
        # the commit state (after the assimilation; plain elastic body, no layer, no bonds); its
        # displacement projected on the committed window displacement. -0.5 = half of what the control
        # achieved springs back on its own (docs/experiments.md 2026-09-24, the spectral finding). ----
        rec_rebound = None
        if getattr(cfg, "rebound_probe", False):
            try:
                from ..mpm.constitutive import lame as _lame
                from ..mpm.function import RolloutSpec as _RS, warp_mpm as _wm
                _lam0, _mu0 = _lame(cfg.young, cfg.poisson)
                _N = len(x)
                _mref = int(getattr(cfg, "mass_ref_n", 0) or 0)
                _mass = (float(_mref) / float(_N)) if (_mref > 0 and _N != _mref) else 1.0
                _spec = _RS(x0=np.ascontiguousarray(x, np.float32), m=_mass, lam=_lam0, mu=_mu0, prm=prm, T=int(cfg.T),
                            Fp=np.ascontiguousarray(Fp, np.float32), v0=np.ascontiguousarray(v_p, np.float32),
                            F0=np.ascontiguousarray(Fc, np.float32), C0=np.ascontiguousarray(C_p, np.float32),
                            device=cfg.device, vol0=vol0)
                with torch.no_grad():
                    _xT, _ = _wm(torch.zeros(_N, 3, 3, device=cfg.device), _spec)
                    # the same from REST (v = 0, C = 0): the elastic-stress part of the free motion alone
                    _spec0 = _RS(x0=_spec.x0, m=_mass, lam=_lam0, mu=_mu0, prm=prm, T=int(cfg.T), Fp=_spec.Fp,
                                 v0=np.zeros_like(_spec.v0), F0=_spec.F0, C0=np.zeros_like(_spec.C0),
                                 device=cfg.device, vol0=vol0)
                    _xT0, _ = _wm(torch.zeros(_N, 3, 3, device=cfg.device), _spec0)
                _d_free = _xT.detach().cpu().numpy().astype(np.float32) - np.asarray(x, np.float32)
                _d_free0 = _xT0.detach().cpu().numpy().astype(np.float32) - np.asarray(x, np.float32)
                _d_prev = np.asarray(x, np.float32) - np.asarray(x_start, np.float32)
                _den = float((_d_prev * _d_prev).sum())
                _reb = float((_d_free * _d_prev).sum() / _den) if _den > 0 else float("nan")
                _reb0 = float((_d_free0 * _d_prev).sum() / _den) if _den > 0 else float("nan")
                _mf = float(np.median(np.linalg.norm(_d_free, axis=1))); _mp = float(np.median(np.linalg.norm(_d_prev, axis=1)))
                _mf0 = float(np.median(np.linalg.norm(_d_free0, axis=1)))
                rec_rebound = {"rebound": _reb, "rebound_rest": _reb0, "free_median": _mf, "free_rest_median": _mf0, "commit_median": _mp}
                log(f"[v2] anim {a + 1}: rebound probe — free displacement projected on the committed one "
                    f"{_reb:+.3f} (free median {_mf:.4f} wu, committed median {_mp:.4f} wu); from rest (v = C = 0): "
                    f"{_reb0:+.3f} (median {_mf0:.4f} wu)")
            except Exception as _e:
                log(f"[v2] anim {a + 1}: rebound probe failed: {_e}")
        # ---- the null-space projection of the window's displacement (cfg.commit_pic; mpm/gridfilter.py,
        # docs/method.md 10.20): x_end <- x_start + G2P(P2G(x_end - x_start)) with the simulation's cubic
        # stencil at the window-start positions; the grid-invisible part is dropped. Before the shift. ----
        if getattr(cfg, "commit_pic", False):
            from ..mpm.gridfilter import grid_project
            _d = np.asarray(x, np.float32) - np.asarray(x_start, np.float32)
            _pd, _ps = grid_project(_d, np.asarray(x_start, np.float32), float(prm.dx), prm.grid_min,
                                    (prm.nx, prm.ny, prm.nz), device=cfg.device)
            x = (np.asarray(x_start, np.float32) + _pd).astype(np.float32)
            rec_pic = _ps
            if (a + 1) % 10 == 1 or _ps["null_share"] > 0.5:
                log(f"[v2] anim {a + 1}: commit projection — null-space share of the window's displacement "
                    f"{100 * _ps['null_share']:.1f} % (removed median {_ps['removed_median']:.4f} wu of "
                    f"{_ps['d_median']:.4f})")
        else:
            rec_pic = None
        # ---- Fickian shifting of the sub-cell arrangement (cfg.shift_sub; mpm/shifting.py, docs/method.md
        # 10.18): the quadrature below the cell is a null space of the objective and nothing else orders
        # it; one explicit diffusion step of the particle concentration at its stability limit, positions
        # only, the outer layer tangentially. Applied to the COMMIT state, so the archived frame and the
        # next window's start are the ordered cloud. ----
        if getattr(cfg, "shift_sub", False):
            from ..mpm.shifting import fickian_shift, native_spacing
            if sp_native is None:
                sp_native = native_spacing(x)
            dxs, sst = fickian_shift(x, sp_native, h_sp=float(cfg.shift_h_sp))
            # the objective's change across the shift, measured (the arrangement is underconstrained,
            # not an exact null space — the independent audit, docs/diagnosis_300k_20260923.md): the
            # fixed-target cell sum before and after, in density units when the run uses them
            _dv0 = _dv1 = None
            try:
                with torch.no_grad():
                    from ..losses.volumetric import d_vol_density as _dvd_s
                    _xt0 = torch.as_tensor(x, device=cfg.device)
                    _xt1 = torch.as_tensor(x + dxs, device=cfg.device)
                    if cfg.loss_units == "density":
                        _dv0 = float(_dvd_s(_xt0, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims, tgt.m_ref, tgt.n_support))
                        _dv1 = float(_dvd_s(_xt1, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims, tgt.m_ref, tgt.n_support))
                    else:
                        _dv0 = float(d_vol(_xt0, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims))
                        _dv1 = float(d_vol(_xt1, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims))
            except Exception:
                pass
            x += dxs
            sst["dvol_rel"] = ((_dv1 - _dv0) / max(abs(_dv0), 1e-12)) if (_dv0 is not None and _dv1 is not None) else float("nan")
            last_shift = sst                                 # attached to this window's record below
            if (a + 1) % 10 == 1 or sst["p99_sp"] > 0.25:
                log(f"[v2] anim {a + 1}: sub-cell shift median {sst['median_sp']:.3f} sp, p99 {sst['p99_sp']:.3f}, "
                    f"max {sst['max_sp']:.2f}; disorder |grad C| h {sst['disorder']:.3f}; tangential on {sst['n_surface']}; "
                    f"cell-sum change across the shift {100 * sst['dvol_rel']:+.3f} %")
        # archive the PROMOTED states (identical to raw when no guard fired)
        ks = max(1, int(cfg.archive_stride))            # archive stride (150k archives)
        frames.extend(f.copy() for f in fr[1:-1][::ks]); frames.append(x.copy())
        F_frames.extend(F_seq[1:-1][::ks]); F_frames.append(Fc.copy())
        if Fg_p is not None:
            Fg_commits.append((len(frames), Fg_p.copy()))

        w = whist[-1]
        if getattr(cfg, "phys_loss", "density") != "density":
            # the optimiser's d_vol under a transport recipe is the loss against the PACED
            # (or transport) target — near zero by construction and hypersensitive in
            # relative terms; the merit, the brake and the convergence tracker must read the
            # cell sum against the FIXED target (2026-09-17 C forensic: the paced value
            # 0.0049 -> 0.0055 read as a 9 % merit regression and froze every ot_pace run)
            with torch.no_grad():
                xt = torch.as_tensor(x, device=cfg.device)
                w = dict(w)
                if cfg.loss_units == "density":
                    from ..losses.volumetric import d_vol_density as _dvd
                    w["d_vol"] = float(_dvd(xt, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims,
                                            tgt.m_ref, tgt.n_support))
                else:
                    w["d_vol"] = float(d_vol(xt, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims))
        # after a local pass the archived state differs from the window's last iterate —
        # the freeze and the logged trace must describe the ARCHIVED state (adversarial
        # finding), so recompute the data terms on the final x
        if lg_tele is not None:
            with torch.no_grad():
                xt = torch.as_tensor(x, device=cfg.device)
                w = dict(w)
                w["d_vol"] = float(d_vol(xt, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx,
                                         tgt.ldims))
                if tgt.sils is not None:
                    w["d_render"] = float(d_render(xt, tgt.sils, tgt.views,
                                                   cfg.render_res, tgt.extent, cfg.sil_k,
                                                   cfg.w_hole, cfg.w_spray))
        d_dt = None
        if tgt.dt3 is not None:      # W1 term on the ARCHIVED state — the freeze track
            with torch.no_grad():    # must see it (Codex finding 6). UNGATED on purpose
                # (stack-review f13: a per-window gate makes the track fall when the
                # GATE dies rather than when particles move — the §7.4 pathology in
                # the convergence signal): a gate-independent geometric statistic.
                xt = torch.as_tensor(x, device=cfg.device)
                d_dt = float(d_w1(xt, tgt.m, tgt.dt3,
                                  tgt.dtgmin, tgt.dtdx, tgt.dtdims))
        d_jd_v = None
        if cfg.w_jdens > 0 and tgt.jd_rho0 is not None:  # density-J prior track (archived)
            with torch.no_grad():
                from ..losses.volumetric import d_jdens as _dj
                xt = torch.as_tensor(x, device=cfg.device)
                d_jd_v = float(_dj(xt, tgt.m, tgt.jd_rho0, tgt.jd_gmin, tgt.jd_dx, tgt.jd_dims))
        d_h1_v = None
        if cfg.w_h1 > 0:                                  # non-local mass-balance track
            with torch.no_grad():
                xt = torch.as_tensor(x, device=cfg.device)
                d_h1_v = float(d_h1(xt, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims))
        d_kde_v = None
        if cfg.w_kde > 0 and tgt.pts is not None:       # particle-scale density track
            with torch.no_grad():                        # (fresh neighbours, ungated)
                from ..losses.volumetric import d_kde as _dk, kde_assign as _ka
                xt = torch.as_tensor(x, device=cfg.device)
                nb = _ka(xt, tgt.pts, cfg.kde_k)
                d_kde_v = float(_dk(xt, tgt.pts, nb, tgt.kde_h, tgt.kde_rho_ref))
        d_fill = None
        if cfg.w_fill > 0 and tgt.tmass3 is not None:   # fill telemetry (f9/F9: the
            with torch.no_grad():                        # term was fully unobservable)
                xt = torch.as_tensor(x, device=cfg.device)
                d_fill = coverage_shortfall(xt, tgt.m, tgt.tmass3, tgt.dtgmin,
                                            tgt.dtdx, tgt.dtdims, cfg.fill_sigma)
        rec = {"animation": a, "iters": len(whist), "loss": w["loss"], "d_vol": w["d_vol"],
               "reattached": n_reattached,
               "shift_median_sp": (last_shift or {}).get("median_sp"), "shift_dvol_rel": (last_shift or {}).get("dvol_rel"),
               "pic_null_share": (rec_pic or {}).get("null_share"),
               "pace_proj_div0": (stats.get("pace_proj") or {}).get("div0"), "pace_proj_div1": (stats.get("pace_proj") or {}).get("div1"), "pace_proj_corr": (stats.get("pace_proj") or {}).get("corr_med"),
               "rebound": (rec_rebound or {}).get("rebound"),
               "grad_norm": w.get("grad_norm"), "d_pbr": w.get("d_pbr"), "d_dt": d_dt,
               "d_sil": w.get("d_sil"), "d_gauss": w.get("d_gauss"), "d_kde": d_kde_v,
               "d_jdens": d_jd_v, "d_h1": d_h1_v, "h1_ratio": stats.get("h1_ratio"),
               "d_fill": d_fill, "g_cos": stats.get("g_cos"),
               "g_raw_cos": stats.get("g_raw_cos"), "g_share": stats.get("g_share"),
               "g_phys_norm": stats.get("g_phys_norm"), "g_rend_norm": stats.get("g_rend_norm"),
               "cont_ratio": stats.get("cont_ratio"), "cont_rejects": stats.get("cont_rejects"),
               "cont_ref_ratio": stats.get("cont_ref_ratio"),
               "render_work": stats.get("render_work"),
               "render_work_x": stats.get("render_work_x"),
               "render_work_F": stats.get("render_work_F"),
               "phys_work": stats.get("phys_work"),
               "phys_work_x": stats.get("phys_work_x"),
               "phys_work_F": stats.get("phys_work_F"),
               "phys_work_v": stats.get("phys_work_v"),
               "step_norm": stats.get("step_norm"),
               "render_cos": stats.get("render_cos"), "phys_cos": stats.get("phys_cos"),
               "predicted_decrease": stats.get("predicted_decrease"),
               "fill_lam": stats.get("fill_lam"),
               "kin": w["kin"], "kin_run": w.get("kin_run"), "kin_var": w.get("kin_var"),
               "alpha_last": w.get("alpha"),
               "d_render": w["d_render"], "lambda": w["lambda"],
               "lambda_capped": stats.get("lambda_capped"),
               "F_kind": "geom" if Fg_p is not None else "physics",
               "dfc_absmax": w["dfc_absmax"], "s_absmax": w["s_absmax"],
               "accepted": stats["accepted"], "rejected": stats["rejected"],
               "v_absmax": float(np.abs(v_p).max()),
               "v_mean": float(np.linalg.norm(v_p, axis=1).mean()),
               "move": float(np.linalg.norm(x - x_start, axis=1).mean()),
               "Jmin": float(batched_det(Fc).min()),
               "Jmin_traj": jmin_traj,
               "clamped": n_out, "nan_x": n_nan, "nan_state": n_ns,
               "F_reset": n_bad, "F_flip": n_flip, "F_invert_steps": n_inv}
        if tgt.gauss is not None:
            from .gauss_loss import gaussian_shape_diagnostics
            rec.update(gaussian_shape_diagnostics(          # on the RENDERED F (F7)
                torch.as_tensor(Fc if Fg_p is None else Fg_p, device=cfg.device),
                tgt.gauss.primitive_sigma,
                reference_spacing=tgt.nn_spacing if tgt.nn_spacing > 0 else None,
                sat=cfg.gauss_cov_sat))
        if lg_tele is not None:
            rec.update(lg_tele)

        # Fixed-scale outer trust gate.  The inner objective contains an adaptive
        # render lambda, so it cannot safely decide whether a whole physical state
        # should be committed across windows.  Normalize each raw channel once per
        # target resolution and require monotone progress in that fixed merit.
        phys_track = rec["d_vol"] + cfg.w_kin * rec["kin"] / (
            tgt.unit_ratio if cfg.loss_units == "density" else 1.0)
        # Fixed merit = SHAPE terms only. b7 forensic (gate v2, 450 paced anims):
        # with d_vol + w_kin*kin in the merit, any motion from a settled state
        # raised kin enough for a >5% merit regression, so the brake rejected
        # 333/450 candidates and pinned the run in place - pacing demands motion,
        # a kinetic merit punishes it. Terminal velocity is a transient, not a
        # shape quality; the brake still catches real regressions (d_vol 62->215).
        components = {"phys": rec["d_vol"]}
        if getattr(cfg, "phys_loss", "density").startswith("ot") and getattr(tgt, "ot_pull", None) is not None:
            # transport recipes: the merit's physics component is the Sinkhorn divergence
            # to the fixed target (what the recipe descends; monotone along a transport
            # path where the cell sum plateaus — C forensic 2026-09-17), the cell sum stays
            # in the record for the tracker and the report
            with torch.no_grad():
                rec["ot_div"] = float(tgt.ot_pull.divergence(torch.as_tensor(x, device=cfg.device), cfg.ot_samples))
            components["phys"] = rec["ot_div"]
        rend_gate = (rec["d_sil"] if rec.get("d_sil") is not None
                     else rec["d_render"])
        if rend_gate is not None:
            components["render"] = rend_gate     # merit reads d_sil only (B1)
        if d_dt is not None:
            components["dt"] = d_dt
        if d_fill is not None:
            components["fill"] = d_fill
        if d_kde_v is not None:
            components["kde"] = d_kde_v
        if d_jd_v is not None:
            components["jdens"] = d_jd_v
        if d_h1_v is not None:
            components["h1"] = d_h1_v
        # mass-ejection veto: isolated-particle count must not increase over a window
        eject_reject = False
        if cfg.eject_veto:
            iso_r = float(cfg.eject_iso_k) * float(max(tgt.nn_spacing, 1e-6))
            iso_cand = _iso_count(x, iso_r)
            iso_start = _iso_count(x_start, iso_r)
            rec.update({"iso_count": iso_cand, "iso_start": iso_start})
            eject_reject = iso_cand > iso_start
        disp = (x - x_start).reshape(-1)
        reversal_cos = None
        if prev_disp is not None:
            reversal_cos = float(np.dot(disp, prev_disp) /
                                 max(np.linalg.norm(disp) * np.linalg.norm(prev_disp), 1e-12))
        rec["reversal_cos"] = reversal_cos
        # a reversal by the gate's own definition (cos below outer_reversal_cos): r300 armed on two
        # near-zero cosines (−0.08, −0.01) twelve windows before l300's alternation and stopped short
        rev_neg_now = reversal_cos is not None and float(reversal_cos) < float(cfg.outer_reversal_cos)
        # the reversal read on the ARRIVED particles only (the paced target's mask; the whole body when
        # fewer than half have arrived): the global cosine reverses transiently on long curved transports
        # (nefertiti, C in g41x) and a global hold engaged there cannot finish the transport
        reversal_cos_arr = reversal_cos
        _am = stats.get("arrived_mask")
        if prev_disp is not None and _am is not None and len(_am) * 3 == len(disp) and float(np.mean(_am)) >= 0.5:
            _m3 = np.repeat(np.asarray(_am, bool), 3)
            _da, _pa = disp[_m3], prev_disp[_m3]
            reversal_cos_arr = float(np.dot(_da, _pa) / max(np.linalg.norm(_da) * np.linalg.norm(_pa), 1e-12))
        rec["reversal_cos_arrived"] = reversal_cos_arr
        rev_neg_now_arr = reversal_cos_arr is not None and float(reversal_cos_arr) < float(cfg.outer_reversal_cos)

        # λ-free plateau tracks, evaluated BEFORE the outer gate: "no track improved"
        # is this pipeline's validated definition of near-stationarity (driver #4 —
        # at small move the residual motion is honest descent UNTIL these tracks
        # stall). The previous latch trigger (normalized merit <= outer_gate_merit_max
        # at small move) fired at anim ~20 of 300 — pace + a large w_kin make moves
        # small long before the descent is done — and then rejected every window to
        # a fake "converged" at anim 27 (final_hires20k_child4_latched forensic).
        rend_track = (rec["d_sil"] if rec.get("d_sil") is not None
                      else rec["d_render"])    # gates read the PURE silhouette (B1)
        improved = best_phys is None or phys_track < best_phys - cfg.tol * abs(best_phys)
        if rend_track is not None and best_rend is not None:
            improved = improved or rend_track < best_rend - cfg.tol * abs(best_rend)
        if d_dt is not None and best_dt is not None:     # own track + tolerance (Opus f4)
            improved = improved or d_dt < best_dt - cfg.tol * abs(best_dt)
        if d_fill is not None and best_fill is not None:  # fill track (stack-review F9)
            improved = improved or d_fill < best_fill - cfg.tol * abs(best_fill)
        if d_kde_v is not None and best_kde is not None:  # particle-scale density track
            improved = improved or d_kde_v < best_kde - cfg.tol * abs(best_kde)
        if d_jd_v is not None and best_jd is not None:    # density-J prior track
            improved = improved or d_jd_v < best_jd - cfg.tol * abs(best_jd)
        if d_h1_v is not None and best_h1 is not None:    # mass-balance track
            improved = improved or d_h1_v < best_h1 - cfg.tol * abs(best_h1)
        if stats.get("pace_bound"):
            # The window exited via the PACE FLOOR: it did exactly the scheduled
            # work, by construction. Plateau accounting must not run against a
            # glidepath (b0 forensic: the validated flagship froze at anim 70/300
            # while descending perfectly on schedule, rev-cos +1.000 - the tol
            # thresholds were calibrated for unpaced 5-12%/commit descent).
            # ...unless the commit REGRESSED a lambda-free track versus the previous
            # accepted commit: the inner L is measured from the window's free
            # rollout, so a 12% cut is easy while the body runs away (r1: 20
            # commits of all-track regression were marked 'improved' and the
            # freeze fired only at a112 with d_vol 82 -> 489).
            regressed = False
            if prev_tracks is not None:
                for k, v in (("phys", phys_track), ("rend", rend_track), ("dt", d_dt)):
                    pv = prev_tracks.get(k)
                    if v is not None and pv is not None and v > pv * (1 + cfg.tol):
                        regressed = True
            if not regressed:
                improved = True
        prev_tracks = {"phys": phys_track, "rend": rend_track, "dt": d_dt}

        outer_reject = False
        outer_gain = None
        if cfg.outer_merit:
            if outer_scales is None:
                outer_scales = {k: max(abs(v), 1e-8) for k, v in components.items()}
            score = float(sum(v / outer_scales[k] for k, v in components.items()))
            # the PHYSICS part of the merit (every component but the render term): what the
            # catastrophe brake below watches (docs/surface_gradient.md 15d, 2026-09-22: under the
            # transport gate of u the arriving front regressed nefertiti's silhouette term by 25 %
            # in one window while the transport improved 10 %; the full-merit brake read it as a
            # runaway and the run died at anim 16 — a render transient is not a physics runaway)
            # 15f (2026-09-22 22:00): the brake reads the PRIMARY objective alone — the transport
            # divergence (or the cell sum) the recipe descends. A runaway regresses it (d_vol
            # 62 -> 215); a transient the trajectory must pass through does not: nefertiti's
            # arriving front spills outside the outline (the stray term d_dt +65 % in one window)
            # while the divergence improves 10 % — under u off the same spill is accepted at
            # -1 % a window and the run recovers to 0.961; rejecting it three times froze the run
            # at anim 16 (the candidate cannot avoid the state the transport passes through)
            score_phys = float(components["phys"] / outer_scales["phys"])
            phys_gain = None
            if outer_prev is not None:
                outer_gain = (outer_prev - score) / max(abs(outer_prev), 1e-8)
                phys_gain = ((outer_prev_phys - score_phys) / max(abs(outer_prev_phys), 1e-8)
                             if outer_prev_phys is not None else outer_gain)
                # Latch evidence must be a SUSTAINED plateau (stale>=2: third
                # consecutive non-improving commit). "small move" was retired as
                # a criterion twice over: with a large w_kin it is reachable at
                # 10% of the descent (s1), and under glidepath pacing every
                # window's move is small BY DESIGN, so one noisy no-improvement
                # commit armed the latch at anim ~59/300 (s3_paced).
                near_stationary = (not improved) and stale >= 2
                # the alternation's onset (10.21 addendum 3): two accepted commits in a row reversing
                # each other is the near-stationarity the tracks cannot see — their up-swings set new
                # bests by a fraction of a per-mille each cycle and disarm the gate for 30-40 windows
                # config.ctrl_rprop_hold_onset: the global step's hold (ctrl_rprop_hold) engages only from the
                # alternation's onset — the transport keeps the growing step (g41u: a hold from the start
                # ended C / beast / nefertiti / V early), the tail gets the held one
                if (getattr(cfg, "ctrl_rprop_hold_onset", False) and rev_neg_now_arr and rev_prev_neg_arr
                        and not getattr(cfg, "ctrl_rprop_hold", False)):
                    cfg.ctrl_rprop_hold = True
                    log(f"[v2] anim {a + 1}: the global step held from here on (two accepted commits reversing in a row "
                        f"on the arrived particles, cos {float(reversal_cos_arr):.2f})")
                if (getattr(cfg, "outer_latch_reversal", False) and rev_neg_now and rev_prev_neg_acc
                        and not outer_latched_rev):
                    outer_latched_rev = True
                    log(f"[v2] anim {a + 1}: outer gate armed at the alternation's onset (two accepted commits "
                        f"reversing in a row, cos {float(reversal_cos):.2f}) — low-gain reversals are rejected from here on")
                outer_gate_latched = outer_gate_latched or near_stationary or outer_latched_rev
                if improved and not stats.get("pace_bound") and not outer_latched_rev:
                    # A REAL track improvement is plateau evidence gone: a latch
                    # armed by a mid-run 3-commit stall self-heals instead of
                    # freezing the run (b6: latched at ~a90 of 450, 5 low-gain
                    # rejects consumed patience while d_vol was still at 182).
                    outer_gate_latched = False
                outer_reject = outer_gate_latched and outer_gain < cfg.outer_merit_tol
                # catastrophe brake, latched or not: pace bounds the INTENDED
                # per-window change, so a fixed-merit regression beyond one pace
                # budget is a runaway (s1: a18 committed at gain=-0.45 and the
                # freeze then held the damaged state; b4 forensic: flat-valley
                # limit cycle, overshoot windows d_vol 62->215 with kin spikes),
                # never a legitimate trade.
                # 2026-09-23 09:40: the primary-objective brake (15f) is the recipe again — the user chose
                # g41 once the video wipes were traced to the surface tracking, not the gate
                brake_reject = phys_gain < -max(cfg.pace, 0.05)
                if eject_reject:
                    # the window launched a particle: discard it like an insane
                    # candidate (shrink the step, cold restart), never commit it
                    brake_reject = True
                    rec["eject_reject"] = 1
                if brake_reject:
                    outer_reject = True
                if ((outer_gate_latched or cfg.outer_reversal_always)
                        and reversal_cos is not None
                        and reversal_cos < cfg.outer_reversal_cos
                        and outer_gain < cfg.outer_reversal_gain):
                    outer_reject = True
            rec.update({"outer_merit": score, "outer_gain": outer_gain,
                        "outer_merit_phys": score_phys, "phys_gain": phys_gain,
                        "reversal_cos": reversal_cos,
                        "outer_gate_latched": int(outer_gate_latched),
                        "outer_accepted": 0 if outer_reject else 1})
            if outer_reject:
                # Undo every mutation made after the window start, including plastic
                # assimilation and optimizer/balancer state.  Rejected trial frames
                # never enter the deliverable trajectory.
                x = x_start
                st = rollback["st"]
                Fp, s = rollback["Fp"], rollback["s"]
                balancer.lam = rollback["lam"]
                # A rejected candidate's LINEAGE is not retried: restoring the
                # warm-start control + moments re-runs the window deterministically
                # and rejection becomes an absorbing state (s4: a62-a69 produced
                # the identical bad candidate to 2 decimals, 8 rejects, frozen).
                # Cold restart gives the next window a genuinely different path.
                dfc_prev, mom_prev = None, None
                del frames[rollback["frames"]:]
                del F_frames[rollback["F_frames"]:]
                del Fg_commits[rollback["Fg_commits"]:]
                if dress is not None:
                    dress.truncate(rollback["frames"])
                guards = rollback["guards"]
                rec.update({"null_commit": 1, "outer_rejected": 1,
                            "brake_reject": int(brake_reject)})
                hist.append(rec)
                # Reject-type split (b4/b6 forensics): a BRAKE reject is one
                # insane candidate, not a stalled run - discard it, shrink the
                # step, keep descending; only LATCHED low-gain rejects are
                # plateau evidence and feed the patience freeze.
                reject_streak += 1
                # v5b forensic: a brake reject followed by a cold restart at the
                # anneal floor REPLAYS the identical candidate (gain -1.15 for 90
                # consecutive windows, rejected every time, no patience consumed:
                # the run could not end). One insane candidate is discarded for
                # free; a REPLAY - the same merit as the previous rejected
                # candidate within the replay-noise tolerance - is a fixed point
                # and must feed the freeze. (REFUTE 2026-09-04 F4: a bare streak
                # count would also charge patience for DISTINCT over-threshold
                # attempts mid-descent, the b4/b6 failure class.)
                replay_tol = max(cfg.outer_merit_tol,
                                 10.0 * float(stats.get("replay_rel", 0.0) or 0.0))
                replay = (last_reject_score is not None and
                          abs(score - last_reject_score)
                          <= replay_tol * max(abs(last_reject_score), 1e-8))
                last_reject_score = score
                rec.update({"reject_streak": reject_streak, "replay": int(replay)})
                if not brake_reject or replay:
                    stale += 1
                if cfg.anneal_stale > 0:
                    anneal = max(0.05, anneal * cfg.anneal_stale)
                if on_commit is not None:
                    F_hold = _id(N) if st["F"] is None else st["F"]
                    v_hold = np.zeros_like(x) if st["v"] is None else st["v"]
                    on_commit(a, x, F_hold, v_hold, rec)
                log(f"[v2] anim {a + 1}: outer merit rejected candidate "
                    f"(gain={outer_gain:.3g}, physics gain={phys_gain:.3g}, reversal={reversal_cos}"
                    f"{', EJECTION ' + str(rec.get('iso_start')) + '->' + str(rec.get('iso_count')) if eject_reject else ''})")
                if stale >= cfg.patience:
                    frozen = True
                if getattr(cfg, "reject_stop", 0) > 0 and reject_streak >= cfg.reject_stop:
                    # early stop: three consecutive rejected candidates of any kind are the
                    # plateau (v7: all terminal streaks; the C replayed a rejected step 12x)
                    log(f"[v2] anim {a + 1}: {reject_streak} consecutive rejected candidates -> "
                        f"early stop at the best commit")
                    frozen = True
                continue
            outer_prev = score
            outer_prev_phys = score_phys
            prev_disp = disp.copy()
            rev_prev_neg_acc = rev_neg_now; rev_prev_neg_arr = rev_neg_now_arr
        else:
            prev_disp = disp.copy()
            rev_prev_neg_acc = rev_neg_now; rev_prev_neg_arr = rev_neg_now_arr
        rec["frame_end"] = len(frames)          # archive index after this commit
        if dress is not None:
            # Tier D post-gate solve (design §4.3): runs only on ACCEPTED commits,
            # on the promoted terminal state; feeds no gate (B1 split) — its
            # telemetry rides in rec for the record only.
            from .dressing import solve_dressing
            rec.update(solve_dressing(dress, x, Fc, cfg.local_dress_iters,
                                      cfg.ls_noise_rel))
            dress.commit_snapshot(len(frames))
        hist.append(rec)
        if on_commit is not None:
            # the viewer renders the GEOMETRIC F when it exists (Sigma = s0^2 Fg Fg^T,
            # PhysGaussian kinematics); F_frames keeps the physics F for metrics
            on_commit(a, x, Fc if Fg_p is None else Fg_p, v_p, rec)

        # ---- plateau freeze on RAW components (λ-free; stops post-convergence sloshing).
        # `improved` was computed above, against the pre-commit bests; the bests only
        # absorb ACCEPTED commits (a gate-rejected candidate must not raise the bar). ----
        best_phys = phys_track if best_phys is None else min(best_phys, phys_track)
        if rend_track is not None:
            best_rend = rend_track if best_rend is None else min(best_rend, rend_track)
        if d_dt is not None:
            best_dt = d_dt if best_dt is None else min(best_dt, d_dt)
        if d_fill is not None:
            best_fill = d_fill if best_fill is None else min(best_fill, d_fill)
        if d_kde_v is not None:
            best_kde = d_kde_v if best_kde is None else min(best_kde, d_kde_v)
        if d_jd_v is not None:
            best_jd = d_jd_v if best_jd is None else min(best_jd, d_jd_v)
        if d_h1_v is not None:
            best_h1 = d_h1_v if best_h1 is None else min(best_h1, d_h1_v)
        reject_streak, last_reject_score = 0, None
        stale = 0 if improved else stale + 1
        if cfg.anneal_stale > 0:     # optimizer-side zigzag damping (docs/oscillation.md)
            anneal = (min(1.0, anneal * (1.0 if getattr(cfg, "ctrl_rprop_hold", False) and getattr(cfg, "ctrl_rprop", False) else 1.15)) if improved
                      else max(0.05, anneal * cfg.anneal_stale))
        if (cfg.anneal_on_reversal > 0 and reversal_cos is not None
                and reversal_cos < cfg.outer_reversal_cos):
            anneal = max(0.05, anneal * cfg.anneal_on_reversal)   # sign-change step control
        rec.update({"improved": int(bool(improved)), "stale": stale, "anneal": anneal,
                    "pace_bound": int(bool(stats.get("pace_bound")))})  # freeze forensics
        if stale >= cfg.patience:
            frozen = True
            log(f"[v2] converged at anim {a + 1} (phys={phys_track:.4f}); holding still")
        # ---- net displacement against summed displacement over the last `patience` windows (config
        # stop_on_cycle; docs/oscillation.md Addendum 9): the outer layer's window-to-window breathing
        # has a net/summed ratio near 0, honest descent near 1, a random walk 1/sqrt(k). Logged every
        # window; a convergence trigger when the flag is on. ----
        if cyc_sub is None:
            cyc_sub = np.sort(np.random.default_rng(0).choice(len(x), min(len(x), 20000), replace=False))
        cyc_hist.append(np.asarray(x[cyc_sub], np.float32).copy())
        k_cyc = max(int(cfg.patience), 2)
        if len(cyc_hist) > k_cyc + 1:
            del cyc_hist[0]
        if len(cyc_hist) == k_cyc + 1:
            summed = np.sum([np.linalg.norm(cyc_hist[i + 1] - cyc_hist[i], axis=1) for i in range(k_cyc)], axis=0)
            net = np.linalg.norm(cyc_hist[-1] - cyc_hist[0], axis=1)
            net_ratio = float(np.median(net) / max(float(np.median(summed)), 1e-12))
            rec["net_ratio"] = net_ratio
            # the same patience as the stale rule: `patience` consecutive windows at or below the
            # random-walk bound (late honest descent hovers just above it — c300 windows 32–60 at
            # 0.41–0.68 against 0.447 — and a single dip must not freeze the run)
            cyc_stale = cyc_stale + 1 if net_ratio <= 1.0 / np.sqrt(k_cyc) else 0
            rec["cyc_stale"] = cyc_stale
            if getattr(cfg, "stop_on_cycle", False) and not frozen and cyc_stale >= k_cyc:
                frozen = True
                log(f"[v2] converged at anim {a + 1}: net / summed displacement over {k_cyc} windows "
                    f"{net_ratio:.3f} <= random walk {1.0 / np.sqrt(k_cyc):.3f} for {cyc_stale} windows "
                    f"(the tail breathes without progress); holding still")
        # ---- sign-history damping of the u channel (config.u_rprop; docs/method.md 10.19): after an
        # ACCEPTED window, a particle whose u flipped sign against the previous accepted window has
        # its bound halved (Rprop eta- = 0.5), one that kept its sign has it raised x1.2 up to the full
        # spacing; floor 0.05; particles with u = 0 (off the layer) keep their scale. ----
        if getattr(cfg, "u_rprop", False) and stats.get("u_final") is not None:
            u_now = np.asarray(stats["u_final"], np.float32)
            if u_scale is None or len(u_scale) != len(u_now):
                u_scale = np.ones(len(u_now), np.float32)
            if u_prev is not None and len(u_prev) == len(u_now):
                prod = u_now * u_prev
                flip, same = prod < 0, prod > 0
                u_scale[flip] *= 0.5
                u_scale[same] = np.minimum(1.0, u_scale[same] * 1.2)
                np.clip(u_scale, float(getattr(cfg, "u_rprop_floor", 0.05)), 1.0, out=u_scale)
                act = u_now != 0
                if act.any():
                    rec["u_flip_frac"] = float(flip[act].mean())
                    rec["u_scale_med"] = float(np.median(u_scale[act]))
                    if (a + 1) % 10 == 0:
                        log(f"[v2] anim {a + 1}: u sign flips {100 * flip[act].mean():.0f} % of the layer, "
                            f"u bound scale median {np.median(u_scale[act]):.2f}, "
                            f"at the floor {100 * (u_scale[act] <= 0.05).mean():.0f} %")
            u_prev = u_now
        # ---- per-particle Rprop on the control step (config.ctrl_rprop; docs/method.md 10.24): after an
        # ACCEPTED window, a particle whose window displacement reversed its previous accepted one has its
        # control step halved (Rprop eta- = 0.5), one that kept its direction has it raised x1.2 up to 1;
        # no floor — the alternation's amplitude is the floor step, and a particle still in transport
        # never reverses. Particles that did not move (either displacement below 1e-4 cell) keep theirs. ----
        if getattr(cfg, "ctrl_rprop", False):
            _d_now = np.asarray(x, np.float32) - np.asarray(x_start, np.float32)
            # config.ctrl_rprop_smooth (10.24, second form): the reversal is read on the displacement averaged
            # over the material neighbourhood (the bond / coherence kNN, frozen at the source) and the scale
            # applied is the neighbourhood mean of the per-particle scales — per-particle scales alone made
            # neighbouring particles' control updates differ by orders of magnitude (ac300: det F 0.39
            # against 0.66), sub-cell control noise the creg term exists to forbid
            _nbr = None
            if getattr(cfg, "ctrl_rprop_smooth", False):
                _kk = int(getattr(cfg, "ctrl_rprop_k", 0) or 0)
                if _kk <= 0 and coh_nbr is not None and len(coh_nbr) == len(_d_now):
                    _nbr = coh_nbr
                else:
                    _kk = _kk if _kk > 0 else 24
                    if not hasattr(run_pipeline, "_rprop_nbr") or run_pipeline._rprop_nbr.shape != (len(_d_now), _kk):
                        from scipy.spatial import cKDTree as _KDn
                        run_pipeline._rprop_nbr = _KDn(src).query(src, k=_kk + 1, workers=-1)[1][:, 1:]
                    _nbr = run_pipeline._rprop_nbr
            def _smooth(v):
                if _nbr is None:
                    return v
                if v.ndim == 1:
                    return (v + v[_nbr].sum(1)) / float(_nbr.shape[1] + 1)
                return (v + v[_nbr].sum(1)) / float(_nbr.shape[1] + 1)
            if ctrl_scale is None or len(ctrl_scale) != len(_d_now):
                ctrl_scale = np.ones(len(_d_now), np.float32)
                ctrl_rev_count = np.zeros(len(_d_now), np.int32); frozen_p = np.zeros(len(_d_now), bool)
            if ctrl_prev_disp is not None and len(ctrl_prev_disp) == len(_d_now):
                _dn_s, _dp_s = _smooth(_d_now), _smooth(ctrl_prev_disp)
                _n0 = np.linalg.norm(_dn_s, axis=1); _n1 = np.linalg.norm(_dp_s, axis=1)
                _tiny = 1e-4 * float(prm.dx)
                _act = (_n0 > _tiny) & (_n1 > _tiny)
                _cos = (_dn_s * _dp_s).sum(1) / np.maximum(_n0 * _n1, 1e-30)
                _flip = _act & (_cos < 0.0); _same = _act & (_cos >= 0.0)
                # config.ctrl_rprop_arrived: a direction change while a particle is still in TRANSPORT (its
                # plan image farther than the pace radius — a curved path, the C's arms round the hole) is
                # not an overshoot; only an ARRIVED particle's reversal halves its step (g41s: C lost 0.011
                # of silIoU to halvings in transit). The arrival is the paced target's own per-particle mask.
                _arr = stats.get("arrived_mask") if getattr(cfg, "ctrl_rprop_arrived", False) else None
                if _arr is not None and len(_arr) == len(_flip):
                    _flip = _flip & np.asarray(_arr, bool)
                ctrl_scale[_flip] *= 0.5
                ctrl_rev_count[_flip] += 1
                ctrl_scale[_same] = np.minimum(1.0, ctrl_scale[_same] * 1.2)
                if _act.any():
                    _cs_app = _smooth(ctrl_scale) if _nbr is not None else ctrl_scale
                    rec["ctrl_flip_frac"] = float(_flip[_act].mean()); rec["ctrl_scale_med"] = float(np.median(_cs_app[_act]))
                    rec["ctrl_scale_lo"] = float((_cs_app[_act] < 0.01).mean())
                    if (a + 1) % 5 == 0:
                        log(f"[v2] anim {a + 1}: control step reversals {100 * _flip[_act].mean():.0f} % of the moving particles, "
                            f"step scale median {np.median(_cs_app[_act]):.3f}, below 0.1: {100 * (_cs_app[_act] < 0.1).mean():.0f} %, "
                            f"below 0.01: {100 * (_cs_app[_act] < 0.01).mean():.0f} %")
            ctrl_prev_disp = _d_now
            ctrl_scale_apply = _smooth(ctrl_scale) if _nbr is not None else ctrl_scale
            # ---- config.freeze_arrived (docs/method.md 10.25; the user 2026-09-24: once optimised, lock it with
            # the plasticity): a particle that has ARRIVED (the paced target's mask) and reversed twice (one full
            # period of the alternation: settled by the rule's own reading) is FROZEN for good — its elastic
            # stretch assimilated in full (F_e -> R_e: no stress of its own), its control zeroed and its update
            # scale 0, its u bound 0, its velocity zeroed at commits. Frozen material is inert, carried by the
            # grid with its neighbours; the frozen set only grows. ----
            # ---- config.settle_eta (docs/method.md 10.26): the settled body's viscosity. A particle that has
            # arrived and reversed twice (the same reading as the freeze) is given the forward model's
            # per-particle viscosity with the time constant of ONE WINDOW, eta = 1 / (T dt) — the
            # quasi-static limit of settled material: whatever motion the grid hands it decays within
            # the window it arises in, instead of being carried into the next window's linearisation.
            # Derived from the discretisation; nothing else changes (control, u, assimilation as before). ----
            if getattr(cfg, "settle_eta", False) and ctrl_prev_disp is not None and frozen_p is not None:
                if settled_p is None or len(settled_p) != len(_d_now):
                    settled_p = np.zeros(len(_d_now), bool)
                _arr_s = stats.get("arrived_mask")
                _arr_s = np.ones(len(_d_now), bool) if _arr_s is None or len(_arr_s) != len(_d_now) else np.asarray(_arr_s, bool)
                settled_p |= _arr_s & (ctrl_rev_count >= 2)
                _eta_w = 1.0 / (float(cfg.T) * float(prm.dt))
                settle_eta_arr = np.where(settled_p, np.float32(_eta_w), np.float32(0.0)).astype(np.float32)
                rec["settled_frac"] = float(settled_p.mean())
                if (a + 1) % 5 == 0:
                    log(f"[v2] anim {a + 1}: settled {100 * settled_p.mean():.1f} % of the particles (viscosity {_eta_w:.3g}, one window)")
            # ---- config.settle_pin (docs/method.md 10.27; the user: the oscillation must be zero): the settled
            # set (arrived, twice reversed — the same reading as settle_eta / freeze) is PINNED inside the
            # rollout: the kernels give it no velocity, no strain, no control, no relaxation move, so the
            # settled body is exactly still frame to frame; its control is zeroed and its scale 0 as for the
            # freeze. A kinematic constraint during the morph only. ----
            if getattr(cfg, "settle_pin", False) and ctrl_prev_disp is not None and frozen_p is not None:
                if settled_p is None or len(settled_p) != len(_d_now):
                    settled_p = np.zeros(len(_d_now), bool)
                _arr_p = stats.get("arrived_mask")
                _arr_p = np.ones(len(_d_now), bool) if _arr_p is None or len(_arr_p) != len(_d_now) else np.asarray(_arr_p, bool)
                if settled_at is None or len(settled_at) != len(_d_now):
                    settled_at = np.full(len(_d_now), -1, np.int32)
                _newly = _arr_p & (ctrl_rev_count >= 2) & (~settled_p)
                # config.settle_pin_clear (10.27 addendum): a particle is pinned only when no UNARRIVED particle
                # lies within the pace radius of it — the paced target's own arrival scale, no new constant. The
                # pinned body then never blocks a channel material still flows through (ap300: the ear's tip
                # starved at 3.9 reference particles against 13.6) and its boundary stays one arrival radius
                # clear of the last arrivals (g41z: det F −0.05…−0.11 at the arrived/in-transit boundary).
                _pr = stats.get("pace_r")
                if getattr(cfg, "settle_pin_clear", False) and _newly.any() and _pr is not None and (~_arr_p).any():
                    from scipy.spatial import cKDTree as _KDc
                    _dn_c, _ = _KDc(np.asarray(x, np.float32)[~_arr_p]).query(np.asarray(x, np.float32)[_newly], k=1,
                                                                           distance_upper_bound=float(_pr), workers=-1)
                    _clear = ~np.isfinite(_dn_c)
                    rec["pin_blocked_frac"] = float(1.0 - _clear.mean())
                    _idx_new = np.nonzero(_newly)[0]
                    _newly = np.zeros_like(_newly); _newly[_idx_new[_clear]] = True
                settled_at[_newly] = a + 1           # pinned from the rollout of window a + 2 on (frames after this commit)
                settled_p |= _newly
                settle_pin_arr = settled_p.astype(np.float32)
                if settled_p.any():
                    ctrl_scale[settled_p] = 0.0
                    ctrl_scale_apply = np.asarray(ctrl_scale_apply, np.float32).copy(); ctrl_scale_apply[settled_p] = 0.0
                    if dfc_prev is not None:
                        _dp = np.asarray(dfc_prev)
                        _axp = [i for i, sz in enumerate(_dp.shape) if sz == len(settled_p)]
                        if _axp:
                            _idx = [slice(None)] * _dp.ndim; _idx[_axp[0]] = settled_p; _dp[tuple(_idx)] = 0.0; dfc_prev = _dp
                    if u_scale is not None and len(u_scale) == len(settled_p):
                        u_scale[settled_p] = 0.0
                    if st.get("v") is not None:
                        st["v"][settled_p] = 0.0
                        if st.get("C") is not None:
                            st["C"][settled_p] = 0.0
                rec["pinned_frac"] = float(settled_p.mean())
                if (a + 1) % 5 == 0:
                    log(f"[v2] anim {a + 1}: pinned {100 * settled_p.mean():.1f} % of the particles")
            if getattr(cfg, "freeze_arrived", False) and ctrl_prev_disp is not None and frozen_p is not None:
                _arr_f = stats.get("arrived_mask")
                _arr_f = np.ones(len(_d_now), bool) if _arr_f is None or len(_arr_f) != len(_d_now) else np.asarray(_arr_f, bool)
                _new = (~frozen_p) & _arr_f & (ctrl_rev_count >= 2)
                if _new.any():
                    Fp[_new] = assimilate_elastic(Fc[_new], Fp[_new], eta=1.0, smin=cfg.assim_smin,
                                                  smax=cfg.assim_smax, isochoric=False)
                    frozen_p |= _new
                if frozen_p.any():
                    ctrl_scale[frozen_p] = 0.0
                    ctrl_scale_apply = np.asarray(ctrl_scale_apply, np.float32).copy(); ctrl_scale_apply[frozen_p] = 0.0
                    if dfc_prev is not None:
                        _dp = np.asarray(dfc_prev)
                        _axf = [i for i, sz in enumerate(_dp.shape) if sz == len(frozen_p)]
                        if _axf:
                            _idx = [slice(None)] * _dp.ndim; _idx[_axf[0]] = frozen_p; _dp[tuple(_idx)] = 0.0; dfc_prev = _dp
                    if u_scale is not None and len(u_scale) == len(frozen_p):
                        u_scale[frozen_p] = 0.0
                    if st.get("v") is not None:
                        st["v"][frozen_p] = 0.0
                        if st.get("C") is not None:
                            st["C"][frozen_p] = 0.0
                    rec["frozen_frac"] = float(frozen_p.mean())
                    if (a + 1) % 5 == 0 or _new.sum() > 0.05 * len(frozen_p):
                        log(f"[v2] anim {a + 1}: frozen {100 * frozen_p.mean():.1f} % of the particles ({int(_new.sum())} new this window)")
        # ---- the next window starts from rest (config.rest_commit; docs/method.md 10.21): the carried momentum
        # of an accepted commit is what the rebound probe measured continuing forward and the next window's
        # control cancelling — the two-window alternation at the resolved scale. v and C zeroed; x, F, Fp kept. ----
        if getattr(cfg, "rest_commit", False) and st.get("v") is not None:
            _gate_thr = float(getattr(cfg, "rest_commit_gate", 1.0) or 0.0)
            _ug = stats.get("u_gate")
            _by_gate = _gate_thr <= 0.0 or (_ug is not None and float(_ug) >= _gate_thr - 1e-6)
            # the reversal latch (10.21, second form): once consecutive accepted windows reverse each
            # other the carried momentum is an overshoot by definition; two accepted commits in a row
            # with a negative reversal cosine = one full period of the alternation
            _rev_neg = reversal_cos is not None and float(reversal_cos) < 0.0
            _by_rev = bool(getattr(cfg, "rest_commit_reversal", False)) and _rev_neg and rev_prev_neg
            rev_prev_neg = _rev_neg
            if not rest_latched and (_by_gate or _by_rev):
                rest_latched = True
                log(f"[v2] anim {a + 1}: windows from rest from here on ("
                    + (f"u transport gate {100 * float(_ug if _ug is not None else 0):.1f} % >= {100 * _gate_thr:.0f} %"
                       if _by_gate else f"two accepted commits reversing in a row, cos {float(reversal_cos):.2f}") + ")")
            if rest_latched and getattr(cfg, "settle_commit", False):
                # ---- config.settle_commit (docs/method.md 10.26): the delivered commit is an EQUILIBRIUM — from
                # the accepted state, one window of zero-control dynamics under the settled body's viscosity
                # (eta = 1 / (T dt), the same constant as settle_eta), the carried velocity and affine state
                # included; the settled positions and F replace the commit's, the velocity is then zeroed.
                # The next window is linearised at rest instead of at a state still moving. ----
                try:
                    from ..mpm.constitutive import lame as _lame_s
                    from ..mpm.function import RolloutSpec as _RS_s, warp_mpm as _wm_s
                    _lam0, _mu0 = _lame_s(cfg.young, cfg.poisson)
                    _N = len(x); _mref = int(getattr(cfg, "mass_ref_n", 0) or 0)
                    _mass = (float(_mref) / float(_N)) if (_mref > 0 and _N != _mref) else 1.0
                    _eta_w = 1.0 / (float(cfg.T) * float(prm.dt))
                    _spec_s = _RS_s(x0=np.ascontiguousarray(x, np.float32), m=_mass, lam=_lam0, mu=_mu0, prm=prm, T=int(cfg.T),
                                    Fp=np.ascontiguousarray(Fp, np.float32), v0=np.ascontiguousarray(st["v"], np.float32),
                                    F0=np.ascontiguousarray(Fc, np.float32),
                                    C0=(np.ascontiguousarray(st["C"], np.float32) if st.get("C") is not None else None),
                                    device=cfg.device, vol0=vol0, eta=np.full(_N, np.float32(_eta_w), np.float32))
                    with torch.no_grad():
                        _xS, _FS = _wm_s(torch.zeros(_N, 3, 3, device=cfg.device), _spec_s)
                    _xS = _xS.detach().cpu().numpy().astype(np.float32); _FS = _FS.detach().cpu().numpy().astype(np.float32).reshape(-1, 3, 3)
                    _ds = float(np.median(np.linalg.norm(_xS - np.asarray(x, np.float32), axis=1)))
                    if np.isfinite(_xS).all() and np.isfinite(_FS).all():
                        x = _xS; Fc = _FS
                        rec["settle_disp"] = _ds
                        if (a + 1) % 5 == 0:
                            log(f"[v2] anim {a + 1}: settle at commit — median displacement {_ds:.5f} wu ({_ds / max(float(prm.dx), 1e-9):.4f} cells)")
                except Exception as _e:
                    log(f"[v2] anim {a + 1}: settle at commit failed: {_e}")
            if rest_latched:
                st["v"] = np.zeros_like(st["v"])
                if st.get("C") is not None:
                    st["C"] = np.zeros_like(st["C"])
            rec["rest_from_here"] = int(rest_latched)

        any_guard = n_out or n_nan or n_ns or n_bad or n_flip or n_inv
        if a % max(1, cfg.animations // 10) == 0 or a == cfg.animations - 1 or any_guard:
            log(f"[v2] anim {a + 1}/{cfg.animations}  L={rec['loss']:.4f}  D_vol={rec['d_vol']:.3f}" +
                (f"  D_r={rec['d_render']:.5f}  lam={rec['lambda']:.3g}" if rec["d_render"] is not None else "") +
                f"  kin={rec['kin']:.4f}  |v|max={rec['v_absmax']:.3f}  move={rec['move']:.4f}" +
                f"  Jmin={rec['Jmin_traj']:.3f}  acc/rej={rec['accepted']}/{rec['rejected']}" +
                (f"  GUARD clamp={n_out} nanx={n_nan} nanst={n_ns} Freset={n_bad} "
                 f"Fflip={n_flip} Finv={n_inv}" if any_guard else ""))

    # Deliverable = the trajectory up to its BEST shape-merit commit. The paced runs
    # reach their best state mid-run (b4: d_vol 62 at a435 of 450) and then wander
    # the flat valley; the archive up to the best commit is continuous (no snap), and
    # the wandering tail is not shown. The full history stays in `history`.
    trunc = None
    deliver_n = len(frames)
    if cfg.best_truncate:
        acc = [r for r in hist if r.get("frame_end") and not r.get("null_commit")
               and r.get("d_vol") is not None]
        if len(acc) >= 2:
            # RESOLUTION-INVARIANT shape merit only: d_vol + d_dt (fixed grids).
            # d_sil is excluded - the c2f rebuild changes render_res mid-run and
            # r3's truncation picked a150 over the genuinely better a217 because
            # the silhouette scalar jumped 53% across that boundary.
            r0 = acc[0]

            def merit(r):
                m = r["d_vol"] / max(abs(r0["d_vol"]), 1e-8)
                if r.get("d_dt") is not None and r0.get("d_dt"):
                    m += r["d_dt"] / max(abs(r0["d_dt"]), 1e-8)
                if r.get("d_kde") is not None and r0.get("d_kde"):
                    m += r["d_kde"] / max(abs(r0["d_kde"]), 1e-8)
                if r.get("d_h1") is not None and r0.get("d_h1"):
                    m += r["d_h1"] / max(abs(r0["d_h1"]), 1e-8)
                return m
            best = min(acc, key=merit)
            if best is not acc[-1] and merit(acc[-1]) > merit(best) * (1 + cfg.tol):
                deliver_n = int(best["frame_end"])
                trunc = {"best_animation": int(best["animation"]) + 1,
                         "frames_kept": deliver_n, "frames_dropped": len(frames) - deliver_n}
                log(f"[v2] deliverable ends at best commit anim {trunc['best_animation']}"
                    f" (deliver {deliver_n} of {len(frames)} frames; all frames archived)")
    if dress is not None:                        # close the archive over any tail
        dress.cover_frames(len(frames))
    return {"truncation": trunc, "deliver_n": deliver_n,   # frames are NEVER dropped
            "dressing": dress.export() if dress is not None else None,
            "frames": frames, "F_frames": F_frames, "history": hist, "guards": guards,
            "Fg_commits": Fg_commits,
            "balancer": {"cap": balancer.cap, "cap_rel": getattr(balancer, "cap_rel", None),
                         "alpha_lam": balancer.alpha_lam},
            "s": s, "Fp": Fp, "n_held": n_held, "converged": frozen, "reattached": n_reattach_total,
            "render_mask": ((surface_w > 0.5) if cfg.render_surface_only else None),
            "pinned": settled_p,                      # config.settle_pin / settle_eta: the settled set at the end (None when off)
            "pinned_at": settled_at}                  # config.settle_pin: the window at which each particle was pinned (-1 never)
