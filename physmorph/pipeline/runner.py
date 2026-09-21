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
from .config import PipelineConfig
from .optimizer import TargetPack, optimize_window
from .render_loss import (LambdaBalancer, d_render, make_views, shade_targets,
                          target_silhouettes)
from .surface_local import surface_local_pass


def _id(N):
    return np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))


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
                           sigma0_from_nn(gaussian_points, cfg.gauss_sigma_scale),
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
        coh_nbr = cKDTree(src).query(src, k=int(cfg.coh_k) + 1, workers=-1)[1][:, 1:]
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
    outer_scales = outer_prev = prev_disp = None
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
            outer_scales = outer_prev = prev_disp = None
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
            frontier=frontier, bond_rest=bond_rest, bond_frag=bond_frag)
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
        dets = batched_det(np.stack(F_seq[1:]))        # one batched det over the window
        n_inv = int((dets <= 0.0).any(0).sum())
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
               "Jmin_traj": float(dets.min()),
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
            if outer_prev is not None:
                outer_gain = (outer_prev - score) / max(abs(outer_prev), 1e-8)
                # Latch evidence must be a SUSTAINED plateau (stale>=2: third
                # consecutive non-improving commit). "small move" was retired as
                # a criterion twice over: with a large w_kin it is reachable at
                # 10% of the descent (s1), and under glidepath pacing every
                # window's move is small BY DESIGN, so one noisy no-improvement
                # commit armed the latch at anim ~59/300 (s3_paced).
                near_stationary = (not improved) and stale >= 2
                outer_gate_latched = outer_gate_latched or near_stationary
                if improved and not stats.get("pace_bound"):
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
                brake_reject = outer_gain < -max(cfg.pace, 0.05)
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
                    f"(gain={outer_gain:.3g}, reversal={reversal_cos}"
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
            prev_disp = disp.copy()
        else:
            prev_disp = disp.copy()
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
            anneal = (min(1.0, anneal * 1.15) if improved
                      else max(0.05, anneal * cfg.anneal_stale))
        if (cfg.anneal_on_reversal > 0 and reversal_cos is not None
                and reversal_cos < cfg.outer_reversal_cos):
            anneal = max(0.05, anneal * cfg.anneal_on_reversal)   # sign-change step control
        rec.update({"improved": int(bool(improved)), "stale": stale, "anneal": anneal,
                    "pace_bound": int(bool(stats.get("pace_bound")))})  # freeze forensics
        if stale >= cfg.patience:
            frozen = True
            log(f"[v2] converged at anim {a + 1} (phys={phys_track:.4f}); holding still")

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
            "render_mask": ((surface_w > 0.5) if cfg.render_surface_only else None)}
