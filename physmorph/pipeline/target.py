"""The fixed target of a run: grids, images and calibrations built once on the device.

TargetPack holds what every window reads: the target mass on the loss grid, the render
targets (silhouettes and the matched shading images), the target's distance transform for
the W1 cleanup, the target points and their device KD-tree for the near-band cleanup, the
local-support term, and the scalars measured once at the source (the unit ratios, the
transport scale, the render weight).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .. import gpu
from ..losses.silhouette import set_kernel
from ..losses.support import SurfaceProximity, TransportSupport
from ..losses.volumetric import (d_vol, d_vol_density, density_units, rasterize_mass_cubic, target_dt_grid,
                                 target_mass_grid)
from ..mpm.state import MPMParams
from .config import PipelineConfig
from ..render.exterior import Lattice, ZhuBridson
from .render_loss import exterior_targets, make_views, shade_targets, target_silhouettes


@dataclass
class Exterior:
    """What the render terms need when they are read on the exterior (render/exterior.py): the field's pitch, the
    lattice and its pitch, the skin of the discs' particle lists, and the target's images from its own discs."""
    pitch: float
    lattice: Lattice
    h: float
    skin: float
    sils: list
    shade: list
    radius: float = 3.0                 # the field's kernel radius and offset, in pitches (D123; 3 and 0.8 = D59's)
    offset: float = 0.8


@dataclass
class TargetPack:
    grid: torch.Tensor                  # target mass on the loss grid
    lgmin: torch.Tensor
    ldx: float
    ldims: tuple
    m: torch.Tensor                     # unit loss-side masses (N,)
    views: list                         # [(theta, phi)]
    sils: list                          # target alpha images
    extent: float                       # render half-extent (1.25 x max |target|)
    shade: list                         # matched shading images
    pgmin: torch.Tensor                 # the morph's normal grid at the render pixel,
    pdx: float                          #   blurred by pblur cells (the drawn surface)
    pdims: tuple
    pblur: float
    dt3: torch.Tensor                   # target DT on its own fine grid (W1 cleanup)
    dtgmin: torch.Tensor
    dtdx: float
    dtdims: tuple
    pts: torch.Tensor                   # target points (N,3)
    knn: gpu.KNN                        # device KD-tree of pts
    nn_spacing: float                   # target median nearest-neighbour spacing
    m_ref: float                        # density-unit constants of the loss grid
    n_support: int
    support: TransportSupport | None
    unit_ratio: float = 1.0             # D_vol(legacy) / D_vol(density) at the source
    unit_grad_ratio: float = 1.0        # |grad D_vol(legacy)| / |grad D_vol(density)|
    grid_ot: object = None              # GridSinkhornLoss, built at the first window
    ot_scale: float | None = None       # transport scale: equal gradient norm with D_vol
    settled_step: float | None = None   # last accepted step (warm start of the search)
    gate: tuple | None = None           # (grid, dx, dims) of the u transport gate: the MPM-cell grid
    ext: Exterior | None = None         # the render terms' exterior, when they are read on it
    relief: object = None               # window/layer.TargetRelief: the relaxation's reference, from the target's surface
    xu: tuple | None = None             # cfg.baseline "xu": (the target mass on the simulation grid, its origin, dx, dims)
    draws: list | None = None           # further samples of the target whose pictures the render's targets average
    # D122 (--surface_density > 1): the body's and the target's particles carry their rest volume relative to the mean
    # (w, the sampler's: the loss-side masses m are the body's w and the target's grids are rasterised with the target's,
    # the dynamics mass is m w) and the spacing of their own over the base spacing (local = w^(1/3)); None for the
    # uniform sample, where every path is as before
    body_w: torch.Tensor | None = None         # the body's w (tgt.m is it as well, the loss-side masses of the body)
    body_local: torch.Tensor | None = None
    local: torch.Tensor | None = None          # the target's w^(1/3)
    w_tgt: torch.Tensor | None = None          # the target's w (its grids are rasterised with it)
    base: object = None                        # the uniform sample (numpy) the extent and the lattice were taken from


def _local(w, like: torch.Tensor):
    """w^(1/3) as a tensor like `like` (None stays None)."""
    return None if w is None else gpu.tensor(w).to(like.dtype).pow(1.0 / 3.0)


def target_relief(tgt_t: torch.Tensor, surface, cfg: PipelineConfig, local: torch.Tensor | None = None):
    """The relaxation's reference (window/layer.TargetRelief) from points of the target mesh's surface and their
    normals. The mesh's normals are taken as they are, all turned at once if as a whole they point into the
    sample (point by point the sample cannot tell: its eight nearest particles put 12-15 % of a sound mesh's
    normals on the wrong side, and turning those gave residuals of tens of pitches, D88). Printed: the reference
    and the rough residual d - dbar on the target sample's own layer, which should share their mean (D105).
    local: the target particles' own spacing over the base (D122)."""
    from .window.layer import TargetRelief, layer_relax_data, layer_spacing
    pts, nrm = (gpu.tensor(np.asarray(v, np.float32)) for v in surface)
    inside = tgt_t[gpu.KNN(tgt_t).query(pts, 8)[1]].mean(1)
    if float(((pts - inside) * nrm).sum(1).sign().mean()) < 0:
        nrm = -nrm
    sp = layer_spacing(tgt_t, local)
    relief = TargetRelief(pts, nrm, sp)
    mask, lnrm, nbr, w = layer_relax_data(tgt_t, sp, k=cfg.layer_k, h_sp=cfg.layer_h_sp, local=local)
    on = mask > 0.5
    d = (lnrm * (tgt_t - (w[..., None] * tgt_t[nbr]).sum(1))).sum(1)
    rough, ref = (d - (w * d[nbr]).sum(1))[on] / sp, relief.at(tgt_t, mask, lnrm, nbr, w, local)[on] / sp
    print(f"[target] relief: {len(pts)} surface points; on the target sample's layer (spacings) the reference has mean "
          f"{float(ref.mean()):+.4f}, rms {float(ref.square().mean().sqrt()):.4f}, the rough residual mean "
          f"{float(rough.mean()):+.4f}, rms {float(rough.square().mean().sqrt()):.4f}", flush=True)
    return relief


def build_target(target_x, prm: MPMParams, cfg: PipelineConfig, draws=None, w=None, w_body=None, base=None) -> TargetPack:
    """draws: further independent samples of the target in its frame. The render term's target pictures are then the
    mean of the pictures the operator draws of every sample: what it draws of the mesh in expectation. Drawn from
    the one sample, a render arm fitted that sample's noise as well, and half its measured lead over the
    physics-only twin was that fit (D90, D91).
    w, w_body (D122, --surface_density > 1): the target's and the body's rest-volume weights (mean 1). The loss-side
    masses are then w, so the rasterised target is the uniform density the F = 1 sample has and the transport
    problem is the same one with more points at the surface; every spacing read from the sample (the shading blur,
    the exterior's pitch and lattice, the near band's berth) is the BASE spacing (gpu.median_kth_spacing with local),
    so the loss's grids and pictures keep their definition, while the proximity's kernel and the layer's depth,
    which act on particles, use each particle's own spacing."""
    set_kernel("cic")
    tgt_t = gpu.tensor(target_x)
    samples = [tgt_t] + [gpu.tensor(d) for d in (draws or [])]
    mean = lambda per: [torch.stack(v).mean(0) for v in zip(*per)]   # noqa: E731  (per sample, a list over views)
    N = tgt_t.shape[0]
    local = _local(w, tgt_t)
    m = torch.ones(N, device=gpu.DEVICE)                      # the loss-side masses: the body's (tgt.m) and the target's
    m_t = m if w is None else gpu.tensor(w)                   #   (the target's grids below), unit for the uniform sample
    if w_body is not None:
        m = gpu.tensor(w_body)
    if cfg.support_weight <= 0:
        support = None
    elif cfg.support_form == "proximity":                    # the fine geometry read from the target surface
        support = SurfaceProximity(tgt_t, local=local)
    else:
        support = TransportSupport(tgt_t, cfg.support_weight, cfg.support_target_ref, cfg.support_form)
    # the loss grid covers the MPM domain (scalar geometry, float32 like the grid itself)
    dmin = np.asarray(prm.grid_min, np.float32)
    dmax = dmin + prm.dx * np.array([prm.nx, prm.ny, prm.nz], np.float32)
    ldx = float((dmax - dmin).max() / cfg.loss_res)
    ldims = (cfg.loss_res,) * 3
    lgmin = torch.tensor(dmin, device=gpu.DEVICE)
    grid = target_mass_grid(tgt_t, m_t, lgmin, ldx, ldims)
    # the u transport gate measures its radius in MPM cells, so its transport map is solved on the
    # MPM-cell grid; that is the loss grid itself unless the loss grid refines with N
    gate = (grid, ldx, ldims)
    if cfg.loss_follows_n and cfg.loss_res != prm.nx:
        gdx, gdims = float((dmax - dmin).max() / prm.nx), (prm.nx,) * 3
        gate = (target_mass_grid(tgt_t, m_t, lgmin, gdx, gdims), gdx, gdims)
    views = make_views(cfg.render_views, cfg.render_elevs)
    # the render's extent and the exterior's lattice from the uniform sample (`base`, D122) when there is one, so the
    # pictures' pixel and the lattice's origin are the F = 1 run's (the F sample's centroid and extremes differ by a fraction of a pitch)
    geom = tgt_t if base is None else gpu.tensor(base)
    extent = float(geom.abs().max()) * 1.25
    sils = mean([target_silhouettes(s, views, cfg.render_res, extent, cfg.sil_k) for s in samples])
    # shading: the morph's normals on a render-pixel grid blurred by the renderer's 1.5 target
    # spacings; the target image is drawn by the same operator at the target (matched), so
    # the loss compares the operator with itself and no target/operator bias enters
    sp_t = gpu.median_kth_spacing(tgt_t, 8, subsample=20000, local=local)       # the base spacing (D122)
    pdx = 2.0 * extent / cfg.render_res
    pdims = tuple(int(np.ceil((dmax - dmin).max() / pdx)) for _ in range(3))
    pblur = 1.5 * sp_t / pdx
    shade = mean([shade_targets(s, views, cfg.render_res, extent, lgmin, pdx, pdims,
                                cfg.sil_k, cfg.pbr_ambient, blur_cells=pblur) for s in samples])
    print(f"[target] shading target: spacing {sp_t:.4f}, normal grid {pdims[0]}^3 at {pdx:.4f} wu "
          f"({pdx / sp_t:.2f} spacings), blur {pblur:.2f} cells", flush=True)
    ext = None
    if cfg.render_exterior:
        # the field's pitch is the volume sample's (0.708 of the 8th-neighbour distance); the discs sit on a lattice
        # of half a render pixel, no coarser than the field resolves (its smallest body, a sphere of 0.8 pitches,
        # holds a node of any lattice up to 0.92 pitches); the discs' particle lists reach one pitch past the kernel.
        # D123 (--exterior_radius): the kernel radius in pitches, the offset in proportion (0.8 at 3), the lattice cap with
        # the offset; at 3 every number is the old one bit for bit
        pitch = 0.708 * sp_t
        radius = cfg.exterior_radius
        offset = 0.8 * (radius / 3.0)
        center = geom.mean(0)
        lattice = Lattice(center, 2.8 * float((geom - center).norm(dim=1).max()))
        h = min(extent / cfg.render_res, 0.92 * pitch * (radius / 3.0))
        per = []
        for s in samples:
            with torch.no_grad():
                p_t, g_t, _, _ = lattice.discs(ZhuBridson(s, pitch, radius=radius, offset=offset), h, refine=False)
            per.append(exterior_targets(p_t, torch.nn.functional.normalize(g_t, dim=1), views, cfg.render_res,
                                        extent, cfg.sil_k, cfg.pbr_ambient))
        e_sils, e_shade = mean([e[0] for e in per]), mean([e[1] for e in per])
        ext = Exterior(pitch, lattice, h, pitch, e_sils, e_shade, radius, offset)
        print(f"[target] exterior: pitch {pitch:.4f} wu, kernel {radius:g} pitches, offset {offset:.3f}, lattice {h / pitch:.2f} pitches = "
              f"{h / (2.0 * extent / cfg.render_res):.2f} render pixels, {len(p_t)} discs on the target"
              + (f"; the pictures are the mean over {len(samples)} samples" if len(samples) > 1 else ""), flush=True)
    # the W1 cleanup's fine target-fitted grid (1.5 extents each way, the box leash's range)
    dtdims = (cfg.dt_res,) * 3
    dtdx = 3.0 * extent / cfg.dt_res
    dtgmin = torch.tensor([-1.5 * extent] * 3, device=gpu.DEVICE)
    dt3 = target_dt_grid(target_mass_grid(tgt_t, m_t, dtgmin, dtdx, dtdims), dtdx, dtdims,
                         clamp=cfg.dt_clamp_frac * extent)
    knn = gpu.KNN(tgt_t)
    d1 = knn.query(tgt_t, 2)[0][:, 1]
    nn_sp = gpu.median(d1 if local is None else d1 / local.to(d1.dtype))         # the base spacing (D122)
    m_ref, n_support = density_units(grid)
    xu = None
    if cfg.baseline.startswith("xu"):         # the baseline's target: the simulation grid, the MPM's cubic B-spline
        xu_gmin = torch.tensor(np.asarray(prm.grid_min, np.float32), device=gpu.DEVICE)
        xu_dims = (prm.nx, prm.ny, prm.nz)
        xu = (rasterize_mass_cubic(tgt_t, m_t * cfg.xu_mass, xu_gmin, prm.dx, xu_dims).detach(), xu_gmin,
              float(prm.dx), xu_dims)
    return TargetPack(grid=grid, lgmin=lgmin, ldx=ldx, ldims=ldims, m=m, views=views, sils=sils,
                      extent=extent, shade=shade, pgmin=lgmin, pdx=pdx, pdims=pdims, pblur=pblur,
                      dt3=dt3, dtgmin=dtgmin, dtdx=dtdx, dtdims=dtdims, pts=tgt_t, knn=knn,
                      nn_spacing=nn_sp, m_ref=m_ref, n_support=n_support, support=support, gate=gate, ext=ext,
                      xu=xu, draws=draws, body_w=None if w_body is None else gpu.tensor(w_body),
                      body_local=_local(w_body, tgt_t), local=local, w_tgt=None if w is None else m_t, base=base)


def rebuild_for_resolution(tgt: TargetPack, target_x, prm: MPMParams,
                           cfg: PipelineConfig) -> TargetPack:
    """Coarse-to-fine: a fresh target at cfg.render_res. The unit ratios and the transport
    scale and solver survive (no weight changes silently at the switch); the render weight
    and the step warm start are recalibrated for the new images."""
    new = build_target(target_x, prm, cfg, draws=tgt.draws, w=tgt.w_tgt, w_body=tgt.body_w, base=tgt.base)
    new.unit_ratio, new.unit_grad_ratio = tgt.unit_ratio, tgt.unit_grad_ratio
    new.ot_scale, new.grid_ot, new.relief = tgt.ot_scale, tgt.grid_ot, tgt.relief
    return new


def calibrate_units(tgt: TargetPack, source_x: torch.Tensor, cfg: PipelineConfig) -> None:
    """Measure the legacy/density ratios of D_vol at the source: unit_ratio converts every
    fixed weight (so the relative weighting equals the legacy one at the source),
    unit_grad_ratio the gradient-magnitude constants (Adam eps, the adaptive-step norm).
    The legacy side is evaluated on the fixed reference grid (cfg.unit_ref_res), the grid
    every legacy weight was tuned on."""
    ref = int(cfg.unit_ref_res)
    if ref > 0 and ref != int(tgt.ldims[0]):
        dims_ref = (ref,) * 3
        ldx_ref = float(tgt.ldx * tgt.ldims[0] / ref)
        grid_ref = target_mass_grid(tgt.pts, tgt.m if tgt.w_tgt is None else tgt.w_tgt, tgt.lgmin, ldx_ref, dims_ref)
    else:
        grid_ref, ldx_ref, dims_ref = tgt.grid, tgt.ldx, tgt.ldims
    xg = source_x.detach().clone().requires_grad_(True)
    L_leg = d_vol(xg, tgt.m, grid_ref, tgt.lgmin, ldx_ref, dims_ref)
    g_leg = torch.autograd.grad(L_leg, xg)[0].norm()
    xg2 = source_x.detach().clone().requires_grad_(True)
    L_den = d_vol_density(xg2, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims,
                          tgt.m_ref, tgt.n_support)
    g_den = torch.autograd.grad(L_den, xg2)[0].norm()
    L_leg, L_den = L_leg.detach(), L_den.detach()
    if float(L_den) <= 0 or float(g_den) <= 0:
        raise ValueError("the unit calibration needs a source that differs from the target")
    tgt.unit_ratio = float(L_leg / L_den)
    tgt.unit_grad_ratio = float(g_leg / g_den)
