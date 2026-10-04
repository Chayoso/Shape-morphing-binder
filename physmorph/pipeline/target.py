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
from ..losses.volumetric import (d_vol, d_vol_density, density_units, target_dt_grid,
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


def build_target(target_x, prm: MPMParams, cfg: PipelineConfig) -> TargetPack:
    set_kernel("cic")
    tgt_t = gpu.tensor(target_x)
    N = tgt_t.shape[0]
    m = torch.ones(N, device=gpu.DEVICE)
    if cfg.support_weight <= 0:
        support = None
    elif cfg.support_form == "proximity":                    # the fine geometry read from the target surface
        support = SurfaceProximity(tgt_t)
    else:
        support = TransportSupport(tgt_t, cfg.support_weight, cfg.support_target_ref, cfg.support_form)
    # the loss grid covers the MPM domain (scalar geometry, float32 like the grid itself)
    dmin = np.asarray(prm.grid_min, np.float32)
    dmax = dmin + prm.dx * np.array([prm.nx, prm.ny, prm.nz], np.float32)
    ldx = float((dmax - dmin).max() / cfg.loss_res)
    ldims = (cfg.loss_res,) * 3
    lgmin = torch.tensor(dmin, device=gpu.DEVICE)
    grid = target_mass_grid(tgt_t, m, lgmin, ldx, ldims)
    # the u transport gate measures its radius in MPM cells, so its transport map is solved on the
    # MPM-cell grid; that is the loss grid itself unless the loss grid refines with N
    gate = (grid, ldx, ldims)
    if cfg.loss_follows_n and cfg.loss_res != prm.nx:
        gdx, gdims = float((dmax - dmin).max() / prm.nx), (prm.nx,) * 3
        gate = (target_mass_grid(tgt_t, m, lgmin, gdx, gdims), gdx, gdims)
    views = make_views(cfg.render_views, cfg.render_elevs)
    extent = float(tgt_t.abs().max()) * 1.25
    sils = target_silhouettes(tgt_t, views, cfg.render_res, extent, cfg.sil_k)
    # shading: the morph's normals on a render-pixel grid blurred by the renderer's 1.5 target
    # spacings; the target image is drawn by the same operator at the target (matched), so
    # the loss compares the operator with itself and no target/operator bias enters
    sp_t = gpu.median_kth_spacing(tgt_t, 8, subsample=20000)
    pdx = 2.0 * extent / cfg.render_res
    pdims = tuple(int(np.ceil((dmax - dmin).max() / pdx)) for _ in range(3))
    pblur = 1.5 * sp_t / pdx
    shade = shade_targets(tgt_t, views, cfg.render_res, extent, lgmin, pdx, pdims,
                          cfg.sil_k, cfg.pbr_ambient, blur_cells=pblur)
    print(f"[target] shading target: spacing {sp_t:.4f}, normal grid {pdims[0]}^3 at {pdx:.4f} wu "
          f"({pdx / sp_t:.2f} spacings), blur {pblur:.2f} cells", flush=True)
    ext = None
    if cfg.render_exterior:
        # the field's pitch is the volume sample's (0.708 of the 8th-neighbour distance); the discs sit on a lattice
        # of half a render pixel, no coarser than the field resolves (its smallest body, a sphere of 0.8 pitches,
        # holds a node of any lattice up to 0.92 pitches); the discs' particle lists reach one pitch past the kernel
        pitch = 0.708 * sp_t
        center = tgt_t.mean(0)
        lattice = Lattice(center, 2.8 * float((tgt_t - center).norm(dim=1).max()))
        h = min(extent / cfg.render_res, 0.92 * pitch)
        with torch.no_grad():
            p_t, g_t, _, _ = lattice.discs(ZhuBridson(tgt_t, pitch), h, refine=False)
        e_sils, e_shade = exterior_targets(p_t, torch.nn.functional.normalize(g_t, dim=1), views, cfg.render_res,
                                           extent, cfg.sil_k, cfg.pbr_ambient)
        ext = Exterior(pitch, lattice, h, pitch, e_sils, e_shade)
        print(f"[target] exterior: pitch {pitch:.4f} wu, lattice {h / pitch:.2f} pitches = "
              f"{h / (2.0 * extent / cfg.render_res):.2f} render pixels, {len(p_t)} discs on the target", flush=True)
    # the W1 cleanup's fine target-fitted grid (1.5 extents each way, the box leash's range)
    dtdims = (cfg.dt_res,) * 3
    dtdx = 3.0 * extent / cfg.dt_res
    dtgmin = torch.tensor([-1.5 * extent] * 3, device=gpu.DEVICE)
    dt3 = target_dt_grid(target_mass_grid(tgt_t, m, dtgmin, dtdx, dtdims), dtdx, dtdims,
                         clamp=cfg.dt_clamp_frac * extent)
    knn = gpu.KNN(tgt_t)
    nn_sp = gpu.median(knn.query(tgt_t, 2)[0][:, 1])
    m_ref, n_support = density_units(grid)
    return TargetPack(grid=grid, lgmin=lgmin, ldx=ldx, ldims=ldims, m=m, views=views, sils=sils,
                      extent=extent, shade=shade, pgmin=lgmin, pdx=pdx, pdims=pdims, pblur=pblur,
                      dt3=dt3, dtgmin=dtgmin, dtdx=dtdx, dtdims=dtdims, pts=tgt_t, knn=knn,
                      nn_spacing=nn_sp, m_ref=m_ref, n_support=n_support, support=support, gate=gate, ext=ext)


def rebuild_for_resolution(tgt: TargetPack, target_x, prm: MPMParams,
                           cfg: PipelineConfig) -> TargetPack:
    """Coarse-to-fine: a fresh target at cfg.render_res. The unit ratios and the transport
    scale and solver survive (no weight changes silently at the switch); the render weight
    and the step warm start are recalibrated for the new images."""
    new = build_target(target_x, prm, cfg)
    new.unit_ratio, new.unit_grad_ratio = tgt.unit_ratio, tgt.unit_grad_ratio
    new.ot_scale, new.grid_ot = tgt.ot_scale, tgt.grid_ot
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
        grid_ref = target_mass_grid(tgt.pts, tgt.m, tgt.lgmin, ldx_ref, dims_ref)
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
