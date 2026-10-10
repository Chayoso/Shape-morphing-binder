"""The prepare stage: the only CPU work of a run, cached on disk.

Meshes are loaded and volume-sampled with trimesh (stratified: one jittered particle per
fill voxel; the samples are cached by physmorph.sampling). The target is rescaled to the
source's volume, since the MPM body cannot change its total volume. The discretisation
follows from the shape: dx = source bounding-box diagonal / cell_diag (D137, cell_ref_n: x (cell_ref_n / N)^(1/3)
above cell_ref_n particles), the domain is the
box leash (1.25 x the larger cloud) plus the 4^3 stencil margin, and the loss grid
follows dx. The near-band berth is the target's median 8th-neighbour distance in
nearest-neighbour spacings. Everything after this runs on the GPU.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from . import gpu
from .mpm.discretisation import derive, report
from .mpm.state import MPMParams
from .sampling import load_normalized
from .sampling.mesh import draws_in_frame, surface_in_frame


@dataclass
class Prepared:
    src: np.ndarray
    tgt: np.ndarray
    v_src: float
    v_tgt: float
    prm: MPMParams
    loss_res: int
    unit_ref_res: int
    nn_berth_k: float
    ppc: float
    tgt_surface: tuple | None = None    # (points, normals) of the target mesh's surface in the target's frame
    tgt_draws: list | None = None       # further independent samples of the target in its frame (the render's target)
    density: dict | None = None         # D132 (match_density): the two samples' fill volumes and the source's scale
    cell_shape: float | None = None     # D137 (cell_ref_n, above it): the shape's MPM cell (diag / cell_diag, as derived
                                        #   before), kept for the u gate and the thin set; None = prm.dx is the shape's cell


def sampling_berth(target: np.ndarray) -> float:
    """The near-band berth in target spacings: median 8th-neighbour distance / median
    nearest-neighbour distance (the sampling's own scale)."""
    target = np.asarray(target)
    if target.ndim != 2 or target.shape[1] != 3 or len(target) < 9 or not np.isfinite(target).all():
        raise ValueError("the sampling berth needs at least nine finite target points")
    d = gpu.knn(gpu.tensor(target), 9)[0]
    spacing = gpu.median(d[:, 1])
    if not spacing > 0:
        raise ValueError("the sampling berth needs a positive target nearest-neighbour spacing")
    ratio = gpu.median(d[:, 8]) / spacing
    if not np.isfinite(ratio) or not ratio > 0:
        raise ValueError("the sampling berth must be finite and positive")
    return float(ratio)


def prepare(src_path: str, tgt_path: str, n: int, seed: int, cell_diag: float, young: float,
            poisson: float, log=print, loss_ref_n: int = 0, floor: bool = False, surface: int = 0,
            draws: int = 1, match_density: bool = False, cell_ref_n: int = 0) -> Prepared:
    fill_src, fill_tgt = ({}, {}) if match_density else (None, None)
    src, v_src = load_normalized(src_path, n, seed, return_volume=True, sample="stratified", fill=fill_src)
    frame = {}
    tgt, v_tgt = load_normalized(tgt_path, n, seed + 1, match_volume=v_src, sample="stratified",
                                 return_volume=True, frame=frame, fill=fill_tgt)
    density = None
    if match_density:
        # D132: the target is matched to the source by the meshes' volumes at a 110^3 fill, but each sample is one
        # jittered particle per voxel of its own, coarser fill, whose volume differs from the 110^3 one by a surface
        # term of the shape: the two samples' number densities n / fill volume differ (the 40k bunny's target 1.2 %
        # below the source's in the deep interior). The body conserves its volume (--volume_exact), so it cannot reach
        # a target sample of another density. The source is rescaled about its centre so that its sample represents
        # the target sample's volume: equal densities; the target, its frame and every measure on it are unchanged
        ratio = fill_tgt["volume"] / fill_src["volume"]
        k_src = ratio ** (1.0 / 3.0)
        src = (src * k_src).astype(np.float32)
        v_src = v_src * ratio
        density = {"source_fill": fill_src["volume"], "target_fill": fill_tgt["volume"], "source_scale": k_src}
        log(f"[v2run] density match: the samples' fill volumes source {fill_src['volume']:.4f}, target "
            f"{fill_tgt['volume']:.4f} wu^3 (target / source density {1.0 / ratio:.4f}); the source rescaled by "
            f"{k_src:.5f}")
    # surface > 0: that many points of the target mesh's own surface, for the relaxation's reference
    tgt_surface = surface_in_frame(tgt_path, frame, surface) if surface > 0 else None
    # draws > 1: that many samples of the target in all, the pipeline's and draws - 1 further independent ones (seeds
    # seed + 1 + 100 k; seed + 2 is left to the evaluation's independent reference), for the render's target
    tgt_draws = (draws_in_frame(tgt_path, frame, n, [seed + 1 + 100 * k for k in range(1, draws)])
                 if draws > 1 else None)
    floor_y = None
    if floor:                                 # both shapes stand on one floor, at the source's lowest point
        floor_y = float(src[:, 1].min())
        tgt = tgt.copy()
        lift = floor_y - float(tgt[:, 1].min())
        tgt[:, 1] += lift
        if tgt_surface is not None:
            tgt_surface[0][:, 1] += lift
        for d in tgt_draws or []:
            d[:, 1] += lift
        log(f"[v2run] floor at y = {floor_y:.3f} wu; the target stands on it")
    log(f"[v2run] volumes: source {v_src:.2f} target(matched) {v_tgt:.2f} wu^3 "
        f"(target bbox diag now {float(np.linalg.norm(tgt.max(0) - tgt.min(0))):.2f})")
    prm = MPMParams()
    g_src, g_tgt = src, tgt
    diag_src = float(np.linalg.norm(g_src.max(0) - g_src.min(0)))
    dx_req = diag_src / float(cell_diag)
    ppc = float(n * dx_req ** 3 / v_src)
    log(f"[disc] cell from the shape: dx = diag {diag_src:.3f} / {cell_diag:g} = {dx_req:.4f} wu "
        f"-> ppc = N dx^3 / V = {ppc:.1f}")
    # the domain: the box leash the objective assumes (1.25 x max |x|, w_box pulls back
    # inside it) plus the 4^3 stencil margin; nothing outside receives grid forces
    dx0 = float((v_src * ppc / n) ** (1.0 / 3.0))
    leash = 1.25 * float(max(np.abs(g_src).max(), np.abs(g_tgt).max()))
    domain_half = leash + 2.0 * dx0
    # the loss grid: the MPM cell, or (loss_ref_n > 0) refined with the particle spacing above
    # loss_ref_n particles, so the transport resolves what the sampling resolves
    per_dx = max(1.0, (n / loss_ref_n) ** (1.0 / 3.0)) if loss_ref_n > 0 else 1.0
    # D137 (cell_ref_n > 0): above cell_ref_n particles the MPM cell follows the particle count as the loss grid does,
    # dx = (diag / cell_diag) x (cell_ref_n / N)^(1/3): the particles per cell stay cell_ref_n's, and a gap of the target
    # a few of the shape's cells wide is no longer inside the cubic kernel's reach of both its sides (D137: at 300k
    # the shape's cell is 6 pitches and the dragon's gaps 1.7-4.5 cells, so their material moved with both rims and
    # tore into beads). The loss grid stays exactly as it was (about one loss cell per finer MPM cell); the shape's cell is kept for
    # the u gate and the thin set (Prepared.cell_shape). At or below cell_ref_n nothing changes.
    k_cell = max(1.0, (n / cell_ref_n) ** (1.0 / 3.0)) if cell_ref_n > 0 else 1.0
    cell_shape = None
    disc = derive(n, v_src, diag_src, prm.dt, young, poisson, ppc=ppc, domain_half=domain_half,
                  loss_cells_per_dx=per_dx)
    if k_cell > 1.0:
        cell_shape, loss_res = disc.dx, disc.loss_res  # the shape's cell and the loss grid, exactly as before
        ppc = ppc / k_cell ** 3                       # the domain (its margin two of the shape's cells) as before
        disc = dataclasses.replace(derive(n, v_src, diag_src, prm.dt, young, poisson, ppc=ppc,
                                          domain_half=domain_half), loss_res=loss_res)
        log(f"[disc] the cell follows N above {cell_ref_n} (D137): dx / {k_cell:.4f} -> ppc {ppc:.1f}; the shape's "
            f"cell {cell_shape:.4f} wu kept for the u gate and the thin set, the loss grid {loss_res}^3 as before")
    prm = dataclasses.replace(prm, dx=disc.dx, nx=disc.grid_n, ny=disc.grid_n, nz=disc.grid_n,
                              grid_min=(disc.grid_min,) * 3)
    if floor_y is not None:
        prm = dataclasses.replace(prm, floor_y=floor_y)
    # the unit calibration measures the legacy ratio on a 0.5 wu reference cell (the grid
    # every legacy weight was tuned on)
    unit_ref_res = int(round(2.0 * domain_half / 0.5))
    log(f"[disc] domain auto: half-width {domain_half:.2f} wu (leash {leash:.2f} + 2 dx), "
        f"grid {disc.grid_n}^3 = {disc.grid_n ** 3 / 1e6:.2f} M cells, unit_ref_res {unit_ref_res} "
        "(0.5 wu reference cell)")
    log(report(disc, src))
    log(report(disc, tgt).splitlines()[-1].replace("[disc] measured", "[disc] TARGET measured"))
    log(f"[disc] loss_res {disc.loss_res} (" + (f"{per_dx:.3f} loss cells per {'shape ' if cell_shape else ''}dx, following N above {loss_ref_n}"
                                                if per_dx > 1.0 else "the MPM cell") + ")")
    berth = sampling_berth(tgt)
    log(f"[v2run] sampling-scale NN berth: nn_berth_k={berth:.17g}")
    return Prepared(src=src, tgt=tgt, v_src=float(v_src), v_tgt=float(v_tgt), prm=prm,
                    loss_res=int(disc.loss_res), unit_ref_res=unit_ref_res, nn_berth_k=berth, ppc=ppc,
                    tgt_surface=tgt_surface, tgt_draws=tgt_draws, density=density, cell_shape=cell_shape)
