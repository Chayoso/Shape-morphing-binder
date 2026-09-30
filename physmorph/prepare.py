"""The prepare stage: the only CPU work of a run, cached on disk.

Meshes are loaded and volume-sampled with trimesh (stratified: one jittered particle per
fill voxel; the samples are cached by physmorph.sampling). The target is rescaled to the
source's volume, since the MPM body cannot change its total volume. The discretisation
follows from the shape: dx = source bounding-box diagonal / cell_diag, the domain is the
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


def sampling_berth(target: np.ndarray, far_k: float) -> float:
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
    if not np.isfinite(ratio) or not 0 < ratio < far_k:
        raise ValueError("the sampling berth must be finite and smaller than nn_far_k")
    return float(ratio)


def prepare(src_path: str, tgt_path: str, n: int, seed: int, cell_diag: float, young: float,
            poisson: float, far_k: float, log=print) -> Prepared:
    src, v_src = load_normalized(src_path, n, seed, return_volume=True, sample="stratified")
    tgt, v_tgt = load_normalized(tgt_path, n, seed + 1, match_volume=v_src, sample="stratified",
                                 return_volume=True)
    log(f"[v2run] volumes: source {v_src:.2f} target(matched) {v_tgt:.2f} wu^3 "
        f"(target bbox diag now {float(np.linalg.norm(tgt.max(0) - tgt.min(0))):.2f})")
    prm = MPMParams()
    diag_src = float(np.linalg.norm(src.max(0) - src.min(0)))
    dx_req = diag_src / float(cell_diag)
    ppc = float(n * dx_req ** 3 / v_src)
    log(f"[disc] cell from the shape: dx = diag {diag_src:.3f} / {cell_diag:g} = {dx_req:.4f} wu "
        f"-> ppc = N dx^3 / V = {ppc:.1f}")
    # the domain: the box leash the objective assumes (1.25 x max |x|, w_box pulls back
    # inside it) plus the 4^3 stencil margin; nothing outside receives grid forces
    dx0 = float((v_src * ppc / n) ** (1.0 / 3.0))
    leash = 1.25 * float(max(np.abs(src).max(), np.abs(tgt).max()))
    domain_half = leash + 2.0 * dx0
    disc = derive(n, v_src, diag_src, prm.dt, young, poisson, ppc=ppc, domain_half=domain_half)
    prm = dataclasses.replace(prm, dx=disc.dx, nx=disc.grid_n, ny=disc.grid_n, nz=disc.grid_n,
                              grid_min=(disc.grid_min,) * 3)
    # the unit calibration measures the legacy ratio on a 0.5 wu reference cell (the grid
    # every legacy weight was tuned on)
    unit_ref_res = int(round(2.0 * domain_half / 0.5))
    log(f"[disc] domain auto: half-width {domain_half:.2f} wu (leash {leash:.2f} + 2 dx), "
        f"grid {disc.grid_n}^3 = {disc.grid_n ** 3 / 1e6:.2f} M cells, unit_ref_res {unit_ref_res} "
        "(0.5 wu reference cell)")
    log(report(disc, src))
    log(report(disc, tgt).splitlines()[-1].replace("[disc] measured", "[disc] TARGET measured"))
    log(f"[disc] loss_res follows dx: {disc.loss_res}")
    berth = sampling_berth(tgt, far_k)
    log(f"[v2run] sampling-scale NN berth: nn_berth_k={berth:.17g}")
    return Prepared(src=src, tgt=tgt, v_src=float(v_src), v_tgt=float(v_tgt), prm=prm,
                    loss_res=int(disc.loss_res), unit_ref_res=unit_ref_res, nn_berth_k=berth, ppc=ppc)
