"""Discretisation contract: derive dx, grid, loss grid and Gaussian size FROM N.

Why (docs/render_controls_physics.md §7). Two of the reported failure modes are
discretisation-level, not optimiser-level:
  * HOLES at low particle count — MLS-MPM needs a minimum particles-per-cell (ppc);
    below ~4 the cubic B-spline stencil loses its partition of unity locally, grid
    nodes with near-zero mass produce runaway velocities (the "small node mass"
    pathology, Steffen et al. 2008) and material tears numerically. Xu et al.'s
    morphing configs use ppc 6; the MPM literature recommends 8 for 3-D.
  * GRID-RESOLUTION SENSITIVITY — the same N at a finer dx has fewer ppc; every
    measured trend that "changes with the grid" is a ppc trend in disguise.
The contract makes dx a function of N and the sampled volume so that ppc is what the
user asked for; the loss grid and the Gaussian rest size follow from the same numbers.
It also reports the explicit-MPM stability numbers (sound speed, CFL) that the
oscillation triage reads (docs/oscillation_triage.md).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class Discretisation:
    N: int
    volume: float               # world units^3 (filled voxel volume of the source)
    ppc: float                  # particles per MPM cell, requested
    dx: float                   # MPM cell size
    grid_n: int                 # nodes per axis (cube)
    grid_min: float             # cube origin (same on every axis)
    loss_res: int               # D_vol raster resolution (cell ~ dx)
    spacing: float              # mean particle spacing (V/N)^(1/3)
    sigma0: float               # Gaussian rest size = sigma_scale * spacing
    rho: float                  # mass density (unit particle mass)
    sound_speed: float
    cfl: float                  # c * dt / dx
    elastic_period: float       # 2 L / c (seconds), L = body extent
    kernel_support_ratio: float # (2 dx) / spacing — must be >> 1 for no holes

    def asdict(self):
        return asdict(self)


def derive(N: int, volume: float, extent: float, dt: float, young: float, poisson: float,
           ppc: float = 8.0, domain_half: float = 16.0, sigma_scale: float = 1.0,
           mass: float = 1.0, loss_cells_per_dx: float = 1.0) -> Discretisation:
    """Choose dx so that a uniformly sampled body of `volume` has `ppc` particles per
    dx^3 cell; the domain cube [-domain_half, domain_half]^3 then fixes the node count.

    extent: body extent (bbox diagonal) — only used for the elastic period.
    Raises when the request is inconsistent (ppc < 1, N < 8, non-positive volume)."""
    if N < 8 or volume <= 0 or ppc < 1:
        raise ValueError("need N >= 8, volume > 0, ppc >= 1")
    dx = float((volume * ppc / N) ** (1.0 / 3.0))
    grid_n = int(np.ceil(2.0 * domain_half / dx))
    dx = 2.0 * domain_half / grid_n                     # exact tiling of the domain
    spacing = float((volume / N) ** (1.0 / 3.0))
    loss_res = int(np.ceil(grid_n * loss_cells_per_dx))
    rho = float(N * mass / volume)
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    mu = young / (2 * (1 + poisson))
    c = float(np.sqrt((lam + 2 * mu) / rho))
    cfl = float(c * dt / dx)
    return Discretisation(N=int(N), volume=float(volume), ppc=float(ppc), dx=float(dx),
                          grid_n=int(grid_n), grid_min=float(-domain_half),
                          loss_res=int(loss_res), spacing=spacing,
                          sigma0=float(sigma_scale * spacing), rho=rho, sound_speed=c,
                          cfl=cfl, elastic_period=float(2.0 * extent / c),
                          kernel_support_ratio=float(2.0 * dx / spacing))


def measure_ppc(x: np.ndarray, dx: float, grid_min: float) -> dict:
    """Occupancy statistics of a cloud on a dx grid: mean/median particles per
    OCCUPIED cell, the fraction of occupied cells below 4 ppc (hole risk) and the
    max nearest-neighbour spacing over the kernel support (2 dx)."""
    from scipy.spatial import cKDTree
    x = np.ascontiguousarray(x, np.float32)
    cell = np.floor((x - grid_min) / dx).astype(np.int64)
    _, counts = np.unique(cell, axis=0, return_counts=True)
    nn = cKDTree(x).query(x, k=2, workers=-1)[0][:, 1] if len(x) > 1 else np.zeros(1)
    return {"ppc_mean": float(counts.mean()), "ppc_median": float(np.median(counts)),
            "occupied_cells": int(len(counts)),
            "frac_cells_below_4ppc": float((counts < 4).mean()),
            "nn_max_over_support": float(nn.max() / (2.0 * dx)),
            "nn_median": float(np.median(nn))}


def report(d: Discretisation, x: np.ndarray | None = None) -> str:
    lines = [f"[disc] N={d.N} V={d.volume:.3f} ppc={d.ppc:g} -> dx={d.dx:.4f} "
             f"grid={d.grid_n}^3 loss_res={d.loss_res} spacing={d.spacing:.4f} "
             f"sigma0={d.sigma0:.4f}",
             f"[disc] rho={d.rho:.2f} c={d.sound_speed:.2f} CFL={d.cfl:.3f} "
             f"elastic period={d.elastic_period*1e3:.1f} ms  support/spacing="
             f"{d.kernel_support_ratio:.2f}"]
    if d.cfl > 0.5:
        lines.append("[disc] WARNING: CFL > 0.5 — explicit MPM will ring or blow up; "
                     "raise dx (lower ppc target), lower dt or lower E")
    if d.kernel_support_ratio < 3.0:
        lines.append("[disc] WARNING: kernel support < 3 spacings — hole risk")
    if x is not None:
        m = measure_ppc(x, d.dx, d.grid_min)
        lines.append(f"[disc] measured ppc mean={m['ppc_mean']:.2f} median="
                     f"{m['ppc_median']:.1f} cells<4ppc={m['frac_cells_below_4ppc']*100:.1f}% "
                     f"nn_max/support={m['nn_max_over_support']:.2f}")
    return "\n".join(lines)
