"""The thin part of the target, and how well a body covers it (measurement only, not in the objective).

Thin set: the target's outer points whose local feature thickness is below two MPM cells. The local thickness
of a point is twice the radius of the largest ball inside the sampled target volume that contains it
(Hildebrand & Ruegsegger 1997), measured on a voxel grid at the target spacing. The MPM kernel and the default
transport grid both work at the cell, so a feature thinner than two cells (two samples across it) is below their
resolution. Outer points: fewer than 0.6 of the median count of target points within two spacings (the census
definition of scripts/probes/settled/thin_regions.py, kept for comparability).

Metrics on the thin set: the share of its points farther than 1.5 target spacings from the body, the same at one
world distance for every N (1.5 spacings of a ref_n sampling), and the gap to the body in spacings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import gpu


def local_thickness(tgt: torch.Tensor, sp: float) -> torch.Tensor:
    """h at every target point (world units): max-ball thickness on a voxel grid of the target spacing."""
    from .render.knn_gpu import knn_self_torch
    lo = tgt.min(0).values - 3 * sp
    dims = [int(math.ceil(float(v))) for v in ((tgt.max(0).values + 3 * sp - lo) / sp)]
    axes = [lo[i] + sp * (torch.arange(dims[i], device=tgt.device) + 0.5) for i in range(3)]
    centers = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    tree = gpu.KNN(tgt)
    d = torch.cat([tree.query(c, 1)[0][:, 0] for c in centers.split(2_000_000)])
    # inside the sampled volume: within the sampling's covering radius (0.75 x the median 8th-neighbour
    # distance, about 1.06 fill pitches on a jittered lattice); interior voxels the jitter misses are enclosed
    # and filled, or h would be capped by the distance to such a hole
    d8 = knn_self_torch(tgt, 9)[0][:, 8]
    r_cover = 0.75 * gpu.median(d8)
    occ = gpu.fill_holes((d <= r_cover).reshape(dims))
    D = gpu.edt(occ)                                           # voxels to the outside
    thick = torch.zeros(dims, dtype=torch.float64, device=tgt.device)
    for r in range(1, int(D.max()) + 1):                       # balls of radius r cover thickness >= 2r
        core = D >= r
        if not bool(core.any()):
            break
        thick[(gpu.edt(~core) <= r) & occ] = 2.0 * r * sp
    ijk = ((tgt - lo) / sp).long()
    for i in range(3):
        ijk[:, i] = ijk[:, i].clamp(0, dims[i] - 1)
    h = thick[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    return torch.where(h > 0, h, torch.full_like(h, 2 * sp))   # a point off the closed volume: thinnest


def outer_mask(P: torch.Tensor, sp: float) -> torch.Tensor:
    """Outer points: fewer than 0.6 of the median count of points within two spacings."""
    from .render.knn_gpu import knn_self_torch
    d, _ = knn_self_torch(P, 41)
    c = (d[:, 1:] < 2.0 * sp).sum(1).float()
    return c < 0.6 * c.median()


@dataclass
class ThinSet:
    points: torch.Tensor          # (M,3) thin outer target points
    thickness: torch.Tensor       # (M,) in MPM cells
    spacing: float                # target median nearest-neighbour spacing
    world: float                  # 1.5 spacings of a ref_n sampling (one distance for every N)
    n_outer: int


def thin_set(target, cell: float, ref_n: int) -> ThinSet:
    tgt = gpu.tensor(target)
    d1 = gpu.knn(tgt, 2)[0][:, 1]
    sp = gpu.median(d1)
    h = local_thickness(tgt, sp) / cell
    outer = outer_mask(tgt, sp)
    sel = outer & (h < 2.0)
    world = 1.5 * sp * (len(tgt) / float(ref_n)) ** (1.0 / 3.0)
    return ThinSet(points=tgt[sel], thickness=h[sel], spacing=sp, world=world, n_outer=int(outer.sum()))


def thin_set_cached(target, cell: float, ref_n: int) -> ThinSet:
    """thin_set, cached on disk beside the samples (D174, exact: the set is a deterministic function of the target
    sample, the cell and ref_n, which key it; 20 s of every 300k run's start, a measurement the run never reads before
    its end). PHYSMORPH_SAMPLE_CACHE=0 disables it, as the sample cache."""
    import hashlib
    import os

    import numpy as np
    arr = np.ascontiguousarray(np.asarray(target, dtype=np.float32))
    cache_on = os.environ.get("PHYSMORPH_SAMPLE_CACHE", "1") != "0"
    key = hashlib.sha1(arr.tobytes() + f"|{arr.shape}|{float(cell)!r}|{int(ref_n)}|thin-v1".encode()).hexdigest()[:20]
    cdir = os.environ.get("PHYSMORPH_CACHE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                          "output", "cache"))
    cpath = os.path.join(cdir, f"thin_{len(arr)}_{key}.npz")
    if cache_on and os.path.exists(cpath):
        try:
            z = np.load(cpath)
            return ThinSet(points=gpu.tensor(z["points"]), thickness=torch.as_tensor(z["thickness"], device=gpu.DEVICE),
                           spacing=float(z["spacing"]), world=float(z["world"]), n_outer=int(z["n_outer"]))
        except Exception:
            pass
    ts = thin_set(arr, cell, ref_n)
    if cache_on:
        try:
            os.makedirs(cdir, exist_ok=True)
            tmp = f"{cpath}.{os.getpid()}.tmp.npz"
            np.savez(tmp, points=ts.points.detach().cpu().numpy(), thickness=ts.thickness.detach().cpu().numpy(),
                     spacing=np.float64(ts.spacing), world=np.float64(ts.world), n_outer=np.int64(ts.n_outer))
            os.replace(tmp, cpath)
        except Exception:
            pass
    return ts


def thin_metrics(x, ts: ThinSet) -> dict:
    """Coverage of the thin set by body positions x: uncovered shares (own and world threshold, also per
    thickness bin < 1 and 1-2 cells) and the gap of the uncovered points in target spacings."""
    if not len(ts.points):
        return {"thin_n": 0}
    d = gpu.KNN(gpu.tensor(x)).query(ts.points, 1)[0][:, 0]
    far = d > 1.5 * ts.spacing
    out = {"thin_n": int(len(ts.points)), "thin_share_of_outer": len(ts.points) / max(ts.n_outer, 1),
           "thin_uncovered": float(far.double().mean()),
           "thin_uncovered_2sp": float((d > 2.0 * ts.spacing).double().mean()),   # gaps a soft threshold near 1.5
           "thin_uncovered_world": float((d > ts.world).double().mean())}         # spacings cannot park points at
    for name, lo, hi in (("lt1", 0.0, 1.0), ("1to2", 1.0, 2.0)):
        b = (ts.thickness >= lo) & (ts.thickness < hi)
        out[f"thin_uncovered_{name}"] = float(far[b].double().mean()) if bool(b.any()) else None
    if bool(far.any()):
        g = (d[far] / ts.spacing).double()
        q = torch.quantile(g, torch.tensor([.5, .9], dtype=g.dtype, device=g.device))
        out.update(thin_gap_median_sp=float(q[0]), thin_gap_p90_sp=float(q[1]))
    return out
