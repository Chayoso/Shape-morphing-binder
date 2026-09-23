"""Exact k-nearest neighbours on the GPU (2026-09-23; docs/experiments.md, the speed pass).

Every per-window neighbour search of the pipeline — the outer layer by asymmetry (32-NN), the
relaxation neighbourhoods (24-NN on the layer), the OT displacement smoothing (k_nb-NN), the DT
isolation gate (8-NN), the fragment mask and the stray census — was a scipy cKDTree query on the
CPU: 1–3 s each at 300k particles, the GPU idle meanwhile, and 5–10× slower again when other
users load the host. This module answers the same query on the GPU with a Warp hash grid: a
radius query with the k best kept in a per-row insertion sort, the radius doubled until every row
has k candidates, scipy for any row still short (never observed on a cloud). Same rows as
`cKDTree.query(x, k)` (self first, distances ascending) up to float32 ties. `PHYSMORPH_KNN=cpu`
restores scipy everywhere.
"""
from __future__ import annotations

import os

import numpy as np

_CPU = os.environ.get("PHYSMORPH_KNN", "") == "cpu"
_state: dict = {"kernel": None, "ok": None}

try:  # warp is a hard dependency of the MPM; the kernel is compiled on first use
    import warp as wp

    @wp.kernel
    def _k_knn(grid: wp.uint64, pts: wp.array(dtype=wp.vec3), radius: float, k: int,
               out_i: wp.array2d(dtype=int), out_d: wp.array2d(dtype=float), cnt: wp.array(dtype=int)):
        i = wp.tid()
        p = pts[i]
        r2 = radius * radius
        n = int(0)
        q = wp.hash_grid_query(grid, p, radius)
        j = int(0)
        while wp.hash_grid_query_next(q, j):
            d2 = wp.length_sq(pts[j] - p)
            if d2 <= r2:
                take = int(0)
                if n < k:
                    take = 1
                elif d2 < out_d[i, k - 1]:
                    take = 1
                if take == 1:
                    pos = int(0)
                    if n < k:
                        pos = n
                        n = n + 1
                    else:
                        pos = k - 1
                    moving = int(1)
                    while moving == 1:
                        if pos > 0:
                            if out_d[i, pos - 1] > d2:
                                out_d[i, pos] = out_d[i, pos - 1]
                                out_i[i, pos] = out_i[i, pos - 1]
                                pos = pos - 1
                            else:
                                moving = 0
                        else:
                            moving = 0
                    out_d[i, pos] = d2
                    out_i[i, pos] = j
        cnt[i] = n

    _state["kernel"] = _k_knn
except Exception:  # pragma: no cover - no warp: scipy only
    wp = None


def _cpu_knn(x: np.ndarray, k: int):
    from scipy.spatial import cKDTree
    return cKDTree(x).query(x, k=k, workers=-1)


def _gpu_ready() -> bool:
    if _state["ok"] is None:
        try:
            import torch
            _state["ok"] = bool(wp is not None and _state["kernel"] is not None and torch.cuda.is_available())
            if _state["ok"]:
                wp.init()
        except Exception:
            _state["ok"] = False
    return bool(_state["ok"])


def knn_self(x, k: int, device: str = "cuda"):
    """(d, idx) of the k nearest points of every point of x within x itself, self included as
    column 0 — the rows of `cKDTree(x).query(x, k)`. x: (N,3) array-like."""
    x = np.ascontiguousarray(np.asarray(x, np.float32))
    N = int(len(x))
    if _CPU or N < 4096 or N <= k or not _gpu_ready():
        return _cpu_knn(x, k)
    kern = _state["kernel"]
    pts = wp.array(x, dtype=wp.vec3, device=device)
    lo, hi = x.min(0), x.max(0)
    ext = float((hi - lo).max())
    vol = float(np.prod(np.maximum(hi - lo, 1e-9)))
    pitch = (vol / N) ** (1.0 / 3.0)
    # the ball holding k points at the cloud's mean density, times 1.5: a surface point has half a ball
    R = 1.5 * pitch * (3.0 * k / (4.0 * np.pi)) ** (1.0 / 3.0)
    dim = int(min(256, max(32, 2 * int(np.ceil(N ** (1.0 / 3.0))))))
    grid = wp.HashGrid(dim, dim, dim, device=device)
    out_i = out_d = cnt = None
    c = None
    for _ in range(6):
        out_i = wp.zeros((N, k), dtype=int, device=device)
        out_d = wp.zeros((N, k), dtype=float, device=device)
        cnt = wp.zeros(N, dtype=int, device=device)
        grid.build(points=pts, radius=float(R))
        wp.launch(kern, dim=N, inputs=[grid.id, pts, float(R), int(k), out_i, out_d, cnt], device=device)
        c = cnt.numpy()
        if int(c.min()) >= k:
            break
        R *= 2.0
        if R > 4.0 * ext:
            break
    d = np.sqrt(out_d.numpy().astype(np.float64))
    idx = out_i.numpy().astype(np.int64)
    short = c < k
    if short.any():
        from scipy.spatial import cKDTree
        dd, ii = cKDTree(x).query(x[short], k=k, workers=-1)
        d[short] = dd
        idx[short] = ii
    return d, idx
