"""Exact k-nearest neighbours on the GPU (2026-09-23; docs/experiments.md, the speed pass).

Every per-window neighbour search of the pipeline — the outer layer by asymmetry (32-NN), the
relaxation neighbourhoods (24-NN on the layer), the OT displacement smoothing (k_nb-NN), the DT
isolation gate (8-NN), the fragment mask and the stray census — was a scipy cKDTree query on the
CPU: 1–3 s each at 300k particles, the GPU idle meanwhile, and 5–10× slower again when other
users load the host. This module answers the same query on the GPU with a Warp hash grid: a
radius query with the k best kept in a per-row insertion sort, the radius doubled until every row
has k candidates, with bounded GPU chunks for any row still short. Same rows as
`cKDTree.query(x, k)` (self first, distances ascending) up to float32 ties. `knn_self_torch`
keeps the result on the device (no 40 MB host copies per call); `knn_self` returns numpy.
`PHYSMORPH_KNN=cpu` restores scipy everywhere.
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


def gpu_available() -> bool:
    if _CPU:
        return False
    if _state["ok"] is None:
        try:
            import torch
            _state["ok"] = bool(wp is not None and _state["kernel"] is not None and torch.cuda.is_available())
            if _state["ok"]:
                wp.init()
        except Exception:
            _state["ok"] = False
    return bool(_state["ok"])


def knn_self_torch(x_t, k: int):
    """(d, idx) CUDA tensors (float32 distances, int64 indices) of the k nearest points of every point
    of the CUDA tensor x_t (N,3) within itself, self included as column 0 — the rows of
    `cKDTree(x).query(x, k)`, without a host round-trip."""
    import torch
    k = int(k)
    if k < 1:
        raise ValueError('k must be positive')
    if x_t.ndim != 2 or x_t.shape[1] != 3:
        raise ValueError('positions must have shape (N,3)')
    if not bool(torch.isfinite(x_t).all()):
        raise ValueError('positions must be finite')
    N = int(x_t.shape[0])
    if not x_t.is_cuda:
        d, i = _cpu_knn(x_t.detach().cpu().numpy().astype(np.float32), k)
        return (torch.as_tensor(np.asarray(d, np.float32), device=x_t.device),
                torch.as_tensor(np.asarray(i, np.int64), device=x_t.device))
    if not gpu_available():
        raise RuntimeError('CUDA kNN is unavailable; CPU fallback is disabled for CUDA tensors')
    if N < 4096 or N <= k:
        return _knn_chunked(x_t, x_t, k, self_query=True)
    kern = _state["kernel"]
    device = str(x_t.device)
    xt = x_t.detach().contiguous().float()
    pts = wp.from_torch(xt, dtype=wp.vec3)
    extent = xt.max(0).values - xt.min(0).values
    ext = float(extent.max())
    vol = float(extent.clamp_min(1e-9).prod())
    pitch = (vol / N) ** (1.0 / 3.0)
    # the ball holding k points at the bbox mean density, times 1.5 (a surface point has half a ball)
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
        c = wp.to_torch(cnt)
        if int(c.min()) >= k:
            break
        R *= 2.0
        if R > 4.0 * ext:
            break
    d = torch.sqrt(wp.to_torch(out_d).clone())
    idx = wp.to_torch(out_i).clone().long()
    short = c < k
    if bool(short.any()):
        dd, ii = _knn_chunked(xt, xt[short], k)
        d[short], idx[short] = dd, ii
    return _self_first(d, idx)


def _self_first(distances, indices):
    """Drop-column-zero consumers require the actual self ID, even at coincident points."""
    import torch
    ids = torch.arange(len(indices), device=indices.device)
    rows = torch.nonzero(indices[:, 0] != ids).squeeze(1)
    if not len(rows):
        return distances, indices
    selected = indices[rows]
    matches = selected == rows[:, None]
    width = indices.shape[1]
    ranks = torch.arange(width, device=indices.device)[None] - matches.long() * width
    order = ranks.argsort(1)
    selected_d = distances[rows].gather(1, order)
    selected_i = selected.gather(1, order)
    missing = ~matches.any(1)
    # When >k particles coincide, the bounded nearest set may omit the self ID.
    selected_d[missing] = torch.cat((torch.zeros(len(rows), 1, device=distances.device,
                                               dtype=distances.dtype), selected_d[:, :-1]), 1)[missing]
    selected_i[missing] = torch.cat((rows[:, None], selected_i[:, :-1]), 1)[missing]
    distances[rows], indices[rows] = selected_d, selected_i
    return distances, indices


def _knn_chunked(reference, queries, k, *, self_query=False):
    """Exact CUDA fallback with bounded temporary storage, including k > N padding.

    Used for small clouds and isolated rows, never SciPy. Double precision avoids
    cancellation near coincident points. Self is first when distances tie.
    """
    import torch
    k = int(k)
    if k < 1:
        raise ValueError('k must be positive')
    if not reference.is_cuda or queries.device != reference.device:
        raise ValueError('reference and queries must share a CUDA device')
    nr, nq = len(reference), len(queries)
    take = min(k, nr)
    result_d = torch.full((nq, k), float('inf'), device=queries.device, dtype=torch.float64)
    result_i = torch.full((nq, k), nr, device=queries.device, dtype=torch.long)
    if not nr or not nq:
        return result_d.float(), result_i
    ref = reference.double()
    for qs in range(0, nq, 128):
        query = queries[qs:qs + 128].double()
        best_d = result_d[qs:qs + len(query), :take].clone()
        best_i = result_i[qs:qs + len(query), :take].clone()
        for rs in range(0, nr, 8192):
            # Direct differences preserve small distances (no squared-norm subtraction).
            delta = query[:, None, :] - ref[None, rs:rs + 8192, :]
            distances = delta.square().sum(-1)
            indices = torch.arange(rs, rs + distances.shape[1], device=queries.device)
            if self_query:
                row = torch.arange(qs, qs + len(query), device=queries.device)
                distances = torch.where(row[:, None] == indices[None], -1., distances)
            merged_d = torch.cat((best_d, distances), dim=1)
            merged_i = torch.cat((best_i, indices[None].expand(len(query), -1)), dim=1)
            order = torch.argsort(merged_d, dim=1, stable=True)[:, :take]
            best_d = merged_d.gather(1, order)
            best_i = merged_i.gather(1, order)
        result_d[qs:qs + len(query), :take] = best_d.clamp_min(0).sqrt()
        result_i[qs:qs + len(query), :take] = best_i
    return result_d.float(), result_i


def knn_self(x, k: int, device: str = "cuda"):
    """(d, idx) numpy arrays of the k nearest points of every point of x within x itself, self
    included as column 0 — the rows of `cKDTree(x).query(x, k)`. x: (N,3) array-like."""
    from physmorph.compute import is_cuda_execution, to_array
    if is_cuda_execution():
        import torch
        d, i = knn_self_torch(torch.as_tensor(to_array(x, np.float32), device=device), k)
        return to_array(d.double()), to_array(i)
    x = np.ascontiguousarray(np.asarray(x, np.float32))
    N = int(len(x))
    if not gpu_available() or N < 4096 or N <= k:
        return _cpu_knn(x, k)
    import torch
    d, i = knn_self_torch(torch.as_tensor(x, device=device), k)
    return d.double().cpu().numpy(), i.cpu().numpy()
