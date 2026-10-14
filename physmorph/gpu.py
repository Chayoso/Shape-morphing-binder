"""Device helpers of the GPU-only pipeline: every run-time array lives on the CUDA device.

The run stage has no CPU fallback. Neighbour queries use CuPy's device KD-tree, grid
morphology uses cupyx.scipy.ndimage, and both exchange memory with torch through DLPack (no
host copies). The prepare stage (mesh loading and volume sampling, physmorph.prepare) is the
only CPU work and is cached on disk. CuPy comes from the isolated dependency folder that
scripts/ops/hyde06_env.sh puts on PYTHONPATH.
"""
from __future__ import annotations

import numpy as np
import torch

DEVICE = "cuda"


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the pipeline runs on a CUDA GPU only (no CPU fallback)")


def tensor(a, dtype=torch.float32) -> torch.Tensor:
    """A CUDA tensor of `a` (numpy array, list, scalar or tensor), contiguous."""
    if torch.is_tensor(a):
        return a.to(device=DEVICE, dtype=dtype).contiguous()
    a = np.ascontiguousarray(a)
    if not a.flags.writeable:                      # a read-only archive member: copy, torch needs writable
        a = a.copy()
    return torch.as_tensor(a, dtype=dtype, device=DEVICE)


def host(t: torch.Tensor) -> np.ndarray:
    """A host copy for the archive (outputs only; nothing computes on it afterwards)."""
    return t.detach().cpu().numpy()


def _cupy():
    import cupy
    from cupy_backends.cuda.libs import nvrtc
    if nvrtc.getVersion()[0] < 12:
        raise RuntimeError("CuPy is bound to a CUDA 11 NVRTC (loaded by torch): import physmorph "
                           "before torch so CuPy's CUDA 12 compiler loads first")
    return cupy


def to_cupy(t: torch.Tensor):
    """A CuPy view of a CUDA tensor (a CPU tensor, from tests, is copied to the device first)."""
    return _cupy().from_dlpack(t.detach().to(DEVICE).contiguous())


def to_torch(a) -> torch.Tensor:
    return torch.from_dlpack(a)


class KNN:
    """Exact nearest neighbours of query points among `points` (CuPy device KD-tree).
    Build once for a fixed point set (the target) and query many times."""

    def __init__(self, points: torch.Tensor):
        from cupyx.scipy.spatial import KDTree
        self.n = int(points.shape[0])
        # float64 like scipy's cKDTree, so distances and medians match it to rounding
        self.tree = KDTree(to_cupy(points.double()))

    def query(self, queries: torch.Tensor, k: int = 1):
        """(distances float64 (M,k), indices int64 (M,k)); a self query returns self first
        (up to exact duplicates), the rows of scipy's cKDTree.query."""
        k = int(min(k, self.n))
        d, i = self.tree.query(to_cupy(queries.double()), k=k)
        d = to_torch(d).double().reshape(-1, k).to(queries.device)
        i = to_torch(i).long().reshape(-1, k).to(queries.device)
        return d, i


def knn(points: torch.Tensor, k: int, queries: torch.Tensor | None = None):
    """One-off exact kNN: (d, idx) of each query (default: every point) among `points`."""
    return KNN(points).query(points if queries is None else queries, k)


def symmetric_rows(nbr: torch.Tensor) -> torch.Tensor:
    """The symmetric closure of neighbour rows (D149): q is in row p iff q is in p's row or p is in q's.

    (N, K) -> (N, K') int64 on nbr's device. Each row lists its neighbours in ascending order, then p's own index as
    padding up to the widest row (K' = the largest degree). The minimum spacing's kernels skip a pair at zero distance,
    so the padding acts on nothing. A self entry of the input is dropped. Each unordered pair sits in both of its rows,
    so a pairwise update that is opposite in its two particles sums to zero over the body."""
    N, K = int(nbr.shape[0]), int(nbr.shape[1])
    dev = nbr.device
    p = torch.arange(N, device=dev, dtype=torch.int64).repeat_interleave(K)
    q = nbr.reshape(-1).to(torch.int64)
    keep = q != p
    p, q = p[keep], q[keep]
    key = torch.unique(torch.cat((p * N + q, q * N + p)))      # sorted: by row, then by column; duplicates merged
    rows, cols = key // N, key % N
    deg = torch.bincount(rows, minlength=N)
    width = max(int(deg.max()), 1) if len(key) else 1
    slot = torch.arange(len(key), device=dev) - (torch.cumsum(deg, 0) - deg)[rows]
    out = torch.arange(N, device=dev, dtype=torch.int64)[:, None].repeat(1, width)
    out[rows, slot] = cols
    return out


def median_kth_spacing(points: torch.Tensor, kth: int, subsample: int | None = None, seed: int = 0) -> float:
    """Median distance to the kth neighbour (kth=1: nearest other point). With `subsample`
    the median is taken on a seeded subset and rescaled by (n_sub/N)^(1/3) to the full
    density, the estimator the layer and shading spacings use."""
    n = int(points.shape[0])
    sub = points
    scale = 1.0
    if subsample is not None and n > subsample:
        idx = np.random.default_rng(seed).choice(n, subsample, replace=False)
        idx = torch.as_tensor(idx, device=points.device)
        sub = points[idx]
        scale = (subsample / n) ** (1.0 / 3.0)
    d, _ = knn(sub, kth + 1)
    return median(d[:, kth]) * scale


def median(t: torch.Tensor) -> float:
    """numpy's median (the mean of the two middle values for an even count); torch.median
    returns the lower one."""
    return float(torch.quantile(t.double().flatten(), 0.5))


def edt(outside: torch.Tensor) -> torch.Tensor:
    """Euclidean distance (in cells, float64) of every True cell to the nearest False cell."""
    from cupyx.scipy import ndimage
    return to_torch(ndimage.distance_transform_edt(to_cupy(outside))).double().to(outside.device)


def label26(occupied: torch.Tensor):
    """Connected components of a 3-D boolean grid under 26-connectivity:
    (labels int32 tensor, count)."""
    from cupyx.scipy import ndimage
    cp = _cupy()
    lab, n = ndimage.label(to_cupy(occupied), structure=cp.ones((3, 3, 3), dtype=cp.int32))
    return to_torch(lab).to(occupied.device), int(n)


def fill_holes(mask: torch.Tensor) -> torch.Tensor:
    from cupyx.scipy import ndimage
    return to_torch(ndimage.binary_fill_holes(to_cupy(mask))).to(mask.device)


def dilate26(occupied: torch.Tensor) -> torch.Tensor:
    """Binary dilation of a 3-D boolean grid by the full 3x3x3 structure."""
    g = occupied.float()[None, None]
    return torch.nn.functional.max_pool3d(g, 3, stride=1, padding=1)[0, 0] > 0
