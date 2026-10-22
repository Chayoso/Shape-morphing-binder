"""D162b (cfg.render_weight_ref "shape"): the render term read on the body's shape, not on its arrangement.

D162 read 1 (docs): referenced to the render term of the body's own particles, the floor-referenced weight read the
arrangement. On the relief the full arm arranges its exterior so that the same shape scores about 27 % below a sampled
arrangement (a / b 0.40 against the floor sample's 0.54), which crosses the floor while 8-13 % of the shape's excess is
left, and the per-window weight turned into an on/off switch. Here the weight's excess is read on b: the render term of
uniform resamples of the body's solid, so the arrangement is dropped and the shape kept (D156h's reading, the D162 X step).

- The solid: the points deeper than c inside the zero set of the Zhu-Bridson field of the particles (D157's field mesh
  at 0.708 target spacings, banded marching cubes); c is measured once a run, the floor sample's zero set against the
  render target's own mesh (the mean signed distance of its vertices: how far out the field puts a body that is
  exactly the target sample).
- The resample: a jittered lattice at the target sample's number density (spacing (target fill volume / N)^(1/3), the
  stratified sampler's density), fixed in the world for the whole run (sites over the MPM domain, the jitter drawn once
  per seed), keeping the points inside the solid. Eight seeds (95 + 100 k, D156h's and the X step's), as many as the
  pictures' own samples; b is the mean of their render terms (target.render_floor, read as the body is read), with
  paint (D160) from each point's nearest particle.
- b_floor: the floor sample (the pictures' own first sample, D162 amendment 2) read the same way, once a run.

CPU for the field mesh and the inside test (torch, open3d's raycasting scene), the device for the render terms."""
from __future__ import annotations

import math
import time

import numpy as np
import torch

SEEDS = tuple(95 + 100 * k for k in range(8))


class _Scene:
    """A triangle mesh in open3d's raycasting scene: signed distances (negative inside)."""

    def __init__(self, V, F, threads=6):
        import open3d as o3d
        self.o3d, self.nt = o3d, threads
        self.V = np.asarray(V, np.float64)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(self.V.astype(np.float32)),
                                 o3d.core.Tensor(np.asarray(F, np.int64).astype(np.uint32)))

    def signed(self, P):
        return self.scene.compute_signed_distance(self.o3d.core.Tensor(np.ascontiguousarray(P, np.float32)),
                                                  nthreads=self.nt, nsamples=3).numpy().astype(np.float64)


def _banded_mc(read, lo, nc, hc, coarse):
    """Marching cubes of a field at level 0 (D157's banded_mc): the coarse grid's values, the half-spacing nodes of the
    cells the level crosses and their ring read exactly, the rest trilinear from the coarse values."""
    import torch.nn.functional as nnf
    from scipy.ndimage import binary_dilation
    from skimage.measure import marching_cubes
    s = coarse > 0.0
    crossed = np.zeros(nc - 1, bool)
    base = tuple(slice(0, k - 1) for k in nc)
    for o in np.ndindex(2, 2, 2):
        crossed |= s[tuple(slice(oi, oi + k - 1) for oi, k in zip(o, nc))] != s[base]
    band = binary_dilation(crossed, structure=np.ones((3, 3, 3), bool))
    nf = 2 * (nc - 1) + 1
    ff = nnf.interpolate(torch.as_tensor(coarse, dtype=torch.float32)[None, None], size=tuple(int(v) for v in nf),
                         mode="trilinear", align_corners=True)[0, 0].numpy().astype(np.float64)
    cells = np.argwhere(band)
    off = np.stack(np.meshgrid(*[np.arange(3)] * 3, indexing="ij"), -1).reshape(1, -1, 3)
    fine = np.unique((2 * cells[:, None, :] + off).reshape(-1, 3), axis=0)
    hm = 0.5 * hc
    ff[fine[:, 0], fine[:, 1], fine[:, 2]] = read(lo + hm * fine)
    verts, faces, _, _ = marching_cubes(ff, level=0.0, spacing=(hm, hm, hm))
    return verts + lo, faces


def zero_set(x, a, threads=6):
    """The zero set of the particles' Zhu-Bridson field at pitch a as a triangle mesh in a raycasting scene (D157's
    field_mesh "zb")."""
    from scipy.spatial import cKDTree
    from ..render.exterior import ZhuBridson
    x = np.asarray(x, np.float64)
    field = ZhuBridson(torch.as_tensor(x, dtype=torch.float32), a)
    R, hc = field.radius, a
    lo = x.min(0) - (R + 2 * hc)
    nc = np.ceil((x.max(0) + (R + 2 * hc) - lo) / hc).astype(int) + 1
    q = lo + hc * np.stack(np.meshgrid(*[np.arange(k) for k in nc], indexing="ij"), -1).reshape(-1, 3)
    near = cKDTree(x).query(q, 1, distance_upper_bound=R, workers=threads)[0] < R
    fc = np.full(len(q), R, np.float64)

    def read(p):
        with torch.no_grad():
            f, _, _ = field(torch.as_tensor(p, dtype=torch.float32), grad=False)
        return np.nan_to_num(f.numpy().astype(np.float64), posinf=R)

    fc[near] = read(q[near])
    del q
    V, F = _banded_mc(read, lo, nc, hc, fc.reshape(*nc))
    return _Scene(V, F, threads)


def closed_mesh(V, F):
    """The mesh with its boundary loops closed by fans from their centroids (D157's close_holes), outward."""
    import trimesh
    t = trimesh.Trimesh(np.asarray(V, np.float64), np.asarray(F, np.int64), process=True)
    if not t.is_watertight:
        E = np.concatenate([t.faces[:, [0, 1]], t.faces[:, [1, 2]], t.faces[:, [2, 0]]])
        _, inv, cnt = np.unique(np.sort(E, 1), axis=0, return_inverse=True, return_counts=True)
        bd = E[cnt[inv.ravel()] == 1]
        Vs, Fs, k = [np.asarray(t.vertices)], [np.asarray(t.faces)], len(t.vertices)
        for comp in trimesh.graph.connected_components(bd, min_len=1):
            e = bd[np.isin(bd[:, 0], comp)]
            Vs.append(np.asarray(t.vertices)[np.unique(e)].mean(0)[None])
            Fs.append(np.stack([e[:, 1], e[:, 0], np.full(len(e), k)], 1))
            k += 1
        t = trimesh.Trimesh(np.concatenate(Vs), np.concatenate(Fs), process=True)
    if t.volume < 0:
        t.invert()
    return np.asarray(t.vertices), np.asarray(t.faces)


class ShapeRead:
    """b(x): the mean render term of eight uniform resamples of the solid of the particles x (module docstring)."""

    def __init__(self, tgt, cfg, floor_x, mesh, target_fill: float, n: int, domain, floor_albedo=None, log=print):
        from .. import gpu
        self.tgt, self.cfg = tgt, cfg
        fx = np.asarray(floor_x, np.float64)
        self.pitch = 0.708 * float(gpu.median_kth_spacing(gpu.tensor(np.asarray(floor_x, np.float32)), 8,
                                                          subsample=20000))
        t0 = time.time()
        true = _Scene(*closed_mesh(*mesh))
        self.c = float(true.signed(zero_set(fx, self.pitch).V).mean())
        self.s = float(target_fill / n) ** (1.0 / 3.0)
        lo, hi = (np.asarray(v.detach().cpu().numpy() if torch.is_tensor(v) else v, np.float64) for v in domain)
        self.lo = lo
        self.dims = np.maximum(np.ceil((hi - lo) / self.s).astype(np.int64), 1)
        m = int(np.prod(self.dims))
        self.jitter = [(np.random.default_rng(sd).uniform(-0.5, 0.5, (m, 3)) * self.s).astype(np.float32) for sd in SEEDS]
        self.b_floor, self.b_floor_se, self.n_floor = self.read(fx, floor_albedo)
        log(f"[v2] render weight referenced to the shape (D162b): field pitch {self.pitch:.4f} wu, the floor sample's "
            f"offset c = {self.c / self.pitch:.3f} pitches, resample spacing {self.s:.4f} wu over a {self.dims.tolist()} "
            f"lattice, b_floor = {self.b_floor:.4e} ± {self.b_floor_se:.1e} ({self.n_floor} points a resample; "
            f"{time.time() - t0:.1f} s)")

    def read(self, x, albedo=None):
        """(b, its standard error over the eight resamples, the mean resample's point count) of the particles x."""
        from scipy.spatial import cKDTree
        from .target import render_floor
        x = np.asarray(x.detach().cpu().numpy() if torch.is_tensor(x) else x, np.float64)
        zs = zero_set(x, self.pitch)
        i0 = np.clip(np.floor((zs.V.min(0) - self.lo) / self.s).astype(np.int64), 0, self.dims - 1)
        i1 = np.clip(np.ceil((zs.V.max(0) - self.lo) / self.s).astype(np.int64) + 1, 1, self.dims)
        ijk = np.stack(np.meshgrid(*[np.arange(i0[d], i1[d]) for d in range(3)], indexing="ij"), -1).reshape(-1, 3)
        flat = np.ravel_multi_index(ijk.T, tuple(int(v) for v in self.dims))
        centres = (self.lo + self.s * (ijk + 0.5)).astype(np.float32)
        tree = cKDTree(x) if albedo is not None else None
        alb = None if albedo is None else np.asarray(albedo.detach().cpu().numpy() if torch.is_tensor(albedo) else albedo,
                                                     np.float32)
        vals, counts = [], []
        for jit in self.jitter:
            p = centres + jit[flat]
            xs = p[zs.signed(p) < -self.c]
            counts.append(len(xs))
            vals.append(render_floor(self.tgt, self.cfg, xs, None if alb is None else alb[tree.query(xs)[1]]))
        v = np.asarray(vals)
        return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v))), int(np.mean(counts))
