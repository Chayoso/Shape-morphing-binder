"""D159: the render's target surface as the target mesh grown by the run's own exterior offset.

The render terms compare the body's exterior (Zhu and Bridson's zero set at the run's pitch a: radius 3 a, offset 0.8 a)
with target pictures. On the frozen path those are the same operator applied to samples of the target (the run's and
its further draws), whose exterior is as blurred as the field at a: it carries no relief below the field's width, and a
detail the body might take there makes the term worse (D158). Here the target surface is the mesh itself, moved out to
the shell the run's exterior stands on: the iso-surface at signed distance c outside the mesh, with c the signed distance
from the mesh of the run's own target exterior (the discs the frozen path draws its pictures from), averaged over the
exterior field's own reach (LocalOffset; D159 amendment 1: one mean for the whole body asked for the fan-closed base
and the crevices the field bridges). c is measured per run, not set: the pictures keep the run's shell above the field's
reach and take the mesh's relief below it. The surface is read as the zero set of f(q) = sd(q) - c(q) (sd < 0 inside
the mesh) by the exterior's own lattice and projection, so its discs are made exactly as the body's are; its normals
are the distance's gradient. Distances and signs are Open3D's RaycastingScene's (exact closest points; the sign by
the parity of three rays), the same operator as D157's corrected referee. Nothing here has mass or acts on the simulation."""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

from .exterior import ZhuBridson


def closed_outward(V, F):
    """The mesh with its boundary loops closed by a fan from each loop's centroid and its faces turned outward (the
    volume positive): D157's referee's preparation. Returns (V, F, loops closed, watertight)."""
    import trimesh
    t = trimesh.Trimesh(np.asarray(V, np.float64), np.asarray(F, np.int64), process=True)
    loops = 0
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
            loops += 1
        t = trimesh.Trimesh(np.concatenate(Vs), np.concatenate(Fs), process=True)
    if t.volume < 0:
        t.invert()
    return np.asarray(t.vertices, np.float64), np.asarray(t.faces, np.int64), loops, bool(t.is_watertight)


class MeshDistance:
    """Signed distances (< 0 inside) and closest points of a closed triangle mesh (Open3D, CPU)."""

    def __init__(self, V, F, threads=0):
        import open3d as o3d
        self.o3d, self.threads = o3d, int(threads)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(np.asarray(V, np.float32)), o3d.core.Tensor(np.asarray(F, np.uint32)))

    def signed(self, P):
        return self.scene.compute_signed_distance(self.o3d.core.Tensor(np.ascontiguousarray(P, np.float32)),
                                                  nthreads=self.threads, nsamples=3).numpy().astype(np.float64)

    def closest(self, P):
        ans = self.scene.compute_closest_points(self.o3d.core.Tensor(np.ascontiguousarray(P, np.float32)),
                                                nthreads=self.threads)
        return ans["points"].numpy().astype(np.float64)


class LocalOffset:
    """c(q): the run's own target exterior's signed distance from the mesh, averaged over the exterior field's own reach
    (amendment 1 of D159, before any run): the discs' signed distances are pooled per lattice cell (one mean over all
    samples' discs in the cell), and each cell takes the mean of the cells within the field's radius R weighted by the
    field's own kernel (1 - r^2 / R^2)^3; a point reads the cell nearest to it. Above the field's reach the shell is
    therefore the run's own exterior (its coarse shape, its bridged crevices, its base where the mesh is open), below it
    the mesh's relief."""

    def __init__(self, P, sd, R, h, origin):
        key = np.floor((P - origin) / h).astype(np.int64)
        uk, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        cnt = np.bincount(inv)
        cen = np.stack([np.bincount(inv, P[:, i]) for i in range(3)], 1) / cnt[:, None]
        mean_sd = np.bincount(inv, sd) / cnt
        tree = cKDTree(cen)
        k = int(min(len(cen), max(16, 2.0 * np.pi * (R / h) ** 2)))   # about twice the cells a disc of radius R holds
        d, nb = tree.query(cen, k, distance_upper_bound=R)
        w = np.where(np.isfinite(d), (1.0 - np.minimum(d, R) ** 2 / R ** 2) ** 3, 0.0)
        val = mean_sd[np.minimum(nb, len(cen) - 1)]
        self.c = (w * val).sum(1) / np.maximum(w.sum(1), 1e-30)
        self.tree, self.cells, self.k = tree, len(cen), k

    def __call__(self, Q):
        return self.c[self.tree.query(Q, 1)[1]]


class MeshShell(ZhuBridson):
    """f(q) = sd(q) - c(q), the shell at the local offset c (LocalOffset) outside the mesh, with the field interface the
    exterior's lattice reads (Lattice.discs: `x` the points whose cells the lattice visits, `pitch`, __call__, project).
    x is the run's own target exterior (the reference discs): the shell lies within a fraction of a pitch of it, well
    inside the ring of two lattice cells the lattice reads around x. The gradient is the distance's,
    sign(sd) (q - closest) / |q - closest| (c varies over the field's reach, slowly against the relief); zero on the mesh."""

    def __init__(self, dist: MeshDistance, x, pitch, offset: LocalOffset):
        super().__init__(x, pitch)
        self.dist, self.local = dist, offset

    def __call__(self, q, grad=True):
        Q = q.detach().double().cpu().numpy()
        sd = self.dist.signed(Q)
        f = torch.as_tensor(sd - self.local(Q), dtype=q.dtype, device=q.device)
        s = torch.ones_like(f)
        if not grad:
            return f, None, s
        d = Q - self.dist.closest(Q)
        n = np.linalg.norm(d, axis=1, keepdims=True)
        g = np.where(n > 1e-12, np.sign(sd)[:, None] * d / np.maximum(n, 1e-30), 0.0)
        return f, torch.as_tensor(g, dtype=q.dtype, device=q.device), s

    def neighbours(self, q, skin, most=256):
        raise NotImplementedError("the mesh shell is a target surface: no particle lists")


def mesh_shell_discs(lattice, h, V, F, pitch, refs, radius):
    """The shell's discs on the exterior's lattice: (points, unit normals, c, info). refs: the run's own target exterior
    discs (a list of (M_k, 3) tensors, one per sample); radius: the exterior field's radius (wu). c is the local offset
    (LocalOffset); the returned c is its mean over the shell's discs, info has its spread."""
    Vc, Fc, loops, tight = closed_outward(V, F)
    dist = MeshDistance(Vc, Fc)
    P = torch.cat([r.detach() for r in refs]).double().cpu().numpy()
    sd = dist.signed(P)
    local = LocalOffset(P, sd, float(radius), float(h), lattice.origin.detach().double().cpu().numpy())
    shell = MeshShell(dist, torch.cat([r.detach() for r in refs]), pitch, local)
    with torch.no_grad():
        p, g, _, _ = lattice.discs(shell, h, refine=False)
    cp = local(p.detach().double().cpu().numpy())
    q = np.percentile(cp, [1, 50, 99])
    info = dict(c=float(cp.mean()), c_sd=float(cp.std()), c_p01=float(q[0]), c_p50=float(q[1]), c_p99=float(q[2]),
                refs_sd_mean=float(sd.mean()), refs_sd_median=float(np.median(sd)), cells=local.cells, cell_k=local.k,
                loops_closed=loops, watertight=tight, faces=len(Fc), discs=len(p))
    return p, torch.nn.functional.normalize(g, dim=1), info["c"], info
