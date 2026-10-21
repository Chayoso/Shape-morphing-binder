"""D159: the render's target surface as the target mesh grown by the run's own exterior offset.

The render terms compare the body's exterior (Zhu and Bridson's zero set at the run's pitch a: radius 3 a, offset 0.8 a)
with target pictures. On the frozen path those are the same operator applied to samples of the target (the run's and
its further draws), whose exterior is as blurred as the field at a: it carries no relief below the field's width, and a
detail the body might take there makes the term worse (D158). Here the target surface is the mesh itself, moved out to
the shell the run's exterior stands on: the iso-surface at signed distance c outside the mesh, with c the mean signed
distance from the mesh of the run's own target exterior (the discs the frozen path draws its pictures from). c is
measured per run, not set: the pictures keep the run's shell and take the mesh's relief. The surface is read as the
zero set of f(q) = sd(q) - c (sd < 0 inside the mesh) by the exterior's own lattice and projection, so its discs are
made exactly as the body's are; its normals are the distance's gradient. Distances and signs are Open3D's
RaycastingScene's (exact closest points; the sign by the parity of three rays), the same operator as D157's corrected
referee. Nothing here has mass or acts on the simulation."""
from __future__ import annotations

import numpy as np
import torch

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


class MeshShell(ZhuBridson):
    """f(q) = sd(q) - c, the shell at signed distance c outside the mesh, with the field interface the exterior's lattice
    reads (Lattice.discs: `x` the points whose cells the lattice visits, `pitch`, __call__, project). x is the run's own
    target exterior (the reference discs): the shell lies where that exterior stands on average, within a fraction of a
    pitch of it, well inside the ring of two lattice cells the lattice reads around x. The gradient is
    sign(sd) (q - closest) / |q - closest|; on the mesh itself (never on the shell, c > 0) it is zero."""

    def __init__(self, dist: MeshDistance, x, pitch, c):
        super().__init__(x, pitch)
        self.dist, self.c = dist, float(c)

    def __call__(self, q, grad=True):
        Q = q.detach().double().cpu().numpy()
        sd = self.dist.signed(Q)
        f = torch.as_tensor(sd - self.c, dtype=q.dtype, device=q.device)
        s = torch.ones_like(f)
        if not grad:
            return f, None, s
        d = Q - self.dist.closest(Q)
        n = np.linalg.norm(d, axis=1, keepdims=True)
        g = np.where(n > 1e-12, np.sign(sd)[:, None] * d / np.maximum(n, 1e-30), 0.0)
        return f, torch.as_tensor(g, dtype=q.dtype, device=q.device), s

    def neighbours(self, q, skin, most=256):
        raise NotImplementedError("the mesh shell is a target surface: no particle lists")


def mesh_shell_discs(lattice, h, V, F, pitch, refs):
    """The shell's discs on the exterior's lattice: (points, unit normals, c, info). refs: the run's own target exterior
    discs (a list of (M_k, 3) tensors, one per sample); c is their mean signed distance from the mesh."""
    Vc, Fc, loops, tight = closed_outward(V, F)
    dist = MeshDistance(Vc, Fc)
    sd = np.concatenate([dist.signed(r.detach().double().cpu().numpy()) for r in refs])
    c = float(sd.mean())
    if not c > 0:
        raise ValueError(f"the run's exterior stands {c:.4g} wu from the mesh: a shell needs it outside (c > 0)")
    shell = MeshShell(dist, torch.cat([r.detach() for r in refs]), pitch, c)
    with torch.no_grad():
        p, g, _, _ = lattice.discs(shell, h, refine=False)
    info = dict(c=c, c_sd=float(sd.std()), loops_closed=loops, watertight=tight, faces=len(Fc), discs=len(p))
    return p, torch.nn.functional.normalize(g, dim=1), c, info
