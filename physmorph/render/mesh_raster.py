"""E3' (2026-10-31, pictures only): the render target's pictures drawn by another renderer, a ray-cast mesh rasteriser.

The frozen path draws the exterior's target pictures from particle samples of the render target (their Zhu-Bridson
discs, shaded_discs / soft_silhouette_multi). Here they are drawn from the render target MESH itself: no particle sample,
no field, no discs. Each view is the loss's own orthographic camera (losses/silhouette._view_basis: right, up, toward the
camera l = right x up), its box (+-extent) and resolution, with the loss's pixel convention (the node of pixel (i, j) at
lattice coordinate (i, j) = (x.right + extent, x.up + extent) / (2 extent) res, as splat_terms reads it). Every pixel is
sampled by ss x ss rays through its unit cell around the node; a ray's first hit (Open3D's RaycastingScene, exact) gives
the triangle and its barycentric coordinates, the normal is the triangle's vertex normals interpolated there (the mesh's
area-weighted vertex normals) and the shade is the loss's lighting, ambient + (1 - ambient) max(n.l, 0) (shaded_discs).
alpha = the share of a pixel's rays that hit; the shaded picture = the mean shade over the hits x alpha, as shaded_discs
returns its shade x alpha. Nothing here acts on the simulation."""
from __future__ import annotations

import numpy as np
import torch


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals (the sum of the incident faces' unnormalised normals), unit length."""
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-30)


def mesh_pictures(mesh, views, res: int, extent: float, ambient: float = 0.25, ss: int = 4, device=None):
    """(silhouettes, shading, info): lists over views of (res, res) float32 tensors on `device`, and a record.
    mesh = (V (n,3), F (m,3)) in the target's frame."""
    import open3d as o3d
    from ..losses.silhouette import _view_basis
    V = np.asarray(mesh[0], np.float64)
    F = np.asarray(mesh[1], np.int64)
    vn = vertex_normals(V, F)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    right, up = _view_basis(torch.zeros(1, 3, dtype=torch.float64), views)
    right, up = right.numpy(), up.numpy()
    l_dir = np.cross(right, up)
    G = res * ss
    off = (np.arange(G, dtype=np.float64) + 0.5) / ss - 0.5            # lattice coordinate of each ray, per axis
    pos = off / res * (2.0 * extent) - extent                          # its plane coordinate
    a, b = np.meshgrid(pos, pos, indexing="ij")                        # a along right (the pixel's i), b along up (j)
    reach = float(np.abs(V).max()) * 4.0 + 4.0 * extent
    sils, shades, neg, hits = [], [], 0, 0
    for v in range(len(views)):
        o = a[..., None] * right[v] + b[..., None] * up[v] + reach * l_dir[v]
        d = np.broadcast_to(-l_dir[v], o.shape)
        rays = np.concatenate([o, d], -1).reshape(-1, 6).astype(np.float32)
        r = scene.cast_rays(o3d.core.Tensor(rays))
        hit = np.isfinite(r["t_hit"].numpy())
        pid = r["primitive_ids"].numpy().astype(np.int64)
        uv = r["primitive_uvs"].numpy().astype(np.float64)
        n = np.zeros((hit.size, 3))
        if hit.any():
            f = F[pid[hit]]
            w = np.stack([1.0 - uv[hit, 0] - uv[hit, 1], uv[hit, 0], uv[hit, 1]], 1)
            n[hit] = (w[:, :, None] * vn[f]).sum(1)
            n[hit] /= np.maximum(np.linalg.norm(n[hit], axis=1, keepdims=True), 1e-30)
        c = n @ l_dir[v]
        neg += int((c[hit] < 0).sum())
        hits += int(hit.sum())
        shades.append(c.reshape(G, G))                                 # n.l at each ray (0 where it misses)
        sils.append(hit.reshape(G, G).astype(np.float64))
    flipped = hits > 0 and neg > 0.5 * hits                            # the faces turned inward: the normals' sign
    out_s, out_a = [], []
    for v in range(len(views)):
        c = -shades[v] if flipped else shades[v]
        sh = np.where(sils[v] > 0, ambient + (1.0 - ambient) * np.clip(c, 0.0, None), 0.0)
        al = sils[v].reshape(res, ss, res, ss).mean((1, 3))
        sp = sh.reshape(res, ss, res, ss).sum((1, 3)) / (ss * ss)       # mean shade over the hits x alpha
        out_a.append(torch.tensor(al, dtype=torch.float32, device=device))
        out_s.append(torch.tensor(sp, dtype=torch.float32, device=device))
    info = dict(rays_per_pixel=ss * ss, hits=hits, back_facing_share=(neg / hits) if hits else None, flipped=bool(flipped),
                vertices=int(len(V)), faces=int(len(F)))
    return out_a, out_s, info
