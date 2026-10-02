"""make_ridge_slab.py OUT_OBJ WAVELENGTH_WU — a closed box (4.454 x 2.4 x 4.454 wu, the gallery body's volume) whose top
face carries ridges y = A sin(2 pi x / wavelength), A = wavelength / 8, running along z (D15's target). Every face is
a regular grid of near-square cells of about 0.019 wu (the sampler's voxeliser subdivides long triangles, so the
mesh has none)."""
import sys
import numpy as np
import trimesh

out, lam = sys.argv[1], float(sys.argv[2])
L, D, A = 4.454, 2.4, float(sys.argv[2]) / 8
nx = nz = 240
ny = 128
xs, zs, ts = np.linspace(-L / 2, L / 2, nx + 1), np.linspace(-L / 2, L / 2, nz + 1), np.linspace(0, 1, ny + 1)
ytop = lambda x: D / 2 + A * np.sin(2 * np.pi * x / lam)


def grid(P):
    """Vertices (nu+1, nv+1, 3) -> (vertices, two triangles per cell)."""
    nu, nv = P.shape[0] - 1, P.shape[1] - 1
    I, K = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    a, b, c, d = (I * (nv + 1) + K).ravel(), ((I + 1) * (nv + 1) + K).ravel(), ((I + 1) * (nv + 1) + K + 1).ravel(), (I * (nv + 1) + K + 1).ravel()
    return P.reshape(-1, 3), np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])


X, Z = np.meshgrid(xs, zs, indexing="ij")
parts = [grid(np.stack([X, ytop(X), Z], -1)), grid(np.stack([X, np.full_like(X, -D / 2), Z], -1))]
Xw, Tw = np.meshgrid(xs, ts, indexing="ij")                    # the two walls at the ends of z follow the ridges
for zc in (-L / 2, L / 2):
    parts.append(grid(np.stack([Xw, -D / 2 + Tw * (ytop(Xw) + D / 2), np.full_like(Xw, zc)], -1)))
Zw, Tz = np.meshgrid(zs, ts, indexing="ij")                    # the two walls at the ends of x
for xc in (-L / 2, L / 2):
    parts.append(grid(np.stack([np.full_like(Zw, xc), -D / 2 + Tz * (ytop(xc) + D / 2), Zw], -1)))
V, F, off = [], [], 0
for v, f in parts:
    V.append(v); F.append(f + off); off += len(v)
mesh = trimesh.Trimesh(np.concatenate(V), np.concatenate(F), process=True)
mesh.merge_vertices()
mesh.fix_normals()
if not mesh.is_watertight:
    raise SystemExit("the ridge slab is not closed")
mesh.export(out)
edges = mesh.edges_unique_length
print(f"{out}: wavelength {lam:.4f} wu, amplitude {A:.4f} wu, {L / lam:.1f} ridges, volume {mesh.volume:.2f} wu^3, {len(mesh.faces)} faces, longest edge {edges.max():.3f} wu")
