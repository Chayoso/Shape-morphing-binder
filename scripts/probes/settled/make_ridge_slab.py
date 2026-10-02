"""make_ridge_slab.py OUT_OBJ WAVELENGTH_WU — a closed box (4.454 x 2.4 x 4.454 wu, the gallery body's volume) whose top
face carries ridges y = A sin(2 pi x / wavelength), A = wavelength / 8, running along z (D15's target)."""
import sys
import numpy as np
import trimesh

out, lam = sys.argv[1], float(sys.argv[2])
L, D, A = 4.454, 2.4, float(sys.argv[2]) / 8
nx, nz = 768, 48
xs, zs = np.linspace(-L / 2, L / 2, nx + 1), np.linspace(-L / 2, L / 2, nz + 1)
X, Z = np.meshgrid(xs, zs, indexing="ij")
top = np.stack([X, D / 2 + A * np.sin(2 * np.pi * X / lam), Z], -1).reshape(-1, 3)
bot = np.stack([X, np.full_like(X, -D / 2), Z], -1).reshape(-1, 3)
n = len(top)
I, K = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
a, b, c, d = (I * (nz + 1) + K).ravel(), ((I + 1) * (nz + 1) + K).ravel(), ((I + 1) * (nz + 1) + K + 1).ravel(), (I * (nz + 1) + K + 1).ravel()
faces = [np.stack([a, c, b], 1), np.stack([a, d, c], 1), np.stack([a + n, b + n, c + n], 1), np.stack([a + n, c + n, d + n], 1)]
for k in (0, nz):                                              # the two walls at the ends of z
    i = np.arange(nx)
    p, q = i * (nz + 1) + k, (i + 1) * (nz + 1) + k
    faces += [np.stack([p, q, q + n], 1), np.stack([p, q + n, p + n], 1)]
for i in (0, nx):                                              # the two walls at the ends of x
    k = np.arange(nz)
    p, q = i * (nz + 1) + k, i * (nz + 1) + k + 1
    faces += [np.stack([p, q, q + n], 1), np.stack([p, q + n, p + n], 1)]
mesh = trimesh.Trimesh(np.concatenate([top, bot]), np.concatenate(faces), process=True)
mesh.fix_normals()
if not mesh.is_watertight:
    raise SystemExit("the ridge slab is not closed")
mesh.export(out)
print(f"{out}: wavelength {lam:.4f} wu, amplitude {A:.4f} wu, {L / lam:.1f} ridges, volume {mesh.volume:.2f} wu^3, {len(mesh.faces)} faces")
