"""truth_view.py OUT_PNG RELIEF_OBJ SMOOTH_OBJ: the relief mesh as a true (right-handed) observer sees each face: from +z
(x right, y up) and from -z (-x right, y up); colour = the vertex offset from the smooth canvas (nearest smooth vertex)."""
import sys
import numpy as np, trimesh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
r = trimesh.load(sys.argv[2], process=False); s = trimesh.load(sys.argv[3], process=False)
V = np.asarray(r.vertices); Vs = np.asarray(s.vertices)
rng = np.random.default_rng(0); idx = rng.choice(len(V), 600000, replace=False); V = V[idx]
d, i = cKDTree(Vs).query(V, 1)
c = V.mean(0)
h = np.sign(np.abs(V[:, 2] - c[2]) - np.abs(Vs[i, 2] - c[2])) * d
fig, ax = plt.subplots(2, 1, figsize=(14, 6))
zt = V[:, 2].max(); zb = V[:, 2].min()
m = V[:, 2] > zt - 2.5
ax[0].scatter(V[m, 0], V[m, 1], c=h[m], s=0.3, cmap="viridis"); ax[0].set_title("true view from +z (camera right = +x)"); ax[0].set_aspect("equal")
m = V[:, 2] < zb + 2.5
ax[1].scatter(-V[m, 0], V[m, 1], c=h[m], s=0.3, cmap="viridis"); ax[1].set_title("true view from -z (camera right = -x)"); ax[1].set_aspect("equal")
plt.tight_layout(); plt.savefig(sys.argv[1], dpi=110)
