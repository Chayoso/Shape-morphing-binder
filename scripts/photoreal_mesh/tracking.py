"""Surface tracking helpers (docs/method.md 10.15): mesh vertices advected with the material (k-NN
binding to the previous frame's particles) and the closest points of a mesh (Open3D ray casting), used
by --track, --surfel_memory and the per-frame re-fit jitter."""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


# ---- surface tracking (docs/method.md 10.15) ---------------------------------------------
def advect_vertices(V, x_prev, x_cur, k, h):
    """Move mesh vertices with the material: each vertex is bound to its k nearest particles of the
    PREVIOUS frame (Gaussian weights of width h = one spacing) and moves by their weighted mean
    displacement to the current frame. Particles that are neighbours at one frame are neighbours at
    the next, so the binding is renewed every frame."""
    kd = cKDTree(x_prev)
    d, j = kd.query(V, k=k, workers=-1)
    w = np.exp(-(d / h) ** 2)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
    return V + (w[:, :, None] * (x_cur[j] - x_prev[j])).sum(1)


def closest_on(mesh, P):
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    r = sc.compute_closest_points(o3d.core.Tensor(np.asarray(P, np.float32)))
    return r["points"].numpy().astype(np.float64)
