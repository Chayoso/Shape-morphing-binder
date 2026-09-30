"""Per-frame QA measures of the photoreal renderer: the bumpiness of a mesh (mean |dihedral angle|, the
H1 measure) and the number of isolated particles (8-NN distance > 3 x median)."""
import numpy as np
from scipy.spatial import cKDTree


def bumpiness(m):
    """Mean absolute dihedral angle (degrees) over the mesh's interior edges: 0 for a plane, small
    for a smooth closed surface, large for a bumpy one. The H1 measure (docs/experiments.md
    2026-09-18 item 4)."""
    if m is None or len(m.triangles) == 0:
        return float("nan")
    m.compute_triangle_normals()
    f = np.asarray(m.triangles); nrm = np.asarray(m.triangle_normals)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0)
    e = np.sort(e, axis=1)
    key = e[:, 0].astype(np.int64) * (f.max() + 1) + e[:, 1]
    tri = np.tile(np.arange(len(f)), 3)
    order = np.argsort(key, kind="stable"); key = key[order]; tri = tri[order]
    same = key[1:] == key[:-1]
    a_, b_ = tri[:-1][same], tri[1:][same]
    cosd = np.clip((nrm[a_] * nrm[b_]).sum(1), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosd)).mean()) if len(cosd) else float("nan")


def isolated_count(x_np):
    d = cKDTree(x_np).query(x_np, k=9, workers=-1)[0][:, -1]
    return int((d > 3.0 * np.median(d)).sum())
