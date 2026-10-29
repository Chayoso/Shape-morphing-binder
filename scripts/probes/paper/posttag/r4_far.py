"""r4_far.py MESH:RAW [...] -- D145 reading 4 (4), observed, not gated: what the display draws for the particles that
pieces145.py finds >= 5 pitches off the body (L97's kept frame RAW; the display mesh of r4_floor.sh). CPU; prints.

Per such particle (apart at 1.5-pitch links, >= 5 pitches from the largest set; pitch from the independent sample):
its set's size, the display field there (pitches; negative = inside a drawn surface), the nearest display vertex's
distance (pitches) and that vertex's component (vertices, enclosed volume in pitches^3, kind by the volume's sign)."""
import json
import sys

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

B = "/data/relcfd/chayo/physmorph_v2"
G = f"{B}/output/gpu"
sys.path.insert(0, f"{B}/repo_r77")
from physmorph.render.exterior import ZhuBridson  # noqa: E402

torch.set_grad_enabled(False)
for spec in sys.argv[1:]:
    mesh, raw = spec.split(":")
    I = np.asarray(np.load(f"{G}/d90/{mesh}_ind300k.npz")["tgt"], np.float64)
    a = 0.708 * float(np.median(cKDTree(I).query(I[::20], 9, workers=6)[0][:, 8]))
    z = np.load(f"{G}/d145t_b1/{mesh}300k_L97_frames12.npz", allow_pickle=True)
    x = np.asarray(z["frames"][[int(r) for r in z["raws"]].index(int(raw))], np.float64)
    pairs = cKDTree(x).query_pairs(1.5 * a, output_type="ndarray")
    _, lab = connected_components(coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])),
                                             shape=(len(x), len(x))), directed=False)
    big = int(np.argmax(np.bincount(lab)))
    ap = np.nonzero(lab != big)[0]
    d = cKDTree(x[lab == big]).query(x[ap], 1, workers=6)[0] / a
    far = ap[d >= 5]
    m = np.load(f"{G}/d145_ext/r4/display_floor/{mesh}_L97_{raw}_mesh.npz")
    V, F = np.asarray(m["V"], np.float64), np.asarray(m["F"], np.int64)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    nv, vlab = connected_components(coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(V), len(V))),
                                    directed=False)
    main = int(np.argmax(np.bincount(vlab)))
    tri = V[F]
    sv = np.bincount(vlab[F[:, 0]], weights=np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0,
                     minlength=nv)
    s = 1.0 if sv[main] >= 0 else -1.0
    nvert = np.bincount(vlab, minlength=nv)
    f = ZhuBridson(torch.as_tensor(x, dtype=torch.float32), a)(torch.as_tensor(x[far], dtype=torch.float32))[0].numpy()
    dv, vi = cKDTree(V).query(x[far], 1, workers=6)
    out = []
    for p, fp, dd, v in zip(far, f, dv, vi):
        c = vlab[v]
        out.append(dict(set=int(np.bincount(lab)[lab[p]]), field_a=round(float(fp) / a, 3), vertex_a=round(float(dd) / a, 2),
                        comp="main" if c == main else ("piece" if s * sv[c] > 0 else "cavity"), comp_verts=int(nvert[c]),
                        comp_vol_a3=round(float(abs(sv[c])) / a ** 3, 4)))
    print(json.dumps(dict(mesh=mesh, raw=int(raw), far=len(far), particles=out)), flush=True)
