"""r4_blob.py OUT_JSONL MESH:RAW [...] -- D145 reading 4 (4), observed, not gated: where visible_apart.py's blobs and bumps
of L97's mid-progress frames sit on the display mesh of item 2 (CPU; run from repo_t with its env).

visible_apart's classify() (unchanged, imported) gives each apart particle of the frame its kind (hidden / bump / blob:
the full display field turns positive on the segment to the nearest particle outside its set). Each such particle's
nearest vertex of r4_mesh.py's display mesh of the same frame (r4/display/MESH_L97_RAW_mesh.npz) names the component
it is under: the main outer surface, an internal cavity (a component of negative enclosed volume) or a separate solid
piece; and its distance to that vertex (pitches)."""
import json
import os
import sys

B = "/data/relcfd/chayo/physmorph_v2"
sys.path.insert(0, f"{B}/repo_t/scripts/probes/paper")
import visible_apart as va  # noqa: E402
import numpy as np  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

G = f"{B}/output/gpu"
X = f"{G}/d145_ext/r4"
for spec in sys.argv[2:]:
    mesh, raw = spec.split(":")
    z = np.load(f"{G}/d145t_b1/{mesh}300k_L97_frames12.npz", allow_pickle=True)
    raws = [int(r) for r in z["raws"]]
    x = np.asarray(z["frames"][raws.index(int(raw))], np.float64)
    tgt = np.asarray(z["tgt"], np.float64)
    a = 0.708 * float(np.median(cKDTree(tgt).query(tgt, 9, workers=6)[0][:, 8]))   # visible_apart's pitch
    ap, kind, height, _ = va.classify(x, a)
    m = np.load(f"{X}/display/{mesh}_L97_{raw}_mesh.npz")
    V, F = np.asarray(m["V"], np.float64), np.asarray(m["F"], np.int64)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    nv, vlab = connected_components(coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(V), len(V))),
                                    directed=False)
    main = int(np.argmax(np.bincount(vlab)))
    tri = V[F]
    sv = np.bincount(vlab[F[:, 0]], weights=np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0,
                     minlength=nv)
    s = 1.0 if sv[main] >= 0 else -1.0
    ck = np.array(["main" if c == main else ("cavity" if s * sv[c] < 0 else "piece") for c in range(nv)])
    d, vi = cKDTree(V).query(x[ap], 1, workers=6)
    under = ck[vlab[vi]]
    row = dict(mesh=mesh, raw=int(raw), pitch=a, apart=int(len(ap)))
    for kn, kv in (("hidden", 0), ("bump", 1), ("blob", 2)):
        sel = kind == kv
        row[kn] = {c: int((sel & (under == c)).sum()) for c in ("main", "cavity", "piece")}
        row[kn + "_dist_median_a"] = round(float(np.median(d[sel] / a)), 2) if sel.any() else None
    with open(sys.argv[1], "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
