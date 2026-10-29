"""r4_undrawn.py OUT_JSONL MESH:LABEL:RAW=ARCHIVE [...] -- D145 reading 4 (4), observed, not gated: are the apart
particles drawn at all? (CPU; repo_r77's exterior field, as r4_mesh.py.)

Per state: the harness's apart particles (outside the largest set connected at 1.5 pitches; pitch from the independent
sample, as r4_mesh.py) and the display field (Zhu and Bridson, radius 3, offset 0.8 pitches) of all particles at each
of them: negative = inside a drawn surface (the body's, or a piece's), positive = no surface is drawn around it (a set
too small or too thin for the field to close). For the undrawn ones: their sets' sizes and their distance to the
largest set (pitches)."""
import json
import sys

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

B = "/data/relcfd/chayo/physmorph_v2"
sys.path.insert(0, f"{B}/repo_r77")
from physmorph.render.exterior import ZhuBridson  # noqa: E402

torch.set_grad_enabled(False)
torch.set_num_threads(6)
pitch = {}
for spec in sys.argv[2:]:
    key, path = spec.split("=", 1)
    mesh, label, raw = key.split(":")
    if mesh not in pitch:
        I = np.asarray(np.load(f"{B}/output/gpu/d90/{mesh}_ind300k.npz")["tgt"], np.float64)
        pitch[mesh] = 0.708 * float(np.median(cKDTree(I).query(I[::20], 9, workers=6)[0][:, 8]))
    a = pitch[mesh]
    z = np.load(path, allow_pickle=True)
    raws = [int(r) for r in z["raws"]]
    k = len(raws) - 1 if raw == "end" else raws.index(int(raw))
    x = np.asarray(z["frames"][k], np.float64)
    tree = cKDTree(x)
    pairs = tree.query_pairs(1.5 * a, output_type="ndarray")
    _, lab = connected_components(coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])),
                                             shape=(len(x), len(x))), directed=False)
    big = int(np.argmax(np.bincount(lab)))
    ap = np.nonzero(lab != big)[0]
    row = dict(mesh=mesh, label=label, raw=raws[k], pitch=a, apart=int(len(ap)))
    if len(ap):
        f = ZhuBridson(torch.as_tensor(x, dtype=torch.float32), a)(torch.as_tensor(x[ap], dtype=torch.float32))[0]
        f = f.numpy()
        und = ap[~(f < 0)]
        sizes = np.bincount(lab)[lab[und]]
        tb = cKDTree(x[lab == big])
        dist = tb.query(x[und], 1, workers=6)[0] / a if len(und) else np.zeros(0)
        dall = tb.query(x[ap], 1, workers=6)[0] / a          # every apart particle's distance to the largest set
        row.update(apart_ge5=int((dall >= 5).sum()), apart_dist_max_a=round(float(dall.max()), 2),
                   drawn=int((f < 0).sum()), undrawn=int(len(und)),
                   undrawn_sets=int(len(np.unique(lab[und]))),
                   undrawn_set_size_max=int(sizes.max()) if len(und) else 0,
                   undrawn_ge5=int((dist >= 5).sum()), undrawn_dist_max_a=round(float(dist.max()), 2) if len(und) else 0)
    with open(sys.argv[1], "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
