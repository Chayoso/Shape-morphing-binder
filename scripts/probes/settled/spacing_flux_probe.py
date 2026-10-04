"""spacing_flux_probe.py FRAMES_NPZ R_OVER_A — D70: where the minimum spacing moves material, on every kept frame, no
simulation. The rule of kernels.k_update is evaluated on the frame as it is (the 16 nearest, half of each overlap,
0.87 of it over one window) and its move is split along the outward normal of the outermost layer (the nearest layer
particle's) for four groups by depth under that layer: the layer itself, to 1.5 pitches, 1.5 to 4, deeper. A rule that
only redistributes material inside the body has no mean outward move in any group; a mean outward move of the layer
is material leaving through the free surface. One JSON row per frame, moves in display pitches per window."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
from physmorph.pipeline.window.layer import layer_by_asymmetry, layer_spacing  # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402

dev = torch.device("cuda")
z = np.load(sys.argv[1], allow_pickle=True)
raws = [int(v) for v in z["raws"]]
tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
a = .708 * float(knn_self_torch(tgt, 9)[0][:, 8].median())
r = float(sys.argv[2]) * a
window = .5 * (1. - (1. - 1. / 20.) ** 40)


def row(x):
    layer, nrm = layer_by_asymmetry(x, layer_spacing(x))[:2]
    on = torch.zeros(len(x), dtype=torch.bool, device=dev)
    on[layer] = True
    d, nb = gpu.KNN(x).query(x, 17)
    d, nb = d.float()[:, 1:], nb[:, 1:]
    over = (r - d).clamp_min(0.)
    rel = x[:, None, :] - x[nb]
    move = window * (over[..., None] * rel / d.clamp_min(1e-9)[..., None]).sum(1) / a
    near = gpu.KNN(x[on]).query(x, 1)
    depth = near[0].float().reshape(-1) / a
    n = nrm[on][near[1].reshape(-1)]
    outward = float((n[on] * (x[on] - x[nb[on]].mean(1))).sum(1).mean()) > 0.      # the layer's normals point out
    out = (move * n).sum(1) * (1. if outward else -1.)
    groups = {"layer": on, "to_1.5": ~on & (depth <= 1.5), "1.5_to_4": ~on & (depth > 1.5) & (depth <= 4.), "deeper": ~on & (depth > 4.)}
    return {k: {"share": float(g.float().mean()), "outward": float(out[g].mean()), "size": float(move[g].norm(dim=1).mean()),
                "pressed": float((over[g] > 0).any(1).float().mean())} for k, g in groups.items()}


with torch.no_grad():
    for i, raw in enumerate(raws):
        x = torch.as_tensor(np.asarray(z["frames"][i], np.float32), device=dev)
        print(json.dumps({"state": raw, **row(x)}), flush=True)
    print(json.dumps({"state": "target", **row(tgt)}), flush=True)
