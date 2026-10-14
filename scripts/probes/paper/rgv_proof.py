"""rgv_proof.py OUT_JSON NAME=RUN_JSON,FRAMES12 [NAME=...] -- the RGV data run is the deliverable's run (read-only
hooks): against the unhooked render arm (D145 B1's L97) and the earlier hooked run (B7's G97, same code and seed), per
pair: the end metrics, the windows, the per-window history (merit, transport energy, silhouette term, lambda) over
the common windows, and the particles' distance at matched simulated frames (median over the outer layer and the bulk,
in pitches; the same particles, in order). The first pair named is the reference; the pair G97 / L97 is the replay
spread of two runs of the same code (CUDA atomics) that the RGV run's distance to either is read against. CPU only."""
import itertools
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

out = sys.argv[1]
runs = {}
for arg in sys.argv[2:]:
    name, paths = arg.split("=", 1)
    j, f = paths.split(",")
    arm = next(iter(json.load(open(j))["arms"].values()))
    hist = [r for r in arm["history"] if r.get("frame_end")]
    z = np.load(f)
    runs[name] = dict(metrics=arm["metrics"], hist=hist, frames=z["frames"], raws=[int(v) for v in np.asarray(z["raws"])],
                      tgt=np.asarray(z["tgt"], np.float64))
tgt = next(iter(runs.values()))["tgt"]
pitch = 0.708 * float(np.median(cKDTree(tgt).query(tgt[::50], 9)[0][:, 8]))
res = {"pitch": pitch, "runs": {}, "pairs": {}}
for nm, r in runs.items():
    m = r["metrics"]
    res["runs"][nm] = {"windows": len(r["hist"]), "kept_frames": len(r["raws"]), "last_raw": int(r["raws"][-1]),
                       **{k: m.get(k) for k in ("silIoU", "chamfer", "hole", "jitter_rel", "detFmin", "minutes")
                          if k in m}}
for (na, a), (nb, b) in itertools.combinations(runs.items(), 2):
    nw = min(len(a["hist"]), len(b["hist"]))
    rel = {}
    for key in ("loss", "transport_energy", "d_sil", "lambda", "selection_merit"):
        va = np.array([h.get(key) if h.get(key) is not None else np.nan for h in a["hist"][:nw]], float)
        vb = np.array([h.get(key) if h.get(key) is not None else np.nan for h in b["hist"][:nw]], float)
        d = np.abs(va - vb) / np.maximum(np.abs(vb), 1e-30)
        rel[key] = {"w1": float(d[0]) if nw else None, "median": float(np.nanmedian(d)) if nw else None,
                    "max_first10": float(np.nanmax(d[:10])) if nw else None}
    common = sorted(set(a["raws"]) & set(b["raws"]))
    pick = common[:: max(1, len(common) // 24)] + [common[-1]]
    traj = []
    for raw in pick:
        xa = np.asarray(a["frames"][a["raws"].index(raw)], np.float64)
        xb = np.asarray(b["frames"][b["raws"].index(raw)], np.float64)
        tree = cKDTree(xa)
        sp = float(np.median(tree.query(xa[::15], 9)[0][:, 8]))
        nbr = tree.query(xa, 33)[1][:, 1:]
        lay = np.linalg.norm(xa - xa[nbr].mean(1), axis=1) >= 0.5 * sp
        dep = cKDTree(xa[lay]).query(xa, 1)[0] / pitch
        d = np.linalg.norm(xa - xb, axis=1) / pitch
        traj.append({"raw": raw, "t": raw / 240.0, "layer": float(np.median(d[lay])), "bulk": float(np.median(d[dep > 2]))})
    res["pairs"][f"{na}|{nb}"] = {"common_windows": nw, "history_rel": rel, "matched": traj,
                                  "layer_max": max(t["layer"] for t in traj), "bulk_max": max(t["bulk"] for t in traj)}
    print(na, nb, "windows", len(a["hist"]), len(b["hist"]), "w1 rel merit/transport/sil",
          {k: round(rel[k]["w1"], 7) for k in ("loss", "transport_energy", "d_sil")},
          "layer/bulk max", round(res["pairs"][f"{na}|{nb}"]["layer_max"], 3), round(res["pairs"][f"{na}|{nb}"]["bulk_max"], 3),
          flush=True)
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res["runs"], indent=1))
