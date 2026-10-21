"""ext_end145.py OUT_JSONL MESH:METHOD:LABEL=ARCHIVE [...] -- D145's methods table: every method's end frame read the same
way (physmorph metrics on the card).

For each archive (the pipeline's archive format: frames, raws, tgt; an end1 archive or a run's frames12): its last frame
that is not NaN (a frame a baseline's sampler could not fill is NaN, convert_b14.py / convert_pair.py) against the
independent 300k sample output/gpu/d90/MESH_ind300k.npz: silhouette IoU and chamfer, the floor (the archive's own target
sample against the independent one), and the thin-part coverage (physmorph.thin) of the independent sample's thin set,
built once per mesh in the shape's MPM cell dx = (ppc x source fill / N)^(1/3) of the tag's L97 run (one sphere, one dx
for every mesh) with ref_n 40000 -- so every method, ours included, is scored on one thin set."""
import json
import sys

from physmorph import gpu  # noqa: F401
import numpy as np
from physmorph import metrics
from physmorph.thin import thin_metrics, thin_set

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
cache = {}


def mesh_refs(mesh):
    if mesh not in cache:
        ind = np.asarray(np.load(f"{G}/d90/{mesh}_ind300k.npz")["tgt"], np.float32)
        prov = json.load(open(f"{G}/d145t_b1/{mesh}300k_L97.json"))["provenance"]
        dx = (prov["ppc"] * prov["density"]["source_fill"] / prov["n"]) ** (1.0 / 3.0)
        cache[mesh] = (ind, thin_set(ind, dx, 40000), dx)
    return cache[mesh]


with open(sys.argv[1], "a") as out:
    for spec in sys.argv[2:]:
        key, path = spec.split("=", 1)
        mesh, method, label = key.split(":")
        ind, ts, dx = mesh_refs(mesh)
        z = np.load(path, allow_pickle=True)
        F = z["frames"]
        k = len(F) - 1
        while k > 0 and not np.isfinite(np.asarray(F[k][:16])).all():
            k -= 1
        x, own = np.asarray(F[k], np.float32), np.asarray(z["tgt"], np.float32)
        th = thin_metrics(x, ts)
        th_floor = thin_metrics(own, ts)
        row = dict(mesh=mesh, method=method, label=label, path=path, end_frame=int(k), n_frames=int(len(F)),
                   sil_ind=metrics.sil_iou(x, ind), chamfer_ind=metrics.chamfer(x, ind), sil_own=metrics.sil_iou(x, own),
                   floor=metrics.sil_iou(own, ind), floor_chamfer=metrics.chamfer(own, ind), dx=dx,
                   thin_ind=th.get("thin_uncovered"), thin_ind_world=th.get("thin_uncovered_world"),
                   thin_n=th.get("thin_n"), thin_floor=th_floor.get("thin_uncovered"))
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps({k2: v for k2, v in row.items() if k2 != "path"}), flush=True)
