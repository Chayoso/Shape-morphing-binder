"""endgen145t.py OUT_JSONL NAME=END_NPZ[,IND_NPZ] [...] -- D145 post-tag: endgen145.py's end check with a chosen reference
(physmorph metrics, on the card). The last frame of each archive against the independent 300k sample (default
output/gpu/d90/MESH_ind300k.npz, MESH = NAME's prefix before "300k_"; IND_NPZ e.g. a run's ind_floor_NAME.npz for the
floor-standing target of B9) and against the archive's own target sample; the floor = own against independent."""
import json
import sys

from physmorph import gpu  # noqa: F401
import numpy as np
from physmorph import metrics

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
with open(sys.argv[1], "a") as out:
    for spec in sys.argv[2:]:
        name, paths = spec.split("=", 1)
        path, ind_path = (paths.split(",", 1) + [None])[:2]
        ind_path = ind_path or f"{G}/d90/{name.split('300k_')[0]}_ind300k.npz"
        z = np.load(path)
        x, own = np.asarray(z["frames"][-1], np.float32), np.asarray(z["tgt"], np.float32)
        ind = np.asarray(np.load(ind_path)["tgt"], np.float32)
        row = dict(run=name, ref=ind_path.split("/")[-1],
                   ind=dict(sil=metrics.sil_iou(x, ind), chamfer=metrics.chamfer(x, ind)),
                   own=dict(sil=metrics.sil_iou(x, own), chamfer=metrics.chamfer(x, own)), floor=metrics.sil_iou(own, ind))
        out.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
