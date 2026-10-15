"""endgen145.py OUT_JSONL MESH=FRAMES_NPZ [...] -- D145: B5's end check for any frames12 archive (the last frame against the
independent 300k sample of output/gpu/d90 and against the archive's own target sample; the floor own vs independent)."""
import json
import sys

from physmorph import gpu  # noqa: F401
import numpy as np
from physmorph import metrics

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
with open(sys.argv[1], "a") as out:
    for spec in sys.argv[2:]:
        mesh, path = spec.split("=", 1)
        z = np.load(path)
        x, own = np.asarray(z["frames"][-1], np.float32), np.asarray(z["tgt"], np.float32)
        ind = np.asarray(np.load(f"{G}/d90/{mesh}_ind300k.npz")["tgt"], np.float32)
        row = dict(run=path.split("/")[-1].replace("_frames12.npz", ""), ind=dict(sil=metrics.sil_iou(x, ind), chamfer=metrics.chamfer(x, ind)),
                   own=dict(sil=metrics.sil_iou(x, own), chamfer=metrics.chamfer(x, own)), floor=metrics.sil_iou(own, ind))
        out.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
