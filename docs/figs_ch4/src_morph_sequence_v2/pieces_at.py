"""pieces_at.py OUT_JSONL NAME=FRAMES12_NPZ KS [...] -- FIGS_V2 (CPU): the pieces off the body at chosen kept frames, by
D145's definition unchanged (tmp/d145r/pieces145.py: connected sets of particles at 1.5 pitches, pitch = 0.708 x the
target sample's median 8th-neighbour distance; every set but the largest is a piece; >= 10 particles or >= 5 pitches off
= the coordinator's marks for a serious piece). KS: comma list of kept-frame indices ("last" allowed)."""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d145r")
from pieces145 import pieces  # noqa: E402

out = open(sys.argv[1], "a")
args = sys.argv[2:]
for i in range(0, len(args), 2):
    name, path = args[i].split("=", 1)
    z = np.load(path, allow_pickle=True)
    fr = z["frames"]
    tgt = np.asarray(z["tgt"], np.float64)
    a = 0.708 * float(np.median(cKDTree(tgt).query(tgt, 9, workers=8)[0][:, 8]))
    ks = [len(fr) - 1 if k == "last" else int(k) for k in args[i + 1].split(",")]
    for k in ks:
        x = np.asarray(fr[k], np.float64)
        _, _, pcs = pieces(x, a, 8)
        sizes = sorted((s for _, s, _ in pcs), reverse=True)
        serious = [dict(particles=s, gap_pitch=round(g, 2)) for _, s, g in pcs if s >= 10 or g >= 5]
        row = dict(run=name, frame=k, t=round(k * 0.05, 2), pieces=len(pcs), particles=int(sum(sizes)),
                   largest=int(sizes[0]) if sizes else 0, farthest_pitch=round(max((g for *_, g in pcs), default=0.0), 2),
                   serious=serious)
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps(row), flush=True)
