"""d152_offsets.py OUT_JSON MESH:SEED [...] -- D152 (H1/H3): where the end surfaces sit, in pitches, along each outer-layer
particle's outward normal (the mean over the other state's 8 nearest outer-layer particles of (x - y) . n): the render
arm L, the twin P and the run's own target sample, each against the independent sample (a positive mean = further out),
and L against P per region (outer layer; B7's rim of three axis views). CPU only, read-only. Measurement only."""
import json
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"


def layer(x):
    tr = cKDTree(x)
    sp = float(np.median(tr.query(x[::15], 9)[0][:, 8]))
    nb = tr.query(x, 33)[1][:, 1:]
    off = x - x[nb].mean(1)
    lay = np.linalg.norm(off, axis=1) >= 0.5 * sp
    return lay, off / np.maximum(np.linalg.norm(off, axis=1, keepdims=True), 1e-12)


def rim3(x, lay, pitch):
    rim = np.zeros(len(x), bool)
    for hh, vv in ((0, 1), (2, 1), (0, 2)):
        ij = np.floor((x[:, [hh, vv]] - x[:, [hh, vv]].min(0)) / pitch).astype(int)
        occ = np.zeros(ij.max(0) + 3, bool)
        occ[ij[:, 0] + 1, ij[:, 1] + 1] = True
        edge = occ & ~ndimage.binary_erosion(occ)
        rim |= edge[ij[:, 0] + 1, ij[:, 1] + 1]
    return rim & lay


def offset(x, nrm, other, pitch):
    lo, _ = layer(other)
    yo = other[lo]
    idx = cKDTree(yo).query(x, 8)[1]
    return (((x[:, None, :] - yo[idx]) * nrm[:, None, :]).sum(2)).mean(1) / pitch


res = {}
for spec in sys.argv[2:]:
    mesh, seed = spec.split(":")
    zl = np.load(f"{G}/d145t_b1/{mesh}300k_L{seed}_end1.npz")
    zp = np.load(f"{G}/d145t_b1/{mesh}300k_P{seed}_end1.npz")
    own = np.asarray(zl["tgt"], np.float64)
    ind = np.asarray(np.load(f"{G}/d90/{mesh}_ind300k.npz")["tgt"], np.float64)
    L, P = np.asarray(zl["frames"][-1], np.float64), np.asarray(zp["frames"][-1], np.float64)
    pitch = 0.708 * float(np.median(cKDTree(own).query(own[::50], 9)[0][:, 8]))
    r = {"pitch": pitch}
    for nm, x in (("L", L), ("P", P), ("own", own)):
        lay, nrm = layer(x)
        rm = rim3(x, lay, pitch)
        h = offset(x[lay], nrm[lay], ind, pitch)
        hr = offset(x[rm], nrm[rm], ind, pitch)
        r[f"{nm}_vs_ind"] = {"layer_mean": float(h.mean()), "layer_rms": float(np.sqrt((h ** 2).mean())),
                             "rim3_mean": float(hr.mean()), "rim3_rms": float(np.sqrt((hr ** 2).mean())),
                             "layer_n": int(lay.sum())}
    lay, nrm = layer(L)
    rm = rim3(L, lay, pitch)
    h = offset(L[lay], nrm[lay], P, pitch)
    hr = offset(L[rm], nrm[rm], P, pitch)
    r["L_vs_P"] = {"layer_mean": float(h.mean()), "layer_rms": float(np.sqrt((h ** 2).mean())),
                   "rim3_mean": float(hr.mean()), "rim3_rms": float(np.sqrt((hr ** 2).mean()))}
    res[spec] = r
    print(spec, json.dumps(r), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
