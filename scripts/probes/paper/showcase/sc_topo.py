"""sc_topo.py OUT_JSONL NAME=NPZ [...] -- SHOWCASE: the topology a run reaches (CPU only). For the run's last frame and for
its own target sample alike: the particles' occupancy on a voxel grid of the target's median nearest-neighbour spacing
(a voxel is occupied when a particle lies within the median 8th-neighbour distance of its centre; one closing step and
the enclosed empty voxels filled: calibrated on the target samples, it gives happy 6, bob 1, A 1, dragon 3 tunnels, while
the covering radius 0.75 d8 with no closing reads the sampling noise as 16-77 micro-tunnels), its Euler characteristic chi = V - E + F - C as a cubical
complex, its pieces b0 (26-connected) and cavities b2, and the tunnels b1 = b0 + b2 - chi. The same operator on both,
so the run's tunnels read against its target's at the same resolution. Also at frames t = 0.5, 1, 2, 4 s and the end
when the archive keeps them (when the holes open)."""
import json
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def euler(M):
    M = np.pad(M, 1)
    C = int(M.sum())
    F = sum(int((M | np.roll(M, 1, ax)).sum()) for ax in range(3))
    E = 0
    for ax in range(3):
        a, b = [d for d in range(3) if d != ax]
        m = M | np.roll(M, 1, a)
        E += int((m | np.roll(m, 1, b)).sum())
    m = M.copy()
    for ax in range(3):
        m = m | np.roll(m, 1, ax)
    return int(m.sum()) - E + F - C


def topo(x, sp, r, lo, dims):
    axes = [lo[i] + sp * (np.arange(dims[i]) + 0.5) for i in range(3)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    t = cKDTree(x)
    d = np.concatenate([t.query(c, 1, workers=8)[0] for c in np.array_split(g, max(1, len(g) // 2_000_000))])
    M = ndimage.binary_fill_holes(ndimage.binary_closing((d <= r).reshape(dims), iterations=1))
    chi = euler(M)
    b0 = ndimage.label(M, structure=np.ones((3, 3, 3)))[1]
    b2 = ndimage.label(~np.pad(M, 1))[1] - 1
    return dict(chi=chi, pieces=int(b0), cavities=int(b2), tunnels=int(b0 + b2 - chi), voxels=int(M.sum()))


with open(sys.argv[1], "a") as out:
    for spec in sys.argv[2:]:
        name, path = spec.split("=", 1)
        z = np.load(path, allow_pickle=True)
        tgt = np.asarray(z["tgt"], np.float64)
        tt = cKDTree(tgt)
        sp = float(np.median(tt.query(tgt, 2, workers=8)[0][:, 1]))
        r = float(np.median(tt.query(tgt, 9, workers=8)[0][:, 8]))
        fr, raws = z["frames"], np.asarray(z["raws"])
        lo = np.minimum(tgt.min(0), np.asarray(fr[-1]).min(0)) - 3 * sp
        hi = np.maximum(tgt.max(0), np.asarray(fr[-1]).max(0)) + 3 * sp
        dims = np.ceil((hi - lo) / sp).astype(int)
        row = dict(run=name, spacing=sp, target=topo(tgt, sp, r, lo, dims))
        ts = {}
        for t_s in (0.5, 1.0, 2.0, 4.0):
            k = int(np.searchsorted(raws, t_s * 240))
            if 0 < k < len(raws) - 1:
                xk = np.asarray(fr[k], np.float64)
                lo_k = np.minimum(lo, xk.min(0) - 3 * sp)
                dims_k = np.ceil((np.maximum(hi, xk.max(0) + 3 * sp) - lo_k) / sp).astype(int)
                ts[f"{raws[k] / 240:.2f}"] = topo(xk, sp, r, lo_k, dims_k)
        row["frames"] = ts
        row["end"] = topo(np.asarray(fr[-1], np.float64), sp, r, lo, dims)
        row["end_s"] = float(raws[-1] / 240)
        out.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
