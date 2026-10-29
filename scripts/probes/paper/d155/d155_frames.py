"""d155_frames.py FRAMES_NPZ OUT_JSON [WORKERS] -- D155 (CPU, read-only): how the visible deformation is spread over a
run, from its kept frames (every 12th simulated frame, 240 frames a second).

pitch = 0.708 x the target sample's median 8th-neighbour distance (D95). Outer layer of a state: |x - mean of its 32
nearest| >= 0.5 x its median 8th-neighbour distance (D152's layer); normals: x minus the mean of its 24 nearest.
Per kept-frame pair (i -> i+1):
  mv_all     mean |dx| over all particles (pitches; the m_ log's move)
  mv_lay     rms |dx| over the END state's outer layer (Lagrangian: the particles that end on the surface)
  mv_lay_n   rms of dx . n over them, n the normal at frame i (the surface moving along itself is invisible)
Per kept frame i:
  path       1 - mean |x_i - x_end| / mean |x_0 - x_end| (D113's arrival)
  lag_lay    rms |x_i - x_end| over the end's outer layer (pitches): how far the surface particles still have to go
  lag_n      rms of (x_i - x_end) . n_end over them: the NORMAL part still to come (tangential sliding is invisible)
  off_rms    the visible surface's distance from the end shape: the frame's own outer layer against the end's outer
             layer, signed normal offset (mean over the 8 nearest end-layer particles of (x - y) . n_end(y)), rms
  gap_p95    the end's outer layer to the frame's outer layer, nearest distance, 95th percentile (missing parts)"""
import json
import sys
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree

path, out = sys.argv[1], sys.argv[2]
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
z = np.load(path)
FR = np.asarray(z["frames"], np.float32)
RAWS = np.asarray(z["raws"]).astype(int)
TGT = np.asarray(z["tgt"], np.float64)
DN = int(z["deliver_n"])
PITCH = 0.708 * float(np.median(cKDTree(TGT).query(TGT[::50], 9)[0][:, 8]))
X_END = FR[-1].astype(np.float64)


def layer_normals(x, k_lay=33, k_n=25):
    tree = cKDTree(x)
    sp = float(np.median(tree.query(x[::15], 9)[0][:, 8]))
    d, nb = tree.query(x, k_lay, workers=2)
    off = x - x[nb[:, 1:]].mean(1)
    lay = np.linalg.norm(off, axis=1) >= 0.5 * sp
    off_n = x - x[nb[:, 1:k_n]].mean(1)
    nrm = off_n / np.maximum(np.linalg.norm(off_n, axis=1, keepdims=True), 1e-12)
    return lay, nrm


LAY_END, N_END = layer_normals(X_END)
YE = X_END[LAY_END]
NE = N_END[LAY_END]
TREE_E = cKDTree(YE)
D0 = float(np.linalg.norm(FR[0].astype(np.float64) - X_END, axis=1).mean())


def per_frame(i):
    x = FR[i].astype(np.float64)
    row = {"i": int(i), "raw": int(RAWS[i])}
    row["path"] = 1.0 - float(np.linalg.norm(x - X_END, axis=1).mean()) / D0
    row["lag_lay"] = float(np.sqrt(((x[LAY_END] - X_END[LAY_END]) ** 2).sum(1).mean()) / PITCH)
    rn = ((x[LAY_END] - X_END[LAY_END]) * N_END[LAY_END]).sum(1) / PITCH   # the normal part still to come
    row["lag_n"] = float(np.sqrt((rn ** 2).mean()))
    row["lag_n_p95"] = float(np.percentile(np.abs(rn), 95))
    row["lag_n_mean"] = float(rn.mean())
    lay, nrm = layer_normals(x)
    xl = x[lay]
    _, j = TREE_E.query(xl, 8, workers=2)
    off = ((xl[:, None, :] - YE[j]) * NE[j]).sum(2).mean(1) / PITCH
    row["off_rms"] = float(np.sqrt((off ** 2).mean()))
    row["off_mean"] = float(off.mean())
    row["off_p95"] = float(np.percentile(np.abs(off), 95))
    g = cKDTree(xl).query(YE, 1, workers=2)[0] / PITCH
    row["gap_mean"] = float(g.mean())
    row["gap_p95"] = float(np.percentile(g, 95))
    row["gap_p99"] = float(np.percentile(g, 99))
    if i + 1 < len(FR):
        y = FR[i + 1].astype(np.float64)
        dx = (y - x) / PITCH
        row["mv_all"] = float(np.linalg.norm(dx, axis=1).mean())
        dl = dx[LAY_END]
        row["mv_lay"] = float(np.sqrt((dl ** 2).sum(1).mean()))
        row["mv_lay_n"] = float(np.sqrt(((dl * nrm[LAY_END]).sum(1) ** 2).mean()))
        row["mv_bulk"] = float(np.sqrt((dx[~LAY_END] ** 2).sum(1).mean()))
    return row


def main():
    with Pool(workers) as pool:
        rows = sorted(pool.map(per_frame, range(len(FR)), chunksize=4), key=lambda r: r["i"])
    res = {"file": path, "pitch": PITCH, "deliver_n": DN, "frames": len(FR), "layer_end": int(LAY_END.sum()),
           "rows": rows}
    json.dump(res, open(out, "w"))
    print(f"{path}: pitch {PITCH:.5f}, {len(FR)} frames, end layer {int(LAY_END.sum())}", flush=True)


if __name__ == "__main__":
    main()
