"""d116_tail.py FRAMES12_NPZ RUN_JSON [TAIL_FRAC=0.3] -- D116: what moves the surface in the tail, frame by frame.
Outer particles: on the last kept frame, those whose 16 nearest neighbours' centroid lies more than 0.25 of their
median neighbour distance away (one-sided neighbourhoods), normal = from the centroid outward. Per consecutive kept
pair in the last TAIL_FRAC of the frames: the displacement of the outer particles split into normal and tangential
parts (rms, in sampling pitches), the median cosine of successive displacements per particle (-1: back and forth),
and each kept frame's place in its window (the delivered frame range of the window from the records)."""
import json, sys, numpy as np
from scipy.spatial import cKDTree
z = np.load(sys.argv[1], allow_pickle=True)
arm = json.load(open(sys.argv[2]))["arms"]; arm = arm[list(arm)[0]]
tail = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
raws, F = np.asarray(z["raws"]), z["frames"]
ends = sorted({int(r["frame_end"]) for r in arm["history"] if r.get("frame_end") and not r.get("null_commit")})
starts = [0] + ends[:-1]
def place(raw):
    for w, (a, b) in enumerate(zip(starts, ends)):
        if a <= raw < b or (raw == b and w == len(ends) - 1):
            return w, (raw - a) / max(b - a, 1)
    return -1, -1.0
xe = np.asarray(F[-1], np.float32)
tree = cKDTree(xe)
d, nb = tree.query(xe, 17)
cen = xe[nb[:, 1:]].mean(1)
off = cen - xe
pitch = np.median(d[:, 1])
outer = np.linalg.norm(off, axis=1) > 0.25 * np.median(d[:, 1:], axis=1)
nrm = -off[outer] / np.linalg.norm(off[outer], axis=1, keepdims=True)
k0 = int(len(raws) * (1 - tail))
rows, prev = [], None
for k in range(max(1, k0), len(raws)):
    dx = (np.asarray(F[k], np.float32) - np.asarray(F[k - 1], np.float32))[outer] / pitch
    dn = (dx * nrm).sum(1)
    dt = dx - dn[:, None] * nrm
    cosv = None
    if prev is not None:
        num = (dx * prev).sum(1); den = np.linalg.norm(dx, axis=1) * np.linalg.norm(prev, axis=1)
        ok = den > 1e-12
        cosv = float(np.median(num[ok] / den[ok])) if ok.any() else None
    w, ph = place(int(raws[k]))
    rows.append(dict(k=k, raw=int(raws[k]), window=w, phase=round(ph, 2), n_rms=float(np.sqrt((dn ** 2).mean())),
                     t_rms=float(np.sqrt((dt ** 2).sum(1).mean())), n_mean=float(dn.mean()), cos_prev=cosv))
    prev = dx
n = np.array([r["n_rms"] for r in rows]); t = np.array([r["t_rms"] for r in rows])
c = np.array([r["cos_prev"] for r in rows if r["cos_prev"] is not None])
print(json.dumps(dict(file=sys.argv[1].split("/")[-1], outer=int(outer.sum()), pitch=float(pitch), pairs=len(rows),
                      n_rms_med=float(np.median(n)), t_rms_med=float(np.median(t)), cos_prev_med=float(np.median(c)),
                      cos_prev_neg_share=float((c < 0).mean()), windows=len(ends))))
by = {}
for r in rows:
    b = "driven" if r["phase"] < 0.5 else "released"
    by.setdefault(b, []).append(r)
for b, rr in by.items():
    print(b, len(rr), "n_rms %.4f t_rms %.4f n_mean %+.5f cos_prev %.3f" % (np.median([r["n_rms"] for r in rr]), np.median([r["t_rms"] for r in rr]),
          np.median([r["n_mean"] for r in rr]), np.median([r["cos_prev"] for r in rr if r["cos_prev"] is not None])))
for r in rows[:12]:
    print(r)
