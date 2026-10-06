"""d116_pulse.py FRAMES12_NPZ RUN_JSON [TAIL_FRAC=0.3] -- D116 (3): is the tail's normal motion a pulse that comes back?
Outer particles and normals as in d116_tail.py. Per consecutive pair of kept frames in the tail, the per-particle
normal displacement dn; per pair of successive pairs the correlation over the particles of dn_k and dn_k+1 (-1: what
one pair moves out the next moves back), split by where the middle frame sits in its window (driven: the first half
of the window's delivered frames, released: the second). And over whole tail windows: the normal excursion
(rms over particles of the largest |n.(x - x_start)| reached among the window's kept frames) against the net normal
move from the window's first kept frame to the next window's first kept frame."""
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
        if a <= raw < b:
            return w, (raw - a) / max(b - a, 1)
    return len(ends), 0.0
xe = np.asarray(F[-1], np.float32)
d, nb = cKDTree(xe).query(xe, 17)
off = xe[nb[:, 1:]].mean(1) - xe
outer = np.linalg.norm(off, axis=1) > 0.25 * np.median(d[:, 1:], axis=1)
nrm = -off[outer] / np.linalg.norm(off[outer], axis=1, keepdims=True)
pitch = float(np.median(d[:, 1]))
k0 = int(len(raws) * (1 - tail))
X = [np.asarray(F[k], np.float32)[outer] for k in range(k0, len(raws))]
P = [place(int(r)) for r in raws[k0:]]
dn = [((b - a) * nrm).sum(1) / pitch for a, b in zip(X, X[1:])]
corr = {"driven": [], "released": []}
for i in range(len(dn) - 1):
    c = float(np.corrcoef(dn[i], dn[i + 1])[0, 1])
    corr["driven" if P[i + 1][1] < 0.5 else "released"].append(c)
by_w = {}
for i, (w, ph) in enumerate(P):
    by_w.setdefault(w, []).append(i)
exc, net = [], []
ws = sorted(by_w)
for w, w2 in zip(ws, ws[1:]):
    idx = by_w[w]
    x0 = X[idx[0]]
    proj = np.stack([((X[i] - x0) * nrm).sum(1) for i in idx[1:] + [by_w[w2][0]]]) / pitch
    exc.append(float(np.sqrt((np.abs(proj).max(0) ** 2).mean())))
    net.append(float(np.sqrt((proj[-1] ** 2).mean())))
exc, net = np.array(exc), np.array(net)
print(json.dumps(dict(file=sys.argv[1].split("/")[-1], pairs=len(dn), corr_driven_mid=float(np.median(corr["driven"])),
                      corr_released_mid=float(np.median(corr["released"])), n_corr=[len(corr["driven"]), len(corr["released"])],
                      windows=len(exc), excursion_rms_med=float(np.median(exc)), net_rms_med=float(np.median(net)),
                      returned_share_med=float(np.median(1 - net / np.maximum(exc, 1e-12))))))
