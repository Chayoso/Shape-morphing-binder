"""d155d_silcourse.py MESH -- D155d forensic (CPU): when does the ramp's silhouette shortfall form? A CPU proxy of the
yardstick silhouette: the particles splatted as discs of 0.8 pitch (the exterior's offset) in the loss's 18 orthographic
views (6 azimuths x elevations 0 / +-0.5 rad, D155b's views18) at 256 px over 1.25 x the independent sample's largest
|x|, against the independent 40k sample splatted alike; per frame the mean over the views of the mismatched pixels over
the union (1 - IoU). Every kept frame of the tag's V seeds and A3D's L seeds, by simulated time and by time after the
arrival; the end value beside the run's own yardstick silhouette (the proxy's check)."""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
O = f"{G}/d155/gate"
mesh = sys.argv[1]
RES = 256


def views18():
    out, elevs, n = [], (0.0, 0.5, -0.5), 6
    for i, phi in enumerate(elevs):
        off = (i / len(elevs)) * (2 * np.pi / n)
        for t in np.linspace(0, 2 * np.pi, n, endpoint=False):
            out.append((float(t + off), float(phi)))
    return out


def basis(theta, phi):
    d = np.array([np.cos(phi) * np.cos(theta), np.sin(phi), np.cos(phi) * np.sin(theta)])
    up = np.array([0.0, 1.0, 0.0])
    u = np.cross(up, d)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    return u, v


ind = np.asarray(np.load(f"{G}/d92/ind_{mesh}.npz")["tgt"], np.float64)
pitch = 0.708 * float(np.median(cKDTree(ind).query(ind[::20], 9)[0][:, 8]))
ext = 1.25 * float(np.abs(ind).max())
px = 2 * ext / RES
r_px = 0.8 * pitch / px
off = [(i, j) for i in range(-int(np.ceil(r_px)), int(np.ceil(r_px)) + 1)
       for j in range(-int(np.ceil(r_px)), int(np.ceil(r_px)) + 1) if i * i + j * j <= r_px * r_px]
V = [basis(*v) for v in views18()]


def masks(x):
    out = []
    for u, v in V:
        a = np.floor((x @ u + ext) / px).astype(np.int64)
        b = np.floor((x @ v + ext) / px).astype(np.int64)
        m = np.zeros((RES, RES), bool)
        for i, j in off:
            aa, bb = np.clip(a + i, 0, RES - 1), np.clip(b + j, 0, RES - 1)
            m[aa, bb] = True
        out.append(m)
    return out


ref = masks(ind)


def mis(x):
    s = []
    for m, r in zip(masks(x), ref):
        s.append(np.logical_xor(m, r).sum() / max(np.logical_or(m, r).sum(), 1))
    return float(np.mean(s))


R = {r["name"]: r for r in (json.loads(line) for line in open(f"{O}/gate_all.jsonl"))}
print(f"{mesh}: pitch {pitch:.4f}, {RES} px, disc {r_px:.2f} px")
for arm in ("V97", "V99", "V101", "A3DL97", "A3DL99", "A3DL101"):
    name = f"{mesh}40k_{arm}"
    z = np.load(f"{O}/{name}_frames12.npz")
    fr, raws, dn = z["frames"], z["raws"], int(z["deliver_n"])
    a = next(iter(json.load(open(f"{O}/{name}.json"))["arms"].values()))
    acc = [h for h in a["history"] if h.get("frame_end") and not h.get("null_commit")]
    t_u = next((h["frame_end"] / 240 for h in acc if (h.get("u_gate") or 0) >= 0.5), 0.0)
    t_s = next((h["frame_end"] / 240 for h in acc if (h.get("morph_s") if h.get("morph_s") is not None else 1.0) >= 1.0), 0.0)
    ta = max(t_u, t_s)
    rows = []
    for k in range(0, len(raws), 2):
        rows.append((raws[k] / 240.0, mis(np.asarray(fr[k], np.float64))))
    i_end = int(np.argmin(np.abs(raws - (dn - 1))))
    e_end = mis(np.asarray(fr[i_end], np.float64))
    pick = [(t, e) for t, e in rows if any(abs(t - q) < 0.06 for q in (1, 2, 3, 3.5, 4, 5, 6, 7, 8, 9, 10))]
    print(f"  {arm:8s} arrival {ta:.2f} s; " + "; ".join(f"{t:.1f}s {e:.4f}" for t, e in pick)
          + f"; END {raws[i_end] / 240:.2f}s {e_end:.4f} (yardstick sil {R[name]['end']['yard_sil'] * 1e3:.3f})", flush=True)
    by_after = [(t - ta, e) for t, e in rows if t >= ta]
    print(f"           after arrival: " + "; ".join(f"+{t:.1f}s {e:.4f}" for t, e in by_after
                                                     if any(abs(t - q) < 0.06 for q in (0, 1, 2, 3, 4, 5, 6, 7, 8))), flush=True)
