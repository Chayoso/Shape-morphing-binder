"""d155b_forensic.py MESH [MESH ...] -- D155b's end-quality forensic (CPU, read-only): where on the target surface the
moving-target runs' ends deviate, and when that deviation forms.

Reference = the independent 40k sample (d92/ind_MESH.npz 'tgt', the yardstick's and the display's referee). Its outer
layer (|y - mean of 32 nearest| >= 0.5 x its median 8th-neighbour distance), normals (y minus the mean of its 24 nearest)
and regions on it: rim (outer layer on the outline of any of the 18 loss views, a one-pitch occupancy grid), thin (max-ball
thickness below two MPM cells), crevice (layer neighbours within 4 pitches on average above the tangent plane, > +0.15),
rest. A state's deviation at each reference point: the signed normal offset of the state's own outer layer from it (the
mean over the state's 8 nearest outer-layer particles of (x - y) . n_y, pitches; + = the run's surface lies outside) and
the gap (the nearest state outer-layer particle, pitches). Per region: rms offset, mean offset, the share of points whose
|offset| > 0.5 pitch. Runs: the tag's V / W at seeds 97 / 99 / 101 and A3 / A3G (L and P) at seed 97, end frame; and the
deviation over time (every 3rd kept frame) for W97 against A3GP97 and V97 against A3GL97, aligned at the arrival (the tag:
u's gate over half the layer; A3G: the ramp's end)."""
import json
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
O = f"{G}/d155/gate"


def views18():
    out, elevs, n = [], (0.0, 0.5, -0.5), 6
    for i, phi in enumerate(elevs):
        off = (i / len(elevs)) * (2 * np.pi / n)
        for t in np.linspace(0, 2 * np.pi, n, endpoint=False):
            out.append((float(t + off), float(phi)))
    return out


def layer_normals(x, k_lay=33, k_n=25):
    tree = cKDTree(x)
    sp = float(np.median(tree.query(x[::15], 9)[0][:, 8]))
    _, nb = tree.query(x, k_lay, workers=2)
    off = x - x[nb[:, 1:]].mean(1)
    lay = np.linalg.norm(off, axis=1) >= 0.5 * sp
    off_n = x - x[nb[:, 1:k_n]].mean(1)
    return lay, off_n / np.maximum(np.linalg.norm(off_n, axis=1, keepdims=True), 1e-12)


def regions(y, lay, nrm, pitch, cell):
    rim = np.zeros(len(y), bool)
    for th, phi in views18():
        right = np.array([np.cos(th), 0.0, -np.sin(th)])
        up = np.array([-np.sin(phi) * np.sin(th), np.cos(phi), -np.sin(phi) * np.cos(th)])
        p = np.stack([y @ right, y @ up], 1)
        ij = np.floor((p - p.min(0)) / pitch).astype(int)
        occ = np.zeros(ij.max(0) + 3, bool)
        occ[ij[:, 0] + 1, ij[:, 1] + 1] = True
        edge = occ & ~ndimage.binary_erosion(occ)
        rim |= edge[ij[:, 0] + 1, ij[:, 1] + 1]
    rim &= lay
    lo = y.min(0) - 3 * pitch
    dims = np.ceil((y.max(0) + 3 * pitch - lo) / pitch).astype(int) + 1
    ijk = np.floor((y - lo) / pitch).astype(int)
    occ = np.zeros(dims, bool)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    occ = ndimage.binary_fill_holes(ndimage.binary_closing(occ, iterations=1))
    D = ndimage.distance_transform_edt(occ)
    thick = np.zeros(dims)
    for r in range(1, int(D.max()) + 1):
        core = D >= r
        if not core.any():
            break
        thick[(ndimage.distance_transform_edt(~core) <= r) & occ] = 2.0 * r * pitch
    h = thick[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    thin = lay & (h > 0) & (h < 2.0 * cell)
    li = np.flatnonzero(lay)
    lt = cKDTree(y[li])
    conc = np.zeros(len(y))
    for i, js in zip(li, lt.query_ball_point(y[li], 4.0 * pitch)):
        js = li[np.asarray(js, int)]
        js = js[js != i]
        if len(js) >= 4:
            v = y[js] - y[i]
            conc[i] = float(np.mean((v @ nrm[i]) / np.maximum(np.linalg.norm(v, axis=1), 1e-12)))
    crev = lay & (conc > 0.15)
    rest = lay & ~rim & ~thin & ~crev
    return {"rim": rim, "thin": thin, "crevice": crev, "rest": rest, "layer": lay}


def deviation(x, Y, NY, R, pitch):
    lay_x, _ = layer_normals(x)
    xl = x[lay_x]
    t = cKDTree(xl)
    _, j = t.query(Y, 8, workers=2)
    off = ((xl[j] - Y[:, None, :]) * NY[:, None, :]).sum(2).mean(1) / pitch
    gap = t.query(Y, 1, workers=2)[0] / pitch
    out = {}
    for k, m in R.items():
        o, g_ = off[m], gap[m]
        out[k] = dict(n=int(m.sum()), rms=float(np.sqrt((o ** 2).mean())), mean=float(o.mean()),
                      over_half=float((np.abs(o) > 0.5).mean()), gap=float(g_.mean()))
    return out


def arrival(name):
    a = next(iter(json.load(open(f"{O}/{name}.json"))["arms"].values()))
    acc = [h for h in a["history"] if h.get("frame_end") and not h.get("null_commit")]
    t = next((h["frame_end"] / 240 for h in acc if (h.get("u_gate") or 0) >= 0.5), 0.0)
    if any("morph_s" in h for h in acc):
        t = max(t, next(h["frame_end"] / 240 for h in acc if (h.get("morph_s") or 0) >= 1.0))
    return t


res = {}
for mesh in sys.argv[1:]:
    ind = np.asarray(np.load(f"{G}/d92/ind_{mesh}.npz")["tgt"], np.float64)
    pitch = 0.708 * float(np.median(cKDTree(ind).query(ind[::20], 9)[0][:, 8]))
    J = json.load(open(f"{O}/{mesh}40k_V97.json"))
    cell = float(J["provenance"]["mpm"]["dx"])
    lay, nrm = layer_normals(ind)
    R_all = regions(ind, lay, nrm, pitch, cell)
    li = np.flatnonzero(lay)
    Y, NY = ind[li], nrm[li]
    R = {k: v[li] for k, v in R_all.items()}
    print(f"{mesh}: pitch {pitch:.4f} wu, cell {cell:.4f}; IND layer {len(li)}: "
          + ", ".join(f"{k} {int(v.sum())}" for k, v in R.items()), flush=True)
    res[mesh] = {"end": {}, "time": {}}
    for arm in ("V97", "V99", "V101", "A3L97", "A3GL97", "W97", "W99", "W101", "A3P97", "A3GP97"):
        z = np.load(f"{O}/{mesh}40k_{arm}_frames12.npz")
        d = deviation(np.asarray(z["frames"][-1], np.float64), Y, NY, R, pitch)
        res[mesh]["end"][arm] = d
        print(f"  end {arm:7s} " + "  ".join(f"{k} rms {v['rms']:.3f} mean {v['mean']:+.3f} >0.5 {v['over_half']:.3f}"
                                                for k, v in d.items() if k != "layer")
              + f"  | layer rms {d['layer']['rms']:.3f}", flush=True)
    for a_, b_ in (("W97", "A3GP97"), ("V97", "A3GL97")):
        for arm in (a_, b_):
            z = np.load(f"{O}/{mesh}40k_{arm}_frames12.npz")
            raws = np.asarray(z["raws"])
            ta = arrival(f"{mesh}40k_{arm}")
            rows = []
            for i in range(0, len(raws), 3):
                t = raws[i] / 240.0
                if t < ta - 1e-9:
                    continue
                d = deviation(np.asarray(z["frames"][i], np.float64), Y, NY, R, pitch)
                rows.append(dict(dt=t - ta, **{k: (v["rms"], v["mean"]) for k, v in d.items()}))
            res[mesh]["time"][arm] = rows
            pick = [r for r in rows if any(abs(r["dt"] - q) < 0.13 for q in (0, 1, 2, 3, 4, 6))] + rows[-1:]
            print(f"  time {arm:7s} (arrival {ta:.2f} s): " + "; ".join(
                f"+{r['dt']:.1f}s layer {r['layer'][0]:.3f} rim {r['rim'][0]:.3f} thin {r['thin'][0]:.3f} crev "
                f"{r['crevice'][0]:.3f}" for r in pick), flush=True)
json.dump(res, open(f"{O}/forensic_b.json", "w"))
