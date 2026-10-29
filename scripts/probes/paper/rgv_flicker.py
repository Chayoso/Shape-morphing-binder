"""rgv_flicker.py OUT_JSON VIZ_NPZ OLD_SCALES_NPZ|none [IMG name=DIR ...] -- the RGV videos' flicker, measured (CPU).

(A) On the particles: each kept frame's colour position on its bar (t in [0, 1]) per outer-layer particle, for the
first design (values held through each window, V1/V2 divided by each window's own 99th percentile, V3 absolute) and the
second (values linear in time between window centres, no per-window rescaling: V1 the render's share of the pull,
V2 |Delta dF_c| / |dF_c(lambda = 0)|, V3 absolute); the mean |t(k+1) - t(k)| over consecutive kept frames, split into
the pairs across a window boundary and the pairs inside a window. For the first design's V1 the boundary change is
split into (a) the per-window reference and (b) the field: t = (log10 v - log10 ref_w - log10 lo) / 3, so
dt = d log10 v / 3 - d log10 ref / 3. For V3, the window-to-window correlation of g Delta u per particle (over the
particles where u acts in both windows): negative = a period-2 dither (D146 cause 2: u re-solved from zero with fresh
Adam every window).
(B) On the pictures (optional, name=DIR of a video's jpg frames): the mean |change| of the 8-bit RGB between
consecutive frames inside the body (pixels away from the backdrop in both, the central box, no text), split the same
way."""
import json
import sys
from pathlib import Path

import numpy as np

out, viz_p, old_p = sys.argv[1:4]
z = np.load(viz_p)
raws = np.asarray(z["raws"]).astype(int)
win_of = np.asarray(z["win_of_frame"]).astype(int)
t0, t1 = np.asarray(z["w_t0"]), np.asarray(z["w_t1"])
centres = 120.0 * (t0 + t1)
W = len(t0)
lay = np.asarray(z["layer"])
R, P = np.asarray(z["g_rend"], np.float64), np.asarray(z["g_phys"], np.float64)
DU, GATE = np.asarray(z["du"], np.float64), np.asarray(z["ugate"])
DN = np.linalg.norm(np.asarray(z["ddfc"], np.float64), axis=2)
D0 = np.asarray(z["dfc0n"], np.float64)
COS = np.asarray(z["cosrp"], np.float64)
COS5 = np.stack([COS[max(0, w - 2):w + 3].mean(0) for w in range(len(COS))])   # the renderer's +c_avg=5
OLD = old_p != "none"                                       # "none": the second design only
old = json.loads(str(np.load(old_p)["scales"])) if OLD else {}
new = json.loads(str(z["scales"]))
gp99 = np.array([np.percentile(P[w][lay[w]], 99) for w in range(W)])
d099 = np.array([np.percentile(D0[w][lay[w]], 99) for w in range(W)])
ACT = lay & (GATE > .5)


def interp(k):
    r = float(raws[k])
    w0 = int(np.clip(np.searchsorted(centres, r, side="right") - 1, 0, W - 1))
    w1 = min(w0 + 1, W - 1)
    span = centres[w1] - centres[w0]
    al = 0.0 if w1 == w0 or span <= 0 else float(np.clip((r - centres[w0]) / span, 0, 1))
    return w0, w1, al


def lin(A, k):
    w0, w1, al = interp(k)
    return (1 - al) * A[w0] + al * A[w1]


def t_old(k):
    if not OLD:
        return {}
    w = win_of[k]
    lo1 = old["v1w_hi"] * 1e-3
    lo2 = old["v2w_hi"] * 1e-2
    tr = np.clip((np.log10(np.maximum(R[w] / gp99[w], 1e-30)) - np.log10(lo1)) / 3, 0, 1)
    tp = np.clip((np.log10(np.maximum(P[w] / gp99[w], 1e-30)) - np.log10(lo1)) / 3, 0, 1)
    t2 = np.clip((np.log10(np.maximum(DN[w] / d099[w], 1e-30)) - np.log10(lo2)) / 2, 0, 1)
    t3 = np.clip(.5 + .5 * DU[w] / old["v3_max"], 0, 1)
    return {"V1_render": tr, "V1_physics": tp, "V2": t2, "V3": t3}


def t_new(k):
    r, p = lin(R, k), lin(P, k)
    tb = np.clip((np.log10(np.maximum(r, 1e-30)) - np.log10(new["b_lo"])) / 4, 0, 1)
    tc = .5 + .5 * np.clip(lin(COS, k), -1, 1)
    t2 = np.clip(lin(DN, k) / np.maximum(lin(D0, k), 1e-30), 0, 1)
    t3 = np.clip(.5 + .5 * lin(DU, k) / new["v3_max"], 0, 1)
    tc5 = .5 + .5 * np.clip(lin(COS5, k), -1, 1)
    return {"A_share": r / np.maximum(r + p, 1e-30), "B_strength": tb, "C_cosine": tc, "C_cosine_avg5": tc5,
            "D_change": t2, "V3": t3}


res = {"pairs": len(raws) - 1, "boundary_pairs": int(np.sum(win_of[1:] != win_of[:-1])), "old": {}, "new": {}}
acc = {"old": {}, "new": {}}
split = {"field": [], "ref": []}
prev_o, prev_n = t_old(0), t_new(0)
for k in range(1, len(raws)):
    cur_o, cur_n = t_old(k), t_new(k)
    bnd = win_of[k] != win_of[k - 1]
    m = lay[win_of[k]] & lay[win_of[k - 1]]                     # outer-layer particles of both frames' windows
    ma = ACT[win_of[k]] & ACT[win_of[k - 1]]
    for des, cur, prev in (("old", cur_o, prev_o), ("new", cur_n, prev_n)):
        for kind in cur:
            mm = ma if kind == "V3" else m
            v = float(np.mean(np.abs(cur[kind][mm] - prev[kind][mm]))) if mm.any() else np.nan
            acc[des].setdefault(kind, {"boundary": [], "inside": []})["boundary" if bnd else "inside"].append(v)
    if bnd and OLD:
        w, wp = win_of[k], win_of[k - 1]
        split["field"].append(float(np.mean(np.abs(np.log10(np.maximum(R[w][m], 1e-30)) -
                                                   np.log10(np.maximum(R[wp][m], 1e-30))))) / 3)
        split["ref"].append(abs(float(np.log10(gp99[w]) - np.log10(gp99[wp]))) / 3)
    prev_o, prev_n = cur_o, cur_n
for des in acc:
    for kind, d in acc[des].items():
        res[des][kind] = {k2: float(np.nanmean(v)) if v else None for k2, v in d.items()}
if OLD:
    res["old_V1_render_boundary_split"] = {"field_mean_abs": float(np.mean(split["field"])),
                                           "reference_mean_abs": float(np.mean(split["ref"]))}
cors = []
for w in range(W - 1):
    m = ACT[w] & ACT[w + 1]
    if m.sum() > 100:
        cors.append(float(np.corrcoef(DU[w][m], DU[w + 1][m])[0, 1]))
cors = np.array(cors)
res["V3_window_to_window_corr"] = {"median": float(np.median(cors)), "negative_share": float(np.mean(cors < 0)),
                                   "p10": float(np.percentile(cors, 10)), "p90": float(np.percentile(cors, 90)),
                                   "n_pairs": int(len(cors)), "from_window_11": float(np.median(cors[10:]))}

# (B) the pictures
img = {}
if len(sys.argv) > 4:
    from PIL import Image
    for arg in sys.argv[4:]:
        name, d = arg.split("=", 1)
        files = sorted(Path(d).glob("*.jpg"))
        ks = [int(f.stem) for f in files]
        b, ins = [], []
        prev = None
        for f, k in zip(files, ks):
            a = np.asarray(Image.open(f), np.int16)
            H, Wd = a.shape[:2]
            if prev is not None and k - 1 == pk:
                box = (slice(int(.12 * H), int(.80 * H)), slice(int(.15 * Wd), int(.85 * Wd)))
                pa, ca = prev[box], a[box]
                bg = np.median(prev[box[0], :10], axis=1, keepdims=True)    # the backdrop: a vertical gradient
                body = (np.abs(pa - bg).max(2) > 12) & (np.abs(ca - bg).max(2) > 12)
                v = float(np.abs(ca - pa)[body].mean()) if body.any() else np.nan
                (b if win_of[k] != win_of[k - 1] else ins).append(v)
            prev, pk = a, k
        img[name] = {"boundary": float(np.nanmean(b)) if b else None, "inside": float(np.nanmean(ins)) if ins else None,
                     "n": len(files)}
res["pictures_mean_abs_rgb_change"] = img
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
