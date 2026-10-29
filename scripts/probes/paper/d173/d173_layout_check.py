"""d173_layout_check.py OUT_JSON -- D173 (a colour layout the twin cannot reach by accident), CPU, before any run.

The twin (lambda = 0) reads only the set of positions, so where each source particle ends is the transport's own map,
blind to the paint. For candidate (source paint, target paint) pairs with the same red budget, this reads that map from
existing bunny twins: each particle's source colour (the source paint at its first kept position, the source sample)
against the target paint at its last kept position, as Cohen's kappa (0 = what an independent placement gives, 1 = the
target layout exactly, < 0 = worse than chance), over all particles and over the outer layer (fewer than 0.7 x the median
neighbour count within 2 pitches: what the pictures see). Also: whether the twins of different scenes are the same run
(D160's half / band / ears / turn twins, seed 97, one code).

Target layouts in the deliverables' front view (the studio camera at azimuth 35 deg, elevation 18: its horizontal image
axis e_h = (cos 35, 0, -sin 35), s = x . e_h; height y):
  band      (reference) target red above its 65 % height quantile; source red above its median height
  half      (reference) target red above its median height; source the same
  checker   2 x 2: red where (s < median s) == (y >= median y) (top-left, bottom-right in s); source red above its median height
  checker_s the same, the split in s at the median of each half's own s (exactly half red); source as checker
  stripes3  three vertical stripes in s at the target's 1/3, 2/3 quantiles, the middle one red; source red its top third
  stripes3o the two outer stripes red (2/3); source red its bottom two thirds"""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/readers")
from fastlib import Frames  # noqa: E402

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
TWINS = {"40k_97": f"{G}/d160/runs40/bunny40k_band_P97_frames12.npz", "40k_99": f"{G}/d160/runs40/bunny40k_band_P99_frames12.npz",
         "300k_97": f"{G}/d162f/runs/band300k_T_97_frames12.npz"}
SAME = [f"{G}/d160/runs40/bunny40k_{s}_P97_frames12.npz" for s in ("band", "half", "ears", "turn")]
AZ = np.radians(35.0)
EH = np.array([np.cos(AZ), 0.0, -np.sin(AZ)])


def layouts(src, tgt):
    ys, yt, st = src[:, 1], tgt[:, 1], tgt @ EH
    my, ms = np.median(yt), np.median(st)
    top = yt >= my
    ms_top, ms_bot = np.median(st[top]), np.median(st[~top])
    q1, q2 = np.quantile(st, [1 / 3, 2 / 3])
    src_half = lambda x: x[:, 1] >= np.median(ys)                              # noqa: E731
    return {
        "band": (src_half, lambda x: x[:, 1] >= np.quantile(yt, 0.65)),
        "half": (src_half, lambda x: x[:, 1] >= my),
        "checker": (src_half, lambda x: (x @ EH < ms) == (x[:, 1] >= my)),
        "checker_s": (src_half, lambda x: np.where(x[:, 1] >= my, x @ EH < ms_top, x @ EH >= ms_bot)),
        "stripes3": (lambda x: x[:, 1] >= np.quantile(ys, 2 / 3), lambda x: (x @ EH >= q1) & (x @ EH < q2)),
        "stripes3o": (lambda x: x[:, 1] < np.quantile(ys, 2 / 3), lambda x: (x @ EH < q1) | (x @ EH >= q2)),
    }


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = float((a == b).mean())
    pa, pb = a.mean(), b.mean()
    pe = float(pa * pb + (1 - pa) * (1 - pb))
    return dict(kappa=(po - pe) / max(1 - pe, 1e-12), agree=po, chance=pe, src_red=float(pa), tgt_red_at_end=float(pb))


def main(out):
    res = dict(twins={}, same_run={})
    for lab, path in TWINS.items():
        fr = Frames(path)
        x0, x1 = np.asarray(fr[0], np.float64), np.asarray(fr[len(fr.raws) - 1], np.float64)
        tgt = np.asarray(fr.get("tgt"), np.float64)
        tree = cKDTree(x1)
        a = .708 * float(np.median(tree.query(x1[:: max(1, len(x1) // 20000)], 9)[0][:, 8]))
        cnt = np.array([len(v) for v in tree.query_ball_point(x1, 2 * a, workers=16, return_sorted=False)])
        layer = cnt < 0.7 * np.median(cnt)
        row = dict(first_raw=int(fr.raws[0]), last_raw=int(fr.raws[-1]), pitch=a, layer_share=float(layer.mean()))
        for name, (sf, tf) in layouts(x0, tgt).items():
            s, t = sf(x0), tf(x1)
            row[name] = dict(all=kappa(s, t), layer=kappa(s[layer], t[layer]), tgt_red_share=float(tf(tgt).mean()))
        res["twins"][lab] = row
        print(lab, json.dumps({k: (v["all"]["kappa"], v["layer"]["kappa"]) for k, v in row.items() if isinstance(v, dict)}), flush=True)
    ref = np.asarray(Frames(SAME[0])[len(Frames(SAME[0]).raws) - 1])
    for p in SAME[1:]:
        f = Frames(p)
        x = np.asarray(f[len(f.raws) - 1])
        res["same_run"][p.rsplit("/", 1)[-1]] = dict(same_shape=x.shape == ref.shape,
                                                    max_abs_diff=float(np.abs(x - ref).max()) if x.shape == ref.shape else None)
    print("same_run", json.dumps(res["same_run"]), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
