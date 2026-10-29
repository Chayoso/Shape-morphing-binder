"""d172_detail.py (D172: d170_detail.py + the pattern region of a built relief) -- was d170_detail.py OUT_JSON SCENE_JSON -- D170: how much of a detailed mesh's own relief (the dragon's scales and ridges, the
bunny's skin wrinkles) a state carries in its particles (CPU: numpy, scipy, Embree through d157_lib; no card).

D161's relief reading (d161_read.relief_read, the same layer rule, foot, high-pass and amplitude) with the letters'
pattern P replaced by the detail height H: at each point of the smooth canvas's surface S (d161_lib.Surface; the canvas is
the mesh after Taubin's smoothing, d161_canvas.py's operator), the signed offset of the detailed mesh along S's outward
normal (Embree, the nearer of the outward and the inward hit; > 0 outside). A state's outer-layer particles give h, their
offset from S at their foot; h and H are high-passed by the same Gaussian (sigma s_bg pitches, over the points read);
A = <h~, H~> / <H~, H~> (1 = the detail at full height), corr = Pearson(h~, H~). share = A / A_ceiling, the ceiling the
detailed mesh's own sample at N (seed + 1) read the same way (what a perfect placement of N particles carries); the floor
the smooth canvas's own sample (seed + 2).

SCENE_JSON: dict(scene, N, tgt (the run's 3D target mesh: the frame), smooth (the canvas), detail (the relief mesh),
s_bg (pitches, default 4), runs {LABEL: PREFIX} (LABEL ends in _SEED), stops {LABEL: raw frame} optional).
Also written: the detail's size against the pitch (rms H in pitches, the distance at which H's autocorrelation on S falls
to 1/2, the share of H's variance left after a Gaussian low-pass of one pitch), at this N."""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d161")
import d157_lib as D7  # noqa: E402
from d161_lib import Surface, target_frame  # noqa: E402

NT = D7.NT
out_path, scene = sys.argv[1], json.load(open(sys.argv[2]))
N = int(scene["N"])
SBG = float(scene.get("s_bg", 4.0))


def thinned_mean(Q, v, s, rng, K=64):
    d0 = cKDTree(Q).query(Q[rng.choice(len(Q), min(2000, len(Q)), replace=False)], 2, workers=NT)[0][:, 1]
    area_per = np.pi * np.median(d0) ** 2 / 0.9
    want = int(K * area_per / (np.pi * (2 * s) ** 2) * len(Q))
    idx = np.arange(len(Q)) if want >= len(Q) else rng.choice(len(Q), max(want, K + 1), replace=False)
    d, i = cKDTree(Q[idx]).query(Q, min(K, len(idx)), workers=NT)
    w = np.exp(-0.5 * (d / s) ** 2) * (d < 2 * s)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-30)
    return (w * v[idx][i]).sum(1)


def layer(X):
    sub = X[np.random.default_rng(0).choice(len(X), min(20000, len(X)), replace=False)]
    sp = float(np.median(cKDTree(sub).query(sub, 9, workers=NT)[0][:, 8])) * (len(sub) / len(X)) ** (1 / 3)
    _, nb = cKDTree(X).query(X, 33, workers=NT)
    off = X - X[nb[:, 1:]].mean(1)
    return np.linalg.norm(off, axis=1) >= 0.5 * sp


def put(path, frame):
    from physmorph.sampling.mesh import _oriented
    m = _oriented(path)
    V = (np.asarray(m.vertices, np.float64) - frame["offset"]) * frame["scale"]
    F = np.asarray(m.faces, np.int64)
    t, _ = D7.close_holes(V, F)
    if t.volume < 0:
        t.invert()
    return D7.Surf(np.asarray(t.vertices, np.float64), np.asarray(t.faces, np.int64))


class Ctx:
    def __init__(self, seed):
        t0 = time.time()
        self.frame, self.x_t = target_frame("assets/isosphere.obj", scene["tgt"], N, seed)
        self.a = a = .708 * D7.median_kth_spacing(self.x_t.astype(np.float64), 8, 20000)
        self.S = Surface(scene["smooth"], self.frame, 1_000_000 if N > 100000 else 400_000, self.x_t.astype(np.float64))
        det = put(scene["detail"], self.frame)
        P, Nn = self.S.P, self.S.N
        H = np.full(len(P), np.nan)
        best = np.full(len(P), np.inf)
        for sgn in (1.0, -1.0):
            rays = np.concatenate([P, sgn * Nn], 1)
            hit, _, _, tt = det.cast(rays)
            tt = np.where(hit, tt, np.inf)
            better = tt < best
            H[better] = sgn * tt[better]
            best = np.minimum(best, tt)
        H[np.abs(H) > 3 * a] = np.nan
        self.H = H
        self.region = np.isfinite(H)
        self.s_bg = SBG * a
        pp = scene["detail"] + ".pattern.npz"
        if scene.get("pattern_region", True) and Path(pp).exists():
            # D172: a built relief (d172_mesh.py) is read where its pattern is, as D161's reader reads its letters: the
            # canvas points within 2 s_bg of a point with P > 0.05
            from d161_pattern import Pattern
            Pt = Pattern.load(pp).at(self.S.P, self.S.N, zpts=self.S.P)
            hot = self.S.P[Pt > 0.05]
            self.region &= cKDTree(hot).query(self.S.P, 1, workers=NT)[0] < 2 * self.s_bg
            self.P = Pt
        rng = np.random.default_rng(seed)
        Hr = np.where(self.region, H, 0.0)
        sub = rng.choice(np.flatnonzero(self.region), min(400000, int(self.region.sum())), replace=False)
        low = thinned_mean(P[sub], Hr[sub], a, rng)
        hp = Hr[sub] - thinned_mean(P[sub], Hr[sub], self.s_bg, rng)
        # the autocorrelation of H on S by distance (pitches), from 4000 points
        q = rng.choice(sub, 4000, replace=False)
        Ps = P[sub]
        tree = cKDTree(Ps)
        bins = np.arange(0.0, 12.25, 0.25)
        num, cnt = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
        Hc = Hr[sub] - Hr[sub].mean()
        var = float((Hc ** 2).mean())
        for qi in q:
            nb = tree.query_ball_point(P[qi], 12 * a)
            d = np.linalg.norm(Ps[nb] - P[qi], axis=1) / a
            b = np.clip(np.digitize(d, bins) - 1, 0, len(bins) - 2)
            np.add.at(num, b, (Hr[qi] - Hr[sub].mean()) * Hc[nb])
            np.add.at(cnt, b, 1)
        ac = num / np.maximum(cnt, 1) / max(var, 1e-30)
        half = next((0.5 * (bins[k] + bins[k + 1]) for k in range(1, len(ac)) if ac[k] < 0.5), None)
        self.size = dict(pitch=a, rms_H_pitch=float(np.sqrt(np.nanmean(H ** 2)) / a),
                         rms_Hhp_pitch=float(np.sqrt((hp ** 2).mean()) / a), region_share=float(self.region.mean()),
                         autocorr_half_pitch=half, var_share_above_1pitch=float(low.var() / max(Hr[sub].var(), 1e-30)),
                         autocorr=[(float(0.5 * (bins[k] + bins[k + 1])), float(ac[k])) for k in range(len(ac))])
        self.secs = time.time() - t0


def relief_read(ctx, Q):
    a = ctx.a
    i, d = ctx.S.foot(Q)
    h = -d
    sel = ctx.region[i] & (np.abs(h) < 3 * a)
    if sel.sum() < 50:
        return dict(A=None, corr=None, n=int(sel.sum()))
    Qs, hs, ps = Q[sel], h[sel], ctx.H[i[sel]]
    rng = np.random.default_rng(1)
    ht = hs - thinned_mean(Qs, hs, ctx.s_bg, rng)
    pt = ps - thinned_mean(Qs, ps, ctx.s_bg, rng)
    A = float(ht @ pt / max(pt @ pt, 1e-30))
    c = float(((ht - ht.mean()) @ (pt - pt.mean())) / max(np.linalg.norm(ht - ht.mean()) * np.linalg.norm(pt - pt.mean()), 1e-30))
    return dict(A=A, corr=c, n=int(sel.sum()), rms_h=float(np.sqrt((ht ** 2).mean()) / a), rms_H=float(np.sqrt((pt ** 2).mean()) / a))


def main():
    from physmorph.sampling.mesh import draws_in_frame, fill_volume_in_frame, sample_in_frame
    res = dict(scene=scene, ctx={}, refs={}, runs={})
    seeds = sorted({int(lab.rsplit("_", 1)[-1]) for lab in scene["runs"]}) or list(scene.get("seeds", [97]))
    ctxs = {}
    for seed in seeds:
        c = ctxs[seed] = Ctx(seed)
        res["ctx"][seed] = dict(c.size, s_bg_pitch=SBG, secs=c.secs)
        print("ctx", seed, json.dumps(res["ctx"][seed])[:400], flush=True)
        f_t = fill_volume_in_frame(scene["tgt"], c.frame, N)
        n_d = int(round(N * fill_volume_in_frame(scene["detail"], c.frame, N) / f_t))
        n_s = int(round(N * fill_volume_in_frame(scene["smooth"], c.frame, N) / f_t))
        refs = {"tgt": c.x_t, "detail_s1": sample_in_frame(scene["detail"], c.frame, n_d, seed + 1)[0],
                "detail_s2": draws_in_frame(scene["detail"], c.frame, n_d, [seed + 2])[0],
                "smooth_s2": draws_in_frame(scene["smooth"], c.frame, n_s, [seed + 2])[0]}
        for k, X in refs.items():
            X = np.asarray(X, np.float64)
            res["refs"][f"{k}{seed}"] = dict(relief=relief_read(c, X[layer(X)]))
            print("ref", k, seed, json.dumps(res["refs"][f"{k}{seed}"]), flush=True)
    stops = scene.get("stops", {})
    for label, prefix in scene["runs"].items():
        seed = int(label.rsplit("_", 1)[-1])
        c = ctxs[seed]
        z = np.load(prefix + "_frames12.npz", allow_pickle=True)
        assert np.abs(np.asarray(z["tgt"], np.float64) - c.x_t).max() < 1e-5, f"{label}: the run's target is not the frame's"
        F, raws = np.asarray(z["frames"], np.float64), np.asarray(z["raws"])
        n = len(F)
        step = 3
        js = set(range(0, n, step)) | {n - 1}
        if label in stops:
            k = int(np.searchsorted(raws, stops[label]))
            js |= {max(0, k - 1), min(n - 1, k)}
        series = []
        for j in sorted(js):
            series.append(dict(raw=int(raws[j]), t=float(raws[j]) / 240.0, relief=relief_read(c, F[j][layer(F[j])])))
        res["runs"][label] = dict(series=series)
        print("run", label, json.dumps(series[-1]), flush=True)
    json.dump(res, open(out_path, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
