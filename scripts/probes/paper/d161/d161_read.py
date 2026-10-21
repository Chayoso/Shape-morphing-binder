"""d161_read.py OUT_JSON SCENE_JSON -- D161's reader (CPU: numpy, scipy, open3d's Embree through d157_lib; no card).

SCENE_JSON: dict(scene, N, smooth (the 3D target's mesh), relief (the render target's mesh; its .pattern.npz and .json
beside it), runs: {LABEL: PREFIX} with LABEL in full97 twin97 full99 twin99 (PREFIX_frames12.npz, PREFIX.json)).

Context: the frame of the smooth mesh at N (seed 97 and 99: d161_lib.target_frame; the run's own target sample must match
it to 1e-6 wu), a = 0.708 x the target sample's median 8th-neighbour distance; the smooth mesh's surface S (d161_lib.
Surface, 400k points at 40k, 1M above) with the pattern P (d161_pattern) on it; the region: S's points within 2 s_bg of
a point with P > 0.05, s_bg = height / 2 (the letters' em in pitches; the logo's scale).

Readings of a state X (particles):
  relief (the run's own particles): the outer layer by the pipeline's rule (offset from the mean of the 32 nearest at
    least 0.5 x the spacing, the median 8th-neighbour distance), each layer particle's outward offset h from the smooth
    surface at its foot, over the layer particles whose foot is in the region and |h| < 3 a; h and P high-passed by
    the same Gaussian (sigma s_bg, over those points) so that the body's own shape error leaves; A = <h~, P~> / <P~, P~>
    (the amplitude carved, pitches, for P in [0, 1]: the relief mesh has amp), corr = Pearson(h~, P~).
  display: the same on the vertices of the display surface (Zhu and Bridson with Solenthaler's correction, D157's
    field_mesh "zbs"), at the end (and at every kept frame with --frames_display).
  images (end): D157's studio look (three GGX lights fixed to the camera) on the 46 held-out views at 384 px, the
    display surface against the relief mesh (crease-aware normals) and against the smooth mesh: the mean squared error
    over the pixels the relief mesh covers where its hit point has P > 0.05 (the pattern's pixels; views with fewer
    than 200 skipped) and over all covered pixels.
  geometry (end, against the smooth mesh): orthographic silhouettes at 512 px over the 18 training and 46 held-out
    views, 1 - IoU; the display surface's area sample (100k points of its visible faces) against the smooth mesh:
    Chamfer-L1 and F at 0.5 / 1 pitch (D157's d3_metrics); the same against the relief mesh.
  pieces (end): particles outside the largest component at radius 2 x the median nearest distance, the largest piece.
  rest (end): the last committed window's records (v_mean, kin, stab_end), windows, wall seconds, lambda, g_share.
Pairs per kept frame (aligned by the simulated frame index): full_s vs twin_s at one seed (the same source particles):
  each twin layer particle's distance to its own full-arm counterpart, in pitches (median, p90; region / elsewhere);
  the surface distance between the two layers (the mean of the two directed nearest-neighbour means); the same surface
  distance twin97 vs twin99 (different samples): the seed spread. References: the run's own smooth target sample, the
  relief mesh's sample at N (sample_in_frame, seed + 1) and an independent one (seed + 2): the reading's floor and
  ceiling."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import d157_lib as D7  # noqa: E402
from d161_lib import Surface, target_frame  # noqa: E402
from d161_pattern import Pattern  # noqa: E402

NT = D7.NT
FRAMES_DISPLAY = "--frames_display" in sys.argv
argv = [a for a in sys.argv[1:] if not a.startswith("--")]
out_path, scene = argv[0], json.load(open(argv[1]))
N = int(scene["N"])
IMG = 384


def thinned_mean(Q, v, s, rng, K=64):
    """Gaussian mean (sigma s, truncated at 2 s) of the values v at the points Q, the population thinned so that the
    K nearest reach about 2 s."""
    d0 = cKDTree(Q).query(Q[rng.choice(len(Q), min(2000, len(Q)), replace=False)], 2, workers=NT)[0][:, 1]
    area_per = np.pi * np.median(d0) ** 2 / 0.9                      # area per point (rough)
    want = int(K * area_per / (np.pi * (2 * s) ** 2) * len(Q))
    idx = np.arange(len(Q)) if want >= len(Q) else rng.choice(len(Q), max(want, K + 1), replace=False)
    d, i = cKDTree(Q[idx]).query(Q, min(K, len(idx)), workers=NT)
    w = np.exp(-0.5 * (d / s) ** 2) * (d < 2 * s)
    w = w / np.maximum(w.sum(1, keepdims=True), 1e-30)
    return (w * v[idx][i]).sum(1)


class Ctx:
    def __init__(self, seed):
        t0 = time.time()
        self.seed = seed
        self.frame, self.x_t = target_frame("assets/isosphere.obj", scene["smooth"], N, seed)
        self.a = a = .708 * D7.median_kth_spacing(self.x_t.astype(np.float64), 8, 20000)
        self.S = Surface(scene["smooth"], self.frame, 1_000_000 if N > 100000 else 400_000, self.x_t.astype(np.float64))
        self.pat = Pattern.load(scene["relief"] + ".pattern.npz")
        self.spec = json.load(open(scene["relief"] + ".json"))["spec"]
        self.P = self.pat.at(self.S.P, self.S.N, zpts=self.S.P)
        self.s_bg = 0.5 * float(self.spec["height"]) * a
        hot = self.S.P[self.P > 0.05]
        self.region = cKDTree(hot).query(self.S.P, 1, workers=NT)[0] < 2 * self.s_bg
        from physmorph.sampling.mesh import _oriented
        def put(path):
            m = _oriented(path)
            V = (np.asarray(m.vertices, np.float64) - self.frame["offset"]) * self.frame["scale"]
            F = np.asarray(m.faces, np.int64)
            t, _ = D7.close_holes(V, F)
            if t.volume < 0:
                t.invert()
            V, F = np.asarray(t.vertices, np.float64), np.asarray(t.faces, np.int64)
            return D7.Surf(V, F, cn=D7.corner_normals(V, F, crease=60.0)), D7.Surf(V, F)
        self.smooth, self.smooth_f = put(scene["smooth"])
        self.relief, self.relief_f = put(scene["relief"])
        self.center = self.x_t.astype(np.float64).mean(0)
        self.radius = float(np.linalg.norm(self.x_t - self.center, axis=1).max())
        self.extent = float(np.abs(self.x_t).max()) * 1.25
        self.views = D7.TRAIN + D7.HELD
        self.ref_masks = [D7.MaskInfo(self.smooth_f.occluded(D7.ortho_rays(v, self.extent, 512)).reshape(512, 512))
                          for v in self.views]
        rng = np.random.default_rng(seed)
        self.ref_pts = {}
        for name, s in (("smooth", self.smooth_f), ("relief", self.relief_f)):
            fa = s.area / s.area.sum()
            fi = rng.choice(len(fa), 100000, p=fa)
            r1, r2 = rng.random((2, 100000))
            sq = np.sqrt(r1)
            V = s.V[s.F[fi]]
            p = (1 - sq)[:, None] * V[:, 0] + (sq * (1 - r2))[:, None] * V[:, 1] + (sq * r2)[:, None] * V[:, 2]
            self.ref_pts[name] = (p, s.fn[fi])
        self.ref_img = {}
        for name, s in (("smooth", self.smooth), ("relief", self.relief)):
            self.ref_img[name] = [D7.render(s, self.center, self.radius, v, IMG) for v in D7.HELD]
        # the pattern's pixels in each held-out view: the relief mesh's hit points with P > 0.05
        self.pmask = []
        for v in D7.HELD:
            cam, dirs, *_ = D7.studio_cam(self.center, self.radius, v, IMG)
            rays = np.concatenate([np.broadcast_to(cam, dirs.shape).reshape(-1, 3), dirs.reshape(-1, 3)], 1)
            hit, pid, uv, tt = self.relief.cast(rays)
            m = np.zeros(len(rays), bool)
            if hit.any():
                hp = rays[hit, :3] + tt[hit, None] * rays[hit, 3:]
                fi, d = self.S.foot(hp)
                m[np.flatnonzero(hit)] = self.P[fi] > 0.05
            self.pmask.append(m.reshape(IMG, IMG))
        self.secs = time.time() - t0


def layer(X, a_sp=None):
    """The pipeline's outer layer (window/layer.layer_by_asymmetry) on the CPU."""
    sub = X[np.random.default_rng(0).choice(len(X), min(20000, len(X)), replace=False)]
    sp = float(np.median(cKDTree(sub).query(sub, 9, workers=NT)[0][:, 8])) * (len(sub) / len(X)) ** (1 / 3)
    _, nb = cKDTree(X).query(X, 33, workers=NT)
    off = X - X[nb[:, 1:]].mean(1)
    return np.linalg.norm(off, axis=1) >= 0.5 * sp


def relief_read(ctx, Q):
    a = ctx.a
    i, d = ctx.S.foot(Q)
    h = -d
    sel = ctx.region[i] & (np.abs(h) < 3 * a)
    if sel.sum() < 50:
        return dict(A=None, corr=None, n=int(sel.sum()))
    Qs, hs, ps = Q[sel], h[sel], ctx.P[i[sel]]
    rng = np.random.default_rng(1)
    ht = hs - thinned_mean(Qs, hs, ctx.s_bg, rng)
    pt = ps - thinned_mean(Qs, ps, ctx.s_bg, rng)
    A = float(ht @ pt / max(pt @ pt, 1e-30))
    c = float(((ht - ht.mean()) @ (pt - pt.mean())) / max(np.linalg.norm(ht - ht.mean()) * np.linalg.norm(pt - pt.mean()), 1e-30))
    return dict(A=A / a, corr=c, n=int(sel.sum()), rms_h=float(np.sqrt((ht ** 2).mean()) / a),
                rms_P=float(np.sqrt((pt ** 2).mean())))


def display_surf(ctx, X):
    surf, _ = D7.field_mesh("zbs", np.asarray(X, np.float64), ctx.a)
    return surf


def images(ctx, surf):
    out = {}
    for ref in ("relief", "smooth"):
        pat, cov = [], []
        for k, v in enumerate(D7.HELD):
            s = D7.render(surf, ctx.center, ctx.radius, v, IMG)
            t = ctx.ref_img[ref][k]
            m = ctx.pmask[k]
            e = ((s[0] - t[0]) ** 2).mean(-1)
            if m.sum() >= 200:
                pat.append(float(e[m].mean()))
            both = (s[2] > 0.5) | (t[2] > 0.5)
            cov.append(float(e[both].mean()))
        out[ref] = dict(mse_pattern=float(np.mean(pat)) if pat else None, n_views_pattern=len(pat),
                        mse_all=float(np.mean(cov)))
    return out


def geometry(ctx, surf):
    errs = []
    for k, v in enumerate(ctx.views):
        m = surf.occluded(D7.ortho_rays(v, ctx.extent, 512)).reshape(512, 512)
        T = ctx.ref_masks[k].m
        errs.append(1.0 - (m & T).sum() / max((m | T).sum(), 1))
    rng = np.random.default_rng(5)
    fa = surf.area / surf.area.sum()
    fi = rng.choice(len(fa), 100000, p=fa)
    r1, r2 = rng.random((2, 100000))
    sq = np.sqrt(r1)
    V = surf.V[surf.F[fi]]
    sP = (1 - sq)[:, None] * V[:, 0] + (sq * (1 - r2))[:, None] * V[:, 1] + (sq * r2)[:, None] * V[:, 2]
    sN = surf.fn[fi]
    out = dict(sil_iou_err_train=float(np.mean(errs[:D7.NTR])), sil_iou_err_held=float(np.mean(errs[D7.NTR:])))
    for name, s in (("smooth", ctx.smooth_f), ("relief", ctx.relief_f)):
        rP, rN = ctx.ref_pts[name]
        d3, _ = D7.d3_metrics(surf, sP, sN, s, rP, rN, ctx.a)
        out[name] = {k: d3[k] for k in ("f0.5", "f1", "chamfer_l1_a", "nc", "signed_mean_a")}
    return out


def pieces(X):
    d1 = cKDTree(X).query(X, 2, workers=NT)[0][:, 1]
    r = 2.0 * float(np.median(d1))
    pr = cKDTree(X).query_pairs(r, output_type="ndarray")
    g = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(X), len(X)))
    nc, lab = connected_components(g, directed=False)
    cnt = np.bincount(lab)
    big = np.argmax(cnt)
    rest = np.sort(cnt[np.arange(nc) != big])[::-1]
    return dict(outside=int(len(X) - cnt[big]), largest_piece=int(rest[0]) if len(rest) else 0,
                pieces_ge10=int((rest >= 10).sum()))


def records(prefix):
    js = json.load(open(prefix + ".json"))
    arm = list(js["arms"].values())[0]
    hist = [h for h in arm["history"] if h.get("frame_end") and not h.get("null_commit")]
    last = hist[-1]
    return dict(windows=len(hist), wall_s=arm.get("seconds", js.get("seconds")),
                v_mean=last.get("v_mean"), kin=last.get("kin"), stab_end=last.get("stab_end"),
                lam_last=last.get("lambda"), g_share_last=last.get("g_share"),
                lam_med10=float(np.median([h.get("lambda") or 0.0 for h in hist[-10:]])),
                render_tgt=js["provenance"].get("render_tgt"), rws=js["provenance"].get("render_weight_scale"))


def surf_dist(A, B):
    return 0.5 * (cKDTree(B).query(A, 1, workers=NT)[0].mean() + cKDTree(A).query(B, 1, workers=NT)[0].mean())


def main():
    res = dict(scene=scene, ctx={}, refs={}, runs={}, pairs={})
    ctxs = {}
    for seed in (97, 99):
        c = Ctx(seed)
        ctxs[seed] = c
        res["ctx"][seed] = dict(pitch=c.a, s_bg_pitch=c.s_bg / c.a, region_share=float(c.region.mean()),
                                P_mean_region=float(c.P[c.region].mean()), secs=c.secs)
        # references: the smooth target sample, the relief mesh's own samples (the ceiling), an independent smooth one
        from physmorph.sampling.mesh import draws_in_frame, fill_volume_in_frame, sample_in_frame
        f_t = fill_volume_in_frame(scene["smooth"], c.frame, N)
        n_r = int(round(N * fill_volume_in_frame(scene["relief"], c.frame, N) / f_t))
        refs = {"smooth_tgt": c.x_t, "smooth_ind": draws_in_frame(scene["smooth"], c.frame, N, [seed + 2])[0],
                "relief_s1": sample_in_frame(scene["relief"], c.frame, n_r, seed + 1)[0],
                "relief_s2": draws_in_frame(scene["relief"], c.frame, n_r, [seed + 2])[0]}
        for k, X in refs.items():
            X = np.asarray(X, np.float64)
            lay = layer(X)
            row = dict(relief=relief_read(c, X[lay]))
            if k in ("smooth_tgt", "relief_s1"):
                surf = display_surf(c, X)
                row["display"] = relief_read(c, surf.V)
            res["refs"][f"{k}{seed}"] = row
            print(k, seed, json.dumps(row), flush=True)
    frames = {}
    for label, prefix in scene["runs"].items():
        seed = int(label[-2:])
        c = ctxs[seed]
        z = np.load(prefix + "_frames12.npz", allow_pickle=True)
        assert np.abs(np.asarray(z["tgt"], np.float64) - c.x_t).max() < 1e-5, f"{label}: the run's target is not the frame's"
        F, raws = np.asarray(z["frames"], np.float64), np.asarray(z["raws"])
        frames[label] = (F, raws)
        series = []
        for j in range(len(F)):
            lay = layer(F[j])
            row = dict(raw=int(raws[j]), t=float(raws[j]) / 240.0, relief=relief_read(c, F[j][lay]))
            if FRAMES_DISPLAY:
                row["display"] = relief_read(c, display_surf(c, F[j]).V)
            series.append(row)
        X = F[-1]
        surf = display_surf(c, X)
        end = dict(relief=series[-1]["relief"], display=relief_read(c, surf.V), images=images(c, surf),
                   geometry=geometry(c, surf), pieces=pieces(X), records=records(prefix))
        res["runs"][label] = dict(series=series, end=end)
        print(label, json.dumps(end), flush=True)
    # pairs: full vs twin at a seed (same particles), twin97 vs twin99 (the seed spread)
    for seed in (97, 99):
        fl, tw = f"full{seed}", f"twin{seed}"
        if fl not in frames or tw not in frames:
            continue
        c = ctxs[seed]
        (Ff, rf), (Ft, rt) = frames[fl], frames[tw]
        common = sorted(set(rf.tolist()) & set(rt.tolist()))
        rows = []
        for r in common:
            xf, xt = Ff[list(rf).index(r)], Ft[list(rt).index(r)]
            lay = layer(xt)
            i, _ = c.S.foot(xt[lay])
            inr = c.region[i]
            d = np.linalg.norm(xf[lay] - xt[lay], axis=1) / c.a
            lf = layer(xf)
            rows.append(dict(raw=int(r), t=r / 240.0, pp_med=float(np.median(d)), pp_p90=float(np.quantile(d, .9)),
                             pp_med_region=float(np.median(d[inr])) if inr.any() else None,
                             pp_med_else=float(np.median(d[~inr])) if (~inr).any() else None,
                             surf=float(surf_dist(xf[lf], xt[lay]) / c.a)))
        # each run's end (whatever its length)
        xf, xt = Ff[-1], Ft[-1]
        res["pairs"][f"full_twin{seed}"] = dict(series=rows, end_surf=float(surf_dist(xf[layer(xf)], xt[layer(xt)]) / c.a))
    if "twin97" in frames and "twin99" in frames:
        (A, ra), (B, rb) = frames["twin97"], frames["twin99"]
        a = ctxs[97].a
        common = sorted(set(ra.tolist()) & set(rb.tolist()))
        rows = [dict(raw=int(r), t=r / 240.0, surf=float(surf_dist(A[list(ra).index(r)][layer(A[list(ra).index(r)])],
                                                                     B[list(rb).index(r)][layer(B[list(rb).index(r)])]) / a))
                for r in common]
        res["pairs"]["twin97_twin99"] = dict(series=rows, end_surf=float(surf_dist(A[-1][layer(A[-1])], B[-1][layer(B[-1])]) / a))
        if "full97" in frames and "full99" in frames:
            (A, ra), (B, rb) = frames["full97"], frames["full99"]
            res["pairs"]["full97_full99"] = dict(end_surf=float(surf_dist(A[-1][layer(A[-1])], B[-1][layer(B[-1])]) / a))
    json.dump(res, open(out_path, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
