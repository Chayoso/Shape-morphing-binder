"""d173_probe.py OUT_JSONL SPEC_JSON -- D173 (1): can a perfect sample, its exterior field and the render's picture carry the
dragon's REAL scales? (no run; one card for the pipeline's own operators, the rest on the CPU)

For each configuration (N, the surface density F, the exterior field radius r in pitches, the picture resolution res):
  sample   a perfect N-sample of the detailed dragon (assets/dragon.obj) in the smoothed dragon's frame (the run's frame:
           3D target d170smooth_dragon.obj, seed 97); F > 1: its outer band (2 base spacings from the surface) drawn F times
           denser, the interior as it was (D122's sampler emulated here, not the pipeline's);
  layer    its outer layer (the pipeline's rule) read against the scale height H (D170's size probe: the detailed mesh's
           offset from the smoothed canvas along its normal), high-passed at 4 base pitches: corr_layer;
  field    the exterior's own operator (render/exterior.make_field "zb", radius r x pf, offset 0.8 r / 3; Lattice.discs at
           h = min(extent / res, 0.92 pf r / 3), target.py's rule; pf = the sample's pitch, the band's for F > 1): the discs'
           offsets from the canvas read the same way: corr_field; the disc count and the card's peak memory;
  picture  the loss's colour-free shading pictures (render_loss.exterior_targets) of those discs in the 18 training
           views at res, against the pictures of the true surfaces drawn the same way (D159's mesh shells of the
           detailed and the smoothed mesh, render/mesh_shell.mesh_shell_discs, c from the r = 3 discs): the pooled pixel
           correlation of (sample - smooth shell) with (detailed shell - smooth shell), both high-passed at 4 base
           pitches over the pixels all three cover: corr_pic (the gate: >= 0.25, the bunny wrinkles' layer reading 0.27);
           rms_truth = the scales' own contrast in the true pictures at that res (the picture lever on the target side).
SPEC_JSON: {"configs": [{"N": 300000, "F": 1, "r": 3, "res": 188}, ...]}."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d161")
import d157_lib as D7  # noqa: E402
from d161_lib import Surface, target_frame  # noqa: E402
from physmorph import gpu  # noqa: E402
from physmorph.pipeline.render_loss import exterior_targets, make_views  # noqa: E402
from physmorph.render.exterior import Lattice, make_field  # noqa: E402
from physmorph.render.mesh_shell import mesh_shell_discs  # noqa: E402
from physmorph.sampling.mesh import fill_volume_in_frame, mesh_in_frame, sample_in_frame  # noqa: E402

B = "/data/relcfd/chayo/physmorph_v2"
SMOOTH = f"{B}/output/gpu/d170/assets/d170smooth_dragon.obj"
DETAIL = f"{B}/repo_d172/assets/dragon.obj"
NT = D7.NT
out_path, spec = sys.argv[1], json.load(open(sys.argv[2]))
VIEWS = make_views(6, (0.0, 0.5, -0.5))


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
    return np.linalg.norm(X - X[nb[:, 1:]].mean(1), axis=1) >= 0.5 * sp


class Ctx:
    """The frame, the canvas S and the scale height H at its points (as d170_detail.py)."""

    def __init__(self, N):
        self.frame, self.x_t = target_frame("assets/isosphere.obj", SMOOTH, N, 97)
        self.a = a = .708 * D7.median_kth_spacing(self.x_t.astype(np.float64), 8, 20000)
        self.S = Surface(SMOOTH, self.frame, 1_000_000, self.x_t.astype(np.float64))
        V, F = mesh_in_frame(DETAIL, self.frame)
        t, _ = D7.close_holes(V, F)
        if t.volume < 0:
            t.invert()
        det = D7.Surf(np.asarray(t.vertices, np.float64), np.asarray(t.faces, np.int64))
        P, Nn = self.S.P, self.S.N
        H, best = np.full(len(P), np.nan), np.full(len(P), np.inf)
        for sgn in (1.0, -1.0):
            hit, _, _, tt = det.cast(np.concatenate([P, sgn * Nn], 1))
            tt = np.where(hit, tt, np.inf)
            better = tt < best
            H[better] = sgn * tt[better]
            best = np.minimum(best, tt)
        H[np.abs(H) > 3 * a] = np.nan
        self.H, self.region, self.s_bg = H, np.isfinite(H), 4.0 * a
        self.meshes = {"detail": mesh_in_frame(DETAIL, self.frame), "smooth": mesh_in_frame(SMOOTH, self.frame)}

    def read(self, Q):
        a = self.a
        i, d = self.S.foot(Q)
        h = -d
        sel = self.region[i] & (np.abs(h) < 3 * a)
        if sel.sum() < 50:
            return dict(A=None, corr=None, n=int(sel.sum()))
        rng = np.random.default_rng(1)
        Qs, hs, ps = Q[sel], h[sel], self.H[i[sel]]
        ht, pt = hs - thinned_mean(Qs, hs, self.s_bg, rng), ps - thinned_mean(Qs, ps, self.s_bg, rng)
        return dict(A=float(ht @ pt / max(pt @ pt, 1e-30)), corr=float(np.corrcoef(ht, pt)[0, 1]), n=int(sel.sum()))


def sample(ctx, N, F, seed):
    f_t = fill_volume_in_frame(SMOOTH, ctx.frame, N)
    n_d = int(round(N * fill_volume_in_frame(DETAIL, ctx.frame, N) / f_t))
    X = np.asarray(sample_in_frame(DETAIL, ctx.frame, n_d, seed)[0], np.float64)
    if F <= 1:
        return X, ctx.a
    # the outer band (2 base spacings below the canvas) F times denser: a denser draw's band points + the base interior
    band = 2.0 * ctx.a / 0.708                      # two layer spacings (the 8th-neighbour distance), layer_h_sp = 2
    Xd = np.asarray(sample_in_frame(DETAIL, ctx.frame, int(F * n_d), seed + 10)[0], np.float64)
    _, dd = ctx.S.foot(Xd)
    _, d0 = ctx.S.foot(X)
    keep = np.concatenate([X[d0 > band], Xd[dd <= band]])
    return keep, ctx.a / F ** (1 / 3)


def pictures(p, n, res, extent):
    sils, shade = exterior_targets(p, torch.nn.functional.normalize(n, dim=1), VIEWS, res, extent, 1.5, 0.25)
    return [s.detach().cpu().numpy() for s in sils], [s.detach().cpu().numpy() for s in shade]


def hp(img, mask, s):
    lo = gaussian_filter(img * mask, s) / np.maximum(gaussian_filter(mask.astype(np.float64), s), 1e-6)
    return (img - lo) * mask


def main():
    res_out = open(out_path, "a")
    ctxs = {}
    for cf in spec["configs"]:
        N, F, r, res = int(cf["N"]), float(cf.get("F", 1)), float(cf["r"]), int(cf["res"])
        t0 = time.time()
        if N not in ctxs:
            ctxs = {N: Ctx(N)}
        ctx = ctxs[N]
        X, pf = sample(ctx, N, F, 98)
        row = dict(N=N, F=F, r=r, res=res, pitch=ctx.a, pitch_field=pf, n_particles=len(X))
        lay = layer(X)
        row["layer"] = ctx.read(X[lay])
        Xt = gpu.tensor(X.astype(np.float32))
        c = Xt.mean(0)
        extent = float(Xt.abs().max()) * 1.25
        lat = Lattice(c, 2.8 * float((Xt - c).norm(dim=1).max()))
        h = min(extent / res, 0.92 * pf * r / 3.0)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        m0 = torch.cuda.memory_allocated()
        try:
            with torch.no_grad():
                fld = make_field("zb", Xt, pf, radius=r, offset=0.8 * r / 3.0)
                p, g, ncell, _ = lat.discs(fld, h, refine=False)
            torch.cuda.synchronize()
            row.update(discs=int(len(p)), lattice_pitch=h / ctx.a, mem_peak_gb=(torch.cuda.max_memory_allocated() - m0) / 1e9)
            P = p.detach().cpu().numpy().astype(np.float64)
            row["field"] = ctx.read(P)
            _, dfoot = ctx.S.foot(P)
            row["discs_deeper_than_3a"] = float((dfoot > 3 * ctx.a).mean())         # interior zero-set pockets
            # the true pictures (mesh shells at this lattice; c from the r = 3 discs of this sample) and the sample's
            with torch.no_grad():
                f3 = make_field("zb", Xt, pf, radius=3.0, offset=0.8)
                p3, _, _, _ = lat.discs(f3, min(extent / res, 0.92 * pf), refine=False)
            sh = {}
            for k, (V, Fm) in ctx.meshes.items():
                q, nq, cc, info = mesh_shell_discs(lat, h, V, Fm, pf, [p3], 3.0 * pf)
                sh[k] = pictures(q, nq, res, extent)
            sb = pictures(p, g, res, extent)
            spx = 4.0 * ctx.a / (2.0 * extent / res)
            num = den1 = den2 = 0.0
            rms_t = []
            for v in range(len(VIEWS)):
                m = (sb[0][v] > 0.5) & (sh["detail"][0][v] > 0.5) & (sh["smooth"][0][v] > 0.5)
                db = hp(sb[1][v] - sh["smooth"][1][v], m, spx)
                dt = hp(sh["detail"][1][v] - sh["smooth"][1][v], m, spx)
                db, dt = db[m] - db[m].mean(), dt[m] - dt[m].mean()
                num += float(db @ dt); den1 += float(db @ db); den2 += float(dt @ dt)
                rms_t.append(float(np.sqrt((dt ** 2).mean())))
            row["picture"] = dict(corr=num / max(np.sqrt(den1 * den2), 1e-30), rms_truth=float(np.mean(rms_t)),
                                  pixel_pitch=2.0 * extent / res / ctx.a)
        except torch.cuda.OutOfMemoryError as e:
            row.update(oom=str(e)[:200], mem_peak_gb=torch.cuda.max_memory_allocated() / 1e9)
        except RuntimeError as e:
            row.update(error=str(e)[:300], mem_peak_gb=torch.cuda.max_memory_allocated() / 1e9)
        row["secs"] = time.time() - t0
        print(json.dumps(row), flush=True)
        res_out.write(json.dumps(row) + "\n")
        res_out.flush()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
