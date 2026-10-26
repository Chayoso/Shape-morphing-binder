"""d160_read.py OUT_JSONL JOBS -- D160's colour readings of runs (measurement only; on a card when one is visible).

JOBS lines: `NAME SCENE FRAMES12 [twin=NAME] [timeline]`
  NAME    a run; FRAMES12 its kept-frame archive (frame 0 the source, the last the delivered end, tgt the target sample)
  SCENE   the appearance scene (physmorph/pipeline/appearance.py). The body's paint is the archive's src_albedo when it
          has one, else the scene's source paint of frame 0: the tag's runs are colour-blind, so their paint can be read
          after the fact (the layout the transport gives).
  twin=   the job NAME of the same seed's lambda = 0 twin (read earlier in the file): the material travel between ends.
Readings on a state's exterior discs (Zhu-Bridson, radius 3, offset 0.8 a, the loss's lattice h = min(extent / 96,
0.92 a) centred on the target as the loss's; each disc's red share the kernel mean of its particles' red indicator,
render/exterior.found_disc_albedo), against the scene's target paint at the disc:
  mis      the share of discs whose shown colour (red share >= 1/2) is not the paint's: the mis-coloured surface share
  recall, false   P(shown red | paint red), P(shown red | paint white)
  travel   over the mis-coloured discs, the distance to the nearest disc of the target sample whose paint is the shown
           colour, in pitches (median, p90): how far that material still is from where its colour is wanted
  floor    the mis share of the target sample's own discs, the sample painted by the scene (the definition's floor)
  img      the delivered state and the independent sample (d92/ind_MESH.npz) painted by the scene, drawn by the loss's
           own colour operator (render_loss.shaded_discs with albedo: Lambertian headlight, ambient 0.25, orthographic,
           the loss's extent) at 256 px in D157's 46 held-out views and in the loss's 18: PSNR and CIELAB dE76 over the
           union of the two coverages (alpha > 0.05), LPIPS (AlexNet) on the whole images; the same of the grey images
           (albedo 1) as the geometry-only reference
  vs_twin  per particle |x_end - x_end(twin)| in pitches, median and p90, over all and over the outer layer (the
           particles some delivered disc of the run reads); and the outer layer's red material, run minus twin: its mean
           height (half, band, ears) or its mean offset along the head direction (turn), in pitches
  timeline (`timeline`) mis at every 4th kept frame and the last, (time s, mis)."""
import json
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

from physmorph.losses.silhouette import set_kernel
from physmorph.pipeline.appearance import RED, WHITE, scene_params, source_red, target_red
from physmorph.pipeline.render_loss import make_views, shaded_discs
from physmorph.render.exterior import Lattice, found_disc_albedo, make_field

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
FPS = 240.0
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_kernel("cic")


def dvec(th, ph):
    return np.array([np.sin(th) * np.cos(ph), np.sin(ph), np.cos(th) * np.cos(ph)])


TRAIN = make_views(6, (0.0, 0.5, -0.5))


def heldout(n=64, cap=12.0):
    """D157's held-out views: a 64-point Fibonacci sphere minus 12 degree caps about the training directions."""
    i = np.arange(n) + 0.5
    pol = np.arccos(1 - 2 * i / n)
    az = np.pi * (1 + 5 ** 0.5) * i
    P = np.stack([np.cos(az) * np.sin(pol), np.cos(pol), np.sin(az) * np.sin(pol)], 1)
    tr = np.array([dvec(t, p) for t, p in TRAIN])
    keep = np.degrees(np.arccos(np.clip(np.abs(P @ tr.T), 0, 1))).min(1) > cap
    return [(float(np.arctan2(p[0], p[2])), float(np.arcsin(np.clip(p[1], -1, 1)))) for p in P[keep]]


HELD = heldout()


def pitch_of(x):
    d, _ = cKDTree(x).query(x, 9)
    return 0.708 * float(np.median(d[:, 8]))


class Ctx:
    """A target's constants (the loss's): pitch, extent, the lattice and its pitch."""

    def __init__(self, tgt):
        self.tgt = np.asarray(tgt, np.float32)
        self.a = pitch_of(self.tgt)
        self.extent = 1.25 * float(np.abs(self.tgt).max())
        t = torch.as_tensor(self.tgt, device=DEV)
        c = t.mean(0)
        self.lattice = Lattice(c, 2.8 * float((t - c).norm(dim=1).max()))
        self.h = min(self.extent / 96, 0.92 * self.a)


_GEOM = {}


def geom(x, ctx, key=None):
    """(points, unit normals, particle lists, kernel weights) of a state's exterior discs; cached by key."""
    if key is not None and key in _GEOM:
        return _GEOM[key]
    xt = torch.as_tensor(np.asarray(x, np.float32), device=DEV)
    f = make_field("zb", xt, ctx.a, device_field=DEV.type == "cuda")
    with torch.no_grad():
        p, g, _, _ = ctx.lattice.discs(f, ctx.h, refine=False)
        idx = f.neighbours(p, ctx.a)
        q = xt.index_select(0, idx.clamp_min(0).reshape(-1)).view(*idx.shape, 3)
        w = (1. - (p[:, None, :] - q).square().sum(-1) / (f.radius * f.radius)).clamp_min(0.) ** 3 * (idx >= 0)
    out = (p, torch.nn.functional.normalize(g, dim=1), idx, w)
    if key is not None:
        _GEOM[key] = out
    return out


def discs(x, ctx, red, key=None):
    """(points, unit normals, red share, particle lists) of a state's exterior discs: each disc's red share the kernel
    mean of its particles' red indicator (render/exterior.found_disc_albedo's weights)."""
    p, n, idx, w = geom(x, ctx, key)
    r = torch.as_tensor(np.asarray(red, np.float32), device=DEV)
    s = (w * r[idx.clamp_min(0)]).sum(1) / w.sum(1).clamp_min(1e-12)
    return p, n, s, idx


def colour(s, prm):
    red, white = torch.tensor(prm["red"], device=s.device), torch.tensor(prm["white"], device=s.device)
    return white + s[:, None] * (red - white)


def layout(p, s, prm, floor_p, floor_want, a):
    pn = p.detach().cpu().numpy()
    sn = s.detach().cpu().numpy()
    want = target_red(prm, pn)
    shown = sn >= 0.5
    bad = shown != want
    out = dict(discs=len(pn), mis=float(bad.mean()), shown_red=float(shown.mean()), want_red=float(want.mean()),
               recall=float(shown[want].mean()) if want.any() else None,
               false=float(shown[~want].mean()) if (~want).any() else None)
    dd = []
    for col in (True, False):
        q = pn[bad & (shown == col)]
        ref = floor_p[floor_want == col]
        if len(q) and len(ref):
            dd.append(cKDTree(ref).query(q)[0] / a)
    dd = np.concatenate(dd) if dd else np.zeros(0)
    out.update(travel_med=float(np.median(dd)) if len(dd) else 0.0,
               travel_p90=float(np.quantile(dd, 0.9)) if len(dd) else 0.0)
    return out


_LP = None


def lpips_batch(A, B):
    global _LP
    import lpips
    if _LP is None:
        _LP = lpips.LPIPS(net="alex", version="0.1", verbose=False).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(A), 8):
            x = torch.as_tensor(np.stack(A[i:i + 8])).permute(0, 3, 1, 2).float() * 2 - 1
            y = torch.as_tensor(np.stack(B[i:i + 8])).permute(0, 3, 1, 2).float() * 2 - 1
            out.append(_LP(x, y).flatten().numpy())
    return np.concatenate(out).astype(np.float64)


def images(p, n, alb, views, ctx, res=256):
    with torch.no_grad():
        s, al = shaded_discs(p, n, views, res, ctx.extent, 1.5, 0.25, albedo=alb)
    return s.permute(1, 2, 3, 0).cpu().numpy(), al.cpu().numpy()          # (V,res,res,C), (V,res,res)


def img_metrics(S, AS, T, AT):
    from skimage.color import rgb2lab
    ps, de = [], []
    for s, a_s, t, a_t in zip(S, AS, T, AT):
        m = (a_s > 0.05) | (a_t > 0.05)
        mse = float(((s - t) ** 2)[m].mean())
        ps.append(10 * np.log10(1.0 / max(mse, 1e-12)))
        if s.shape[-1] == 3:
            de.append(float(np.linalg.norm(rgb2lab(np.clip(s, 0, 1)) - rgb2lab(np.clip(t, 0, 1)), axis=-1)[m].mean()))
    rgb = lambda X: [np.repeat(x, 3, -1) if x.shape[-1] == 1 else x for x in X]  # noqa: E731
    lp = lpips_batch(rgb(S), rgb(T))
    return dict(psnr=float(np.mean(ps)), lpips=float(np.mean(lp)), de=float(np.mean(de)) if de else None)


def independent_mesh(path, options):
    """Explicit mesh metadata supports campaign filenames without mesh/count prefixes."""
    explicit = next((token[5:] for token in options if token.startswith("mesh=")), None)
    return explicit if explicit is not None else path.split("/")[-1].split("40k")[0].split("300k")[0].split("500k")[0]


def main():
    out_path, jobs = sys.argv[1], [ln.split() for ln in open(sys.argv[2]) if ln.strip() and not ln.startswith("#")]
    ctxs, ends = {}, {}
    out = open(out_path, "a")
    for job in jobs:
        name, scene, path = job[:3]
        twin = next((j[5:] for j in job[3:] if j.startswith("twin=")), None)
        z = np.load(path, allow_pickle=True)
        X, raws, tgt = z["frames"], np.asarray(z["raws"]), np.asarray(z["tgt"], np.float32)
        if int(raws[0]) != 0:
            print(json.dumps(dict(name=name, skipped="no source frame")), flush=True)
            continue
        src = np.asarray(X[0], np.float32)
        prm = scene_params(scene, src, tgt)
        if "src_albedo" in z.files:
            red = (np.abs(np.asarray(z["src_albedo"]) - np.asarray(RED, np.float32)).sum(1) < 1e-6)
            assert np.array_equal(red, source_red(prm, src)), "the archive's paint is not the scene's"
        else:
            red = source_red(prm, src)
        key = (len(tgt), float(tgt[:, 1].sum()))
        if key not in ctxs:                                     # the target's constants and its independent sample
            ctx = Ctx(tgt)
            ctx.floor = {}
            mesh = independent_mesh(path, job[3:])
            ctx.ind = np.asarray(np.load(f"{G}/d92/ind_{mesh}.npz")["tgt"], np.float32)
            ctxs[key] = ctx
        ctx = ctxs[key]
        if scene not in ctx.floor:                              # per scene: the floor and the independent images
            fw = target_red(prm, ctx.tgt)
            pf, nf, sf, _ = discs(ctx.tgt, ctx, fw, key=("tgt", key))
            pfn = pf.cpu().numpy()
            fl = layout(pf, sf, prm, pfn, target_red(prm, pfn), ctx.a)
            pi, ni, si, _ = discs(ctx.ind, ctx, target_red(prm, ctx.ind), key=("ind", key))
            imgs = {}
            for vs, views in (("held", HELD), ("train", TRAIN)):
                imgs[vs] = (images(pi, ni, colour(si, prm), views, ctx),
                            images(pi, ni, torch.ones(len(pi), 1, device=DEV), views, ctx))
            ctx.floor[scene] = (pfn, target_red(prm, pfn), fl, imgs)
        pfn, fwant, fl, imgs = ctx.floor[scene]
        xe = np.asarray(X[-1], np.float32)
        p, n, s, idx = discs(xe, ctx, red, key=("end", path))
        row = dict(name=name, scene=scene, n=len(xe), deliver_n=int(z["deliver_n"]), t_end=float(raws[-1] / FPS),
                   pitch=ctx.a, src_red=float(red.mean()), floor=fl["mis"], end=layout(p, s, prm, pfn, fwant, ctx.a))
        row["img"] = {}
        for vs, views in (("held", HELD), ("train", TRAIN)):
            (Tc, Ac), (Tg, Ag) = imgs[vs]
            Sc, ASc = images(p, n, colour(s, prm), views, ctx)
            Sg, ASg = images(p, n, torch.ones(len(p), 1, device=DEV), views, ctx)
            row["img"][vs] = dict(colour=img_metrics(Sc, ASc, Tc, Ac), grey=img_metrics(Sg, ASg, Tg, Ag))
        outer = np.unique(idx[idx >= 0].cpu().numpy())
        ends[name] = (xe, outer, red, prm)
        if twin and twin in ends:
            xt, _, _, _ = ends[twin]
            d = np.linalg.norm(xe - xt, axis=1) / ctx.a
            h = (np.array([0.0, 1.0, 0.0]) if prm["scene"] != "turn" else np.asarray(prm["head_dir"]))
            ro = outer[red[outer]]
            row["vs_twin"] = dict(twin=twin, d_med=float(np.median(d)), d_p90=float(np.quantile(d, 0.9)),
                                  d_outer_med=float(np.median(d[outer])), d_outer_p90=float(np.quantile(d[outer], 0.9)),
                                  red_outer_shift=float(((xe[ro] - xt[ro]) @ h).mean() / ctx.a) if len(ro) else None,
                                  red_outer_n=int(len(ro)))
        if "timeline" in job[3:]:
            tl = []
            for k in list(range(0, len(X) - 1, 4)) + [len(X) - 1]:
                pk, _, sk, _ = discs(np.asarray(X[k], np.float32), ctx, red)
                tl.append((float(raws[k] / FPS), layout(pk, sk, prm, pfn, fwant, ctx.a)["mis"]))
            row["timeline"] = tl
        out.write(json.dumps(row) + "\n")
        out.flush()
        e = row["end"]
        print(f"{name:24s} {scene:5s} mis {e['mis']:.4f} (floor {fl['mis']:.4f}) recall {e['recall']} false "
              f"{e['false']} travel {e['travel_med']:.1f}/{e['travel_p90']:.1f} | held colour PSNR "
              f"{row['img']['held']['colour']['psnr']:.2f} LPIPS {row['img']['held']['colour']['lpips']:.4f} dE "
              f"{row['img']['held']['colour']['de']:.2f} | grey PSNR {row['img']['held']['grey']['psnr']:.2f}"
              + (f" | vs twin {row['vs_twin']}" if "vs_twin" in row else ""), flush=True)


if __name__ == "__main__":
    main()
