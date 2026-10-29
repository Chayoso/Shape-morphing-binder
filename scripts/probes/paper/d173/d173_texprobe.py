"""d173_texprobe.py OUT_JSONL -- D173 (3): the textured duck's representability ceiling (no run; one card for the exterior's
operators). For N in 40k / 300k: the run's frame (the duck, assets/bob.obj, seed 97), the texture (appearance.TextureMap:
Keenan Crane's bob_diffuse.png on the bob.obj triangles), the loss's target pictures (the target sample's exterior discs,
Zhu-Bridson radius 3, the lattice of target.py, each disc's colour by render/exterior.found_disc_albedo over the sample's
looked-up colours, render_loss.exterior_targets at the run's resolution: 96 px at 40k, 188 at 300k) against the same
pictures of an independent perfect sample painted the same way: PSNR (covered pixels) and SSIM (whole image, RGB) in the
46 held-out views (D157) and the 18 training views. The same with each disc coloured by its nearest particle (the
identity, no kernel). The texture's feature size: the half-correlation distance of its luminance on the surface, pitches."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path.cwd()))
from physmorph import gpu  # noqa: E402,F811  (first: CuPy binds its CUDA 12 compiler before torch loads)
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d161")
import d157_lib as D7  # noqa: E402
from d161_lib import target_frame  # noqa: E402
from physmorph import gpu  # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import torch  # noqa: E402
from physmorph.pipeline.appearance import TextureMap  # noqa: E402
from physmorph.pipeline.render_loss import exterior_targets, make_views  # noqa: E402
from physmorph.render.exterior import Lattice, found_disc_albedo, make_field  # noqa: E402
from physmorph.sampling.mesh import draws_in_frame  # noqa: E402

A = "/data/relcfd/chayo/physmorph_v2/output/gpu/d173/assets"
TGT = "assets/bob.obj"
TRAIN = make_views(6, (0.0, 0.5, -0.5))


def pictures(X, col, a, res, extent, lat, views, nn=False):
    Xt = gpu.tensor(X.astype(np.float32))
    ct = gpu.tensor(col.astype(np.float32))
    with torch.no_grad():
        f = make_field("zb", Xt, a, radius=3.0, offset=0.8)
        p, g, _, _ = lat.discs(f, min(extent / res, 0.92 * a), refine=False)
        if nn:
            c = ct[gpu.KNN(Xt).query(p, 1)[1][:, 0]]
        else:
            c = found_disc_albedo(f, p, ct, ct.mean(0), a)
        s, sh = exterior_targets(p, torch.nn.functional.normalize(g, dim=1), views, res, extent, 1.5, 0.25, albedo=c)
    return [x.cpu().numpy() for x in s], [x.cpu().numpy() for x in sh]


def compare(A_, B_):
    ps, ss = [], []
    for (sa, ia), (sb, ib) in zip(zip(*A_), zip(*B_)):
        m = (sa > 0.5) & (sb > 0.5)
        if m.sum() < 50:
            continue
        mse = float(((ia - ib) ** 2).mean(0)[m].mean())
        ps.append(10 * np.log10(1.0 / max(mse, 1e-12)))
        ss.append(float(structural_similarity(np.moveaxis(ia, 0, -1), np.moveaxis(ib, 0, -1), channel_axis=-1, data_range=1.0)))
    return dict(psnr=float(np.mean(ps)), ssim=float(np.mean(ss)), views=len(ps))


def main():
    out = open(sys.argv[1], "a")
    for N in (40000, 300000):
        frame, x_t = target_frame("assets/isosphere.obj", TGT, N, 97)
        x_t = x_t.astype(np.float64)
        a = .708 * D7.median_kth_spacing(x_t, 8, 20000)
        tex = TextureMap(f"{A}/d173bob_tex.obj", f"{A}/d173bob_diffuse.png", frame, TGT)
        res = int(round(96 * (N / 40000) ** (1 / 3)))
        extent = float(np.abs(x_t).max()) * 1.25
        c = gpu.tensor(x_t.mean(0).astype(np.float32))
        lat = Lattice(c, 2.8 * float(np.linalg.norm(x_t - x_t.mean(0), axis=1).max()))
        X2 = np.asarray(draws_in_frame(TGT, frame, N, [99])[0], np.float64)
        row = dict(N=N, pitch=a, res=res, pixel_pitch=2 * extent / res / a)
        # the texture's feature size on the surface (luminance autocorrelation)
        rng = np.random.default_rng(0)
        sub = rng.choice(len(tex.P), 200000, replace=False)
        P, L = tex.P[sub], (tex.C[sub] @ np.array([0.2126, 0.7152, 0.0722])).astype(np.float64)
        Lc = L - L.mean()
        tree = cKDTree(P)
        bins = np.arange(0, 12.25, 0.25)
        num, cnt = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
        for q in rng.choice(len(P), 3000, replace=False):
            nb = tree.query_ball_point(P[q], 12 * a)
            d = np.linalg.norm(P[nb] - P[q], axis=1) / a
            b = np.clip(np.digitize(d, bins) - 1, 0, len(bins) - 2)
            np.add.at(num, b, Lc[q] * Lc[nb])
            np.add.at(cnt, b, 1)
        ac = num / np.maximum(cnt, 1) / max(float((Lc ** 2).mean()), 1e-30)
        row["texture_half_corr_pitch"] = next((float(0.5 * (bins[k] + bins[k + 1])) for k in range(1, len(ac)) if ac[k] < 0.5), None)
        for nm, views in (("held", D7.HELD), ("train", TRAIN)):
            for nn in (False, True):
                T = pictures(x_t, tex.at(x_t), a, res, extent, lat, views, nn)
                C = pictures(X2, tex.at(X2), a, res, extent, lat, views, nn)
                row[f"{nm}_{'nn' if nn else 'kernel'}"] = compare(C, T)
        # the twin's floor: the target pictures against the texture's mean colour on the same discs
        T = pictures(x_t, tex.at(x_t), a, res, extent, lat, D7.HELD)
        M = pictures(x_t, np.repeat(tex.at(x_t).mean(0, keepdims=True), len(x_t), 0), a, res, extent, lat, D7.HELD)
        row["held_flat_colour"] = compare(M, T)
        print(json.dumps(row), flush=True)
        out.write(json.dumps(row) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
