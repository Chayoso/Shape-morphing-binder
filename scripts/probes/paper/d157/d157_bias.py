"""d157_bias.py OUT_JSONL SPEC [SPEC ...] -- D157 (4b): the outward bias of the silhouette term's 2 : 1 asymmetry.

SPEC = MESH:SEED:N:ARM:TWIN (N 300k | 40k; ARM / TWIN the archives' labels, L / P or V / W). Measurement only, CPU.
Per mesh-seed the run's own exterior target pictures are rebuilt as prepare.py and target.py build them: the run's
target sample and its further draws (draws_in_frame, seeds seed + 1 + 100 k, k = 1 .. draws - 1; their count from the
run's provenance), each sample's exterior discs (Zhu and Bridson's field at the run's pitch, radius 3, offset 0.8, the
lattice of the run: centre the target's mean, reach 2.8 x its radius, h = min(box / res, 0.92 pitch), refine off), their
soft silhouettes (CIC, k = 1.5) averaged over the samples. Per state (the own sample, the arm's end, the twin's end) its
exterior discs on the same lattice, moved uniformly along their unit normals by delta pitches (-0.6 .. 0.6, steps of
0.01): the deficit D(delta) = mean relu(a_t - a)^2 and the excess X(delta) = mean relu(a - a_t)^2 over the views and
pixels; the term at (w_hole, w_spray) = (2, 1) is 2D + X, at (1, 1) D + X. argmin by a parabola through the grid minimum
and its neighbours. At the run's resolution, and (bunny, dragon) at half and double it. The target pictures' edge width:
pixels with 0.1 < a_t < 0.9 over the length of the a_t = 0.5 outline, in pitches, for the mean over the draws and for
the run's own sample alone."""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as nnf
from scipy.spatial import cKDTree
from skimage.measure import perimeter

from physmorph.losses.silhouette import set_kernel, soft_silhouette_multi
from physmorph.pipeline.render_loss import make_views
from physmorph.render.exterior import Lattice, ZhuBridson
from physmorph.sampling.mesh import draws_in_frame, load_normalized

torch.set_grad_enabled(False)
NT = int(os.environ.get("D157_THREADS", "6"))
torch.set_num_threads(NT)
set_kernel("cic")
G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
REPO = "/data/relcfd/chayo/physmorph_v2/repo_t1"
VIEWS = make_views(6, (0.0, 0.5, -0.5))
DELTAS = np.round(np.arange(-0.6, 0.6001, 0.01), 4)
GATE = ("armadilo", "bunny", "dragon", "homer", "teapot")


def median_kth_spacing(x, kth, subsample, seed=0):
    n = len(x)
    sub, scale = x, 1.0
    if subsample is not None and n > subsample:
        sub = x[np.random.default_rng(seed).choice(n, subsample, replace=False)]
        scale = (subsample / n) ** (1.0 / 3.0)
    d = cKDTree(sub).query(sub, kth + 1)[0][:, kth].astype(np.float32)
    return float(np.median(d.astype(np.float64))) * scale


def archive(mesh, seed, N, label):
    if N == "300k":
        return f"{G}/d145t_b1/{mesh}300k_{label}{seed}_end1.npz"
    if mesh in GATE:
        return f"{G}/d155/gate/{mesh}40k_{label}{seed}_frames12.npz"
    return f"{G}/d154/g40runs/{mesh}40k_{label}{seed}_end1.npz"


def discs(x, pitch, lattice, h):
    """The run's exterior discs of a state (Lattice.discs, refine off, as target.py and the objective build them)."""
    p, g, _, _ = lattice.discs(ZhuBridson(torch.as_tensor(x, dtype=torch.float32), pitch), h, refine=False)
    return p, nnf.normalize(g, dim=1)


def edge_width(a_t, px, pitch):
    w = []
    for a in a_t:
        a = a.numpy()
        per = perimeter(a > 0.5, neighborhood=4)
        w.append(((a > 0.1) & (a < 0.9)).sum() / max(per, 1e-9))
    return float(np.mean(w)) * px / pitch


def profile(p, n, a_t, res, extent, pitch):
    D, X = [], []
    for d in DELTAS:
        a = soft_silhouette_multi(p + float(d) * pitch * n, VIEWS, res, extent, 1.5)
        D.append(float(torch.clamp(a_t - a, min=0).pow(2).mean()))
        X.append(float(torch.clamp(a - a_t, min=0).pow(2).mean()))
    return np.array(D), np.array(X)


def argmin(L):
    i = int(np.argmin(L))
    if 0 < i < len(L) - 1:
        y0, y1, y2 = L[i - 1], L[i], L[i + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den > 0 else 0.0
        return float(DELTAS[i] + off * (DELTAS[1] - DELTAS[0])), i in (0, len(L) - 1)
    return float(DELTAS[i]), True


def main():
    out_path, specs = sys.argv[1], sys.argv[2:]
    os.chdir(REPO)
    with open(out_path, "a") as out:
        for spec in specs:
            mesh, seed, N, arm, twin = spec.split(":")
            seed = int(seed)
            n = 300000 if N == "300k" else 40000
            t0 = time.time()
            za = np.load(archive(mesh, seed, N, arm), allow_pickle=True)
            tgt = np.asarray(za["tgt"], np.float32)
            js = json.load(open(archive(mesh, seed, N, arm).replace("_end1.npz", ".json").replace("_frames12.npz", ".json")))
            prov = js["provenance"]
            ndraw = int(prov.get("render_target_draws", 1))
            res0 = int(js["arms"][next(iter(js["arms"]))]["config"].get("render_res", 0) or 0)
            fill_src, fill_tgt, frame = {}, {}, {}
            _, v_src = load_normalized("assets/isosphere.obj", n, seed, return_volume=True, sample="stratified",
                                       fill=fill_src)
            tg, _ = load_normalized(f"assets/{mesh}.obj", n, seed + 1, match_volume=v_src, sample="stratified",
                                    return_volume=True, frame=frame, fill=fill_tgt)
            same = float(np.abs(tg - tgt).max())
            draws = draws_in_frame(f"assets/{mesh}.obj", frame, n, [seed + 1 + 100 * k for k in range(1, ndraw)]) \
                if ndraw > 1 else []
            samples = [tgt] + [np.asarray(d, np.float32) for d in draws]
            pitch = .708 * median_kth_spacing(tgt, 8, 20000)
            extent = float(np.abs(tgt).max()) * 1.25
            tt = torch.as_tensor(tgt)
            center = tt.mean(0)
            lattice = Lattice(center, 2.8 * float((tt - center).norm(dim=1).max()))
            run_res = 188 if N == "300k" else 96
            ress = [run_res] + ([run_res // 2, 2 * run_res] if mesh in ("bunny", "dragon") and seed == 97 else [])
            states = {"own": tgt, arm: np.asarray(za["frames"][-1], np.float32),
                      twin: np.asarray(np.load(archive(mesh, seed, N, twin), allow_pickle=True)["frames"][-1],
                                       np.float32)}
            row = dict(mesh=mesh, seed=seed, n=N, draws=len(samples), run_res=run_res, res_cfg=res0, pitch=pitch,
                       extent=extent, tgt_reproduced_maxdiff=same, res={})
            for res in ress:
                h = min(extent / res, 0.92 * pitch)
                px = 2 * extent / res
                sils = []
                for s in samples:
                    p_t, _ = discs(s, pitch, lattice, h)
                    sils.append(soft_silhouette_multi(p_t, VIEWS, res, extent, 1.5))
                a_t = torch.stack(sils).mean(0)
                rr = dict(h_a=h / pitch, px_a=px / pitch, edge_mean_a=edge_width(a_t, px, pitch),
                          edge_single_a=edge_width(sils[0], px, pitch), states={})
                for name, x in states.items():
                    p, nn_ = discs(x, pitch, lattice, h)
                    D, X = profile(p, nn_, a_t, res, extent, pitch)
                    m21, e21 = argmin(2 * D + X)
                    m11, e11 = argmin(D + X)
                    rr["states"][name] = dict(discs=len(p), argmin21=m21, argmin11=m11, bias=m21 - m11,
                                              at_edge=bool(e21 or e11), D0=float(D[60]), X0=float(X[60]),
                                              D=D.tolist(), X=X.tolist())
                row["res"][str(res)] = rr
                print(mesh, seed, N, res, {k: (round(v["argmin21"], 3), round(v["argmin11"], 3), round(v["bias"], 3))
                                           for k, v in rr["states"].items()}, "edge", round(rr["edge_mean_a"], 3),
                      round(rr["edge_single_a"], 3), flush=True)
            row["seconds"] = time.time() - t0
            out.write(json.dumps(row) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
