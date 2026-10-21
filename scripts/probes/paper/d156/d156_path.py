"""d156_path.py OUT_JSONL MESH:SEED [...] -- D156's diagnostic of the render loss itself: does it fall all the way from
the hull to the true mesh along a carve?

CPU, d156_read.Ctx (the run's own frame and render operator: pitch, box, lattice, the eight-draw pictures of the true
mesh at 96 px). The carve path: the solids S_t = {x : (1 - t) sd_hull(x) + t sd_mesh(x) < 0}, t = 0 (the hull) ... 1
(the mesh), sd the signed distances (open3d; the mesh's boundary loops fan-closed) on a grid of half a pitch over the
hull's box plus three pitches, each solid meshed by marching cubes and sampled by the pipeline's own stratified sampler
(sampling.mesh.stratified_draws, seed + 1) at the hull sample's number density: every S_t is read as a target sample
of itself would be, the same sampling noise at each t. At each t: the run's silhouette and shading terms (d_exterior)
of the sample's exterior against the true mesh's pictures; the floor: the same for Mp (an independent sample of the
mesh). Earlier attempt (path97_cut_v1.jsonl, not used): cutting the hull's sample at the mesh's exact surface also cut
the sampler's own outer shell, which a sample of the mesh keeps, so its end sat inside the mesh's sample.
A loss the render can descend from the hull to the mesh falls in t; a rise or a bump means the pictures' loss does not
ask for the carve."""
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from skimage.measure import marching_cubes

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d156_read as R  # noqa: E402
from physmorph.sampling.mesh import stratified_draws  # noqa: E402

G = "/data/relcfd/chayo/physmorph_v2/output/gpu/d156/runs40"
TS = np.round(np.linspace(0.0, 1.0, 11), 2)
out = open(sys.argv[1], "a")


def sdf(surf, P):
    return surf.scene.compute_signed_distance(o3d.core.Tensor(P.astype(np.float32)), nthreads=R.NT,
                                              nsamples=3).numpy().astype(np.float64)


def terms(ctx, x):
    p, n = ctx.discs(x)
    s, b = R.d_exterior(p, n, *ctx.pic, R.TRAIN, R.RES_RUN, ctx.box, R.cfg.sil_k, R.cfg.w_hole, R.cfg.w_spray,
                        R.cfg.pbr_ambient)
    return float(s), float(b)


for spec in sys.argv[2:]:
    mesh, seed = spec.split(":")
    seed = int(seed)
    ctx = R.Ctx(mesh, seed, "hull", f"{G}/{mesh}40k_L_{seed}_frames12.npz")
    h = 0.5 * ctx.a
    lo = ctx.hull.V.min(0) - 3 * ctx.a
    nc = np.ceil((ctx.hull.V.max(0) + 3 * ctx.a - lo) / h).astype(int) + 1
    g = [lo[i] + h * np.arange(nc[i]) for i in range(3)]
    P = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
    sh, sm = sdf(ctx.hull, P).reshape(*nc), sdf(ctx.true, P).reshape(*nc)
    density = len(ctx.H) / ctx.vol_hull
    rows = []
    for t in TS:
        f = (1 - t) * sh + t * sm
        v, fc, _, _ = marching_cubes(f, level=0.0, spacing=(h, h, h))
        S = trimesh.Trimesh(v + lo, fc, process=True)
        if S.volume < 0:
            S.invert()
        n_t = int(round(density * S.volume))
        x = stratified_draws(S, n_t, [seed + 1])[0]
        sil, pbr = terms(ctx, x)
        rows.append(dict(t=float(t), n=n_t, volume=float(S.volume), sil=sil, pbr=pbr))
        print(json.dumps(dict(mesh=mesh, t=float(t), n=n_t, sil=sil, pbr=pbr)), flush=True)
    fs, fp = terms(ctx, ctx.Mp)
    row = dict(mesh=mesh, seed=seed, grid=[int(k) for k in nc], h=h, path=rows, floor=dict(sil=fs, pbr=fp),
               monotone_sil=bool(all(b["sil"] <= a["sil"] for a, b in zip(rows, rows[1:]))),
               monotone_pbr=bool(all(b["pbr"] <= a["pbr"] for a, b in zip(rows, rows[1:]))))
    out.write(json.dumps(row) + "\n")
    out.flush()
