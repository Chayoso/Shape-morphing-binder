"""d160_split.py OUT_JSONL SCENE SEED DUMP_DIR RUN_JSON -- D160 step 1, part B (measurement only, on a card): at every
committed window's state of a diagnostic re-run (--term_dump: each term's per-particle position gradient as the
objective weighs it: g_ot = ot_scale grad transport, g_surf = ot_scale grad proximity, g_near, g_spray, g_rend =
lambda grad (silhouette + shading)), the colour's push at the colour front against the physics' push-back.

The render's gradient is split at the same state by the pipeline's own operators (the run's target pack rebuilt by
prepare + build_target with the scene's appearance; the discs found and read as Objective.render_terms):
  route  lambda x grad of the shading term through the discs' albedos alone (points and normals held): the colour's push
  geom   lambda x grad of silhouette + shading through the points and normals (albedos held)
(route + geom is checked against the dump's g_rend). The front: the outer particles some mis-coloured disc reads (a
disc whose shown colour, red share >= 1/2, is not the paint's at it). Per window, over the front:
  rms of |route|, |geom|, |phys| = |g_ot + g_surf|, |clean| = |g_near + g_spray|
  cos(route, phys) (summed over the front) and the share of front particles where route . phys < 0
  along the needed direction d (unit, from the particle to the nearest target-sample disc that wants its colour):
  the mean of -route . d and of -phys . d (positive = toward where its colour is wanted), and the net
  -(route + geom + phys + clean) . d: whether the total descent carries the front material toward its colour
plus the window's lambda, |g_phys|, the telemetry's render_work / phys_work and step_norm."""
import glob
import json
import sys
from pathlib import Path

from physmorph import gpu  # noqa: F401  (before torch)
import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(gpu.__file__).resolve().parents[1]))
from scripts.pipeline_run import parse_args, prepared, run_config  # noqa: E402
from physmorph.pipeline.appearance import make_appearance, source_red, target_red  # noqa: E402
from physmorph.pipeline.render_loss import d_exterior  # noqa: E402
from physmorph.pipeline.target import build_target  # noqa: E402
from physmorph.render.exterior import make_field, tracked  # noqa: E402

out_path, scene, seed, dump, run_json = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
mesh = Path(run_json).name.split("40k")[0]
args = parse_args(["--tgt", f"assets/{mesh}.obj", "--n", "40000", "--seed", str(seed), "--appearance", scene])
cfg0, prep = prepared(args, seed)
cfg = run_config(args, cfg0, prep)
app = make_appearance(scene, prep.src, prep.tgt, prep.tgt_draws)
tgt = build_target(prep.tgt, prep.prm, cfg, draws=prep.tgt_draws, appearance=app)
e = tgt.ext
red_src = source_red(app.params, prep.src)
arm = next(iter(json.load(open(run_json))["arms"].values()))
hist = {h["animation"]: h for h in arm["history"] if h.get("frame_end") and not h.get("null_commit")}
# the target sample's discs and the colour each wants, for the needed direction
fld_t = make_field("zb", tgt.pts, e.pitch, radius=e.radius, offset=e.offset, device_field=True)
with torch.no_grad():
    p_t = e.lattice.discs(fld_t, e.h, refine=False)[0].cpu().numpy()
want_t = target_red(app.params, p_t)
trees = {True: cKDTree(p_t[want_t]), False: cKDTree(p_t[~want_t])}
dev = torch.device("cuda")
out = open(out_path, "a")
for f in sorted(glob.glob(f"{dump}/terms_*.npz")):
    a = int(Path(f).stem.split("_")[1])
    z = np.load(f)
    x = torch.as_tensor(z["x"], device=dev, dtype=torch.float32)
    lam = float(hist.get(a, {}).get("lambda") or 0.0)
    with torch.no_grad():
        discs = tracked(make_field(e.field, x, e.pitch, radius=e.radius, offset=e.offset, device_field=True),
                        e.lattice, e.h, e.skin)
    xg = x.clone().requires_grad_(True)
    p, n, _, c = discs.read(xg, e.albedo, e.albedo_ref)
    _, col_route = d_exterior(p.detach(), n.detach(), e.sils, e.shade, tgt.views, cfg.render_res, tgt.extent, cfg.sil_k,
                              cfg.w_hole, cfg.w_spray, cfg.pbr_ambient, albedo=c)
    g_route = torch.autograd.grad(col_route, xg)[0] * lam
    xg2 = x.clone().requires_grad_(True)
    p2, n2, _, c2 = discs.read(xg2, e.albedo, e.albedo_ref)
    sil, col_geom = d_exterior(p2, n2, e.sils, e.shade, tgt.views, cfg.render_res, tgt.extent, cfg.sil_k, cfg.w_hole,
                               cfg.w_spray, cfg.pbr_ambient, albedo=c2.detach())
    g_geom = torch.autograd.grad(sil + cfg.w_pbr * col_geom, xg2)[0] * lam
    G = {k: torch.as_tensor(z[k], device=dev) for k in ("g_ot", "g_surf", "g_near", "g_spray", "g_rend")}
    phys, clean = G["g_ot"] + G["g_surf"], G["g_near"] + G["g_spray"]
    check = float((g_route + g_geom - G["g_rend"]).norm() / G["g_rend"].norm().clamp_min(1e-30))
    # the front: particles read by mis-coloured discs
    with torch.no_grad():
        r = torch.as_tensor(red_src.astype(np.float32), device=dev)
        valid = discs.idx >= 0
        q = x[discs.idx.clamp_min(0)]
        w = (1. - (discs.p0[:, None] - q).square().sum(-1) / discs.radius ** 2).clamp_min(0.) ** 3 * valid
        s = (w * r[discs.idx.clamp_min(0)]).sum(1) / w.sum(1).clamp_min(1e-12)
        want = torch.as_tensor(target_red(app.params, discs.p0.cpu().numpy()), device=dev)
        bad = (s >= .5) != want
        front_idx = torch.unique(discs.idx[bad][discs.idx[bad] >= 0])
        outer_idx = torch.unique(discs.idx[discs.idx >= 0])
    F = front_idx
    xf = x[F].cpu().numpy()
    redf = red_src[F.cpu().numpy()]
    d = np.zeros_like(xf)
    for col in (True, False):
        m = redf == col
        if m.any():
            _, j = trees[col].query(xf[m])
            v = p_t[want_t == col][j] - xf[m]
            d[m] = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    dT = torch.as_tensor(d, device=dev, dtype=torch.float32)
    rms = lambda g: float(g[F].square().sum(1).mean().sqrt())  # noqa: E731
    gr, gp = g_route[F], phys[F]
    tot = (g_route + g_geom + phys + clean)[F]
    h = hist.get(a, {})
    row = dict(window=a, t=(h.get("frame_end") or 0) / 240.0, lam=lam, g_phys_norm=h.get("g_phys_norm"),
               render_work=h.get("render_work"), phys_work=h.get("phys_work"), step_norm=h.get("step_norm"),
               move=h.get("move"), front=int(len(F)), outer=int(len(outer_idx)), mis=float(bad.float().mean()),
               split_check=check,
               rms_route=rms(g_route), rms_geom=rms(g_geom), rms_phys=rms(phys), rms_clean=rms(clean),
               cos_route_phys=float((gr * gp).sum() / (gr.norm() * gp.norm()).clamp_min(1e-30)),
               opposed_share=float(((gr * gp).sum(1) < 0).float().mean()),
               route_along=float((-(gr * dT).sum(1)).mean()), phys_along=float((-(gp * dT).sum(1)).mean()),
               geom_along=float((-(g_geom[F] * dT).sum(1)).mean()), clean_along=float((-(clean[F] * dT).sum(1)).mean()),
               total_along=float((-(tot * dT).sum(1)).mean()),
               route_along_unit=float((-(gr * dT).sum(1) / gr.norm(dim=1).clamp_min(1e-30)).mean()))
    out.write(json.dumps(row) + "\n")
    out.flush()
    print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
