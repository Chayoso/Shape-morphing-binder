"""d156_demand.py OUT_JSONL RUN_PREFIX [RUN_PREFIX ...] -- D156's diagnostic (pre-registered item (5)'s "does the render's
demand point into the excess, and how much of it does the transport hold"): at a D156 run's end state, the per-particle
descent directions of the render terms (against the true mesh's pictures) and of the transport (to the hull), read on
the particles in the hull's excess over the mesh. A measurement on a card (the pipeline's own operators, repo_d156).

RUN_PREFIX: output/gpu/d156/runs40/MESH40k_ARM_SEED (its .json gives the run's flags, its _frames12.npz the end state).
The run's TargetPack is rebuilt from its flags (scripts/pipeline_run.prepared / run_config, target.build_target with
the render target's samples), the transport as the window objective builds it (GridSinkhornLoss, blur one loss cell),
its scale ot_scale and the render weight lambda from the run's last committed window (for a twin, lambda of its render
arm L at the same seed). At the end state x:
  g_t = ot_scale d(transport)/dx, g_s = d(silhouette)/dx, g_p = w_pbr d(shading)/dx (the exterior's tracked discs, as
  Objective.render_terms), g_r = lambda (g_s + g_p).
On the particles in R = hull minus mesh (open3d occupancy of the hull and of the fan-closed mesh in the run's frame), with
u = the unit direction to the nearest point of the mesh (inward, toward what the pictures show): the mean of -g . u
for each (> 0: its descent moves the particle toward the mesh), the share of the particles whose -g_r . u > 0, and
lambda |g_r . u| / |g_t . u| summed over R (the render's inward demand against the transport's hold). The same on the
outer-layer particles outside R as the scale."""
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh

sys.path.insert(0, str(Path.cwd()))
from physmorph import gpu  # noqa: E402
from physmorph.losses.grid_ot import GridSinkhornLoss  # noqa: E402
from physmorph.pipeline.render_loss import d_exterior  # noqa: E402
from physmorph.pipeline.target import build_target, calibrate_units  # noqa: E402
from physmorph.render.exterior import make_field, tracked  # noqa: E402
from physmorph.sampling.mesh import _oriented, load_normalized  # noqa: E402
from scripts.pipeline_run import parse_args, prepared, run_config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d157_lib import close_holes  # noqa: E402


def scene(V, F):
    s = o3d.t.geometry.RaycastingScene()
    s.add_triangles(o3d.core.Tensor(np.asarray(V, np.float32)), o3d.core.Tensor(np.asarray(F, np.uint32)))
    return s


def last_committed(hist):
    return [h for h in hist if h.get("frame_end") and not h.get("null_commit")][-1]


out = open(sys.argv[1], "a")
for prefix in sys.argv[2:]:
    js = json.load(open(prefix + ".json"))
    pv = js["provenance"]
    arm = list(js["arms"].values())[0]
    argv = ["--tgt", pv["tgt"], "--render_tgt", pv["render_tgt"], "--n", str(pv["n"]), "--seed", str(pv["seed"])]
    args = parse_args(argv)
    cfg0, prep = prepared(args, args.seed)
    cfg = run_config(args, cfg0, prep)
    tgt = build_target(prep.tgt, prep.prm, cfg, draws=prep.tgt_draws, render_x=prep.render_x)
    calibrate_units(tgt, gpu.tensor(prep.src), cfg)
    rec = last_committed(arm["history"])
    lam = float(rec["lambda"] or 0.0)
    lam_from = "own"
    if lam == 0.0:                                           # a twin: its render arm's lambda (same seed)
        jl = json.load(open(prefix.replace("_P_", "_L_") + ".json"))
        lam, lam_from = float(last_committed(list(jl["arms"].values())[0]["history"])["lambda"]), "L"
    ot_scale = float(rec["ot_scale"])
    ot = GridSinkhornLoss(tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims, eps=float(tgt.ldx) ** 2, iters=cfg.ot_iters,
                          tol=min(cfg.ot_tol, 1e-3), mass_total=float(tgt.m.sum()), cuda_blocks=True, support=tgt.support)
    x = gpu.tensor(np.asarray(np.load(prefix + "_frames12.npz", allow_pickle=True)["frames"][-1], np.float32))
    xg = x.clone().requires_grad_(True)
    g_t = ot_scale * torch.autograd.grad(ot.state_energy(xg, tgt.m), xg)[0]
    e = tgt.ext
    discs = tracked(make_field(e.field, x.detach(), e.pitch, radius=e.radius, offset=e.offset, device_field=True),
                    e.lattice, e.h, e.skin)
    xg = x.clone().requires_grad_(True)
    p, n, _ = discs.read(xg)
    sil, pbr = d_exterior(p, n, e.sils, e.shade, tgt.views, cfg.render_res, tgt.extent, cfg.sil_k, cfg.w_hole,
                          cfg.w_spray, cfg.pbr_ambient)
    g_s = torch.autograd.grad(sil, xg, retain_graph=True)[0]
    g_p = cfg.w_pbr * torch.autograd.grad(pbr, xg)[0]
    G = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in (("t", g_t), ("s", g_s), ("p", g_p))}
    G["r"] = lam * (G["s"] + G["p"])
    # the meshes in the run's frame (the target's: the hull's load_normalized, as prepare)
    _, v_src = load_normalized(args.src, args.n, args.seed, return_volume=True, sample="stratified")
    frame = {}
    load_normalized(args.tgt, args.n, args.seed + 1, match_volume=v_src, sample="stratified", frame=frame)
    put = lambda path: ((np.asarray(_oriented(path).vertices, np.float64) - frame["offset"]) * frame["scale"],  # noqa: E731
                        np.asarray(_oriented(path).faces, np.int64))
    tc, _ = close_holes(*put(args.render_tgt))
    hv, hf = put(args.tgt)
    st, sh = scene(tc.vertices, tc.faces), scene(hv, hf)
    xc = x.cpu().numpy()
    q = o3d.core.Tensor(xc.astype(np.float32))
    in_t = st.compute_occupancy(q, nsamples=3).numpy() > 0.5
    in_h = sh.compute_occupancy(q, nsamples=3).numpy() > 0.5
    R = in_h & ~in_t
    cp = st.compute_closest_points(q)["points"].numpy().astype(np.float64)
    u = cp - xc
    dist = np.linalg.norm(u, axis=1)
    u = u / np.maximum(dist, 1e-12)[:, None]
    # the outer layer outside R (the scale): particles within one pitch of the hull's surface, inside the mesh
    dh = sh.compute_distance(q).numpy()
    lay = (~R) & in_t & (dh < e.pitch)
    row = dict(run=prefix.split("/")[-1], lam=lam, lam_from=lam_from, ot_scale=ot_scale, n_R=int(R.sum()),
               n_layer=int(lay.sum()), R_depth_median_pitch=float(np.median(dist[R]) / e.pitch) if R.any() else None,
               sil=float(sil), pbr=float(pbr))
    for name, m in (("R", R), ("layer", lay)):
        if not m.any():
            continue
        s = {k: float((-(G[k][m] * u[m]).sum(1)).mean()) for k in ("t", "s", "p", "r")}       # inward descent per particle
        s["share_render_inward"] = float(((-(G["r"][m] * u[m]).sum(1)) > 0).mean())
        s["share_transport_inward"] = float(((-(G["t"][m] * u[m]).sum(1)) > 0).mean())
        s["render_over_transport_abs"] = float(np.abs((G["r"][m] * u[m]).sum(1)).sum()
                                               / max(np.abs((G["t"][m] * u[m]).sum(1)).sum(), 1e-30))
        s["rms_r"] = float(np.sqrt((G["r"][m] ** 2).sum(1).mean()))
        s["rms_t"] = float(np.sqrt((G["t"][m] ** 2).sum(1).mean()))
        row[name] = s
    out.write(json.dumps(row) + "\n")
    out.flush()
    print(json.dumps(row), flush=True)
