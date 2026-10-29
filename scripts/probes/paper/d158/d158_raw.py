"""d158_raw.py OUT_DIR MESH DENSE_NPZ[,...] KEPT[,KEPT...] [+res_loop=188] [+draws=8] [+seed=97] -- D158 follow-up (a
measurement): the dumped layers drawn as the LOOP draws its discs, with the exterior field's own normals (the field's
gradient at the disc, Tracked.read; the targets' normals too) in place of D115's display normals (the field's averaged
twice over the base reach), the slope of the drawn height tilting them as before. Per dumped frame and picture (the
four of d158_fit.py): the render terms of the run's surface (h = 0), the relief alone (h_rel), its parts one value per
outer particle can hold (P_W h_rel, P_F h_rel), the smooth part (h_low) and the whole layer (h). One JSON line a frame
in OUT_DIR/MESH300k_L97/raw.jsonl."""
import json, os, sys
from pathlib import Path
REPO = os.environ.get("D158_REPO", "/data/relcfd/chayo/physmorph_v2/repo_freeze1015")
sys.path.insert(0, REPO)
from physmorph import gpu                                      # noqa: E402,F401
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as nnf                              # noqa: E402
from physmorph.pipeline.config import PipelineConfig           # noqa: E402
from physmorph.pipeline.render_loss import d_exterior, exterior_targets, make_views  # noqa: E402
from physmorph.render.exterior import Lattice, make_field      # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.sampling.mesh import draws_in_frame, load_normalized  # noqa: E402

dev = torch.device(os.environ.get("D158_DEV", "cuda"))           # cpu: the follow-up without a card
opts = dict(s[1:].split("=", 1) for s in sys.argv[1:] if s.startswith("+"))
argv = [s for s in sys.argv if not s.startswith("+")]
out, MESH = Path(argv[1]), argv[2]
KEPT = [int(v) for v in argv[4].split(",")]
RES_LOOP, NDRAW, SEED = int(opts.get("res_loop", 188)), int(opts.get("draws", 8)), int(opts.get("seed", 97))
d = out / f"{MESH}300k_L97"
head = json.load(open(d / "head.json"))
cfg = PipelineConfig()
z = np.load(head["frames12"], allow_pickle=True)
tgt_np = np.asarray(z["tgt"], np.float32)
tgt = torch.as_tensor(tgt_np, device=dev)
denses = [torch.as_tensor(np.asarray(np.load(p)["tgt"], np.float32), device=dev) for p in argv[3].split(",")]
pitch = lambda s: .708 * float(knn_self_torch(s, 9)[0][:, 8].median())   # noqa: E731
a, a_c = pitch(tgt), pitch(denses[0])
center = tgt.mean(0)
radius = float((tgt - center).norm(dim=1).max())
extent = float(tgt.abs().max()) * 1.25
res = int(np.ceil(2.0 * extent / a_c))
h_lat = 0.92 * a_c
lat = Lattice(center, 2.8 * radius)
views = make_views(cfg.render_views, cfg.render_elevs)


def exterior(s, a_field):
    with torch.no_grad():
        p, g, _, _ = lat.discs(make_field("zb", s, a_field, radius=3., offset=.8), h_lat)
    return p, nnf.normalize(g, dim=1)


def mean_pics(exts, R):
    per = [exterior_targets(pp, nn, views, R, extent, cfg.sil_k, cfg.pbr_ambient) for pp, nn in exts]
    return ([torch.stack(v).mean(0) for v in zip(*[e[0] for e in per])],
            [torch.stack(v).mean(0) for v in zip(*[e[1] for e in per])])


dense_ext = [exterior(s, a_c) for s in denses]
os.chdir(REPO)
n_run = len(tgt_np)
frame = {}
_, v_src = load_normalized("assets/isosphere.obj", n_run, SEED, return_volume=True, sample="stratified", fill={})
load_normalized(f"assets/{MESH}.obj", n_run, SEED + 1, match_volume=v_src, sample="stratified", return_volume=True,
                frame=frame, fill={})
draws = draws_in_frame(f"assets/{MESH}.obj", frame, n_run, [SEED + 1 + 100 * k for k in range(1, NDRAW)]) if NDRAW > 1 else []
loop_ext = [exterior(s, a) for s in [tgt] + [torch.as_tensor(np.asarray(dd, np.float32), device=dev) for dd in draws]]
SETTINGS = {"dense_ac": (*mean_pics(dense_ext, res), res), "dense_loop": (*mean_pics(dense_ext, RES_LOOP), RES_LOOP),
            "loop_ac": (*mean_pics(loop_ext, res), res), "loop_loop": (*mean_pics(loop_ext, RES_LOOP), RES_LOOP)}
K = 16
for k in KEPT:
    dz = np.load(d / f"dump_{k:04d}.npz")
    T = {q: torch.as_tensor(dz[q], device=dev) for q in ("p", "n", "x", "h", "h_low", "h_rel", "rel_uW", "rel_uF")}
    p, x = T["p"], T["x"]
    with torch.no_grad():
        _, g, _ = make_field("zb", x, a, radius=3., offset=.8)(p)
    n = nnf.normalize(g, dim=1)                                # the field's own normal (the loop's, the targets')
    _, nb = knn_self_torch(p, K + 1)
    nb = nb[:, 1:]
    side = ((n[nb] * n[:, None]).sum(-1) > 0.5).float()
    ref = torch.where(n[:, :1].abs() < .9, n.new_tensor((1., 0., 0.)), n.new_tensor((0., 1., 0.))).expand_as(n)
    t1 = nnf.normalize(torch.linalg.cross(n, ref), dim=1)
    t2 = torch.linalg.cross(n, t1)
    dd = p[nb] - p[:, None]
    D = torch.stack(((dd * t1[:, None]).sum(-1), (dd * t2[:, None]).sum(-1)), -1)
    w = (torch.exp(-(dd.norm(dim=-1) / (1.5 * h_lat)) ** 2) * side)[..., None]
    A = torch.linalg.solve((D * w).transpose(1, 2) @ D + 1e-6 * h_lat ** 2 * torch.eye(2, device=dev), (D * w).transpose(1, 2))

    def drawn(hh, base):
        g2 = (A @ (hh[nb] - hh[:, None])[..., None]).squeeze(-1)
        return p + hh[:, None] * base, nnf.normalize(base - g2[:, :1] * t1 - g2[:, 1:] * t2, dim=1)

    var = {"zero": torch.zeros_like(T["h"]), "rel": T["h_rel"], "rel_uW": T["rel_uW"], "rel_uF": T["rel_uF"],
           "low": T["h_low"], "h": T["h"]}
    row = dict(kept=k, t_s=k / 20.0, normal_angle_deg_vs_display=float(torch.rad2deg(torch.acos((n * T["n"]).sum(1).clamp(-1, 1))).median()),
               raw={}, display={})
    with torch.no_grad():
        for sname, st in SETTINGS.items():
            for lab, base in (("raw", n), ("display", T["n"])):
                row[lab][sname] = {vn: [float(t) for t in d_exterior(*drawn(vv, base), st[0], st[1], views, st[2], extent,
                                                                      cfg.sil_k, ambient=cfg.pbr_ambient)]
                                   for vn, vv in var.items()}
    with open(d / "raw.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    tot = lambda t: t[0] + t[1]                                 # noqa: E731
    for sname in SETTINGS:
        r = row["raw"][sname]
        z0 = r["zero"]
        print(f"{MESH} t {k / 20:.2f} {sname:10s} raw normals: relief alone total {(tot(r['rel']) - tot(z0)) / tot(z0):+.3f} "
              f"sil {(r['rel'][0] - z0[0]) / z0[0]:+.3f} shading {(r['rel'][1] - z0[1]) / z0[1]:+.3f}; P_W relief "
              f"{(tot(r['rel_uW']) - tot(z0)) / tot(z0):+.3f}; P_F relief {(tot(r['rel_uF']) - tot(z0)) / tot(z0):+.3f}; "
              f"relief on the smooth part {(tot(r['h']) - tot(r['low'])) / tot(r['low']):+.3f}; whole layer {(tot(r['h']) - tot(z0)) / tot(z0):+.3f}",
              flush=True)
