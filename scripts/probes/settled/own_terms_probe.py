"""own_terms_probe.py MESH N FRAMES_NPZ|- [seed=97] — D113: the run's OWN physics and render terms at the source and at
every kept frame of a run, by the pipeline's own setup (pipeline_run.py's defaults: surface proximity, the loss grid and
the render pictures following N, the render terms on the exterior against the mean pictures of eight draws).

Per state, one JSON line: its raw index (-1: the source), the transport energy TE (Sinkhorn divergence + surface
proximity, length^2), the physics geometry ot_scale x TE (the released motion and the end drift need velocities, which
the kept frames do not carry; they are zero at the source, at rest), the silhouette and shading terms on the exterior
(discs found afresh at that state), and the arrival: the share of particles within two target spacings of their
nearest target point and the mean path done, 1 - mean |x_t - x_end| / mean |x_0 - x_end|. FRAMES_NPZ "-": the source
alone (the window records carry everything after it). "ref": FRAMES_NPZ is an independent sample of the target (D90's
`{mesh}_ind300k.npz`, whose `tgt` field is that sample, not the run's target), read as a state at rest (D114's floor)."""
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
from physmorph.losses.grid_ot import GridSinkhornLoss          # noqa: E402
from physmorph.losses.volumetric import d_vol_density          # noqa: E402
from physmorph.pipeline.config import PipelineConfig           # noqa: E402
from physmorph.pipeline.render_loss import d_exterior          # noqa: E402
from physmorph.pipeline.target import build_target, calibrate_units  # noqa: E402
from physmorph.prepare import prepare                          # noqa: E402
from physmorph.render.exterior import Tracked, ZhuBridson      # noqa: E402

mesh, n, frames_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
seed = next((int(s[5:]) for s in sys.argv[4:] if s.startswith("seed=")), 97)
cfg0 = PipelineConfig(loss_follows_n=True)
prep = prepare("assets/isosphere.obj", f"assets/{mesh}.obj", n, seed, 26.0, cfg0.young, cfg0.poisson,
               log=lambda s: None, loss_ref_n=cfg0.mass_ref_n, draws=8)
per_dx = max(1.0, (n / cfg0.mass_ref_n) ** (1.0 / 3.0))
res = int(np.ceil(cfg0.render_res * per_dx))
cfg = dataclasses.replace(cfg0, render_exterior=True, render_res=res, loss_res=prep.loss_res,
                          unit_ref_res=prep.unit_ref_res, nn_berth_k=prep.nn_berth_k)
prm, src = prep.prm, gpu.tensor(prep.src)
tgt = build_target(prep.tgt, prm, cfg, draws=prep.tgt_draws)
calibrate_units(tgt, src, cfg)
ot = GridSinkhornLoss(tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims, eps=float(tgt.ldx) ** 2, iters=cfg.ot_iters,
                      tol=min(cfg.ot_tol, 1e-3), mass_total=float(tgt.m.sum()), cuda_blocks=True, support=tgt.support)
xg = src.detach().clone().requires_grad_(True)
gd = torch.autograd.grad(d_vol_density(xg, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims, tgt.m_ref, tgt.n_support),
                         xg)[0].norm()
xg = src.detach().clone().requires_grad_(True)
gt = torch.autograd.grad(ot.state_energy(xg, tgt.m), xg)[0].norm()
ot_scale = float(gd / gt.clamp_min(1e-30))
knn = gpu.KNN(tgt.pts)
sp_t = gpu.median(knn.query(tgt.pts, 2)[0][:, 1])
e = tgt.ext
states = [(-1, src)]
x_end = None
if frames_path != "-":
    z = np.load(frames_path, allow_pickle=True)
    dt = float(np.abs(np.asarray(z["tgt"], np.float32) - prep.tgt).max())
    if dt > 1e-5 and "ref" not in sys.argv[4:]:
        raise SystemExit(f"the run's target sample differs from this setup's ({dt:.3g})")
    states += [(int(r), None) for r in z["raws"]]
    frames = z["frames"]
    x_end = gpu.tensor(np.asarray(frames[-1], np.float32))
print(json.dumps({"mesh": mesh, "n": n, "seed": seed, "ot_scale": ot_scale, "render_res": res,
                  "target_spacing": sp_t, "states": len(states)}), flush=True)
path0 = None if x_end is None else float((src - x_end).norm(dim=1).mean())
for i, (raw, x) in enumerate(states):
    if x is None:
        x = gpu.tensor(np.asarray(frames[i - 1], np.float32))
    with torch.no_grad():
        te = float(ot.state_energy(x, tgt.m))
        disc = Tracked(ZhuBridson(x, e.pitch), e.lattice, e.h, e.skin)
        p, nrm, _ = disc.read(x)
        sil, pbr = d_exterior(p, nrm, e.sils, e.shade, tgt.views, cfg.render_res, tgt.extent, cfg.sil_k, cfg.w_hole,
                              cfg.w_spray, cfg.pbr_ambient)
        near = float((knn.query(x, 1)[0][:, 0] <= 2.0 * sp_t).float().mean())
        path = None if x_end is None else 1.0 - float((x - x_end).norm(dim=1).mean()) / max(path0, 1e-12)
    print(json.dumps({"state": raw, "TE": te, "phys_geom": ot_scale * te, "sil": float(sil), "pbr": float(pbr),
                      "render": float(sil) + cfg.w_pbr * float(pbr), "near2sp": near, "path": path,
                      "discs": int(len(p))}), flush=True)
