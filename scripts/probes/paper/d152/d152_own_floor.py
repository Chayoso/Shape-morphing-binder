"""d152_own_floor.py OUT_JSONL MESH SEED -- D152 (H2): the render term's own floor on the tag (D114's floor, redone on
freeze-2026-10-15). The run's setup is rebuilt from the run's own JSON (B1's L, output/gpu/d145t_b1/MESH300k_LSEED.json:
its config and provenance, prepare() with pipeline_run.py's arguments, the eight picture draws), and the render terms
(silhouette, shading on the exterior, discs found afresh at the state and read there, as own_terms_probe.py) and the
transport energy are scored for:
  ind  D90's independent sample of the target (seed + 2, outside the eight draws), at rest: the FLOOR, what a sample of
       the target itself scores against the run's pictures;
  own  the run's own target sample (one of the eight draws, so below the floor by construction);
  L, P the render arm's and the twin's delivered ends (end1);
and, to check the setup, the arm's record at its delivered window (d_sil, d_pbr of the record whose frame_end is the
delivered raw + 1). Run from repo_t (the tag's code), one card. Measurement only."""
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from physmorph import gpu                                      # noqa: E402  (before torch)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
from physmorph.losses.grid_ot import GridSinkhornLoss          # noqa: E402
from physmorph.pipeline.config import PipelineConfig           # noqa: E402
from physmorph.pipeline.render_loss import d_exterior          # noqa: E402
from physmorph.pipeline.target import build_target             # noqa: E402
from physmorph.prepare import prepare                          # noqa: E402
from physmorph.render.exterior import Tracked, ZhuBridson      # noqa: E402

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
out_path, mesh, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
D = f"{G}/d145t_b1/{mesh}300k"
J = json.load(open(f"{D}_L{seed}.json"))
prov = J["provenance"]
arm = next(iter(J["arms"].values()))
c = dict(arm["config"])
fields = {f.name for f in dataclasses.fields(PipelineConfig)}
kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in c.items() if k in fields}
cfg = PipelineConfig(**kw)
missing = sorted(fields - set(c))
assert cfg.exterior_field == "zb" and cfg.render_exterior, (cfg.exterior_field, cfg.render_exterior)
prep = prepare(prov["src"], prov["tgt"], prov["n"], seed, prov["cell_diag"], cfg.young, cfg.poisson,
               log=lambda s: None, loss_ref_n=cfg.mass_ref_n if cfg.loss_follows_n else 0, floor=bool(prov["floor"]),
               surface=0, draws=prov["render_target_draws"], match_density=prov["match_density"],
               cell_ref_n=cfg.mass_ref_n if prov["cell_follows_n"] else 0)
zl, zp = np.load(f"{D}_L{seed}_end1.npz"), np.load(f"{D}_P{seed}_end1.npz")
dt = float(np.abs(np.asarray(zl["tgt"], np.float32) - prep.tgt).max())
if dt > 1e-5:
    raise SystemExit(f"the run's target sample differs from this setup's ({dt:.3g})")
prm = prep.prm
assert abs(prm.dx - prov["mpm"]["dx"]) < 1e-6, (prm.dx, prov["mpm"]["dx"])
tgt = build_target(prep.tgt, prm, cfg, draws=prep.tgt_draws)
try:
    ot = GridSinkhornLoss(tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims, eps=float(tgt.ldx) ** 2, iters=cfg.ot_iters,
                          tol=min(cfg.ot_tol, 1e-3), mass_total=float(tgt.m.sum()), cuda_blocks=True,
                          support=tgt.support)
except Exception as exc:                                       # the transport energy is a side reading only
    print(f"no transport energy: {exc!r}", flush=True)
    ot = None
e = tgt.ext
hist = arm["history"]
raw_L = int(zl["raws"][-1])
rec = next((h for h in hist if h.get("frame_end") == raw_L + 1), None)
jp = json.load(open(f"{D}_P{seed}.json"))
hp = next(iter(jp["arms"].values()))["history"]
raw_P = int(zp["raws"][-1])
rec_p = next((h for h in hp if h.get("frame_end") == raw_P + 1), None)
ind = np.asarray(np.load(f"{G}/d90/{mesh}_ind300k.npz")["tgt"], np.float32)
states = (("ind", ind), ("own", prep.tgt), ("L", np.asarray(zl["frames"][-1], np.float32)),
          ("P", np.asarray(zp["frames"][-1], np.float32)))
with open(out_path, "a") as out:
    for name, xs in states:
        x = gpu.tensor(np.asarray(xs, np.float32))
        with torch.no_grad():
            try:
                te = float(ot.state_energy(x, tgt.m)) if ot is not None else None
            except Exception as exc:
                print(f"no transport energy: {exc!r}", flush=True)
                te = None
            disc = Tracked(ZhuBridson(x, e.pitch), e.lattice, e.h, e.skin)
            p, nrm, _ = disc.read(x)
            sil, pbr = d_exterior(p, nrm, e.sils, e.shade, tgt.views, cfg.render_res, tgt.extent, cfg.sil_k,
                                  cfg.w_hole, cfg.w_spray, cfg.pbr_ambient)
        row = dict(mesh=mesh, seed=seed, state=name, render_res=cfg.render_res, views=len(tgt.views),
                   draws=prov["render_target_draws"], sil=float(sil), pbr=float(pbr),
                   render=float(sil) + cfg.w_pbr * float(pbr), TE=te, discs=int(len(p)),
                   cfg_fields_not_in_json=missing)
        if name == "L" and rec:
            row.update(rec_frame_end=rec["frame_end"], rec_d_sil=rec.get("d_sil"), rec_d_pbr=rec.get("d_pbr"),
                       rec_transport_energy=rec.get("transport_energy"))
        if name == "P" and rec_p:
            row.update(rec_frame_end=rec_p["frame_end"], rec_d_sil=rec_p.get("d_sil"), rec_d_pbr=rec_p.get("d_pbr"),
                       rec_transport_energy=rec_p.get("transport_energy"))
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps(row), flush=True)
