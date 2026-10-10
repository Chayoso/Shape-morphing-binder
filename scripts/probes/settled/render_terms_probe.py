"""render_terms_probe.py FRAMES_NPZ [RES] [ref=NPZ] — D62: the render terms of every kept frame of a run on one yardstick, both
definitions: on the particle cloud (d_render, d_pbr) and on the exterior (d_exterior, the discs found at that frame),
each against the target drawn by the same operator. The pipeline's own functions and constants at render resolution
RES (default 96, the fine level); the particle shading's normal grid spans three render extents about the origin (the
run's spans the MPM domain: the same operator on another box, the same box for every run read here).
One JSON line per frame: its raw index, the two silhouette and the two shading terms, the discs. `ref=NPZ`: the
targets are drawn from that file's sample in place of the run's own (an independent sample of the mesh, D90: the
run's own sample is also the render term's target)."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as nnf                              # noqa: E402
from physmorph.losses.silhouette import set_kernel             # noqa: E402
from physmorph.pipeline.config import PipelineConfig           # noqa: E402
from physmorph.pipeline.render_loss import (d_exterior, d_pbr, d_render, exterior_targets, make_views,  # noqa: E402
                                            shade_targets, target_silhouettes)
from physmorph.render.exterior import Lattice, ZhuBridson      # noqa: E402

dev = torch.device("cuda")
cfg = PipelineConfig()
opts = sys.argv[2:]
res = next((int(s) for s in opts if s.isdigit()), cfg.render_res)
ref = next((s[4:] for s in opts if s.startswith("ref=")), None)
z = np.load(sys.argv[1], allow_pickle=True)
raws, frames = [int(v) for v in z["raws"]], z["frames"]
tgt = torch.as_tensor(np.asarray(np.load(ref)["tgt"] if ref else z["tgt"], np.float32), device=dev)
set_kernel("cic")
views = make_views(cfg.render_views, cfg.render_elevs)
extent = float(tgt.abs().max()) * 1.25
# pitch=WU (D122): the field's pitch and the shading blur at that pitch (a reference that is a surface-dense sample has no one median)
pitch_opt = next((s[6:] for s in opts if s.startswith("pitch=")), None)
sp_t = gpu.median_kth_spacing(tgt, 8, subsample=20000) if pitch_opt is None else float(pitch_opt) / .708
pdx = 2. * extent / res
gmin = torch.full((3,), -1.5 * extent, device=dev)
pdims = (int(np.ceil(3. * extent / pdx)),) * 3
pblur = 1.5 * sp_t / pdx
sils = target_silhouettes(tgt, views, res, extent, cfg.sil_k)
shade = shade_targets(tgt, views, res, extent, gmin, pdx, pdims, cfg.sil_k, cfg.pbr_ambient, blur_cells=pblur)
pitch = .708 * sp_t
center = tgt.mean(0)
lattice = Lattice(center, 2.8 * float((tgt - center).norm(dim=1).max()))
h = min(extent / res, .92 * pitch)


def discs(x):
    p, g, _, _ = lattice.discs(ZhuBridson(x, pitch), h, refine=False)
    return p, nnf.normalize(g, dim=1)


with torch.no_grad():
    e_sils, e_shade = exterior_targets(*discs(tgt), views, res, extent, cfg.sil_k, cfg.pbr_ambient)
    print(f"N {len(tgt)}; render {res} px, extent {extent:.3f} wu; pitch {pitch:.4f} wu, lattice {h / pitch:.2f} pitches = {h / pdx:.2f} pixels")
    for raw, frame in zip(raws, frames):
        x = torch.as_tensor(np.asarray(frame, np.float32), device=dev)
        p, n = discs(x)
        sil_e, pbr_e = d_exterior(p, n, e_sils, e_shade, views, res, extent, cfg.sil_k, cfg.w_hole, cfg.w_spray, cfg.pbr_ambient)
        row = dict(state=raw, discs=len(p),
                   particles=dict(sil=float(d_render(x, sils, views, res, extent, cfg.sil_k, cfg.w_hole, cfg.w_spray)),
                                  pbr=float(d_pbr(x, shade, views, res, extent, gmin, pdx, pdims, cfg.sil_k, cfg.pbr_ambient, pblur))),
                   exterior=dict(sil=float(sil_e), pbr=float(pbr_e)))
        print(json.dumps(row), flush=True)
