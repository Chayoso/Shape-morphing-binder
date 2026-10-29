"""d152_yard_floor.py OUT_JSONL MESH:SEED [...] -- D152 (H1/H2): the yardstick's floor. The yardstick is
render_terms_probe.py (repo_r77, 96 px, the exterior's discs, the targets drawn from D90's independent sample
output/gpu/d90/MESH_ind300k.npz). Here the same operator, unchanged, scores per MESH:SEED three states:
  own  the run's own target sample (the end1 archive's `tgt`; L and P of one seed share it), at rest: what a run that put
       every particle on its own target sample would score (the yardstick's sampling floor, as the display's `own`);
  L, P the render arm's and the twin's delivered end (end1), to check that this script reproduces t3end_*.log.
Run from repo_r77 (the yardstick's code), one card. Measurement only."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from physmorph import gpu                                      # noqa: E402,F401  (before torch)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as nnf                              # noqa: E402
from physmorph.losses.silhouette import set_kernel             # noqa: E402
from physmorph.pipeline.config import PipelineConfig           # noqa: E402
from physmorph.pipeline.render_loss import (d_exterior, d_pbr, d_render, exterior_targets, make_views,  # noqa: E402
                                            shade_targets, target_silhouettes)
from physmorph.render.exterior import Lattice, ZhuBridson      # noqa: E402

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
dev = torch.device("cuda")
cfg = PipelineConfig()
res = cfg.render_res_hi
out = open(sys.argv[1], "a")
set_kernel("cic")
views = make_views(cfg.render_views, cfg.render_elevs)
for spec in sys.argv[2:]:
    mesh, seed = spec.split(":")
    ind = torch.as_tensor(np.asarray(np.load(f"{G}/d90/{mesh}_ind300k.npz")["tgt"], np.float32), device=dev)
    zl = np.load(f"{G}/d145t_b1/{mesh}300k_L{seed}_end1.npz")
    zp = np.load(f"{G}/d145t_b1/{mesh}300k_P{seed}_end1.npz")
    own = np.asarray(zl["tgt"], np.float32)
    same = float(np.abs(own - np.asarray(zp["tgt"], np.float32)).max())
    tgt = ind
    extent = float(tgt.abs().max()) * 1.25
    sp_t = gpu.median_kth_spacing(tgt, 8, subsample=20000)
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
        for name, xs in (("own", own), ("L", np.asarray(zl["frames"][-1], np.float32)),
                         ("P", np.asarray(zp["frames"][-1], np.float32))):
            x = torch.as_tensor(xs, device=dev)
            p, n = discs(x)
            sil_e, pbr_e = d_exterior(p, n, e_sils, e_shade, views, res, extent, cfg.sil_k, cfg.w_hole, cfg.w_spray,
                                      cfg.pbr_ambient)
            row = dict(mesh=mesh, seed=int(seed), state=name, res=res, own_LP_same=same, discs=len(p),
                       particles=dict(sil=float(d_render(x, sils, views, res, extent, cfg.sil_k, cfg.w_hole, cfg.w_spray)),
                                      pbr=float(d_pbr(x, shade, views, res, extent, gmin, pdx, pdims, cfg.sil_k,
                                                      cfg.pbr_ambient, pblur))),
                       exterior=dict(sil=float(sil_e), pbr=float(pbr_e)))
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps(row), flush=True)
    torch.cuda.empty_cache()
