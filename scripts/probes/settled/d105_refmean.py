"""d105_refmean.py MESH [FRAMES_NPZ] — the relief reference's mean against the layer's own rough residual (D105). The
relaxation moves a layer particle by -frac (d - dbar - ref) along its normal; d - dbar has a mean of about zero over a layer
(a residual measured against the neighbours' mean), so a reference with a mean mu leaves a push of frac mu outward on every
particle that no arrangement of the layer can answer. Prints, in spacings, on the target sample's own layer and on the last
kept frame of FRAMES_NPZ (if given): the layer's d - dbar, the reference read there (TargetRelief.at), and their
difference. Run on D88's reference (the mesh's value read at the nearest surface point, before D105) it gave a reference
mean of +0.011 to +0.031 against d - dbar's -0.001 to -0.005, and the mesh's own value a mean of +0.0002 / +0.0014."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
from physmorph.prepare import prepare                          # noqa: E402
from physmorph.pipeline.config import PipelineConfig           # noqa: E402
from physmorph.pipeline.target import target_relief            # noqa: E402
from physmorph.pipeline.window.layer import layer_relax_data, layer_spacing  # noqa: E402

mesh = sys.argv[1]
cfg = PipelineConfig()
prep = prepare("assets/isosphere.obj", f"assets/{mesh}.obj", 300000, 97, 26.0, cfg.young, cfg.poisson,
               log=lambda s: None, surface=75000)
tgt = gpu.tensor(prep.tgt)
relief = target_relief(tgt, prep.tgt_surface, cfg)
sp = layer_spacing(tgt)
print(f"{mesh}: spacing {sp:.4f} wu", flush=True)


def layer_rows(x, label):
    x = x.contiguous()
    s_x = layer_spacing(x)
    mask, nrm, nbr, w = layer_relax_data(x, s_x, k=cfg.layer_k, h_sp=cfg.layer_h_sp)
    on = mask > 0.5
    d = (nrm * (x - (w[..., None] * x[nbr]).sum(1))).sum(1)
    rough = d - (w * d[nbr]).sum(1)
    ref = relief.at(x, mask, nrm, nbr, w)
    near = on & (ref != 0)
    r, f = rough[on] / sp, ref[on] / sp
    print(f"  {label}: layer {int(on.sum())} of {len(x)}, near the surface {float(near.float().sum() / on.float().sum()):.3f}; "
          f"d - dbar mean {float(r.mean()):+.4f} rms {float(r.square().mean().sqrt()):.4f}; ref mean {float(f.mean()):+.4f} "
          f"rms {float(f.square().mean().sqrt()):.4f}; (d - dbar - ref) mean {float((r - f).mean()):+.4f}; "
          f"d mean {float(d[on].mean() / sp):+.4f}", flush=True)


layer_rows(tgt, "target sample")
if len(sys.argv) > 2:
    z = np.load(sys.argv[2], allow_pickle=True)
    layer_rows(gpu.tensor(np.asarray(z["frames"][-1], np.float32)), f"end frame raw {int(z['raws'][-1])}")
