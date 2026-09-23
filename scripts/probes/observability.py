"""Observability probe: which viewer-visible defects does each render loss SEE?

On a converged archived state, compute per-particle |dL/dx| for the silhouette
loss and for the gauss (diff_gauss) loss, and cross-tabulate with floater status
(target-NN distance > 2x target spacing) and with interior/surface membership.
Answers root_analysis §1 quantitatively: the fraction of floaters that receive
ANY restoring gradient from each observation model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from physmorph.mpm import MPMParams  # noqa: E402
from physmorph.pipeline import PipelineConfig  # noqa: E402
from physmorph.pipeline.render_loss import d_render  # noqa: E402
from physmorph.pipeline.runner import _surface_weights, build_target  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--gauss_res", type=int, default=96)
    args = ap.parse_args()
    d = np.load(args.npz)
    dn = int(d["deliver_n"]) if "deliver_n" in d.files else len(d["frames"])
    src, tgt_x, x = d["src"], d["tgt"], d["frames"][dn - 1].astype(np.float32)
    print(f"[obs] state = delivered frame {dn} of {len(d['frames'])}")
    N = len(x)
    prm = MPMParams()
    cfg = PipelineConfig(T=20, iters=8, animations=8, loss_res=64)
    cfg.lambda_auto = 0.5
    cfg.use_gauss_loss, cfg.gauss_res = True, args.gauss_res
    cfg.gauss_children, cfg.render_surface_only, cfg.surface_grad_frac = 4, True, 0.5
    cfg.gauss_in_objective = False
    tgt = build_target(tgt_x, prm, cfg)
    sw = _surface_weights(src, cfg.surface_grad_k, 0.5, cfg.surface_grad_floor) > 0.5
    tgt.gauss.configure_source(src, sw)
    tt = cKDTree(tgt_x)
    sp = float(np.median(tt.query(tgt_x, k=2)[0][:, 1]))
    dtg = tt.query(x)[0]
    floater = dtg > 2 * sp
    far = dtg > 3 * sp
    print(f"[obs] N={N} floaters(>2sp)={floater.sum()} far(>3sp)={far.sum()} "
          f"surface parents={sw.sum()} gauss res={tgt.gauss.res}")

    xt = torch.tensor(x, device="cuda", requires_grad=True)
    L_sil = d_render(xt, tgt.sils, tgt.views, cfg.render_res, tgt.extent,
                     cfg.sil_k, cfg.w_hole, cfg.w_spray)
    (g_sil,) = torch.autograd.grad(L_sil, xt)
    xt2 = torch.tensor(x, device="cuda", requires_grad=True)
    mask = torch.as_tensor(sw, device="cuda")
    L_g = tgt.gauss.loss(xt2, None, mask=mask)
    (g_g,) = torch.autograd.grad(L_g, xt2)
    gs = g_sil.norm(dim=1).cpu().numpy()
    gg = g_g.norm(dim=1).cpu().numpy()
    print(f"[obs] L_sil={float(L_sil):.5f} L_gauss={float(L_g):.5f}")

    def report(tag, sel):
        n = int(sel.sum())
        if n == 0:
            print(f"[obs] {tag}: n=0"); return
        es = (gs[sel] ** 2).sum() / max((gs ** 2).sum(), 1e-30)
        eg = (gg[sel] ** 2).sum() / max((gg ** 2).sum(), 1e-30)
        med_s, med_g = np.median(gs), np.median(gg)
        seen_s = (gs[sel] > 0.1 * med_s).mean()
        seen_g = (gg[sel] > 0.1 * med_g).mean()
        zero_s = (gs[sel] == 0).mean()
        zero_g = (gg[sel] == 0).mean()
        print(f"[obs] {tag:22s} n={n:6d}  sil: energy_share={es:.3f} seen(>0.1med)={seen_s:.2f} zero={zero_s:.2f}"
              f"  | gauss: energy_share={eg:.3f} seen={seen_g:.2f} zero={zero_g:.2f}")

    report("ALL", np.ones(N, bool))
    report("floaters >2sp", floater)
    report("far floaters >3sp", far)
    report("body (<=2sp)", ~floater)
    report("surface parents", sw)
    report("interior parents", ~sw)
    report("floater & surface", floater & sw)
    report("floater & interior", floater & ~sw)
    # directional agreement on floaters: does the gradient point TOWARD the target?
    _, nn = tt.query(x[floater])
    to_tgt = tgt_x[nn] - x[floater]
    to_tgt /= np.linalg.norm(to_tgt, axis=1, keepdims=True) + 1e-12
    for tag, g in (("sil", g_sil), ("gauss", g_g)):
        gf = -g.cpu().numpy()[floater]
        nrm = np.linalg.norm(gf, axis=1)
        ok = nrm > 0
        cos = (gf[ok] * to_tgt[ok]).sum(1) / (nrm[ok] + 1e-30)
        print(f"[obs] floaters with nonzero {tag} grad: {ok.mean():.2f}; "
              f"median cos(-grad, to-target) = {np.median(cos) if ok.any() else float('nan'):+.3f}; "
              f"frac cos>0.5 = {(cos > 0.5).mean() if ok.any() else float('nan'):.2f}")


if __name__ == "__main__":
    main()
