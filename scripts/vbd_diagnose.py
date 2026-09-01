"""VBD convergence diagnosis — WHY does the block-descent crawl? (one instrumented commit)

Measures, at full particle count, everything the crawl hypothesis space needs:
  1. per-sweep energy / |gradE| / mean accepted step  -> stall shape (fast drop then flat?)
  2. per-TERM gradient norms on u at exit (elastic vs D_vol vs lambda*D_render vs leash)
     -> which term owns the residual
  3. SMOOTH/ROUGH decomposition of the exit gradient on the node grid: k Jacobi
     neighbour-average passes -> if the residual is dominated by the SMOOTH component,
     that is the textbook multigrid signature (GS kills rough error, stalls on smooth
     modes) and coarse-grid correction is the indicated fix; if ROUGH dominates, the
     block solves themselves are wrong-scaled.

Run (hyde06, free GPU):
  CUDA_VISIBLE_DEVICES=0 python scripts/vbd_diagnose.py --n 20000 --sweeps 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from physmorph.losses.volumetric import d_vol  # noqa: E402
from physmorph.mpm import MPMParams  # noqa: E402
from physmorph.pipeline.config import PipelineConfig  # noqa: E402
from physmorph.pipeline.render_loss import LambdaBalancer, d_render  # noqa: E402
from physmorph.pipeline.runner import build_target  # noqa: E402
from physmorph.sampling import load_normalized  # noqa: E402
from physmorph.vbd import QuasiStaticGrid  # noqa: E402


def smooth_on_grid(qs, g, iters):
    """k Jacobi neighbour-average passes of a node field over the FULL grid."""
    nx, ny, nz = qs.dims
    full = torch.zeros(nx * ny * nz, 3, device=g.device)
    full[qs.active] = g
    f = full.reshape(nx, ny, nz, 3)
    for _ in range(iters):
        s = torch.zeros_like(f)
        s[1:] += f[:-1]; s[:-1] += f[1:]
        s[:, 1:] += f[:, :-1]; s[:, :-1] += f[:, 1:]
        s[:, :, 1:] += f[:, :, :-1]; s[:, :, :-1] += f[:, :, 1:]
        f = s / 6.0
    return f.reshape(-1, 3)[qs.active]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--sweeps", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="output/vbd_diagnose")
    args = ap.parse_args()

    src = load_normalized("assets/isosphere.obj", args.n, 1)
    tgt_x = load_normalized("assets/bunny.obj", args.n, 2)
    prm = MPMParams()
    cfg = PipelineConfig(lambda_auto=0.5, device=args.device)
    tgt = build_target(tgt_x, prm, cfg)
    qs = QuasiStaticGrid(src, prm.grid_min, prm.dx, (prm.nx, prm.ny, prm.nz),
                         cfg.vbd_young, cfg.poisson, args.device)
    Fe0 = torch.eye(3, device=args.device).expand(len(src), 3, 3).contiguous()

    def parts(u):
        E_el, _ = qs.elastic(u, Fe0)
        disp, _ = qs.kinematics(u)
        xn = qs.x0 + disp
        lv = d_vol(xn, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims)
        lb = torch.clamp(xn.abs() - tgt.extent, min=0).pow(2).sum(1).mean()
        lr = d_render(xn, tgt.sils, tgt.views, cfg.render_res, tgt.extent,
                      cfg.sil_k, cfg.w_hole, cfg.w_spray)
        return E_el, lv, lb, lr

    u0 = torch.zeros(qs.A, 3, device=args.device, requires_grad=True)
    E_el, lv, lb, lr = parts(u0)
    gp = torch.autograd.grad(E_el + lv + cfg.w_box * lb, u0, retain_graph=True)[0]
    gr = torch.autograd.grad(lr, u0)[0]
    lam_r = LambdaBalancer(0.5).update(float(gp.norm()), float(gr.norm()))

    def energy(u):
        E_el, lv, lb, lr = parts(u)
        return E_el + lv + cfg.w_box * lb + lam_r * lr

    print(f"[diag] N={args.n} A={qs.A} nodes  lam_r={lam_r:.1f}  sweeps<={args.sweeps}",
          flush=True)
    u, info = qs.solve(energy, sweeps=args.sweeps, tol=1e-4, step=cfg.vbd_step)

    E_tr = info["energy"]
    print("[diag] per-sweep (every 5th): sweep | E | |gradE| | mean step t")
    for i in range(0, len(info["gnorms"]), 5):
        print(f"   {i:4d} | {E_tr[i+1]:10.4f} | {info['gnorms'][i]:9.4f} | "
              f"{info['step_means'][i]:8.4f}", flush=True)
    print(f"[diag] gnorm {info['gnorm0']:.3f} -> {info['gnorm']:.3f} "
          f"(ratio {info['gnorm']/info['gnorm0']:.4f}, converged={info['converged']})")

    # ---- per-term gradients at exit ----
    ue = u.detach().requires_grad_(True)
    E_el, lv, lb, lr = parts(ue)
    terms = {"elastic": E_el, "d_vol": lv, "box": cfg.w_box * lb, "lam_d_render": lam_r * lr}
    g_terms = {}
    keys = list(terms)
    for i, k in enumerate(keys):
        g_terms[k] = torch.autograd.grad(terms[k], ue, retain_graph=i < len(keys) - 1)[0]
    print("[diag] exit gradient by term: " +
          "  ".join(f"{k}={float(g.norm()):.4f}" for k, g in g_terms.items()), flush=True)

    # ---- smooth/rough decomposition of the exit gradient ----
    g = sum(g_terms.values())
    out_rows = []
    for it in (1, 3, 8, 20):
        gs = smooth_on_grid(qs, g, it)
        cos = float((g * gs).sum() / (g.norm() * gs.norm() + 1e-30))
        frac = float(gs.norm() / g.norm())
        out_rows.append({"jacobi_iters": it, "cos": cos, "norm_ratio": frac})
        print(f"[diag] smooth({it:2d} iters): cos(g, smooth g)={cos:.4f}  "
              f"|smooth g|/|g|={frac:.4f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(f"{args.out}.json").write_text(json.dumps({
        "n": args.n, "nodes": qs.A, "lam_r": lam_r, "gnorm0": info["gnorm0"],
        "gnorm": info["gnorm"], "converged": info["converged"],
        "energy": E_tr, "gnorms": info["gnorms"], "step_means": info["step_means"],
        "g_terms": {k: float(v.norm()) for k, v in g_terms.items()},
        "smooth": out_rows}))
    print(f"saved {args.out}.json", flush=True)


if __name__ == "__main__":
    main()
