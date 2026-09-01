"""Gradient-flow analysis of the v2 pipeline (feeds docs/gradient_analysis.md).

At representative commits along a live morph, measures AT THE WINDOW-START DECISION POINT
(dFc = 0, where the balancer acts):

  1. per-term gradient norms on dFc through the MPM adjoint: D_vol, D_render, kin, box —
     and the two weighting policies on the same numbers:
       v1 (PhysMorph-GS): fixed lambda -> contribution ratio lam*|g_ren|/|g_vol| (the
       "render term is inert" number),
       v2: lambda = a*|g_phys|/|g_ren| (norm-balanced, EMA) -> contribution `a` by design;
  2. render-vs-physics gradient ALIGNMENT (cosine on dFc): does render agree, orthogonally
     complement, or fight the mass objective?
  3. surface concentration of the image-space pull |dD_render/dx_T| vs local density decile
     (the "render feedback is surface-dominant" claim, quantified) — contrasted with the
     volumetric pull |dD_vol/dx_T|;
  4. F-field health along the run: singular-value anisotropy percentiles, det F, plus
     ejection telemetry (outside_frac) — with ablation arms --w_box 0 and --assim 0.

Run (hyde06, idle GPU):
  CUDA_VISIBLE_DEVICES=1 python scripts/grad_analysis.py --out output/grad_analysis
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from physmorph.losses.volumetric import d_vol  # noqa: E402
from physmorph.mpm import MPMParams  # noqa: E402
from physmorph.mpm.conditioning import condition_F  # noqa: E402
from physmorph.mpm.constitutive import lame  # noqa: E402
from physmorph.mpm.function import RolloutSpec, warp_mpm_full  # noqa: E402
from physmorph.pipeline.config import PipelineConfig  # noqa: E402
from physmorph.pipeline.optimizer import optimize_window  # noqa: E402
from physmorph.pipeline.render_loss import LambdaBalancer, d_render  # noqa: E402
from physmorph.pipeline.runner import build_target  # noqa: E402
from physmorph.plasticity import assimilate_elastic  # noqa: E402
from physmorph.sampling import load_normalized  # noqa: E402


def _id(n):
    return np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))


def _norm(g):
    return float(g.norm().item())


def f_stats(F):
    S = np.linalg.svd(F.reshape(-1, 3, 3), compute_uv=False)
    aniso = S[:, 0] / np.maximum(S[:, 2], 1e-9)
    det = np.linalg.det(F.reshape(-1, 3, 3))
    return {"aniso_p50": float(np.percentile(aniso, 50)),
            "aniso_p95": float(np.percentile(aniso, 95)),
            "aniso_max": float(aniso.max()),
            "detF_min": float(det.min()), "detF_p05": float(np.percentile(det, 5)),
            "detF_mean": float(det.mean())}


def probe(x0, st, Fp, prm, cfg, tgt, commit):
    """Per-term gradients at the window-start decision point (dFc = 0)."""
    dev = cfg.device
    N, T = len(x0), cfg.T
    lam0, mu0 = lame(cfg.young, cfg.poisson)
    spec = RolloutSpec(x0=x0, m=1.0, lam=lam0, mu=mu0, prm=prm, T=T,
                       F0=st["F"], Fp=Fp, v0=st["v"], C0=st["C"], device=dev)
    dFc = torch.zeros(T, N, 3, 3, device=dev, requires_grad=True)
    xT, FT, vT = warp_mpm_full(dFc, spec)
    lv = d_vol(xT, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims)
    lr = d_render(xT, tgt.sils, tgt.views, cfg.render_res, tgt.extent,
                  cfg.sil_k, cfg.w_hole, cfg.w_spray)
    lk = vT.pow(2).sum(1).mean()
    lbox = torch.clamp(xT.abs() - tgt.extent, min=0).pow(2).sum(1).mean()

    # image-space pulls (before the MPM adjoint) — per-particle
    gx_ren = torch.autograd.grad(lr, xT, retain_graph=True)[0]
    gx_vol = torch.autograd.grad(lv, xT, retain_graph=True)[0]
    # control-space gradients (through the T-step MPM adjoint)
    g_vol = torch.autograd.grad(lv, dFc, retain_graph=True)[0]
    g_ren = torch.autograd.grad(lr, dFc, retain_graph=True)[0]
    g_kin = torch.autograd.grad(lk, dFc, retain_graph=True)[0]
    g_box = torch.autograd.grad(lbox, dFc)[0]

    n_vol, n_ren, n_kin, n_box = map(_norm, (g_vol, g_ren, g_kin, g_box))
    g_phys = g_vol + cfg.w_kin * g_kin + cfg.w_box * g_box
    n_phys = _norm(g_phys)
    cos = float((g_vol.flatten() @ g_ren.flatten()
                 / (g_vol.norm() * g_ren.norm() + 1e-30)).item())

    # surface concentration: local kNN radius decile vs mean |image-space pull|
    x_np = xT.detach().cpu().numpy()
    r = cKDTree(x_np).query(x_np, k=17)[0][:, 1:].mean(1)     # local spacing
    gren_mag = gx_ren.norm(dim=1).cpu().numpy()
    gvol_mag = gx_vol.norm(dim=1).cpu().numpy()
    hi = r >= np.percentile(r, 90)                             # sparse = surface band
    lo = r <= np.percentile(r, 10)                             # dense = interior

    lam_fixed_v1 = 0.5                                         # v1 morph_mass default
    return {
        "commit": commit,
        "loss": {"d_vol": float(lv), "d_render": float(lr), "kin": float(lk),
                 "box": float(lbox)},
        "grad_norm_dFc": {"d_vol": n_vol, "d_render": n_ren, "kin": n_kin, "box": n_box,
                          "phys_total": n_phys},
        "v1_fixed_lambda_contrib": lam_fixed_v1 * n_ren / max(n_vol, 1e-30),
        "v2_lambda": cfg.lambda_auto * n_phys / max(n_ren, 1e-30),
        "v2_contrib_by_construction": cfg.lambda_auto,
        "cos_render_vs_dvol": cos,
        "surface_conc_render": float(gren_mag[hi].mean() / max(gren_mag[lo].mean(), 1e-30)),
        "surface_conc_dvol": float(gvol_mag[hi].mean() / max(gvol_mag[lo].mean(), 1e-30)),
        "gx_ren_absmax": float(gren_mag.max()), "gx_vol_absmax": float(gvol_mag.max()),
    }


def run_arm(src, tgt_x, prm, cfg, probe_at, log=print):
    """Runner-mirror loop with probe injection; returns (probes, per-commit stats)."""
    N = len(src)
    tgt = build_target(tgt_x, prm, cfg)
    balancer = LambdaBalancer(cfg.lambda_auto if cfg.lambda_auto > 0 else 0.5,
                              cfg.lambda_ema)
    balancer.alpha_lam = cfg.lambda_auto            # arm may be phys-only; pack has sils
    x, st, Fp, s = src.copy(), {"F": None, "v": None, "C": None}, _id(N), None
    probes, commits = [], []
    for a in range(cfg.animations):
        if a in probe_at:
            probes.append(probe(x, st, Fp, prm, cfg, tgt, a))
            log(f"  probe @ commit {a}: done")
        fr, F_seq, end, s, whist, stats = optimize_window(
            x, prm, cfg, tgt, balancer, F0=st["F"], Fp=Fp, v0=st["v"], C0=st["C"],
            s_init=s, log=lambda *_: None)
        if not whist:
            log(f"  arm stopped at commit {a} (no accepted step)")
            break
        x = np.ascontiguousarray(fr[-1], np.float32)
        Fc, nb, nf, _ = condition_F(end["F"], clamp=False)
        st = {"F": Fc, "v": end["v"], "C": end["C"]}
        if cfg.assim > 0:
            Fp = assimilate_elastic(Fc, Fp, eta=cfg.assim,
                                    smin=cfg.assim_smin, smax=cfg.assim_smax)
        out_frac = float((np.abs(x) > tgt.extent).any(1).mean())
        rec = {"commit": a, "kin": whist[-1]["kin"], "d_vol": whist[-1]["d_vol"],
               "outside_frac": out_frac, "F_reset": nb, "F_flip": nf, **f_stats(Fc)}
        commits.append(rec)
    return probes, commits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="assets/isosphere.obj")
    ap.add_argument("--tgt", default="assets/bunny.obj")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--animations", type=int, default=10)
    ap.add_argument("--probe_at", default="0,3,6,9")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="output/grad_analysis")
    args = ap.parse_args()

    src = load_normalized(args.src, args.n, args.seed)
    tgt = load_normalized(args.tgt, args.n, args.seed + 1)
    prm = MPMParams()
    probe_at = {int(s) for s in args.probe_at.split(",")}
    base = dict(T=args.T, iters=args.iters, animations=args.animations,
                lambda_auto=0.5, w_kin=5.0)

    arms = {
        "default": PipelineConfig(**base),
        "no_box": PipelineConfig(**base, w_box=0.0),
        "no_assim": PipelineConfig(**base, assim=0.0),
    }
    out = {"provenance": {**vars(args), "dx": prm.dx, "dt": prm.dt,
                          "smoothing": prm.smoothing}}
    for name, cfg in arms.items():
        print(f"[grad] ===== arm {name} =====", flush=True)
        probes, commits = run_arm(src, tgt, prm, cfg,
                                  probe_at if name == "default" else set())
        out[name] = {"probes": probes, "commits": commits}
        if commits:
            c = commits[-1]
            print(f"[grad] {name}: commits={len(commits)}  kin_last={c['kin']:.4f}  "
                  f"outside={c['outside_frac']*100:.3f}%  anisoP95={c['aniso_p95']:.3f}  "
                  f"detFmin={c['detF_min']:.4f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(f"{args.out}.json").write_text(json.dumps(out))
    print(f"saved {args.out}.json", flush=True)

    for p in out["default"]["probes"]:
        print(f"[probe c{p['commit']}] |g_vol|={p['grad_norm_dFc']['d_vol']:.3g} "
              f"|g_ren|={p['grad_norm_dFc']['d_render']:.3g} "
              f"v1_contrib={p['v1_fixed_lambda_contrib']:.2e} "
              f"v2_lam={p['v2_lambda']:.3g} cos={p['cos_render_vs_dvol']:.3f} "
              f"surfconc_ren={p['surface_conc_render']:.2f} "
              f"surfconc_vol={p['surface_conc_dvol']:.2f}", flush=True)


if __name__ == "__main__":
    main()
