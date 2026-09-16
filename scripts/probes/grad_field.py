"""Gradient-field probe: how much of the body the render loss reaches, how large the
gradients are, and where (heatmaps). For an archived run, at selected commits:

  g_r(x)   = dD_render/dx        the image loss's direct pull on particle positions
  g_v(x)   = dD_vol/dx           the mass-matching pull (density units when the run used them)
  g_r(dFc) = dD_render(x_T)/dFc  the same image loss routed THROUGH the MPM adjoint over one
                                 window (T steps, dFc = 0, v = C = 0 at the window start):
                                 which particles' CONTROL the image gradient reaches

Reports, per commit: the share of all / SURFACE / INTERIOR particles with |g| above a floor
(1e-3 of the max), magnitude percentiles, and the reach of g_r(dFc) by depth below the
surface (deciles of the distance to the nearest surface particle). Renders heatmaps
(log10 |g|, two views) with the gallery splat and writes a JSON.
Usage: python grad_field.py OUT_DIR run [frac1,frac2,...]   (default 0.05,0.35,1.0)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physmorph.losses.volumetric import d_vol, d_vol_density  # noqa: E402
from physmorph.mpm import MPMParams  # noqa: E402
from physmorph.mpm.constitutive import lame  # noqa: E402
from physmorph.mpm.function import RolloutSpec, warp_mpm_ext  # noqa: E402
from physmorph.mpm.traj import compute_rest_volumes  # noqa: E402
from physmorph.pipeline.config import PipelineConfig  # noqa: E402
from physmorph.pipeline.render_loss import d_render  # noqa: E402
from physmorph.pipeline.runner import build_target  # noqa: E402
from scripts.make_gif import splat  # noqa: E402

OUT, name = sys.argv[1], sys.argv[2]
FRACS = [float(f) for f in (sys.argv[3] if len(sys.argv) > 3 else "0.05,0.35,1.0").split(",")]
DEV = "cuda" if torch.cuda.is_available() else "cpu"

z = np.load(os.path.join(OUT, name + ".npz"))
fr, tgt, src = z["frames"], z["tgt"], z["src"]
Ff = z["F_frames"] if "F_frames" in z.files else None
dn = int(z["deliver_n"]) if "deliver_n" in z.files else len(fr)
jp = os.path.join(OUT, name.split("_render_")[0].split("_phys_")[0] + ".json")
prov = json.load(open(jp))["provenance"] if os.path.exists(jp) else {}
args = {k: v for k, v in prov.items() if k != "mpm"}
mpm = prov.get("mpm", {})
prm = MPMParams(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in mpm.items()}) if mpm else MPMParams()
cfg = PipelineConfig()
cfg.loss_res = int(args.get("loss_res", cfg.loss_res))
cfg.loss_units = args.get("loss_units", cfg.loss_units)
cfg.lambda_auto = 0.5                      # build the silhouettes even for a physics-only run
cfg.render_res = int(args.get("render_res", cfg.render_res))
cfg.unit_ref_res = int(args.get("unit_ref_res", cfg.unit_ref_res))
cfg.device = DEV
if cfg.loss_units == "density" and "ppc" in args and float(args["ppc"]) > 0:
    from physmorph.mpm.discretisation import derive
    v_src = float(args.get("_v_src", 0.0)) or None
    # loss_res followed dx in density units: reproduce it from the archived prm
    cfg.loss_res = int(round((prm.grid_min[0] * -2) / prm.dx))
pack = build_target(tgt, prm, cfg)
lam, mu = lame(cfg.young, cfg.poisson)
print(f"{name}: N={len(src)} dx={prm.dx} loss_res={cfg.loss_res} units={cfg.loss_units} render_res={cfg.render_res} "
      f"views={len(pack.views)} device={DEV}")


def surface_mask(x, k=24, thr=0.32):
    """kNN one-sidedness: |mean(neighbour offsets)| / mean distance > thr -> surface."""
    d, nb = cKDTree(x).query(x, k=k + 1, workers=-1)
    off = x[nb[:, 1:]] - x[:, None, :]
    score = np.linalg.norm(off.mean(1), axis=1) / (d[:, 1:].mean(1) + 1e-9)
    return score > thr


def render_heat(x, val, res, az, el, extent):
    """Per-pixel max of val over the splatted particles (NaN where empty)."""
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    right = np.array([ca, 0.0, -sa], np.float32)
    up = np.array([-sa * se, ce, -ca * se], np.float32)
    p = np.stack([x @ right, x @ up], 1)
    ij = np.floor((p + extent) / (2 * extent) * res).astype(np.int64)
    ok = (ij >= 0).all(1) & (ij < res).all(1)
    img = np.full((res, res), np.nan, np.float32)
    v = val[ok]
    order = np.argsort(v)                       # last write wins -> max over a 3x3 footprint
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            i2 = np.clip(ij[ok, 0] + ox, 0, res - 1)
            j2 = np.clip(ij[ok, 1] + oy, 0, res - 1)
            flat = i2 * res + j2
            cur = img.reshape(-1)[flat[order]]
            upd = np.where(np.isnan(cur), v[order], np.maximum(cur, v[order]))
            np.put(img.reshape(-1), flat[order], upd)
    return img.T[::-1]


def stats(g, surf, floor_rel=1e-3):
    n = np.linalg.norm(g.reshape(len(g), -1), axis=1)
    floor = floor_rel * n.max() if n.max() > 0 else 0.0
    act = n > floor
    q = np.percentile(n, [50, 90, 99, 100])
    return dict(active_all=float(act.mean()), active_surface=float(act[surf].mean()),
                active_interior=float(act[~surf].mean()), mean_surface=float(n[surf].mean()),
                mean_interior=float(n[~surf].mean()), p50=float(q[0]), p90=float(q[1]),
                p99=float(q[2]), max=float(q[3]), floor=float(floor)), n


results = {"run": name, "frames": []}
heat = []
T = int(args.get("T", 20))
ext = float(np.abs(tgt).max()) * 1.08
for f in FRACS:
    k = min(dn - 1, int(round(f * (dn - 1))))
    x = np.ascontiguousarray(fr[k], np.float32)
    surf = surface_mask(x)
    depth = cKDTree(x[surf]).query(x, k=1, workers=-1)[0]
    xt = torch.as_tensor(x, device=DEV).requires_grad_(True)
    Lr = d_render(xt, pack.sils, pack.views, cfg.render_res, pack.extent, cfg.sil_k)
    (gr,) = torch.autograd.grad(Lr, xt)
    if cfg.loss_units == "density":
        Lv = d_vol_density(xt, pack.m, pack.grid, pack.lgmin, pack.ldx, pack.ldims, pack.m_ref, pack.n_support)
    else:
        Lv = d_vol(xt, pack.m, pack.grid, pack.lgmin, pack.ldx, pack.ldims)
    (gv,) = torch.autograd.grad(Lv, xt)
    gr, gv = gr.cpu().numpy(), gv.cpu().numpy()
    # through the adjoint: dD_render(x_T)/dFc over one window from this state
    F0 = np.ascontiguousarray(Ff[k], np.float32) if Ff is not None and k < len(Ff) else None
    vol0 = compute_rest_volumes(src, 1.0, prm, DEV)
    spec = RolloutSpec(x0=x, m=1.0, lam=lam, mu=mu, prm=prm, T=T, device=DEV, vol0=vol0, F0=F0)
    dfc = torch.zeros(T, len(x), 3, 3, device=DEV, requires_grad=True)
    xT, FT, vT, FgT, V = warp_mpm_ext(dfc, spec)
    LrT = d_render(xT, pack.sils, pack.views, cfg.render_res, pack.extent, cfg.sil_k)
    (gc,) = torch.autograd.grad(LrT, dfc)
    gc = gc.abs().amax(0).cpu().numpy()          # (N,3,3) max over the window
    sr, nr = stats(gr, surf)
    sv, nv = stats(gv, surf)
    sc, nc = stats(gc, surf)
    # reach by depth decile
    dq = np.quantile(depth, np.linspace(0, 1, 11))
    reach = []
    for a, b in zip(dq[:-1], dq[1:]):
        m = (depth >= a) & (depth <= b)
        reach.append(dict(depth_lo=float(a), depth_hi=float(b), share_active=float((nc[m] > sc["floor"]).mean()),
                          mean=float(nc[m].mean())))
    rec = dict(frac=f, frame=k, surface_share=float(surf.mean()), L_render=float(Lr), L_vol=float(Lv),
               L_render_T=float(LrT), g_render_x=sr, g_vol_x=sv, g_render_dfc=sc, reach_by_depth=reach)
    results["frames"].append(rec)
    print(f"  frame {k} ({f:.2f}): surface {surf.mean():.1%} | g_render(x): active all {sr['active_all']:.1%} surface {sr['active_surface']:.1%} interior {sr['active_interior']:.1%}, "
          f"mean surf/int {sr['mean_surface']:.2e}/{sr['mean_interior']:.2e} | g_vol(x): active all {sv['active_all']:.1%} surface {sv['active_surface']:.1%}, mean surf/int {sv['mean_surface']:.2e}/{sv['mean_interior']:.2e} | "
          f"g_render(dFc): active all {sc['active_all']:.1%} surface {sc['active_surface']:.1%} interior {sc['active_interior']:.1%}; reach deciles "
          + " ".join(f"{r['share_active']:.2f}" for r in reach))
    heat.append((k, x, nr, nv, nc))

json.dump(results, open(os.path.join(OUT, f"grad_field_{name}.json"), "w"), indent=1)
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [("|dD_render/dx|", 2), ("|dD_vol/dx|", 3), ("|dD_render/dFc| (adjoint, max over window)", 4)]
    views = [(0.6, 0.18), (2.2, 0.18)]
    fig, axes = plt.subplots(len(heat) * len(views), len(cols), figsize=(4.6 * len(cols), 4.2 * len(heat) * len(views)))
    axes = np.atleast_2d(axes)
    RES = 260
    for hi, (k, x, nr, nv, nc) in enumerate(heat):
        for vi, (az, el) in enumerate(views):
            for ci, (title, idx) in enumerate(cols):
                val = (nr, nv, nc)[ci]
                lv = np.log10(val + 1e-12 * max(val.max(), 1e-30))
                img = render_heat(x, lv, RES, az, el, ext)
                ax = axes[hi * len(views) + vi, ci]
                im = ax.imshow(img, cmap="inferno", vmin=np.nanpercentile(lv, 1), vmax=lv.max())
                ax.set_title(f"frame {k} · az {az} · {title}", fontsize=9); ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.04, label="log10 |g|")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, f"grad_field_{name}.png"), dpi=100)
    print("figure saved")
except Exception as e:
    print("no figure:", e)
