"""Step 0 — what does the EXISTING morph_mass actually produce, with and without render guidance?

physmorph.morph.morph_mass optimises the deformation-gradient control field dFc per frame with

    L = D_vol  +  render_lambda * D_img  +  cohesion

where D_vol is the volumetric mass-matching objective (the Xu et al. loss) and D_img is the
multi-view soft silhouette. render_lambda = 0 is therefore the physics/3-D-only baseline and
render_lambda > 0 is the render-guided arm, in ONE code path — no reimplementation, no separate
package. This measures both with the same metrics so the comparison is honest.

Metrics (all from raw simulation state; the renderer is never consumed):
  chamfer      symmetric mean nearest-neighbour distance to the target point set
  sil_iou      multi-view silhouette IoU at a higher resolution than the loss uses
  detF_min     minimum det F over the returned frames (inversion check)
  d_vol        the optimiser's own objective, final value

Run (hyde06):
  CUDA_VISIBLE_DEVICES=0 python scripts/morph_measure.py --lambdas 0,0.5 --out output/step0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from physmorph.losses.silhouette import (d_img, ring_thetas, soft_silhouette,  # noqa: E402
                                         target_silhouettes)
from physmorph.experimental.morph import morph_mass  # noqa: E402  (v1, quarantined)
from physmorph.mpm import MPMParams  # noqa: E402
from physmorph.sampling import load_mesh, sample_volume  # noqa: E402


def load(path, n, seed=1):
    x = sample_volume(load_mesh(path), n, seed=seed).astype(np.float32)
    x -= x.mean(0)
    return (x * (8.0 / (np.linalg.norm(x.max(0) - x.min(0)) + 1e-9))).astype(np.float32)


def chamfer(a, b):
    da = cKDTree(b).query(a, k=1, workers=-1)[0]
    db = cKDTree(a).query(b, k=1, workers=-1)[0]
    return float(da.mean() + db.mean())


def sil_iou(x, tgt, views=8, res=128, thr=0.5, device="cuda"):
    """Multi-view silhouette IoU. Measured at higher res than the loss grid so it is not the
    quantity being optimised (the loss uses render_res, default 64, and MSE not IoU)."""
    xt = torch.tensor(x, device=device)
    tt = torch.tensor(tgt, device=device)
    extent = float(np.abs(tgt).max()) * 1.25
    ious = []
    with torch.no_grad():
        for th in ring_thetas(views):
            a = soft_silhouette(xt, float(th), res, extent) > thr
            b = soft_silhouette(tt, float(th), res, extent) > thr
            u = (a | b).sum().item()
            ious.append(((a & b).sum().item() / u) if u else 1.0)
    return float(np.mean(ious))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="assets/isosphere.obj")
    ap.add_argument("--tgt", default="assets/bunny.obj")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--ntgt", type=int, default=30000)
    ap.add_argument("--K", type=int, default=30, help="morph frames")
    ap.add_argument("--T", type=int, default=15, help="MPM steps per frame (taped)")
    ap.add_argument("--inner", type=int, default=4, help="Adam iters per frame")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--render_views", type=int, default=6)
    ap.add_argument("--render_res", type=int, default=64)
    ap.add_argument("--cohesion_w", type=float, default=8.0, help="kNN stray penalty (anti-ejection)")
    ap.add_argument("--dfc_clip", type=float, default=0.04)
    ap.add_argument("--loss_res", type=int, default=32, help="D_vol grid res; domain is 32 units, so 32 -> 1.0 units/cell")
    ap.add_argument("--lambdas", default="0,0.5", help="render_lambda values to compare")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="output/step0")
    args = ap.parse_args()

    # morph_mass builds ONE unit-mass vector sized by the SOURCE and reuses it to rasterize the
    # TARGET (morph.py:85,99), so the two clouds must have the same particle count. That is not an
    # accident: D_vol compares mass grids, and unequal counts of unit-mass particles would compare
    # clouds of different total mass. Honour the constraint rather than patching the loss.
    if args.ntgt != args.n:
        print(f"[step0] note: ntgt {args.ntgt} -> {args.n} (D_vol needs equal particle counts)",
              flush=True)
        args.ntgt = args.n
    src = load(args.src, args.n, args.seed)
    tgt = load(args.tgt, args.ntgt, args.seed + 1)
    prm = MPMParams()                      # morphing line keeps the legacy F-smoothing (0.955)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"[step0] src={args.src} ({len(src)}) -> tgt={args.tgt} ({len(tgt)}) | "
          f"K={args.K} T={args.T} inner={args.inner} lr={args.lr} | dx={prm.dx} "
          f"smoothing={prm.smoothing}", flush=True)
    print(f"[step0] baseline chamfer (undeformed sphere vs target) = {chamfer(src, tgt):.4f}",
          flush=True)

    out = {"provenance": vars(args), "arms": {}}
    for lam in [float(s) for s in args.lambdas.split(",")]:
        tag = f"lam{lam:g}"
        print(f"\n[step0] ===== ARM render_lambda={lam:g} "
              f"({'3-D / mass only (Xu et al. objective)' if lam == 0 else 'render-guided'}) =====",
              flush=True)
        t0 = time.time()
        frames, Fs, hist = morph_mass(src, tgt, prm, T=args.T, K=args.K, inner=args.inner,
                                      lr=args.lr, render_lambda=lam,
                                      render_views=args.render_views,
                                      render_res=args.render_res, loss_res=args.loss_res,
                                      cohesion_w=args.cohesion_w, dfc_clip=args.dfc_clip,
                                      save_F=True,
                                      promote_F=True)   # keep F across frames or detF is trivially 1
        dt = time.time() - t0
        xf = frames[-1]
        detf = min(float(np.linalg.det(F).min()) for F in Fs)
        # the two loss TERMS at the final state, in their own units — this is what decides whether
        # render_lambda is doing anything at all (a term 3 orders below the other is inert)
        with torch.no_grad():
            xt = torch.tensor(xf, device="cuda")
            mt = torch.ones(len(xf), device="cuda")
            extent = float(np.abs(tgt).max()) * 1.25
            thetas = ring_thetas(args.render_views)
            tsil = target_silhouettes(torch.tensor(tgt, device="cuda"), thetas,
                                      args.render_res, extent)
            dimg = float(d_img(xt, tsil, thetas, args.render_res, extent).item())
        rec = {"chamfer": chamfer(xf, tgt), "sil_iou": sil_iou(xf, tgt),
               "d_img_final": dimg, "lambda_times_d_img": lam * dimg,
               "d_vol_final": hist[-1]["d_vol"] if hist else None,
               "d_vol_hist": [h["d_vol"] for h in hist],
               "move_hist": [h["move"] for h in hist],
               "detF_min": detf, "frames": len(frames), "seconds": dt,
               "converged_at": next((h["frame"] for h in hist if h.get("held")), None)}
        out["arms"][tag] = rec
        np.savez_compressed(f"{args.out}_{tag}.npz", src=src, tgt=tgt,
                            frames=np.stack(frames), Fs=np.stack(Fs))
        print(f"[step0] ARM {tag}: chamfer={rec['chamfer']:.4f}  silIoU={rec['sil_iou']:.4f}  "
              f"D_vol={rec['d_vol_final']:.2f}  D_img={dimg:.5f}  lam*D_img={lam*dimg:.4f}  "
              f"detFmin={detf:.4f}  ({dt/60:.1f} min)", flush=True)

    Path(f"{args.out}.json").write_text(json.dumps(out))
    print("\n[step0] SUMMARY")
    for tag, r in out["arms"].items():
        print(f"  {tag:8s} chamfer={r['chamfer']:.4f}  silIoU={r['sil_iou']:.4f}  "
              f"D_vol={r['d_vol_final']:.2f}  D_img={r['d_img_final']:.5f}  "
              f"lam*D_img={r['lambda_times_d_img']:.4f}  detFmin={r['detF_min']:.4f}", flush=True)
    print(f"saved {args.out}.json", flush=True)


if __name__ == "__main__":
    main()
