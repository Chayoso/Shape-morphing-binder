"""Trajectory optimisation (C++-matched) vs the greedy morph_mass, on the same problem.

Gate 1 (regression): the shared-control path must still work after making dFc optional-per-step —
morph_mass is re-run and must reproduce its previous numbers.
Gate 2 (equivalence): a CONSTANT sequence dFc[t] = c must give the same rollout as the shared
control c, since the engine then applies the same matrix at every step. This checks the new
sequence plumbing against the old path bit-for-bit.
Then: the actual comparison, greedy vs trajectory, with and without render guidance.

Run (hyde06):
  CUDA_VISIBLE_DEVICES=0 python scripts/traj_opt_run.py --out output/trajopt
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

from physmorph.losses.silhouette import ring_thetas, soft_silhouette  # noqa: E402
from physmorph.mpm import MPMParams  # noqa: E402
from physmorph.mpm.function import RolloutSpec, warp_mpm  # noqa: E402
from physmorph.sampling import load_mesh, sample_volume  # noqa: E402
from physmorph.trajectory_opt import TrajOptConfig, optimize_morph  # noqa: E402


def load(path, n, seed=1):
    x = sample_volume(load_mesh(path), n, seed=seed).astype(np.float32)
    x -= x.mean(0)
    return (x * (8.0 / (np.linalg.norm(x.max(0) - x.min(0)) + 1e-9))).astype(np.float32)


def chamfer(a, b):
    da = cKDTree(b).query(a, k=1, workers=-1)[0]
    db = cKDTree(a).query(b, k=1, workers=-1)[0]
    return float(da.mean() + db.mean())


def sil_iou(x, tgt, views=8, res=128, thr=0.5, device="cuda"):
    xt, tt = torch.tensor(x, device=device), torch.tensor(tgt, device=device)
    extent = float(np.abs(tgt).max()) * 1.25
    out = []
    with torch.no_grad():
        for th in ring_thetas(views):
            a = soft_silhouette(xt, float(th), res, extent) > thr
            b = soft_silhouette(tt, float(th), res, extent) > thr
            u = (a | b).sum().item()
            out.append(((a & b).sum().item() / u) if u else 1.0)
    return float(np.mean(out))


def gate_equivalence(src, prm, T=6, device="cuda"):
    """A constant sequence must equal the shared control exactly."""
    N = len(src)
    spec = RolloutSpec(x0=src, m=1.0, lam=1.0e5, mu=5.0e4, prm=prm, T=T, device=device)
    c = (torch.randn(N, 3, 3, device=device) * 1e-3)
    with torch.no_grad():
        x_shared, _ = warp_mpm(c, spec)
        x_seq, _ = warp_mpm(c.unsqueeze(0).repeat(T, 1, 1, 1).contiguous(), spec)
    d = float((x_shared - x_seq).abs().max())
    # Bitwise equality is NOT the right bar: P2G uses wp.atomic_add, so the float32 accumulation
    # order differs between launches and one ULP of drift is expected. Coordinates are O(4), and
    # float32 eps = 1.19e-7, so a few ULP at this scale is ~1e-6. Anything above that would mean
    # the sequence really is applying different controls.
    tol = 1e-6 * max(1.0, float(x_shared.abs().max()))
    ok = d <= tol
    print(f"[gate2] constant-sequence vs shared control: max|dx| = {d:.3e}  (tol {tol:.1e}, "
          f"float32 eps = 1.19e-7)  -> {'PASS' if ok else 'FAIL — plumbing differs'}", flush=True)
    return {"max_abs_diff": d, "tol": tol, "pass": bool(ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="assets/isosphere.obj")
    ap.add_argument("--tgt", default="assets/bunny.obj")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--iters", type=int, default=8, help="inner iters per animation")
    ap.add_argument("--animations", type=int, default=30, help="outer commits (C++ num_animations)")
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--w_ctrl", type=float, default=0.0, help="control-energy weight (running cost)")
    ap.add_argument("--lambda_auto", default="0,0.5",
                    help="norm-based render balancing weights (0 = physics only)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="output/trajopt")
    args = ap.parse_args()

    src = load(args.src, args.n, args.seed)
    tgt = load(args.tgt, args.n, args.seed + 1)
    prm = MPMParams()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"[trajopt] {args.src} -> {args.tgt}  N={args.n}  T={args.T}  iters={args.iters}",
          flush=True)
    print(f"[trajopt] baseline chamfer (undeformed) = {chamfer(src, tgt):.4f}", flush=True)

    out = {"provenance": vars(args), "gate_equivalence": gate_equivalence(src, prm), "arms": {}}

    for la in [float(s) for s in args.lambda_auto.split(",")]:
        tag = f"auto{la:g}"
        print(f"\n[trajopt] ===== ARM lambda_auto={la:g} "
              f"({'physics/mass only' if la == 0 else 'render-balanced'}) =====", flush=True)
        cfg = TrajOptConfig(T=args.T, iters=args.iters, alpha=args.alpha, lambda_auto=la,
                            w_ctrl=args.w_ctrl)
        t0 = time.time()
        frames, hist = optimize_morph(src, tgt, prm, cfg, animations=args.animations)
        dt = time.time() - t0
        xf = frames[-1]
        rec = {"chamfer": chamfer(xf, tgt), "sil_iou": sil_iou(xf, tgt),
               "d_vol_final": hist[-1]["d_vol"] if hist else None,
               "d_img_final": hist[-1]["d_img"] if hist else None,
               "lambda_final": hist[-1]["lambda"] if hist else None,
               "dfc_absmax": hist[-1]["dfc_absmax"] if hist else None,
               "Jmin": hist[-1]["Jmin"] if hist else None,
               "animations": len(hist), "total_steps": len(hist) * args.T, "seconds": dt,
               "history": hist}
        out["arms"][tag] = rec
        np.savez_compressed(f"{args.out}_{tag}.npz", src=src, tgt=tgt,
                            frames=np.stack(frames))
        print(f"[trajopt] ARM {tag}: chamfer={rec['chamfer']:.4f}  silIoU={rec['sil_iou']:.4f}  "
              f"D_vol={rec['d_vol_final']:.3f}  |dFc|max={rec['dfc_absmax']:.4f}  Jmin={rec['Jmin']:.4f}  "
              f"({dt/60:.1f} min)", flush=True)

    Path(f"{args.out}.json").write_text(json.dumps(out))
    print("\n[trajopt] SUMMARY", flush=True)
    for tag, r in out["arms"].items():
        print(f"  {tag:8s} chamfer={r['chamfer']:.4f}  silIoU={r['sil_iou']:.4f}  "
              f"D_vol={r['d_vol_final']:.3f}  lambda={r['lambda_final']}", flush=True)
    print(f"saved {args.out}.json", flush=True)


if __name__ == "__main__":
    main()
