"""load_equal.py OUT_JSON [N] [SEED] [MESH] -- B9b, the frozen check of assim_load at the window level (pre-registered
in docs/experiments.md, B9b, "The loads' own effect", before its runs). The frozen recipe's 40k bunny (seed 97) in the
scene's one-body window with no gravity and no floor, the world drag 0.9 (the frozen dynamics, as V0) and the frame
fixed at launch: two windows, the first from rest, the second from the first's committed end (moving and deformed), each
under random controls (1e-2 a step) committed once. Per window: the zero-control continuations with and without the
loads (window.zero_end_F, free_zero_end_F), the assim_load reading (runner.load_free_F) against the frozen reading (the
committed end F), and the assimilated Fp of both; the frozen reading's own replay (the window committed again from the
same start: CUDA atomics) as the rounding's scale. Pass: the two continuations and the two Fp agree within the larger of
1e-5 and the replay's own difference. CUDA; run on hyde06."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import physmorph  # noqa: E402,F401  (before torch)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from physmorph import gpu  # noqa: E402
from physmorph.mpm.traj import compute_rest_volumes  # noqa: E402
from physmorph.pipeline.scene.objective import SceneObjective  # noqa: E402
from physmorph.pipeline.scene.raster import BodyRaster  # noqa: E402
from physmorph.pipeline.scene.rollout import commit_rollout  # noqa: E402
from physmorph.pipeline.scene.runner import load_free_F  # noqa: E402
from physmorph.pipeline.scene.spec import SceneSpec, layout  # noqa: E402
from physmorph.pipeline.scene.window import Body, SceneWindow  # noqa: E402
from physmorph.pipeline.target import build_target, calibrate_units, target_relief  # noqa: E402
from physmorph.pipeline.window.setup import StartState  # noqa: E402
from physmorph.plasticity import assimilate_elastic  # noqa: E402
from scripts.pipeline_run import parse_args as run_args, prepared, run_config  # noqa: E402


def main():
    out = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 97
    mesh = sys.argv[4] if len(sys.argv) > 4 else "bunny"
    gpu.require_cuda()
    base = run_args(["--n", str(n), "--tgt", f"assets/{mesh}.obj", "--seed", str(seed), "--out", str(out), "--drag", "0"])
    cfg0, prep = prepared(base, seed)
    cfg = run_config(base, cfg0, prep)
    X = np.asarray(prep.src, np.float64)
    X = (X - X.mean(0)).astype(np.float32)
    x = gpu.tensor(X)
    tgt = build_target(prep.tgt, prep.prm, cfg, draws=prep.tgt_draws)
    if prep.tgt_surface is not None:
        tgt.relief = target_relief(tgt.pts, prep.tgt_surface, cfg)
    calibrate_units(tgt, x, cfg)
    vp = (prep.density["target_fill"] if prep.density else prep.v_src) / n
    spec = SceneSpec(bodies=1, rest=True, g=0.0, G=0.0, mu=0.0, yaw=(0.0, 0.0), floor=False, frame="fixed", drag=0.9,
                     seconds=8.33)
    _, _, _, prm, info = layout(spec, [X.astype(np.float64)], [vp], cfg.young, cfg.poisson, cfg.mass_ref_n,
                                prep.prm.dx, prep.prm.dt, cfg.T,
                                radius=float(np.linalg.norm(prep.tgt - prep.tgt.mean(0), axis=1).max()))
    vol0 = compute_rest_volumes(x, 1.0, prm, cfg.device)
    coh = gpu.knn(x, cfg.coh_k + 1)[1][:, 1:]
    bonds = (coh, (x[coh] - x[:, None]).norm(dim=2), torch.zeros(n, device=x.device))
    body = Body(start=0, stop=n, Xq=x, tgt=tgt, vp=vp, mass=cfg.mass_ref_n / n, cfg=cfg, raster=BodyRaster(prep.prm, cfg),
                fixed=(x.mean(0), torch.eye(3, device=x.device)))
    gen = torch.Generator().manual_seed(7)
    start = StartState(x=x, Fp=torch.eye(3, device=x.device).repeat(n, 1, 1))
    u = torch.zeros(n, device=x.device)
    rows = []
    for w in range(2):
        leaf = (1e-2 * torch.randn(cfg.T, n, 3, 3, generator=gen)).to(x.device)
        win = SceneWindow(start, prm, cfg, [body], vol0, [bonds], mu=0.0)
        obj = SceneObjective(win)
        c1 = commit_rollout(win, obj, leaf, u, 0.0)
        F_ctrl = c1.end_F.reshape(-1, 3, 3).clone()
        end = (c1.x[-1].clone(), c1.end_F.clone(), c1.end_v.clone(), c1.end_C.clone(),
               None if c1.end_J is None else c1.end_J.clone())
        c1 = None
        c2 = commit_rollout(win, obj, leaf, u, 0.0)                      # the replay: the rounding's scale
        F_rep = c2.end_F.reshape(-1, 3, 3).clone()
        c2 = None
        win.load(leaf, u)                                                # the committed controls in the buffers
        F_zl, F_zf = win.zero_end_F(), win.free_zero_end_F()
        Fp = start.Fp
        read = load_free_F(F_ctrl, F_zl, F_zf)
        Fp_load = assimilate_elastic(read, Fp, eta=cfg.assim, smin=cfg.assim_smin, smax=cfg.assim_smax,
                                     isochoric=not cfg.assim_volume)
        Fp_frz = assimilate_elastic(F_ctrl, Fp, eta=cfg.assim, smin=cfg.assim_smin, smax=cfg.assim_smax,
                                    isochoric=not cfg.assim_volume)
        Fp_rep = assimilate_elastic(F_rep, Fp, eta=cfg.assim, smin=cfg.assim_smin, smax=cfg.assim_smax,
                                    isochoric=not cfg.assim_volume)
        r = {"window": w + 1, "moving_start": w > 0,
             "start_speed_rms": 0.0 if start.v is None else float(start.v.norm(dim=1).square().mean().sqrt()),
             "zl_vs_zf": float((F_zl - F_zf).abs().max()), "zl_vs_I": float((F_zl - torch.eye(3, device=x.device)).abs().max()),
             "read_vs_frozen": float((read - F_ctrl).abs().max()), "replay_F": float((F_rep - F_ctrl).abs().max()),
             "Fp_load_vs_frozen": float((Fp_load - Fp_frz).abs().max()), "Fp_replay": float((Fp_rep - Fp_frz).abs().max()),
             "Fp_step": float((Fp_frz - Fp).abs().max())}
        tol = max(1e-5, r["Fp_replay"])
        r["pass"] = r["zl_vs_zf"] <= max(1e-5, r["replay_F"]) and r["Fp_load_vs_frozen"] <= tol
        rows.append(r)
        print(json.dumps(r), flush=True)
        xe, Fe, ve, Ce, Je = end
        start = StartState(x=xe, Fp=Fp_frz, F=Fe.reshape(-1, 3, 3), v=ve, C=Ce, J=Je)
        win = obj = None
        torch.cuda.empty_cache()
    res = {"n": n, "seed": seed, "mesh": mesh, "windows": rows, "pass": all(r["pass"] for r in rows), "scene": info}
    out.write_text(json.dumps(res, indent=1))
    print(f"LOAD-EQUAL {'PASS' if res['pass'] else 'FAIL'}; saved {out}")


if __name__ == "__main__":
    main()
