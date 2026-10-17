"""one_equal.py OUT_JSON [N] [SEED] [MESH] -- B9b: the one-body scene against the single-body window without gravity, on
the frozen recipe's 40k bunny (seed 97): one window's rollout and terms under the same controls (zero, and random leaves of
1e-3 and 1e-2 a step, u zero), from the same start (the source about its centroid, at rest).

  A  the frozen window (window/setup.Window, window/objective.Objective) on the run's own grid (prep.prm, drag 0.9).
  B  the scene's window of one body (scene/window.SceneWindow, scene/objective.SceneObjective) in the frame fixed at the
     start, on A's grid with A's drag 0.9 (so no deformation damping): the same definition; expected equal to rounding.
  C  the scene's window in the body's Buckens frame on A's grid, the world drag 0 and the deformation damped at 0.9 / s
     (mpm/bodies, fix 4): B9b's definition without the loads; a body without loads has no rigid motion (the controls keep
     its momentum), so expected equal to A up to rounding and the frame's turn.
  D  as C on the scene's own grid (scene/spec.layout with B9b's at-rest spec, g 0, no floor): the grid placed from the
     scene's extents, not the single-body cube; the transfers' discretisation differs (positions relative to the nodes).
  E  as D with the floor (the gated floor and its position constraint, the plane half a pitch under the lowest particle).

Per arm and control: the end positions against A (largest and RMS, in sample pitches), the end velocities' RMS against
A's RMS, each term (transport lv, kinetic lk, silhouette lr, shading lpbr, d_sil, stability lstab) and its relative
difference to A; and (amended 2026-10-17 17:50 CDT, after the first check stopped on the zero control's kinetic terms
of a body at rest, 6.5e-11 and 3.5e-13) every arm's window replayed REPLAYS times per control: the arm's own replay
range of each term (CUDA atomics sum in another order at every replay) and its positions' largest difference from A's
first replay over all replays. CUDA; run on hyde06."""
from __future__ import annotations

import dataclasses
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
from physmorph.pipeline.scene.spec import SceneSpec, layout  # noqa: E402
from physmorph.pipeline.scene.window import Body, SceneWindow  # noqa: E402
from physmorph.pipeline.target import build_target, calibrate_units, target_relief  # noqa: E402
from physmorph.pipeline.window.objective import Objective  # noqa: E402
from physmorph.pipeline.window.rollout import eval_terms  # noqa: E402
from physmorph.pipeline.window.setup import StartState, Window  # noqa: E402
from scripts.pipeline_run import parse_args as run_args, prepared, run_config  # noqa: E402

TERMS = ("lv", "lk", "lr", "lpbr", "d_sil", "lstab")
REPLAYS = 8                 # each arm's window evaluated 8 times per control: its own replay range (the amended check)


def main():
    out = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 97
    mesh = sys.argv[4] if len(sys.argv) > 4 else "bunny"
    gpu.require_cuda()
    base = run_args(["--n", str(n), "--tgt", f"assets/{mesh}.obj", "--seed", str(seed), "--out", str(out)])
    cfg0, prep = prepared(base, seed)
    cfg = run_config(base, cfg0, prep)
    prm_a = prep.prm
    X = np.asarray(prep.src, np.float64)
    X = (X - X.mean(0)).astype(np.float32)
    x = gpu.tensor(X)
    tgt = build_target(prep.tgt, prm_a, cfg, draws=prep.tgt_draws)
    if prep.tgt_surface is not None:
        tgt.relief = target_relief(tgt.pts, prep.tgt_surface, cfg)
    calibrate_units(tgt, x, cfg)
    vp = (prep.density["target_fill"] if prep.density else prep.v_src) / n
    sp = vp ** (1.0 / 3.0)
    coh = gpu.knn(x, cfg.coh_k + 1)[1][:, 1:]
    bonds = (coh, (x[coh] - x[:, None]).norm(dim=2), torch.zeros(n, device=x.device))
    start = StartState(x=x, Fp=torch.eye(3, device=x.device).repeat(n, 1, 1))

    def scene_prm(floor: bool):
        spec = SceneSpec(bodies=1, rest=True, g=0.0, G=0.0, mu=0.0, yaw=(0.0, 0.0), floor=floor, drag=0.0, seconds=8.33)
        _, _, _, prm, info = layout(spec, [X.astype(np.float64)], [vp], cfg.young, cfg.poisson, cfg.mass_ref_n,
                                    prm_a.dx, prm_a.dt, cfg.T, radius=float(np.linalg.norm(prep.tgt - prep.tgt.mean(0),
                                                                                          axis=1).max()))
        return prm, info

    prm_d, info_d = scene_prm(False)
    prm_e, info_e = scene_prm(True)
    arms = {
        "A": ("frozen", prm_a, None),
        "B": ("scene", prm_a, "fixed"),
        "C": ("scene", dataclasses.replace(prm_a, drag=0.0), "body"),
        "D": ("scene", prm_d, "body"),
        "E": ("scene", prm_e, "body"),
    }
    gen = torch.Generator().manual_seed(2)
    noise = torch.randn(cfg.T, n, 3, 3, generator=gen)
    controls = {"zero": torch.zeros(cfg.T, n, 3, 3), "rand1e-3": 1e-3 * noise, "rand1e-2": 1e-2 * noise}
    u = torch.zeros(n, device=x.device)
    res = {"n": n, "seed": seed, "mesh": mesh, "pitch": sp, "dx": prm_a.dx,
           "grids": {k: {"grid_min": list(v[1].grid_min), "n": [v[1].nx, v[1].ny, v[1].nz], "drag": v[1].drag,
                         "floor_y": v[1].floor_y} for k, v in arms.items()},
           "scene_info_D": info_d, "scene_info_E": info_e, "arms": {}}
    ref = {}
    for name, (kind, prm, frame) in arms.items():
        vol0 = compute_rest_volumes(x, 1.0, prm, cfg.device)
        if kind == "frozen":
            win = Window(start, prm, cfg, tgt, vol0, bonds)
            obj = Objective(win)
        else:
            body = Body(start=0, stop=n, Xq=x, tgt=tgt, vp=vp, mass=cfg.mass_ref_n / n, cfg=cfg,
                        raster=BodyRaster(prm_a, cfg),
                        fixed=(x.mean(0), torch.eye(3, device=x.device)) if frame == "fixed" else None)
            win = SceneWindow(start, prm, cfg, [body], vol0, [bonds], mu=0.0)
            obj = SceneObjective(win)
        row = {}
        for cname, leaf in controls.items():
            e = eval_terms(win, obj, leaf.to(x.device), u)
            xT, vT = e.xT.detach().clone(), e.vT.detach().clone()
            terms = {k: float(getattr(e, k)) for k in TERMS if getattr(e, k) is not None}
            if name == "A":
                ref[cname] = (xT, vT, terms)
            xa, va, ta = ref[cname]
            dxT = (xT - xa).norm(dim=1)
            # the same window replayed (CUDA atomics: each replay sums in another order): the arm's own replay spread
            # of every term, and its positions' largest difference from A's first replay over all replays
            reps = {k: [v] for k, v in terms.items()}
            dx_all = float(dxT.max())
            for _ in range(REPLAYS - 1):
                e2 = eval_terms(win, obj, leaf.to(x.device), u)
                for k in reps:
                    reps[k].append(float(getattr(e2, k)))
                dx_all = max(dx_all, float((e2.xT - xa).norm(dim=1).max()))
                e2 = None
            row[cname] = {"moved_rms_sp": float((xT - x).norm(dim=1).square().mean().sqrt()) / sp,
                          "dx_max_sp": float(dxT.max()) / sp, "dx_rms_sp": float(dxT.square().mean().sqrt()) / sp,
                          "dx_max_sp_all": dx_all / sp,
                          "dv_rms_rel": float((vT - va).norm(dim=1).square().mean().sqrt())
                          / max(float(va.norm(dim=1).square().mean().sqrt()), 1e-30),
                          "terms": terms, "replays": reps,
                          "terms_rel": {k: (v - ta[k]) / max(abs(ta[k]), 1e-30) for k, v in terms.items() if k in ta},
                          "in_domain": bool(e.in_domain), "jt": e.jt}
            print(f"[{name}] {cname}: moved {row[cname]['moved_rms_sp']:.4g} sp; vs A max {row[cname]['dx_max_sp']:.3g} "
                  f"rms {row[cname]['dx_rms_sp']:.3g} sp, dv {row[cname]['dv_rms_rel']:.3g}; terms rel "
                  + " ".join(f"{k}={v:+.2e}" for k, v in row[cname]["terms_rel"].items()), flush=True)
        res["arms"][name] = row
        win = obj = None
        torch.cuda.empty_cache()
    out.write_text(json.dumps(res, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
