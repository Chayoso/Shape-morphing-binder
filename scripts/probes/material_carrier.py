"""Tier M carrier probe — is the per-particle material leaf `s` a carrier the render
channel can actually USE?  (docs/local_global_design.md Appendix A; docs/root_analysis.md)

The v2 root analysis says the render covector's UNIQUE content is high frequency and the
MPM adjoint low-passes it before it reaches the control `dFc`.  Appendix A proposes the
existing per-particle log-Lame leaf `s = (s_lam, s_mu)` (`cfg.opt_material`,
`optimizer.material()`, `warp_mpm_full(dfc, spec, lam_t, mu_t)`) as a deferred replacement
carrier, on the claim that "no cheap exact annulling control exists" (A.1).

This probe decides that claim with numbers, on a WARM state of the flagship config:

  A  warm state   4 accepted commits of the dress_bench setup (no gauss), full promotion
                  (x, F, v, C) + isochoric plastic assimilation of Fp.
  B  gradients    ||dL_render/ds||, ||dL_phys/ds||, cos, and the same on dFc; the
                  render-to-phys ratio per leaf; kNN band decomposition of the per-particle
                  render-covector magnitude on each leaf.
  C  cancellation the decisive test.  A band-limited material perturbation delta_s makes a
                  terminal displacement delta_x; fit dFc (Adam) to reproduce delta_x and
                  report the residual fraction + the control cost it needed.  Then the
                  REVERSE (band-limited dFc perturbation, fit s) for the asymmetry.
  D  locality     share of the render covector's material energy on surface parents.

Read-only w.r.t. the pipeline: nothing here is imported by production code.

Run (hyde06, GPU 2 only):
  cd /tmp/pm31 && OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    CUDA_VISIBLE_DEVICES=2 <python> scripts/probes/material_carrier.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch  # noqa: E402
import warp as wp  # noqa: E402

from physmorph.losses.volumetric import d_vol  # noqa: E402
from physmorph.mpm import MPMParams  # noqa: E402
from physmorph.mpm.conditioning import condition_F  # noqa: E402
from physmorph.mpm.constitutive import lame  # noqa: E402
from physmorph.mpm.function import RolloutSpec, warp_mpm_full  # noqa: E402
from physmorph.mpm.traj import Trajectory, compute_rest_volumes  # noqa: E402
from physmorph.pipeline import PipelineConfig  # noqa: E402
from physmorph.pipeline.optimizer import optimize_window  # noqa: E402
from physmorph.pipeline.render_loss import LambdaBalancer, d_render  # noqa: E402
from physmorph.pipeline.runner import _surface_weights, build_target  # noqa: E402
from physmorph.plasticity import assimilate_elastic  # noqa: E402
from physmorph.sampling import load_normalized as load  # noqa: E402

RES: dict = {}
PEAK = {"mib": 0}


def emit(key, val):
    RES[key] = val
    print(f"[res] {key} = {val}", flush=True)


def start_mem_watch(period=5.0):
    """Sample this process's own GPU footprint (hard rule: stay under 8 GB)."""
    import os
    import subprocess
    import threading

    def poll():
        pid = str(os.getpid())
        while True:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=20).stdout
                for line in out.splitlines():
                    p, mem = [c.strip() for c in line.split(",")[:2]]
                    if p == pid:
                        PEAK["mib"] = max(PEAK["mib"], int(mem))
            except Exception:
                pass
            time.sleep(period)

    threading.Thread(target=poll, daemon=True).start()


# ---------------------------------------------------------------- kNN smoothing
class Smoother:
    """Repeated mean over {self} U kNN — a genuine low-pass (a self-excluding neighbour
    mean has an eigenvalue near -1 on checkerboard modes and is NOT one)."""

    def __init__(self, x: np.ndarray, k: int, device: str):
        from scipy.spatial import cKDTree
        idx = cKDTree(x).query(x, k=k + 1, workers=-1)[1]      # col 0 == self
        self.idx = torch.as_tensor(np.ascontiguousarray(idx), device=device)

    def apply(self, f: torch.Tensor, rounds: int, axis: int = 0) -> torch.Tensor:
        """Smooth `rounds` times along the particle axis `axis`."""
        g = f.movedim(axis, 0) if axis != 0 else f
        for _ in range(rounds):
            g = g[self.idx].mean(1)
        return g.movedim(0, axis) if axis != 0 else g


BAND_ROUNDS = (1, 2, 4, 8, 16)
BAND_NAMES = ("b0 <1", "b1 1-2", "b2 2-4", "b3 4-8", "b4 8-16", "b5 >=16")


def band_energies(sm: Smoother, m: torch.Tensor):
    """Successive-smoothing band split of a per-particle scalar field.

    Bands are S_{r_i} - S_{r_{i+1}} plus the coarsest residual S_16.  They are not
    orthogonal, so fractions are reported against the SUM of band energies."""
    stages, cur, prev_r = [m], m, 0
    for r in BAND_ROUNDS:
        cur = sm.apply(cur, r - prev_r)
        stages.append(cur)
        prev_r = r
    bands = [stages[i] - stages[i + 1] for i in range(len(BAND_ROUNDS))] + [stages[-1]]
    e = np.array([float(b.pow(2).sum()) for b in bands])
    return e / max(e.sum(), 1e-30)


# ---------------------------------------------------------------- rollouts
def rollout_nograd(spec: RolloutSpec, dfc_t, lam_t=None, mu_t=None) -> torch.Tensor:
    """Plain forward rollout, no tape (the optimizer's eval_terms path)."""
    N, T = spec.x0.shape[0], spec.T
    with torch.no_grad():
        lam_e = lam_t.detach().cpu().numpy() if lam_t is not None else spec.lam
        mu_e = mu_t.detach().cpu().numpy() if mu_t is not None else spec.mu
        dc = dfc_t.detach().contiguous()
        seq = [wp.from_torch(dc[t].view(N, 3, 3), dtype=wp.mat33) for t in range(T)]
        tr = Trajectory(spec.x0, spec.m, lam_e, mu_e, spec.prm, T, F0=spec.F0,
                        Fp=spec.Fp, v0=spec.v0, C0=spec.C0, dFc=seq,
                        device=spec.device, requires_grad=False, vol0=spec.vol0)
        tr.rollout()
        return wp.to_torch(tr.x[T]).clone()


# ---------------------------------------------------------------- A: warm state
def warm_state(args, prm, cfg, tgt, src, sw, vol0):
    balancer = LambdaBalancer(cfg.lambda_auto, cfg.lambda_ema, cfg.lambda_cap)
    N = src.shape[0]
    x = src.copy()
    st = {"F": None, "v": None, "C": None}
    Fp = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))
    accepted, tries, ctrl_costs, absmax = 0, 0, [], []
    while accepted < args.commits and tries < args.commits + 4:
        tries += 1
        t0 = time.perf_counter()
        fr, F_seq, end, _s, whist, stats = optimize_window(
            x, prm, cfg, tgt, balancer, F0=st["F"], Fp=Fp, v0=st["v"], C0=st["C"],
            log=lambda *_: None, surface_w=sw, vol0=vol0)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        ok = bool(whist)
        if ok:
            accepted += 1
            x = np.ascontiguousarray(fr[-1], np.float32)
            Fc, _nb, _nf, _ = condition_F(end["F"], clamp=False)
            st = {"F": Fc, "v": np.nan_to_num(end["v"]).astype(np.float32),
                  "C": np.nan_to_num(end["C"]).astype(np.float32)}
            if cfg.assim > 0:
                Fp = assimilate_elastic(Fc, Fp, eta=cfg.assim, smin=cfg.assim_smin,
                                        smax=cfg.assim_smax, isochoric=cfg.assim_iso)
            dfc = stats.get("dfc")
            if dfc is not None:
                ctrl_costs.append(float(cfg.w_ctrl * (dfc ** 2).sum() / (cfg.T * N)))
                absmax.append(float(np.abs(dfc).max()))
            print(f"[warm] commit {tries}: {dt:.2f}s accepted={ok} "
                  f"d_vol={whist[-1]['d_vol']:.3f} d_sil={whist[-1]['d_sil']} "
                  f"kin={whist[-1]['kin']:.4g} dfc_absmax={whist[-1]['dfc_absmax']:.4g} "
                  f"acc/rej={stats['accepted']}/{stats['rejected']}", flush=True)
        else:
            print(f"[warm] commit {tries}: {dt:.2f}s NULL (no accepted step)", flush=True)
    emit("warm_accepted", accepted)
    emit("warm_dfc_absmax", [round(a, 6) for a in absmax])
    emit("warm_ctrl_cost", [float(f"{c:.6g}") for c in ctrl_costs])
    return x, st, Fp


# ---------------------------------------------------------------- fits
def _fit_once(make_leaf, forward, target_dx, x0T, iters, lr, tag):
    """One Adam fit of `leaf` against a fixed terminal displacement.

    Returns (best_residual_frac, best_leaf_snapshot, trace).  The best ITERATE is kept,
    not the last one: a diverging step size must not be scored on where it landed."""
    leaf, post = make_leaf()
    opt = torch.optim.Adam([leaf], lr=lr)
    tn = float(target_dx.norm())
    trace, best, best_p = [], np.inf, leaf.detach().clone()
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        xT = forward(leaf)
        r = (xT - x0T) - target_dx
        r.pow(2).sum().backward()
        frac = float(r.detach().norm()) / max(tn, 1e-30)
        if frac < best:
            best, best_p = frac, leaf.detach().clone()
        opt.step()
        if post is not None:
            post(leaf)
        if it % 20 == 0 or it == iters - 1:
            trace.append((it, round(frac, 5)))
    print(f"[{tag}] lr={lr:g} best_residual_frac={best:.5f} trace={trace}", flush=True)
    return best, best_p, trace


def fit_sweep(kind, spec, target_dx, x0T, iters, lrs, dev, tag, lam0=None, mu0=None,
              clamp=1.0, w_ctrl=1e-3):
    """Fit over a LADDER of step sizes and keep the best iterate found anywhere.

    A single hand-picked lr cannot decide a reachability question: Adam moves every
    element of the leaf by ~lr per step, so one step size either overshoots (measured:
    lr=1e-3 on dFc leaves the residual ABOVE 1.0 after 3 steps) or never arrives."""
    N, T = spec.x0.shape[0], spec.T
    zero_dfc = torch.zeros(T, N, 3, 3, device=dev)

    if kind == "dfc":
        def make_leaf():
            return torch.zeros(T, N, 3, 3, device=dev, requires_grad=True), None

        def forward(p):
            return warp_mpm_full(p, spec, None, None)[0]
    else:
        def make_leaf():
            p = torch.zeros(2, N, device=dev, requires_grad=True)

            def post(q):
                with torch.no_grad():
                    q.clamp_(-clamp, clamp)
            return p, post

        def forward(p):
            return warp_mpm_full(zero_dfc, spec, lam0 * torch.exp(p[0]),
                                 mu0 * torch.exp(p[1]))[0]

    per_lr, best, best_p, best_lr = {}, np.inf, None, None
    for lr in lrs:
        f, p, _tr = _fit_once(make_leaf, forward, target_dx, x0T, iters, lr, tag)
        per_lr[f"{lr:g}"] = round(f, 5)
        if f < best:
            best, best_p, best_lr = f, p, lr

    # verify the winner on an independent no-grad rollout (the graph path and the
    # commit path are not bit-identical; the optimizer validates its own replay too)
    with torch.no_grad():
        if kind == "dfc":
            xT = rollout_nograd(spec, best_p)
        else:
            xT = rollout_nograd(spec, zero_dfc, lam0 * torch.exp(best_p[0]),
                                mu0 * torch.exp(best_p[1]))
        got = xT - x0T
        frac = float((got - target_dx).norm()) / max(float(target_dx.norm()), 1e-30)
        cos = float((got * target_dx).sum() /
                    max(got.norm() * target_dx.norm(), 1e-30))
        gain = float(got.norm() / max(float(target_dx.norm()), 1e-30))
    out = {"best_frac": round(best, 5), "replay_frac": round(frac, 5),
           "cos": round(cos, 5), "amplitude_ratio": round(gain, 5),
           "best_lr": best_lr, "per_lr": per_lr,
           "absmax": float(f"{float(best_p.abs().max()):.6g}")}
    if kind == "dfc":
        out["ctrl_cost"] = float(f"{float(w_ctrl * best_p.pow(2).sum() / (T * N)):.6g}")
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--commits", type=int, default=4)
    ap.add_argument("--fit_iters", type=int, default=100)
    ap.add_argument("--lr_dfc", type=float, nargs="+",
                    default=[3e-3, 1e-3, 3e-4, 1e-4, 3e-5])
    ap.add_argument("--lr_s", type=float, nargs="+",
                    default=[1e-1, 3e-2, 1e-2, 3e-3])
    ap.add_argument("--ds_max", type=float, default=0.3)
    ap.add_argument("--knn", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="output/material_carrier.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    start_mem_watch()

    prm = MPMParams()
    src = load("assets/isosphere.obj", args.n, 1)
    tgt_x = load("assets/bunny.obj", args.n, 2)
    cfg = PipelineConfig(T=20, iters=8, animations=8, loss_res=64)
    cfg.lambda_auto, cfg.w_kin = 0.5, 5.0
    cfg.w_dt, cfg.w_nn, cfg.w_jvol = 0.2, 0.2, 50.0
    cfg.use_gauss_loss = False                       # NO gauss (probe directive)
    cfg.render_surface_only, cfg.surface_grad_frac = True, 0.50
    cfg.assim_iso = True
    cfg.warm_start = True                            # telemetry only: exposes stats["dfc"]
    dev = cfg.device
    N, T = args.n, cfg.T

    tgt = build_target(tgt_x, prm, cfg)
    swf = _surface_weights(src, cfg.surface_grad_k, cfg.surface_grad_frac,
                           cfg.surface_grad_floor)
    surf = swf > 0.5
    sw = np.ascontiguousarray(surf, np.float32)
    vol0 = compute_rest_volumes(src, 1.0, prm, dev)
    emit("N", N)
    emit("T", T)
    emit("surface_frac", round(float(surf.mean()), 4))

    # ---- A: warm state -------------------------------------------------------
    t_a = time.perf_counter()
    x, st, Fp = warm_state(args, prm, cfg, tgt, src, sw, vol0)
    emit("A_seconds", round(time.perf_counter() - t_a, 1))

    lam0, mu0 = lame(cfg.young, cfg.poisson)
    emit("lam0", lam0)
    emit("mu0", mu0)
    spec = RolloutSpec(x0=np.ascontiguousarray(x, np.float32), m=1.0, lam=lam0, mu=mu0,
                       prm=prm, T=T, F0=st["F"], Fp=Fp, v0=st["v"], C0=st["C"],
                       device=dev, vol0=vol0)
    sm = Smoother(np.ascontiguousarray(x, np.float32), args.knn, dev)
    surf_t = torch.as_tensor(surf, device=dev)

    # ---- B: gradients on both leaves ----------------------------------------
    s = torch.zeros(2, N, device=dev, requires_grad=True)
    dfc = torch.zeros(T, N, 3, 3, device=dev, requires_grad=True)
    xT, FT, vT = warp_mpm_full(dfc, spec, lam0 * torch.exp(s[0]), mu0 * torch.exp(s[1]))
    L_r = d_render(xT, tgt.sils, tgt.views, cfg.render_res, tgt.extent, cfg.sil_k,
                   cfg.w_hole, cfg.w_spray)
    L_p = (d_vol(xT, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims)
           + cfg.w_kin * vT.pow(2).sum(1).mean())
    emit("B_d_render", float(L_r.detach()))
    emit("B_d_phys", float(L_p.detach()))

    gr_dfc, gr_s = torch.autograd.grad(L_r, (dfc, s), retain_graph=True)
    gp_dfc, gp_s = torch.autograd.grad(L_p, (dfc, s))
    del xT, FT, vT, L_r, L_p

    def per_particle(g, axis=1):
        """Reduce a leaf gradient to a per-particle magnitude field (N,)."""
        m = g.pow(2)
        for a in sorted([d for d in range(g.dim()) if d != axis], reverse=True):
            m = m.sum(a)
        return m.sqrt()

    def stat(gr, gp, name):
        nr, np_ = float(gr.norm()), float(gp.norm())
        cos = float((gr * gp).sum() / max(nr * np_, 1e-30))
        emit(f"B_{name}_render_norm", float(f"{nr:.6g}"))
        emit(f"B_{name}_phys_norm", float(f"{np_:.6g}"))
        emit(f"B_{name}_cos", round(cos, 5))
        emit(f"B_{name}_render_over_phys", float(f"{nr / max(np_, 1e-30):.6g}"))

    stat(gr_s, gp_s, "s")
    stat(gr_dfc, gp_dfc, "dfc")
    mag_s, mag_c = per_particle(gr_s), per_particle(gr_dfc)
    magp_s, magp_c = per_particle(gp_s), per_particle(gp_dfc)

    for nm, m in (("s", mag_s), ("dfc", mag_c),
                  ("s_phys", magp_s), ("dfc_phys", magp_c)):
        f = band_energies(sm, m)
        emit(f"B_bands_{nm}", {k: round(float(v), 5) for k, v in zip(BAND_NAMES, f)})
        emit(f"B_hf_{nm}", round(float(f[0] + f[1]), 5))   # rounds < 2 == the fine bands

    # ---- D: where the material signal lives ---------------------------------
    for nm, m in (("s", mag_s), ("dfc", mag_c)):
        e = m.pow(2)
        frac = float(e[surf_t].sum() / max(float(e.sum()), 1e-30))
        emit(f"D_surface_energy_frac_{nm}", round(frac, 5))
        emit(f"D_surface_enrichment_{nm}", round(frac / max(float(surf.mean()), 1e-9), 4))

    del gr_dfc, gr_s, gp_dfc, gp_s, s, dfc
    torch.cuda.empty_cache()

    # ---- C: cancellability ---------------------------------------------------
    zero_dfc = torch.zeros(T, N, 3, 3, device=dev)
    x0T = rollout_nograd(spec, zero_dfc)
    # CUDA atomics make a replay non-bit-identical (optimizer f2): every delta below
    # must be read against this floor.
    noise = float((rollout_nograd(spec, zero_dfc) - x0T).norm())
    emit("C0_replay_noise", float(f"{noise:.6g}"))

    # C1: material perturbation -> can dFc reproduce it?
    g = torch.randn(2, N, device=dev)
    ds = sm.apply(g, 4, axis=1)
    ds = ds * (args.ds_max / float(ds.abs().max()))
    xT_m = rollout_nograd(spec, zero_dfc, lam0 * torch.exp(ds[0]), mu0 * torch.exp(ds[1]))
    dx_m = xT_m - x0T
    emit("C1_ds_absmax", round(float(ds.abs().max()), 5))
    emit("C1_dx_norm", float(f"{float(dx_m.norm()):.6g}"))
    emit("C1_dx_rms_per_particle", float(f"{float(dx_m.pow(2).sum(1).mean().sqrt()):.6g}"))
    emit("C1_x0T_scale", float(f"{float(x0T.norm()):.6g}"))
    emit("C1_dx_over_noise", round(float(dx_m.norm()) / max(noise, 1e-30), 2))
    t_c = time.perf_counter()
    r1 = fit_sweep("dfc", spec, dx_m, x0T, args.fit_iters, args.lr_dfc, dev,
                   "C1 fit dFc", w_ctrl=cfg.w_ctrl)
    for k, v in r1.items():
        emit(f"C1_{k}", v)
    emit("C1_seconds", round(time.perf_counter() - t_c, 1))

    # C2 (reverse): dFc perturbation -> can s reproduce it?
    gc = torch.randn(T, N, 3, 3, device=dev)
    dc = sm.apply(gc, 4, axis=1)
    dc = dc / float(dc.abs().max())                       # unit-max band-limited field
    trial = 0.02
    xT_c = rollout_nograd(spec, trial * dc)
    n_trial = float((xT_c - x0T).norm())
    scale = trial * float(dx_m.norm()) / max(n_trial, 1e-30)   # match |delta_x| to C1
    dc = scale * dc
    xT_c = rollout_nograd(spec, dc)
    dx_c = xT_c - x0T
    emit("C2_dfc_absmax", float(f"{float(dc.abs().max()):.6g}"))
    emit("C2_dx_norm", float(f"{float(dx_c.norm()):.6g}"))
    emit("C2_dfc_ctrl_cost", float(f"{float(1e-3 * dc.pow(2).sum() / (T * N)):.6g}"))
    t_c = time.perf_counter()
    r2 = fit_sweep("s", spec, dx_c, x0T, args.fit_iters, args.lr_s, dev, "C2 fit s",
                   lam0=lam0, mu0=mu0, clamp=cfg.mat_clamp)
    for k, v in r2.items():
        emit(f"C2_{k}", v)
    emit("C2_seconds", round(time.perf_counter() - t_c, 1))

    emit("gpu_peak_mib", PEAK["mib"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(RES, indent=1))
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
