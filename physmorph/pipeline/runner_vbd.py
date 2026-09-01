"""run_vbd_pipeline — quasi-static VBD-MPM outer loop (docs/method.md §7).

Same contract as runner.run_pipeline (frames / F_frames / history / guards / n_held /
converged) so scripts and gates treat it as just another arm. One frame per commit:
there is no substep trajectory — each commit IS an equilibrium.

Per commit: freeze the CIC stencil at the current cloud (total-Lagrangian within the
commit), solve the grid energy with colored block descent, apply the incremental map to
x and F, repair-and-count F pathologies, assimilate an eta-fraction of the elastic
stretch into Fp (identical channel to the dynamic path), track the RAW loss components
for the plateau freeze. Rest is by construction (no velocities exist), so gate G3's
drift term is trivially zero and only tail jitter is informative.
"""
from __future__ import annotations

import numpy as np
import torch

from ..losses.volumetric import d_vol
from ..mpm.conditioning import condition_F
from ..mpm.state import MPMParams
from ..plasticity import assimilate_elastic
from ..vbd import QuasiStaticGrid
from .config import PipelineConfig
from .render_loss import LambdaBalancer, d_render
from .runner import build_target


def _id(N):
    return np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))


def run_vbd_pipeline(source_x, target_x, prm: MPMParams, cfg: PipelineConfig, log=print,
                     on_commit=None, on_sweep=None):
    dev = cfg.device
    src = np.ascontiguousarray(source_x, np.float32)
    N = src.shape[0]
    assert target_x.shape[0] == N, "D_vol needs equal particle counts"
    tgt = build_target(target_x, prm, cfg)
    balancer = LambdaBalancer(cfg.lambda_auto, cfg.lambda_ema)

    dmin = np.asarray(prm.grid_min, np.float32)
    dmax = dmin + prm.dx * np.array([prm.nx, prm.ny, prm.nz], np.float32)
    lo, hi = dmin + 2 * prm.dx, dmax - 2 * prm.dx

    x = src.copy()
    F = _id(N)
    Fp = _id(N)
    frames, F_frames, hist = [x.copy()], [F.copy()], []
    guards = {"clamped": 0, "nan_x": 0, "nan_state": 0, "F_reset": 0, "F_flip": 0,
              "F_invert_steps": 0}
    best_phys, best_rend, stale, frozen, n_held = None, None, 0, False, 0

    log(f"[vbd] N={N} sweeps<={cfg.vbd_sweeps} tol={cfg.vbd_tol} commits={cfg.animations} "
        f"render={'on(a=%g)' % cfg.lambda_auto if cfg.lambda_auto > 0 else 'OFF'} "
        f"assim={cfg.assim}")

    for a in range(cfg.animations):
        if frozen:
            if cfg.hold_after_converge:
                frames.append(x.copy()); F_frames.append(F_frames[-1].copy())
                hist.append({"animation": a, "held": 1})
                n_held += 1
                continue
            break
        x_start = x.copy()
        qs = QuasiStaticGrid(x_start, prm.grid_min, prm.dx,
                             (prm.nx, prm.ny, prm.nz), cfg.vbd_young, cfg.poisson, dev)
        Fe0 = torch.as_tensor(
            np.einsum("nij,njk->nik", F, np.linalg.inv(Fp)).astype(np.float32), device=dev)

        def parts(u):
            E_el, _ = qs.elastic(u, Fe0)
            disp, _ = qs.kinematics(u)
            xn = qs.x0 + disp
            lv = d_vol(xn, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims)
            lb = torch.clamp(xn.abs() - tgt.extent, min=0).pow(2).sum(1).mean()
            lr = (d_render(xn, tgt.sils, tgt.views, cfg.render_res, tgt.extent,
                           cfg.sil_k, cfg.w_hole, cfg.w_spray) if balancer.active else None)
            return E_el, lv, lb, lr

        # per-commit lambda from gradient norms at u=0 (EMA across commits)
        lam_r = 0.0
        if balancer.active:
            u0 = torch.zeros(qs.A, 3, device=dev, requires_grad=True)
            E_el, lv, lb, lr = parts(u0)
            gp = torch.autograd.grad(E_el + lv + cfg.w_box * lb, u0, retain_graph=True)[0]
            gr = torch.autograd.grad(lr, u0)[0]
            lam_r = balancer.update(float(gp.norm()), float(gr.norm()))

        def energy(u):
            E_el, lv, lb, lr = parts(u)
            E = E_el + lv + cfg.w_box * lb
            return E if lr is None else E + lam_r * lr

        sweep_cb = None
        if on_sweep is not None:                       # live viewer: publish DURING the solve
            ny, nz = prm.ny, prm.nz
            lin = qs.active
            npos = (torch.stack([lin // (ny * nz), (lin // nz) % ny, lin % nz], 1).float()
                    * prm.dx + torch.as_tensor(prm.grid_min, device=dev))
            npos_np = npos.cpu().numpy().astype(np.float32)

            def sweep_cb(u_s, g_s, gn_s, si, _a=a, _F=F, _qs=qs, _np=npos_np):
                with torch.no_grad():
                    disp_s, Ap_s = _qs.kinematics(u_s)
                    x_s = (_qs.x0 + disp_s).cpu().numpy().astype(np.float32)
                    F_s = (Ap_s.cpu().numpy().astype(np.float32) @ _F)
                    nq = np.stack([u_s.norm(dim=1).cpu().numpy(),
                                   g_s.norm(dim=1).cpu().numpy()], 1).astype(np.float32)
                on_sweep(_a, si, gn_s, x_s, F_s, _np, nq)

        u, info = qs.solve(energy, sweeps=cfg.vbd_sweeps, tol=cfg.vbd_tol,
                           step=cfg.vbd_step, ls=cfg.vbd_ls, on_sweep=sweep_cb)

        with torch.no_grad():
            disp, Ap = qs.kinematics(u)
            x_new = (qs.x0 + disp).cpu().numpy().astype(np.float32)
            F_new = (Ap.cpu().numpy().astype(np.float32) @ F)
            E_el, lv, lb, lr = parts(u)

        n_nan = int((~np.isfinite(x_new).all(1)).sum())
        n_out = int(((x_new < lo) | (x_new > hi)).any(1).sum())
        x = np.clip(np.nan_to_num(x_new), lo, hi).astype(np.float32)
        Fc, n_bad, n_flip, _ = condition_F(F_new, clamp=False)
        n_inv = int((np.linalg.det(Fc) <= 0).sum())
        guards["clamped"] += n_out; guards["nan_x"] += n_nan
        guards["F_reset"] += n_bad; guards["F_flip"] += n_flip
        guards["F_invert_steps"] += n_inv
        F = Fc
        if cfg.assim > 0:
            Fp = assimilate_elastic(F, Fp, eta=cfg.assim,
                                    smin=cfg.assim_smin, smax=cfg.assim_smax)

        frames.append(x.copy()); F_frames.append(F.copy())
        rec = {"animation": a, "loss": float(energy(u).detach()), "E_el": float(E_el),
               "d_vol": float(lv), "d_render": float(lr) if lr is not None else None,
               "lambda": lam_r if balancer.active else None, "kin": 0.0, "v_mean": 0.0,
               "sweeps": info["sweeps"], "gnorm": info["gnorm"], "gnorm0": info["gnorm0"],
               "solve_converged": info["converged"],
               "move": float(np.linalg.norm(x - x_start, axis=1).mean()),
               "Jmin": float(np.linalg.det(F).min()),
               "clamped": n_out, "nan_x": n_nan, "F_reset": n_bad, "F_flip": n_flip,
               "F_invert_steps": n_inv}
        hist.append(rec)
        if on_commit is not None:                    # live viewer hook (scripts/live_viewer.py)
            on_commit(a, x, F, rec)

        phys_track = rec["d_vol"] + rec["E_el"]
        rend_track = rec["d_render"]
        improved = best_phys is None or phys_track < best_phys - cfg.tol * abs(best_phys)
        if rend_track is not None and best_rend is not None:
            improved = improved or rend_track < best_rend - cfg.tol * abs(best_rend)
        best_phys = phys_track if best_phys is None else min(best_phys, phys_track)
        if rend_track is not None:
            best_rend = rend_track if best_rend is None else min(best_rend, rend_track)
        stale = 0 if improved else stale + 1
        if stale >= cfg.patience:
            frozen = True
            log(f"[vbd] converged at commit {a + 1}; holding still")

        any_guard = n_out or n_nan or n_bad or n_flip or n_inv
        if a % max(1, cfg.animations // 10) == 0 or a == cfg.animations - 1 or any_guard:
            log(f"[vbd] {a + 1}/{cfg.animations}  E={rec['loss']:.4f}  E_el={rec['E_el']:.3f}"
                f"  D_vol={rec['d_vol']:.3f}" +
                (f"  D_r={rec['d_render']:.5f}  lam={rec['lambda']:.3g}"
                 if rec["d_render"] is not None else "") +
                f"  sweeps={rec['sweeps']}{'' if rec['solve_converged'] else '(!)'}  "
                f"move={rec['move']:.4f}  Jmin={rec['Jmin']:.3f}" +
                (f"  GUARD out={n_out} nan={n_nan} Frst={n_bad} Fflp={n_flip} Finv={n_inv}"
                 if any_guard else ""))

    return {"frames": frames, "F_frames": F_frames, "history": hist, "guards": guards,
            "s": None, "Fp": Fp, "n_held": n_held, "converged": frozen}
