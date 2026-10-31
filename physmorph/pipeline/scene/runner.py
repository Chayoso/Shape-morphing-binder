"""run_scene — the window loop of the multi-body scene (D147).

As run.runner.run_pipeline, per window: optimise the controls (scene/solve.SceneOptimizer), promote the committed state,
make half the elastic stretch plastic, archive every simulated frame. Unlike it, every window is committed (time always
advances: the scene's physics runs its whole length, a collision or a landing raises the merit whatever the control
does), there is no selection, stop rule or truncation, and each body is recorded in its own frame: its centre of mass,
velocity, angular velocity and momentum, its frame's rotation, its terms, the contact's impulse and the bodies' gap.
"""
from __future__ import annotations

import torch

from ... import gpu
from ...mpm.traj import compute_rest_volumes
from ...plasticity import assimilate_elastic
from ...prof import timed
from ..render_loss import LambdaBalancer
from ..run.runner import GUARDS
from ..run.state import FrameStore, fragment_mask, promote
from ..window import StartState
from ..window.setup import domain_bounds
from .frame import rigid_velocity
from .solve import SceneOptimizer, SceneTarget

_STATS = ("g_share", "g_cos", "g_phys_norm", "g_rend_norm", "ls_trials", "accepted", "rejected", "replay_rel",
          "inactive", "raster_k", "raster_extent", "zero_reason",
          "null_reason", "zero_commit", "t_start", "t_grad", "t_ls", "t_commit", "u_gate", "ext_builds", "L_split")


def momentum(x, v, C, mass: float, dx: float) -> dict:
    """A body's linear momentum, its angular momentum about its centre of mass (APIC's affine part included, Jiang et
    al. 2015), its centre of mass and velocity, and its least-squares angular velocity."""
    c = x.mean(0)
    r = x - c
    Cm = C.reshape(-1, 3, 3)
    skew = torch.stack((Cm[:, 2, 1] - Cm[:, 1, 2], Cm[:, 0, 2] - Cm[:, 2, 0], Cm[:, 1, 0] - Cm[:, 0, 1]), 1)
    L = mass * (torch.linalg.cross(r, v, dim=1) + (dx * dx / 3.0) * skew).double().sum(0)
    vc, w = rigid_velocity(x, v)
    return {"com": c.tolist(), "v_com": vc.tolist(), "P": (mass * v.double().sum(0)).tolist(), "L": L.tolist(),
            "omega": w.tolist()}


def cross_gap(x, bodies) -> dict:
    """The bodies' nearest particles: their distance and how many particles of each lie within 1.5 sample spacings of
    the other body (a stuck patch)."""
    a, b = (x[bd.rows].contiguous() for bd in bodies)
    sp = bodies[0].vp ** (1.0 / 3.0)
    dab = gpu.knn(b, 1, queries=a)[0][:, 0]
    dba = gpu.knn(a, 1, queries=b)[0][:, 0]
    return {"gap": float(dab.min()), "gap_sp": float(dab.min()) / sp,
            "near": [int((dab < 1.5 * sp).sum()), int((dba < 1.5 * sp).sum())]}


def run_scene(bodies: list, x0: torch.Tensor, v0: torch.Tensor, prm, cfg, windows: int, mu: float, log=print,
              F_stride: int | None = None, assim_free: bool = True, schedule=None, dump: dict | None = None,
              settle_from: int | None = None) -> dict:
    """bodies: scene/window.Body per body (rows, the frame's reference, its built and calibrated TargetPack, the sample's
    volume per particle, the dynamics mass); x0, v0: the launch state in body order (CUDA tensors); schedule:
    scene/schedule.MorphSchedule (each window's target on the way, by the window's end time) or None (the final target
    from the start); dump: {"t0", "t1", "bodies", "path"} (scene/dump.py: a forensic record of the committed windows that
    start in [t0, t1); None: none)."""
    gpu.require_cuda()
    N = x0.shape[0]
    tgt = SceneTarget(bodies, mu)
    balancer = LambdaBalancer(cfg.lambda_auto, cfg.lambda_ema, None, cap_rel=20.0)
    lo, hi = domain_bounds(prm)
    x, v = x0.clone(), v0.clone()
    F = C = J = None
    Fp = torch.eye(3, device=gpu.DEVICE).repeat(N, 1, 1)
    vol0 = torch.cat([compute_rest_volumes(x0[bd.rows].contiguous(), 1.0, prm, cfg.device) for bd in bodies])
    coh = [gpu.knn(x0[bd.rows].contiguous(), cfg.coh_k + 1)[1][:, 1:] for bd in bodies]
    rest = [None] * len(bodies)
    frames = FrameStore(x0, F_stride or cfg.T, volume=cfg.volume_exact != "off")
    hist, guards, dfc_prev = [], {k: 0 for k in GUARDS}, None
    log(f"[scene] {len(bodies)} bodies, N={N} ({', '.join(str(bd.n) for bd in bodies)}), {windows} windows, "
        f"grid {prm.nx}x{prm.ny}x{prm.nz}, g={-prm.f_ext[1]:.4g}, floor {prm.floor_y:.3f}, mu {mu:g}")
    for a in range(windows):
        s = None if schedule is None else schedule.apply(bodies, (a + 1) * 2 * cfg.T * prm.dt)
        bonds = []
        with timed("frag_bonds"):
            for b, bd in enumerate(bodies):
                xb = x[bd.rows]
                d_now = (xb[coh[b]] - xb[:, None, :]).norm(dim=2)
                frag = fragment_mask(xb, prm)
                rest[b] = d_now if rest[b] is None else torch.where(frag[:, None], rest[b], d_now)
                bonds.append((coh[b], rest[b], frag.float()))
        start = StartState(x=x.clone(), Fp=Fp, F=F, v=v, C=C, J=J)
        # fix 3: a commit whose replay disagrees with the accepted candidate is a null window, as in the single-body path:
        # no simulated time, solved again from the same start with the attempt's controls as its warm start, up to the
        # single-body patience; then the uncontrolled continuation is committed and recorded as such
        retries = []
        settling = settle_from is not None and a >= settle_from
        if settling:
            # the scene's settle (2026-10-31, a scene option): zero controls and no layer relaxation, as the single-body
            # runs' settle windows (--settle_windows with --no-settle_relax), in the single-body world: every velocity
            # damped at the single-body drag (MPMParams.drag; the scene's flight runs without it, so its deformation drag
            # then reads zero, mpm/bodies.deformation_drag_rate); the body comes to rest under the loads alone
            import dataclasses
            from types import SimpleNamespace
            from .rollout import commit_rollout
            prm_s = dataclasses.replace(prm, drag=float(type(prm).drag))
            opt = SceneOptimizer(start, prm_s, dataclasses.replace(cfg, layer_relax=False), tgt, balancer, vol0, bonds, 1.0,
                                 None, lambda *_: None)
            with torch.no_grad():
                opt.dFc.zero_()
                opt.u.zero_()
            c_ = commit_rollout(opt.win, opt.obj, opt.dFc, opt.u, 0.0)
            res = SimpleNamespace(commit=c_ if c_.valid else None, hist=[],
                                  stats={"dfc": opt.dFc.detach()[:cfg.T].clone(), "settle": 1, "commit_reason": c_.reason,
                                         "commit_E_final": float(c_.E_final), "selection_merit": float(c_.E_final)})
        while not settling:
            opt = SceneOptimizer(start, prm, cfg, tgt, balancer, vol0, bonds, 1.0, None, lambda *_: None,
                                 retry_mismatch=len(retries) < cfg.patience)
            res = opt.run(dfc_prev)
            if not res.stats.get("retry"):
                break
            retries.append(res.stats.get("null_reason"))
            dfc_prev = res.stats["dfc"]
            log(f"[scene] window {a + 1}: the commit's replay disagrees ({retries[-1]}); solved again from its start "
                f"({len(retries)} of {cfg.patience})")
            res = opt = None
        stats = res.stats
        if res.commit is None:
            log(f"[scene] window {a + 1}: no valid rollout ({stats.get('commit_reason')}); the scene ends here")
            hist.append({"animation": a, "invalid": 1, "commit_reason": stats.get("commit_reason")})
            break
        commit = res.commit
        dfc_prev = stats["dfc"]
        with timed("promote"):
            x, F, v, C, J, counts = promote(commit, lo, hi)
        for k in GUARDS:
            guards[k] += counts[k]
        # the settle's controls are zero, so the controls' share the scene assimilates (assim_free) is nothing there
        if cfg.assim > 0 and not (settling and assim_free):
            with timed("assim"):
                # the controls' deformation without the loads (scene/window.free_end_F), or (assim_free False) the
                # loaded end as the frozen rule reads it: the impact's compression, the weight's sag made rest shape
                F_assim = opt.win.free_end_F() if assim_free else F
                Fp = assimilate_elastic(F_assim, Fp, eta=cfg.assim, smin=cfg.assim_smin, smax=cfg.assim_smax,
                                        isochoric=not cfg.assim_volume)
        with timed("frames"):
            frames.add_window(commit.x[1:-1], commit.F[1:-1], x, F,
                              Js=commit.J[1:-1] if J is not None else None, J_end=J)
        if tgt.grid_ot is not None:
            tgt.grid_ot.repeat = not (counts["clamped"] or counts["nan_x"])
        with timed("record"), torch.no_grad():
            impulse = wp_impulse(opt.win)
            per = []
            for b, (bd, part) in enumerate(zip(bodies, opt.obj.body_terms(x))):
                _, _, Rb = bd.frame(x[bd.rows])
                ang = float(torch.rad2deg(torch.arccos(((torch.trace(Rb) - 1) / 2).clamp(-1, 1))))
                per.append({**part, **momentum(x[bd.rows], v[bd.rows], C[bd.rows], bd.mass, prm.dx),
                            "R": Rb.reshape(-1).tolist(), "R_angle": ang})
            rec = {"animation": a, "t": (a + 1) * 2 * cfg.T * prm.dt, "frame_end": len(frames), "bodies": per,
                   "contact_impulse": impulse, **(cross_gap(x, bodies) if len(bodies) == 2 else {}), **counts,
                   "lambda": res.hist[-1]["lambda"] if res.hist else opt.lam_r, "loss": stats.get("commit_E_final"),
                   "selection_merit": stats.get("selection_merit"), "Jmin_traj": commit.jmin_traj, **device_memory(),
                   **{k: stats.get(k) for k in _STATS}, "morph_s": s, "replay_retries": retries, "settle": stats.get("settle")}
        if dump is not None and dump["t0"] <= a * 2 * cfg.T * prm.dt < dump["t1"]:
            from .dump import window_dump
            window_dump(dump["path"], a, a * 2 * cfg.T * prm.dt, opt, commit, dump["bodies"])
        res.commit = commit = None
        hist.append(rec)
        P = [sum(bp["P"][k] for bp in per) for k in range(3)]
        log(f"[scene] window {a + 1}/{windows} t={rec['t']:.2f}s  merit={rec['loss']:.5g}  lam={rec['lambda']:.3g}  "
            f"g_share={rec.get('g_share') or 0:.2f}  gap={rec.get('gap_sp', float('nan')):.2f}sp near={rec.get('near')}  "
            f"impulse=({impulse[0]:+.3g},{impulse[1]:+.3g},{impulse[2]:+.3g})  P=({P[0]:+.4g},{P[1]:+.4g},{P[2]:+.4g})  "
            + "  ".join(f"b{b}: com=({bp['com'][0]:+.2f},{bp['com'][1]:+.2f}) |w|={sum(w * w for w in bp['omega']) ** 0.5:.3f}"
                        f" R={bp['R_angle']:.1f}deg sil={bp['d_sil']:.4g}" for b, bp in enumerate(per))
            + (f"  raster={rec['raster_k']}" if any(rec.get("raster_k") or []) else "")
            + (f"  left out={rec['inactive']}" if rec.get("inactive") else "")
            + (f"  replay retries={len(retries)}" if retries else "")
            + (f"  zero-control commit ({stats.get('zero_reason')})" if stats.get("zero_commit") else ""))
    return {"frames": frames, "history": hist, "guards": guards, "Fp": Fp.cpu().numpy()}


def wp_impulse(win) -> list:
    import warp as wp
    return wp.to_torch(win.tr.impulse).double().reshape(-1).tolist()


def device_memory() -> dict:
    """A record: the device's free memory and Warp's memory pool in use (MiB), to see that a window's trajectories are
    released (D148's leak: a reference cycle through the tape kept them until the garbage collector ran)."""
    import warp as wp
    free, _ = torch.cuda.mem_get_info()
    out = {"gpu_free_mib": free / 2 ** 20, "torch_alloc_mib": torch.cuda.memory_allocated() / 2 ** 20}
    used = getattr(wp, "get_mempool_used_mem_current", None)
    if used is not None:
        try:
            out["warp_pool_used_mib"] = used("cuda") / 2 ** 20
        except Exception:                                   # noqa: BLE001  (a record only)
            pass
    return out
