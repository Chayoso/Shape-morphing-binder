"""run_xu_paper: Xu et al. as published, reimplemented on this simulator (D118; PipelineConfig.xu_protocol "paper").

The morph is a chain of episodes (the paper's "simulation networks", V-D: "We construct a new simulation network using
the final layer of the previous optimized network"): each is optimised by window/xu_episode.optimize_episode and kept
whatever its line search did, and the next starts from its end state (x, F, v, C). Nothing of our protocol: no
released half, outer acceptance, rejection brake, stopping rule, best-window delivery, plastic assimilation (the
paper's material is elastic), warm start (Algorithm 1 starts every control at zero) or material bonds. The number of
episodes is the morph's length in the paper's timesteps over the network's ten (cfg.animations, scripts/pipeline_run.py
--xu_timesteps); the delivered trajectory is every episode.
"""
from __future__ import annotations

import time

import torch

from ... import gpu
from ...losses.volumetric import d_vol_density, d_vol_xu
from ...mpm.traj import compute_rest_volumes
from ..config import PipelineConfig
from ..target import build_target, calibrate_units
from ..window import StartState
from ..window.setup import domain_bounds
from ..window.xu_episode import XU, optimize_episode
from .state import FrameStore, promote

GUARDS = ("clamped", "nan_x", "nan_state", "F_reset", "F_flip", "F_invert_steps")


def run_xu_paper(source_x, target_x, prm, cfg: PipelineConfig, log=print, F_stride: int | None = None,
                 draws=None) -> dict:
    """Morph source -> target with Xu et al.'s method. Returns the dict run_pipeline returns (frames, history, guards,
    the delivered slice: all frames)."""
    gpu.require_cuda()
    src = gpu.tensor(source_x)
    N = src.shape[0]
    if len(target_x) != N:
        raise ValueError(f"source and target need the same particle count (got {N} vs {len(target_x)})")
    steps, layer = XU.layout(prm.dt)
    log(f"[xu] Xu et al. as published: {cfg.animations} episodes of {steps} driven steps ({steps * prm.dt:.4f} s), "
        f"one control layer held over the first {layer} step(s); {XU.passes} passes x {XU.iters} iterations of Adam "
        f"(alpha {XU.alpha:g}) with a halving line search (at most {XU.ls_iters} trials); every episode kept; "
        f"loss {cfg.baseline} ({cfg.xu_form}), loss mass {cfg.xu_mass:.4g} a particle")
    tgt = build_target(target_x, prm, cfg, draws=draws)
    calibrate_units(tgt, src, cfg)
    lo, hi = domain_bounds(prm)
    x = src.clone()
    F = v = C = None
    Fp = torch.eye(3, device=gpu.DEVICE).repeat(N, 1, 1)         # elastic: no plastic state ever forms
    vol0 = compute_rest_volumes(src, 1.0, prm, cfg.device)
    frames = FrameStore(src, F_stride or steps)
    hist, guards, completed = [], {k: 0 for k in GUARDS}, True
    for a in range(cfg.animations):
        t0 = time.time()
        x_start = x
        res = optimize_episode(StartState(x=x, Fp=Fp, F=F, v=v, C=C), prm, cfg, tgt, vol0, log=log)
        commit = res.commit
        if commit.reason == "nonfinite":
            log(f"[xu] episode {a + 1}: the committed rollout is not finite; the chain ends here")
            hist.append({"animation": a, "null_commit": 1, "commit_reason": commit.reason, **_stats(res.stats)})
            completed = False
            break
        x, F, v, C, counts = promote(commit, lo, hi)
        for k in GUARDS:
            guards[k] += counts[k]
        frames.add_window(commit.x[1:-1], commit.F[1:-1], x, F)
        rec = _record(a, res, x, x_start, v, F, counts, commit, tgt, cfg)
        rec.update(frame_end=len(frames), seconds=time.time() - t0)
        hist.append(rec)
        res.commit = commit = None                              # release the episode's buffers before the next
        log(f"[xu] episode {a + 1}/{cfg.animations}: loss {rec['loss_start']:.5g} -> {rec['loss']:.5g} "
            f"(Xu {rec['transport_energy']:.5g}), steps accepted {rec['accepted']} of {XU.passes * XU.iters}, "
            f"line search failed {rec['ls_failed']}, |F~|max {rec['ctrl_absmax']:.3g}, |v|max {rec['v_absmax']:.3f}, "
            f"kin {rec['kin']:.4g}, {rec['seconds']:.1f} s" + (f", GUARD {counts}" if any(counts.values()) else ""))
    return {"truncation": None, "deliver_n": len(frames), "frames": frames, "history": hist, "guards": guards,
            "Fp": gpu.host(Fp), "n_held": 0, "converged": completed, "balancer": {}}


def _stats(s: dict) -> dict:
    keep = ("steps", "layer", "accepted", "ls_failed", "ls_trials", "ctrl_absmax", "ctrl_rms", "passes")
    return {**{k: s[k] for k in keep if k in s}, "loss_start": s.get("L0")}


def _record(a, res, x, x_start, v, F, counts, commit, tgt, cfg) -> dict:
    """The episode's history record, measured on the promoted state: the loss read at the episode's driven end (the
    commit's, with the spray cleanup for xu_spray), Xu's loss alone there, and the density-unit D_vol as our runs log."""
    with torch.no_grad():
        d_vol = float(d_vol_density(x, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims, tgt.m_ref, tgt.n_support))
        energy = float(d_vol_xu(x, tgt.m * cfg.xu_mass, *tgt.xu, **cfg.xu_kw()))
        jmin = float(torch.linalg.det(F).min())
    return {"animation": a, "loss": float(commit.E_final), "transport_energy": energy, "d_vol": d_vol,
            "selection_merit": float(commit.E_final), "kin": float(v.pow(2).sum(1).mean()),
            "v_absmax": float(v.abs().max()), "v_mean": float(v.norm(dim=1).mean()),
            "com": x.mean(0).tolist(), "v_com": v.mean(0).tolist(), "move": float((x - x_start).norm(dim=1).mean()),
            "Jmin": jmin, "Jmin_traj": float(commit.jmin_traj), "commit_reason": commit.reason, **counts,
            **_stats(res.stats)}
