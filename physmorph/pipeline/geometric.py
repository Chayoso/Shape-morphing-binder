"""Joint dFc optimization with stress-mediated geometric rendering.

This intentionally small path implements docs/geometric_control_contract.md.
Legacy optimizer experiments remain available separately. No state projection,
opacity fitting, render gain balancing, local dressing, or alternating updates.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
import warp as wp

from ..mpm.function import RolloutSpec, warp_mpm_geometry
from ..mpm.traj import Trajectory


@dataclass
class GeometricConfig:
    iterations: int = 20
    step_size: float = .005
    line_search_steps: int = 12
    render_weight: float = 1.0  # fixed objective definition, never a gradient-norm ratio
    rms_decay: float = .9
    rms_eps: float = 1e-5
    armijo: float = 1e-4
    min_determinant: float = 1e-4
    max_geom_condition: float = 20.


def restrict_render_control(gradient: torch.Tensor, surface: torch.Tensor):
    """Mask AFTER the full x/F_geom adjoint: endpoint masking alone leaks inward."""
    shape = (1, -1, 1, 1) if gradient.ndim == 4 else (-1, 1, 1)
    return gradient * surface.to(device=gradient.device, dtype=gradient.dtype).reshape(shape)


def forward_geometry(control: torch.Tensor, spec: RolloutSpec):
    """Identical no-tape candidate/replay; includes all intermediate geometry."""
    c = control.detach().contiguous()
    d = ([wp.from_torch(t.contiguous(), dtype=wp.mat33) for t in c]
         if c.ndim == 4 else wp.from_torch(c, dtype=wp.mat33))
    tr = Trajectory(spec.x0, spec.m, spec.lam, spec.mu, spec.prm, spec.T,
                    F0=spec.F0, Fp=spec.Fp, v0=spec.v0, C0=spec.C0, dFc=d,
                    vol0=spec.vol0, device=spec.device, requires_grad=False,
                    track_geometry=True, F_geom0=spec.F_geom0)
    tr.rollout()
    n = tr.N
    state = (wp.to_torch(tr.x[-1]), wp.to_torch(tr.F[-1]).reshape(n, 9),
             wp.to_torch(tr.v[-1]), wp.to_torch(tr.F_geom[-1]).reshape(n, 9))
    return state, tr


def trajectory_health(tr: Trajectory, cfg: GeometricConfig) -> dict:
    """Raw-state checks, including interior substeps; never read rendered images."""
    low = torch.tensor(tr.prm.grid_min, device=wp.to_torch(tr.x[0]).device) + 2*tr.prm.dx
    high = low + (torch.tensor([tr.prm.nx, tr.prm.ny, tr.prm.nz], device=low.device)-5)*tr.prm.dx
    min_model, min_geom, max_condition = float("inf"), float("inf"), 0.
    for t in range(tr.T + 1):
        arrays = [wp.to_torch(a[t]) for a in (tr.x, tr.v, tr.C, tr.F, tr.F_geom)]
        if not all(bool(torch.isfinite(a).all()) for a in arrays):
            return {"valid": False, "reason": "nonfinite_state", "substep": t}
        x, _, _, fm, fg = arrays
        if bool(((x < low) | (x > high)).any()):
            return {"valid": False, "reason": "incomplete_grid_stencil", "substep": t}
        jm, jg = float(torch.linalg.det(fm).min()), float(torch.linalg.det(fg).min())
        min_model, min_geom = min(min_model, jm), min(min_geom, jg)
        if min(jm, jg) <= cfg.min_determinant:
            return {"valid": False, "reason": "orientation", "substep": t}
        sv = torch.linalg.svdvals(fg)
        max_condition = max(max_condition, float((sv[:, 0]/sv[:, -1]).max()))
        if max_condition > cfg.max_geom_condition:
            return {"valid": False, "reason": "geometric_condition", "substep": t}
        if t < tr.T:
            effective = fm + wp.to_torch(tr._dfc(t))
            fp = wp.to_torch(tr.Fp)
            if (float(torch.linalg.det(effective).min()) <= cfg.min_determinant
                    or float(torch.linalg.det(fp).min()) <= cfg.min_determinant):
                return {"valid": False, "reason": "constitutive_orientation", "substep": t}
    return {"valid": True, "min_det_model": min_model, "min_det_geom": min_geom,
            "max_geom_condition": max_condition}


def next_window_spec(spec: RolloutSpec, tr: Trajectory) -> RolloutSpec:
    """Promote physical state and cumulative geometry together, without assimilation."""
    return replace(spec, x0=tr.x[-1].numpy().copy(), F0=tr.F[-1].numpy().copy(),
                   v0=tr.v[-1].numpy().copy(), C0=tr.C[-1].numpy().copy(),
                   F_geom0=tr.F_geom[-1].numpy().copy())


def verify_replay(reference: Trajectory, replay: Trajectory):
    """Scalar equality cannot detect corrupted C or other unobserved restart state."""
    for name in ("x", "v", "C", "F", "F_geom"):
        for t, (a, b) in enumerate(zip(getattr(reference, name), getattr(replay, name))):
            aa, bb = wp.to_torch(a), wp.to_torch(b)
            if not torch.allclose(aa, bb, rtol=1e-5, atol=1e-5):
                err = float((aa-bb).abs().max())
                raise RuntimeError(f"final replay changed {name}[{t}], max error {err}")


def optimize_geometric_window(spec: RolloutSpec, physical_loss, render_loss,
                              surface: np.ndarray, cfg: GeometricConfig | None = None,
                              initial_control=None, on_iteration=None):
    """One shared rollout, one joint candidate, one joint Armijo decision.

    physical_loss(state, control) and render_loss(x, F_geom) return scalars.
    Surface restriction changes the search direction; the *unmasked* derivative
    of the actual joint scalar is used for its slope. A conflicting non-descent
    direction is reported and stopped, never replaced with a physics-only step.
    """
    cfg = GeometricConfig() if cfg is None else cfg
    if cfg.render_weight < 0 or cfg.step_size <= 0 or cfg.line_search_steps < 1:
        raise ValueError("invalid joint objective or line search configuration")
    if not 0 <= cfg.rms_decay < 1 or cfg.rms_eps <= 0 or not 0 < cfg.armijo < 1:
        raise ValueError("invalid preconditioner or Armijo configuration")
    if spec.vol0 is None:
        raise ValueError("joint geometry requires persistent source vol0")
    if np.asarray(surface).dtype != bool or np.asarray(surface).shape != (len(spec.x0),):
        raise ValueError("surface must be a frozen boolean mask of shape (N,)")
    # Torch and Warp use the same logical device name in this pipeline.
    mask = torch.as_tensor(surface, device=spec.device)
    if not bool(mask.any()):
        raise ValueError("empty render surface")
    control = (torch.zeros(spec.T, len(spec.x0), 3, 3, device=spec.device)
               if initial_control is None else initial_control.detach().clone())
    if control.shape != (spec.T, len(spec.x0), 3, 3):
        raise ValueError("control must be a per-substep (T,N,3,3) sequence")
    history, rms, accepted_steps = [], torch.zeros_like(control), 0

    def evaluate(c):
        with torch.no_grad():
            state, tr = forward_geometry(c, spec)
            health = trajectory_health(tr, cfg)
            if not health["valid"]:
                return float("inf"), None, None, tr, health
            lp, lr = physical_loss(state, c), render_loss(state[0], state[3])
            loss = float(lp + cfg.render_weight*lr)
            if not np.isfinite(loss):
                health = {"valid": False, "reason": "nonfinite_objective"}
                loss = float("inf")
            return loss, float(lp), float(lr), tr, health

    value, _, _, trajectory, health = evaluate(control)
    if not health["valid"]:
        raise RuntimeError(f"invalid starting rollout: {health}")
    for iteration in range(cfg.iterations):
        control.requires_grad_(True)
        state = warp_mpm_geometry(control, spec)
        lp = physical_loss(state, control)
        lr = render_loss(state[0], state[3])
        gp = torch.autograd.grad(lp, control, retain_graph=True)[0]
        gr = torch.autograd.grad(lr, control)[0]
        if not bool(torch.isfinite(gp).all() & torch.isfinite(gr).all()):
            raise RuntimeError("nonfinite joint adjoint")
        gs = restrict_render_control(gr, mask)
        direction_grad = gp + cfg.render_weight*gs
        true_grad = gp + cfg.render_weight*gr
        trial_rms = cfg.rms_decay*rms + (1-cfg.rms_decay)*direction_grad.square()
        denom = (trial_rms/(1-cfg.rms_decay**(accepted_steps+1))).sqrt() + cfg.rms_eps
        delta = -cfg.step_size*direction_grad/denom
        slope = float((true_grad*delta).sum())
        value = float((lp+cfg.render_weight*lr).detach())
        record = {"iteration": iteration, "accepted": False, "loss_before": value,
                  "physics_before": float(lp.detach()), "render_before": float(lr.detach()),
                  "g_physics": float(gp.norm()), "g_render_raw": float(gr.norm()),
                  "g_render_surface": float(gs.norm()),
                  "g_render_interior": float(gs[:, ~mask].norm()),
                  "raw_render_interior": float(gr[:, ~mask].norm()),
                  "true_slope": slope, "render_weight": cfg.render_weight}
        control = control.detach()
        if not np.isfinite(slope) or slope >= 0:
            record["reason"] = "restricted_direction_is_not_joint_descent"
        else:
            for backtrack in range(cfg.line_search_steps):
                fraction = .5**backtrack
                candidate = control + fraction*delta
                new, np_, nr, tr, check = evaluate(candidate)
                # Compare the same scalar at every candidate, with a true slope.
                if check["valid"] and new < value and new <= value + cfg.armijo*fraction*slope:
                    control, trajectory, health = candidate.detach(), tr, check
                    rms, accepted_steps = trial_rms.detach(), accepted_steps+1
                    record.update(accepted=True, loss=new, physics=np_, render=nr,
                                  backtracks=backtrack, step_fraction=fraction,
                                  render_linear_work=float((gr*(fraction*delta)).sum()),
                                  physics_linear_work=float((gp*(fraction*delta)).sum()),
                                  **health)
                    break
            if not record["accepted"]:
                record.update(reason="joint_line_search_failed", candidate_health=check)
        history.append(record)
        if on_iteration is not None:
            on_iteration(record, trajectory)
        if not record["accepted"]:
            break
    replay, _, _, final_tr, final_health = evaluate(control)
    if not final_health["valid"]:
        raise RuntimeError(f"invalid final replay: {final_health}")
    accepted = [r for r in history if r["accepted"]]
    expected = accepted[-1]["loss"] if accepted else value
    if abs(replay-expected) > 1e-5*max(abs(expected), 1e-3):
        raise RuntimeError(f"final replay changed objective: {expected} -> {replay}")
    verify_replay(trajectory, final_tr)
    return {"control": control.detach(), "trajectory": final_tr, "history": history,
            "health": final_health, "loss": replay, "accepted_steps": accepted_steps}
