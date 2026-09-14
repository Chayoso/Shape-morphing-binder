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
    min_surface_density_ratio: float = .1
    min_surface_triangle_area_ratio: float = .01
    max_surface_edge_ratio: float = 4.


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
                    track_geometry=True, F_geom0=spec.F_geom0,
                    surface0=spec.surface0, surface_F0=spec.surface_F0, surface_faces=spec.surface_faces,
                    surface_reference0=spec.surface_reference0, surface_density0=spec.surface_density0)
    tr.rollout()
    n = tr.N
    state = (wp.to_torch(tr.x[-1]), wp.to_torch(tr.F[-1]).reshape(n, 9),
             wp.to_torch(tr.v[-1]), wp.to_torch(tr.F_geom[-1]).reshape(n, 9))
    if tr.surface_x is not None:
        state += (wp.to_torch(tr.surface_x[-1]), wp.to_torch(tr.surface_F[-1]))
    return state, tr


def trajectory_health(tr: Trajectory, cfg: GeometricConfig) -> dict:
    """Raw-state checks, including interior substeps; never read rendered images."""
    low = torch.tensor(tr.prm.grid_min, device=wp.to_torch(tr.x[0]).device) + 2*tr.prm.dx
    high = low + (torch.tensor([tr.prm.nx, tr.prm.ny, tr.prm.nz], device=low.device)-5)*tr.prm.dx
    min_model, min_geom, max_condition = float("inf"), float("inf"), 0.
    surface_rho0 = None
    if tr.surface_x is not None:
        surface_rho0 = (wp.to_torch(tr.surface_density[0]) if tr.surface_density0 is None else
                        torch.as_tensor(tr.surface_density0, device=low.device))
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
        if tr.surface_x is not None:
            sx, sf = wp.to_torch(tr.surface_x[t]), wp.to_torch(tr.surface_F[t])
            if not bool(torch.isfinite(sx).all() and torch.isfinite(sf).all()):
                return {"valid": False, "reason": "surface_nonfinite", "substep": t}
            if bool(((sx < low) | (sx > high)).any()):
                return {"valid": False, "reason": "surface_grid_stencil", "substep": t}
            if float(torch.linalg.det(sf).min()) <= cfg.min_determinant:
                return {"valid": False, "reason": "surface_orientation", "substep": t}
            ssv = torch.linalg.svdvals(sf)
            if float((ssv[:, 0]/ssv[:, -1]).max()) > cfg.max_geom_condition:
                return {"valid": False, "reason": "surface_condition", "substep": t}
            rho = wp.to_torch(tr.surface_density[t])
            if (not bool(torch.isfinite(rho).all()) or float(surface_rho0.min()) <= 1e-8
                    or bool((rho < cfg.min_surface_density_ratio*surface_rho0).any())):
                return {"valid": False, "reason": "surface_mass_support", "substep": t}
    result = {"valid": True, "min_det_model": min_model, "min_det_geom": min_geom,
              "max_geom_condition": max_condition}
    if tr.surface_x is not None:
        skin_sv = torch.linalg.svdvals(torch.stack([wp.to_torch(f) for f in tr.surface_F]))
        result.update(min_surface_det=float(skin_sv.prod(-1).min()),
                      max_surface_condition=float((skin_sv[..., 0]/skin_sv[..., -1]).max()),
                      min_surface_density_ratio=float(torch.stack([wp.to_torch(d) for d in tr.surface_density]).div(
                          surface_rho0).min()))
    if tr.surface_faces is not None:
        faces = torch.as_tensor(tr.surface_faces, dtype=torch.long, device=low.device)
        tri = torch.stack([wp.to_torch(a) for a in tr.surface_x])[:, faces]
        e1, e2 = tri[..., 1, :]-tri[..., 0, :], tri[..., 2, :]-tri[..., 0, :]
        rest = torch.as_tensor(tr.surface_reference0, device=low.device)[faces]
        r1, r2 = rest[:, 1]-rest[:, 0], rest[:, 2]-rest[:, 0]
        rest_area = torch.linalg.cross(r1, r2).norm(dim=-1)
        normals = torch.linalg.cross(e1, e2)
        area = normals.norm(dim=-1)
        if float(rest_area.min()) <= 1e-12:
            return {"valid": False, "reason": "initial_surface_degenerate"}
        area_ratio = float((area/rest_area).min())
        edge = torch.stack([e1.norm(dim=-1), e2.norm(dim=-1), (e2-e1).norm(dim=-1)], -1)
        rest_edge = torch.stack([r1.norm(dim=-1), r2.norm(dim=-1), (r2-r1).norm(dim=-1)], -1)
        edge_ratio = float((edge/rest_edge.clamp_min(1e-12)).max())
        # Compare current face orientation against local transported tangents.
        sf = torch.stack([wp.to_torch(a) for a in tr.surface_F])[:, faces].mean(2)
        # F is cumulative from the persistent original reference; no inverse of
        # an averaged rotation is needed (that average can be singular).
        n_expected = torch.linalg.cross(torch.einsum("tfij,fj->tfi", sf, r1),
                                       torch.einsum("tfij,fj->tfi", sf, r2))
        if not bool(torch.isfinite(n_expected).all()) or float(n_expected.norm(dim=-1).min()) <= 1e-12:
            return {"valid": False, "reason": "surface_expected_normal_degenerate"}
        cosine = (normals*n_expected).sum(-1)/(area*n_expected.norm(dim=-1)).clamp_min(1e-20)
        if (area_ratio < cfg.min_surface_triangle_area_ratio or edge_ratio > cfg.max_surface_edge_ratio
                or float(cosine.min()) <= 0.):
            return {"valid": False, "reason": "surface_triangle_deformation",
                    "min_area_ratio": area_ratio, "max_edge_ratio": edge_ratio, "min_normal_cosine": float(cosine.min())}
        result.update(min_surface_triangle_area_ratio=area_ratio, max_surface_edge_ratio=edge_ratio,
                      min_surface_normal_cosine=float(cosine.min()))
    return result


def next_window_spec(spec: RolloutSpec, tr: Trajectory) -> RolloutSpec:
    """Promote physical state and cumulative geometry together, without assimilation."""
    return replace(spec, x0=tr.x[-1].numpy().copy(), F0=tr.F[-1].numpy().copy(),
                   v0=tr.v[-1].numpy().copy(), C0=tr.C[-1].numpy().copy(),
                   F_geom0=tr.F_geom[-1].numpy().copy(),
                   surface0=None if tr.surface_x is None else tr.surface_x[-1].numpy().copy(),
                   surface_F0=None if tr.surface_F is None else tr.surface_F[-1].numpy().copy(),
                   surface_reference0=tr.surface_reference0,
                   surface_density0=(None if tr.surface_x is None else tr.surface_density[0].numpy().copy())
                       if tr.surface_density0 is None else tr.surface_density0.copy())


def verify_replay(reference: Trajectory, replay: Trajectory):
    """Scalar equality cannot detect corrupted C or other unobserved restart state."""
    for name in ("x", "v", "C", "F", "F_geom", "surface_x", "surface_F", "surface_density"):
        if getattr(reference, name) is None and getattr(replay, name) is None:
            continue
        if getattr(reference, name) is None or getattr(replay, name) is None:
            raise RuntimeError(f"final replay changed availability of {name}")
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
