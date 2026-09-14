"""Surface actuation using measured MPM response and a constrained trust region.

This is a different task formulation from the legacy weighted sum: reduce the
image residual while keeping mass and physics-core losses nonincreasing. All
three are evaluated on one physical rollout. No image-gradient norm gain.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch
from scipy.optimize import minimize

from .geometric import GeometricConfig, forward_geometry, trajectory_health, verify_replay


@dataclass
class ResponseConfig:
    iterations: int = 6
    fd_strain: float = .005
    radius: float = .05
    max_radius: float = .15
    min_radius: float = 1e-5
    max_control_rms: float = .3
    max_particle_control: float = .4
    attempts: int = 8
    acceptance_ratio: float = .1
    max_response_error: float = .2
    objective: str = "image"  # identical basis/constraints for physics-only comparison


class SurfaceStrainBasis:
    """Six symmetric global modes, optionally times linear rest-coordinate fields.

    Coherent within the selected material shell, discontinuous at its hard mask.
    No field is supported on an interior particle. Modes are whitened in the
    mean surface Frobenius metric, so |z| is actual RMS dFc per substep. The same
    dFc is applied on EVERY substep; changing horizon changes the control problem.
    """
    def __init__(self, rest_x, surface, device="cpu", linear=False):
        x = torch.as_tensor(rest_x, dtype=torch.float32, device=device)
        mask = torch.as_tensor(surface, device=device)
        if mask.dtype != torch.bool or mask.shape != (len(x),) or int(mask.sum()) < 4:
            raise ValueError("basis requires a boolean material surface mask")
        mats = torch.zeros(6, 3, 3, dtype=x.dtype, device=device)
        for a in range(3):
            mats[a, a, a] = 1.
        for j, (a, b) in enumerate(((0, 1), (0, 2), (1, 2)), 3):
            mats[j, a, b] = mats[j, b, a] = 2**-.5
        fields = [torch.ones(len(x), dtype=x.dtype, device=device)]
        if linear:
            centered = x-x[mask].mean(0)
            scale = centered[mask].square().mean(0).sqrt().clamp_min(1e-6)
            fields.extend((centered/scale).unbind(1))
        spatial = torch.stack(fields)*mask[None, :]
        raw = torch.einsum("kn,aij->kanij", spatial, mats).reshape(-1, len(x), 3, 3)
        flat = raw.flatten(1).double()
        gram = flat@flat.T/int(mask.sum())
        L = torch.linalg.cholesky(gram)
        self.modes = torch.linalg.solve_triangular(L, flat, upper=False).reshape_as(raw).float()
        self.mask, self.count = mask, len(self.modes)

    def field(self, coefficients):
        return torch.einsum("k,knij->nij", coefficients, self.modes)

    def expand(self, coefficients, steps):
        return self.field(coefficients).unsqueeze(0).expand(steps, -1, -1, -1).contiguous()

    def peak(self, coefficients):
        return float(self.field(coefficients).flatten(1).norm(dim=1).max())


def quadratic_model(residual, jacobian):
    """Half-squared residual model, with Jacobian columns as the first axis."""
    r, j = residual.flatten().double(), jacobian.flatten(1).double()
    return {"value": float(.5*(r@r)), "g": (j@r).cpu().numpy(), "H": (j@j.T).cpu().numpy()}


def model_value(model, delta):
    return model["value"] + model["g"]@delta + .5*delta@model["H"]@delta


def solve_constrained_response(models, coefficients, radius, max_control_rms, objective="image"):
    """Small convex quadratic subproblem. Positive row scaling preserves its solution.

    The physical constraints are not scalarized into the image objective. Their
    normalization only changes the units of the same inequalities for SLSQP.
    """
    if objective not in ("image", "physics"):
        raise ValueError("objective must be image or physics")
    z = np.asarray(coefficients, np.float64)
    goal = models[objective]
    scale = max(abs(goal["value"]), 1e-12)
    def fun(d):
        return (model_value(goal, d)-goal["value"])/scale
    def jac(d):
        return (goal["g"]+goal["H"]@d)/scale
    constraints = []
    for key in ("mass", "physics"):
        m = models[key]
        s = max(abs(m["value"]), 1e-12)
        constraints.append({"type": "ineq",
            "fun": lambda d, m=m, s=s: (m["value"]-model_value(m, d))/s,
            "jac": lambda d, m=m, s=s: -(m["g"]+m["H"]@d)/s})
    constraints.extend([
        {"type": "ineq", "fun": lambda d: 1-(d@d)/radius**2,
         "jac": lambda d: -2*d/radius**2},
        {"type": "ineq", "fun": lambda d: 1-((z+d)@(z+d))/max_control_rms**2,
         "jac": lambda d: -2*(z+d)/max_control_rms**2},
    ])
    result = minimize(fun, np.zeros_like(z), jac=jac, method="SLSQP", constraints=constraints,
                      options={"ftol": 1e-11, "maxiter": 300})
    feasible = all(float(c["fun"](result.x)) >= -1e-7 for c in constraints)
    return result.x, {"success": bool(result.success and feasible), "message": str(result.message),
                       "iterations": int(result.nit), "predicted_decrease": float(-fun(result.x)*scale)}


def optimize_response_window(spec, basis, residuals, cfg=None, health_cfg=None, on_iteration=None):
    """Measure local response, solve one joint constrained candidate, re-evaluate it.

    residuals(state, control) returns flattened mass/physics/image residuals,
    each defining its scalar as 0.5*sum(r**2). No second derivative or Warp JVP is
    assumed; the native raster's finite-step behavior is measured explicitly.
    """
    cfg = ResponseConfig() if cfg is None else cfg
    health_cfg = GeometricConfig() if health_cfg is None else health_cfg
    if spec.vol0 is None or min(cfg.fd_strain, cfg.radius, cfg.max_control_rms, cfg.max_particle_control) <= 0:
        raise ValueError("positive strain bounds and persistent source vol0 are required")
    if cfg.objective not in ("image", "physics"):
        raise ValueError("unsupported objective")
    z = torch.zeros(basis.count, device=spec.device)
    radius, evaluations, history = cfg.radius, 0, []
    started = time.monotonic()
    def evaluate(coeff):
        nonlocal evaluations
        evaluations += 1
        if basis.peak(coeff) > cfg.max_particle_control:
            return None, None, {"valid": False, "reason": "particle_control_bound"}
        with torch.no_grad():
            c = basis.expand(coeff, spec.T)
            state, tr = forward_geometry(c, spec)
            health = trajectory_health(tr, health_cfg)
            if not health["valid"]:
                return None, tr, health
            rr = {k: v.detach().flatten() for k, v in residuals(state, c).items()}
            if set(rr) != {"mass", "physics", "image"} or not all(bool(torch.isfinite(v).all()) for v in rr.values()):
                return None, tr, {"valid": False, "reason": "invalid_residuals"}
        return rr, tr, health
    def values(rr):
        return {k: float(.5*v.double().square().sum()) for k, v in rr.items()}
    current, accepted_tr, health = evaluate(z)
    repeat, repeated_tr, repeat_health = evaluate(z)
    if current is None or repeat is None:
        raise RuntimeError(f"invalid initial rollout: {health}, {repeat_health}")
    verify_replay(accepted_tr, repeated_tr)
    replay_noise = {k: float((current[k]-repeat[k]).double().norm()) for k in current}
    start_values = values(current)
    # Scalar tolerance is numerical, not a permission to trade mass for images.
    scalar_tol = {k: max(10*abs(start_values[k]-values(repeat)[k]), 1e-9*max(start_values[k], 1e-3))
                  for k in current}
    for iteration in range(cfg.iterations):
        before = values(current)
        columns = {k: [] for k in current}
        fd_info = []
        failed = False
        for index in range(basis.count):
            unit = torch.zeros_like(z)
            unit[index] = 1.
            epsilon = min(cfg.fd_strain, radius*.25)
            # Probe peak bounds are checked in the same physical dFc units as candidates.
            for _ in range(5):
                plus, _, hp = evaluate(z+epsilon*unit)
                minus, _, hm = evaluate(z-epsilon*unit)
                if plus is not None and minus is not None:
                    break
                epsilon *= .5
            if plus is None or minus is None:
                failed = True
                break
            for k in columns:
                columns[k].append((plus[k]-minus[k])/(2*epsilon))
            fd_info.append({"epsilon_rms_strain": epsilon, "peak_perturbation": basis.peak(epsilon*unit),
                            "image_signal_norm": float((plus["image"]-minus["image"]).double().norm())})
        if failed:
            history.append({"iteration": iteration, "accepted": False, "reason": "invalid_response_probes"})
            break
        jacobians = {k: torch.stack(v) for k, v in columns.items()}
        models = {k: quadratic_model(current[k], jacobians[k]) for k in current}
        # A held-out combination checks the response prediction at half the probe scale.
        direction = torch.arange(1, basis.count+1, device=z.device, dtype=z.dtype)
        direction /= direction.norm()
        check_step = min(cfg.fd_strain, radius*.25)*.5
        check, _, _ = evaluate(z+check_step*direction)
        response_error = None
        if check is not None:
            response_error = {}
            for k in current:
                observed = check[k]-current[k]
                predicted = check_step*torch.einsum("k,kr->r", direction, jacobians[k])
                response_error[k] = float((observed-predicted).double().norm()/observed.double().norm().clamp_min(1e-20))
        record = {"iteration": iteration, "accepted": False, "before": before,
                  "response_validation_error": response_error, "fd": fd_info,
                  "image_response_eigenvalues": np.linalg.eigvalsh(models["image"]["H"]).tolist()}
        required_responses = ("mass", "physics", "image") if cfg.objective == "image" else ("mass", "physics")
        if response_error is None or any(response_error[k] > cfg.max_response_error for k in required_responses):
            record.update(reason="unreliable_response_model", rollout_evaluations=evaluations,
                          seconds=time.monotonic()-started)
            history.append(record)
            if on_iteration is not None:
                on_iteration(record, accepted_tr)
            break
        for attempt in range(cfg.attempts):
            delta, sub = solve_constrained_response(models, z.cpu().numpy(), radius, cfg.max_control_rms, cfg.objective)
            record["subproblem"] = sub
            if not sub["success"]:
                record["reason"] = "subproblem_failed"
                break
            prediction = sub["predicted_decrease"]
            if prediction <= scalar_tol[cfg.objective]:
                record["reason"] = "no_measured_feasible_descent"
                break
            d = torch.as_tensor(delta, device=z.device, dtype=z.dtype)
            if basis.peak(d) > cfg.max_particle_control:
                trial, tr, trial_health = None, None, {"valid": False, "reason": "particle_increment_bound"}
            else:
                trial, tr, trial_health = evaluate(z+d)
            accepted = False
            if trial is not None:
                after = values(trial)
                gain = before[cfg.objective]-after[cfg.objective]
                ratio = gain/prediction
                accepted = (gain > scalar_tol[cfg.objective] and ratio >= cfg.acceptance_ratio
                    and after["mass"] <= before["mass"]+scalar_tol["mass"]
                    and after["physics"] <= before["physics"]+scalar_tol["physics"])
                if cfg.objective == "image":
                    accepted = accepted and after["image"] < before["image"]-scalar_tol["image"]
                if accepted:
                    z, current, accepted_tr, health = (z+d).detach(), trial, tr, trial_health
                    record.update(accepted=True, after=after, radius=radius, ratio=ratio, attempt=attempt,
                                  peak_control=basis.peak(z), rms_control=float(z.norm()),
                                  control_increment_rms=float(d.norm()), health=health)
                    if ratio > .75 and float(d.norm()) > .8*radius:
                        radius = min(cfg.max_radius, radius*1.6)
                    break
            radius *= .5
            record.update(reason="actual_rollout_rejected", candidate_health=trial_health)
            if radius < cfg.min_radius:
                record["reason"] = "trust_region_exhausted"
                break
        record.update(rollout_evaluations=evaluations, seconds=time.monotonic()-started)
        history.append(record)
        if on_iteration is not None:
            on_iteration(record, accepted_tr)
        if not record["accepted"]:
            break
    final, replay, final_health = evaluate(z)
    if final is None:
        raise RuntimeError(f"invalid final response replay: {final_health}")
    verify_replay(accepted_tr, replay)
    for k, value in values(final).items():
        if abs(value-values(current)[k]) > max(scalar_tol[k], 1e-6*max(abs(value), 1e-3)):
            raise RuntimeError(f"final response replay changed {k}")
    return {"coefficients": z, "control": basis.expand(z, spec.T), "trajectory": replay,
            "history": history, "initial": start_values, "final": values(final), "health": final_health,
            "replay_residual_noise": replay_noise, "scalar_tolerances": scalar_tol,
            "rollout_evaluations": evaluations, "seconds": time.monotonic()-started}
