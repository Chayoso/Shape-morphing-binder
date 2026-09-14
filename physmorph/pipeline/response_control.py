"""Surface actuation using measured MPM response and a constrained trust region.

This is a different task formulation from the legacy weighted sum: reduce the
image residual while keeping mass and physics-core losses nonincreasing. All
three are evaluated on one physical rollout. No image-gradient norm gain.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import time

import numpy as np
import torch
import warp as wp
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
    minimum_physics_progress: float = 0.0  # fraction of feasible local 3-D/core progress
    geometric_response_constraints: bool = False
    global_surface_checks: bool = False


class SurfaceStrainBasis:
    """Six symmetric global modes, optionally times linear rest-coordinate fields.

    Coherent within the selected material shell, discontinuous at its hard mask.
    No field is supported on an interior particle. Modes are whitened in the
    mean surface Frobenius metric, so |z| is actual RMS dFc per substep. The same
    dFc is applied on EVERY substep; changing horizon changes the control problem.
    """
    def __init__(self, rest_x, surface, device="cpu", linear=False, patches=0, patch_radius=1.35):
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
        self.patch_centers = None
        if patches:
            if linear or patches < 2 or patches > int(mask.sum()) or patch_radius <= 0:
                raise ValueError("patch modes require distinct centers and positive radius; no linear mix")
            directions = x-x[mask].mean(0)
            directions /= directions.norm(dim=1, keepdim=True).clamp_min(1e-8)
            skin = directions[mask]
            chosen = [int(torch.argmax(skin[:, 1]))]
            distances = torch.full((len(skin),), float("inf"), device=device)
            for _ in range(patches-1):
                distances = torch.minimum(distances, (skin-skin[chosen[-1]]).square().sum(1))
                chosen.append(int(torch.argmax(distances)))
            centers = skin[chosen]
            r = torch.cdist(directions, centers)/patch_radius
            weights = (1-r).clamp_min(0).pow(4)*(1+4*r)
            if bool((weights[mask].sum(1) < 1e-8).any()):
                raise ValueError("patches do not cover the material shell")
            weights /= weights.sum(1, keepdim=True).clamp_min(1e-8)
            fields = list(weights.unbind(1))
            self.patch_centers = centers
        if linear:
            centered = x-x[mask].mean(0)
            scale = centered[mask].square().mean(0).sqrt().clamp_min(1e-6)
            fields.extend((centered/scale).unbind(1))
        spatial = torch.stack(fields)*mask[None, :]
        raw = torch.einsum("kn,aij->kanij", spatial, mats).reshape(-1, len(x), 3, 3)
        flat = raw.flatten(1).double()
        gram = flat@flat.T/int(mask.sum())
        self.gram_condition = float(torch.linalg.cond(gram))
        if not np.isfinite(self.gram_condition) or self.gram_condition > 1e10:
            raise ValueError("surface basis is numerically rank deficient")
        L = torch.linalg.cholesky(gram)
        self.modes = torch.linalg.solve_triangular(L, flat, upper=False).reshape_as(raw).float()
        self.mask, self.count = mask, len(self.modes)

    def field(self, coefficients):
        return torch.einsum("k,knij->nij", coefficients, self.modes)

    def expand(self, coefficients, steps):
        return self.field(coefficients).unsqueeze(0).expand(steps, -1, -1, -1).contiguous()

    def peak(self, coefficients):
        return float(self.field(coefficients).flatten(1).norm(dim=1).max())


class TemporalSurfaceBasis(SurfaceStrainBasis):
    """Independent piecewise-constant controls with time-weighted physical units.

    Rows of modes/mask enumerate (phase, particle), only to express exact peak
    constraints. expand returns the actual (T,N,3,3) MPM control sequence.
    """
    def __init__(self, spatial, steps, phases=2):
        if not 1 <= phases <= steps:
            raise ValueError("temporal phases must partition the rollout")
        self.steps, self.phases, self.particles = steps, phases, len(spatial.mask)
        self.phase_index = torch.arange(steps, device=spatial.modes.device)*phases//steps
        counts = torch.bincount(self.phase_index, minlength=phases)
        weights = counts.to(spatial.modes.dtype)/steps
        k, n = spatial.count, self.particles
        self.modes = spatial.modes.new_zeros(phases*k, phases*n, 3, 3)
        for phase in range(phases):
            self.modes[phase*k:(phase+1)*k, phase*n:(phase+1)*n] = spatial.modes/weights[phase].sqrt()
        self.count = phases*k
        self.mask = spatial.mask.repeat(phases)
        self.patch_centers, self.gram_condition = spatial.patch_centers, spatial.gram_condition

    def expand(self, coefficients, steps):
        if steps != self.steps:
            raise ValueError("temporal basis horizon changed")
        fields = self.field(coefficients).reshape(self.phases, self.particles, 3, 3)
        return fields[self.phase_index].contiguous()


def quadratic_model(residual, jacobian):
    """Half-squared residual model, with Jacobian columns as the first axis."""
    r, j = residual.flatten(), jacobian.flatten(1)
    h = torch.zeros((len(j), len(j)), device=j.device, dtype=torch.float64)
    g = torch.zeros(len(j), device=j.device, dtype=torch.float64)
    value = torch.zeros((), device=j.device, dtype=torch.float64)
    # HQ Jacobians can occupy several GiB; do not make a full float64 copy.
    for start in range(0, len(r), 1 << 20):
        rb, jb = r[start:start+(1 << 20)].double(), j[:, start:start+(1 << 20)].double()
        h += jb@jb.T
        g += jb@rb
        value += .5*(rb@rb)
    return {"value": float(value), "g": g.cpu().numpy(), "H": h.cpu().numpy()}


def model_value(model, delta):
    return model["value"] + model["g"]@delta + .5*delta@model["H"]@delta


def solve_constrained_response(models, coefficients, radius, max_control_rms, objective="image",
                               particle_modes=None, max_particle_control=None,
                               loss_bounds=None, initial_delta=None, state_inequality=None):
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
        return (goal["g"]@d + .5*d@goal["H"]@d)/scale
    def jac(d):
        return (goal["g"]+goal["H"]@d)/scale
    constraints = []
    for key in ("mass", "physics"):
        m = models[key]
        s = max(abs(m["value"]), 1e-12)
        upper = m["value"] if loss_bounds is None else loss_bounds[key]
        constraints.append({"type": "ineq",
            "fun": lambda d, m=m, s=s, upper=upper: (upper-model_value(m, d))/s,
            "jac": lambda d, m=m, s=s: -(m["g"]+m["H"]@d)/s})
    constraints.extend([
        {"type": "ineq", "fun": lambda d: 1-(d@d)/radius**2,
         "jac": lambda d: -2*d/radius**2},
        {"type": "ineq", "fun": lambda d: 1-((z+d)@(z+d))/max_control_rms**2,
         "jac": lambda d: -2*(z+d)/max_control_rms**2},
    ])
    # Constraint generation avoids treating a blocked particle as a blocked body.
    # Individual PSD quadratics let the solver use feasible tangent directions.
    active = set()
    initial = np.zeros_like(z) if initial_delta is None else np.asarray(initial_delta, np.float64)
    peak_ok = True
    iterations = 0
    for generation in range(16):
        # Solve in a unit trust ball. Small physical strain radii otherwise put
        # O(1/radius) constraint gradients beside tiny objective steps in SLSQP.
        # This is an exact change of variables, including every chain rule.
        unit_constraints = [{"type": c["type"],
            "fun": lambda u, c=c: c["fun"](radius*u),
            "jac": lambda u, c=c: radius*c["jac"](radius*u)} for c in constraints]
        result = minimize(lambda u: fun(radius*u), initial/radius,
                          jac=lambda u: radius*jac(radius*u), method="SLSQP", constraints=unit_constraints,
                          options={"ftol": 1e-11, "maxiter": 300})
        result.x = radius*result.x
        iterations += int(result.nit)
        if (particle_modes is None and state_inequality is None) or not np.isfinite(result.x).all():
            break
        violations = []
        if particle_modes is not None:
            pm = np.asarray(particle_modes, np.float64).reshape(len(z), -1, 9)
            cap = float(max_particle_control)*(1-1e-5)
            for kind, coeff in (("total", z+result.x), ("increment", result.x)):
                squared = np.einsum("k,knc->nc", coeff, pm, optimize=True)
                squared = np.einsum("nc,nc->n", squared, squared)
                for p in np.flatnonzero(squared > cap**2*(1+1e-7)):
                    violations.append((float(squared[p]/cap**2), kind, int(p)))
        if state_inequality is not None:
            predicted = state_inequality["value"]+state_inequality["J"]@result.x
            for p in np.flatnonzero(predicted > 1.+1e-7):
                violations.append((float(predicted[p]), "geometry", int(p)))
        peak_ok = not violations
        if peak_ok:
            break
        added = 0
        for _, kind, p in sorted(violations, reverse=True):
            if (kind, p) in active:
                continue
            active.add((kind, p))
            if kind == "geometry":
                row, value = state_inequality["J"][p], state_inequality["value"][p]
                constraints.append({"type": "ineq",
                    "fun": lambda d, row=row, value=value: 1-value-row@d,
                    "jac": lambda d, row=row: -row})
            else:
                h = pm[:, p]@pm[:, p].T/cap**2
                offset = z if kind == "total" else np.zeros_like(z)
                constraints.append({"type": "ineq",
                    "fun": lambda d, h=h, o=offset: 1-(o+d)@h@(o+d),
                    "jac": lambda d, h=h, o=offset: -2*h@(o+d)})
            added += 1
            if added == 8:
                break
        if not added:
            break
        initial = result.x
    feasible = all(float(c["fun"](result.x)) >= -1e-7 for c in constraints)
    # An inexact but feasible decreasing step remains a valid trust-region
    # candidate. The full nonlinear rollout is the acceptance authority.
    usable = np.isfinite(result.x).all() and (result.success or -fun(result.x) > 1e-10)
    return result.x, {"success": bool(usable and feasible and peak_ok),
                       "optimizer_converged": bool(result.success), "message": str(result.message),
                       "iterations": iterations, "particle_constraints": sum(k != "geometry" for k, _ in active),
                       "geometric_constraints": sum(k == "geometry" for k, _ in active),
                       "predicted_decrease": float(-fun(result.x)*scale)}


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
    if not 0 <= cfg.minimum_physics_progress < 1:
        raise ValueError("minimum physics progress must be in [0,1)")
    z = torch.zeros(basis.count, device=spec.device)
    particle_modes = basis.modes[:, basis.mask].flatten(2).cpu().numpy().astype(np.float64)
    diagnostic_health = (replace(health_cfg, max_geom_condition=float("inf"))
                         if cfg.geometric_response_constraints else health_cfg)
    loss_keys = ("mass", "physics", "image")
    radius, evaluations, history = cfg.radius, 0, []
    started = time.monotonic()
    def evaluate(coeff, diagnostic=False):
        nonlocal evaluations
        evaluations += 1
        if not diagnostic and basis.peak(coeff) > cfg.max_particle_control:
            return None, None, {"valid": False, "reason": "particle_control_bound"}
        with torch.no_grad():
            c = basis.expand(coeff, spec.T)
            state, tr = forward_geometry(c, spec)
            health = trajectory_health(tr, diagnostic_health if diagnostic else health_cfg)
            if not health["valid"]:
                return None, tr, health
            if cfg.global_surface_checks and not diagnostic:
                from .surface_validity import no_surface_intersections
                global_health = no_surface_intersections(tr)
                if not global_health["valid"]:
                    return None, tr, global_health
                health.update(global_health)
            rr = {k: v.detach().flatten() for k, v in residuals(state, c).items()}
            if set(rr) != {"mass", "physics", "image"} or not all(bool(torch.isfinite(v).all()) for v in rr.values()):
                return None, tr, {"valid": False, "reason": "invalid_residuals"}
            if cfg.geometric_response_constraints:
                geometry = torch.stack([wp.to_torch(a) for a in tr.F_geom])
                if tr.surface_F is not None:
                    geometry = torch.cat([geometry, torch.stack([wp.to_torch(a) for a in tr.surface_F])], 1)
                sv = torch.linalg.svdvals(geometry)
                condition = (sv[..., 0]/sv[..., -1].clamp_min(1e-12)).amax(0)
                rr["condition"] = torch.log(condition)/np.log(health_cfg.max_geom_condition)
        return rr, tr, health
    def values(rr):
        return {k: float(.5*rr[k].double().square().sum()) for k in loss_keys}
    current, accepted_tr, health = evaluate(z)
    repeat, repeated_tr, repeat_health = evaluate(z)
    if current is None or repeat is None:
        raise RuntimeError(f"invalid initial rollout: {health}, {repeat_health}")
    verify_replay(accepted_tr, repeated_tr)
    replay_noise = {k: float((current[k]-repeat[k]).double().norm()) for k in current}
    start_values = values(current)
    # Scalar tolerance is numerical, not a permission to trade mass for images.
    scalar_tol = {k: max(10*abs(start_values[k]-values(repeat)[k]), 1e-9*max(start_values[k], 1e-3))
                  for k in loss_keys}
    for iteration in range(cfg.iterations):
        before = values(current)
        columns = {k: [] for k in current}
        fd_info = []
        failed = False
        for index in range(basis.count):
            unit = torch.zeros_like(z)
            unit[index] = 1.
            epsilon = min(cfg.fd_strain, radius*.25)
            # Probes may cross control and (when linearized) condition bounds.
            # They are never committed; finite/grid/orientation checks remain.
            # Accepted candidates always use the complete original health gate.
            for _ in range(5):
                plus, _, hp = evaluate(z+epsilon*unit, diagnostic=True)
                minus, _, hm = evaluate(z-epsilon*unit, diagnostic=True)
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
        del columns
        models = {k: quadratic_model(current[k], jacobians[k]) for k in loss_keys}
        state_inequality = None
        if cfg.geometric_response_constraints:
            state_inequality = {"value": current["condition"].double().cpu().numpy(),
                                "J": jacobians["condition"].T.double().cpu().numpy()}
        # A held-out combination checks the response prediction at half the probe scale.
        direction = torch.arange(1, basis.count+1, device=z.device, dtype=z.dtype)
        direction /= direction.norm()
        check_step = min(cfg.fd_strain, radius*.25)*.5
        check, _, _ = evaluate(z+check_step*direction, diagnostic=True)
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
            bounds, reference_delta = None, None
            if cfg.minimum_physics_progress > 0 and cfg.objective == "image":
                reference_delta, reference_info = solve_constrained_response(models, z.cpu().numpy(),
                    radius, cfg.max_control_rms, "physics", particle_modes, cfg.max_particle_control,
                    state_inequality=state_inequality)
                record["local_physics_reference"] = reference_info
                if reference_info["success"]:
                    bounds = {key: models[key]["value"]-cfg.minimum_physics_progress*max(0.,
                        models[key]["value"]-model_value(models[key], reference_delta)) for key in ("mass", "physics")}
                    record["predicted_physics_bounds"] = bounds
                else:
                    record["reason"] = "physics_reference_subproblem_failed"
                    radius *= .5
                    continue
            delta, sub = solve_constrained_response(models, z.cpu().numpy(), radius, cfg.max_control_rms,
                cfg.objective, particle_modes, cfg.max_particle_control, bounds, reference_delta, state_inequality)
            record["subproblem"] = sub
            if not sub["success"]:
                record["reason"] = "subproblem_failed"
                radius *= .5
                if radius < cfg.min_radius:
                    break
                continue
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
                    record.pop("reason", None)
                    record.pop("candidate_health", None)
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
