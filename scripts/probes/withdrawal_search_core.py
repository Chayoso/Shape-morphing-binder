"""P331: one frozen-origin joint-body search; evidence only, never adoption."""
from copy import deepcopy
import math

import torch

from physmorph.pipeline.affine_braking import affine_ball_step, projected_affine_check
from physmorph.pipeline.frozen_withdrawal_window import FrozenWithdrawalWindow
from scripts.probes.prepared_withdrawal import head_closure, scalar_closure


CONSTRAINTS = ('head_volume', 'head_render', 'head_silhouette', 'head_merit',
               'coast_volume', 'coast_render', 'coast_silhouette', 'coast_stored')
EPS = torch.finfo(torch.float32).eps


class SearchFailure(RuntimeError):
    """The driver can retain this incremental report while preserving the cause."""
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def allowance(value):
    if not math.isfinite(value):
        raise ValueError('Nonfinite scalar allowance')
    return 32*EPS*max(abs(value), 1e-12)


def joint_trust_radius(history, nodes):
    rms = history.get('body_update_modes_rms')
    if (not isinstance(rms, (list, tuple)) or len(rms) != 2 or type(nodes) is not int or nodes <= 0
            or any(type(x) not in (float, int) or not math.isfinite(x) or x < 0 for x in rms)):
        raise ValueError('Invalid original two-mode update RMS')
    radius = math.sqrt(nodes)*math.hypot(*rms)
    if not math.isfinite(radius):
        raise ValueError('Nonfinite original joint trust radius')
    return radius


def project_joint(coefficients):
    """The actual production per-node SIX-vector radius, with native arithmetic."""
    result = coefficients.clone()
    result.div_(result.norm(dim=1, keepdim=True).clamp_min(1.0))
    return result


def energies(values, dt):
    """Per-ID mean squared speeds; includes first coast displacement, not V0."""
    X, V = values['coast_X'].double(), values['coast_V'][1:].double()
    return dict(geometric=((X[1:]-X[:-1])/dt).square().sum(-1).mean(0),
                stored=V.square().sum(-1).mean(0))


def energy_comparison(current, origin, masks):
    result = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        row = dict(particles=count, geometric=None, stored=None)
        if count:
            for key in ('geometric', 'stored'):
                actual, before = current[key][mask], origin[key][mask]
                difference = actual-before
                floor = 32*EPS*before.abs().clamp_min(1e-12)
                row[key] = dict(mean=float(actual.mean()), origin_mean=float(before.mean()),
                    mean_change=float(difference.mean()), worsened_ids=int((difference > 0).sum()),
                    worsened_beyond_roundoff_ids=int((difference > floor).sum()),
                    max_increase=float(difference.max()),
                    difference_p95=float(torch.quantile(difference, .95)))
        result[name] = row
    return result


def run_search(packet, *, record, raw_observe):
    """At most 3 originals + 11 trials + 3 confirmations on one private model.

    record(label, values, info) must persist the given same-forward witness
    synchronously. info['arrays'] contains owned device arrays; the remainder is
    JSON-ready. raw_observe returns a JSON-ready dict with a boolean 'passed'.
    Empty secondary arrived-free cohorts remain null; empty primary free cohorts
    stop inconclusively. Ordinary failed gates return a report. Unexpected errors
    preserve the current forward through record, then raise SearchFailure.
    """
    report = dict(scope='pre_assimilation_frozen_policy_joint_body_search_no_adoption',
        status='preparing', constraint_order=list(CONSTRAINTS), baselines=[], proposals=[],
        confirmations=[], errors=[], selected=None, provisional_candidate_found=False,
        candidate_found=False, confirmed=False, adopted=False,
        objective='mean over fixed start-free IDs and all coast steps of |dx/dt|^2',
        coast_policy='Original Fp/pins/layer/bonds; learned controls withdrawn; not actual commit',
        constraint_merit='Derivative uses complete tensor merit; all values/gates use original float callback',
        max_candidate_forwards=11, max_confirmation_forwards=3)
    model = None
    binding = None
    evaluator = packet.get('evaluate_merit')
    try:
        if not callable(record) or not callable(raw_observe) or not callable(evaluator) or not callable(getattr(evaluator, 'terms', None)):
            raise ValueError('Missing callback/complete-merit capability')
        owner = packet['rollout']
        c0 = owner.coefficients.detach().clone()
        if (c0.ndim != 2 or c0.shape[1] != 6 or not len(c0) or c0.dtype != torch.float32
                or not bool(torch.isfinite(c0).all()) or bool((c0.square().sum(-1) > 1+1e-6).any())):
            raise ValueError('Invalid original joint FP32 coefficients')
        if not torch.equal(c0, packet['controls']['body']):
            raise ValueError('Original coefficient binding differs')
        N, T, dt = len(packet['x0']), owner.spec.T, float(packet['dt'])
        if T < 1 or not math.isfinite(dt) or dt <= 0 or dt != owner.spec.prm.dt:
            raise ValueError('Invalid original time discretization')
        for name in ('pins', 'start_arrived'):
            value = packet[name]
            if not torch.is_tensor(value) or value.shape != (N,) or value.dtype != torch.bool or value.device != c0.device:
                raise ValueError('Invalid fixed cohort: '+name)
        masks = dict(start_free=~packet['pins'], start_arrived_free=~packet['pins'] & packet['start_arrived'])
        report['cohorts'] = {name: int(mask.sum()) for name, mask in masks.items()}
        report.update(N=N, T=T, dt=dt, dx=float(owner.spec.prm.dx), nodes=len(c0),
                      lambda_render=float(packet['lambda_render']))
        radius = joint_trust_radius(packet['history'], len(c0))
        report['initial_radius'] = radius
        report['radius_rule'] = 'sqrt(node_count)*hypot(displacement_update_RMS,terminal_update_RMS); no factor 3'
        if not report['cohorts']['start_free']:
            report['status'] = 'inconclusive_empty_start_free'
            return report
        if radius == 0:
            report['status'] = 'inconclusive_zero_original_update'
            return report
        accepted = {k: packet[k].detach().clone() for k in ('positions', 'V', 'F', 'C')}
        binding = evaluator.binding_digest()
        report['merit_binding_before'] = binding
        model = FrozenWithdrawalWindow(owner)
        reference = packet['reference']
        baseline_rows, original_energy = [], None
        grad, jacobian, origin_constraint = None, None, None

        def forward(label, coefficients, *, baseline=False, gradient=False, extra=None):
            """Exactly one forward, with an archival finally before any replacement."""
            values = None
            row = dict(label=label, baseline=baseline, valid=False, passed=False,
                       raw=dict(measured=False), **(extra or {}))
            arrays = dict(coefficients=coefficients.detach().clone(),
                **{name+'_ids': torch.nonzero(mask).flatten() for name, mask in masks.items()})
            current_energy = None
            derivatives = None
            try:
                with torch.set_grad_enabled(gradient):
                    values = model.evaluate(coefficients[:, 3:], coefficients[:, :3])
                    row['health'] = dict(values.get('health', {}), valid=bool(values.get('valid')),
                        pins_exact=bool(values.get('pins_exact')), min_det=float(values['min_det']))
                    row['all_fields_finite'] = all(bool(torch.isfinite(v).all())
                        for v in values.values() if torch.is_tensor(v))
                    if (values['coast_X'].shape != (T+1, N, 3) or values['coast_V'].shape != (T+1, N, 3)):
                        raise ValueError('Invalid full coast layout')
                    row['valid'] = bool(values['valid'] and values['pins_exact'] and row['all_fields_finite'])
                    if baseline:
                        row['head_closure'] = head_closure(values, accepted, owner.spec)
                    if row['valid']:
                        head = reference.terms(values['x'])
                        coast = reference.terms(values['coast_X'][-1])
                        exact = evaluator(values)
                        current_energy = energies(values, dt)
                        objective = current_energy['geometric'][masks['start_free']].mean()
                        stored = current_energy['stored'][masks['start_free']].mean()
                        tensors = (head['volume'], head['render'], head['silhouette'], None,
                                   coast['volume'], coast['render'], coast['silhouette'], stored)
                        scalars = [float(v.detach()) if v is not None else exact['merit'] for v in tensors]
                        row.update(objective=float(objective.detach()), constraints=dict(zip(CONSTRAINTS, scalars)),
                            original_merit=exact,
                            prepared_head={k: float(v.detach()) for k, v in head.items()},
                            prepared_coast_end={k: float(v.detach()) for k, v in coast.items()})
                        row['prepared_head']['weighted_render'] = packet['lambda_render']*row['prepared_head']['render']
                        row['prepared_coast_end']['weighted_render'] = packet['lambda_render']*row['prepared_coast_end']['render']
                        numbers = (row['objective'], *scalars, *row['prepared_head'].values(), *row['prepared_coast_end'].values(), *exact.values())
                        if not all(math.isfinite(v) for v in numbers):
                            raise ValueError('Nonfinite candidate observations')
                        if float(exact['lambda_render']) != float(packet['lambda_render']):
                            raise ValueError('Original lambda binding differs')
                        arrays.update({key+'_coast_energy_per_id': value.detach().clone() for key, value in current_energy.items()})
                        if baseline:
                            row['scalar_closure'] = scalar_closure(exact['merit'], packet['history']['loss'], c0.dtype)
                            row['baseline_gate'] = row['scalar_closure']['passed'] and all(v['passed'] for v in row['head_closure'].values())
                        if gradient:
                            whole = evaluator.terms(values)['merit']
                            targets = (objective, *tensors[:3], whole, *tensors[4:])
                            if not all(torch.is_tensor(v) and v.requires_grad for v in targets):
                                raise ValueError('Missing differentiated objective/constraint channel')
                            gs = [torch.autograd.grad(v, coefficients, retain_graph=i < len(targets)-1)[0].detach().clone()
                                  for i, v in enumerate(targets)]
                            if not all(bool(torch.isfinite(v).all()) for v in gs):
                                raise ValueError('Nonfinite objective/constraint derivative')
                            derivatives = gs[0], torch.stack(gs[1:])
                            arrays['objective_gradient'] = derivatives[0].clone()
                            arrays['constraint_gradients'] = derivatives[1].clone()
                        # Raw observation is made even for a prepared-constraint
                        # failure, without evaluating any later model generation.
                        with torch.no_grad():
                            raw = raw_observe(label, values, baseline=baseline)
                        if not isinstance(raw, dict) or type(raw.get('passed')) is not bool:
                            raise ValueError('Invalid raw observation result')
                        row['raw'] = dict(measured=True, **deepcopy(raw))
                        if baseline:
                            row['passed'] = bool(row['baseline_gate'] and raw['passed'])
                        else:
                            row['nonlinear_constraints'] = {key: value <= ceilings[key] for key, value in row['constraints'].items()}
                            row['objective_decrease'] = dict(reference_min=baseline_min, roundoff_floor=objective_floor,
                                decrease=baseline_min-row['objective'], passed=baseline_min-row['objective'] > objective_floor)
                            row['per_id_energy_change'] = energy_comparison(current_energy, original_energy, masks)
                            row['passed'] = bool(all(row['nonlinear_constraints'].values()) and row['objective_decrease']['passed'] and raw['passed'])
            except Exception as exc:
                row['error'] = dict(type=type(exc).__name__, message=str(exc))
                raise
            finally:
                report['baselines' if baseline else 'confirmations' if label.startswith('confirm_') else 'proposals'].append(row)
                if values is not None:
                    metadata = deepcopy(row)
                    metadata.pop('label')  # Already supplied as the callback argument.
                    record(label, values, dict(metadata, arrays=arrays))
            detached_energy = None if current_energy is None else {k: v.detach() for k, v in current_energy.items()}
            return row, detached_energy, derivatives

        for index in range(3):
            controls = c0.clone().requires_grad_(index == 2)
            row, current, derivatives = forward('baseline_'+str(index), controls, baseline=True, gradient=index == 2)
            baseline_rows.append(row)
            if not row['passed']:
                report['status'] = 'baseline_gate_failed'
                return report
            if index == 2:
                if row['raw'].get('baseline_ready') is not True:
                    raise ValueError('Raw baseline envelope was not frozen after three originals')
                original_energy = {k: v.detach().clone() for k, v in current.items()}
                grad, jacobian = derivatives
                origin_constraint = torch.tensor([row['constraints'][k] for k in CONSTRAINTS], dtype=torch.float64, device=c0.device)
        ceilings = {k: max(row['constraints'][k] for row in baseline_rows) for k in CONSTRAINTS}
        report['baseline_maxima'] = dict(ceilings)
        ceilings = {k: value+allowance(value) for k, value in ceilings.items()}
        accepted_merit_cap = packet['history']['loss']+allowance(packet['history']['loss'])
        ceilings['head_merit'] = min(ceilings['head_merit'], accepted_merit_cap)
        report.update(ceilings=ceilings, accepted_merit_cap=accepted_merit_cap,
            ceiling_rule='max of 3 originals + 32*FP32eps*max(abs(maximum),1e-12); head merit also capped at accepted merit plus same allowance')
        baseline_min = min(row['objective'] for row in baseline_rows)
        objective_floor = allowance(baseline_min)
        report.update(baseline_objective_min=baseline_min, objective_roundoff_floor=objective_floor,
                      origin='baseline_2', gradient_norm=float(grad.double().norm()))
        if not bool((grad != 0).any()):
            report['status'] = 'inconclusive_zero_objective_gradient'
            return report
        bounds = torch.tensor([ceilings[k] for k in CONSTRAINTS], dtype=torch.float64, device=c0.device)-origin_constraint
        report['linear_trials'] = []
        for halving in range(11):
            current_radius = radius*2.**(-halving)
            step, solve = affine_ball_step(grad, jacobian, bounds, current_radius)
            trial = dict(halving=halving, radius=current_radius, solve=solve, forward_ran=False)
            report['linear_trials'].append(trial)
            if step is None:
                continue
            candidate = project_joint((c0.double()+step).to(c0.dtype))
            delta = candidate.double()-c0.double()
            affine = projected_affine_check(delta, jacobian, bounds, c0.dtype)
            norm = float(delta.norm())
            radius_tolerance = 32*EPS*(current_radius+float(c0.double().norm()))
            directional = float((grad.double()*delta).sum())
            trial.update(projected_affine=affine, projected_norm=norm, radius_tolerance=radius_tolerance,
                         projected_directional_derivative=directional,
                         predicted_decrease=-directional, predicted_decrease_floor=objective_floor,
                         prediction_resolved=math.isfinite(directional) and -directional > objective_floor)
            if not affine['passed'] or norm > current_radius+radius_tolerance or not trial['prediction_resolved']:
                trial['status'] = 'projected_step_rejected'
                continue
            trial['forward_ran'] = True
            row, _, _ = forward(f'candidate_h{halving:02d}', candidate,
                extra=dict(halving=halving, radius=current_radius, linear_trial=deepcopy(trial)))
            if not row['passed']:
                continue
            report['selected'] = row['label']
            report['provisional_candidate_found'] = True
            for repeat in range(3):
                forward('confirm_'+str(repeat), candidate,
                        extra=dict(candidate_label=row['label'], identical_controls=True))
            report['confirmed'] = all(r['passed'] for r in report['confirmations'])
            report['candidate_found'] = report['confirmed']
            report['status'] = ('confirmed_candidate_no_adoption' if report['confirmed']
                                else 'confirmation_failed_no_adoption')
            return report
        report['status'] = 'no_candidate_passed'
        return report
    except Exception as exc:
        report['errors'].append(dict(type=type(exc).__name__, message=str(exc)))
        report['status'] = 'error'
        raise SearchFailure(str(exc), report) from exc
    finally:
        if model is not None:
            model.close()
        if binding is not None:
            try:
                report['merit_binding_after'] = evaluator.binding_digest()
                report['merit_binding_unchanged'] = report['merit_binding_after'] == binding
                if not report['merit_binding_unchanged']:
                    report['status'] = 'binding_changed'
                    raise SearchFailure('Prepared merit binding changed', report)
            except SearchFailure:
                raise
            except Exception as exc:
                report['status'] = 'binding_check_failed'
                report['errors'].append(dict(type=type(exc).__name__, message=str(exc)))
                raise SearchFailure('Cannot verify prepared merit binding', report) from exc
