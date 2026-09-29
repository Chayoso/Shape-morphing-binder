"""Experimental live-merit search registration; actual continuation is separate.

The only selectable numerical state comes from the search's last confirmed
forward. The ordinary runner receives its raw head and applies its own handoff.
"""
import math

import torch

from .post_assimilation_window import PostAssimilationWindow
from .window_selection import _own


def search(context, successor, *, record, raw_observe):
    # Import the bounded experiment driver only for this opt-in capability.
    from scripts.probes.withdrawal_search_core import run_search

    context._live()
    if (len(context._choices) != 1 or context._post_pins is not None
            or context._accepted_velocity is None):
        raise ValueError('Post-assimilation search requires an unused live context with accepted V')
    evaluator = context._evaluate_merit
    if (not callable(record) or not callable(raw_observe)
            or not callable(getattr(evaluator, 'terms', None))
            or not callable(getattr(evaluator, 'binding_digest', None))):
        raise ValueError('Post-assimilation search requires complete live merit and evidence callbacks')
    cfg, owner = context._cfg, context._owner
    arrived = context._original[5].get('arrived_mask')
    if arrived is None:
        raise ValueError('Missing current prepared start-arrived cohort')
    arrived = torch.as_tensor(arrived, device=context._start.device)
    if arrived.dtype != torch.bool or arrived.shape != context._pins.shape:
        raise ValueError('Invalid prepared start-arrived cohort')
    model = PostAssimilationWindow(owner, successor, cfg)
    # This lazy model checks the registration boundary; run_search owns its own
    # numerical model and closes it before returning. No second forward here.
    context._model.close()
    context._model = model
    context._post_pins = model._next_pins.detach().clone()
    original = context._choices[context._identity]['values']
    packet = dict(rollout=owner, controls=dict(body=owner.coefficients.detach().clone()),
        x0=context._start.detach().clone(), positions=original['positions'].detach().clone(),
        V=context._accepted_velocity.detach().clone(), F=original['F'].detach().clone(),
        C=original['C'].detach().clone(), pins=context._pins.detach().clone(),
        start_arrived=arrived.detach().clone(), dt=owner.spec.prm.dt,
        lambda_render=float(context._original[4][-1].get('lambda') or 0.),
        history=_own(context._original[4][-1]), reference=context._reference,
        evaluate_merit=evaluator)
    binding = evaluator.binding_digest()
    confirmed = []

    def receive(coefficients, values, info):
        context._live()
        report = info.get('report', {})
        if (confirmed or info.get('label') != 'confirm_2' or not report.get('confirmed')
                or not report.get('candidate_found') or evaluator.binding_digest() != binding):
            raise ValueError('Invalid confirmed search receipt')
        confirmed.append((_own(coefficients), _own(values), _own(info)))

    report = run_search(packet, record=record, raw_observe=raw_observe,
                        successor=successor, cfg=cfg, on_confirmed=receive)
    context._live()
    if (evaluator.binding_digest() != binding
            or report.get('merit_binding_unchanged') is False
            or (report.get('candidate_found') and report.get('merit_binding_unchanged') is not True)):
        raise ValueError('Current full-merit binding changed during search')
    if not report.get('candidate_found'):
        if confirmed:
            raise ValueError('Rejected search supplied a confirmed candidate')
        return context.original(), report
    if not report.get('confirmed') or len(confirmed) != 1:
        raise ValueError('Confirmed search must provide exactly one same-forward snapshot')
    coefficients, values, info = confirmed[0]
    if (info.get('candidate_label') != report.get('selected')
            or coefficients.shape != owner.coefficients.shape
            or coefficients.device != owner.coefficients.device
            or coefficients.dtype != owner.coefficients.dtype
            or not bool(torch.isfinite(coefficients).all())
            or bool((coefficients.square().sum(-1) > 1+1e-6).any())):
        raise ValueError('Confirmed coefficients or candidate identity differ')
    failures, post_det = context._health(values)
    metrics = None
    if not failures:
        with torch.no_grad():
            metrics = _own(evaluator(values))
        if not all(math.isfinite(float(v)) for v in metrics.values()):
            failures.append('nonfinite_merit')
        elif not context._floor <= metrics['merit'] <= context._ceiling:
            failures.append('original_merit_or_pace')
    choice, label = object(), 'post_assimilation_confirm_2'
    context._labels.add(label)
    entry = dict(label=label, identity=False, eligible=not failures, failures=failures,
        values=values, metrics=metrics, coefficients=coefficients, result=None,
        certificate=dict(passed=not failures, search=_own(report),
            source_label=info['label'], source_generation=info['generation'],
            scope='Fixed estimated successor only; actual admission/coast/continuation not certified'))
    context._choices[choice] = entry
    if entry['eligible']:
        entry['result'] = context._candidate_result(entry, post_det)
    return choice, report
