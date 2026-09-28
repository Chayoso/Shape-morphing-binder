"""Read-only decomposition of the actual fixed-stencil PIC endpoint correction.

All numerical arrays and reductions remain on the input Torch device. Only JSON
scalars/lists cross the reporting boundary. Candidate endpoints are diagnostics,
not replayed physics or evidence of future motion/quality.
"""
from __future__ import annotations

import math

import torch

NAMES = ('mpm_advection', 'bond_residual', 'surface_u', 'layer_residual')
ROUNDOFF_EPS = 64
NOISE_FACTOR = 10


@torch.no_grad()
def decompose_pic(operator, start, raw, promoted, pin, sums, layer_mask, spacing):
    """Split J into Q(H-I) components using the supplied owned start operator.

    ``sums`` is (4,N,3), in ``motion_accounting.NAMES`` order. Pins are the
    window-start bool mask. The free/layer-free cohorts are fixed here, before
    any candidate endpoint is considered. The original operator and all inputs
    are read-only; returned endpoints own their storage.
    """
    if not torch.is_tensor(start) or start.ndim != 2 or start.shape[1] != 3 or len(start) == 0:
        raise ValueError('start must be nonempty Nx3 Torch positions')
    if start.dtype not in (torch.float32, torch.float64):
        raise ValueError('positions must use float32 or float64')
    n = len(start)
    for name, value, shape in (('raw', raw, start.shape), ('promoted', promoted, start.shape),
                               ('sums', sums, (4, n, 3))):
        if (not torch.is_tensor(value) or value.shape != shape or value.device != start.device
                or value.dtype != start.dtype):
            raise ValueError(f'{name} shape/device/dtype does not match start')
    if (not torch.is_tensor(pin) or pin.shape != (n,) or pin.device != start.device
            or pin.dtype != torch.bool):
        raise ValueError('pin must be a same-device window-start bool mask')
    if not torch.is_tensor(layer_mask) or layer_mask.shape != (n,) or layer_mask.device != start.device:
        raise ValueError('layer_mask must be a same-device particle mask')
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError('spacing must be finite and positive')
    if not all(bool(torch.isfinite(value).all()) for value in (start, raw, promoted, sums, layer_mask)):
        raise ValueError('component inputs must be finite')
    if not torch.equal(operator.x0, start):
        raise ValueError('operator reference differs from the actual window start')
    if not torch.equal(raw[pin], start[pin]):
        raise ValueError('raw endpoint changed a window-start pin')
    if not torch.equal(promoted[pin], start[pin]):
        raise ValueError('owned promoted endpoint changed a window-start pin')

    free = ~pin
    layer = layer_mask if layer_mask.dtype == torch.bool else layer_mask >= .5
    displacement = raw - start
    actual = promoted - raw
    packed = sums.permute(1, 0, 2).reshape(n, 12)
    filtered = operator.apply_H(packed).reshape(n, 4, 3).permute(1, 0, 2)
    # Q is applied AFTER H: pinned columns/masses still belong to the operator.
    jumps = torch.where(free[None, :, None], filtered - sums, 0.)
    h_displacement = operator.apply_H(displacement)
    h_repeat = operator.apply_H(displacement)
    map_jump = torch.where(free[:, None], h_displacement - displacement, 0.)
    repeat = torch.where(free[:, None], h_repeat - h_displacement, 0.)
    component_sum_error = sums.sum(0) - displacement
    original_error = map_jump - actual
    linearity_error = jumps.sum(0) - map_jump
    total_error = jumps.sum(0) - actual

    # Subtracting stored positions carries their rounding scale; displacement
    # alone is an insufficient absolute scale near rest. Report allowance and
    # measured repeat separately, rather than calling allowance measured noise.
    scale = torch.maximum(torch.maximum(start.abs(), raw.abs()), sums.abs().sum(0)).clamp_min(spacing)
    base = ROUNDOFF_EPS * torch.finfo(start.dtype).eps * scale
    allowance = base + NOISE_FACTOR * repeat.abs()

    def gate(error, tolerance):
        return dict(passed=bool((error.abs() <= tolerance).all()),
                    max_abs_wu=float(error.abs().max()),
                    rms_wu=float(error.double().square().sum(-1).mean().sqrt()),
                    max_tolerance_ratio=float((error.abs() / tolerance).max()))

    gates = dict(component_sum_equals_displacement=gate(component_sum_error, base),
                 original_operator_equals_owned_endpoint=gate(original_error, allowance),
                 component_linearity=gate(linearity_error, allowance),
                 component_jump_sum_equals_actual=gate(total_error, 2 * allowance),
                 pins_exact=True)
    valid = all(value['passed'] for value in gates.values() if isinstance(value, dict))
    endpoints = dict(raw=raw.clone(), current=promoted.clone(),
                     advection_only=torch.where(pin[:, None], start, raw + jumps[0]).clone(),
                     preserve_relaxation=torch.where(pin[:, None], start, raw + jumps[:3].sum(0)).clone())
    report = dict(
        valid=valid, N=n, pinned_count=int(pin.sum()), spacing_wu=float(spacing), dtype=str(start.dtype),
        component_names=list(NAMES),
        definition='D=raw-start; J_i=Q(H-I)s_i; actual J=owned promoted-raw; Q follows H',
        cohorts_definition='window-start unpinned IDs; layer_free also uses frozen layer_mask>=0.5',
        residual_definition='bond/layer residuals include their recorded arithmetic roundoff',
        scope='same accepted rollout, frozen stencil and components; candidate geometry only, no future response',
        numerical_policy=dict(roundoff_eps_multiple=ROUNDOFF_EPS, repeat_factor=NOISE_FACTOR,
                              base_scale='max(abs(start),abs(raw),sum(abs(components)),spacing) per coordinate',
                              repeat_scope='two D evaluations on the same supplied operator',
                              resolution='candidate-current RMS >10*max(original error, repeat, linearity error, roundoff allowance RMS)',
                              ratio_floor='actual-jump RMS must exceed the same 10x numerical context; otherwise fractions are null',
                              independent_repeat_count=1),
        gates=gates, cohorts={})

    def rms(value):
        return value.square().sum(-1).mean().sqrt()

    def magnitude(value):
        result = rms(value)
        return dict(rms_wu=float(result), rms_sp=float(result / spacing))

    for name, mask in (('all_free', free), ('layer_free', free & layer)):
        count = int(mask.sum())
        if not count:
            report['cohorts'][name] = dict(particles=0, interpretation='empty_cohort')
            continue
        j = actual[mask].double()
        comp = jumps[:, mask].double()
        energy = j.square().sum(-1).mean()
        gram = torch.einsum('inc,jnc->ij', comp, comp) / count
        fraction = (comp * j[None]).sum((1, 2)) / count
        closure = dict(original_operator_error=magnitude(original_error[mask].double()),
                       repeat_error=magnitude(repeat[mask].double()),
                       component_linearity_error=magnitude(linearity_error[mask].double()),
                       roundoff_allowance=magnitude(base[mask].double()))
        noise = max(value['rms_wu'] for value in closure.values())
        actual_resolved = float(energy.sqrt()) > NOISE_FACTOR * noise
        entries = {}
        for index, component in enumerate(NAMES):
            entries[component] = dict(**magnitude(comp[index]),
                signed_projection_onto_actual_wu2=float(fraction[index]),
                signed_fraction=float(fraction[index] / energy) if actual_resolved else None)
        a, r = comp[0], comp[1:].sum(0)
        aa, rr, ar = a.square().sum(-1).mean(), r.square().sum(-1).mean(), (a * r).sum(-1).mean()
        candidates = {}
        for candidate, endpoint in endpoints.items():
            jump = (endpoint - raw)[mask].double()
            change = (endpoint - promoted)[mask].double()
            signal = float(rms(change))
            resolved = valid and signal > NOISE_FACTOR * noise
            candidates[candidate] = dict(**magnitude(jump),
                signed_fraction=float((jump * j).sum(-1).mean() / energy) if actual_resolved else None,
                change_from_current=magnitude(change),
                change_exceeds_10x_numerical_context=resolved,
                interpretation=('closure_failed' if not valid else 'resolved_endpoint_change' if resolved
                                else 'inconclusive_at_numerical_context'))
        report['cohorts'][name] = dict(
            particles=count, actual_jump=magnitude(j), actual_jump_above_numerical_floor=actual_resolved,
            components=entries,
            gram_mean_wu2=gram.tolist(), gram_mean_sp2=(gram / spacing**2).tolist(),
            component_diagonal_energy_wu2=float(gram.diag().sum()),
            pair_cross_energy_wu2=float(gram.sum() - gram.diag().sum()),
            summed_component_energy_wu2=float(gram.sum()), actual_jump_energy_wu2=float(energy),
            advection_residual=dict(advection=magnitude(a), residual=magnitude(r),
                advection_energy_wu2=float(aa), residual_energy_wu2=float(rr),
                dot_mean_wu2=float(ar), cross_energy_wu2=float(2 * ar),
                cancellation_fraction=(float(2 * ar / (aa + rr))
                                       if float((aa + rr).sqrt()) > NOISE_FACTOR * noise else None),
                advection_to_actual_rms=float((aa / energy).sqrt()) if actual_resolved else None),
            numerical_context=closure, candidates=candidates)
    return dict(report=report, endpoints=endpoints)
