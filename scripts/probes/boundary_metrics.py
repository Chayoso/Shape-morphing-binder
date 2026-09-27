"""Torch-only reporting for a controlled one-step PIC-boundary replay.

The caller must hold every non-position input fixed across the three branches.
This helper checks the specified positions/cohorts; it cannot verify that caller
contract from the terminal states alone. Only JSON scalar telemetry leaves the
input device. No renderer, settling policy or physical-quality metric is used.
"""
from __future__ import annotations

import math

import torch


_BRANCHES = ('pic', 'pic_repeat', 'free_raw')
_STAT_NAMES = ('mean', 'median', 'p95', 'max', 'rms')


def _distribution(values):
    if values.numel() == 0:
        return dict(count=0, **{name: None for name in _STAT_NAMES})
    values = values.double()
    return dict(count=values.numel(), mean=float(values.mean()),
                median=float(torch.quantile(values, .5)), p95=float(torch.quantile(values, .95)),
                max=float(values.max()), rms=float(values.square().mean().sqrt()))


def _norm(v):
    return v.double().reshape(v.shape[0], -1).norm(dim=1) if len(v) else v.new_empty(0, dtype=torch.float64)


def _motion(v, spacing):
    norms = _norm(v)
    return dict(wu=_distribution(norms), sp=_distribution(norms/spacing))


def _signed(values):
    result = _distribution(values)
    result.update(min=float(values.min()) if values.numel() else None,
                  p05=float(torch.quantile(values.double(), .05)) if values.numel() else None)
    return result


def _ratio(numerator, denominator):
    return float(numerator/denominator) if bool(denominator > 0) else None


def _reversal(previous, following, threshold):
    valid = (_norm(previous) > threshold) & (_norm(following) > threshold)
    negative = (previous.double()*following.double()).sum(1) < 0
    count = int(valid.sum())
    reversed_count = int((valid & negative).sum())
    return dict(eligible=count, reversals=reversed_count,
                fraction=float(negative[valid].double().mean()) if count else None), valid, negative


def _reversal_comparison(prior_pic, prior_raw, pic, free_raw, repeat, threshold):
    pic_rev, pic_valid, pic_negative = _reversal(prior_pic, pic, threshold)
    raw_rev, raw_valid, raw_negative = _reversal(prior_raw, free_raw, threshold)
    repeat_rev, _, _ = _reversal(prior_pic, repeat, threshold)
    common = pic_valid & raw_valid
    common_count = int(common.sum())
    return dict(pic=pic_rev, free_raw=raw_rev, pic_repeat=repeat_rev,
        common_eligible=dict(count=common_count,
            pic_fraction=float(pic_negative[common].double().mean()) if common_count else None,
            free_raw_fraction=float(raw_negative[common].double().mean()) if common_count else None,
            pic_only_count=int((common & pic_negative & ~raw_negative).sum()),
            free_raw_only_count=int((common & ~pic_negative & raw_negative).sum())))


def _response_noise(response, noise, spacing, threshold):
    result = dict(response=_motion(response, spacing), pic_repeat_noise=_motion(noise, spacing))
    if not len(response):
        result.update(rms_signal_to_noise=None, noise_status='empty_cohort',
                      fraction_above_pointwise_repeat_and_threshold=None)
        return result
    signal_norm, noise_norm = _norm(response), _norm(noise)
    signal_rms, noise_rms = signal_norm.square().mean().sqrt(), noise_norm.square().mean().sqrt()
    result.update(rms_signal_to_noise=_ratio(signal_rms, noise_rms),
                  noise_status='positive_measured_repeat_noise' if bool(noise_rms > 0)
                  else 'zero_measured_repeat_noise',
                  fraction_above_pointwise_repeat_and_threshold=float(
                      (signal_norm > noise_norm.clamp_min(threshold)).double().mean()))
    return result


def _alignment(response, boundary_j, spacing, threshold):
    response, opposing = response.double(), -boundary_j.double()
    rn, jn = _norm(response), _norm(opposing)
    j_valid = jn > threshold
    pair_valid = j_valid & (rn > threshold)
    dot = (response*opposing).sum(1)
    total_dot = dot.sum()
    denom = opposing.square().sum()
    cosine_denom = (response.square().sum()*denom).sqrt()
    parallel = dot[j_valid]/jn[j_valid]
    cosine = (dot[pair_valid]/(rn[pair_valid]*jn[pair_valid])).clamp(-1, 1)
    return dict(global_projection_coefficient=_ratio(total_dot, denom),
                global_cosine=_ratio(total_dot, cosine_denom),
                boundary_direction_count=int(j_valid.sum()),
                direction_pair_count=int(pair_valid.sum()),
                parallel_displacement_wu=_signed(parallel),
                parallel_displacement_sp=_signed(parallel/spacing),
                cosine=_signed(cosine),
                positive_cosine_fraction=float((cosine > 0).double().mean()) if len(cosine) else None)


def _state_difference(pic, other, name, mask, spacing):
    width = 3 if name == 'v1' else 9
    delta = pic[name][mask].reshape(-1, width) - other[name][mask].reshape(-1, width)
    norms = _norm(delta)
    if name == 'v1':
        return dict(wu_per_s=_distribution(norms), sp_per_s=_distribution(norms/spacing))
    return dict(units='dimensionless' if name == 'F1' else '1/s', frobenius=_distribution(norms))


def _validate(data, branches, dt, spacing):
    if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    if isinstance(spacing, bool) or not math.isfinite(spacing) or spacing <= 0:
        raise ValueError('spacing must be finite and positive')
    if set(branches) != set(_BRANCHES):
        raise ValueError(f'branches must be exactly {_BRANCHES}')
    reference = data['previous']
    if not torch.is_tensor(reference) or reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError('previous must be a Torch (N,3) tensor')
    if reference.dtype not in (torch.float32, torch.float64) or reference.device.type not in ('cpu', 'cuda'):
        raise ValueError('positions must be float32/float64 on CPU/CUDA')
    n = len(reference)
    def check(tensor, label, shapes, boolean=False):
        if not torch.is_tensor(tensor) or tuple(tensor.shape) not in shapes:
            raise ValueError(f'{label} has invalid shape/type')
        if tensor.device != reference.device or tensor.dtype != (torch.bool if boolean else reference.dtype):
            raise ValueError(f'{label} must match the positions device and expected dtype')
        if not boolean and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f'{label} must be finite')
    for key in ('previous', 'raw', 'promoted'):
        check(data[key], key, {(n, 3)})
    for key in ('pin', 'layer_mask'):
        check(data[key], key, {(n,)}, boolean=True)
    for arm in _BRANCHES:
        for key in ('x0', 'x1', 'pre_layer', 'v1'):
            check(branches[arm][key], f'{arm}.{key}', {(n, 3)})
        for key in ('F1', 'C1'):
            check(branches[arm][key], f'{arm}.{key}', {(n, 9), (n, 3, 3)})
    expected_free = torch.where(data['pin'][:, None], data['promoted'], data['raw'])
    for arm in _BRANCHES:
        expected = expected_free if arm == 'free_raw' else data['promoted']
        if not torch.equal(branches[arm]['x0'], expected):
            raise ValueError(f'{arm}.x0 violates the fixed boundary intervention')


@torch.no_grad()
def summarize_boundary(data, branches, dt, spacing):
    """Return JSON scalars for fixed next-window free and layer-free particle IDs.

    data: previous/raw/promoted (N,3), next-window pin/layer_mask bool (N,).
    Each pic/pic_repeat/free_raw branch: x0/x1/pre_layer/v1 (N,3), F1/C1
    (N,9) or (N,3,3). All tensors share position dtype/device. Only free_raw
    starts from raw, with NEXT-window pins held at promoted. State differences
    are reported; initial non-position state equality remains the caller's duty.
    """
    _validate(data, branches, dt, spacing)
    pin = data['pin']
    j = data['promoted']-data['raw']
    last_raw = data['raw']-data['previous']
    last_saved = last_raw+j
    step = {arm: values['x1']-values['x0'] for arm, values in branches.items()}
    applied_j = branches['pic']['x0']-branches['free_raw']['x0']
    response = step['pic']-step['free_raw']
    repeat_noise = step['pic_repeat']-step['pic']
    endpoint_difference = branches['pic']['x1']-branches['free_raw']['x1']
    components = {}
    for arm, values in branches.items():
        advection = dt*values['v1']
        components[arm] = dict(advection=advection,
            pre_layer_residual=values['pre_layer']-values['x0']-advection,
            layer=values['x1']-values['pre_layer'])
    induced_components = dict(
        advection=dt*(branches['pic']['v1']-branches['free_raw']['v1']),
        pre_layer_residual=components['pic']['pre_layer_residual']-components['free_raw']['pre_layer_residual'],
        layer=components['pic']['layer']-components['free_raw']['layer'])
    induced_closure = (induced_components['advection']+induced_components['pre_layer_residual']
                       +induced_components['layer']-response)
    threshold = spacing*1e-4
    result = dict(N=len(pin), dt=float(dt), spacing=float(spacing), direction_threshold_wu=float(threshold),
                  definitions=dict(
                      cohorts='same NEXT-window unpinned IDs; layer_free additionally uses frozen NEXT layer_mask',
                      boundary_j='promoted-raw; applied x0 intervention is zero on NEXT-window pins',
                      induced_response='(pic.x1-pic.x0)-(free_raw.x1-free_raw.x0), not pic.x1-free_raw.x1',
                      induced_components='pic minus free_raw advection, pre-layer residual and layer displacement; vector sum equals induced response up to arithmetic roundoff, but components can cancel',
                      saved_last_step='raw-previous+j; alternative last step is raw-previous',
                      pre_layer_residual='pre_layer-x0-dt*v1; not attributed to pure bond motion',
                      alignment='positive projection on -j means induced next response opposes the preceding PIC remap',
                      global_alignment='whole-cohort dot/norm reductions; per-particle directions use the stated length threshold',
                      repeat_noise='one identical PIC replay; a measured numerical floor, not a confidence bound',
                      reversal='negative dot product, with BOTH vector lengths strictly >1e-4*spacing',
                      whole_alternative_reversal='PIC prior is raw-previous+j; free_raw prior is raw-previous; BOTH prior and next vectors change',
                      shared_preceding_reversal='both branches use raw-previous+j as the same prior vector; only the next response changes',
                      state_response='terminal F/v/C differences; caller must prove equal initial non-position inputs'),
                  no_physical_quality_claim=True, pin_checks={}, cohorts={})
    for arm, values in branches.items():
        drift = values['x1'][pin]-values['x0'][pin]
        result['pin_checks'][arm] = dict(count=int(pin.sum()),
            position_exact=torch.equal(values['x1'][pin], values['x0'][pin]),
            pre_layer_position_exact=torch.equal(values['pre_layer'][pin], values['x0'][pin]),
            max_position_drift_wu=float(_norm(drift).max()) if len(drift) else None,
            v1_zero=bool(torch.count_nonzero(values['v1'][pin]) == 0),
            C1_zero=bool(torch.count_nonzero(values['C1'][pin]) == 0),
            F1_same_as_pic=torch.equal(values['F1'][pin].reshape(-1, 9), branches['pic']['F1'][pin].reshape(-1, 9)))
    result['pin_checks']['F1_scope'] = 'cross-branch equality only; initial F is not supplied'
    for name, mask in (('all_free', ~pin), ('layer_free', ~pin & data['layer_mask'])):
        count = int(mask.sum())
        cohort = dict(count=count, boundary_j=_motion(j[mask], spacing),
                      applied_boundary=_motion(applied_j[mask], spacing),
                      last_raw_step=_motion(last_raw[mask], spacing),
                      last_saved_step=_motion(last_saved[mask], spacing), branches={})
        for arm, values in branches.items():
            advection = components[arm]['advection']
            pre_residual = components[arm]['pre_layer_residual']
            layer = components[arm]['layer']
            closure = advection+pre_residual+layer-step[arm]
            cohort['branches'][arm] = dict(
                advection=_motion(advection[mask], spacing),
                pre_layer_residual=_motion(pre_residual[mask], spacing),
                layer=_motion(layer[mask], spacing), total=_motion(step[arm][mask], spacing),
                component_closure=_motion(closure[mask], spacing))
        cohort['induced_next_response'] = _response_noise(response[mask], repeat_noise[mask], spacing, threshold)
        cohort['induced_components'] = {
            component: dict(**_motion(delta[mask], spacing),
                            alignment_with_negative_j=_alignment(delta[mask], j[mask], spacing, threshold))
            for component, delta in induced_components.items()}
        cohort['induced_components']['sum_vs_induced_response_closure'] = _motion(induced_closure[mask], spacing)
        cohort['endpoint_difference'] = _motion(endpoint_difference[mask], spacing)
        cohort['endpoint_identity_residual'] = _motion((endpoint_difference-applied_j-response)[mask], spacing)
        cohort['alignment_with_negative_j'] = _alignment(response[mask], j[mask], spacing, threshold)
        cohort['reversals'] = _reversal_comparison(
            last_saved[mask], last_raw[mask], step['pic'][mask], step['free_raw'][mask],
            step['pic_repeat'][mask], threshold)
        cohort['reversals']['scope'] = 'whole alternative boundary: both prior and next vectors change'
        cohort['reversals']['shared_preceding_saved'] = _reversal_comparison(
            last_saved[mask], last_saved[mask], step['pic'][mask], step['free_raw'][mask],
            step['pic_repeat'][mask], threshold)
        cohort['state_differences'] = {
            label: {field: _state_difference(branches[a], branches[b], field, mask, spacing)
                    for field in ('v1', 'F1', 'C1')}
            for label, a, b in (('pic_minus_free_raw', 'pic', 'free_raw'),
                                ('pic_repeat_minus_pic', 'pic_repeat', 'pic'))}
        result['cohorts'][name] = cohort
    return result
