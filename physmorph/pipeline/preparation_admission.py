"""Shared accepted-displacement reversal and transit-ray admission primitives.

The restricted preview owns current-run history. It does not commit, assimilate,
release pins, or predict a different prefix's history.
"""
from physmorph.compute import array_api as np, KDTree
from .settlement import pin_arrival_evidence


def smooth_material(value, neighbors):
    if neighbors is None:
        return value
    return (value + value[neighbors].sum(1)) / float(neighbors.shape[1] + 1)


def update_reversal(displacement, previous, scale, reversals, frozen, neighbors,
                    arrived_start, dx):
    """Exact ordinary arithmetic, returning owned state rather than mutating input."""
    displacement = np.asarray(displacement, np.float32).copy()
    if scale is None or len(scale) != len(displacement):
        scale = np.ones(len(displacement), np.float32)
        reversals = np.zeros(len(displacement), np.int32)
        frozen = np.zeros(len(displacement), bool)
    else:
        scale, reversals, frozen = scale.copy(), reversals.copy(), frozen.copy()
    active = np.zeros(len(displacement), bool)
    flip = np.zeros(len(displacement), bool)
    if previous is not None and len(previous) == len(displacement):
        now, prev = smooth_material(displacement, neighbors), smooth_material(previous, neighbors)
        n0 = np.linalg.norm(now, axis=1); n1 = np.linalg.norm(prev, axis=1)
        tiny = 1e-4 * float(dx)
        active = (n0 > tiny) & (n1 > tiny)
        cosine = (now * prev).sum(1) / np.maximum(n0 * n1, 1e-30)
        flip = active & (cosine < 0.0); same = active & (cosine >= 0.0)
        if arrived_start is not None and len(arrived_start) == len(flip):
            flip = flip & np.asarray(arrived_start, bool)
        scale[flip] *= 0.5
        reversals[flip] += 1
        scale[same] = np.minimum(1.0, scale[same] * 1.2)
    return dict(displacement=displacement, scale=scale, reversals=reversals, frozen=frozen,
                scale_apply=smooth_material(scale, neighbors), active=active, flip=flip)


def reversal_candidates(arrived, reversals, settled):
    return arrived & (reversals >= 2) & (~settled)


def exclude_transit_rays(x, images, arrived, newly, support_radius):
    """Caller supplies the existing max(pace radius, 2 dx) support radius."""
    xq = np.asarray(x, np.float32)[~arrived]
    dq = np.asarray(images, np.float32)[~arrived] - xq
    length = np.linalg.norm(dq, axis=1)
    ns = np.maximum(1, np.ceil(length / float(support_radius)).astype(np.int64))
    rep = np.repeat(np.arange(len(xq)), ns + 1)
    k = np.arange(len(rep)) - np.repeat(np.cumsum(np.concatenate([np.asarray([0], dtype=ns.dtype), ns[:-1] + 1])), ns + 1)
    s = (k / np.repeat(ns, ns + 1)).astype(np.float32)[:, None]
    samples = xq[rep] + s * dq[rep]
    distance, _ = KDTree(samples).query(np.asarray(x, np.float32)[newly], k=1,
                                       distance_upper_bound=float(support_radius), workers=-1)
    clear = ~np.isfinite(distance)
    indices = np.nonzero(newly)[0]
    result = np.zeros_like(newly); result[indices[clear]] = True
    return dict(newly=result, samples=samples, blocked_fraction=float(1.0-clear.mean()))


def accumulate_pins(settled, settled_at, newly, window_index):
    settled, settled_at = settled.copy(), settled_at.copy()
    settled_at[newly] = window_index + 1
    settled |= newly
    return settled, settled_at


class AdmissionHistory:
    """Opaque owned snapshot; exported arrays are fresh copies."""
    def __init__(self, *, scale, previous, reversals, frozen, settled, settled_at,
                 pins, neighbors):
        self.__arrays = {name: None if value is None else np.asarray(value).copy()
                         for name, value in locals().copy().items() if name != 'self'}

    def arrays(self):
        return {key: None if value is None else value.copy() for key, value in self.__arrays.items()}


def validate_preview_config(cfg):
    from .window_selection import validate_config
    validate_config(cfg)
    required = ('ctrl_rprop', 'ctrl_rprop_smooth', 'ctrl_rprop_arrived', 'settle_pin',
                'settle_pin_ray', 'settle_pin_assim', 'settle_pin_slip')
    unsupported = ('body_rprop', 'freeze_arrived', 'settle_eta', 'settle_pin_confirm',
                   'settle_pin_clear', 'settle_pin_still', 'settle_pin_stuck', 'settle_pin_follow',
                   'settle_pin_yield', 'settle_pin_kkt', 'settle_pin_kkt_dry', 'assim_consensus',
                   'w_grow', 'layer_F', 'ctrl_rprop_hold_onset')
    if (not all(getattr(cfg, name, False) for name in required)
            or any(getattr(cfg, name, False) for name in unsupported) or cfg.ctrl_rprop_k != 8):
        raise ValueError('Admission preview requires the registered reversal/ray policy without alternative state changes')


def preview_active_admission(history, x_start, x_accepted, plan_img, pace_r,
                             arrived_start, *, cfg, dx, window_index):
    """Conditional accepted-original preview; ordinary outer acceptance is separate."""
    validate_preview_config(cfg)
    if not isinstance(history, AdmissionHistory):
        raise ValueError('Admission preview requires owned current history')
    state = history.arrays()
    start, end = np.asarray(x_start, np.float32), np.asarray(x_accepted, np.float32)
    n = len(start)
    if (start.shape != (n, 3) or end.shape != start.shape
            or not np.isfinite(start).all() or not np.isfinite(end).all()
            or plan_img is None or np.shape(plan_img) != start.shape or not np.isfinite(plan_img).all()
            or arrived_start is None or np.shape(arrived_start) != (n,)
            or pace_r is None or not np.isfinite(pace_r) or pace_r <= 0
            or not np.isfinite(dx) or dx <= 0 or type(window_index) is not int or window_index < 0):
        raise ValueError('Invalid current-window admission evidence')
    for name in ('scale', 'reversals', 'frozen', 'settled', 'settled_at', 'pins'):
        if state[name] is None or state[name].shape != (n,):
            raise ValueError('Missing initialized current history: '+name)
    if (state['previous'] is None or state['previous'].shape != start.shape
            or state['neighbors'] is None or state['neighbors'].shape != (n, 8)
            or state['neighbors'].dtype.kind not in 'iu'
            or (state['neighbors'] < 0).any() or (state['neighbors'] >= n).any()):
        raise ValueError('Current reversal history/neighborhood missing or invalid')
    if (state['scale'].dtype != np.dtype(np.float32) or state['previous'].dtype != np.dtype(np.float32)
            or not np.isfinite(state['scale']).all() or not np.isfinite(state['previous']).all()
            or (state['scale'] < 0).any() or (state['scale'] > 1).any()
            or state['settled'].dtype != np.dtype(bool) or state['frozen'].dtype != np.dtype(bool)
            or state['reversals'].dtype != np.dtype(np.int32) or state['settled_at'].dtype != np.dtype(np.int32)
            or (state['reversals'] < 0).any()):
        raise ValueError('Current history dtype or numerical values invalid')
    if (not np.isin(state['pins'], [0, 1]).all()
            or not np.array_equal(state['settled'], state['pins'] > .5)
            or state['frozen'].any()):
        raise ValueError('Current settled/pin state is inconsistent or contains frozen particles')
    result = update_reversal(end-start, state['previous'], state['scale'], state['reversals'],
                             state['frozen'], state['neighbors'], arrived_start, dx)
    arrival = pin_arrival_evidence(end, plan_img, pace_r, arrived_start, require_start=False)
    newly = reversal_candidates(arrival.eligible, result['reversals'], state['settled'])
    telemetry = arrival.telemetry()
    samples = None
    if newly.any() and (~arrival.eligible).any():
        ray = exclude_transit_rays(end, plan_img, arrival.eligible, newly, max(float(pace_r), 2.0*float(dx)))
        newly, samples = ray['newly'], ray['samples']
        telemetry['pin_ray_blocked_frac'] = ray['blocked_fraction']
    settled, settled_at = accumulate_pins(state['settled'], state['settled_at'], newly, window_index)
    result['scale'][settled] = 0.0
    result['scale_apply'] = np.asarray(result['scale_apply'], np.float32).copy()
    result['scale_apply'][settled] = 0.0
    return dict(**result, settled=settled, settled_at=settled_at, newly=newly,
                pins=settled.astype(np.float32), arrival=arrival, ray_samples=samples, telemetry=telemetry,
                scope='Passive successor admission only; u_rprop history and next controlled optimization bounds are not previewed')
