"""Current original-head passive preparation, with no borrowed successor state.

Admission and geometric policies are detached previews. No derivative through
their discrete decisions, next controlled optimizer state, or adoption is claimed.
"""
from copy import deepcopy

import torch

from physmorph.compute import array_api as np, to_array
from ..mpm.traj import Trajectory
from ..mpm.withdrawal import OwnedWithdrawal
from ..plasticity.assimilation_adjoint import assimilate_handoff
from .preparation_admission import preview_active_admission
from .preparation_geometry import prepare_bonds, prepare_layer_geometry


def prepare_original_successor(context, history):
    """Prepare an owned passive successor from this live context's original head.

    The OT-derived u gate is deliberately neutral: u=0 and layer_F is forbidden.
    Compare this field separately rather than reporting full policy-array identity.
    The original choice/result and ordinary runner history are never modified.
    """
    context._live()
    cfg, spec = context._cfg, context._owner.spec
    if (len(context._choices) != 1 or context._post_pins is not None
            or spec.vol0 is None or spec.Fp is None
            or spec.prm.gate_r_hi > spec.prm.gate_r_lo):
        raise ValueError('Current preparation requires unused original context and fixed material state')
    original = context._original
    frames, _, end, _, _, stats = original
    x = np.ascontiguousarray(to_array(frames[-1], copy=True), np.float32)
    start = np.ascontiguousarray(to_array(frames[0], copy=True), np.float32)
    predicted = preview_active_admission(history, start, x, stats.get('plan_img'),
        stats.get('pace_r'), stats.get('arrived_mask'), cfg=cfg, dx=spec.prm.dx,
        window_index=context._window)
    device = context._start.device
    old = torch.as_tensor(history.arrays()['pins'], device=device) > .5
    if not torch.equal(old, context._pins):
        raise ValueError('Current admission history does not match the head pin state')
    pins = torch.as_tensor(predicted['pins'], device=device) > .5
    new = torch.as_tensor(predicted['newly'], device=device)
    F = torch.as_tensor(to_array(end['F']), device=device).reshape(-1, 3, 3)
    P = torch.as_tensor(to_array(spec.Fp), device=device).reshape_as(F)
    with torch.no_grad():
        next_P = assimilate_handoff(F, P, old, new, eta=cfg.assim,
            isochoric=cfg.assim_iso, smin=cfg.assim_smin, smax=cfg.assim_smax,
            settle_pin_assim=cfg.settle_pin_assim, fp64=cfg.assim_fp64)
    velocity = np.ascontiguousarray(to_array(end['v'], copy=True), np.float32)
    affine = np.ascontiguousarray(to_array(end['C'], copy=True), np.float32)
    pin_array = to_array(pins)
    velocity[pin_array] = 0.
    affine[pin_array] = 0.
    layer, spacing = prepare_layer_geometry(x, cfg)
    bonds = None
    if cfg.bonds:
        if spec.bond_nbr is None or spec.bond_rest is None:
            raise ValueError('Current preparation requires current material bond history')
        rest, fragmented = prepare_bonds(x, spec.bond_nbr, spec.bond_rest, spec.prm)
        bonds = (spec.bond_nbr, rest, fragmented.astype(np.float32), spec.bond_threshold)
    elif spec.bond_nbr is not None:
        raise ValueError('Current head bond policy differs from configuration')
    Fg = end.get('Fg')
    if Fg is not None:
        Fg = to_array(Fg, copy=True).reshape(-1, 3, 3)
    prepared = Trajectory(x, spec.m, spec.lam, spec.mu, deepcopy(spec.prm), spec.T,
        Fp=to_array(next_P), F0=to_array(F), v0=velocity, C0=affine, Fg0=Fg,
        vol0=spec.vol0, eta=spec.eta, pin=predicted['pins'], pin_slip=spec.pin_slip,
        layer=layer, bonds=bonds, track_geom=Fg is not None,
        requires_grad=False, persistent=False, device=spec.device)
    return OwnedWithdrawal.capture(prepared, 0), predicted, dict(
        scope='Current original-head passive successor preview; ordinary outer acceptance remains separate',
        layer_spacing=spacing, neutralized_policy_fields=['layer_ug'] if layer is not None else [],
        controls=dict(stress='zero', surface_u='zero', body='absent', layer_F=False),
        next_controlled_optimizer_state='not previewed', admission_derivative='not represented')
