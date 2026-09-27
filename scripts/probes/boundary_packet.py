"""Owned first-step replay inputs from an already executed accepted trajectory.

Capture is an explicit host archive boundary and launches no numerical kernels.
Replay executes only step zero, retaining the original horizon because the body,
bond and surface channels use it. CUDA callers must enter ``cuda_execution``.
"""
from __future__ import annotations

from dataclasses import asdict
import json

import numpy as np
import torch
import warp as wp

from physmorph.compute import is_cuda_execution, to_array, to_host, warp_array
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory


def _host(value):
    return to_host(wp.to_torch(value))


def capture_step(tr: Trajectory) -> dict:
    """Copy authoritative inputs and original step-one outputs, with no replay.

    The caller must capture an executed accepted buffer before it is reused.
    ``frag_step``/neighbor counts are shared scratch at the last executed step;
    they are deliberately omitted and recomputed by each first-step replay.
    """
    if tr.T < 1:
        raise ValueError('first-step capture requires T >= 1')
    arrays = {name: _host(value) for name, value in {
        'x0': tr.x[0], 'v0': tr.v[0], 'F0': tr.F[0], 'C0': tr.C[0],
        'Fp': tr.Fp, 'm': tr.m, 'lam': tr.lam, 'mu': tr.mu, 'eta': tr.eta,
        'pin': tr.pin, 'vol': tr.vol, 'dFc': tr._dfc(0),
        'original_x1': tr.x[1], 'original_v1': tr.v[1],
        'original_F1': tr.F[1], 'original_C1': tr.C[1],
        'original_pre_layer': tr.xu[1] if tr.layer else tr.x[1],
    }.items()}
    meta = dict(schema_version=1, N=int(tr.N), T=int(tr.T), prm=asdict(tr.prm),
                pin_mode=int(tr.pin_mode), pin_slip=bool(tr.pin_mode == 1),
                gate=bool(tr.gate), gate_n0=float(tr.gate_n0) if tr.gate else None,
                has_body_control=tr.body_control is not None,
                layer_present=bool(tr.layer), layer_F=bool(tr.layer_F),
                bonds_present=bool(tr.bonds), track_geom=bool(tr.track_geom))
    if tr.body_control is not None:
        arrays['body_control'] = _host(tr.body_control)
    if tr.layer:
        for name in ('mask', 'nrm', 'nbr', 'w', 'u', 'ug'):
            arrays['layer_' + name] = _host(getattr(tr, 'layer_' + name))
        for name in ('nbr', 'w'):
            arrays['layer_' + name] = arrays['layer_' + name].reshape(tr.N, tr.layer_K)
        meta.update(layer_K=int(tr.layer_K), layer_frac=float(tr.layer_frac),
                    layer_frac_u=float(tr.layer_frac_u))
        if tr.layer_F:
            arrays['layer_g'] = _host(tr.layer_g).reshape(tr.N, tr.layer_K, 3)
            inv_depth = float(tr.layer_inv_depth)
            meta.update(layer_inv_depth=inv_depth,
                        layer_depth=1.0 / inv_depth if inv_depth > 0 else 0.0)
    else:
        arrays['layer_mask'] = np.zeros(tr.N, dtype=np.float32)
    if tr.bonds:
        for name in ('nbr', 'rest', 'frag'):
            arrays['bond_' + name] = _host(getattr(tr, 'bond_' + name))
        for name in ('nbr', 'rest'):
            arrays['bond_' + name] = arrays['bond_' + name].reshape(tr.N, tr.bond_K)
        meta.update(bond_K=int(tr.bond_K), frag_thr=float(tr.frag_thr))
    if tr.track_geom:
        arrays['Fg0'], arrays['original_Fg1'] = _host(tr.Fg[0]), _host(tr.Fg[1])
    # Canonical JSON containers also own the dataclass's tuple/list metadata.
    meta = json.loads(json.dumps(meta, allow_nan=False))
    return {'meta': meta, 'arrays': arrays}


def replay_step(packet: dict, x0override=None, device='cpu') -> dict:
    """Reconstruct the frozen controls/materials and return owned Torch outputs.

    A position override changes only initial geometry. Support gates and bond
    activation legitimately respond to that branch's density; gate normalization
    remains the value resolved by the captured trajectory. The caller is
    responsible for keeping pinned anchors identical between branches.
    """
    meta, arrays = packet['meta'], packet['arrays']
    if meta.get('schema_version') != 1 or int(meta['T']) < 1:
        raise ValueError('unsupported packet schema or empty original horizon')
    is_cuda = torch.device(device).type == 'cuda'
    if is_cuda != is_cuda_execution():
        raise RuntimeError('replay device must match the caller cuda_execution context')
    n, T = int(meta['N']), int(meta['T'])
    params = dict(meta['prm'])
    if meta['gate']:
        # Prevent the override geometry from redefining the nominal count.
        params['gate_n0'] = float(meta['gate_n0'])
    prm = MPMParams(**params)
    x0 = to_array(arrays['x0'] if x0override is None else x0override, copy=True)
    if x0.shape != (n, 3):
        raise ValueError(f'x0 override must have shape ({n}, 3)')

    def a(name):
        return to_array(arrays[name])

    def w(name, dtype):
        return warp_array(a(name), dtype=dtype, device=device, requires_grad=False)

    layer = None
    if meta['layer_present']:
        layer = (a('layer_mask'), a('layer_nrm'), a('layer_nbr'), a('layer_w'),
                 float(meta['layer_frac']),
                 a('layer_g') if meta['layer_F'] else None,
                 float(meta.get('layer_depth', 0.0)), a('layer_ug'))
    bonds = None
    if meta['bonds_present']:
        bonds = (a('bond_nbr'), a('bond_rest'), a('bond_frag'), float(meta['frag_thr']))
    tr = Trajectory(
        x0, a('m'), a('lam'), a('mu'), prm, T,
        Fp=a('Fp'), v0=a('v0'), F0=a('F0'), C0=a('C0'),
        dFc=w('dFc', wp.mat33), eta=a('eta'), pin=a('pin'),
        pin_slip=bool(meta['pin_mode'] == 1), device=device,
        requires_grad=False, persistent=False, vol0=a('vol'),
        Fg0=a('Fg0') if meta['track_geom'] else None, track_geom=meta['track_geom'],
        bonds=bonds, layer=layer,
        layer_u=w('layer_u', wp.float32) if layer is not None else None,
        body_control=w('body_control', wp.vec3) if meta['has_body_control'] else None)
    if tr.layer:
        tr.layer_frac_u = float(meta['layer_frac_u'])
        if tr.layer_F:
            tr.layer_inv_depth = float(meta['layer_inv_depth'])
    tr.step(0)

    def own(value):
        return wp.to_torch(value).detach().clone()

    result = {name: own(value) for name, value in {
        'x0': tr.x[0], 'x1': tr.x[1],
        'pre_layer': tr.xu[1] if tr.layer else tr.x[1],
        'v1': tr.v[1], 'F1': tr.F[1], 'C1': tr.C[1],
    }.items()}
    if tr.bonds:
        result.update(bond_active=own(tr.frag_step), neighbor_count=own(tr.ncount_b))
    if tr.track_geom:
        result.update(Fg0=own(tr.Fg[0]), Fg1=own(tr.Fg[1]))
    return result
