"""Owned, detached control-withdrawal inputs; not a differentiable handoff.

Capture the actual prepared trajectory after its policies have been finalized.
No assimilation, pin admission or layer preparation is performed here. Replays
retain those captured policies and withdraw only future dFc, u and body control.
CUDA callers must keep capture/reconstruction inside ``cuda_execution``.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from math import isfinite
from numbers import Integral

import torch
import warp as wp

from physmorph.compute import array_api as np, is_cuda_execution, to_array
from .state import MPMParams
from .traj import Trajectory


def _device_context(device):
    device = wp.get_device(device)
    if device.is_cuda != is_cuda_execution():
        raise RuntimeError('Withdrawal device must match the caller cuda_execution context')
    if device.is_cuda and device.ordinal != torch.cuda.current_device():
        raise RuntimeError('Withdrawal device must match the active CUDA device')
    return str(device)


@dataclass(frozen=True)
class OwnedWithdrawal:
    """Private snapshot; public accessors and every trajectory own their buffers.

    ``capture(tr, step)`` requires the caller to know that ``step`` is populated.
    A step index alone cannot distinguish pre-commit from post-commit state.
    ``trajectory()`` constructs but does not run/capture a no-grad rollout.
    """
    _arrays: dict = field(repr=False)
    _metadata: dict = field(repr=False)

    @classmethod
    def capture(cls, tr: Trajectory, step: int):
        device = _device_context(tr.device)
        if isinstance(step, bool) or not isinstance(step, Integral) or not 0 <= step <= tr.T:
            raise ValueError('Withdrawal step must be an integer in [0,T]')
        values = dict(x0=tr.x[step], v0=tr.v[step], C0=tr.C[step], F0=tr.F[step],
                      Fp=tr.Fp, m=tr.m, lam=tr.lam, mu=tr.mu, eta=tr.eta,
                      pin=tr.pin, vol=tr.vol)
        arrays = {name: to_array(value) for name, value in values.items()}
        meta = dict(schema='owned_withdrawal_v1', N=int(tr.N), T=int(tr.T), step=int(step),
                    device=device, captured_device=device, prm=asdict(tr.prm),
                    pin_mode=int(tr.pin_mode), gate=bool(tr.gate),
                    gate_n0=float(tr.gate_n0) if tr.gate else None,
                    track_geom=bool(tr.track_geom), layer_present=bool(tr.layer),
                    layer_F=bool(tr.layer_F), bonds_present=bool(tr.bonds),
                    source_body_control=tr.body_control is not None,
                    scope='Detached frozen-state withdrawal; no commit or preparation map is applied')
        if tr.track_geom:
            arrays['Fg0'] = to_array(tr.Fg[step])
        if tr.layer:
            for name in ('mask', 'nrm', 'nbr', 'w', 'ug'):
                arrays['layer_'+name] = to_array(getattr(tr, 'layer_'+name))
            for name in ('nbr', 'w'):
                arrays['layer_'+name] = arrays['layer_'+name].reshape(tr.N, tr.layer_K)
            meta.update(layer_K=int(tr.layer_K), layer_frac=float(tr.layer_frac),
                        layer_frac_u=float(tr.layer_frac_u))
            if tr.layer_F:
                arrays['layer_g'] = to_array(tr.layer_g).reshape(tr.N, tr.layer_K, 3)
                meta['layer_inv_depth'] = float(tr.layer_inv_depth)
        else:
            arrays['layer_mask'] = np.zeros(tr.N, dtype=np.float32)
        if tr.bonds:
            for name in ('nbr', 'rest', 'frag'):
                arrays['bond_'+name] = to_array(getattr(tr, 'bond_'+name))
            for name in ('nbr', 'rest'):
                arrays['bond_'+name] = arrays['bond_'+name].reshape(tr.N, tr.bond_K)
            meta.update(bond_K=int(tr.bond_K), frag_thr=float(tr.frag_thr))
        return cls.from_arrays(arrays, meta)

    @classmethod
    def from_arrays(cls, arrays, metadata, *, device=None):
        """Own an archival mapping; host-to-device upload is an explicit I/O input.

        This validates schema, layout and policy scalars, without repairing or
        silently sanitizing numerical state. Runtime state guards remain the
        caller's responsibility. The original capture device is retained.
        """
        meta = deepcopy(metadata)
        if meta.get('schema') != 'owned_withdrawal_v1':
            raise ValueError('Unsupported withdrawal schema')
        json.dumps(meta, allow_nan=False)
        if (not isinstance(meta.get('captured_device'), str) or not meta['captured_device']
                or set(meta['prm']) != set(asdict(MPMParams()))):
            raise ValueError('Withdrawal requires captured device and complete MPM parameters')
        dev = _device_context(meta['device'] if device is None else device)
        for key in ('N', 'T', 'step', 'pin_mode'):
            if isinstance(meta[key], bool) or not isinstance(meta[key], Integral):
                raise ValueError(f'Withdrawal {key} must be integer')
        n, horizon = meta['N'], meta['T']
        if n < 1 or horizon < 1 or not 0 <= meta['step'] <= horizon or meta['pin_mode'] not in (0, 1):
            raise ValueError('Invalid withdrawal dimensions, step or pin mode')
        for key in ('gate', 'track_geom', 'layer_present', 'layer_F', 'bonds_present', 'source_body_control'):
            if type(meta[key]) is not bool:
                raise ValueError(f'Withdrawal {key} must be bool')
        prm = MPMParams(**meta['prm'])
        if meta['gate'] != bool(prm.gate_r_hi > prm.gate_r_lo):
            raise ValueError('Withdrawal gate policy differs from captured parameters')
        if meta['gate'] and (not isfinite(meta['gate_n0']) or meta['gate_n0'] <= 0):
            raise ValueError('Active withdrawal gate requires its positive resolved normalization')
        if meta['layer_F'] and not meta['layer_present']:
            raise ValueError('layer_F requires captured layer data')
        layouts = {name: ((n, 3), 'float32') for name in ('x0', 'v0')}
        layouts.update({name: ((n, 3, 3), 'float32') for name in ('C0', 'F0', 'Fp')})
        layouts.update({name: ((n,), 'float32') for name in ('m', 'lam', 'mu', 'eta', 'pin', 'vol', 'layer_mask')})
        if meta['track_geom']:
            layouts['Fg0'] = ((n, 3, 3), 'float32')
        for kind, enabled in (('layer', meta['layer_present']), ('bond', meta['bonds_present'])):
            if enabled:
                k = meta[kind+'_K']
                if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
                    raise ValueError(f'Invalid {kind} neighbor count')
                layouts[kind+'_nbr'] = ((n, k), 'int32')
                layouts[kind+('_w' if kind == 'layer' else '_rest')] = ((n, k), 'float32')
        if meta['layer_present']:
            layouts.update(layer_nrm=((n, 3), 'float32'), layer_ug=((n,), 'float32'))
            for key in ('layer_frac', 'layer_frac_u'):
                if not isfinite(meta[key]):
                    raise ValueError(f'Nonfinite {key}')
            if meta['layer_F']:
                layouts['layer_g'] = ((n, meta['layer_K'], 3), 'float32')
                if not isfinite(meta['layer_inv_depth']) or meta['layer_inv_depth'] < 0:
                    raise ValueError('Invalid captured inverse layer depth')
        if meta['bonds_present']:
            layouts['bond_frag'] = ((n,), 'float32')
            if not isfinite(meta['frag_thr']):
                raise ValueError('Nonfinite bond threshold')
        if set(arrays) != set(layouts):
            raise ValueError('Withdrawal arrays do not match the declared policies')
        owned = {}
        for name, (shape, dtype) in layouts.items():
            value = to_array(arrays[name])
            if value.shape != shape or str(value.dtype) != dtype:
                raise ValueError(f'Withdrawal {name} requires {shape} {dtype}')
            if name in ('layer_nbr', 'bond_nbr') and bool(((value < 0) | (value >= n)).any()):
                raise ValueError(f'Withdrawal {name} indices must be in [0,N)')
            owned[name] = value.copy()
        meta['device'] = dev
        return cls(owned, meta)

    def arrays(self):
        """Return independent arrays on the captured/reconstruction device."""
        _device_context(self._metadata['device'])
        return {name: value.copy() for name, value in self._arrays.items()}

    def metadata(self):
        """Return owned JSON-serializable metadata; scalar observation only."""
        return deepcopy(self._metadata)

    def trajectory(self, *, persistent=False):
        """Construct a private no-grad replay with original T; launch no steps."""
        meta, a = self._metadata, self._arrays
        device = _device_context(meta['device'])
        prm = MPMParams(**deepcopy(meta['prm']))
        if meta['gate']:
            prm.gate_n0 = meta['gate_n0']  # Keep the original resolved source count.
        layer = None
        if meta['layer_present']:
            inverse = meta.get('layer_inv_depth', 0.)
            layer = (a['layer_mask'], a['layer_nrm'], a['layer_nbr'], a['layer_w'],
                     meta['layer_frac'], a.get('layer_g'), 1./inverse if inverse > 0 else 0.,
                     a['layer_ug'])
        bonds = ((a['bond_nbr'], a['bond_rest'], a['bond_frag'], meta['frag_thr'])
                 if meta['bonds_present'] else None)
        tr = Trajectory(a['x0'], a['m'], a['lam'], a['mu'], prm, meta['T'],
            Fp=a['Fp'], v0=a['v0'], F0=a['F0'], C0=a['C0'], eta=a['eta'], pin=a['pin'],
            dFc=wp.zeros(meta['N'], dtype=wp.mat33, device=device),
            pin_slip=meta['pin_mode'] == 1, device=device, requires_grad=False,
            persistent=bool(persistent), vol0=a['vol'], Fg0=a.get('Fg0'),
            track_geom=meta['track_geom'], layer=layer, bonds=bonds,
            layer_u=wp.zeros(meta['N'], dtype=wp.float32, device=device) if layer else None,
            body_control=None)
        if tr.layer:
            tr.layer_frac_u = meta['layer_frac_u']
            if tr.layer_F:
                tr.layer_inv_depth = meta['layer_inv_depth']
        return tr
