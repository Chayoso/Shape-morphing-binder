"""Reduced-body head -> assimilation -> fixed prepared-successor derivative.

Successor policies are supplied from an actual step-zero capture. This does
not differentiate admission/preparation or certify a candidate for adoption.
The caller independently checks basepoint closure against that captured state.
"""
from copy import deepcopy
from dataclasses import asdict

import torch
import warp as wp

from ..mpm.post_assimilation_adjoint import PostAssimilationAdjoint
from ..mpm.withdrawal import OwnedWithdrawal
from .frozen_withdrawal_window import FrozenWithdrawalWindow
from .window_selection import validate_config


class PostAssimilationWindow(FrozenWithdrawalWindow):
    """Owned fixed policies; its owner's lifetime and every forward lease apply."""

    _scope_label = 'post_assimilation_fixed_successor_withdrawal'

    def __init__(self, owner, successor, cfg, *, capture=True):
        super().__init__(owner, capture=capture)
        validate_config(cfg)
        unsupported = [name for name in ('w_grow', 'assim_consensus', 'freeze_arrived',
                       'layer_F', 'settle_pin_kkt_dry') if getattr(cfg, name, False)]
        if unsupported:
            raise ValueError('Post-assimilation window does not support '+', '.join(unsupported))
        if not isinstance(successor, OwnedWithdrawal):
            raise TypeError('Post-assimilation window requires an OwnedWithdrawal successor')
        meta = successor.metadata()
        spec = self.spec
        if (meta['step'] != 0 or meta['N'] != len(spec.x0) or meta['T'] != spec.T
                or cfg.T != spec.T or meta['device'] != self._device
                or meta['prm'] != asdict(spec.prm)
                or meta['pin_mode'] != int(bool(spec.pin_slip))):
            raise ValueError('Successor step/N/T/device/parameters/pin mode must match the head')
        if meta['gate'] or spec.prm.gate_r_hi > spec.prm.gate_r_lo:
            raise ValueError('Post-assimilation window does not support active gate policies')
        if meta['layer_F']:
            raise ValueError('Post-assimilation window does not support successor layer_F')
        if (meta['layer_present'] and meta['layer_frac_u'] != 1./spec.T
                or meta.get('layer_inv_depth', 0.) != 0.):
            raise ValueError('Successor layer control fraction/inverse depth is unsupported')
        arrays = successor.arrays()
        device = self.coefficients.device
        def tensor(value):
            return torch.as_tensor(value, device=device).detach().clone()
        for name, head_name in (('m', 'm'), ('lam', 'lam'), ('mu', 'mu'), ('vol', 'vol0')):
            value = getattr(spec, head_name)
            if value is None and name in ('lam', 'mu'):
                value = 0.
            if value is None:
                raise ValueError('Successor requires fixed head material: '+name)
            value = tensor(wp.to_torch(value) if isinstance(value, wp.array) else value).to(torch.float32)
            if name == 'vol' and value.shape != (meta['N'],):
                raise ValueError('Head vol0 must be an explicit (N,) array')
            try:
                value = torch.broadcast_to(value, (meta['N'],))
            except RuntimeError as error:
                raise ValueError('Invalid head material shape: '+name) from error
            if not torch.equal(tensor(arrays[name]), value):
                raise ValueError('Successor fixed material differs from head: '+name)
        pins = tensor(arrays['pin'])
        old = tensor(spec.pin) if spec.pin is not None else torch.zeros_like(pins)
        if (not bool(((pins == 0) | (pins == 1)).all())
                or not bool(((old == 0) | (old == 1)).all())):
            raise ValueError('Head and successor pins must be binary')
        self._next_pins = pins.bool()
        if bool(((old > .5) & ~self._next_pins).any()):
            raise ValueError('Successor pin release is unsupported')
        self._cfg = deepcopy(cfg)
        self._successor_metadata = meta
        self._successor_eta = arrays['eta']
        self._successor_layer = None
        if meta['layer_present']:
            self._successor_layer = (arrays['layer_mask'], arrays['layer_nrm'], arrays['layer_nbr'],
                arrays['layer_w'], meta['layer_frac'], None, 0., arrays['layer_ug'])
        self._successor_bonds = ((arrays['bond_nbr'], arrays['bond_rest'], arrays['bond_frag'], meta['frag_thr'])
                                 if meta['bonds_present'] else None)

    def _make_adjoint(self):
        cfg = self._cfg
        adjoint = PostAssimilationAdjoint(self.spec, next_pins=self._next_pins,
            successor_layer=self._successor_layer, successor_bonds=self._successor_bonds,
            successor_eta=self._successor_eta, eta=cfg.assim, isochoric=cfg.assim_iso,
            smin=cfg.assim_smin, smax=cfg.assim_smax, settle_pin_assim=cfg.settle_pin_assim,
            capture=False, fp64=cfg.assim_fp64)
        # With layer_F disabled and u=0 this fraction has no numerical effect,
        # but preserve the captured policy before constructing any CUDA graph.
        if adjoint.coast.layer:
            adjoint.coast.layer_frac_u = self._successor_metadata['layer_frac_u']
        adjoint.capture_enabled = bool(self.capture and adjoint.cuda)
        if adjoint.capture_enabled:
            with adjoint._scope():
                adjoint._capture()
        return adjoint

    def _coast_contract(self, head, coast, out, snapshots):
        pins = self._next_pins
        Fp = wp.to_torch(coast.Fp).detach().clone()
        finite = bool(torch.isfinite(Fp).all())
        determinant = torch.linalg.det(Fp).min()
        valid = finite and float(determinant) > 0
        return dict(pins=pins, anchor=out.x,
            velocity=torch.where(pins[:, None], torch.zeros_like(out.v), out.v),
            C=torch.where(pins[:, None, None], torch.zeros_like(snapshots['C_sequence'][-1]),
                          snapshots['C_sequence'][-1]),
            pin_step=0, finite=finite, valid=valid,
            snapshots=dict(coast_Fp=Fp, coast_pins=pins.detach().clone()),
            health=dict(coast_Fp_valid=valid, coast_Fp_min_det=float(determinant),
                        pin_state_scope='head old pins after steps; coast next pins from step0 at head endpoint'))

    def close(self):
        super().close()
        self._successor_layer = self._successor_bonds = self._successor_eta = None
        self._next_pins = None
