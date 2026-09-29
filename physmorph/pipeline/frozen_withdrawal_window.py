"""Private reduced-body lookahead, before assimilation or new pin admission.

The head has the exact FrozenBodyWindow basis, bounds and energy. The coast
withdraws learned controls through WithdrawalAdjoint's single tape; it keeps
the original Fp, pins, layer and bonds. This is not an actual commit evaluator
or a candidate adoption API. Full-state snapshots are owned evidence, not new
differentiable output channels beyond the joint bridge's public outputs.
"""
from copy import deepcopy

import torch
import warp as wp

from ..compute import cuda_module, is_cuda_execution
from ..mpm.withdrawal_adjoint import WithdrawalAdjoint
from .endpoint_contract import endpoint_bounds, valid_endpoint
from .frozen_body_window import FrozenBodyWindow


def _preflight(device, stream=None):
    """Metadata/context only: reject before cloning or reading numerical inputs."""
    dev = wp.get_device(device)
    if dev.is_cuda != is_cuda_execution():
        raise RuntimeError('Withdrawal adapter requires matching cuda_execution context')
    if not dev.is_cuda:
        return None
    current = torch.cuda.current_stream(torch.device(str(dev)))
    if (torch.cuda.current_device() != dev.ordinal
            or (stream is not None and current.cuda_stream != stream.cuda_stream)):
        raise RuntimeError('Withdrawal adapter is bound to its constructor CUDA device/stream')
    if (wp.get_stream(dev).cuda_stream != current.cuda_stream
            or cuda_module().cuda.get_current_stream().ptr != current.cuda_stream):
        raise RuntimeError('Withdrawal adapter requires matching Torch/CuPy/Warp streams')
    return current


class FrozenWithdrawalWindow:
    """Callback-lived adapter; closing its owner also expires this adapter."""

    _scope_label = 'pre_assimilation_frozen_policy_withdrawal'

    def __init__(self, owner, *, capture=True):
        if not isinstance(owner, FrozenBodyWindow) or owner.closed:
            raise ValueError('Withdrawal adapter requires a live FrozenBodyWindow')
        if not owner.spec.body_ctrl or owner.spec.body_modes != 2:
            raise ValueError('Withdrawal adapter requires two body modes')
        self._device = str(wp.get_device(owner.spec.device))
        self._stream = _preflight(self._device)
        self._owner = owner
        self.spec = deepcopy(owner.spec)
        for key in ('idx', 'weights', 'gate', 'coefficients', 'stress'):
            setattr(self, key, getattr(owner, key).detach().clone())
        self.surface_u = None if owner.surface_u is None else owner.surface_u.detach().clone()
        self.capture = bool(capture)
        self.adjoint = None
        self.closed = False
        self.generation = 0

    def _live(self):
        if self.closed or self._owner.closed:
            raise RuntimeError('Frozen withdrawal window expired')

    def _gradient_lease(self, gradient, generation):
        self._live()
        if generation != self.generation:
            raise RuntimeError('Frozen withdrawal gradient is stale')
        return gradient

    def close(self):
        self.closed = True
        self.generation += 1
        if self.adjoint is not None:
            self.adjoint.forward_generation += 1
        self.adjoint = None

    def _make_adjoint(self):
        return WithdrawalAdjoint(self.spec, capture=self.capture)

    def _coast_contract(self, head, coast, out, snapshots):
        """Default shared boundary; subclasses may supply an explicit handoff."""
        return dict(pins=wp.to_torch(head.pin) > .5, anchor=wp.to_torch(head.x[0]),
                    velocity=out.v, C=snapshots['C_sequence'][-1], pin_step=1,
                    finite=True, valid=True, snapshots={}, health={})

    def evaluate(self, terminal, displacement=None):
        """Return the original head merit packet plus a frozen-policy coast.

        X/V and joint bridge F/Fg outputs remain differentiable. Head full F/Fg/C
        snapshots and coast C are detached copies from that *same* forward.
        Never replay to obtain or replace missing endpoint state.
        """
        # Failed replacements also expire both Warp outputs and the Torch-only
        # body energy graph. Invalidate before any control validation/copies.
        self.generation += 1
        generation = self.generation
        self._live()
        _preflight(self._device, self._stream)
        displacement = self.coefficients[:, :3] if displacement is None else displacement
        for name, value in (('displacement', displacement), ('terminal', terminal)):
            if (not torch.is_tensor(value) or value.shape != self.coefficients[:, :3].shape
                    or value.device != self.coefficients.device or value.dtype != self.coefficients.dtype
                    or not bool(torch.isfinite(value).all())):
                raise ValueError('Invalid '+name+' coefficients')
        coeff = torch.cat((displacement, terminal), dim=1)
        if bool((coeff.detach().square().sum(-1) > 1+1e-6).any()):
            raise ValueError('Terminal coefficients exceed remaining joint radius')
        if self.adjoint is None:
            self.adjoint = self._make_adjoint()
        # Preserve the exact production diagnostic expression and summation order.
        n = len(self.idx)
        field = self.spec.prm.dx * (coeff[self.idx]*self.weights[..., None]).sum(1)*self.gate
        body = field.reshape(n, 2, 3).permute(1, 0, 2).reshape(2*n, 3).contiguous()
        out = self.adjoint.apply(self.stress, self.surface_u, body)
        energy = (field/self.spec.prm.dx).square().sum(1).mean()
        with torch.no_grad():
            head, coast = self.adjoint.head, self.adjoint.coast
            snapshots = {}
            for name in ('F', 'Fg', 'C'):
                seq = torch.stack([wp.to_torch(a).reshape(n, 3, 3) for a in getattr(head, name)])
                snapshots[name+'_initial'] = seq[0].clone()
                snapshots[name+'_sequence'] = seq[1:].clone()
            coast_C = torch.stack([wp.to_torch(a).reshape(n, 3, 3) for a in coast.C])
            boundary = self._coast_contract(head, coast, out, snapshots)
            coast_pinned = boundary['pins']
            pinned = wp.to_torch(head.pin) > .5
            start = wp.to_torch(head.x[0])
            bounds = endpoint_bounds(self.spec.prm, out.x)
            head_finite = all(bool(torch.isfinite(v).all()) for v in
                              (out.x, out.F, out.v, out.Fg, out.V, out.X, energy,
                               *snapshots.values(), wp.to_torch(head.v[0]), start))
            coast_finite = all(bool(torch.isfinite(v).all()) for v in
                               (out.coast_X, out.coast_V, out.coast_F, out.coast_Fg, coast_C)) and boundary['finite']
            head_det = torch.linalg.det(snapshots['F_sequence']).min()
            elastic_det = torch.stack([torch.linalg.det(
                wp.to_torch(head.F[t]).reshape(n, 3, 3)+self.stress[t]).min()
                for t in range(self.spec.T)]).min()
            coast_det = torch.linalg.det(out.coast_F.reshape(self.spec.T+1, n, 3, 3)).min()
            head_pins = torch.equal(out.X[:, pinned], start[pinned][None].expand(self.spec.T, -1, -1))
            coast_pins = torch.equal(out.coast_X[:, coast_pinned], boundary['anchor'][coast_pinned][None].expand(self.spec.T+1, -1, -1))
            # Incoming v0/C0 can be nonzero. Every simulated pinned step must
            # zero both velocity and APIC state; fixed x alone is insufficient.
            head_pin_state = (bool((out.V[:, pinned] == 0).all())
                              and bool((snapshots['C_sequence'][:, pinned] == 0).all()))
            coast_pin_state = (bool((out.coast_V[boundary['pin_step']:, coast_pinned] == 0).all())
                               and bool((coast_C[boundary['pin_step']:, coast_pinned] == 0).all()))
            head_pins = head_pins and head_pin_state
            coast_pins = coast_pins and coast_pin_state
            head_valid = (head_finite and valid_endpoint(out.X, bounds)
                          and float(torch.minimum(head_det, elastic_det)) > 0 and head_pins)
            coast_valid = (coast_finite and valid_endpoint(out.coast_X, bounds)
                           and float(coast_det) > 0 and coast_pins and boundary['valid'])
            # These equalities bind the detached archive fields to public outputs.
            same_forward = (torch.equal(snapshots['F_sequence'][-1].reshape_as(out.F), out.F)
                            and torch.equal(snapshots['Fg_sequence'][-1].reshape_as(out.Fg), out.Fg)
                            and torch.equal(out.x, out.X[-1]) and torch.equal(out.v, out.V[-1])
                            and torch.equal(out.coast_X[0], out.x) and torch.equal(out.coast_V[0], boundary['velocity'])
                            and torch.equal(out.coast_F[0], out.F) and torch.equal(out.coast_Fg[0], out.Fg)
                            and torch.equal(coast_C[0], boundary['C']))
        result = dict(x=out.x, F=out.F, C=snapshots['C_sequence'][-1].clone(), v=out.v,
                      Fg=out.Fg, V=out.V, positions=out.X, body_energy=energy,
                      coast_X=out.coast_X, coast_V=out.coast_V, coast_F=out.coast_F,
                      coast_Fg=out.coast_Fg, coast_C=coast_C,
                      valid=bool(head_valid and coast_valid and same_forward),
                      pins_exact=bool(head_pins and coast_pins),
                      min_det=float(torch.minimum(head_det, elastic_det)),
                      health=dict(head_valid=bool(head_valid), coast_valid=bool(coast_valid),
                                  head_finite=head_finite, coast_finite=coast_finite,
                                  head_pins_exact=head_pins, coast_pins_exact=coast_pins,
                                  pin_state_scope='zero post-step V/C; incoming boundary v0/C0 excluded',
                                  coast_min_det=float(coast_det), same_forward=same_forward),
                      scope=self._scope_label, **snapshots, **boundary['snapshots'])
        result['health'].update(boundary['health'])
        # An expired callback must not leave a live private reverse computation.
        for value in result.values():
            if torch.is_tensor(value) and value.requires_grad:
                value.register_hook(lambda gradient, generation=generation:
                                    self._gradient_lease(gradient, generation))
        return result
