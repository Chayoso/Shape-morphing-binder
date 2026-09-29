"""Controlled head -> assimilation/pin projection -> passive coast derivative.

Next-pin admission and successor layer/bond policies are supplied, owned and
fixed. This is a conditional handoff derivative, not the runner's full policy
derivative. The original pre-assimilation WithdrawalAdjoint is unchanged.
"""
from copy import deepcopy

import torch
import warp as wp

from physmorph.compute import to_array
from physmorph.plasticity.assimilation_adjoint import assimilate_handoff, _validate
from .traj import Trajectory
from .withdrawal_adjoint import WithdrawalAdjoint, _own, _reject_gradient_inputs


class PostAssimilationAdjoint(WithdrawalAdjoint):
    """Fixed materials/policies; same differentiable control/output API as P326.

    Explicit successor_layer/successor_bonds may be None when disabled. They are
    actual prepared values, not instructions to recompute geometry. next_pins
    includes all old pins; release/follow/yield/KKT and growth/consensus are not
    represented here. Captured execution uses ordered Warp and Torch graphs on
    one shared stream. Only first control derivatives are supported.
    """
    def __init__(self, spec, *, next_pins, successor_layer, successor_bonds, successor_eta,
                 eta=.5, isochoric=False, smin=.2, smax=5., settle_pin_assim=True,
                 capture=True):
        _reject_gradient_inputs((next_pins, successor_layer, successor_bonds, successor_eta))
        super().__init__(spec, capture=False)
        self.capture_enabled = bool(capture and self.cuda)
        self.eta, self.isochoric = eta, bool(isochoric)
        self.smin, self.smax = smin, smax
        self.settle_pin_assim = bool(settle_pin_assim)
        with self._scope():
            pins = torch.as_tensor(to_array(next_pins), device=self.device)
            if pins.shape != (self.N,) or pins.dtype != torch.bool:
                raise ValueError('next_pins must be a boolean (N,) array')
            self.next_pins = pins.detach().clone()
            current = wp.to_torch(self.head.pin)
            torch._assert_async(((current == 0) | (current == 1)).all(), 'Head pins must be binary')
            self.old_pins = (current > 0).detach().clone()
            torch._assert_async((~self.old_pins | self.next_pins).all(), 'Pin release is outside the handoff scope')
            self.new_pins = self.next_pins & ~self.old_pins
            # Drop every reference to the provisional pre-assimilation coast
            # before allocating its replacement; avoid three live trajectories.
            self._ports, self.seeds, self.seed_tensors, self.grad_arrays = (), {}, {}, []
            self.coast = None
            common = {name: _own(getattr(spec, name)) for name in
                      ('x0', 'm', 'lam', 'mu', 'Fp', 'v0', 'F0', 'C0', 'vol0', 'Fg0')}
            prm = deepcopy(self.head.prm)
            if self.head.gate:
                prm.gate_n0 = self.head.gate_n0
            self.coast = Trajectory(prm=prm, T=self.T, device=self.dev,
                requires_grad=True, persistent=True, track_geom=True,
                pin=to_array(self.next_pins.to(torch.float32)), pin_slip=bool(spec.pin_slip),
                layer=_own(successor_layer), bonds=_own(successor_bonds), eta=_own(successor_eta),
                dFc=wp.zeros(self.N, dtype=wp.mat33, device=self.dev),
                layer_u=wp.zeros(self.N, dtype=wp.float32, device=self.dev) if successor_layer is not None else None,
                body_control=None, **common)
            self.coast.Fp = wp.clone(self.coast.Fp, requires_grad=True)
            self.boundary_F = wp.to_torch(self.head.F[self.T]).detach().requires_grad_()
            self.fixed_Fp = wp.to_torch(self.head.Fp).detach()
            _validate(self.boundary_F, self.fixed_Fp, eta, smin, smax)
            self._configure_seeds()
            self.head_tape = self.coast_tape = None
            self.head_forward_graph = self.coast_forward_graph = None
            self.head_backward_graph = self.coast_backward_graph = None
            self.boundary_forward_graph = self.boundary_backward_graph = None
            if self.capture_enabled:
                self._capture()

    def _configure_seeds(self):
        head, coast = self.head, self.coast
        self._ports = ((head.x[self.T],), (head.F[self.T],), (head.v[self.T],), (head.Fg[self.T],),
                       tuple(head.v[1:]), tuple(head.x[1:]), tuple(coast.x), tuple(coast.v),
                       tuple(coast.F), tuple(coast.Fg))
        self.seed_tensors, self.seeds = {}, {}
        for port in self._ports:
            for array in port:
                if array not in self.seeds:
                    value = torch.zeros_like(wp.to_torch(array))
                    self.seed_tensors[array] = value
                    self.seeds[array] = wp.from_torch(value, dtype=array.dtype)
        head_arrays = set(array for port in self._ports[:6] for array in port)
        head_arrays.add(head.C[self.T])
        self.head_seed_tensors = {a: torch.zeros_like(wp.to_torch(a)) for a in head_arrays}
        self.head_seeds = {a: wp.from_torch(v, dtype=a.dtype) for a, v in self.head_seed_tensors.items()}
        coast_arrays = set(array for port in self._ports[6:] for array in port)
        self.coast_seeds = {a: self.seeds[a] for a in coast_arrays}
        self.grad_arrays, seen = [], set()
        for tr in (head, coast):
            for value in vars(tr).values():
                for array in value if isinstance(value, (list, tuple)) else (value,):
                    if isinstance(array, wp.array) and array.grad is not None:
                        key = (str(array.grad.device), array.grad.ptr)
                        if key not in seen:
                            seen.add(key)
                            self.grad_arrays.append(array.grad)

    def _record_head(self):
        self.head_tape = wp.Tape()
        with self.head_tape:
            self.head.rollout()

    def _record_coast(self):
        self.coast_tape = wp.Tape()
        with self.coast_tape:
            self.coast.rollout()

    def _boundary_forward(self):
        with torch.enable_grad():
            self.boundary_Fp = assimilate_handoff(self.boundary_F, self.fixed_Fp,
                self.old_pins, self.new_pins, eta=self.eta, isochoric=self.isochoric,
                smin=self.smin, smax=self.smax, settle_pin_assim=self.settle_pin_assim)
        with torch.no_grad():
            wp.to_torch(self.coast.Fp).copy_(self.boundary_Fp)
            for name in ('x', 'v', 'C', 'F', 'Fg'):
                source = wp.to_torch(getattr(self.head, name)[self.T])
                if name in ('v', 'C'):
                    mask = self.next_pins.reshape((self.N,) + (1,) * (source.ndim-1))
                    source = torch.where(mask, torch.zeros_like(source), source)
                wp.to_torch(getattr(self.coast, name)[0]).copy_(source)

    def _boundary_vjp(self):
        if not self.boundary_Fp.requires_grad:
            return torch.zeros_like(self.boundary_F)
        return torch.autograd.grad(self.boundary_Fp, self.boundary_F,
            grad_outputs=wp.to_torch(self.coast.Fp.grad), retain_graph=True)[0]

    def _head_covectors(self, fp_to_F):
        with torch.no_grad():
            for array, value in self.head_seed_tensors.items():
                if array in self.seed_tensors:
                    value.copy_(self.seed_tensors[array])
                else:
                    value.zero_()
            for name in ('x', 'v', 'C', 'F', 'Fg'):
                gradient = wp.to_torch(getattr(self.coast, name)[0].grad)
                if name in ('v', 'C'):
                    mask = self.next_pins.reshape((self.N,) + (1,) * (gradient.ndim-1))
                    gradient = torch.where(mask, torch.zeros_like(gradient), gradient)
                self.head_seed_tensors[getattr(self.head, name)[self.T]].add_(gradient)
            self.head_seed_tensors[self.head.F[self.T]].add_(fp_to_F)

    def _capture(self):
        # Warm forward, both reverse tapes and the spectral pullback on the
        # constructor's aligned stream before any library/graph capture.
        self._record_head()
        self._boundary_forward()
        self._record_coast()
        self._zero_grads()
        self.coast_tape.backward(grads=self.coast_seeds)
        self._head_covectors(self._boundary_vjp())
        self.head_tape.backward(grads=self.head_seeds)
        wp.synchronize_stream(self.warp_stream)
        with wp.ScopedCapture(stream=self.warp_stream) as captured:
            self._record_head()
        self.head_forward_graph = captured.graph
        with wp.ScopedCapture(stream=self.warp_stream) as captured:
            self._record_coast()
        self.coast_forward_graph = captured.graph
        with wp.ScopedCapture(stream=self.warp_stream) as captured:
            self.coast_tape.backward(grads=self.coast_seeds)
        self.coast_backward_graph = captured.graph
        with wp.ScopedCapture(stream=self.warp_stream) as captured:
            self.head_tape.backward(grads=self.head_seeds)
        self.head_backward_graph = captured.graph
        self.boundary_forward_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.boundary_forward_graph, stream=self.stream):
            self._boundary_forward()
        self.boundary_forward_graph.replay()
        self.boundary_backward_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.boundary_backward_graph, stream=self.stream):
            self.captured_fp_to_F = self._boundary_vjp()
        self._zero_grads()

    def forward(self):
        self.forward_generation += 1
        with self._scope():
            if self.capture_enabled:
                wp.capture_launch(self.head_forward_graph)
                self.boundary_forward_graph.replay()
                wp.capture_launch(self.coast_forward_graph)
            else:
                self._record_head()
                self._boundary_forward()
                self._record_coast()

    def backward(self):
        with self._scope():
            self._zero_grads()
            if self.capture_enabled:
                wp.capture_launch(self.coast_backward_graph)
                self.boundary_backward_graph.replay()
                self._head_covectors(self.captured_fp_to_F)
                wp.capture_launch(self.head_backward_graph)
            else:
                self.coast_tape.backward(grads=self.coast_seeds)
                self._head_covectors(self._boundary_vjp())
                self.head_tape.backward(grads=self.head_seeds)
