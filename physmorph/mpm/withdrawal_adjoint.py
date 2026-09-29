"""Joint control-to-withdrawal adjoint on two original-horizon trajectories.

This is a pre-assimilation continuation with frozen prepared policies, not the
runner's commit/admission/re-preparation map. No optimizer/objective uses it by
default. CUDA instances are bound to their constructor's Torch stream.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import fields, is_dataclass
from numbers import Integral
from typing import NamedTuple

import torch
import warp as wp

from physmorph.compute import cuda_module, is_cuda_execution, to_array
from .traj import Trajectory


class WithdrawalOutputs(NamedTuple):
    x: torch.Tensor
    F: torch.Tensor
    v: torch.Tensor
    Fg: torch.Tensor
    V: torch.Tensor
    X: torch.Tensor
    coast_X: torch.Tensor
    coast_V: torch.Tensor
    coast_F: torch.Tensor
    coast_Fg: torch.Tensor


def _reject_gradient_inputs(value):
    if isinstance(value, (torch.Tensor, wp.array)):
        if value.requires_grad:
            raise ValueError('WithdrawalAdjoint supports fixed initial/material/config inputs only')
    elif is_dataclass(value):
        for item in fields(value):
            _reject_gradient_inputs(getattr(value, item.name))
    elif isinstance(value, dict):
        for item in value.values():
            _reject_gradient_inputs(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _reject_gradient_inputs(item)


def _own(value):
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return tuple(_own(item) for item in value)
    if hasattr(value, 'shape'):
        result = to_array(value, copy=True)
        return result.item() if result.ndim == 0 else result
    return deepcopy(value)


class WithdrawalAdjoint:
    """Fixed materials; dFc(T,N,3,3), optional u(N), body(modes*N,3).

    ``apply`` returns owned head outputs (the existing five plus X) and coast
    X/V/F/Fg sequences including their shared time-zero boundary. Coast F/Fg are
    (T+1,N,9). Initial state is fixed; only supplied control leaves receive grads.
    CUDA construction, forward and backward require a matching cuda_execution
    context and the same Torch stream. capture=False uses plain device kernels.
    """
    def __init__(self, spec, *, capture=True):
        _reject_gradient_inputs(spec)
        if spec.vol0 is None:
            raise ValueError('WithdrawalAdjoint requires explicit source vol0')
        if isinstance(spec.T, bool) or not isinstance(spec.T, Integral) or spec.T < 1:
            raise ValueError('WithdrawalAdjoint requires integer T>=1')
        if spec.body_modes not in (1, 2) or (spec.body_ctrl and spec.T < 2):
            raise ValueError('Body control requires T>=2 and one or two modes')
        if len(spec.x0.shape) != 2 or tuple(spec.x0.shape[1:]) != (3,) or spec.x0.shape[0] < 1:
            raise ValueError('WithdrawalAdjoint requires nonempty x0(N,3)')
        self.dev = str(wp.get_device(spec.device))
        self.device = torch.device(self.dev)
        self.cuda = self.device.type == 'cuda'
        self.stream = torch.cuda.current_stream(self.device) if self.cuda else None
        # Keep the Warp-owned stream installed by cuda_execution. A nested
        # Torch-only stream switch has no established ordering for CuPy inputs.
        self.warp_stream = wp.get_stream(self.dev) if self.cuda else None
        self.T, self.N = int(spec.T), int(spec.x0.shape[0])
        self.forward_generation = 0
        self.capture_enabled = bool(capture and self.cuda)
        self.g_fwd = self.g_bwd = self.tape = None
        with self._scope():
            common = {name: _own(getattr(spec, name)) for name in
                      ('x0', 'm', 'lam', 'mu', 'Fp', 'v0', 'F0', 'C0', 'vol0', 'Fg0', 'eta', 'pin', 'layer')}
            common.update(device=self.dev, requires_grad=True, persistent=True, track_geom=True,
                          pin_slip=bool(spec.pin_slip), bonds=_own(spec.bonds()))
            self.dc = torch.zeros(self.T, self.N, 3, 3, dtype=torch.float32, device=self.device)
            self.dc_wp = [wp.from_torch(value, dtype=wp.mat33, requires_grad=True) for value in self.dc]
            self.u = torch.zeros(self.N, dtype=torch.float32, device=self.device)
            self.u_wp = wp.from_torch(self.u, dtype=wp.float32, requires_grad=True) if spec.layer is not None else None
            self.body = torch.zeros(spec.body_modes*self.N, 3, dtype=torch.float32, device=self.device)
            self.body_wp = wp.from_torch(self.body, dtype=wp.vec3, requires_grad=True) if spec.body_ctrl else None
            self.head = Trajectory(prm=deepcopy(spec.prm), T=self.T, dFc=self.dc_wp,
                                   layer_u=self.u_wp, body_control=self.body_wp, **common)
            coast_prm = deepcopy(self.head.prm)
            if self.head.gate:
                coast_prm.gate_n0 = self.head.gate_n0
            self.coast = Trajectory(prm=coast_prm, T=self.T,
                dFc=wp.zeros(self.N, dtype=wp.mat33, device=self.dev),
                layer_u=wp.zeros(self.N, dtype=wp.float32, device=self.dev) if spec.layer is not None else None,
                body_control=None, **common)
            head, coast = self.head, self.coast
            # This alias is the joint derivative: no bridge boundary or detached copy.
            for name in ('x', 'v', 'C', 'F', 'Fg'):
                getattr(coast, name)[0] = getattr(head, name)[self.T]
            for name in ('Fp', 'm', 'lam', 'mu', 'eta', 'pin', 'vol'):
                setattr(coast, name, getattr(head, name))
            if head.layer:
                for name in ('layer_mask', 'layer_nrm', 'layer_nbr', 'layer_w', 'layer_ug'):
                    setattr(coast, name, getattr(head, name))
                if head.layer_F:
                    coast.layer_g = head.layer_g
            if head.bonds:
                for name in ('bond_nbr', 'bond_rest', 'bond_frag'):
                    setattr(coast, name, getattr(head, name))
            self._ports = ((head.x[self.T],), (head.F[self.T],), (head.v[self.T],), (head.Fg[self.T],),
                           tuple(head.v[1:]), tuple(head.x[1:]), tuple(coast.x), tuple(coast.v),
                           tuple(coast.F), tuple(coast.Fg))
            self.seed_tensors, self.seeds = {}, {}
            for port in self._ports:
                for array in port:
                    if array not in self.seeds:
                        seed = torch.zeros_like(wp.to_torch(array))
                        self.seed_tensors[array] = seed
                        self.seeds[array] = wp.from_torch(seed, dtype=array.dtype)
            # Some state/constant fields alias across segments. Zero each actual
            # gradient storage only once; never rely on array/output names.
            self.grad_arrays, seen = [], set()
            for tr in (head, coast):
                for value in vars(tr).values():
                    for array in value if isinstance(value, (list, tuple)) else (value,):
                        if isinstance(array, wp.array) and array.grad is not None:
                            key = (str(array.grad.device), array.grad.ptr)
                            if key not in seen:
                                seen.add(key)
                                self.grad_arrays.append(array.grad)
            if self.capture_enabled:
                self._record_forward()   # Compile/warm all kernels outside capture.
                self._zero_grads()
                self.tape.backward(grads=self.seeds)
                wp.synchronize_stream(self.warp_stream)
                with wp.ScopedCapture(stream=self.warp_stream) as captured:
                    self._record_forward()
                self.g_fwd = captured.graph
                with wp.ScopedCapture(stream=self.warp_stream) as captured:
                    self._zero_grads()
                    self.tape.backward(grads=self.seeds)
                self.g_bwd = captured.graph

    @contextmanager
    def _scope(self):
        if self.cuda != is_cuda_execution():
            raise RuntimeError('WithdrawalAdjoint device requires matching cuda_execution context')
        if not self.cuda:
            yield
            return
        if (torch.cuda.current_device() != self.device.index or
                torch.cuda.current_stream(self.device).cuda_stream != self.stream.cuda_stream):
            raise RuntimeError('WithdrawalAdjoint is bound to its constructor CUDA device/stream')
        if (wp.get_stream(self.dev).cuda_stream != self.stream.cuda_stream or
                cuda_module().cuda.get_current_stream().ptr != self.stream.cuda_stream):
            raise RuntimeError('WithdrawalAdjoint requires matching Torch/CuPy/Warp streams')
        with cuda_module().cuda.ExternalStream(self.stream.cuda_stream), wp.ScopedStream(self.warp_stream):
            yield

    def _zero_grads(self):
        for gradient in self.grad_arrays:
            gradient.zero_()

    def _record_forward(self):
        self.tape = wp.Tape()
        with self.tape:
            self.head.rollout()
            self.coast.rollout()

    def forward(self):
        """Replay current private controls; direct calls also invalidate contexts."""
        self.forward_generation += 1
        with self._scope():
            if self.g_fwd is None:
                self._record_forward()
            else:
                wp.capture_launch(self.g_fwd)

    def backward(self):
        with self._scope():
            if self.g_bwd is None:
                self._zero_grads()
                self.tape.backward(grads=self.seeds)
            else:
                wp.capture_launch(self.g_bwd)

    def apply(self, dFc, u=None, body=None):
        # A failed validation/copy must not leave an older autograd context valid.
        self.forward_generation += 1
        with self._scope():
            for name, value, shape, enabled in (
                    ('dFc', dFc, tuple(self.dc.shape), True),
                    ('u', u, tuple(self.u.shape), self.u_wp is not None),
                    ('body', body, tuple(self.body.shape), self.body_wp is not None)):
                if value is None and name == 'u':
                    continue
                if (value is None) != (not enabled):
                    raise ValueError(f'{name} presence must match prepared controls')
                if value is not None and (not torch.is_tensor(value) or tuple(value.shape) != shape
                                          or value.dtype != torch.float32 or value.device != self.device):
                    raise ValueError(f'{name} requires {shape} float32 on {self.device}')
            return WithdrawalOutputs(*_JointWithdrawal.apply(dFc, u, body, self))


class _JointWithdrawal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dFc, u, body, adj):
        ctx.set_materialize_grads(False)
        ctx.adj = adj
        ctx.requested = tuple(value is not None and value.requires_grad for value in (dFc, u, body))
        with adj._scope():
            adj.dc.copy_(dFc.detach())
            adj.u.zero_() if u is None else adj.u.copy_(u.detach())
            if body is not None:
                adj.body.copy_(body.detach())
            adj.forward()
            ctx.generation = adj.forward_generation
            outputs = []
            for index, port in enumerate(adj._ports):
                width = 9 if index in (1, 3, 8, 9) else 3
                if index < 4:
                    outputs.append(wp.to_torch(port[0]).reshape(adj.N, width).clone())
                else:
                    outputs.append(torch.stack([wp.to_torch(array).reshape(adj.N, width) for array in port]))
            return tuple(outputs)

    @staticmethod
    def backward(ctx, *gradients):
        adj = ctx.adj
        if ctx.generation != adj.forward_generation:
            raise RuntimeError('WithdrawalAdjoint backward has stale forward buffers')
        with adj._scope(), torch.no_grad():
            for seed in adj.seed_tensors.values():
                seed.zero_()
            for index, (port, gradient) in enumerate(zip(adj._ports, gradients)):
                if gradient is not None:
                    for t, array in enumerate(port):
                        value = gradient if index < 4 else gradient[t]
                        adj.seed_tensors[array].add_(value.reshape(adj.seed_tensors[array].shape))
            adj.backward()
            dc = torch.stack([wp.to_torch(value.grad).reshape(adj.N, 3, 3).clone()
                              for value in adj.dc_wp]) if ctx.requested[0] else None
            u = wp.to_torch(adj.u_wp.grad).clone() if ctx.requested[1] else None
            body = wp.to_torch(adj.body_wp.grad).clone() if ctx.requested[2] else None
            return dc, u, body, None
