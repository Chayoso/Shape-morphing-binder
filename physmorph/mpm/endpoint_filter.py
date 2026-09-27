"""Fixed-stencil PIC/XPIC and its exact constant-stencil adjoint.

P = S^-1 W D^-1 W^T M, with the legacy 1e-12 lower bounds on S and D.
H = I-(I-P)^order is finite-order XPIC, not an idempotent projection.
Only a supplied field/endpoint is differentiated; the window-start stencil,
reference positions and masses are owned constants. All numerical work stays
on the input Torch device. CUDA index_add_ may have atomic reduction noise.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import torch


def _cubic(t: torch.Tensor) -> torch.Tensor:
    a = t.abs()
    return torch.where(a < 1, .5 * a**3 - a**2 + 2 / 3,
                       torch.where(a < 2, (2 - a)**3 / 6, torch.zeros_like(a)))


class _LinearXPIC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, field, prepared, transpose):
        ctx.prepared = prepared
        ctx.transpose = transpose
        return prepared._apply(field, transpose)

    @staticmethod
    def backward(ctx, grad_output):
        # Re-enter the same linear primitive so create_graph also has the
        # correct adjoint; no forward scatter/gather intermediates are saved.
        return _LinearXPIC.apply(grad_output, ctx.prepared, not ctx.transpose), None, None


@dataclass(frozen=True, init=False, eq=False)
class FixedEndpointFilter:
    """Prepare once at a window start, reuse for gradient/evaluation/commit.

    x0 is a finite float32/float64 Torch tensor of shape (N,3). grid_min may
    be three Python constants or a tensor on the same device. m is None
    (unit masses), a nonnegative scalar or an (N,) tensor on that device;
    zero mass is allowed and follows the legacy clamped-denominator convention.
    Tensor inputs are detached and copied. Private buffers must not be mutated;
    the public x0 accessor returns a copy. Construct another object to refresh
    the stencil. No gradient is defined through x0, m or grid_min.
    """

    _x0: torch.Tensor
    _mass: torch.Tensor
    _weights: tuple[torch.Tensor, ...]
    _indices: tuple[torch.Tensor, ...]
    _weight_sum: torch.Tensor
    _grid_mass: torch.Tensor
    _dims: tuple[int, int, int]
    _order: int

    def __init__(self, x0: torch.Tensor, dx, grid_min, dims, m=None, order=5):
        if not torch.is_tensor(x0):
            raise TypeError("x0 must be a Torch tensor")
        if x0.ndim != 2 or x0.shape[1] != 3 or x0.dtype not in (torch.float32, torch.float64):
            raise ValueError("x0 must have shape (N,3) and dtype float32 or float64")
        if x0.device.type not in ("cpu", "cuda"):
            raise ValueError("only Torch CPU/CUDA devices are supported")
        if not bool(torch.isfinite(x0).all()):
            raise ValueError("x0 must be finite")
        if isinstance(dx, bool) or not isinstance(dx, Real) or not math.isfinite(dx) or dx <= 0:
            raise ValueError("dx must be a finite positive scalar")
        dims = tuple(dims)
        if len(dims) != 3 or any(isinstance(v, bool) or not isinstance(v, Integral) or v <= 0 for v in dims):
            raise ValueError("dims must contain three positive integers")
        dims = tuple(int(v) for v in dims)
        if isinstance(order, bool) or not isinstance(order, Integral) or order < 1:
            raise ValueError("order must be a positive integer")
        if torch.is_tensor(grid_min) and grid_min.device != x0.device:
            raise ValueError("grid_min tensor must be on the x0 device")
        if torch.is_tensor(m) and m.device != x0.device:
            raise ValueError("mass tensor must be on the x0 device")

        with torch.no_grad():
            reference = x0.detach().clone().contiguous()
            origin = torch.as_tensor(grid_min, dtype=x0.dtype, device=x0.device).detach().clone()
            if origin.shape != (3,) or not bool(torch.isfinite(origin).all()):
                raise ValueError("grid_min must contain three finite coordinates")
            mass = (torch.ones(len(x0), dtype=x0.dtype, device=x0.device) if m is None
                    else torch.as_tensor(m, dtype=x0.dtype, device=x0.device).detach().clone())
            if mass.numel() == 1:
                mass = mass.reshape(()).expand(len(x0)).clone()
            if mass.shape != (len(x0),) or not bool(torch.isfinite(mass).all()) or bool((mass < 0).any()):
                raise ValueError("m must be a finite nonnegative scalar or (N,) tensor")
            nx, ny, nz = dims
            p = (reference - origin) * (1.0 / float(dx))
            base = torch.floor(p).long() - 1
            weights, indices = [], []
            for oi in range(4):
                for oj in range(4):
                    for ok in range(4):
                        node = base + torch.tensor((oi, oj, ok), device=x0.device)
                        t = node.to(x0.dtype) - p
                        w = _cubic(t[:, 0]) * _cubic(t[:, 1]) * _cubic(t[:, 2])
                        valid = ((node[:, 0] >= 0) & (node[:, 0] < nx)
                                 & (node[:, 1] >= 0) & (node[:, 1] < ny)
                                 & (node[:, 2] >= 0) & (node[:, 2] < nz))
                        weights.append(w * valid)
                        indices.append(((node[:, 0] * ny + node[:, 1]) * nz
                                        + node[:, 2]).clamp(0, nx * ny * nz - 1))
            weight_sum = torch.stack(weights).sum(0).clamp_min(1e-12)
            grid_mass = reference.new_zeros(nx * ny * nz)
            for w, ids in zip(weights, indices):
                grid_mass.index_add_(0, ids, w * mass)
            grid_mass.clamp_min_(1e-12)
        for name, value in dict(_x0=reference, _mass=mass, _weights=tuple(weights),
                                _indices=tuple(indices), _weight_sum=weight_sum,
                                _grid_mass=grid_mass, _dims=dims, _order=int(order)).items():
            object.__setattr__(self, name, value)

    @property
    def x0(self):
        """An independent copy of the owned reference positions."""
        return self._x0.clone()

    @property
    def order(self):
        return self._order

    def _check_field(self, field):
        if not torch.is_tensor(field):
            raise TypeError("field must be a Torch tensor")
        if field.ndim != 2 or field.shape[0] != len(self._x0) or field.shape[1] < 1:
            raise ValueError("field must have shape (N,C), C>=1")
        if field.device != self._x0.device or field.dtype != self._x0.dtype:
            raise ValueError("field must match the prepared dtype and device")

    def _pic(self, field, transpose):
        # P uses mass on scatter and retained-stencil normalization on gather.
        # P^T reverses those placements; it is generally different from P.
        scatter_field = field / self._weight_sum[:, None] if transpose else field
        grid = field.new_zeros((len(self._grid_mass), field.shape[1]))
        for w, ids in zip(self._weights, self._indices):
            sw = w if transpose else w * self._mass
            grid.index_add_(0, ids, sw[:, None] * scatter_field)
        grid = grid / self._grid_mass[:, None]
        out = torch.zeros_like(field)
        for w, ids in zip(self._weights, self._indices):
            gw = w * self._mass if transpose else w
            out += gw[:, None] * grid[ids]
        return out if transpose else out / self._weight_sum[:, None]

    def _apply(self, field, transpose):
        residual = field.clone()
        for _ in range(self._order):
            residual = residual - self._pic(residual, transpose)
        return field - residual

    def apply_H(self, d: torch.Tensor) -> torch.Tensor:
        """Apply H to an (N,C) field, with custom H^T reverse-mode derivative."""
        self._check_field(d)
        return _LinearXPIC.apply(d, self, False)

    def apply_HT(self, g: torch.Tensor) -> torch.Tensor:
        """Apply H^T, including both mass and retained-weight normalization."""
        self._check_field(g)
        return _LinearXPIC.apply(g, self, True)

    def endpoint(self, raw: torch.Tensor, pin_mask=None) -> torch.Tensor:
        """x0 + Q H(raw-x0); gradient reaches raw only, as H^T Q g.

        pin_mask is an optional (N,) Torch bool/numeric tensor on the same
        device; values >0.5 pin a row, as in the rollout. The mask is copied.
        Q acts after H: a fixed output row does not imply a zero column of H.
        """
        self._check_field(raw)
        if raw.shape[1] != 3:
            raise ValueError("raw endpoint must have shape (N,3)")
        result = self._x0 + self.apply_H(raw - self._x0)
        if pin_mask is not None:
            if not torch.is_tensor(pin_mask):
                raise TypeError("pin_mask must be a Torch tensor")
            if pin_mask.shape != (len(raw),) or pin_mask.device != raw.device:
                raise ValueError("pin_mask must have shape (N,) on the prepared device")
            if pin_mask.dtype != torch.bool and not bool(torch.isfinite(pin_mask).all()):
                raise ValueError("pin_mask must be finite")
            pins = (pin_mask.detach() > .5).clone()
            # Restore the reference exactly, including a signed zero, as the
            # legacy commit does; the raw pullback is still H^T Q.
            result = torch.where(pins[:, None], self._x0, result)
        return result
