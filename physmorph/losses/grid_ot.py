"""Fixed-target Sinkhorn divergence on the loss grid (settled transport).

The body's mass and the target's are both measured on the loss grid; the debiased
Sinkhorn divergence between them is solved in the log domain with epsilon scaling from the
squared grid diameter down to the blur (one loss cell squared). The squared Euclidean cost
is separable by axis, so every sweep is three one-dimensional log-sum-exp passes (a Warp
kernel inside a captured CUDA graph during no-grad evaluations). The gradient uses the
envelope theorem: the converged potentials, differentiated through the rasterisation
weights. state_energy adds the residual drift |T dt v|^2 and the local support term.
grid_transport_displacement is the label-free transport map of the same quadrature, used
for the per-window transport gate of the outer-layer control.
"""
from __future__ import annotations

import torch
import warp as wp

from .volumetric import rasterize_mass

def _grid_measure(grid, grid_min, dx, dims):
    ids = torch.nonzero(grid > 0, as_tuple=False).flatten()
    if not ids.numel():
        raise ValueError('grid transport requires nonempty measures')
    nx, ny, nz = dims
    nodes = torch.stack((ids // (ny * nz), (ids // nz) % ny, ids % nz), 1)
    return grid_min + dx * nodes.to(grid.dtype), grid[ids] / grid[ids].sum()


@torch.no_grad()
def _grid_potentials(xa, a, y, b, eps, iters, tol):
    if not eps > 0 or not tol > 0 or iters < 1:
        raise ValueError('grid transport requires positive blur, tolerance and sweep budget')
    if len(xa) * len(y) > (1 << 27):
        raise ValueError('grid transport support exceeds the dense solve memory budget')
    C = ((xa * xa).sum(1)[:, None] - 2. * xa @ y.T
         + (y * y).sum(1)[None, :]).clamp_min(0.)
    la, lb = a.log(), b.log()
    f, g = torch.zeros_like(a), torch.zeros_like(b)
    span = torch.maximum(xa.max(0).values, y.max(0).values) - torch.minimum(xa.min(0).values, y.min(0).values)
    level = max(0, int(torch.ceil(torch.log2(span.square().sum().clamp_min(eps) / eps))))
    for iteration in range(iters):
        temperature = eps * 2. ** level
        # Parallel half-updates preserve f == g exactly for identical measures.
        fn = -temperature * torch.logsumexp((g[None, :] - C) / temperature + lb[None, :], 1)
        gn = -temperature * torch.logsumexp((f[:, None] - C) / temperature + la[:, None], 0)
        check = iteration % 4 == 3 or iteration == iters - 1
        if check:
            ea = (a * torch.expm1((f - fn) / temperature).abs()).sum()
            eb = (b * torch.expm1((g - gn) / temperature).abs()).sum()
            error = float(torch.maximum(ea, eb))
        f, g = .5 * (f + fn), .5 * (g + gn)
        if check and error < tol:
            if level == 0:
                break
            level -= 1
    if level != 0 or temperature != eps:
        raise ValueError('grid transport did not reach the requested blur within the sweep budget')
    if not error <= tol:
        raise ValueError(f'grid transport did not converge: marginal error {error:g} > {tol:g}')
    return f, g


class _GridTransportNotConverged(ValueError):
    pass


@wp.kernel(enable_backward=False)
def _grid_logsumexp_axis(field: wp.array(dtype=float), cost: wp.array2d(dtype=float),
                        temperature: wp.array(dtype=float), width: int, stride: int,
                        result: wp.array(dtype=float)):
    i = wp.tid()
    coordinate = (i // stride) % width
    start = i - coordinate * stride
    maximum = float(-wp.inf)
    for j in range(width):
        value = field[start + j * stride] - cost[coordinate, j] / temperature[0]
        if wp.isnan(value):
            result[i] = value
            return
        maximum = wp.max(maximum, value)
    if not wp.isfinite(maximum):
        result[i] = maximum  # Includes an entirely empty (-inf) grid line.
        return
    total = float(0.)
    for j in range(width):
        value = field[start + j * stride] - cost[coordinate, j] / temperature[0]
        total += wp.exp(value - maximum)
    result[i] = maximum + wp.log(total)


class GridSinkhornLoss:
    """Fixed-target Sinkhorn divergence with a separable full-grid envelope."""

    def __init__(self, target, grid_min, dx, dims, *, eps, iters=400, tol=1e-3,
                 mass_total=None, cuda_blocks=False, support=None):
        if not eps > 0 or not tol > 0 or iters < 1 or not float(target.sum()) > 0:
            raise ValueError('grid transport requires positive mass, blur, tolerance and sweep budget')
        self.dims, self.eps, self.iters, self.tol = dims, eps, iters, tol
        self.grid_min, self.dx = grid_min, dx
        self.mass_total = mass_total
        self.support = support
        self.b = target.detach() / target.sum()
        self.costs = [((torch.arange(n, device=target.device, dtype=target.dtype)[:, None]
                       - torch.arange(n, device=target.device, dtype=target.dtype)[None, :]) * dx).square()
                      for n in dims]
        self.diameter2 = dx ** 2 * sum((n - 1) ** 2 for n in dims)
        self.cuda_blocks = False
        self._solve_graphs = {}
        self.target_potential, _ = self.solve(self.b, self.b)
        self.cuda_blocks = cuda_blocks

    def transform(self, dual, log_weights, temperature):
        # Squared Euclidean cost is additive by axis. Log-domain separability
        # is exact and avoids both dense 3D costs and underflow of exp(-C/eps).
        if (self.cuda_blocks and dual.is_cuda and not torch.is_grad_enabled()
                and dual.dtype == log_weights.dtype == self.b.dtype == torch.float32):
            # Same stable log-domain reduction, without grid-by-axis temporaries.
            # Use Torch's CURRENT stream, including the non-default capture stream.
            temp = torch.as_tensor(temperature, device=dual.device, dtype=dual.dtype).reshape(1)
            field = (dual / temp + log_weights).reshape(self.dims).reshape(-1)
            stream = wp.stream_from_torch(dual.device)
            stride = field.numel()
            for width, cost in zip(self.dims, self.costs):
                stride //= width
                result = torch.empty_like(field)
                wp.launch(_grid_logsumexp_axis, dim=field.numel(),
                          inputs=[wp.from_torch(field), wp.from_torch(cost),
                                  wp.from_torch(temp), width, stride],
                          outputs=[wp.from_torch(result)], device=stream.device, stream=stream)
                field = result
            return -temp * field
        field = (dual / temperature + log_weights).reshape(self.dims)
        for axis, cost in enumerate(self.costs):
            field = field.movedim(axis, -1)
            field = torch.logsumexp(field.unsqueeze(-2) - cost / temperature, -1)
            field = field.movedim(-1, axis)
        return -temperature * field.reshape(-1)

    @torch.no_grad()
    def solve(self, a, b):
        if (self.cuda_blocks and a.is_cuda and self.iters % 4 == 0
                and a.device == b.device == self.b.device
                and a.dtype == b.dtype == self.b.dtype
                and a.shape == b.shape == self.b.shape):
            with torch.cuda.device(a.device):
                return self._solve_cuda_blocks(a, b)
        la, lb = a.log(), b.log()
        f, g = torch.zeros_like(a), torch.zeros_like(b)
        level = max(0, int(torch.ceil(torch.log2(a.new_tensor(max(self.diameter2, self.eps) / self.eps)))))
        for iteration in range(self.iters):
            temperature = self.eps * 2. ** level
            fn = self.transform(g, lb, temperature)
            # Identical measures keep f == g exactly under the parallel update.
            gn = fn if a is b else self.transform(f, la, temperature)
            check = iteration % 4 == 3 or iteration == self.iters - 1
            if check:
                ea = (torch.exp(la + (f - fn) / temperature) - a).abs().sum()
                eb = (torch.exp(lb + (g - gn) / temperature) - b).abs().sum()
                error = float(torch.maximum(ea, eb))
            f, g = .5 * (f + fn), .5 * (g + gn)
            if check and error < self.tol:
                if level == 0:
                    break
                level -= 1
        if level != 0 or temperature != self.eps or not error <= self.tol:
            raise _GridTransportNotConverged(f'grid transport did not converge: marginal error {error:g}')
        return f, g

    def _solve_cuda_blocks(self, a, b):
        """Capture four unchanged sweeps; keep the original residual/blur schedule.

        Every call starts from zero duals. Graphs own their buffers, not a temporal
        warm start: line-search values must not depend on previous trials.
        Like the owning optimizer, a loss instance is used serially.
        """
        same = a is b
        if same not in self._solve_graphs:
            la, lb = a.log().clone(), b.log().clone()
            f, g = torch.zeros_like(a), torch.zeros_like(b)
            temp, error = a.new_tensor(self.eps), a.new_zeros(())
            abuf, bbuf = a.clone(), b.clone()

            def block():
                for i in range(4):
                    fn = self.transform(g, lb, temp)
                    gn = fn if same else self.transform(f, la, temp)
                    if i == 3:
                        ea = (torch.exp(la + (f - fn) / temp) - abuf).abs().sum()
                        eb = (torch.exp(lb + (g - gn) / temp) - bbuf).abs().sum()
                        error.copy_(torch.maximum(ea, eb))
                    f.copy_(.5 * (f + fn))
                    g.copy_(.5 * (g + gn))

            stream = torch.cuda.Stream(device=a.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                block()
                block()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                block()
            self._solve_graphs[same] = (graph, la, lb, f, g, temp, error, abuf, bbuf)
        graph, la, lb, f, g, temp, error, abuf, bbuf = self._solve_graphs[same]
        la.copy_(a.log())
        lb.copy_(b.log())
        abuf.copy_(a)
        bbuf.copy_(b)
        f.zero_()
        g.zero_()
        level = max(0, int(torch.ceil(torch.log2(a.new_tensor(max(self.diameter2, self.eps) / self.eps)))))
        for iteration in range(0, self.iters, 4):
            temperature = self.eps * 2. ** level
            temp.fill_(temperature)
            graph.replay()
            residual = float(error)
            if residual < self.tol:
                if level == 0:
                    break
                level -= 1
        if level != 0 or temperature != self.eps or not residual <= self.tol:
            raise _GridTransportNotConverged(f'grid transport did not converge: marginal error {residual:g}')
        # The next cross/self solve rewrites the capture buffers.
        return f.clone(), g.clone()

    def __call__(self, current):
        total = current.sum()
        if self.mass_total is not None and abs(float(total.detach()) - self.mass_total) > 1e-6 * self.mass_total:
            # Never reward disappearance beyond the raster boundary.
            return total * 0. + float('inf')
        if not float(total.detach()) > 0:
            raise ValueError('grid transport requires nonempty measures')
        a = current / total
        needs_gradient = torch.is_grad_enabled() and current.requires_grad
        try:
            f, g = self.solve(a, self.b)
            fs, gs = self.solve(a, a)
        except _GridTransportNotConverged:
            if needs_gradient:
                raise
            # An unsolved trial is inadmissible, never an approximate descent step.
            return total * 0. + float('inf')
        # Self solves are symmetric (f == g). Subtract potentials before reducing
        # so tiny divergences do not lose precision against O(1) self-energies.
        value = ((a.double() * (f.double() - fs.double())).sum()
                 + (self.b.double() * (g.double() - self.target_potential.double())).sum()).to(current.dtype)
        if not needs_gradient:
            return value
        with torch.no_grad():
            # Includes zero-mass nodes, so entering an empty CIC node has the
            # correct one-sided derivative even at grid-aligned particles.
            phi = (self.transform(g, self.b.log(), self.eps)
                   - self.transform(gs, a.log(), self.eps))
        envelope = (a * phi).sum()
        return value.detach() + (envelope - envelope.detach())

    def state_energy(self, x, mass, velocity, horizon):
        """Transport plus squared residual displacement, both in length-squared units.

        The physically simulated release tail supplies the actual endpoint.
        The horizon sets the position/velocity conversion without another weight.
        This is a terminal cost, not an edit to the simulated positions/velocities.
        """
        if self.mass_total is not None:
            actual = rasterize_mass(x, mass, self.grid_min, self.dx, self.dims)
            if abs(float(actual.sum().detach()) - self.mass_total) > 1e-6 * self.mass_total:
                # Inward velocity must not conceal particles already outside.
                return actual.sum() * 0. + velocity.sum() * 0. + float('inf')
        drift = horizon * velocity
        current = rasterize_mass(x, mass, self.grid_min, self.dx, self.dims)
        value = self(current) + (mass * drift.square().sum(1)).sum() / mass.sum()
        return value if self.support is None else self.support(value, x)


@torch.no_grad()
def grid_transport_displacement(x, mass, target_grid, grid_min, dx, dims,
                                *, eps, iters=400, tol=1e-2):
    """Label-independent transport map on the same spatial mass quadrature."""
    current = rasterize_mass(x, mass, grid_min, dx, dims)
    xa, a = _grid_measure(current, grid_min, dx, dims)
    yb, b = _grid_measure(target_grid, grid_min, dx, dims)

    def mapped(y, weights):
        _, g = _grid_potentials(xa, a, y, weights, eps, iters, tol)
        lb = weights.log()
        result = torch.empty_like(x)
        chunk = max(256, (1 << 26) // len(y))
        yy = (y * y).sum(1)[None, :]
        for start in range(0, len(x), chunk):
            q = x[start:start + chunk]
            cost = (q * q).sum(1)[:, None] - 2. * q @ y.T + yy
            result[start:start + chunk] = torch.softmax((g[None, :] - cost) / eps + lb[None, :], 1) @ y
        return result

    return mapped(yb, b) - mapped(xa, a)

