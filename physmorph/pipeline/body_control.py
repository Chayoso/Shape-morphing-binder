"""Cell-scale external force, parameterised by nominal window displacement.

The basis is frozen at window start. The first pulse has zero impulse and unit
free displacement. The optional second pulse has zero free displacement and
nonzero impulse. Elasticity, contact and damping determine the actual motion.
"""
from __future__ import annotations

from physmorph.compute import array_api as np, is_cuda_execution
import torch


def rest_to_rest_pulse(T: int, dt: float) -> np.ndarray:
    if T < 2 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("body control needs T >= 2 and positive finite dt")
    q = T - 1 - 2 * np.arange(T, dtype=np.float64)
    return (q / (dt * dt * np.dot(T - np.arange(T), q))).astype(np.float32)


def terminal_velocity_pulse(T: int, dt: float) -> np.ndarray:
    """Zero free displacement; terminal velocity 1/(T*dt) per unit coefficient."""
    a = rest_to_rest_pulse(T, dt).astype(np.float64)
    return (1 / (T * dt) ** 2 - (T + 1) / (2 * T) * a).astype(np.float32)


class BodyControlBasis:
    """Trilinear vector coefficients on occupied nodes of the MPM lattice."""

    def __init__(self, x: np.ndarray, origin, dx: float, device="cpu"):
        x = np.asarray(x, np.float32)
        if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all() or dx <= 0:
            raise ValueError("finite (N,3) positions and positive dx required")
        r = (x - np.asarray(origin, np.float32)) / float(dx)
        base = np.floor(r).astype(np.int64)
        f = r - base
        offsets = np.asarray([(i, j, k) for i in (0, 1) for j in (0, 1)
                              for k in (0, 1)], np.int64)
        node_rows = (base[:, None] + offsets).reshape(-1, 3)
        if is_cuda_execution():
            # CuPy14 unique(axis=0) compares rows in a Python loop. A bounded
            # integer lattice key gives the identical lexicographic node order.
            lower = node_rows.min(0)
            widths = node_rows.max(0) - lower + 1
            width_y, width_z = int(widths[1]), int(widths[2])
            if int(widths[0]) * width_y * width_z >= 2**63:
                raise ValueError('body lattice key exceeds int64 capacity')
            shifted = node_rows - lower
            keys = (shifted[:, 0] * width_y + shifted[:, 1]) * width_z + shifted[:, 2]
            nodes, inv = np.unique(keys, return_inverse=True)
        else:
            nodes, inv = np.unique(node_rows, axis=0, return_inverse=True)
        weights = np.prod(np.where(offsets[None] != 0, f[:, None], 1 - f[:, None]), axis=2)
        self.idx = torch.as_tensor(inv.reshape(len(x), 8), device=device)
        self.weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
        self.n_nodes, self.dx, self.device = len(nodes), float(dx), device

    def zeros(self, modes=1):
        if modes not in (1, 2):
            raise ValueError("body control supports one or two temporal modes")
        return torch.zeros(self.n_nodes, 3 * modes, device=self.device, requires_grad=True)

    def expand(self, coefficients):
        return self.dx * (coefficients[self.idx] * self.weights[..., None]).sum(1)

    @torch.no_grad()
    def displacement_step_scale(self, particle_scale, active, arrived):
        """Restrict raw RPROP scales; any live transit contribution protects its node.

        This conditions the optimizer's displacement-mode update only. It neither
        scales the physical force nor changes the independently optimized brake.
        Pinned/passive particles must be absent from ``active``.
        """
        n = len(self.idx)
        if any(value.shape != (n,) or value.device != self.idx.device
               for value in (particle_scale, active, arrived)):
            raise ValueError('body step inputs must be device-matched particle vectors')
        if (not torch.isfinite(particle_scale).all() or (particle_scale < 0).any()
                or (particle_scale > 1).any()):
            raise ValueError('RPROP scales must be finite and in [0,1]')
        active, arrived = active.bool(), arrived.bool()
        weights = self.weights * active[:, None]
        ids = self.idx.reshape(-1)
        denom = self.weights.new_zeros(self.n_nodes)
        numerator = torch.zeros_like(denom)
        transit = torch.zeros_like(denom)
        denom.index_add_(0, ids, weights.reshape(-1))
        numerator.index_add_(0, ids, (weights * particle_scale[:, None]).reshape(-1))
        transit.index_add_(0, ids, ((weights > 0) & ~arrived[:, None]).float().reshape(-1))
        node_scale = torch.where(denom > 0, numerator / denom.clamp_min(1e-30), 0.)
        node_scale = torch.where(transit > 0, 1., node_scale).clamp(0., 1.)
        effective = (node_scale[self.idx] * self.weights).sum(1)
        occupied = denom > 0
        moving = active & ~arrived
        settled = active & arrived
        telemetry = dict(
            body_step_node_mean=float(node_scale[occupied].mean()) if occupied.any() else None,
            body_step_transit_nodes_frac=float((transit[occupied] > 0).float().mean()) if occupied.any() else None,
            body_step_transit_min=float(effective[moving].min()) if moving.any() else None,
            body_step_arrived_median=float(effective[settled].median()) if settled.any() else None)
        return node_scale[:, None], telemetry
