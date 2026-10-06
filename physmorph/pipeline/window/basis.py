"""The window's stress control on the grid (D109).

The control dFc is one 3x3 per driven step on every grid node the window's particles touch; each particle reads it
with the simulation's own cubic B-spline weights (kernels.base_node, constitutive.weight) at its start position,
dFc_p = sum_i w_ip C_i (G2P). A free per-particle control let a gradient concentrated on a few particles move those
particles alone, out of the body (D108), and its sub-cell part moved the layer away from the target late in the run
(D107): the grid carries no stress that differs below a cell. Frozen for the window, as the layer graph is.
"""
from __future__ import annotations

import torch


class _Interp(torch.autograd.Function):
    """out = W @ flat with the adjoint W^T @ grad (both sparse CSR, built once)."""

    @staticmethod
    def forward(ctx, flat, W, Wt):
        ctx.Wt = Wt
        return W @ flat

    @staticmethod
    def backward(ctx, grad):
        return ctx.Wt @ grad.contiguous(), None, None


class GridBasis:
    """G2P of node controls to the particles and its mass-weighted P2G inverse, with the MPM's cubic B-spline."""

    def __init__(self, x0: torch.Tensor, dx: float, grid_min, dims):
        x0 = x0.detach().double()
        dev = x0.device
        N = x0.shape[0]
        X = (x0 - torch.as_tensor(grid_min, dtype=torch.float64, device=dev)) / dx
        base = torch.floor(X).long() - 1                    # kernels.base_node: the 4^3 stencil of the cubic B-spline
        o = torch.arange(4, device=dev)
        r = base[None].double() + o[:, None, None].double() - X[None]                             # (4, N, 3), in cells
        ar = r.abs()
        w = torch.where(ar < 1.0, 0.5 * ar ** 3 - ar ** 2 + 2.0 / 3.0,
                        torch.where(ar < 2.0, (2.0 - ar).clamp_min(0.0) ** 3 / 6.0, torch.zeros_like(ar)))
        a, b, c = torch.meshgrid(o, o, o, indexing="ij")
        a, b, c = a.reshape(-1), b.reshape(-1), c.reshape(-1)                                     # the 64 nodes
        weight = (w[a, :, 0] * w[b, :, 1] * w[c, :, 2]).T                                         # (N, 64)
        node = base[:, None, :] + torch.stack((a, b, c), 1)[None]                                 # (N, 64, 3)
        nx, ny, nz = (int(d) for d in dims)
        inside = ((node >= 0) & (node < torch.as_tensor((nx, ny, nz), device=dev))).all(-1)
        weight = torch.where(inside, weight, torch.zeros_like(weight))
        weight = weight / weight.sum(1, keepdim=True).clamp_min(1e-12)                          # a truncated stencil
        key = (node[..., 0].clamp(0, nx - 1) * ny + node[..., 1].clamp(0, ny - 1)) * nz + node[..., 2].clamp(0, nz - 1)
        uniq, inv = torch.unique(key.reshape(-1), return_inverse=True)
        self.N, self.M = N, int(uniq.numel())
        self.node_ijk = torch.stack((uniq // (ny * nz), (uniq // nz) % ny, uniq % nz), 1)        # (M, 3) grid indices
        rows = torch.arange(N, device=dev).repeat_interleave(64)
        vals = weight.reshape(-1).float()
        W = torch.sparse_coo_tensor(torch.stack((rows, inv)), vals, (N, self.M)).coalesce()
        self.W = W.to_sparse_csr()
        self.Wt = W.t().coalesce().to_sparse_csr()
        self.mass = (self.Wt @ torch.ones(N, 1, device=dev)).reshape(-1)                         # sum_p w_ip

    def zeros(self, T: int, device=None) -> torch.Tensor:
        return torch.zeros(T, self.M, 3, 3, device=device or self.mass.device)

    def to_particles(self, C: torch.Tensor) -> torch.Tensor:
        """(T, M, 3, 3) node controls -> (T, N, 3, 3) particle controls (differentiable in C)."""
        T = C.shape[0]
        flat = C.permute(1, 0, 2, 3).reshape(self.M, T * 9)
        return _Interp.apply(flat, self.W, self.Wt).reshape(self.N, T, 3, 3).permute(1, 0, 2, 3)

    def to_nodes(self, dfc: torch.Tensor) -> torch.Tensor:
        """(T, N, 3, 3) particle controls -> (T, M, 3, 3): each node's mass-weighted mean of its particles' (P2G)."""
        T = dfc.shape[0]
        flat = dfc.detach().permute(1, 0, 2, 3).reshape(self.N, T * 9).contiguous()
        return ((self.Wt @ flat) / self.mass.clamp_min(1e-12)[:, None]).reshape(self.M, T, 3, 3).permute(1, 0, 2, 3)

    def dot(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """The inner product of two node gradients in the nodes' mass metric, sum_i a_i . b_i / m_i: for a smooth
        gradient the per-particle one, so the step control's legacy norms keep their meaning."""
        return ((a * b).sum(dim=(0, 2, 3)) / self.mass.clamp_min(1e-12)).sum()
