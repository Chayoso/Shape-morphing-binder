"""The exterior's field (render/exterior.ZhuBridson) on the device in one Warp kernel (D138, the runtime phase).

The same field as ZhuBridson's tensor form: f(q) = |q - xbar(q)| - offset, xbar the mean of the particles within the
radius R of q weighted by (1 - d^2 / R^2)^3, the summed weight s, and with `grad` the field's gradient in q, written out
by the chain rule (dhat - J^T dhat, J = d xbar / d q, as Tracked.read writes the normal). The particles are binned in
cells of side R (counting sort on the device), so the 27 cells around a point hold every particle within R of it, as
in ZhuBridson's bins. Only the order of the sums differs from the tensor form (float rounding); the tensor form padded
every point's candidates to the fullest cell's count and differentiated through them, which made a disc search 1.1 s
of a 300k window (docs/experiments.md D138).
"""
from __future__ import annotations

import torch
import warp as wp


@wp.kernel(enable_backward=False)
def k_zb_field(q: wp.array(dtype=wp.vec3), xs: wp.array(dtype=wp.vec3), starts: wp.array(dtype=int),
               lo: wp.vec3, inv_side: float, nx: int, ny: int, nz: int, R2: float, offset: float, want_grad: int,
               f: wp.array(dtype=float), g: wp.array(dtype=wp.vec3), s: wp.array(dtype=float)):
    i = wp.tid()
    qi = q[i]
    c = (qi - lo) * inv_side
    S = float(0.0)
    P = wp.vec3(0.0, 0.0, 0.0)
    dW = wp.vec3(0.0, 0.0, 0.0)
    PdW = wp.mat33(0.0)
    if not (wp.abs(c[0]) < 1.0e6 and wp.abs(c[1]) < 1.0e6 and wp.abs(c[2]) < 1.0e6):   # non-finite or far away
        c = wp.vec3(-10.0, -10.0, -10.0)
    ix = int(wp.floor(c[0]))
    iy = int(wp.floor(c[1]))
    iz = int(wp.floor(c[2]))
    for ox in range(-1, 2):
        cx = ix + ox
        if cx < 0 or cx >= nx:
            continue
        for oy in range(-1, 2):
            cy = iy + oy
            if cy < 0 or cy >= ny:
                continue
            for oz in range(-1, 2):
                cz = iz + oz
                if cz < 0 or cz >= nz:
                    continue
                cell = (cx * ny + cy) * nz + cz
                for k in range(starts[cell], starts[cell + 1]):
                    p = xs[k]
                    r = qi - p
                    u = wp.dot(r, r) / R2
                    if u < 1.0:
                        t = 1.0 - u
                        w = t * t * t
                        S = S + w
                        P = P + w * p
                        if want_grad != 0:
                            dw = (-6.0 / R2) * (t * t) * r
                            dW = dW + dw
                            PdW = PdW + wp.outer(p, dw)
    s[i] = S
    xbar = P / wp.max(S, 1.0e-12)
    d = qi - xbar
    L = wp.length(d)
    if S > 1.0e-6:
        f[i] = L - offset
    else:
        f[i] = float(wp.inf)
    if want_grad != 0:
        Sc = wp.max(S, 1.0e-12)
        J = (PdW - wp.outer(xbar, dW)) / Sc
        dhat = d / wp.max(L, 1.0e-30)
        g[i] = dhat - wp.transpose(J) @ dhat


class DeviceBins:
    """The particles sorted into a dense grid of cells of side `side` (the field's radius) over their bounding box with a
    margin of one cell: the sorted positions and each cell's start (CSR)."""

    def __init__(self, x: torch.Tensor, side: float):
        self.side = float(side)
        self.lo = x.min(0).values - side
        hi = x.max(0).values + side
        self.dims = [int(v) + 1 for v in torch.floor((hi - self.lo) / side).tolist()]
        nx, ny, nz = self.dims
        c = torch.floor((x - self.lo) / side).long()
        cell = (c[:, 0] * ny + c[:, 1]) * nz + c[:, 2]
        order = torch.argsort(cell)
        counts = torch.bincount(cell, minlength=nx * ny * nz)
        self.starts = torch.zeros(nx * ny * nz + 1, dtype=torch.int32, device=x.device)
        self.starts[1:] = torch.cumsum(counts, 0).to(torch.int32)
        self.xs = x[order].contiguous()


def zb_field(x: torch.Tensor, bins: DeviceBins, radius: float, offset: float, q: torch.Tensor, grad: bool = True):
    """(f, g or None, s) of ZhuBridson at the points q, on the device."""
    q = q.detach().float().contiguous()
    n = q.shape[0]
    f = torch.empty(n, device=q.device)
    s = torch.empty(n, device=q.device)
    g = torch.zeros(n if grad else 1, 3, device=q.device)
    if n:
        lo = bins.lo.tolist()
        wp.launch(k_zb_field, dim=n, inputs=[wp.from_torch(q, dtype=wp.vec3), wp.from_torch(bins.xs, dtype=wp.vec3),
                                             wp.from_torch(bins.starts), wp.vec3(*lo), 1.0 / bins.side,
                                             bins.dims[0], bins.dims[1], bins.dims[2], float(radius) ** 2,
                                             float(offset), int(grad)],
                  outputs=[wp.from_torch(f), wp.from_torch(g, dtype=wp.vec3), wp.from_torch(s)],
                  device=str(q.device), stream=wp.stream_from_torch(q.device))
    return f, (g if grad else None), s
