"""The exterior's field (render/exterior.ZhuBridson) on the device in one Warp kernel (D138, the runtime phase).

The same field as ZhuBridson's tensor form: f(q) = |q - xbar(q)| - offset, xbar the mean of the particles within the
radius R of q weighted by (1 - d^2 / R^2)^3, the summed weight s, and with `grad` the field's gradient in q, written out
by the chain rule (dhat - J^T dhat, J = d xbar / d q, as Tracked.read writes the normal). The particles are binned in
cells of side R (counting sort on the device), so the 27 cells around a point hold every particle within R of it, as
in ZhuBridson's bins. Only the order of the sums differs from the tensor form (float rounding); the tensor form padded
every point's candidates to the fullest cell's count and differentiated through them, which made a disc search 1.1 s
of a 300k window (docs/experiments.md D138).

D140: the anisotropic field (render/exterior.Anisotropic) in one kernel of the same form, the cells' side its largest
kernel's reach.
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
        self.order = order                                   # D140: per-particle data in the same order


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


@wp.kernel(enable_backward=False)
def k_aniso_field(q: wp.array(dtype=wp.vec3), xs: wp.array(dtype=wp.vec3), cis: wp.array(dtype=wp.mat33),
                  starts: wp.array(dtype=int), lo: wp.vec3, inv_side: float, nx: int, ny: int, nz: int,
                  level: float, scale: float, want_grad: int,
                  f: wp.array(dtype=float), g: wp.array(dtype=wp.vec3), s: wp.array(dtype=float)):
    """D140: the anisotropic field (render/exterior.Anisotropic) at the points q: Phi = sum_j (1 - s_j^2 / 9)^3 on
    s_j < 3, s_j^2 = (q - x_j)^T C_j^-1 (q - x_j); f = (level - Phi^(1/3)) scale, its gradient, s = Phi.
    The cells' side is the largest kernel's reach, so the 27 cells around a point hold every particle that reaches it."""
    i = wp.tid()
    qi = q[i]
    c = (qi - lo) * inv_side
    P = float(0.0)
    G = wp.vec3(0.0, 0.0, 0.0)
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
                    r = qi - xs[k]
                    Cr = cis[k] @ r
                    u = wp.dot(r, Cr) / 9.0
                    if u < 1.0:
                        t = 1.0 - u
                        P = P + t * t * t
                        if want_grad != 0:
                            G = G + (-2.0 / 3.0) * (t * t) * Cr
    s[i] = P
    cr = wp.pow(wp.max(P, 1.0e-30), 1.0 / 3.0)
    if P > 0.0:
        f[i] = (level - cr) * scale
    else:
        f[i] = float(wp.inf)
    if want_grad != 0:
        g[i] = (-scale / (3.0 * cr * cr)) * G


def aniso_field(bins: DeviceBins, cis_sorted: torch.Tensor, level: float, scale: float, q: torch.Tensor,
                grad: bool = True):
    """(f, g or None, s) of the anisotropic field at the points q (D140), with the particles' inverse covariances in
    the bins' order; on the points' device (CUDA or Warp's CPU)."""
    q = q.detach().float().contiguous()
    n = q.shape[0]
    f = torch.empty(n, device=q.device)
    s = torch.empty(n, device=q.device)
    g = torch.zeros(n if grad else 1, 3, device=q.device)
    if n:
        lo = bins.lo.tolist()
        stream = {"stream": wp.stream_from_torch(q.device)} if q.is_cuda else {}
        wp.launch(k_aniso_field, dim=n, inputs=[wp.from_torch(q, dtype=wp.vec3), wp.from_torch(bins.xs, dtype=wp.vec3),
                                                wp.from_torch(cis_sorted, dtype=wp.mat33), wp.from_torch(bins.starts),
                                                wp.vec3(*lo), 1.0 / bins.side, bins.dims[0], bins.dims[1], bins.dims[2],
                                                float(level), float(scale), int(grad)],
                  outputs=[wp.from_torch(f), wp.from_torch(g, dtype=wp.vec3), wp.from_torch(s)],
                  device=str(q.device), **stream)
    return f, (g if grad else None), s


@wp.func
def zb_sums(qi: wp.vec3, xs: wp.array(dtype=wp.vec3), starts: wp.array(dtype=int), lo: wp.vec3, inv_side: float,
            nx: int, ny: int, nz: int, R2: float):
    """Zhu and Bridson's sums at qi: the summed weight S, the weighted positions P, the weights' gradient dW and
    sum p (x) grad w."""
    c = (qi - lo) * inv_side
    S = float(0.0)
    P = wp.vec3(0.0, 0.0, 0.0)
    dW = wp.vec3(0.0, 0.0, 0.0)
    PdW = wp.mat33(0.0)
    if not (wp.abs(c[0]) < 1.0e6 and wp.abs(c[1]) < 1.0e6 and wp.abs(c[2]) < 1.0e6):
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
                        dw = (-6.0 / R2) * (t * t) * r
                        dW = dW + dw
                        PdW = PdW + wp.outer(p, dw)
    return S, P, dW, PdW


@wp.func
def cbrt_real(v: float):
    return wp.sign(v) * wp.pow(wp.abs(v), 1.0 / 3.0)


@wp.func
def ev_max3(J: wp.mat33):
    """The largest real part of the eigenvalues of J (render/exterior.ev_max's closed form)."""
    T = J[0, 0] + J[1, 1] + J[2, 2]
    M = (J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0] + J[0, 0] * J[2, 2] - J[0, 2] * J[2, 0]
         + J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1])
    D = wp.determinant(J)
    p = M - T * T / 3.0
    q = -2.0 * T * T * T / 27.0 + M * T / 3.0 - D
    disc = q * q / 4.0 + p * p * p / 27.0
    y = float(0.0)
    if disc <= 0.0:
        pn = wp.max(-p, 1.0e-30)
        arg = wp.clamp(1.5 * q / (-pn) * wp.sqrt(3.0 / pn), -1.0, 1.0)
        y = 2.0 * wp.sqrt(pn / 3.0) * wp.cos(wp.acos(arg) / 3.0)
    else:
        sd = wp.sqrt(disc)
        y1 = cbrt_real(-q / 2.0 + sd) + cbrt_real(-q / 2.0 - sd)
        y = wp.max(y1, -y1 / 2.0)
    return y + T / 3.0


@wp.func
def zbs_ev(qi: wp.vec3, xs: wp.array(dtype=wp.vec3), starts: wp.array(dtype=int), lo: wp.vec3, inv_side: float,
           nx: int, ny: int, nz: int, R2: float):
    S, P, dW, PdW = zb_sums(qi, xs, starts, lo, inv_side, nx, ny, nz, R2)
    Sc = wp.max(S, 1.0e-12)
    xbar = P / Sc
    return ev_max3((PdW - wp.outer(xbar, dW)) / Sc)


@wp.kernel(enable_backward=False)
def k_zbs_field(q: wp.array(dtype=wp.vec3), xs: wp.array(dtype=wp.vec3), starts: wp.array(dtype=int),
                lo: wp.vec3, inv_side: float, nx: int, ny: int, nz: int, R2: float, offset: float,
                t_low: float, t_high: float, h: float, want_grad: int,
                f: wp.array(dtype=float), g: wp.array(dtype=wp.vec3), s: wp.array(dtype=float),
                ev_out: wp.array(dtype=float), gev_out: wp.array(dtype=wp.vec3)):
    """D144: Zhu and Bridson's field with Solenthaler et al.'s correction (render/exterior.ZhuBridsonSolenthaler):
    f = |q - xbar| - offset fac(EVmax); with want_grad its gradient (Zhu and Bridson's less offset fac' grad EVmax,
    the last by central differences of step h where fac varies) and EVmax with its difference gradient (want_grad 2:
    everywhere)."""
    i = wp.tid()
    qi = q[i]
    S, P, dW, PdW = zb_sums(qi, xs, starts, lo, inv_side, nx, ny, nz, R2)
    s[i] = S
    Sc = wp.max(S, 1.0e-12)
    xbar = P / Sc
    J = (PdW - wp.outer(xbar, dW)) / Sc
    ev = ev_max3(J)
    gam = (t_high - ev) / (t_high - t_low)
    fac = float(1.0)
    dfac = float(0.0)
    if ev > t_low:
        fac = wp.clamp(gam * gam * gam - 3.0 * gam * gam + 3.0 * gam, 0.0, 1.0)
        if ev < t_high:
            dfac = -3.0 * (1.0 - gam) * (1.0 - gam) / (t_high - t_low)
    d = qi - xbar
    L = wp.length(d)
    if S > 1.0e-6:
        f[i] = L - offset * fac
    else:
        f[i] = float(wp.inf)
    if want_grad != 0:
        dhat = d / wp.max(L, 1.0e-30)
        gr = dhat - wp.transpose(J) @ dhat
        gev = wp.vec3(0.0, 0.0, 0.0)
        if dfac != 0.0 or want_grad == 2:
            for k in range(3):
                e = wp.vec3(0.0, 0.0, 0.0)
                e[k] = h
                gev[k] = (zbs_ev(qi + e, xs, starts, lo, inv_side, nx, ny, nz, R2)
                          - zbs_ev(qi - e, xs, starts, lo, inv_side, nx, ny, nz, R2)) / (2.0 * h)
        g[i] = gr - offset * dfac * gev
        if want_grad == 2:
            ev_out[i] = ev
            gev_out[i] = gev


def zbs_field(x: torch.Tensor, bins: DeviceBins, radius: float, offset: float, t_low: float, t_high: float, h: float,
              q: torch.Tensor, grad: bool = True, want_ev: bool = False):
    """(f, g or None, s, EVmax or None, grad EVmax or None) of ZhuBridsonSolenthaler at the points q, on the device."""
    q = q.detach().float().contiguous()
    n = q.shape[0]
    f = torch.empty(n, device=q.device)
    s = torch.empty(n, device=q.device)
    g = torch.zeros(n if grad else 1, 3, device=q.device)
    ev = torch.zeros(n if want_ev else 1, device=q.device)
    gev = torch.zeros(n if want_ev else 1, 3, device=q.device)
    if n:
        lo = bins.lo.tolist()
        stream = {"stream": wp.stream_from_torch(q.device)} if q.is_cuda else {}
        wp.launch(k_zbs_field, dim=n, inputs=[wp.from_torch(q, dtype=wp.vec3), wp.from_torch(bins.xs, dtype=wp.vec3),
                                              wp.from_torch(bins.starts), wp.vec3(*lo), 1.0 / bins.side,
                                              bins.dims[0], bins.dims[1], bins.dims[2], float(radius) ** 2,
                                              float(offset), float(t_low), float(t_high), float(h),
                                              2 if want_ev else int(grad)],
                  outputs=[wp.from_torch(f), wp.from_torch(g, dtype=wp.vec3), wp.from_torch(s), wp.from_torch(ev),
                           wp.from_torch(gev, dtype=wp.vec3)],
                  device=str(q.device), **stream)
    return f, (g if grad or want_ev else None), s, (ev if want_ev else None), (gev if want_ev else None)
