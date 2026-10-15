"""The torque-free layer update (D150 (b), `layer_torque_free`): the outer layer's position update below the grid (the
relaxation and u) also loses its part along three TORQUE modes, n_p x v0_p, besides the six rigid modes (n, r x n) of
kernels.k_layer_project (D98) and solve.free_of_rigid (D94).

A position update below the grid moves x and leaves v, so it changes APIC's angular momentum by exactly sum m dx x v
(the last line of the APIC G2P proof, Jiang et al. 2015: the displacement must be dt v). For the layer's normal
displacement dx_p = s_p n_p this is sum s_p (n_p x v_p). With v0 the window start's velocity, frozen like the start's r
for the rigid modes, a displacement free of the modes n_p x v0_p satisfies sum m dx x v0 = 0 exactly; what is left is
sum m dx x (v_t - v0), first order in the window's change of velocity. Without the switch none of this runs (the frozen
kernels of kernels.py and adjoints.py are unchanged).

Here: the relaxation's step with its moments on the nine modes (k_layer_relax_t), the projection (k_layer_project_t, the
inverse Gram matrix as nine 3x3 blocks in one array, row block i = mode i of (n, r x n, n x v0)), and the projection's
hand adjoint (the moments' share summed in bins, as adjoints.k_layer_project_adj)."""
from __future__ import annotations

import numpy as np
import torch
import warp as wp

_NBINS = 4096
_BSUM = 64


@wp.kernel
def k_layer_relax_t(d: wp.array(dtype=float), mask: wp.array(dtype=float), nrm: wp.array(dtype=wp.vec3),
                    rn: wp.array(dtype=wp.vec3), nv: wp.array(dtype=wp.vec3), nbr: wp.array(dtype=int),
                    w: wp.array(dtype=float), K: int, frac: float, ref: wp.array(dtype=float),
                    s: wp.array(dtype=float), b: wp.array(dtype=wp.vec3)):
    """kernels.k_layer_relax with the third moment b[2] += s (n x v0)."""
    p = wp.tid()
    if mask[p] < 0.5:
        s[p] = 0.0
        return
    dbar = float(0.0)
    for a in range(K):
        dbar = dbar + w[p * K + a] * d[nbr[p * K + a]]
    sp = -frac * (d[p] - dbar - ref[p])
    s[p] = sp
    wp.atomic_add(b, 0, sp * nrm[p])
    wp.atomic_add(b, 1, sp * rn[p])
    wp.atomic_add(b, 2, sp * nv[p])


@wp.kernel
def k_layer_project_t(x_in: wp.array(dtype=wp.vec3), s: wp.array(dtype=float), b: wp.array(dtype=wp.vec3),
                      mask: wp.array(dtype=float), nrm: wp.array(dtype=wp.vec3), rn: wp.array(dtype=wp.vec3),
                      nv: wp.array(dtype=wp.vec3), M: wp.array(dtype=wp.mat33),
                      u: wp.array(dtype=float), frac_u: float, ug: wp.array(dtype=float),
                      x_out: wp.array(dtype=wp.vec3)):
    """kernels.k_layer_project on the nine modes (n, r x n, n x v0): the relaxation's step s loses its part along all
    of them; u's displacement is free of them already (the optimiser's projection)."""
    p = wp.tid()
    if mask[p] < 0.5:
        x_out[p] = x_in[p]
        return
    ct = M[0] @ b[0] + M[1] @ b[1] + M[2] @ b[2]
    cr = M[3] @ b[0] + M[4] @ b[1] + M[5] @ b[2]
    cv = M[6] @ b[0] + M[7] @ b[1] + M[8] @ b[2]
    sp = s[p] - wp.dot(nrm[p], ct) - wp.dot(rn[p], cr) - wp.dot(nv[p], cv)
    x_out[p] = x_in[p] + (frac_u * ug[p] * u[p] + sp) * nrm[p]


@wp.kernel(enable_backward=False)
def k_layer_project_t_adj(mask: wp.array(dtype=float), nrm: wp.array(dtype=wp.vec3), rn: wp.array(dtype=wp.vec3),
                          nv: wp.array(dtype=wp.vec3), frac_u: float, ug: wp.array(dtype=float), has_adj_u: int,
                          adj_x_out: wp.array(dtype=wp.vec3), adj_x_in: wp.array(dtype=wp.vec3),
                          adj_s: wp.array(dtype=float), adj_u: wp.array(dtype=float), bins: wp.array(dtype=wp.vec3)):
    """The adjoint of k_layer_project_t but for the moments b: each layer particle's q = n . adj_x in q n, q (r x n) and
    q (n x v0), summed into one of _NBINS bins (k_layer_project_t_adj_b sums them)."""
    p = wp.tid()
    a = adj_x_out[p]
    adj_x_in[p] = adj_x_in[p] + a
    if mask[p] < 0.5:
        return
    q = wp.dot(a, nrm[p])
    adj_s[p] = adj_s[p] + q
    if has_adj_u != 0:
        adj_u[p] = adj_u[p] + frac_u * ug[p] * q
    k = p % _NBINS
    wp.atomic_add(bins, 3 * k, q * nrm[p])
    wp.atomic_add(bins, 3 * k + 1, q * rn[p])
    wp.atomic_add(bins, 3 * k + 2, q * nv[p])


@wp.kernel(enable_backward=False)
def k_layer_project_t_adj_b(bins: wp.array(dtype=wp.vec3), M: wp.array(dtype=wp.mat33), adj_b: wp.array(dtype=wp.vec3)):
    """The bins' sums S_n, S_r, S_v, _BSUM threads of _NBINS / _BSUM bins each: adj b_j = -sum_i M[3 i + j]^T S_i."""
    t = wp.tid()
    Sn = wp.vec3(0.0, 0.0, 0.0)
    Sr = wp.vec3(0.0, 0.0, 0.0)
    Sv = wp.vec3(0.0, 0.0, 0.0)
    for i in range(_NBINS // _BSUM):
        k = t * (_NBINS // _BSUM) + i
        Sn = Sn + bins[3 * k]
        Sr = Sr + bins[3 * k + 1]
        Sv = Sv + bins[3 * k + 2]
    for j in range(3):
        wp.atomic_add(adj_b, j, -(wp.transpose(M[j]) @ Sn + wp.transpose(M[3 + j]) @ Sr + wp.transpose(M[6 + j]) @ Sv))


def modes(x0: torch.Tensor, lmask, lnrm, nv) -> torch.Tensor:
    """(N, 9) float64: the layer's nine modes n, r x n (r about the starting centre of mass) and n x v0, masked; nv is
    n x v0 masked (torque_modes)."""
    xt = x0.double()
    mt, nt = torch.as_tensor(lmask, device=xt.device).double(), torch.as_tensor(lnrm, device=xt.device).double()
    rn = torch.linalg.cross(xt - xt.mean(0), nt, dim=1) * mt[:, None]
    return torch.cat((nt * mt[:, None], rn, torch.as_tensor(nv, device=xt.device).double()), 1)


def torque_modes(lmask: torch.Tensor, lnrm: torch.Tensor, v0: torch.Tensor | None) -> torch.Tensor:
    """n_p x v0_p on the layer (zero off it, and zero for a start at rest: those modes then drop out of the Gram
    inverse, a pseudo-inverse)."""
    if v0 is None:
        return torch.zeros_like(lnrm)
    return torch.linalg.cross(lnrm, v0.to(lnrm.dtype), dim=1) * lmask[:, None].to(lnrm.dtype)


def gram_blocks(A: torch.Tensor, device) -> wp.array:
    """The pseudo-inverse of A^T A (9 x 9) as nine 3x3 blocks, row-major (block 3 i + j = rows of mode i, columns of
    mode j)."""
    Minv = torch.linalg.pinv(A.T @ A).cpu().numpy()
    blocks = np.stack([Minv[3 * i:3 * i + 3, 3 * j:3 * j + 3] for i in range(3) for j in range(3)]).astype(np.float32)
    return wp.array(blocks, dtype=wp.mat33, device=device)


def record_project(tape, tr, t: int, layer_u, N: int):
    """Record k_layer_project_t's adjoint for step t of Trajectory `tr` on `tape`."""
    from .adjoints import _dummy_float
    dev = tr.device
    xu, s, b, xo = tr.xu[t + 1], tr.ls[t + 1], tr.lb[t + 1], tr._x_out(t)
    if getattr(tr, "_proj_bins_t", None) is None:
        tr._proj_bins_t = wp.zeros(3 * _NBINS, dtype=wp.vec3, device=dev)
    bins = tr._proj_bins_t
    has_u = layer_u.grad is not None
    a_u = layer_u.grad if has_u else _dummy_float(dev)

    def backward():
        bins.zero_()
        wp.launch(k_layer_project_t_adj, dim=N, inputs=[tr.layer_mask, tr.layer_nrm, tr.layer_rn, tr.layer_nv,
                                                        tr.layer_frac_u, tr.layer_ug, int(has_u), xo.grad, xu.grad,
                                                        s.grad, a_u, bins], device=dev)
        wp.launch(k_layer_project_t_adj_b, dim=_BSUM, inputs=[bins, tr.layer_Mt, b.grad], device=dev)

    tape.record_func(backward=backward, arrays=[a for a in (xu, s, b, xo, layer_u) if a.grad is not None])


def u_modes(g: torch.Tensor, n: torch.Tensor, r: torch.Tensor, v0: torch.Tensor | None = None) -> torch.Tensor:
    """The modes u's displacement g u n is made free of (window.solve.free_of_rigid), float64 columns: g n and g (r x n),
    the six rigid modes (D94); with v0 also g (n x v0), the three torque modes (D150 (b))."""
    A = torch.cat((g[:, None] * n, g[:, None] * torch.linalg.cross(r, n, dim=1)), 1).double()
    if v0 is None:
        return A
    return torch.cat((A, (g[:, None] * torch.linalg.cross(n, v0.to(n.dtype), dim=1)).double()), 1)
