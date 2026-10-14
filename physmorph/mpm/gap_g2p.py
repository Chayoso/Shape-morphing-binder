"""The G2P of a split step (D148, `--gap_split`): kernels.k_g2p with each gap node's velocity taken from the particle's
side, hard (sig = 0: side 1's field where the share s is 1, side 0's otherwise) or soft (sig > 0, D148c: s v1 +
(1 - s) v0), and its hand adjoints, per particle and in the trajectory's cell order (adjoints.k_g2p_adj and
k_g2p_adj_warp with the same selection or weights). The shares are gap_kernels.gap_weight's; the particles without a
gap node in their stencil (near = 0) read the total field only, as k_g2p.
"""
from __future__ import annotations

import warp as wp

from .adjoints import _weight_and_grad
from .constitutive import bspline_w, weight
from .gap_kernels import gap_weight, near_gap
from .kernels import WARP_LANES, _warp_sum3, base_node, cell_key, gid, valid_pos


@wp.func
def blend(s: float, sig: float, v0: wp.vec3, v1: wp.vec3) -> wp.vec3:
    """The node's velocity seen by a particle with share s of side 1."""
    if sig > 0.0:
        return s * v1 + (1.0 - s) * v0
    if s > 0.0:
        return v1
    return v0


@wp.kernel
def k_g2p_gap(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
              C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
              dFc: wp.array(dtype=wp.mat33), F_new: wp.array(dtype=wp.mat33),
              grid_v: wp.array(dtype=wp.vec3), grid_v1: wp.array(dtype=wp.vec3),
              q: wp.array(dtype=wp.vec3), plane_m: wp.array(dtype=wp.vec3), plane_n: wp.array(dtype=wp.vec3),
              near: wp.array(dtype=int), sig: float,
              gmin: wp.vec3, dx: float, inv_dx: float, dt: float,
              nx: int, ny: int, nz: int):
    """kernels.k_g2p, each gap node's velocity by the particle's share of side 1."""
    p = wp.tid()
    xp = x[p]
    if not valid_pos(xp):
        return
    qp = q[p]
    vnew = wp.vec3(0.0, 0.0, 0.0)
    Cnew = wp.mat33(0.0)
    C0 = 3.0 * inv_dx * inv_dx
    b = base_node(xp, gmin, inv_dx)
    nr = near_gap(b, near, nx, ny, nz)
    for oi in range(4):
        for oj in range(4):
            for ok in range(4):
                i = b[0] + oi
                j = b[1] + oj
                k = b[2] + ok
                if i >= 0 and i < nx and j >= 0 and j < ny and k >= 0 and k < nz:
                    xg = gmin + wp.vec3(float(i), float(j), float(k)) * dx
                    dgp = xg - xp
                    w = weight(dgp, inv_dx)
                    g = gid(i, j, k, ny, nz)
                    vg = grid_v[g]
                    if nr != 0:
                        s = gap_weight(qp, plane_m[g], plane_n[g], sig)
                        if s > 0.0:
                            vg = blend(s, sig, vg, grid_v1[g])
                    vnew = vnew + w * vg
                    Cnew = Cnew + C0 * w * wp.outer(vg, dgp)
    v[p] = vnew
    C[p] = Cnew
    F_new[p] = (wp.identity(n=3, dtype=float) + dt * Cnew) @ (F[p] + dFc[p])


@wp.kernel(enable_backward=False)
def k_g2p_adj_gap(x: wp.array(dtype=wp.vec3), F: wp.array(dtype=wp.mat33), dFc: wp.array(dtype=wp.mat33),
                  grid_v: wp.array(dtype=wp.vec3), grid_v1: wp.array(dtype=wp.vec3),
                  q: wp.array(dtype=wp.vec3), plane_m: wp.array(dtype=wp.vec3), plane_n: wp.array(dtype=wp.vec3),
                  near: wp.array(dtype=int), sig: float,
                  gmin: wp.vec3, dx: float, inv_dx: float, dt: float, nx: int, ny: int, nz: int,
                  adj_v: wp.array(dtype=wp.vec3), adj_C: wp.array(dtype=wp.mat33), adj_Fnew: wp.array(dtype=wp.mat33),
                  adj_x: wp.array(dtype=wp.vec3), adj_F: wp.array(dtype=wp.mat33), adj_dFc: wp.array(dtype=wp.mat33),
                  has_adj_dFc: int, adj_grid_v: wp.array(dtype=wp.vec3), adj_grid_v1: wp.array(dtype=wp.vec3)):
    """The adjoint of k_g2p_gap (adjoints.k_g2p_adj with each node's velocity by the particle's share)."""
    p = wp.tid()
    xp = x[p]
    if not valid_pos(xp):
        return
    qp = q[p]
    C0 = 3.0 * inv_dx * inv_dx
    b = base_node(xp, gmin, inv_dx)
    nr = near_gap(b, near, nx, ny, nz)
    Cnew = wp.mat33(0.0)
    for oi in range(4):
        for oj in range(4):
            for ok in range(4):
                i = b[0] + oi
                j = b[1] + oj
                k = b[2] + ok
                if i >= 0 and i < nx and j >= 0 and j < ny and k >= 0 and k < nz:
                    dgp = gmin + wp.vec3(float(i), float(j), float(k)) * dx - xp
                    w = bspline_w(dgp[0] * inv_dx) * bspline_w(dgp[1] * inv_dx) * bspline_w(dgp[2] * inv_dx)
                    g = gid(i, j, k, ny, nz)
                    vg = grid_v[g]
                    if nr != 0:
                        s = gap_weight(qp, plane_m[g], plane_n[g], sig)
                        if s > 0.0:
                            vg = blend(s, sig, vg, grid_v1[g])
                    Cnew = Cnew + C0 * w * wp.outer(vg, dgp)
    a_vnew = adj_v[p]
    B = F[p] + dFc[p]
    A = wp.identity(n=3, dtype=float) + dt * Cnew
    a_Fn = adj_Fnew[p]
    a_B = wp.transpose(A) @ a_Fn
    adj_F[p] = adj_F[p] + a_B
    if has_adj_dFc != 0:
        adj_dFc[p] = adj_dFc[p] + a_B
    a_Cnew = adj_C[p] + dt * (a_Fn @ wp.transpose(B))
    a_CnT = wp.transpose(a_Cnew)
    a_xp = wp.vec3(0.0, 0.0, 0.0)
    for oi in range(4):
        for oj in range(4):
            for ok in range(4):
                i = b[0] + oi
                j = b[1] + oj
                k = b[2] + ok
                if i >= 0 and i < nx and j >= 0 and j < ny and k >= 0 and k < nz:
                    dgp = gmin + wp.vec3(float(i), float(j), float(k)) * dx - xp
                    w, dw = _weight_and_grad(dgp, inv_dx)
                    g = gid(i, j, k, ny, nz)
                    s = float(0.0)
                    if nr != 0:
                        s = gap_weight(qp, plane_m[g], plane_n[g], sig)
                    vg = grid_v[g]
                    if s > 0.0:
                        vg = blend(s, sig, vg, grid_v1[g])
                    Ad = a_Cnew @ dgp
                    c = w * a_vnew + (C0 * w) * Ad
                    if s > 0.0:
                        if sig > 0.0:
                            wp.atomic_add(adj_grid_v1, g, s * c)
                            wp.atomic_add(adj_grid_v, g, (1.0 - s) * c)
                        else:
                            wp.atomic_add(adj_grid_v1, g, c)
                    else:
                        wp.atomic_add(adj_grid_v, g, c)
                    a_w = wp.dot(a_vnew, vg) + C0 * wp.dot(vg, Ad)
                    a_xp = a_xp - ((C0 * w) * (a_CnT @ vg) + a_w * dw)
    adj_x[p] = adj_x[p] + a_xp


@wp.kernel(enable_backward=False)
def k_g2p_adj_gap_warp(order: wp.array(dtype=int), n: int,
                       x: wp.array(dtype=wp.vec3), F: wp.array(dtype=wp.mat33), dFc: wp.array(dtype=wp.mat33),
                       grid_v: wp.array(dtype=wp.vec3), grid_v1: wp.array(dtype=wp.vec3), Cn: wp.array(dtype=wp.mat33),
                       q: wp.array(dtype=wp.vec3), plane_m: wp.array(dtype=wp.vec3), plane_n: wp.array(dtype=wp.vec3),
                       near: wp.array(dtype=int), sig: float,
                       gmin: wp.vec3, dx: float, inv_dx: float, dt: float, nx: int, ny: int, nz: int,
                       adj_v: wp.array(dtype=wp.vec3), adj_C: wp.array(dtype=wp.mat33),
                       adj_Fnew: wp.array(dtype=wp.mat33), adj_x: wp.array(dtype=wp.vec3),
                       adj_F: wp.array(dtype=wp.mat33), adj_dFc: wp.array(dtype=wp.mat33), has_adj_dFc: int,
                       adj_grid_v: wp.array(dtype=wp.vec3), adj_grid_v1: wp.array(dtype=wp.vec3)):
    """k_g2p_adj_gap in the trajectory's cell order (adjoints.k_g2p_adj_warp's form): the lanes on lane 0's stencil sum
    their grid adjoint per field before one atomic per node and field; the new affine field read from the forward."""
    blk, lane = wp.tid()
    qi = blk * WARP_LANES + lane
    p = int(0)
    ok = False
    if qi < n:
        p = order[qi]
        ok = valid_pos(x[p])
    C0 = 3.0 * inv_dx * inv_dx
    b = wp.vec3i(0, 0, 0)
    key = int(-1)
    nr = int(0)
    xp = wp.vec3(0.0, 0.0, 0.0)
    qp = wp.vec3(0.0, 0.0, 0.0)
    a_vnew = wp.vec3(0.0, 0.0, 0.0)
    a_Cnew = wp.mat33(0.0)
    if ok:
        xp = x[p]
        qp = q[p]
        b = base_node(xp, gmin, inv_dx)
        nr = near_gap(b, near, nx, ny, nz)
        key = cell_key(b, ny, nz)
        Cnew = Cn[p]
        a_vnew = adj_v[p]
        B = F[p] + dFc[p]
        A = wp.identity(n=3, dtype=float) + dt * Cnew
        a_Fn = adj_Fnew[p]
        a_B = wp.transpose(A) @ a_Fn
        adj_F[p] = adj_F[p] + a_B
        if has_adj_dFc != 0:
            adj_dFc[p] = adj_dFc[p] + a_B
        a_Cnew = adj_C[p] + dt * (a_Fn @ wp.transpose(B))
    k0 = wp.tile_extract(wp.tile(key), 0)
    b0 = wp.vec3i(wp.tile_extract(wp.tile(b[0]), 0), wp.tile_extract(wp.tile(b[1]), 0),
                  wp.tile_extract(wp.tile(b[2]), 0))
    same = ok and key == k0
    a_CnT = wp.transpose(a_Cnew)
    a_xp = wp.vec3(0.0, 0.0, 0.0)
    for oi in range(4):
        for oj in range(4):
            for ok_ in range(4):
                c0 = wp.vec3(0.0, 0.0, 0.0)
                c1 = wp.vec3(0.0, 0.0, 0.0)
                if ok:
                    i = b[0] + oi
                    j = b[1] + oj
                    k = b[2] + ok_
                    if i >= 0 and i < nx and j >= 0 and j < ny and k >= 0 and k < nz:
                        dgp = gmin + wp.vec3(float(i), float(j), float(k)) * dx - xp
                        w, dw = _weight_and_grad(dgp, inv_dx)
                        g = gid(i, j, k, ny, nz)
                        s = float(0.0)
                        if nr != 0:
                            s = gap_weight(qp, plane_m[g], plane_n[g], sig)
                        vg = grid_v[g]
                        if s > 0.0:
                            vg = blend(s, sig, vg, grid_v1[g])
                        Ad = a_Cnew @ dgp
                        c = w * a_vnew + (C0 * w) * Ad
                        a_w = wp.dot(a_vnew, vg) + C0 * wp.dot(vg, Ad)
                        a_xp = a_xp - ((C0 * w) * (a_CnT @ vg) + a_w * dw)
                        if s > 0.0:
                            if sig > 0.0:
                                c1 = s * c
                                c0 = (1.0 - s) * c
                            else:
                                c1 = c
                        else:
                            c0 = c
                        if not same:
                            if s > 0.0:
                                wp.atomic_add(adj_grid_v1, g, c1)
                                if sig > 0.0:
                                    wp.atomic_add(adj_grid_v, g, c0)
                            else:
                                wp.atomic_add(adj_grid_v, g, c0)
                            c0 = wp.vec3(0.0, 0.0, 0.0)
                            c1 = wp.vec3(0.0, 0.0, 0.0)
                sc0 = _warp_sum3(c0)
                sc1 = _warp_sum3(c1)
                if lane == 0 and k0 >= 0:
                    i0 = b0[0] + oi
                    j0 = b0[1] + oj
                    k0_ = b0[2] + ok_
                    if i0 >= 0 and i0 < nx and j0 >= 0 and j0 < ny and k0_ >= 0 and k0_ < nz:
                        g0 = gid(i0, j0, k0_, ny, nz)
                        wp.atomic_add(adj_grid_v, g0, sc0)
                        wp.atomic_add(adj_grid_v1, g0, sc1)
    if ok:
        adj_x[p] = adj_x[p] + a_xp
