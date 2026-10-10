"""Hand-written adjoints (D138, the runtime phase): the same vector-Jacobian products as Warp's generated adjoints of
kernels.k_p2g and kernels.k_g2p (in one pass over each particle's 64 nodes), kernels.k_update and kernels.k_layer_project.

Warp's generated adjoint of a kernel replays the forward body and keeps every node's intermediate values for the
reverse sweep; for the 4^3-node loops of the transfers that cost 2.7x (P2G) and 17x (G2P) the forward kernel at 300k
particles (Warp's kernel timing, docs/experiments.md D138): G2P's adjoint was 57 % of an adjoint sweep. Here the node
loop is run once forward to rebuild the particle's new velocity and affine field (G2P only), then once more taking each
node's contribution to every adjoint directly. The arithmetic is the chain rule of the same expressions, so the
gradients agree with the generated ones to float rounding (tests/test_runtime_exact.py); the grid side accumulates with
atomics in either form, whose order the GPU does not fix.

The forward kernels are unchanged and still launched; on a tape (Trajectory.step) they are launched without being
recorded and these adjoints are recorded in their place (wp.Tape.record_func).
"""
from __future__ import annotations

import warp as wp

from .constitutive import bspline_dw, bspline_w
from .kernels import WARP_LANES, _warp_sum3, base_node, cell_key, gid, valid_pos


@wp.func
def _weight_and_grad(dgp: wp.vec3, inv_dx: float):
    """(w, dw/d dgp) of the tensor-product cubic B-spline."""
    a0 = dgp[0] * inv_dx
    a1 = dgp[1] * inv_dx
    a2 = dgp[2] * inv_dx
    w0 = bspline_w(a0)
    w1 = bspline_w(a1)
    w2 = bspline_w(a2)
    d0 = bspline_dw(a0)
    d1 = bspline_dw(a1)
    d2 = bspline_dw(a2)
    return w0 * w1 * w2, inv_dx * wp.vec3(d0 * w1 * w2, w0 * d1 * w2, w0 * w1 * d2)


@wp.kernel(enable_backward=False)
def k_g2p_adj(x: wp.array(dtype=wp.vec3), F: wp.array(dtype=wp.mat33), dFc: wp.array(dtype=wp.mat33),
              grid_v: wp.array(dtype=wp.vec3),
              gmin: wp.vec3, dx: float, inv_dx: float, dt: float, nx: int, ny: int, nz: int,
              adj_v: wp.array(dtype=wp.vec3), adj_C: wp.array(dtype=wp.mat33), adj_Fnew: wp.array(dtype=wp.mat33),
              adj_x: wp.array(dtype=wp.vec3), adj_F: wp.array(dtype=wp.mat33), adj_dFc: wp.array(dtype=wp.mat33),
              has_adj_dFc: int, adj_grid_v: wp.array(dtype=wp.vec3)):
    """The adjoint of kernels.k_g2p (v, C and F_new of each particle from the grid velocities)."""
    p = wp.tid()
    xp = x[p]
    if not valid_pos(xp):
        return
    C0 = 3.0 * inv_dx * inv_dx
    b = base_node(xp, gmin, inv_dx)
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
                    vg = grid_v[gid(i, j, k, ny, nz)]
                    Cnew = Cnew + C0 * w * wp.outer(vg, dgp)
    a_vnew = adj_v[p]
    # F_new = (I + dt Cnew)(F + dFc)
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
                    vg = grid_v[g]
                    Ad = a_Cnew @ dgp
                    wp.atomic_add(adj_grid_v, g, w * a_vnew + (C0 * w) * Ad)
                    a_w = wp.dot(a_vnew, vg) + C0 * wp.dot(vg, Ad)
                    a_xp = a_xp - ((C0 * w) * (a_CnT @ vg) + a_w * dw)
    adj_x[p] = adj_x[p] + a_xp


@wp.kernel(enable_backward=False)
def k_g2p_adj_warp(order: wp.array(dtype=int), n: int,
                   x: wp.array(dtype=wp.vec3), F: wp.array(dtype=wp.mat33), dFc: wp.array(dtype=wp.mat33),
                   grid_v: wp.array(dtype=wp.vec3), Cn: wp.array(dtype=wp.mat33),
                   gmin: wp.vec3, dx: float, inv_dx: float, dt: float, nx: int, ny: int, nz: int,
                   adj_v: wp.array(dtype=wp.vec3), adj_C: wp.array(dtype=wp.mat33), adj_Fnew: wp.array(dtype=wp.mat33),
                   adj_x: wp.array(dtype=wp.vec3), adj_F: wp.array(dtype=wp.mat33), adj_dFc: wp.array(dtype=wp.mat33),
                   has_adj_dFc: int, adj_grid_v: wp.array(dtype=wp.vec3)):
    """k_g2p_adj over the particles in `order` (kernels.k_p2g_warp's cell order), WARP_LANES threads a block: the
    lanes on lane 0's stencil sum their grid adjoint before one atomic per node; the per-particle arithmetic is
    k_g2p_adj's, but the particle's new affine field is read from the forward (C[t + 1]) instead of being gathered
    from the grid again (the same expression, k_g2p's)."""
    blk, lane = wp.tid()
    q = blk * WARP_LANES + lane
    p = int(0)
    ok = False
    if q < n:
        p = order[q]
        ok = valid_pos(x[p])
    C0 = 3.0 * inv_dx * inv_dx
    b = wp.vec3i(0, 0, 0)
    key = int(-1)
    xp = wp.vec3(0.0, 0.0, 0.0)
    a_vnew = wp.vec3(0.0, 0.0, 0.0)
    a_Cnew = wp.mat33(0.0)
    if ok:
        xp = x[p]
        b = base_node(xp, gmin, inv_dx)
        key = cell_key(b, ny, nz)
        Cnew = Cn[p]                                    # the forward's new affine field, C[t + 1]
        a_vnew = adj_v[p]
        # F_new = (I + dt Cnew)(F + dFc)
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
                c = wp.vec3(0.0, 0.0, 0.0)
                if ok:
                    i = b[0] + oi
                    j = b[1] + oj
                    k = b[2] + ok_
                    if i >= 0 and i < nx and j >= 0 and j < ny and k >= 0 and k < nz:
                        dgp = gmin + wp.vec3(float(i), float(j), float(k)) * dx - xp
                        w, dw = _weight_and_grad(dgp, inv_dx)
                        g = gid(i, j, k, ny, nz)
                        vg = grid_v[g]
                        Ad = a_Cnew @ dgp
                        c = w * a_vnew + (C0 * w) * Ad
                        a_w = wp.dot(a_vnew, vg) + C0 * wp.dot(vg, Ad)
                        a_xp = a_xp - ((C0 * w) * (a_CnT @ vg) + a_w * dw)
                        if not same:
                            wp.atomic_add(adj_grid_v, g, c)
                            c = wp.vec3(0.0, 0.0, 0.0)
                sc = _warp_sum3(c)
                if lane == 0 and k0 >= 0:
                    i0 = b0[0] + oi
                    j0 = b0[1] + oj
                    k0_ = b0[2] + ok_
                    if i0 >= 0 and i0 < nx and j0 >= 0 and j0 < ny and k0_ >= 0 and k0_ < nz:
                        wp.atomic_add(adj_grid_v, gid(i0, j0, k0_, ny, nz), sc)
    if ok:
        adj_x[p] = adj_x[p] + a_xp


@wp.kernel(enable_backward=False)
def k_p2g_adj(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
              C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
              dFc: wp.array(dtype=wp.mat33), P: wp.array(dtype=wp.mat33),
              m: wp.array(dtype=float), vol: wp.array(dtype=float),
              nbr: wp.array(dtype=int), frag: wp.array(dtype=float), bond_K: int,
              gmin: wp.vec3, dx: float, inv_dx: float, dt: float, drag: float, nx: int, ny: int, nz: int,
              adj_gm: wp.array(dtype=float), adj_gv: wp.array(dtype=wp.vec3),
              adj_x: wp.array(dtype=wp.vec3), adj_v: wp.array(dtype=wp.vec3), adj_C: wp.array(dtype=wp.mat33),
              adj_F: wp.array(dtype=wp.mat33), adj_dFc: wp.array(dtype=wp.mat33), has_adj_dFc: int,
              adj_P: wp.array(dtype=wp.mat33)):
    """The adjoint of kernels.k_p2g (mass and momentum of each particle onto its 64 nodes); m, vol, the bonds and the
    fragment flags are constants of the tape."""
    p = wp.tid()
    xp = x[p]
    if not valid_pos(xp):
        return
    Feff = F[p] + dFc[p]
    C0 = 3.0 * inv_dx * inv_dx
    s = -C0 * dt * vol[p]
    G = s * (P[p] @ wp.transpose(Feff)) + m[p] * C[p]
    frag_p = bond_K > 0 and frag[p] > 0.5
    vp = v[p]
    if frag_p:
        vs = wp.vec3(0.0, 0.0, 0.0)
        for a in range(bond_K):
            vs = vs + v[nbr[p * bond_K + a]]
        vp = vs / float(bond_K)
    mp = m[p]
    damp = 1.0 - dt * drag
    mv = mp * vp * damp
    b = base_node(xp, gmin, inv_dx)
    a_mv = wp.vec3(0.0, 0.0, 0.0)
    a_G = wp.mat33(0.0)
    a_xp = wp.vec3(0.0, 0.0, 0.0)
    GT = wp.transpose(G)
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
                    agv = adj_gv[g]
                    a_w = adj_gm[g] * mp + wp.dot(agv, mv + G @ dgp)
                    a_mv = a_mv + w * agv
                    a_G = a_G + w * wp.outer(agv, dgp)
                    a_xp = a_xp - (w * (GT @ agv) + a_w * dw)
    adj_x[p] = adj_x[p] + a_xp
    a_vp = (mp * damp) * a_mv
    if frag_p:
        a_vs = a_vp / float(bond_K)
        for a in range(bond_K):
            wp.atomic_add(adj_v, nbr[p * bond_K + a], a_vs)
    else:
        wp.atomic_add(adj_v, p, a_vp)
    adj_C[p] = adj_C[p] + mp * a_G
    adj_P[p] = adj_P[p] + s * (a_G @ Feff)
    a_Feff = s * (wp.transpose(a_G) @ P[p])
    adj_F[p] = adj_F[p] + a_Feff
    if has_adj_dFc != 0:
        adj_dFc[p] = adj_dFc[p] + a_Feff


_DUMMY: dict = {}


def _dummy_mat(device):
    a = _DUMMY.get(str(device))
    if a is None:
        a = wp.zeros(1, dtype=wp.mat33, device=device)
        _DUMMY[str(device)] = a
    return a


def record_g2p(tape, tr, t: int, dfc, N: int):
    """Record kernels.k_g2p's adjoint for step t of Trajectory `tr` on `tape` (the forward was launched unrecorded)."""
    prm, dev = tr.prm, tr.device
    gmin, inv_dx = wp.vec3(*prm.grid_min), 1.0 / prm.dx
    x, F, gv, v1, C1, Fn = tr.x[t], tr.F[t], tr.gvel[t], tr.v[t + 1], tr.C[t + 1], tr.Fraw[t + 1]
    a_dfc = dfc.grad if dfc.grad is not None else _dummy_mat(dev)

    def backward():
        tail = [gmin, prm.dx, inv_dx, prm.dt, prm.nx, prm.ny, prm.nz,
                v1.grad, C1.grad, Fn.grad, x.grad, F.grad, a_dfc, int(dfc.grad is not None), gv.grad]
        if tr.order is not None:                    # cell order (D138 batch 4, kernels.k_p2g_warp)
            wp.launch_tiled(k_g2p_adj_warp, dim=[tr.order_blocks], inputs=[tr.order, N, x, F, dfc, gv, C1] + tail,
                            block_dim=WARP_LANES, device=dev)
        else:
            wp.launch(k_g2p_adj, dim=N, inputs=[x, F, dfc, gv] + tail, device=dev)

    def multi(G):                                   # three cotangents (k_g2p_adj3)
        has = int(dfc.grad is not None)
        A = [_struct(G2PAdj, v=G(v1)[c], C=G(C1)[c], Fnew=G(Fn)[c], x=G(x)[c], F=G(F)[c],
                     dFc=G(dfc)[c] if has else a_dfc, grid_v=G(gv)[c]) for c in range(3)]
        order = tr.order if tr.order is not None else tr.order_id
        lanes = 1 if str(dev).startswith("cpu") else WARP_LANES
        wp.launch_tiled(k_g2p_adj3, dim=[(N + lanes - 1) // lanes],
                        inputs=[order, N, x, F, dfc, gv, C1, gmin, prm.dx, inv_dx, prm.dt, prm.nx, prm.ny, prm.nz,
                                *A, has], block_dim=lanes, device=dev)

    backward.multi = multi
    backward.arrays3 = [a for a in (x, F, gv, v1, C1, Fn, dfc) if a.grad is not None]
    tape.record_func(backward=backward, arrays=[a for a in (x, F, gv, v1, C1, Fn, dfc) if a.grad is not None])


def record_p2g(tape, tr, t: int, dfc, bnb, bnc, bK: int, N: int):
    """Record kernels.k_p2g's adjoint for step t of Trajectory `tr` on `tape` (the forward was launched unrecorded)."""
    prm, dev = tr.prm, tr.device
    gmin, inv_dx = wp.vec3(*prm.grid_min), 1.0 / prm.dx
    x, v, C, F, P, gm, gmom = tr.x[t], tr.v[t], tr.C[t], tr.F[t], tr.P[t], tr.gm[t], tr.gmom[t]
    a_dfc = dfc.grad if dfc.grad is not None else _dummy_mat(dev)

    def backward():
        wp.launch(k_p2g_adj, dim=N, inputs=[x, v, C, F, dfc, P, tr.m, tr.vol, bnb, bnc, bK,
                                            gmin, prm.dx, inv_dx, prm.dt, prm.drag, prm.nx, prm.ny, prm.nz,
                                            gm.grad, gmom.grad, x.grad, v.grad, C.grad, F.grad, a_dfc,
                                            int(dfc.grad is not None), P.grad], device=dev)

    def multi(G):                                   # three cotangents (k_p2g_adj3)
        has = int(dfc.grad is not None)
        A = [_struct(P2GAdj, gm=G(gm)[c], gv=G(gmom)[c], x=G(x)[c], v=G(v)[c], C=G(C)[c], F=G(F)[c],
                     dFc=G(dfc)[c] if has else a_dfc, P=G(P)[c]) for c in range(3)]
        wp.launch(k_p2g_adj3, dim=N, inputs=[x, v, C, F, dfc, P, tr.m, tr.vol, bnb, bnc, bK,
                                             gmin, prm.dx, inv_dx, prm.dt, prm.drag, prm.nx, prm.ny, prm.nz,
                                             *A, has], device=dev)

    backward.multi = multi
    backward.arrays3 = [a for a in (x, v, C, F, P, gm, gmom, dfc) if a.grad is not None]
    tape.record_func(backward=backward,
                     arrays=[a for a in (x, v, C, F, P, gm, gmom, dfc) if a.grad is not None])


@wp.kernel(enable_backward=False)
def k_update_adj(x_in: wp.array(dtype=wp.vec3), s: float, dt: float,
                 nbr: wp.array(dtype=int), rest: wp.array(dtype=float), frag: wp.array(dtype=float), bond_K: int,
                 bond_frac: float, snbr: wp.array(dtype=int), space_K: int, space_r: wp.array(dtype=float),
                 adj_x_out: wp.array(dtype=wp.vec3), adj_F_out: wp.array(dtype=wp.mat33),
                 adj_x_in: wp.array(dtype=wp.vec3), adj_v: wp.array(dtype=wp.vec3),
                 adj_F_in: wp.array(dtype=wp.mat33), adj_F_new: wp.array(dtype=wp.mat33)):
    """The adjoint of kernels.k_update (the smoothing blend of F, the advection, the bonds' and the minimum spacing's
    position projections); the rest lengths, the neighbour lists and the spacings are constants of the tape."""
    p = wp.tid()
    aF = adj_F_out[p]
    adj_F_new[p] = adj_F_new[p] + (1.0 - s) * aF
    adj_F_in[p] = adj_F_in[p] + s * aF
    ax = adj_x_out[p]
    a_xp = ax
    adj_v[p] = adj_v[p] + dt * ax
    I = wp.identity(n=3, dtype=float)
    if bond_K > 0 and frag[p] > 0.5:
        c = bond_frac / float(bond_K)
        for a in range(bond_K):
            j = nbr[p * bond_K + a]
            d = x_in[j] - x_in[p]
            L = wp.length(d)
            r = rest[p * bond_K + a]
            if L > r and L > 1.0e-9:
                # (L - r) d / L = d - r d / L: its Jacobian in d is (1 - r / L) I + r d d^T / L^3 (symmetric)
                Jd = (1.0 - r / L) * I + (r / (L * L * L)) * wp.outer(d, d)
                a_d = c * (Jd @ ax)
                wp.atomic_add(adj_x_in, j, a_d)
                a_xp = a_xp - a_d
    if space_K > 0:
        c2 = bond_frac * 0.5
        for a in range(space_K):
            q = snbr[p * space_K + a]
            d = x_in[p] - x_in[q]
            L = wp.length(d)
            r = 0.5 * (space_r[p] + space_r[q])
            if L < r and L > 1.0e-9:
                # (r - L) d / L = r d / L - d: its Jacobian in d is (r / L - 1) I - r d d^T / L^3 (symmetric)
                Jd = (r / L - 1.0) * I - (r / (L * L * L)) * wp.outer(d, d)
                a_d = c2 * (Jd @ ax)
                wp.atomic_add(adj_x_in, q, -a_d)
                a_xp = a_xp + a_d
    wp.atomic_add(adj_x_in, p, a_xp)


_NBINS = 4096
_BSUM = 64                                         # threads that sum the bins (k_layer_project_adj_b)


@wp.kernel(enable_backward=False)
def k_layer_project_adj(mask: wp.array(dtype=float), nrm: wp.array(dtype=wp.vec3), rn: wp.array(dtype=wp.vec3),
                        frac_u: float, ug: wp.array(dtype=float), has_adj_u: int,
                        adj_x_out: wp.array(dtype=wp.vec3), adj_x_in: wp.array(dtype=wp.vec3),
                        adj_s: wp.array(dtype=float), adj_u: wp.array(dtype=float), bins: wp.array(dtype=wp.vec3)):
    """The adjoint of kernels.k_layer_project but for the rigid-mode moments b: each layer particle's share of their
    adjoint (q n and q (r x n), q = n . adj_x) is summed into one of _NBINS bins (k_layer_project_adj_b sums the bins);
    the generated adjoint added every particle's share into the same two vectors."""
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
    wp.atomic_add(bins, 2 * k, q * nrm[p])
    wp.atomic_add(bins, 2 * k + 1, q * rn[p])


@wp.kernel(enable_backward=False)
def k_layer_project_adj_b(bins: wp.array(dtype=wp.vec3), M11: wp.mat33, M12: wp.mat33, M21: wp.mat33, M22: wp.mat33,
                          adj_b: wp.array(dtype=wp.vec3)):
    """The bins' sums, _BSUM threads of _NBINS / _BSUM bins each: b's adjoint, -(M11^T Sn + M21^T Sr) and
    -(M12^T Sn + M22^T Sr), each thread's part added (M is linear)."""
    t = wp.tid()
    Sn = wp.vec3(0.0, 0.0, 0.0)
    Sr = wp.vec3(0.0, 0.0, 0.0)
    for i in range(_NBINS // _BSUM):
        k = t * (_NBINS // _BSUM) + i
        Sn = Sn + bins[2 * k]
        Sr = Sr + bins[2 * k + 1]
    wp.atomic_add(adj_b, 0, -(wp.transpose(M11) @ Sn + wp.transpose(M21) @ Sr))
    wp.atomic_add(adj_b, 1, -(wp.transpose(M12) @ Sn + wp.transpose(M22) @ Sr))


_DUMMY_F: dict = {}


def _dummy_float(device):
    a = _DUMMY_F.get(str(device))
    if a is None:
        a = wp.zeros(1, dtype=float, device=device)
        _DUMMY_F[str(device)] = a
    return a


def record_update(tape, tr, t: int, x_next, F_next, bnb, brest, bnc, bK: int, snbr, N: int):
    """Record kernels.k_update's adjoint for step t of Trajectory `tr` on `tape`."""
    prm, dev = tr.prm, tr.device
    x, v1, F, Fn = tr.x[t], tr.v[t + 1], tr.F[t], tr.Fraw[t + 1]
    frac = 1.0 / float(tr.control_steps)

    def backward():
        wp.launch(k_update_adj, dim=N, inputs=[x, prm.smoothing, prm.dt, bnb, brest, bnc, bK, frac,
                                               snbr, tr.space_K, tr.space_r, x_next.grad, F_next.grad,
                                               x.grad, v1.grad, F.grad, Fn.grad], device=dev)

    def multi(G):                                   # three cotangents (k_update_adj3)
        A = [_struct(UpdAdj, x_out=G(x_next)[c], F_out=G(F_next)[c], x_in=G(x)[c], v=G(v1)[c], F_in=G(F)[c],
                     F_new=G(Fn)[c]) for c in range(3)]
        wp.launch(k_update_adj3, dim=N, inputs=[x, prm.smoothing, prm.dt, bnb, brest, bnc, bK, frac,
                                                snbr, tr.space_K, tr.space_r, *A], device=dev)

    backward.multi = multi
    backward.arrays3 = [x, x_next, v1, F, Fn, F_next]
    tape.record_func(backward=backward, arrays=[x, x_next, v1, F, Fn, F_next])


def record_layer_project(tape, tr, t: int, layer_u, N: int):
    """Record kernels.k_layer_project's adjoint for step t of Trajectory `tr` on `tape`."""
    dev = tr.device
    xu, s, b, xo = tr.xu[t + 1], tr.ls[t + 1], tr.lb[t + 1], tr.x[t + 1]
    if getattr(tr, "_proj_bins", None) is None:
        tr._proj_bins = wp.zeros(2 * _NBINS, dtype=wp.vec3, device=dev)
    bins = tr._proj_bins
    has_u = layer_u.grad is not None
    a_u = layer_u.grad if has_u else _dummy_float(dev)

    def backward():
        bins.zero_()
        wp.launch(k_layer_project_adj, dim=N, inputs=[tr.layer_mask, tr.layer_nrm, tr.layer_rn, tr.layer_frac_u,
                                                      tr.layer_ug, int(has_u), xo.grad, xu.grad, s.grad, a_u, bins],
                  device=dev)
        wp.launch(k_layer_project_adj_b, dim=_BSUM, inputs=[bins, *tr.layer_M, b.grad], device=dev)

    def multi(G):                                   # three cotangents: the one-channel kernels, channel by channel
        for c in range(3):
            bins.zero_()
            wp.launch(k_layer_project_adj, dim=N, inputs=[tr.layer_mask, tr.layer_nrm, tr.layer_rn, tr.layer_frac_u,
                                                          tr.layer_ug, int(has_u), G(xo)[c], G(xu)[c], G(s)[c],
                                                          G(layer_u)[c] if has_u else a_u, bins], device=dev)
            wp.launch(k_layer_project_adj_b, dim=_BSUM, inputs=[bins, *tr.layer_M, G(b)[c]], device=dev)

    backward.multi = multi
    backward.arrays3 = [a for a in (xu, s, b, xo, layer_u) if a.grad is not None]
    tape.record_func(backward=backward, arrays=[a for a in (xu, s, b, xo, layer_u) if a.grad is not None])


# ── three cotangents in one sweep (D138 batch 6) ─────────────────────────────────────────────────────────────────
# A window's gradient needs the control gradients of three objectives at the same rollout (physics, cleanup, render:
# PCGrad and the render weight read them apart). The tape is linear in its seeds, so one sweep can carry the three
# cotangents: every generated adjoint is launched once per channel, and the transfers' and the update's hand-written
# adjoints below run the three channels in one pass, sharing the particle's stencil, weights and neighbour geometry.
# Each channel's arithmetic is the one-channel kernel's, expression for expression (bit for bit on the CPU, where the
# atomics keep the particle order); function.PersistentAdjoint.vjp3 walks the tape. A record's `multi` attribute is
# its three-channel backward, called with G(a) -> [a.grad, channel 1's, channel 2's].


@wp.struct
class G2PAdj:
    v: wp.array(dtype=wp.vec3)          # v[t + 1]
    C: wp.array(dtype=wp.mat33)         # C[t + 1]
    Fnew: wp.array(dtype=wp.mat33)      # Fraw[t + 1]
    x: wp.array(dtype=wp.vec3)
    F: wp.array(dtype=wp.mat33)
    dFc: wp.array(dtype=wp.mat33)
    grid_v: wp.array(dtype=wp.vec3)


@wp.func
def _g2p_adj_head(A: G2PAdj, p: int, B: wp.mat33, Amat: wp.mat33, dt: float, has_adj_dFc: int):
    """One channel of k_g2p_adj's particle part: writes the F and dFc adjoints, returns (a_vnew, a_Cnew)."""
    a_vnew = A.v[p]
    a_Fn = A.Fnew[p]
    a_B = wp.transpose(Amat) @ a_Fn
    A.F[p] = A.F[p] + a_B
    if has_adj_dFc != 0:
        A.dFc[p] = A.dFc[p] + a_B
    a_Cnew = A.C[p] + dt * (a_Fn @ wp.transpose(B))
    return a_vnew, a_Cnew


@wp.func
def _g2p_adj_node(w: float, dw: wp.vec3, dgp: wp.vec3, vg: wp.vec3, C0: float, a_vnew: wp.vec3, a_Cnew: wp.mat33,
                  a_CnT: wp.mat33, a_xp: wp.vec3):
    """One channel of k_g2p_adj's node: (the node's grid-velocity adjoint, the updated position adjoint)."""
    Ad = a_Cnew @ dgp
    c = w * a_vnew + (C0 * w) * Ad
    a_w = wp.dot(a_vnew, vg) + C0 * wp.dot(vg, Ad)
    return c, a_xp - ((C0 * w) * (a_CnT @ vg) + a_w * dw)


@wp.kernel(enable_backward=False)
def k_g2p_adj3(order: wp.array(dtype=int), n: int,
               x: wp.array(dtype=wp.vec3), F: wp.array(dtype=wp.mat33), dFc: wp.array(dtype=wp.mat33),
               grid_v: wp.array(dtype=wp.vec3), Cn: wp.array(dtype=wp.mat33),
               gmin: wp.vec3, dx: float, inv_dx: float, dt: float, nx: int, ny: int, nz: int,
               A0: G2PAdj, A1: G2PAdj, A2: G2PAdj, has_adj_dFc: int):
    """k_g2p_adj_warp for three cotangents, launched tiled (one lane a block on the CPU, WARP_LANES on CUDA)."""
    blk, lane = wp.tid()
    q = blk * wp.block_dim() + lane
    p = int(0)
    ok = False
    if q < n:
        p = order[q]
        ok = valid_pos(x[p])
    C0 = 3.0 * inv_dx * inv_dx
    b = wp.vec3i(0, 0, 0)
    key = int(-1)
    xp = wp.vec3(0.0, 0.0, 0.0)
    v0 = wp.vec3(0.0, 0.0, 0.0)
    v1 = wp.vec3(0.0, 0.0, 0.0)
    v2 = wp.vec3(0.0, 0.0, 0.0)
    c0 = wp.mat33(0.0)
    c1 = wp.mat33(0.0)
    c2 = wp.mat33(0.0)
    if ok:
        xp = x[p]
        b = base_node(xp, gmin, inv_dx)
        key = cell_key(b, ny, nz)
        Cnew = Cn[p]                                    # the forward's new affine field, C[t + 1]
        B = F[p] + dFc[p]
        Amat = wp.identity(n=3, dtype=float) + dt * Cnew
        v0, c0 = _g2p_adj_head(A0, p, B, Amat, dt, has_adj_dFc)
        v1, c1 = _g2p_adj_head(A1, p, B, Amat, dt, has_adj_dFc)
        v2, c2 = _g2p_adj_head(A2, p, B, Amat, dt, has_adj_dFc)
    k0 = wp.tile_extract(wp.tile(key), 0)
    b0 = wp.vec3i(wp.tile_extract(wp.tile(b[0]), 0), wp.tile_extract(wp.tile(b[1]), 0),
                  wp.tile_extract(wp.tile(b[2]), 0))
    same = ok and key == k0
    t0 = wp.transpose(c0)
    t1 = wp.transpose(c1)
    t2 = wp.transpose(c2)
    x0 = wp.vec3(0.0, 0.0, 0.0)
    x1 = wp.vec3(0.0, 0.0, 0.0)
    x2 = wp.vec3(0.0, 0.0, 0.0)
    for oi in range(4):
        for oj in range(4):
            for ok_ in range(4):
                g0 = wp.vec3(0.0, 0.0, 0.0)
                g1 = wp.vec3(0.0, 0.0, 0.0)
                g2 = wp.vec3(0.0, 0.0, 0.0)
                if ok:
                    i = b[0] + oi
                    j = b[1] + oj
                    k = b[2] + ok_
                    if i >= 0 and i < nx and j >= 0 and j < ny and k >= 0 and k < nz:
                        dgp = gmin + wp.vec3(float(i), float(j), float(k)) * dx - xp
                        w, dw = _weight_and_grad(dgp, inv_dx)
                        g = gid(i, j, k, ny, nz)
                        vg = grid_v[g]
                        g0, x0 = _g2p_adj_node(w, dw, dgp, vg, C0, v0, c0, t0, x0)
                        g1, x1 = _g2p_adj_node(w, dw, dgp, vg, C0, v1, c1, t1, x1)
                        g2, x2 = _g2p_adj_node(w, dw, dgp, vg, C0, v2, c2, t2, x2)
                        if not same:
                            wp.atomic_add(A0.grid_v, g, g0)
                            wp.atomic_add(A1.grid_v, g, g1)
                            wp.atomic_add(A2.grid_v, g, g2)
                            g0 = wp.vec3(0.0, 0.0, 0.0)
                            g1 = wp.vec3(0.0, 0.0, 0.0)
                            g2 = wp.vec3(0.0, 0.0, 0.0)
                s0 = _warp_sum3(g0)
                s1 = _warp_sum3(g1)
                s2 = _warp_sum3(g2)
                if lane == 0 and k0 >= 0:
                    i0 = b0[0] + oi
                    j0 = b0[1] + oj
                    k0_ = b0[2] + ok_
                    if i0 >= 0 and i0 < nx and j0 >= 0 and j0 < ny and k0_ >= 0 and k0_ < nz:
                        gg = gid(i0, j0, k0_, ny, nz)
                        wp.atomic_add(A0.grid_v, gg, s0)
                        wp.atomic_add(A1.grid_v, gg, s1)
                        wp.atomic_add(A2.grid_v, gg, s2)
    if ok:
        A0.x[p] = A0.x[p] + x0
        A1.x[p] = A1.x[p] + x1
        A2.x[p] = A2.x[p] + x2


@wp.struct
class P2GAdj:
    gm: wp.array(dtype=float)
    gv: wp.array(dtype=wp.vec3)
    x: wp.array(dtype=wp.vec3)
    v: wp.array(dtype=wp.vec3)
    C: wp.array(dtype=wp.mat33)
    F: wp.array(dtype=wp.mat33)
    dFc: wp.array(dtype=wp.mat33)
    P: wp.array(dtype=wp.mat33)


@wp.func
def _p2g_adj_node(A: P2GAdj, g: int, w: float, dw: wp.vec3, dgp: wp.vec3, mp: float, mv: wp.vec3, G: wp.mat33,
                  GT: wp.mat33, a_mv: wp.vec3, a_G: wp.mat33, a_xp: wp.vec3):
    """One channel of k_p2g_adj's node: the updated (a_mv, a_G, a_xp)."""
    agv = A.gv[g]
    a_w = A.gm[g] * mp + wp.dot(agv, mv + G @ dgp)
    return a_mv + w * agv, a_G + w * wp.outer(agv, dgp), a_xp - (w * (GT @ agv) + a_w * dw)


@wp.func
def _p2g_adj_tail(A: P2GAdj, p: int, a_xp: wp.vec3, a_mv: wp.vec3, a_G: wp.mat33, mp: float, damp: float,
                  frag_p: bool, nbr: wp.array(dtype=int), bond_K: int, s: float, Feff: wp.mat33, Pp: wp.mat33,
                  has_adj_dFc: int):
    """One channel of k_p2g_adj's particle part (after the node loop)."""
    A.x[p] = A.x[p] + a_xp
    a_vp = (mp * damp) * a_mv
    if frag_p:
        a_vs = a_vp / float(bond_K)
        for a in range(bond_K):
            wp.atomic_add(A.v, nbr[p * bond_K + a], a_vs)
    else:
        wp.atomic_add(A.v, p, a_vp)
    A.C[p] = A.C[p] + mp * a_G
    A.P[p] = A.P[p] + s * (a_G @ Feff)
    a_Feff = s * (wp.transpose(a_G) @ Pp)
    A.F[p] = A.F[p] + a_Feff
    if has_adj_dFc != 0:
        A.dFc[p] = A.dFc[p] + a_Feff


@wp.kernel(enable_backward=False)
def k_p2g_adj3(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
               C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
               dFc: wp.array(dtype=wp.mat33), P: wp.array(dtype=wp.mat33),
               m: wp.array(dtype=float), vol: wp.array(dtype=float),
               nbr: wp.array(dtype=int), frag: wp.array(dtype=float), bond_K: int,
               gmin: wp.vec3, dx: float, inv_dx: float, dt: float, drag: float, nx: int, ny: int, nz: int,
               A0: P2GAdj, A1: P2GAdj, A2: P2GAdj, has_adj_dFc: int):
    """k_p2g_adj for three cotangents."""
    p = wp.tid()
    xp = x[p]
    if not valid_pos(xp):
        return
    Feff = F[p] + dFc[p]
    C0 = 3.0 * inv_dx * inv_dx
    s = -C0 * dt * vol[p]
    G = s * (P[p] @ wp.transpose(Feff)) + m[p] * C[p]
    frag_p = bond_K > 0 and frag[p] > 0.5
    vp = v[p]
    if frag_p:
        vs = wp.vec3(0.0, 0.0, 0.0)
        for a in range(bond_K):
            vs = vs + v[nbr[p * bond_K + a]]
        vp = vs / float(bond_K)
    mp = m[p]
    damp = 1.0 - dt * drag
    mv = mp * vp * damp
    b = base_node(xp, gmin, inv_dx)
    m0 = wp.vec3(0.0, 0.0, 0.0)
    m1 = wp.vec3(0.0, 0.0, 0.0)
    m2 = wp.vec3(0.0, 0.0, 0.0)
    G0 = wp.mat33(0.0)
    G1 = wp.mat33(0.0)
    G2 = wp.mat33(0.0)
    x0 = wp.vec3(0.0, 0.0, 0.0)
    x1 = wp.vec3(0.0, 0.0, 0.0)
    x2 = wp.vec3(0.0, 0.0, 0.0)
    GT = wp.transpose(G)
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
                    m0, G0, x0 = _p2g_adj_node(A0, g, w, dw, dgp, mp, mv, G, GT, m0, G0, x0)
                    m1, G1, x1 = _p2g_adj_node(A1, g, w, dw, dgp, mp, mv, G, GT, m1, G1, x1)
                    m2, G2, x2 = _p2g_adj_node(A2, g, w, dw, dgp, mp, mv, G, GT, m2, G2, x2)
    Pp = P[p]
    _p2g_adj_tail(A0, p, x0, m0, G0, mp, damp, frag_p, nbr, bond_K, s, Feff, Pp, has_adj_dFc)
    _p2g_adj_tail(A1, p, x1, m1, G1, mp, damp, frag_p, nbr, bond_K, s, Feff, Pp, has_adj_dFc)
    _p2g_adj_tail(A2, p, x2, m2, G2, mp, damp, frag_p, nbr, bond_K, s, Feff, Pp, has_adj_dFc)


@wp.struct
class UpdAdj:
    x_out: wp.array(dtype=wp.vec3)
    F_out: wp.array(dtype=wp.mat33)
    x_in: wp.array(dtype=wp.vec3)
    v: wp.array(dtype=wp.vec3)
    F_in: wp.array(dtype=wp.mat33)
    F_new: wp.array(dtype=wp.mat33)


@wp.func
def _update_adj_head(A: UpdAdj, p: int, s: float, dt: float):
    """One channel of k_update_adj's particle part: writes F_new, F_in and v, returns the x_out adjoint."""
    aF = A.F_out[p]
    A.F_new[p] = A.F_new[p] + (1.0 - s) * aF
    A.F_in[p] = A.F_in[p] + s * aF
    ax = A.x_out[p]
    A.v[p] = A.v[p] + dt * ax
    return ax


@wp.kernel(enable_backward=False)
def k_update_adj3(x_in: wp.array(dtype=wp.vec3), s: float, dt: float,
                  nbr: wp.array(dtype=int), rest: wp.array(dtype=float), frag: wp.array(dtype=float), bond_K: int,
                  bond_frac: float, snbr: wp.array(dtype=int), space_K: int, space_r: wp.array(dtype=float),
                  A0: UpdAdj, A1: UpdAdj, A2: UpdAdj):
    """k_update_adj for three cotangents: the bonds' and the spacing's Jacobians found once, applied to each."""
    p = wp.tid()
    ax0 = _update_adj_head(A0, p, s, dt)
    ax1 = _update_adj_head(A1, p, s, dt)
    ax2 = _update_adj_head(A2, p, s, dt)
    a0 = ax0
    a1 = ax1
    a2 = ax2
    I = wp.identity(n=3, dtype=float)
    if bond_K > 0 and frag[p] > 0.5:
        c = bond_frac / float(bond_K)
        for a in range(bond_K):
            j = nbr[p * bond_K + a]
            d = x_in[j] - x_in[p]
            L = wp.length(d)
            r = rest[p * bond_K + a]
            if L > r and L > 1.0e-9:
                Jd = (1.0 - r / L) * I + (r / (L * L * L)) * wp.outer(d, d)
                d0 = c * (Jd @ ax0)
                d1 = c * (Jd @ ax1)
                d2 = c * (Jd @ ax2)
                wp.atomic_add(A0.x_in, j, d0)
                wp.atomic_add(A1.x_in, j, d1)
                wp.atomic_add(A2.x_in, j, d2)
                a0 = a0 - d0
                a1 = a1 - d1
                a2 = a2 - d2
    if space_K > 0:
        c2 = bond_frac * 0.5
        for a in range(space_K):
            q = snbr[p * space_K + a]
            d = x_in[p] - x_in[q]
            L = wp.length(d)
            r = 0.5 * (space_r[p] + space_r[q])
            if L < r and L > 1.0e-9:
                Jd = (r / L - 1.0) * I - (r / (L * L * L)) * wp.outer(d, d)
                d0 = c2 * (Jd @ ax0)
                d1 = c2 * (Jd @ ax1)
                d2 = c2 * (Jd @ ax2)
                wp.atomic_add(A0.x_in, q, -d0)
                wp.atomic_add(A1.x_in, q, -d1)
                wp.atomic_add(A2.x_in, q, -d2)
                a0 = a0 + d0
                a1 = a1 + d1
                a2 = a2 + d2
    wp.atomic_add(A0.x_in, p, a0)
    wp.atomic_add(A1.x_in, p, a1)
    wp.atomic_add(A2.x_in, p, a2)


def _struct(cls, **arrays):
    s = cls()
    for k, a in arrays.items():
        setattr(s, k, a)
    return s
