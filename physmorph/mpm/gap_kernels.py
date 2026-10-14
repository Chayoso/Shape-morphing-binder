"""The two sides of a target gap on their own grid fields (D148, `--gap_split`): the kernels of a split step.

The frozen P2G still deposits every particle into the total field (mass m, momentum p). At a gap node (gap_split.py:
a node outside the target whose kernel box holds two pieces of it; plane point m_g, unit normal n_g, n_g = 0 elsewhere)
a particle's side is the side of the plane its window-start transport image q_p lies on (q_p frozen for the window).
k_p2g_side1 deposits each particle's side-1 part at gap nodes into a second field (m1, p1); side 0 is the total minus
side 1. k_grid_op_gap: at a node with both sides (and, in variant B, a particle volume below one cell) the sides'
post-force velocities v0 = (p - p1) / (m - m1) + dt f, v1 = p1 / m1 + dt f; approaching, (v0 - v1) . n_g > 0, both take
the total field's velocity p / m + dt f (the single field's, so a pressed interface carries what the bulk carries);
separating, each keeps its own (free separation, Bardenhagen et al. 2000). Every other node is kernels.k_grid_op's
arithmetic on the total field. k_g2p_gap gathers a particle's velocity and affine field from its side's field at gap
nodes and from the total elsewhere (kernels.k_g2p's arithmetic); k_g2p_adj_gap is its hand adjoint (adjoints.k_g2p_adj
with the same selection). The sides and the planes are constants of the window's tape.
"""
from __future__ import annotations

import warp as wp

from .adjoints import _weight_and_grad
from .constitutive import bspline_w, weight
from .kernels import base_node, bond_velocity, gid, valid_pos


@wp.func
def gap_side(q: wp.vec3, pm: wp.vec3, pn: wp.vec3) -> int:
    """1 when the node is a gap node and q lies on its plane's positive side, else 0."""
    if pn[0] == 0.0 and pn[1] == 0.0 and pn[2] == 0.0:
        return 0
    if wp.dot(q - pm, pn) > 0.0:
        return 1
    return 0


@wp.kernel
def k_p2g_side1(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
                dFc: wp.array(dtype=wp.mat33), P: wp.array(dtype=wp.mat33),
                m: wp.array(dtype=float), vol: wp.array(dtype=float),
                nbr: wp.array(dtype=int), frag: wp.array(dtype=float), bond_K: int,
                q: wp.array(dtype=wp.vec3), plane_m: wp.array(dtype=wp.vec3), plane_n: wp.array(dtype=wp.vec3),
                grid_m1: wp.array(dtype=float), grid_v1: wp.array(dtype=wp.vec3),
                gmin: wp.vec3, dx: float, inv_dx: float, dt: float, drag: float,
                nx: int, ny: int, nz: int):
    """kernels.k_p2g's deposit, at the gap nodes where the particle is on side 1 only."""
    p = wp.tid()
    xp = x[p]
    if not valid_pos(xp):
        return
    b = base_node(xp, gmin, inv_dx)
    qp = q[p]
    any1 = int(0)
    for oi in range(4):
        for oj in range(4):
            for ok in range(4):
                i = b[0] + oi
                j = b[1] + oj
                k = b[2] + ok
                if i >= 0 and i < nx and j >= 0 and j < ny and k >= 0 and k < nz:
                    g = gid(i, j, k, ny, nz)
                    any1 = any1 + gap_side(qp, plane_m[g], plane_n[g])
    if any1 == 0:
        return
    Feff = F[p] + dFc[p]
    C0 = 3.0 * inv_dx * inv_dx
    G = -C0 * dt * vol[p] * (P[p] @ wp.transpose(Feff)) + m[p] * C[p]   # total-PK1 form (k_p2g's)
    vp = v[p]
    if bond_K > 0 and frag[p] > 0.5:
        vp = bond_velocity(v, nbr, p, bond_K)
    mv = m[p] * vp * (1.0 - dt * drag)
    for oi in range(4):
        for oj in range(4):
            for ok in range(4):
                i = b[0] + oi
                j = b[1] + oj
                k = b[2] + ok
                if i >= 0 and i < nx and j >= 0 and j < ny and k >= 0 and k < nz:
                    g = gid(i, j, k, ny, nz)
                    if gap_side(qp, plane_m[g], plane_n[g]) == 1:
                        xg = gmin + wp.vec3(float(i), float(j), float(k)) * dx
                        dgp = xg - xp
                        w = weight(dgp, inv_dx)
                        wp.atomic_add(grid_m1, g, w * m[p])
                        wp.atomic_add(grid_v1, g, w * (mv + G @ dgp))


@wp.func
def _walls_floor(vg: wp.vec3, i: int, j: int, k: int, nx: int, ny: int, nz: int, wall_nodes: int, node_y: float,
                 floor_y: float, friction: float):
    """kernels.k_grid_op's domain walls and floor, the same arithmetic in the same order."""
    out = vg
    if wall_nodes > 0:
        vx = out[0]
        vy = out[1]
        vz = out[2]
        if i < wall_nodes and vx < 0.0:
            vx = 0.0
        if i >= nx - wall_nodes and vx > 0.0:
            vx = 0.0
        if j < wall_nodes and vy < 0.0:
            vy = 0.0
        if j >= ny - wall_nodes and vy > 0.0:
            vy = 0.0
        if k < wall_nodes and vz < 0.0:
            vz = 0.0
        if k >= nz - wall_nodes and vz > 0.0:
            vz = 0.0
        out = wp.vec3(vx, vy, vz)
    if node_y < floor_y and out[1] < 0.0:
        vt = wp.vec3(out[0], 0.0, out[2])
        tl = wp.length(vt)
        vn = -out[1]
        if tl > 1.0e-8:
            vt = vt * wp.max(0.0, 1.0 - friction * vn / tl)
        out = wp.vec3(vt[0], 0.0, vt[2])
    return out


@wp.kernel
def k_grid_op_gap(grid_m: wp.array(dtype=float), grid_mom: wp.array(dtype=wp.vec3),
                  grid_m1: wp.array(dtype=float), grid_mom1: wp.array(dtype=wp.vec3),
                  plane_n: wp.array(dtype=wp.vec3), gvol: wp.array(dtype=float), use_vol: int,
                  grid_vel: wp.array(dtype=wp.vec3), grid_vel1: wp.array(dtype=wp.vec3),
                  dt: float, f_ext: wp.vec3, gmin_y: float, dx: float, nx: int, ny: int, nz: int,
                  floor_y: float, friction: float, wall_nodes: int, stats: wp.array(dtype=float)):
    """The grid operation of a split step. stats (a record, off the tape): [0] the mass at split nodes, [1] the mass
    at split nodes in the separating branch, [2] the split nodes, [3] the separating ones."""
    g = wp.tid()
    mg = grid_m[g]
    if mg > 1.0e-12:
        vg = grid_mom[g] / mg + dt * f_ext
        v0 = vg
        v1 = vg
        pn = plane_n[g]
        m1 = grid_m1[g]
        if (pn[0] != 0.0 or pn[1] != 0.0 or pn[2] != 0.0) and m1 > 1.0e-12:
            m0 = mg - m1
            split = m0 > 1.0e-12
            if split and use_vol != 0:
                split = gvol[g] < dx * dx * dx
            if split:
                a1 = grid_mom1[g] / m1 + dt * f_ext
                a0 = (grid_mom[g] - grid_mom1[g]) / m0 + dt * f_ext
                wp.atomic_add(stats, 0, mg)
                wp.atomic_add(stats, 2, 1.0)
                if wp.dot(a0 - a1, pn) <= 0.0:           # separating: each side its own velocity
                    v0 = a0
                    v1 = a1
                    wp.atomic_add(stats, 1, mg)
                    wp.atomic_add(stats, 3, 1.0)
        i = g // (ny * nz)
        j = (g // nz) % ny
        k = g % nz
        node_y = gmin_y + float(j) * dx
        grid_vel[g] = _walls_floor(v0, i, j, k, nx, ny, nz, wall_nodes, node_y, floor_y, friction)
        grid_vel1[g] = _walls_floor(v1, i, j, k, nx, ny, nz, wall_nodes, node_y, floor_y, friction)
    else:
        grid_vel[g] = wp.vec3(0.0, 0.0, 0.0)
        grid_vel1[g] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def k_g2p_gap(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
              C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
              dFc: wp.array(dtype=wp.mat33), F_new: wp.array(dtype=wp.mat33),
              grid_v: wp.array(dtype=wp.vec3), grid_v1: wp.array(dtype=wp.vec3),
              q: wp.array(dtype=wp.vec3), plane_m: wp.array(dtype=wp.vec3), plane_n: wp.array(dtype=wp.vec3),
              gmin: wp.vec3, dx: float, inv_dx: float, dt: float,
              nx: int, ny: int, nz: int):
    """kernels.k_g2p, each node's velocity from the particle's side's field."""
    p = wp.tid()
    xp = x[p]
    if not valid_pos(xp):
        return
    qp = q[p]
    vnew = wp.vec3(0.0, 0.0, 0.0)
    Cnew = wp.mat33(0.0)
    C0 = 3.0 * inv_dx * inv_dx
    b = base_node(xp, gmin, inv_dx)
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
                    if gap_side(qp, plane_m[g], plane_n[g]) == 1:
                        vg = grid_v1[g]
                    vnew = vnew + w * vg
                    Cnew = Cnew + C0 * w * wp.outer(vg, dgp)
    v[p] = vnew
    C[p] = Cnew
    F_new[p] = (wp.identity(n=3, dtype=float) + dt * Cnew) @ (F[p] + dFc[p])


@wp.kernel(enable_backward=False)
def k_g2p_adj_gap(x: wp.array(dtype=wp.vec3), F: wp.array(dtype=wp.mat33), dFc: wp.array(dtype=wp.mat33),
                  grid_v: wp.array(dtype=wp.vec3), grid_v1: wp.array(dtype=wp.vec3),
                  q: wp.array(dtype=wp.vec3), plane_m: wp.array(dtype=wp.vec3), plane_n: wp.array(dtype=wp.vec3),
                  gmin: wp.vec3, dx: float, inv_dx: float, dt: float, nx: int, ny: int, nz: int,
                  adj_v: wp.array(dtype=wp.vec3), adj_C: wp.array(dtype=wp.mat33), adj_Fnew: wp.array(dtype=wp.mat33),
                  adj_x: wp.array(dtype=wp.vec3), adj_F: wp.array(dtype=wp.mat33), adj_dFc: wp.array(dtype=wp.mat33),
                  has_adj_dFc: int, adj_grid_v: wp.array(dtype=wp.vec3), adj_grid_v1: wp.array(dtype=wp.vec3)):
    """The adjoint of k_g2p_gap (adjoints.k_g2p_adj with the side's field at each node)."""
    p = wp.tid()
    xp = x[p]
    if not valid_pos(xp):
        return
    qp = q[p]
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
                    g = gid(i, j, k, ny, nz)
                    vg = grid_v[g]
                    if gap_side(qp, plane_m[g], plane_n[g]) == 1:
                        vg = grid_v1[g]
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
                    s1 = gap_side(qp, plane_m[g], plane_n[g]) == 1
                    vg = grid_v[g]
                    if s1:
                        vg = grid_v1[g]
                    Ad = a_Cnew @ dgp
                    c = w * a_vnew + (C0 * w) * Ad
                    if s1:
                        wp.atomic_add(adj_grid_v1, g, c)
                    else:
                        wp.atomic_add(adj_grid_v, g, c)
                    a_w = wp.dot(a_vnew, vg) + C0 * wp.dot(vg, Ad)
                    a_xp = a_xp - ((C0 * w) * (a_CnT @ vg) + a_w * dw)
    adj_x[p] = adj_x[p] + a_xp


def record_g2p_gap(tape, tr, t: int, dfc, N: int):
    """Record k_g2p_gap's adjoint for step t of Trajectory `tr` on `tape` (the forward was launched unrecorded)."""
    from .adjoints import _dummy_mat
    prm, dev = tr.prm, tr.device
    gmin, inv_dx = wp.vec3(*prm.grid_min), 1.0 / prm.dx
    x, F, gv, gv1, v1, C1, Fn = tr.x[t], tr.F[t], tr.gvel[t], tr.gvel1[t], tr.v[t + 1], tr.C[t + 1], tr.Fraw[t + 1]
    a_dfc = dfc.grad if dfc.grad is not None else _dummy_mat(dev)

    def backward():
        wp.launch(k_g2p_adj_gap, dim=N, inputs=[x, F, dfc, gv, gv1, tr.gap_q, tr.gap_m, tr.gap_n,
                                                gmin, prm.dx, inv_dx, prm.dt, prm.nx, prm.ny, prm.nz,
                                                v1.grad, C1.grad, Fn.grad, x.grad, F.grad, a_dfc,
                                                int(dfc.grad is not None), gv.grad, gv1.grad], device=dev)

    tape.record_func(backward=backward, arrays=[a for a in (x, F, gv, gv1, v1, C1, Fn, dfc) if a.grad is not None])


def attach(tr, gap, A, Z, n_grid: int, rg: bool):
    """Allocate the split step's arrays on Trajectory `tr`. gap = (plane_m (ngrid, 3), plane_n (ngrid, 3), q (N, 3) or
    None, vp): vp 0 = variant A, > 0 = variant B with that volume per particle (the sample's fill volume / N)."""
    pm, pn, q, vp = gap
    prm, dev, T = tr.prm, tr.device, tr.T
    tr.gap_m, tr.gap_n = A(pm, wp.vec3), A(pn, wp.vec3)
    tr.gap_q = A(q, wp.vec3) if q is not None else Z(wp.vec3)
    tr.gap_vp = float(vp)
    g1 = [wp.zeros(prm.ngrid, dtype=wp.float32, device=dev, requires_grad=rg) for _ in range(n_grid)]
    p1 = [wp.zeros(prm.ngrid, dtype=wp.vec3, device=dev, requires_grad=rg) for _ in range(n_grid)]
    v1 = [wp.zeros(prm.ngrid, dtype=wp.vec3, device=dev, requires_grad=rg) for _ in range(n_grid)]
    tr.gm1 = [g1[t % n_grid] for t in range(T)]
    tr.gmom1 = [p1[t % n_grid] for t in range(T)]
    tr.gvel1 = [v1[t % n_grid] for t in range(T)]
    # variant B's node volume: one buffer a step on a tape (the grid kernel's generated adjoint replays its branch)
    nv = (T if rg else 1) if tr.gap_vp > 0 else 1
    tr.gap_vol = [wp.zeros(prm.ngrid, dtype=float, device=dev) for _ in range(nv)]
    tr.gap_ones = wp.ones(tr.N, dtype=float, device=dev) if tr.gap_vp > 0 else None
    tr.gap_stats = [wp.zeros(4, dtype=float, device=dev) for _ in range(T)]          # a record a step


def grid_op(tr, t: int):
    """Step t's grid operation with the gap nodes' two fields."""
    from .contact import k_node_volume
    from .kernels import WALL_NODES
    p, dev = tr.prm, tr.device
    gv = tr.gap_vol[t % len(tr.gap_vol)]
    if tr.gap_vp > 0:                                   # variant B: the node's particle volume, off the tape
        gv.zero_()
        J = tr.J[t] if tr.carries else tr.gap_ones
        wp.launch(k_node_volume, dim=tr.N, inputs=[tr.x[t], J, tr.gap_vp, gv, wp.vec3(*p.grid_min), p.dx, 1.0 / p.dx,
                                                   p.nx, p.ny, p.nz], device=dev, record_tape=False)
    tr.gap_stats[t].zero_()
    wp.launch(k_grid_op_gap, dim=p.ngrid, inputs=[tr.gm[t], tr.gmom[t], tr.gm1[t], tr.gmom1[t], tr.gap_n, gv,
                                                  int(tr.gap_vp > 0), tr.gvel[t], tr.gvel1[t], p.dt, wp.vec3(*p.f_ext),
                                                  p.grid_min[1], p.dx, p.nx, p.ny, p.nz, p.floor_y, p.floor_friction,
                                                  WALL_NODES, tr.gap_stats[t]], device=dev)


def record(tr) -> list:
    """The last rollout's split record over its steps: [split mass, separating mass, split node-steps, separating
    node-steps] (mass summed over steps)."""
    import torch
    return [float(v) for v in torch.stack([wp.to_torch(s) for s in tr.gap_stats]).sum(0).tolist()]


def p2g_side1(tr, t: int, dfc, bnb, bnc, bK: int):
    """Step t's side-1 deposit (after the frozen P2G; on a tape by its generated adjoint)."""
    prm = tr.prm
    if tr.share_grid or tr.persistent:
        tr.gm1[t].zero_()
        tr.gmom1[t].zero_()
    wp.launch(k_p2g_side1, dim=tr.N, inputs=[tr.x[t], tr.v[t], tr.C[t], tr.F[t], dfc, tr.P[t], tr.m, tr.vol, bnb, bnc,
                                             bK, tr.gap_q, tr.gap_m, tr.gap_n, tr.gm1[t], tr.gmom1[t],
                                             wp.vec3(*prm.grid_min), prm.dx, 1.0 / prm.dx, prm.dt, prm.drag, prm.nx,
                                             prm.ny, prm.nz], device=tr.device)


def g2p(tr, t: int, dfc, tape):
    """Step t's G2P from each particle's side's field (the hand adjoint on a hand tape, else the generated one)."""
    prm = tr.prm
    wp.launch(k_g2p_gap, dim=tr.N, inputs=[tr.x[t], tr.v[t + 1], tr.C[t + 1], tr.F[t], dfc, tr.Fraw[t + 1], tr.gvel[t],
                                           tr.gvel1[t], tr.gap_q, tr.gap_m, tr.gap_n, wp.vec3(*prm.grid_min), prm.dx,
                                           1.0 / prm.dx, prm.dt, prm.nx, prm.ny, prm.nz],
              device=tr.device, record_tape=tape is None)
    if tape is not None:
        record_g2p_gap(tape, tr, t, dfc, tr.N)
