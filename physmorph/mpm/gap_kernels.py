"""The two sides of a target gap on their own grid fields (D148, `--gap_split`): the deposit and grid kernels of a split
step (the G2P kernels in gap_g2p.py, the step's Python in gap_step.py).

The frozen P2G still deposits every particle into the total field (mass m, momentum p). At a gap node (gap_split.py:
a node outside the target whose kernel box holds two pieces of it; plane point m_g, unit normal n_g, n_g = 0 elsewhere)
a particle's share of side 1 is s = 1 when its window-start transport image q_p lies on the plane's positive side and
0 otherwise (sig = 0, the hard sides of D148 / D148b); with sig > 0 (D148c, soft sides) s = Phi((q_p - m_g) . n_g /
sig), the plan's mass beyond the plane under its own Gaussian blur (sig^2 = eps / 2 of the plan's kernel exp(-|x-y|^2 /
eps)). k_p2g_side1 deposits s times each particle's deposit at gap nodes into a second field (m1, p1); side 0 is the
total minus side 1. k_grid_op_gap: at a node with both sides (and, in variant B, a particle volume below one cell) the
sides' post-force velocities v0 = (p - p1) / (m - m1) + dt f, v1 = p1 / m1 + dt f; approaching, (v0 - v1) . n_g > 0,
both take the total field's velocity p / m + dt f (the single field's); separating, each keeps its own (free
separation, Bardenhagen et al. 2000). Every other node is kernels.k_grid_op's arithmetic on the total field. The
sides, the planes and `near` (1 where a gap node lies in the stencil of a particle whose base node this is: the
particles without one skip the split's kernels) are constants of the window's tape. k_p2g_side1_adj is the deposit's
hand adjoint (adjoints.k_p2g_adj's arithmetic, weighted by s).
"""
from __future__ import annotations

import warp as wp

from .adjoints import _weight_and_grad
from .constitutive import weight
from .kernels import base_node, bond_velocity, gid, valid_pos


@wp.func
def gap_weight(q: wp.vec3, pm: wp.vec3, pn: wp.vec3, sig: float) -> float:
    """The particle's share of side 1 at a node: 0 off the gap nodes; hard (sig = 0) the plane's side of q; soft
    (sig > 0) the normal CDF of q's signed distance to the plane over sig."""
    if pn[0] == 0.0 and pn[1] == 0.0 and pn[2] == 0.0:
        return 0.0
    d = wp.dot(q - pm, pn)
    if sig > 0.0:
        return 0.5 * (1.0 + wp.erf(d / (sig * 1.4142135623730951)))
    if d > 0.0:
        return 1.0
    return 0.0


@wp.func
def near_gap(b: wp.vec3i, near: wp.array(dtype=int), nx: int, ny: int, nz: int) -> int:
    """1 when a gap node may lie in the stencil of base node b (outside the node range: 1, the kernels then look)."""
    if b[0] >= 0 and b[0] < nx and b[1] >= 0 and b[1] < ny and b[2] >= 0 and b[2] < nz:
        return near[gid(b[0], b[1], b[2], ny, nz)]
    return 1


@wp.func
def side_share(p: int, g: int, qp: wp.vec3, plane_m: wp.array(dtype=wp.vec3), plane_n: wp.array(dtype=wp.vec3),
               sig: float, box0: wp.array(dtype=wp.vec3i), boxd: wp.array(dtype=wp.vec3i),
               boxi: wp.array(dtype=wp.vec3i), boxlab: wp.array(dtype=int), ncell: wp.vec3i,
               rep: wp.array(dtype=int), clampk: int) -> float:
    """The particle's share of side 1 at node g: by the node's pieces (D148c, gap_walls.py) where the node has a box
    and the particle's representative cell lies in one of its two pieces, else gap_weight's (the plane and the point).
    clampk = 1 (D148d): a representative cell outside the box is read at the box cell closest to it, and boxlab holds
    each box cell's nearest occupied cell's piece (gap_walls.BoxPieces(nearest=True)): the nearest wall."""
    bi = boxi[g]
    if bi[0] >= 0:
        rp = rep[p]
        if rp >= 0:
            a0 = box0[g]
            d = boxd[g]
            li = rp // (ncell[1] * ncell[2]) - a0[0]
            lj = (rp // ncell[2]) % ncell[1] - a0[1]
            lk = rp % ncell[2] - a0[2]
            if clampk != 0:
                li = wp.min(wp.max(li, 0), d[0] - 1)
                lj = wp.min(wp.max(lj, 0), d[1] - 1)
                lk = wp.min(wp.max(lk, 0), d[2] - 1)
            if li >= 0 and li < d[0] and lj >= 0 and lj < d[1] and lk >= 0 and lk < d[2]:
                lb = boxlab[bi[0] + (li * d[1] + lj) * d[2] + lk]
                if lb == bi[2]:
                    return 1.0
                if lb == bi[1]:
                    return 0.0
    return gap_weight(qp, plane_m[g], plane_n[g], sig)


@wp.kernel
def k_p2g_side1(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
                dFc: wp.array(dtype=wp.mat33), P: wp.array(dtype=wp.mat33),
                m: wp.array(dtype=float), vol: wp.array(dtype=float),
                nbr: wp.array(dtype=int), frag: wp.array(dtype=float), bond_K: int,
                q: wp.array(dtype=wp.vec3), plane_m: wp.array(dtype=wp.vec3), plane_n: wp.array(dtype=wp.vec3),
                near: wp.array(dtype=int), sig: float, box0: wp.array(dtype=wp.vec3i), boxd: wp.array(dtype=wp.vec3i),
                boxi: wp.array(dtype=wp.vec3i), boxlab: wp.array(dtype=int), ncell: wp.vec3i, rep: wp.array(dtype=int), clampk: int,
                grid_m1: wp.array(dtype=float), grid_v1: wp.array(dtype=wp.vec3),
                gmin: wp.vec3, dx: float, inv_dx: float, dt: float, drag: float,
                nx: int, ny: int, nz: int):
    """kernels.k_p2g's deposit times the particle's share of side 1, at the gap nodes."""
    p = wp.tid()
    xp = x[p]
    if not valid_pos(xp):
        return
    b = base_node(xp, gmin, inv_dx)
    if near_gap(b, near, nx, ny, nz) == 0:
        return
    qp = q[p]
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
                    s = side_share(p, g, qp, plane_m, plane_n, sig, box0, boxd, boxi, boxlab, ncell, rep, clampk)
                    if s > 0.0:
                        xg = gmin + wp.vec3(float(i), float(j), float(k)) * dx
                        dgp = xg - xp
                        w = weight(dgp, inv_dx)
                        wp.atomic_add(grid_m1, g, s * (w * m[p]))
                        wp.atomic_add(grid_v1, g, s * (w * (mv + G @ dgp)))


@wp.kernel(enable_backward=False)
def k_p2g_side1_adj(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                    C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
                    dFc: wp.array(dtype=wp.mat33), P: wp.array(dtype=wp.mat33),
                    m: wp.array(dtype=float), vol: wp.array(dtype=float),
                    nbr: wp.array(dtype=int), frag: wp.array(dtype=float), bond_K: int,
                    q: wp.array(dtype=wp.vec3), plane_m: wp.array(dtype=wp.vec3), plane_n: wp.array(dtype=wp.vec3),
                    near: wp.array(dtype=int), sig: float, box0: wp.array(dtype=wp.vec3i), boxd: wp.array(dtype=wp.vec3i),
                    boxi: wp.array(dtype=wp.vec3i), boxlab: wp.array(dtype=int), ncell: wp.vec3i, rep: wp.array(dtype=int), clampk: int,
                    gmin: wp.vec3, dx: float, inv_dx: float, dt: float, drag: float, nx: int, ny: int, nz: int,
                    adj_gm1: wp.array(dtype=float), adj_gv1: wp.array(dtype=wp.vec3),
                    adj_x: wp.array(dtype=wp.vec3), adj_v: wp.array(dtype=wp.vec3), adj_C: wp.array(dtype=wp.mat33),
                    adj_F: wp.array(dtype=wp.mat33), adj_dFc: wp.array(dtype=wp.mat33), has_adj_dFc: int,
                    adj_P: wp.array(dtype=wp.mat33)):
    """The adjoint of k_p2g_side1 (adjoints.k_p2g_adj with each node's term times the share s, a constant of the tape);
    m, vol, the bonds, the fragment flags, the sides and the planes are constants of the tape."""
    p = wp.tid()
    xp = x[p]
    if not valid_pos(xp):
        return
    b = base_node(xp, gmin, inv_dx)
    if near_gap(b, near, nx, ny, nz) == 0:
        return
    qp = q[p]
    Feff = F[p] + dFc[p]
    C0 = 3.0 * inv_dx * inv_dx
    sc = -C0 * dt * vol[p]
    G = sc * (P[p] @ wp.transpose(Feff)) + m[p] * C[p]
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
                    g = gid(i, j, k, ny, nz)
                    s = side_share(p, g, qp, plane_m, plane_n, sig, box0, boxd, boxi, boxlab, ncell, rep, clampk)
                    if s > 0.0:
                        dgp = gmin + wp.vec3(float(i), float(j), float(k)) * dx - xp
                        w, dw = _weight_and_grad(dgp, inv_dx)
                        agv = s * adj_gv1[g]
                        a_w = (s * adj_gm1[g]) * mp + wp.dot(agv, mv + G @ dgp)
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
    adj_P[p] = adj_P[p] + sc * (a_G @ Feff)
    a_Feff = sc * (wp.transpose(a_G) @ P[p])
    adj_F[p] = adj_F[p] + a_Feff
    if has_adj_dFc != 0:
        adj_dFc[p] = adj_dFc[p] + a_Feff


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
