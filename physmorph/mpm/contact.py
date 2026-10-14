"""Contact between bodies on their own grid fields (D147): the grid operation of a multi-body step.

Each body has its own grid field (mass, momentum) and gathers its particles' new velocities from it alone, so two
bodies within the cubic kernel's reach of each other no longer share one velocity (on one grid they stick: Hu et al.
2018, "the kernel support across the boundary naturally causes velocity field smoothing"). At a node where both fields
have mass, the multi-velocity rule of Bardenhagen et al. (2000, 2001) acts on the pair against their centre-of-mass
velocity, with one normal for the pair (the difference of the two fields' mass gradients, so the two changes are equal
and opposite and the momentum and angular momentum of the pair are kept):

    in contact when V_0 + V_1 >= dx^3 (the two fields' node volumes fill the cell) and (v_0 - v_cm) . n > 0;
    then v_0 <- v_0 - d n - min(1, mu d / |t|) t  (d = (v_0 - v_cm) . n, t its tangential rest; Coulomb),
         v_1 <- v_1 + (m_0 / m_1) (d n + min(1, mu d / |t|) t).

The volume gate is the touching condition: for two half-spaces touching at a plane, a partition-of-unity kernel gives
Phi(a) + Phi(-a) = 1 at every node (Phi its cumulative), so the node volumes add up to the cell exactly; with a gap they
fall short, with an overlap they exceed it. Without it the rule acts wherever both kernels reach a node, up to four
cells apart. The floor is the same rule with the floor as a body of infinite mass whose node volume is the kernel's part
below it: V_b >= dx^3 Phi((y_i - y_floor) / dx). The node volumes are each particle's current volume (the sample's
volume per particle times the tracked J) and decide the branch only: they are deposited outside the tape, per step,
because the contact kernel's generated adjoint replays its forward body (the gate included) at backward time.
"""
from __future__ import annotations

import warp as wp

from .kernels import WALL_NODES, base_node, gid, valid_pos, weight


@wp.func
def bspline_cdf(d: float) -> float:
    """Phi(d): the part of the 1-D cubic B-spline (support [-2, 2], unit integral) below d."""
    if d <= -2.0:
        return 0.0
    if d >= 2.0:
        return 1.0
    a = -wp.abs(d)
    lo = float(0.0)
    if a <= -1.0:
        lo = (2.0 + a) * (2.0 + a) * (2.0 + a) * (2.0 + a) / 24.0
    else:
        lo = 1.0 / 24.0 + (-a * a * a * a / 8.0 - a * a * a / 3.0 + 2.0 * a / 3.0) + 1.0 / 8.0 + 1.0 / 3.0
    if d > 0.0:
        return 1.0 - lo
    return lo


@wp.func
def _grad_m(gm: wp.array(dtype=float), i: int, j: int, k: int, nx: int, ny: int, nz: int, dx: float):
    """The node masses' central difference (one-sided at the box faces)."""
    ip = wp.min(i + 1, nx - 1)
    im = wp.max(i - 1, 0)
    jp = wp.min(j + 1, ny - 1)
    jm = wp.max(j - 1, 0)
    kp = wp.min(k + 1, nz - 1)
    km = wp.max(k - 1, 0)
    return wp.vec3((gm[gid(ip, j, k, ny, nz)] - gm[gid(im, j, k, ny, nz)]) / (float(ip - im) * dx),
                   (gm[gid(i, jp, k, ny, nz)] - gm[gid(i, jm, k, ny, nz)]) / (float(jp - jm) * dx),
                   (gm[gid(i, j, kp, ny, nz)] - gm[gid(i, j, km, ny, nz)]) / (float(kp - km) * dx))


@wp.func
def _walls_floor(v: wp.vec3, i: int, j: int, k: int, nx: int, ny: int, nz: int, on_floor: bool, friction: float):
    """kernels.k_grid_op's separating domain walls, and the floor where the gate put the node in contact with it."""
    vx = v[0]
    vy = v[1]
    vz = v[2]
    if i < WALL_NODES and vx < 0.0:
        vx = 0.0
    if i >= nx - WALL_NODES and vx > 0.0:
        vx = 0.0
    if j < WALL_NODES and vy < 0.0:
        vy = 0.0
    if j >= ny - WALL_NODES and vy > 0.0:
        vy = 0.0
    if k < WALL_NODES and vz < 0.0:
        vz = 0.0
    if k >= nz - WALL_NODES and vz > 0.0:
        vz = 0.0
    out = wp.vec3(vx, vy, vz)
    if on_floor and out[1] < 0.0:
        vt = wp.vec3(out[0], 0.0, out[2])
        tl = wp.length(vt)
        vn = -out[1]
        if tl > 1.0e-8:
            vt = vt * wp.max(0.0, 1.0 - friction * vn / tl)
        out = wp.vec3(vt[0], 0.0, vt[2])
    return out


@wp.kernel
def k_grid_op_pair(gm0: wp.array(dtype=float), gmom0: wp.array(dtype=wp.vec3), gvol0: wp.array(dtype=float),
                   gvel0: wp.array(dtype=wp.vec3), gm1: wp.array(dtype=float), gmom1: wp.array(dtype=wp.vec3),
                   gvol1: wp.array(dtype=float), gvel1: wp.array(dtype=wp.vec3),
                   dt: float, f_ext: wp.vec3, gmin_y: float, dx: float, nx: int, ny: int, nz: int,
                   floor_y: float, mu: float, impulse: wp.array(dtype=wp.vec3)):
    """The grid operation of two bodies (kernels.k_grid_op per field, the walls and the floor included) with the
    gated contact between them. impulse[0] += body 0's momentum change by the contact (a record)."""
    g = wp.tid()
    i = g // (ny * nz)
    j = (g // nz) % ny
    k = g % nz
    m0 = gm0[g]
    m1 = gm1[g]
    v0 = wp.vec3(0.0, 0.0, 0.0)
    v1 = wp.vec3(0.0, 0.0, 0.0)
    if m0 > 1.0e-12:
        v0 = gmom0[g] / m0 + dt * f_ext
    if m1 > 1.0e-12:
        v1 = gmom1[g] / m1 + dt * f_ext
    cell = dx * dx * dx
    if m0 > 1.0e-12 and m1 > 1.0e-12:
        if gvol0[g] + gvol1[g] >= cell:
            n = _grad_m(gm1, i, j, k, nx, ny, nz, dx) - _grad_m(gm0, i, j, k, nx, ny, nz, dx)
            ln = wp.length(n)
            if ln > 1.0e-12:
                nn = n / ln
                w0 = v0 - (m0 * v0 + m1 * v1) / (m0 + m1)
                d = wp.dot(w0, nn)
                if d > 0.0:
                    t0 = w0 - d * nn
                    tl = wp.length(t0)
                    r = float(0.0)
                    if tl > 1.0e-12:
                        r = wp.min(1.0, mu * d / tl)
                    dv0 = -d * nn - r * t0
                    v0 = v0 + dv0
                    v1 = v1 - (m0 / m1) * dv0
                    wp.atomic_add(impulse, 0, m0 * dv0)
    h = (gmin_y + float(j) * dx - floor_y) / dx
    need = cell * bspline_cdf(h)
    if m0 > 1.0e-12:
        gvel0[g] = _walls_floor(v0, i, j, k, nx, ny, nz, h < 2.0 and gvol0[g] >= need, mu)
    else:
        gvel0[g] = wp.vec3(0.0, 0.0, 0.0)
    if m1 > 1.0e-12:
        gvel1[g] = _walls_floor(v1, i, j, k, nx, ny, nz, h < 2.0 and gvol1[g] >= need, mu)
    else:
        gvel1[g] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel(enable_backward=False)
def k_node_volume(x: wp.array(dtype=wp.vec3), J: wp.array(dtype=float), vp: float, gvol: wp.array(dtype=float),
                  gmin: wp.vec3, dx: float, inv_dx: float, nx: int, ny: int, nz: int):
    """Each particle's current volume, vp J, onto its 64 nodes (the gate's node volume; outside the tape)."""
    p = wp.tid()
    xp = x[p]
    if not valid_pos(xp):
        return
    b = base_node(xp, gmin, inv_dx)
    for oi in range(4):
        for oj in range(4):
            for ok in range(4):
                i = b[0] + oi
                j = b[1] + oj
                k = b[2] + ok
                if i >= 0 and i < nx and j >= 0 and j < ny and k >= 0 and k < nz:
                    xg = gmin + wp.vec3(float(i), float(j), float(k)) * dx
                    wp.atomic_add(gvol, gid(i, j, k, ny, nz), weight(xg - xp, inv_dx) * vp * J[p])
