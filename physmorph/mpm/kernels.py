"""MLS-MPM forward kernels (Warp). Equations refer to docs/SPEC.md §3.3.

Oracle: DiffMPMLib3D/ForwardSimulation.cpp. Cubic B-spline 4^3 stencil,
C0 = 3/dx^2, APIC affine. Stress uses F_e = (F+dFc) Fp^{-1} (D1).
"""
from __future__ import annotations

import warp as wp

from .constitutive import weight, pk1_fixed_corotated


@wp.func
def gid(i: int, j: int, k: int, ny: int, nz: int) -> int:
    return (i * ny + j) * nz + k


@wp.func
def base_node(xp: wp.vec3, gmin: wp.vec3, inv_dx: float) -> wp.vec3i:
    return wp.vec3i(
        int(wp.floor((xp[0] - gmin[0]) * inv_dx)) - 1,
        int(wp.floor((xp[1] - gmin[1]) * inv_dx)) - 1,
        int(wp.floor((xp[2] - gmin[2]) * inv_dx)) - 1,
    )


@wp.func
def valid_pos(xp: wp.vec3) -> bool:
    """Guard against NaN/Inf/runaway positions (floor(NaN) -> illegal index)."""
    return (xp[0] == xp[0] and xp[1] == xp[1] and xp[2] == xp[2]
            and wp.abs(xp[0]) < 1.0e5 and wp.abs(xp[1]) < 1.0e5 and wp.abs(xp[2]) < 1.0e5)


# ── stress — eq (3'), oracle P_op_1 ─────────────────────────────────────────
@wp.kernel
def k_stress(F: wp.array(dtype=wp.mat33), dFc: wp.array(dtype=wp.mat33),
             Fp: wp.array(dtype=wp.mat33), lam: wp.array(dtype=float),
             mu: wp.array(dtype=float), P: wp.array(dtype=wp.mat33)):
    p = wp.tid()
    Fe = (F[p] + dFc[p]) @ wp.inverse(Fp[p])
    P[p] = pk1_fixed_corotated(Fe, lam[p], mu[p])


# ── guidance velocity injection (distributed over substeps) ─────────────────
# Adds a small per-particle velocity each substep so elasticity can resist
# overshoot, instead of a single large velocity override (anti-ejection).
@wp.kernel
def k_add_guidance(v: wp.array(dtype=wp.vec3), d: wp.array(dtype=wp.vec3), gain: float):
    p = wp.tid()
    v[p] = v[p] + gain * d[p]


# ── grid reset ──────────────────────────────────────────────────────────────
@wp.kernel
def k_zero_scalar(a: wp.array(dtype=float)):
    a[wp.tid()] = 0.0


@wp.kernel
def k_zero_vec(a: wp.array(dtype=wp.vec3)):
    a[wp.tid()] = wp.vec3(0.0, 0.0, 0.0)


# ── P2G — eq (4)(5), oracle SingleParticle_to_grid ──────────────────────────
@wp.kernel
def k_p2g(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
          C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
          dFc: wp.array(dtype=wp.mat33), P: wp.array(dtype=wp.mat33),
          m: wp.array(dtype=float), vol: wp.array(dtype=float),
          grid_m: wp.array(dtype=float), grid_v: wp.array(dtype=wp.vec3),
          gmin: wp.vec3, dx: float, inv_dx: float, dt: float, drag: float,
          nx: int, ny: int, nz: int):
    p = wp.tid()
    xp = x[p]
    if not valid_pos(xp):
        return
    Feff = F[p] + dFc[p]
    C0 = 3.0 * inv_dx * inv_dx
    G = -C0 * dt * vol[p] * (P[p] @ wp.transpose(Feff)) + m[p] * C[p]   # eq (5) affine
    mv = m[p] * v[p] * (1.0 - dt * drag)
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
                    wp.atomic_add(grid_m, g, w * m[p])
                    wp.atomic_add(grid_v, g, w * (mv + G @ dgp))


# ── grid op — eq (6), oracle SingleNode_op ──────────────────────────────────
# Out-of-place (momentum in -> velocity out). In-place read-write of a
# differentiable array breaks Warp's adjoint, so we keep momentum and velocity
# in distinct arrays. In-place forward use passes the same array for in/out.
@wp.kernel
def k_grid_op(grid_m: wp.array(dtype=float), grid_mom: wp.array(dtype=wp.vec3),
              grid_vel: wp.array(dtype=wp.vec3), dt: float, f_ext: wp.vec3,
              gmin_y: float, dx: float, ny: int, nz: int, floor_y: float, friction: float):
    g = wp.tid()
    mg = grid_m[g]
    if mg > 1.0e-12:
        vg = grid_mom[g] / mg + dt * f_ext
        # floor boundary: separating + Coulomb-style friction (rubber bounces, honey splats)
        j = (g // nz) % ny
        node_y = gmin_y + float(j) * dx
        if node_y < floor_y and vg[1] < 0.0:
            vt = wp.vec3(vg[0], 0.0, vg[2])                 # tangential
            tl = wp.length(vt)
            vn = -vg[1]                                     # normal (into floor) magnitude
            if tl > 1.0e-8:
                vt = vt * wp.max(0.0, 1.0 - friction * vn / tl)   # friction
            vg = wp.vec3(vt[0], 0.0, vt[2])
        grid_vel[g] = vg
    else:
        grid_vel[g] = wp.vec3(0.0, 0.0, 0.0)


# ── G2P — eq (7)(8), oracle Grid_to_SingleParticle + op_2 ────────────────────
@wp.kernel
def k_g2p(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
          C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
          dFc: wp.array(dtype=wp.mat33), F_new: wp.array(dtype=wp.mat33),
          grid_v: wp.array(dtype=wp.vec3), eta: wp.array(dtype=float),
          gmin: wp.vec3, dx: float, inv_dx: float, dt: float,
          nx: int, ny: int, nz: int, v_max: float, eta_sym: int, eta_mode: int):
    p = wp.tid()
    xp = x[p]
    if not valid_pos(xp):
        return
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
                    vg = grid_v[gid(i, j, k, ny, nz)]
                    vnew = vnew + w * vg
                    Cnew = Cnew + C0 * w * wp.outer(vg, dgp)
    if v_max > 0.0:                          # clamp escape velocity (anti-scatter)
        sp = wp.length(vnew)
        if sp > v_max:
            vnew = vnew * (v_max / sp)
    # VISCOUS DISSIPATION (rubber<->honey). eta_sym=1 (OBJECTIVE, default for new work):
    # damp only sym(C) = strain rate D; the skew part (spin W) is preserved, so a rigid
    # rotation loses NO energy (material frame indifference). eta_sym=0 = legacy full-C
    # damping (damps spin too -> non-objective; kept for reproducing old results).
    # eta_mode 1 (EXPONENTIAL, dt-consistent): the per-step factor of a continuum decay rate eta is
    # exp(-dt*eta) exactly, so the recovered eta means the same thing at any dt and no clamp is
    # needed. eta_mode 0 (LEGACY linear 1-dt*eta) is the first-order truncation of it: its
    # continuum-equivalent rate is -ln(1-dt*eta)/dt > eta (at dt=1/480: +0.10% at eta=1, +7.45% at
    # eta=65), and inside the optimiser's own box (eta<=400) it can go negative -> the max(0,.)
    # clamp fires and kills C in a single step. Kept as default for old-table reproduction.
    fac = float(0.0)
    if eta_mode == 1:
        fac = wp.exp(-dt * eta[p])
    else:
        fac = wp.max(0.0, 1.0 - dt * eta[p])
    if eta_sym == 1:
        Csym = 0.5 * (Cnew + wp.transpose(Cnew))
        Cnew = (Cnew - Csym) + Csym * fac
    else:
        Cnew = Cnew * fac
    v[p] = vnew
    C[p] = Cnew
    F_new[p] = (wp.identity(n=3, dtype=float) + dt * Cnew) @ (F[p] + dFc[p])


# ── update (smoothing + advect) — eq (9), oracle op_2 + smooth ──────────────
# Functional: read curr (x_in, F_in), write next (x_out, F_out). In-place use
# passes the same array for in/out (element-wise, race-free).
@wp.kernel
def k_update(x_in: wp.array(dtype=wp.vec3), x_out: wp.array(dtype=wp.vec3),
             v: wp.array(dtype=wp.vec3), F_in: wp.array(dtype=wp.mat33),
             F_new: wp.array(dtype=wp.mat33), F_out: wp.array(dtype=wp.mat33),
             dt: float, s: float):
    p = wp.tid()
    F_out[p] = (1.0 - s) * F_new[p] + s * F_in[p]   # blend new with OLD F
    x_out[p] = x_in[p] + dt * v[p]


# ── particle-level separating floor (SHARP contact for drop heroes) ─────────
@wp.kernel
def k_floor_clamp(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                  floor_y: float, friction: float):
    p = wp.tid()
    xp = x[p]
    if xp[1] < floor_y:
        vp = v[p]
        vt = 1.0 - friction
        # separating, non-penetrating: clamp to floor, kill downward velocity (the
        # bounce comes from the MATERIAL's elastic rebound, so rubber bounces, honey splats)
        x[p] = wp.vec3(xp[0], floor_y, xp[2])
        v[p] = wp.vec3(vp[0] * vt, wp.max(0.0, vp[1]), vp[2] * vt)


# ── volume (one-time) — eq (10), oracle CalculatePointCloudVolumes ──────────
@wp.kernel
def k_volume(x: wp.array(dtype=wp.vec3), m: wp.array(dtype=float),
             grid_m: wp.array(dtype=float), vol: wp.array(dtype=float),
             gmin: wp.vec3, dx: float, inv_dx: float, nx: int, ny: int, nz: int):
    p = wp.tid()
    xp = x[p]
    if not valid_pos(xp):
        vol[p] = 0.0
        return
    mass = float(0.0)
    b = base_node(xp, gmin, inv_dx)
    for oi in range(4):
        for oj in range(4):
            for ok in range(4):
                i = b[0] + oi
                j = b[1] + oj
                k = b[2] + ok
                if i >= 0 and i < nx and j >= 0 and j < ny and k >= 0 and k < nz:
                    xg = gmin + wp.vec3(float(i), float(j), float(k)) * dx
                    w = weight(xg - xp, inv_dx)
                    mass = mass + w * grid_m[gid(i, j, k, ny, nz)]
    cell = dx * dx * dx
    rho = mass / cell
    if rho > 1.0e-12:
        vol[p] = m[p] / rho
    else:
        vol[p] = 0.0
