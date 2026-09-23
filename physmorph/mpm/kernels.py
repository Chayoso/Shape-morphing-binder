"""MLS-MPM forward kernels (Warp). Equations refer to docs/SPEC.md §3.3.

Oracle: DiffMPMLib3D/ForwardSimulation.cpp. Cubic B-spline 4^3 stencil,
C0 = 3/dx^2, APIC affine. Stress uses F_e = (F+dFc) Fp^{-1} (D1);
the P scratch stores dPsi/d(F+dFc), not the elastic PK1 dPsi/dF_e.
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
    Fpi = wp.inverse(Fp[p])
    Fe = (F[p] + dFc[p]) @ Fpi
    Pe = pk1_fixed_corotated(Fe, lam[p], mu[p])
    # Chain rule for Psi((F+dFc) Fp^-1): Ptotal = Pe Fp^-T.  The P2G
    # product below is then Ptotal(F+dFc)^T = Pe Fe^T (symmetric tau).
    P[p] = Pe @ wp.transpose(Fpi)


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
          m: wp.array(dtype=float), vol: wp.array(dtype=float), omega: wp.array(dtype=float),
          nbr: wp.array(dtype=int), frag: wp.array(dtype=float), bond_K: int,
          grid_m: wp.array(dtype=float), grid_v: wp.array(dtype=wp.vec3),
          gmin: wp.vec3, dx: float, inv_dx: float, dt: float, drag: float,
          nx: int, ny: int, nz: int):
    p = wp.tid()
    xp = x[p]
    if not valid_pos(xp):
        return
    Feff = F[p] + dFc[p]
    C0 = 3.0 * inv_dx * inv_dx
    # omega[p] = support gate on the APIC affine term (1 = plain APIC; k_support_gate)
    G = -C0 * dt * vol[p] * (P[p] @ wp.transpose(Feff)) + omega[p] * m[p] * C[p]   # total-PK1 form
    vp = v[p]
    if bond_K > 0 and frag[p] > 0.5:
        # FRAGMENT (its occupied-cell component is not the body's): the grid cannot couple
        # it to the body; use the material transfer — the mean velocity of its frozen
        # source neighbours (material PIC; k_update does the position projection)
        vs = wp.vec3(0.0, 0.0, 0.0)
        for a in range(bond_K):
            vs = vs + v[nbr[p * bond_K + a]]
        vp = vs / float(bond_K)
    mv = m[p] * vp * (1.0 - dt * drag)
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


# ── support gate (Yao-Zhao 2026, arXiv 2603.03860 §support-gated APIC) ──────
# A particle whose 3^3-cell neighbourhood is depleted transfers PIC momentum only: the
# affine term m*C*(x_g - x_p) of a fringe particle (steep velocity gradient at the front)
# is what hands the empty-side nodes an outward velocity, and with no other particle on
# those nodes nothing pulls it back (numerical fracture). omega is piecewise constant in
# x, so the P2G adjoint reads it as a constant (launched with record_tape=False).
@wp.kernel
def k_cell_count(x: wp.array(dtype=wp.vec3), gmin: wp.vec3, inv_dx: float,
                 nx: int, ny: int, nz: int, cnt: wp.array(dtype=int)):
    p = wp.tid()
    xp = x[p]
    if not valid_pos(xp):
        return
    i = int(wp.floor((xp[0] - gmin[0]) * inv_dx))
    j = int(wp.floor((xp[1] - gmin[1]) * inv_dx))
    k = int(wp.floor((xp[2] - gmin[2]) * inv_dx))
    if i >= 0 and i < nx and j >= 0 and j < ny and k >= 0 and k < nz:
        wp.atomic_add(cnt, gid(i, j, k, ny, nz), 1)


@wp.kernel
def k_support_gate(x: wp.array(dtype=wp.vec3), cnt: wp.array(dtype=int), gmin: wp.vec3,
                   inv_dx: float, nx: int, ny: int, nz: int, n0: float, r_lo: float,
                   r_hi: float, omega: wp.array(dtype=float), ncount: wp.array(dtype=float)):
    p = wp.tid()
    xp = x[p]
    omega[p] = 1.0
    ncount[p] = 0.0
    if not valid_pos(xp):
        return
    ci = int(wp.floor((xp[0] - gmin[0]) * inv_dx))
    cj = int(wp.floor((xp[1] - gmin[1]) * inv_dx))
    ck = int(wp.floor((xp[2] - gmin[2]) * inv_dx))
    n = int(0)
    for oi in range(3):
        for oj in range(3):
            for ok in range(3):
                i = ci - 1 + oi
                j = cj - 1 + oj
                k = ck - 1 + ok
                if i >= 0 and i < nx and j >= 0 and j < ny and k >= 0 and k < nz:
                    n = n + cnt[gid(i, j, k, ny, nz)]
    ncount[p] = float(n)
    s = (float(n) / n0 - r_lo) / (r_hi - r_lo)
    s = wp.clamp(s, 0.0, 1.0)
    omega[p] = s * s * (3.0 - 2.0 * s)


@wp.kernel
def k_frag_step(ncount: wp.array(dtype=float), frag_commit: wp.array(dtype=float),
                frag: wp.array(dtype=float)):
    """Decoupling flag for THIS step: a particle with no other particle in the 3^3 cells around
    its own (count <= 1 counts only itself) is decoupled now — the grid cannot act on it — or
    it was flagged at the commit (the runner's fragment mask). Evaluated every step: the
    single-particle leaders of the expansion phase clear the fracture gap inside one window
    (150k bob: 54 particles flung 1.3-3.7 wu, static afterwards), and a mask fixed at the
    window start bonds them one window too late."""
    p = wp.tid()
    if ncount[p] <= 1.0 or frag_commit[p] > 0.5:
        frag[p] = 1.0
    else:
        frag[p] = 0.0


# ── material re-coupling of decoupled particles (numerical-fracture repair) ────
# Implemented inside k_p2g (material-PIC velocity) and k_update (bond projection); the
# decoupling test is the 3^3-cell count of the support-gate kernels. An explicit bond
# spring was tried first (2026-09-16) and rejected: a linear spring with a multi-wu
# extension integrated explicitly is unstable at these stiffnesses.


# ── grid op — eq (6), oracle SingleNode_op ──────────────────────────────────
# Out-of-place (momentum in -> velocity out). In-place read-write of a
# differentiable array breaks Warp's adjoint, so we keep momentum and velocity
# in distinct arrays. In-place forward use passes the same array for in/out.
WALL_NODES = 2   # cubic B-spline half-support: a particle within 2 cells of the box edge has a truncated stencil


@wp.kernel
def k_grid_op(grid_m: wp.array(dtype=float), grid_mom: wp.array(dtype=wp.vec3),
              grid_vel: wp.array(dtype=wp.vec3), dt: float, f_ext: wp.vec3,
              gmin_y: float, dx: float, nx: int, ny: int, nz: int, floor_y: float, friction: float,
              wall_nodes: int):
    g = wp.tid()
    mg = grid_m[g]
    if mg > 1.0e-12:
        vg = grid_mom[g] / mg + dt * f_ext
        j = (g // nz) % ny
        # domain walls (separating): the outward normal velocity is zeroed on the outermost
        # `wall_nodes` node layers of every face, tangential and inward motion stay free.
        # Without them the box edge was a TRAP: a particle within the stencil half-support of
        # the edge deposits on and gathers from a truncated stencil, loses momentum each step
        # and freezes there — the 150k C shed 600–900 particles per window into that band
        # (the "chunks" re-attached by the net sat at the box corners, static, 1.5–3 wu from
        # any target point) while the 40k C never reached it. A wall the material can slide
        # along and be pulled back from is the oracle's domain treatment; the trap was not.
        if wall_nodes > 0:
            i = g // (ny * nz)
            k = g % nz
            vx = vg[0]
            vy = vg[1]
            vz = vg[2]
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
            vg = wp.vec3(vx, vy, vz)
        # floor boundary: separating + Coulomb-style friction (rubber bounces, honey splats)
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
             dt: float, s: float,
             nbr: wp.array(dtype=int), rest: wp.array(dtype=float),
             frag: wp.array(dtype=float), bond_K: int, bond_frac: float):
    p = wp.tid()
    F_out[p] = (1.0 - s) * F_new[p] + s * F_in[p]   # blend new with OLD F
    xp = x_in[p] + dt * v[p]
    if bond_K > 0 and frag[p] > 0.5:
        # MATERIAL RE-COUPLING (decoupled particle): project toward the rest lengths of
        # its frozen source bonds (re-based at the window start) — position-based, one
        # full projection per step, tension only (compression is the grid's business)
        acc = wp.vec3(0.0, 0.0, 0.0)
        for a in range(bond_K):
            j = nbr[p * bond_K + a]
            d = x_in[j] - x_in[p]
            L = wp.length(d)
            r = rest[p * bond_K + a]
            if L > r and L > 1.0e-9:
                acc = acc + (L - r) * d / L
        xp = xp + bond_frac * acc / float(bond_K)      # bond_frac = 1/T: re-join over one window
    x_out[p] = xp


# ── outer-layer relaxation — a particle-scale POSITION projection in the FORWARD model ──
# docs/surface_gradient.md §6 (2026-09-19). The gradient-stage analysis (40k bunny, 8
# windows) showed the render covector to be a surface signal (99 % on the outer layer) whose
# bump-band content (65 % uncorrelated at 2 spacings) the control cannot act on: after the
# MPM adjoint every channel's control gradient has the grid's correlation length (~4
# spacings) and the window's response is 83-90 % smooth. A sub-cell surface bump is not
# reachable through the control stress. It is not reachable through a particle FORCE either:
# a force on one particle accelerates its cell's momentum, which P2G/G2P average over the
# ~50 particles of the cell — a critically damped spring on one bump moved it 2.4 % in a
# window (tests/test_layer_relax.py, first form). Sub-cell relative motion is not a momentum
# mode of the discretisation; it is a POSITION mode, and the same is true of the material
# bonds (k_update projects positions, not forces). So the relaxation is a per-step position
# projection: for each outer-layer particle p (frozen per window: mask, reference normal n_p,
# K same-side layer neighbours with Gaussian weights of h = 2 spacings), with c_p the
# weighted centroid of the neighbours at the current (advected) positions and
# d_p = n_p . (x_p - c_p) the plane residual,
#   x_p <- x_p - frac (d_p - dbar_p) n_p,   frac = 1/T
# the ROUGH part of the residual (what its neighbourhood mean does not explain: the sampling
# noise; curvature and features, which the neighbours share, cancel in d - dbar) is removed
# over one window, as the bonds re-join over one window. Both kernels are on the tape; the
# frozen arrays are constants.
@wp.kernel
def k_layer_resid(x: wp.array(dtype=wp.vec3),
                  mask: wp.array(dtype=float), nrm: wp.array(dtype=wp.vec3),
                  nbr: wp.array(dtype=int), w: wp.array(dtype=float), K: int,
                  d: wp.array(dtype=float)):
    p = wp.tid()
    if mask[p] < 0.5:
        d[p] = 0.0
        return
    # w rows are normalised on the host (layer rows sum to 1): a division by a loop-accumulated
    # weight sum inside the kernel broke the adjoint (gradients 1e23; scratch layer_adj2.py,
    # 2026-09-19) — no division here
    c = wp.vec3(0.0, 0.0, 0.0)
    for a in range(K):
        c = c + w[p * K + a] * x[nbr[p * K + a]]
    d[p] = wp.dot(nrm[p], x[p] - c)


@wp.kernel
def k_layer_project(x_in: wp.array(dtype=wp.vec3), d: wp.array(dtype=float),
                    mask: wp.array(dtype=float), nrm: wp.array(dtype=wp.vec3),
                    nbr: wp.array(dtype=int), w: wp.array(dtype=float), K: int,
                    frac: float, u: wp.array(dtype=float), frac_u: float, ug: wp.array(dtype=float),
                    x_out: wp.array(dtype=wp.vec3)):
    """Outer-layer position update: the relaxation (frac) and the POSITION-MODE CONTROL
    CHANNEL u (docs/surface_gradient.md §7): u[p] is a per-window normal displacement leaf
    of the optimiser, applied frac_u = 1/T per step, so the render covector reaches it
    without the grid's low-pass (its adjoint is the identity times the physics response)."""
    p = wp.tid()
    if mask[p] < 0.5:
        x_out[p] = x_in[p]
        return
    dbar = float(0.0)
    for a in range(K):
        dbar = dbar + w[p * K + a] * d[nbr[p * K + a]]
    x_out[p] = x_in[p] + (frac_u * ug[p] * u[p] - frac * (d[p] - dbar)) * nrm[p]


# ── geometric deformation gradient — the RENDER kinematics ───────────────────
# Fg_{t+1} = (I + dt C_{t+1}) Fg_t: transported by the actual spatial velocity
# derivative only. It receives NO control addition and NO temporal smoothing, so a
# Gaussian covariance Sigma = sigma0^2 Fg Fg^T (PhysGaussian kinematics) can change
# only when material moves (docs/render_controls_physics.md §3: the control's direct
# F route was measured to change the image at zero motion). Fg is not used by the
# constitutive law; the physics keeps the smoothed, controlled F.
@wp.kernel
def k_geom_update(C: wp.array(dtype=wp.mat33), Fg_in: wp.array(dtype=wp.mat33),
                  Fg_out: wp.array(dtype=wp.mat33), dt: float):
    p = wp.tid()
    Fg_out[p] = (wp.identity(n=3, dtype=float) + dt * C[p]) @ Fg_in[p]


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


# ── P3: the u channel THROUGH THE DEFORMATION GRADIENT (docs/final_plan.md 2) ────────
# The step's u displacement delta_p = frac_u u_p n_p is a displacement field on the outer
# layer. Its gradient over the frozen same-side neighbourhood,
#   G = sum_a (delta_a - delta_p) (x) g_a       (tangential, least squares; g from
#                                                 surface_recon.layer_grad_weights)
#     + delta_p (x) n_p / depth                  (normal: the layer below did not move),
# and F <- (I + G) F, so a rough u is a strain the stress resists in the following steps
# and the adjoint reaches u through F. The relaxation projection stays outside F (it is a
# constraint, like contact). k_update writes Fu[t+1]; this kernel writes F[t+1].
@wp.kernel
def k_layer_F(u: wp.array(dtype=float), ug: wp.array(dtype=float), mask: wp.array(dtype=float), nrm: wp.array(dtype=wp.vec3),
              nbr: wp.array(dtype=int), g: wp.array(dtype=wp.vec3), K: int,
              frac_u: float, inv_depth: float,
              F_in: wp.array(dtype=wp.mat33), F_out: wp.array(dtype=wp.mat33)):
    p = wp.tid()
    if mask[p] < 0.5:
        F_out[p] = F_in[p]
        return
    dp = (frac_u * ug[p] * u[p]) * nrm[p]
    G = wp.outer(dp, nrm[p]) * inv_depth
    for a in range(K):
        q = nbr[p * K + a]
        dq = (frac_u * ug[q] * u[q]) * nrm[q]
        G = G + wp.outer(dq - dp, g[p * K + a])
    F_out[p] = (wp.identity(n=3, dtype=float) + G) * F_in[p]
