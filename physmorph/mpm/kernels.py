"""MLS-MPM forward kernels (Warp). Equations refer to docs/SPEC.md §3.3.

Oracle: DiffMPMLib3D/ForwardSimulation.cpp. Cubic B-spline 4^3 stencil,
C0 = 3/dx^2, APIC affine. Stress uses F_e = (F+dFc) Fp^{-1} (D1);
the P scratch stores dPsi/d(F+dFc), not the elastic PK1 dPsi/dF_e.
With the exact volume (D129) the stress reads (J / det F)^(1/3) (F+dFc) in place of F+dFc.
"""
from __future__ import annotations

import warp as wp

from .constitutive import weight, pk1_fixed_corotated, pk1_fixed_corotated_polar


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


@wp.kernel
def k_stress_polar(F: wp.array(dtype=wp.mat33), dFc: wp.array(dtype=wp.mat33),
                   Fp: wp.array(dtype=wp.mat33), lam: wp.array(dtype=float),
                   mu: wp.array(dtype=float), P: wp.array(dtype=wp.mat33)):
    p = wp.tid()
    Fpi = wp.inverse(Fp[p])
    Fe = (F[p] + dFc[p]) @ Fpi
    Pe = pk1_fixed_corotated_polar(Fe, lam[p], mu[p])
    P[p] = Pe @ wp.transpose(Fpi)


# ── the volume the motion makes, carried in the smoothed F (D131, --volume_exact carried) ─────────────────────
# The smoothing (k_update, s = prm.smoothing) keeps (1 - s) of every step's deformation increment in F, so the stress
# read J = det F ~ 1.00-1.01 where the motion's own volume was J ~ 4 (D128). The tracked J is the motion's volume
# (k_volume_update_motion), and the smoothed F is rescaled to it after every step (k_volume_carry); the stress reads
# F + dFc as on the old path. det F is floored at 1e-6 as the constitutive model floors J: an inverted F is rejected
# by the trajectory checks, the floor only keeps the rejected trial finite.
@wp.func
def volume_scale(F: wp.mat33, J: float) -> float:
    return wp.pow(wp.max(J, 1.0e-6) / wp.max(wp.determinant(F), 1.0e-6), 1.0 / 3.0)


@wp.kernel
def k_volume_update_motion(C: wp.array(dtype=wp.mat33), J_in: wp.array(dtype=float), J_out: wp.array(dtype=float),
                           dt: float):
    """The motion's own volume: J_{t+1} = J_t det(I + dt C_{t+1}), the determinant of the geometric deformation
    gradient Fg (k_geom_update) carried as a scalar; the control takes no part (D129, D130)."""
    p = wp.tid()
    J_out[p] = J_in[p] * wp.determinant(wp.identity(n=3, dtype=float) + dt * C[p])


@wp.kernel
def k_volume_carry(F_in: wp.array(dtype=wp.mat33), J: wp.array(dtype=float), F_out: wp.array(dtype=wp.mat33)):
    """The smoothed F carries the motion's volume (D131, --volume_exact carried): after the blend (k_update) F is
    rescaled to det F = J (k_volume_update_motion's), F_out = (J / det F)^(1/3) F; its isochoric part, the smoothing's
    shape, is unchanged. The stress then reads F + dFc as on the old path. D130 read (J / det F)^(1/3) (F + dFc) on an F
    whose volume nothing held: F -> lam F changed only the control's weight (F + dFc / lam) and the descent shrank lam
    (det F 0.41 -> 0.03 on the 300k dragon), until the trajectory check rejected the trials."""
    p = wp.tid()
    F_out[p] = volume_scale(F_in[p], J[p]) * F_in[p]


# ── grid reset ──────────────────────────────────────────────────────────────
@wp.kernel
def k_zero_scalar(a: wp.array(dtype=float)):
    a[wp.tid()] = 0.0


@wp.kernel
def k_zero_vec(a: wp.array(dtype=wp.vec3)):
    a[wp.tid()] = wp.vec3(0.0, 0.0, 0.0)


# ── P2G — eq (4)(5), oracle SingleParticle_to_grid ──────────────────────────
@wp.func
def bond_velocity(v: wp.array(dtype=wp.vec3), nbr: wp.array(dtype=int), p: int, K: int):
    # A function call replays the dynamic sum before differentiating w(x) * vp.
    # Inlining this loop in k_p2g leaves its accumulator at zero in Warp's replay.
    vs = wp.vec3(0.0, 0.0, 0.0)
    for a in range(K):
        vs = vs + v[nbr[p * K + a]]
    return vs / float(K)


@wp.kernel
def k_p2g(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
          C: wp.array(dtype=wp.mat33), F: wp.array(dtype=wp.mat33),
          dFc: wp.array(dtype=wp.mat33), P: wp.array(dtype=wp.mat33),
          m: wp.array(dtype=float), vol: wp.array(dtype=float),
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
    G = -C0 * dt * vol[p] * (P[p] @ wp.transpose(Feff)) + m[p] * C[p]   # total-PK1 form
    vp = v[p]
    if bond_K > 0 and frag[p] > 0.5:
        # FRAGMENT (its occupied-cell component is not the body's): the grid cannot couple
        # it to the body; use the material transfer — the mean velocity of its frozen
        # source neighbours (material PIC; k_update does the position projection)
        vp = bond_velocity(v, nbr, p, bond_K)
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


# ── the 3^3-cell particle count (the material bonds' decoupling test, k_frag_step) ─────────
# Piecewise constant in x, computed outside the tape.
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
def k_neighbour_count(x: wp.array(dtype=wp.vec3), cnt: wp.array(dtype=int), gmin: wp.vec3,
                      inv_dx: float, nx: int, ny: int, nz: int, ncount: wp.array(dtype=float)):
    """The particles in the 3^3 cells around each particle's own (k_cell_count's counts), itself included."""
    p = wp.tid()
    xp = x[p]
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
          grid_v: wp.array(dtype=wp.vec3),
          gmin: wp.vec3, dx: float, inv_dx: float, dt: float,
          nx: int, ny: int, nz: int):
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
             frag: wp.array(dtype=float), bond_K: int, bond_frac: float,
             snbr: wp.array(dtype=int), space_K: int, space_r: wp.array(dtype=float)):
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
    if space_K > 0:
        # MINIMUM SPACING (D70): the grid holds about 200 particles a cell and cannot see two of them pressed
        # together, so where the material is stretched and torn the arrangement under the surface ends uneven
        # (D68, D69). A particle is moved away from each of its frozen neighbours (the nearest at the window's
        # start) that is nearer than the pair's spacing, by half the overlap, over one window like the bonds. The
        # spacing is per particle (space_r[p], the pitch of its own rest volume: a surface-dense sample, D122, has
        # two) and a pair's is the mean of the two, the one value of a uniform sample
        push = wp.vec3(0.0, 0.0, 0.0)
        for a in range(space_K):
            q = snbr[p * space_K + a]
            d = x_in[p] - x_in[q]
            L = wp.length(d)
            r = 0.5 * (space_r[p] + space_r[q])
            if L < r and L > 1.0e-9:
                push = push + (r - L) * d / L
        xp = xp + bond_frac * 0.5 * push
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
def k_layer_relax(d: wp.array(dtype=float), mask: wp.array(dtype=float), nrm: wp.array(dtype=wp.vec3),
                  rn: wp.array(dtype=wp.vec3), nbr: wp.array(dtype=int), w: wp.array(dtype=float), K: int,
                  frac: float, ref: wp.array(dtype=float),
                  s: wp.array(dtype=float), b: wp.array(dtype=wp.vec3)):
    """The relaxation's normal displacement of this step, s = -frac (d - dbar - ref), and its
    moments on the six rigid modes of the layer: b[0] += s n, b[1] += s (r x n). The rough
    residual d - dbar is relaxed towards ref[p], the same quantity on the target's own surface
    where the particle stands (zero where none is given): relaxed towards zero, the target's
    relief below a cell went with the sampling noise (D82, D88)."""
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


@wp.kernel
def k_layer_project(x_in: wp.array(dtype=wp.vec3), s: wp.array(dtype=float), b: wp.array(dtype=wp.vec3),
                    mask: wp.array(dtype=float), nrm: wp.array(dtype=wp.vec3), rn: wp.array(dtype=wp.vec3),
                    M11: wp.mat33, M12: wp.mat33, M21: wp.mat33, M22: wp.mat33,
                    u: wp.array(dtype=float), frac_u: float, ug: wp.array(dtype=float),
                    x_out: wp.array(dtype=wp.vec3)):
    """Outer-layer position update: the relaxation and the POSITION-MODE CONTROL CHANNEL u
    (docs/surface_gradient.md §7): u[p] is a per-window normal displacement leaf of the
    optimiser, applied frac_u = 1/T per step, so the render covector reaches it without the
    grid's low-pass (its adjoint is the identity times the physics response). The relaxation
    loses its part along the six rigid modes of the layer (M = the inverse Gram matrix of the
    modes n and r x n, r about the window's starting centre of mass): a position constraint
    below the grid moves no mass as a whole and turns none (D97: it carried 94 % of the body's
    net drift; u has been made so in D94)."""
    p = wp.tid()
    if mask[p] < 0.5:
        x_out[p] = x_in[p]
        return
    ct = M11 @ b[0] + M12 @ b[1]
    cr = M21 @ b[0] + M22 @ b[1]
    sp = s[p] - wp.dot(nrm[p], ct) - wp.dot(rn[p], cr)
    x_out[p] = x_in[p] + (frac_u * ug[p] * u[p] + sp) * nrm[p]


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
