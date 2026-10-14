"""D147 throwaway prototype (branch wip/d147): two elastic spheres collide in MLS-MPM and must bounce apart.

Contact rule under test (multi-velocity field, Bardenhagen et al. 2000/2001 with a volume gate): each body has its own
grid field (mass, momentum); at a node where both fields have mass, the normal is the difference of the two fields'
mass gradients (central differences of the node masses, n from body 0 into body 1), the bodies are in contact when
their node volumes fill the cell (V0 + V1 >= dx^3, the touching condition for two half-spaces under any partition-of-
unity kernel: Phi(a) + Phi(-a) = 1) and approaching ((v0 - vcm) . n > 0); then the relative normal velocity is
removed and the tangential one reduced by Coulomb's mu times the normal change. Equal and opposite changes at one node:
linear and angular momentum are conserved by construction.

Modes: multi (the rule), multi_nogate (Bardenhagen's original: both masses + approaching), single (one field, the
frozen code: the bodies share one velocity where their kernels overlap).
No morph objective, no controls in the forward runs; --gradcheck tapes a short window through the contact with a
constant control on body 0 and compares the adjoint with central finite differences.
Run on hyde06 only.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import physmorph  # noqa: F401  (import before torch: the CuPy / NVRTC order, physmorph/__init__.py)
import torch
import warp as wp

from physmorph import gpu
from physmorph.mpm import kernels as K
from physmorph.mpm.constitutive import lame
from physmorph.sampling import load_normalized


# ── the contact grid operation ──────────────────────────────────────────────────────────────────────────────────
@wp.func
def _grad_m(gm: wp.array(dtype=float), i: int, j: int, k: int, nx: int, ny: int, nz: int, inv2dx: float):
    """Central difference of the node masses (one-sided at the box faces)."""
    ip = wp.min(i + 1, nx - 1)
    im = wp.max(i - 1, 0)
    jp = wp.min(j + 1, ny - 1)
    jm = wp.max(j - 1, 0)
    kp = wp.min(k + 1, nz - 1)
    km = wp.max(k - 1, 0)
    gx = (gm[K.gid(ip, j, k, ny, nz)] - gm[K.gid(im, j, k, ny, nz)]) * inv2dx * 2.0 / float(ip - im)
    gy = (gm[K.gid(i, jp, k, ny, nz)] - gm[K.gid(i, jm, k, ny, nz)]) * inv2dx * 2.0 / float(jp - jm)
    gz = (gm[K.gid(i, j, kp, ny, nz)] - gm[K.gid(i, j, km, ny, nz)]) * inv2dx * 2.0 / float(kp - km)
    return wp.vec3(gx, gy, gz)


@wp.func
def bspline_cdf(d: float) -> float:
    """Phi(d): the part of the 1-D cubic B-spline (support [-2, 2], unit integral) below d."""
    if d <= -2.0:
        return 0.0
    if d >= 2.0:
        return 1.0
    a = -wp.abs(d)
    if a <= -1.0:
        lo = (2.0 + a) * (2.0 + a) * (2.0 + a) * (2.0 + a) / 24.0
    else:
        lo = 1.0 / 24.0 + (-a * a * a * a / 8.0 - a * a * a / 3.0 + 2.0 * a / 3.0) + 1.0 / 8.0 + 1.0 / 3.0
    if d > 0.0:
        return 1.0 - lo
    return lo


@wp.func
def _boundary(vg: wp.vec3, i: int, j: int, k: int, nx: int, ny: int, nz: int, wall_nodes: int,
              node_y: float, floor_y: float, friction: float):
    """k_grid_op's separating walls and floor, unchanged."""
    vx = vg[0]
    vy = vg[1]
    vz = vg[2]
    if wall_nodes > 0:
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
def k_grid_op_contact(gm0: wp.array(dtype=float), gmom0: wp.array(dtype=wp.vec3),
                      gm1: wp.array(dtype=float), gmom1: wp.array(dtype=wp.vec3),
                      gvol0: wp.array(dtype=float), gvol1: wp.array(dtype=float),
                      gvel0: wp.array(dtype=wp.vec3), gvel1: wp.array(dtype=wp.vec3),
                      dt: float, f_ext: wp.vec3, gmin_y: float, dx: float, nx: int, ny: int, nz: int,
                      floor_y: float, friction: float, wall_nodes: int, mu_c: float, cell_vol: float, gate: int,
                      flag: wp.array(dtype=int), floor_gate: int):
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
    flag[g] = 0
    if m0 > 1.0e-12 and m1 > 1.0e-12:
        inv2dx = 0.5 / dx
        n = _grad_m(gm1, i, j, k, nx, ny, nz, inv2dx) - _grad_m(gm0, i, j, k, nx, ny, nz, inv2dx)
        ln = wp.length(n)
        touching = True
        if gate != 0:
            touching = gvol0[g] + gvol1[g] >= cell_vol
        if ln > 1.0e-12 and touching:
            nn = n / ln
            vcm = (m0 * v0 + m1 * v1) / (m0 + m1)
            w0 = v0 - vcm
            dn = wp.dot(w0, nn)
            if dn > 0.0:                                  # approaching: remove the normal part, Coulomb on the rest
                t0 = w0 - dn * nn
                tl = wp.length(t0)
                ratio = float(0.0)
                if tl > 1.0e-12:
                    ratio = wp.min(1.0, mu_c * dn / tl)
                dv0 = -dn * nn - ratio * t0
                v0 = v0 + dv0
                v1 = v1 - (m0 / m1) * dv0
                flag[g] = 1
    node_y = gmin_y + float(j) * dx
    # the floor: k_grid_op's (every node below floor_y), or (floor_gate) the same gate as between the bodies, the floor a
    # body of infinite mass whose node volume is the part of the node's kernel below it: in contact when the field's
    # node volume and the floor's fill the cell, V_b >= dx^3 (1 - Phi((floor_y - y_i) / dx)) = dx^3 Phi((y_i - floor_y) / dx)
    fy0 = floor_y
    fy1 = floor_y
    if floor_gate != 0:
        fy0 = -1.0e9
        fy1 = -1.0e9
        h = (node_y - floor_y) / dx
        if h < 2.0:                                       # the floor's kernel share is non-zero (as m1 > 0 between bodies)
            need = cell_vol * bspline_cdf(h)
            if gvol0[g] >= need:
                fy0 = 1.0e9
            if gvol1[g] >= need:
                fy1 = 1.0e9
    if m0 > 1.0e-12:
        gvel0[g] = _boundary(v0, i, j, k, nx, ny, nz, wall_nodes, node_y, fy0, friction)
    else:
        gvel0[g] = wp.vec3(0.0, 0.0, 0.0)
    if m1 > 1.0e-12:
        gvel1[g] = _boundary(v1, i, j, k, nx, ny, nz, wall_nodes, node_y, fy1, friction)
    else:
        gvel1[g] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel(enable_backward=False)
def k_p2g_volume(x: wp.array(dtype=wp.vec3), J: wp.array(dtype=float), vp: float, gvol: wp.array(dtype=float),
                 gmin: wp.vec3, inv_dx: float, nx: int, ny: int, nz: int, dx: float):
    """Each particle's current volume (the sample's volume per particle x the tracked J) onto its 64 nodes: the gate
    only, never differentiated."""
    p = wp.tid()
    xp = x[p]
    if not K.valid_pos(xp):
        return
    b = K.base_node(xp, gmin, inv_dx)
    for oi in range(4):
        for oj in range(4):
            for ok in range(4):
                i = b[0] + oi
                j = b[1] + oj
                k = b[2] + ok
                if i >= 0 and i < nx and j >= 0 and j < ny and k >= 0 and k < nz:
                    xg = gmin + wp.vec3(float(i), float(j), float(k)) * dx
                    wp.atomic_add(gvol, K.gid(i, j, k, ny, nz), K.weight(xg - xp, inv_dx) * vp * J[p])


@wp.kernel(enable_backward=False)
def k_grid_moments(gm: wp.array(dtype=float), gvel: wp.array(dtype=wp.vec3), gmin: wp.vec3, dx: float,
                   ny: int, nz: int, out: wp.array(dtype=wp.vec3)):
    """out[0] += m v, out[1] += x x m v, out[2] += (m, m |v|^2 / 2, 0) of the field's nodes (after the grid op)."""
    g = wp.tid()
    m = gm[g]
    if m <= 1.0e-12:
        return
    v = gvel[g]
    xg = gmin + wp.vec3(float(g // (ny * nz)), float((g // nz) % ny), float(g % nz)) * dx
    wp.atomic_add(out, 0, m * v)
    wp.atomic_add(out, 1, wp.cross(xg, m * v))
    wp.atomic_add(out, 2, wp.vec3(m, 0.5 * m * wp.dot(v, v), 0.0))


@wp.kernel
def k_loss(v: wp.array(dtype=wp.vec3), n0: int, n1: int, out: wp.array(dtype=float)):
    """Body 1's centre-of-mass x velocity (the gradient check's loss)."""
    p = wp.tid()
    if p >= n0:
        wp.atomic_add(out, 0, v[p][0] / float(n1))


# ── the scene ──────────────────────────────────────────────────────────────────────────────────────────────────
class Scene:
    def __init__(self, a):
        dev = "cuda"
        self.a, self.dev = a, dev
        x1, vol = load_normalized("assets/isosphere.obj", a.n, a.seed, sample="stratified", return_volume=True)
        x2 = load_normalized("assets/isosphere.obj", a.n, a.seed + 50, sample="stratified")
        R = float(np.abs(x1).max())
        diag = float(np.linalg.norm(x1.max(0) - x1.min(0)))
        self.dx = dx = diag / 26.0                       # the shape's cell, as prepare
        self.R, self.vol_body, self.N0 = R, float(vol), a.n
        gap = a.gap                                     # surface gap at the start (wu)
        off = R + 0.5 * gap
        b = a.b * R                                     # impact parameter (y offset of body 1)
        c0, c1 = np.array([-off, -0.5 * b, 0.0]), np.array([off, 0.5 * b, 0.0])
        x = np.concatenate([x1 + c0, x2 + c1]).astype(np.float32)
        v = np.zeros_like(x)
        v[:a.n, 0], v[a.n:, 0] = a.v, -a.v
        N = len(x)
        lo = x.min(0) - (4.0 * R + 4 * dx)
        hi = x.max(0) + (4.0 * R + 4 * dx)
        if a.floor:
            lo[1] = x[:, 1].min() - a.drop - 3 * dx
        self.nxyz = tuple(int(np.ceil((hi[d] - lo[d]) / dx)) for d in range(3))
        self.gmin = wp.vec3(*lo.tolist())
        self.lo = lo
        self.ngrid = int(np.prod(self.nxyz))
        self.floor_y = float(x[:, 1].min()) - a.drop if a.floor else -1.0e9
        lam0, mu0 = lame(a.young, a.poisson)
        self.m_p = 40000.0 / a.n                          # each body is the canonical body (mass 40000 units)
        self.vp = self.vol_body / a.n                     # the sample's volume per particle
        A = lambda arr, dt: wp.array(arr, dtype=dt, device=dev)                 # noqa: E731
        I3 = np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))
        self.N = N
        self.x, self.v = A(x, wp.vec3), A(v, wp.vec3)
        self.C = wp.zeros(N, dtype=wp.mat33, device=dev)
        self.F, self.Fp, self.Fnew = A(I3, wp.mat33), A(I3, wp.mat33), A(I3, wp.mat33)
        self.dFc = wp.zeros(N, dtype=wp.mat33, device=dev)
        self.P = wp.zeros(N, dtype=wp.mat33, device=dev)
        self.m = wp.full(N, self.m_p, dtype=float, device=dev)
        self.lam = wp.full(N, lam0, dtype=float, device=dev)
        self.mu = wp.full(N, mu0, dtype=float, device=dev)
        self.J = wp.ones(N, dtype=float, device=dev)
        self.nbr0 = wp.zeros(1, dtype=int, device=dev)
        self.rest0 = wp.zeros(1, dtype=float, device=dev)
        self.frag0 = wp.zeros(N, dtype=float, device=dev)
        nf = 1 if a.mode == "single" else 2
        self.gm = [wp.zeros(self.ngrid, dtype=float, device=dev) for _ in range(nf)]
        self.gmom = [wp.zeros(self.ngrid, dtype=wp.vec3, device=dev) for _ in range(nf)]
        self.gvel = [wp.zeros(self.ngrid, dtype=wp.vec3, device=dev) for _ in range(nf)]
        self.gvol = [wp.zeros(self.ngrid, dtype=float, device=dev) for _ in range(2)]
        self.flag = wp.zeros(self.ngrid, dtype=int, device=dev)
        self.mom = wp.zeros(3, dtype=wp.vec3, device=dev)
        # rest volumes as the pipeline: MPM's estimate from the grid mass at the source (traj.compute_rest_volumes)
        self.vol = wp.zeros(N, dtype=float, device=dev)
        gm_tmp = wp.zeros(self.ngrid, dtype=float, device=dev)
        gv_tmp = wp.zeros(self.ngrid, dtype=wp.vec3, device=dev)
        nx, ny, nz = self.nxyz
        wp.launch(K.k_p2g, dim=N, inputs=[self.x, self.v, self.C, self.F, self.dFc, self.P, self.m, self.vol,
                                          self.nbr0, self.frag0, 0, gm_tmp, gv_tmp, self.gmin, dx, 1.0 / dx, 0.0, 0.0,
                                          nx, ny, nz], device=dev)
        wp.launch(K.k_volume, dim=N, inputs=[self.x, self.m, gm_tmp, self.vol, self.gmin, dx, 1.0 / dx, nx, ny, nz],
                  device=dev)
        self.sl = [slice(0, a.n), slice(a.n, N)]
        self.c_body = [c0, c1]
        lam_, mu_ = lam0, mu0
        rho = self.m_p / self.vp
        self.c_sound = float(np.sqrt((lam_ + 2 * mu_) / rho))
        print(f"[proto] N={N} ({a.n} per body), R={R:.3f} wu, dx={dx:.4f}, grid {self.nxyz}, rho={rho:.1f}, "
              f"c={self.c_sound:.2f} wu/s, v_rel/c={2 * a.v / self.c_sound:.3f}, mode={a.mode}, mu={a.mu}, "
              f"b={a.b}, gravity={a.g}, floor={a.floor}", flush=True)

    def _s(self, arr, b):
        s = self.sl[b]
        return arr[s.start:s.stop]

    def step(self, dfc=None, record=False):
        a, dev, dx = self.a, self.dev, self.dx
        nx, ny, nz = self.nxyz
        inv_dx = 1.0 / dx
        dfc = self.dFc if dfc is None else dfc
        wp.launch(K.k_stress_polar, dim=self.N, inputs=[self.F, dfc, self.Fp, self.lam, self.mu, self.P], device=dev)
        for f in range(len(self.gm)):
            self.gm[f].zero_()
            self.gmom[f].zero_()
        bodies = [(0, 0), (1, 0)] if a.mode == "single" else [(0, 0), (1, 1)]
        for b, f in bodies:
            S = lambda arr: self._s(arr, b)                                        # noqa: E731
            wp.launch(K.k_p2g, dim=a.n, inputs=[S(self.x), S(self.v), S(self.C), S(self.F), S(dfc), S(self.P),
                                                S(self.m), S(self.vol), self.nbr0, S(self.frag0), 0,
                                                self.gm[f], self.gmom[f], self.gmin, dx, inv_dx, a.dt, 0.0,
                                                nx, ny, nz], device=dev)
        fext = wp.vec3(0.0, -a.g, 0.0)
        if a.mode == "single":
            wp.launch(K.k_grid_op, dim=self.ngrid, inputs=[self.gm[0], self.gmom[0], self.gvel[0], a.dt, fext,
                                                          float(self.lo[1]), dx, nx, ny, nz, self.floor_y, a.mu,
                                                          K.WALL_NODES], device=dev)
        else:
            gate = 1 if a.mode == "multi" else 0
            if gate or a.floor_gate:
                for b in range(2):
                    self.gvol[b].zero_()
                    wp.launch(k_p2g_volume, dim=a.n, inputs=[self._s(self.x, b), self._s(self.J, b), self.vp,
                                                             self.gvol[b], self.gmin, inv_dx, nx, ny, nz, dx],
                              device=dev, record_tape=False)
            wp.launch(k_grid_op_contact, dim=self.ngrid,
                      inputs=[self.gm[0], self.gmom[0], self.gm[1], self.gmom[1], self.gvol[0], self.gvol[1],
                              self.gvel[0], self.gvel[1], a.dt, fext, float(self.lo[1]), dx, nx, ny, nz,
                              self.floor_y, a.mu, K.WALL_NODES, a.mu, dx ** 3, gate, self.flag, int(a.floor_gate)],
                      device=dev)
        for b, f in bodies:
            S = lambda arr: self._s(arr, b)                                        # noqa: E731
            wp.launch(K.k_g2p, dim=a.n, inputs=[S(self.x), S(self.v), S(self.C), S(self.F), S(dfc), S(self.Fnew),
                                                self.gvel[f], self.gmin, dx, inv_dx, a.dt, nx, ny, nz], device=dev)
        wp.launch(K.k_volume_update_motion, dim=self.N, inputs=[self.C, self.J, self.J, a.dt], device=dev)
        wp.launch(K.k_update, dim=self.N, inputs=[self.x, self.x, self.v, self.F, self.Fnew, self.F, a.dt, a.smoothing,
                                                  self.nbr0, self.rest0, self.frag0, 0, 0.0, self.nbr0, 0, self.rest0],
                  device=dev)
        wp.launch(K.k_volume_carry, dim=self.N, inputs=[self.F, self.J, self.F], device=dev)

    def moments(self):
        """Per field (body): momentum, angular momentum about the origin, mass and kinetic energy on the grid
        after the grid operation (what G2P hands to the particles)."""
        out = []
        nx, ny, nz = self.nxyz
        for f in range(len(self.gm)):
            self.mom.zero_()
            wp.launch(k_grid_moments, dim=self.ngrid, inputs=[self.gm[f], self.gvel[f], self.gmin, self.dx, ny, nz,
                                                               self.mom], device=self.dev)
            out.append(self.mom.numpy().astype(np.float64).copy())
        return out

    def particles(self):
        x = wp.to_torch(self.x)
        v = wp.to_torch(self.v)
        return x, v


def cross_gap(x, sl, sp):
    """Smallest distance from a body-0 particle to a body-1 particle (wu and particle spacings), and the counts of
    particles of each body within 1.5 spacings of the other body."""
    x0, x1 = x[sl[0]].contiguous(), x[sl[1]].contiguous()
    d01 = gpu.knn(x1, 1, queries=x0)[0][:, 0]
    d10 = gpu.knn(x0, 1, queries=x1)[0][:, 0]
    return float(d01.min()), float(d01.min()) / sp, int((d01 < 1.5 * sp).sum()), int((d10 < 1.5 * sp).sum())


def run(a):
    sc = Scene(a)
    sp = sc.vp ** (1.0 / 3.0)
    rec, first_contact = [], None
    t0 = time.time()
    for s in range(a.steps):
        sc.step()
        if s % a.every == 0 or s == a.steps - 1:
            wp.synchronize()
            x, v = sc.particles()
            mo = sc.moments()
            ncon = int(wp.to_torch(sc.flag).sum()) if a.mode != "single" else 0
            com = [x[sl].mean(0).double().cpu().numpy() for sl in sc.sl]
            vcm = [v[sl].mean(0).double().cpu().numpy() for sl in sc.sl]
            g, gs, n01, n10 = cross_gap(x, sc.sl, sp)
            P = sum(m[0] for m in mo)
            L = sum(m[1] for m in mo)
            KE = float(sum(m[2][1] for m in mo))
            Pb = [(sc.m_p * v[sl].double().sum(0)).cpu().numpy() for sl in sc.sl]
            r = {"step": s + 1, "t": (s + 1) * a.dt, "com0": com[0].tolist(), "com1": com[1].tolist(),
                 "v0": vcm[0].tolist(), "v1": vcm[1].tolist(), "P": P.tolist(), "L": L.tolist(), "KE": KE,
                 "P0": Pb[0].tolist(), "P1": Pb[1].tolist(), "contact_nodes": ncon, "gap": g, "gap_sp": gs,
                 "near01": n01, "near10": n10, "Jmin": float(wp.to_torch(sc.J).min()),
                 "low": [float(x[sl, 1].min()) - sc.floor_y for sl in sc.sl] if a.floor else None}
            rec.append(r)
            if ncon and first_contact is None:
                first_contact = r
            if s % (a.every * 10) == 0:
                print(f"[proto] step {s + 1:5d} t={r['t']:.3f}  v0x={vcm[0][0]:+.4f} v1x={vcm[1][0]:+.4f} "
                      f"P=({P[0]:+.3e},{P[1]:+.3e},{P[2]:+.3e}) L=({L[0]:+.3e},{L[1]:+.3e},{L[2]:+.3e}) "
                      f"KE={KE:.4g} contact={ncon} gap={g:.4f} wu ({gs:.2f} sp) near={n01}/{n10} "
                      f"Jmin={r['Jmin']:.3f}", flush=True)
    wall = time.time() - t0
    # the bounce: centre-of-mass velocities along the line of centres before and after
    pre, post = rec[0], rec[-1]
    nrm = np.asarray(post["com1"]) - np.asarray(post["com0"])
    nrm /= np.linalg.norm(nrm)
    n0 = np.array([1.0, 0.0, 0.0])
    vrel_pre = float(np.dot(np.asarray(pre["v1"]) - np.asarray(pre["v0"]), n0))
    vrel_post = float(np.dot(np.asarray(post["v1"]) - np.asarray(post["v0"]), nrm))
    P_all = np.array([r["P"] for r in rec])
    L_all = np.array([r["L"] for r in rec])
    t_all = np.array([r["t"] for r in rec])
    M = 2 * 40000.0
    P_pred = P_all[0][None, :] + np.outer(t_all - t_all[0], [0.0, -a.g * M, 0.0])
    summ = {"args": vars(a), "R": sc.R, "dx": sc.dx, "spacing": sp, "c": sc.c_sound, "wall_s": wall,
            "first_contact": first_contact, "end": post,
            "restitution": -vrel_post / vrel_pre if vrel_pre != 0 else None,
            "P_drift_max_rel": float(np.abs(P_all - P_pred).max() / max(np.abs(P_all[0]).max(), 1e-30)) if not a.floor else None,
            "L_drift_max": float(np.abs(L_all - L_all[0]).max()) if not a.floor and a.g == 0 else None,
            "L_scale": float(np.abs(L_all[0]).max()) if not a.floor and a.g == 0 else None,
            "contact_steps": int(sum(1 for r in rec if r["contact_nodes"] > 0)) * a.every,
            "end_near": [post["near01"], post["near10"]], "end_gap_sp": post["gap_sp"]}
    print(json.dumps({k: summ[k] for k in ("restitution", "P_drift_max_rel", "L_drift_max", "L_scale",
                                           "contact_steps", "end_near", "end_gap_sp", "wall_s")}), flush=True)
    if first_contact is not None:
        print(f"[proto] first contact at t={first_contact['t']:.3f}: gap {first_contact['gap']:.4f} wu = "
              f"{first_contact['gap_sp']:.2f} spacings", flush=True)
    with open(a.out + ".json", "w") as fh:
        json.dump({"summary": summ, "records": rec}, fh)
    return sc, summ


def gradcheck(a):
    """A taped window through the contact: a constant control dFc = eps * D on body 0 (D a fixed random symmetric
    field), L = body 1's centre-of-mass x velocity at the end. dL/d eps by the tape against central differences."""
    sc = Scene(a)
    # advance to just before the contact (no tape)
    while True:
        sc.step()
        x, _ = sc.particles()
        if cross_gap(x, sc.sl, sc.vp ** (1 / 3))[0] < 2.0 * sc.dx:
            break
    state = {k: wp.clone(getattr(sc, k)) for k in ("x", "v", "C", "F", "J")}
    rng = np.random.default_rng(1)
    D = rng.normal(size=(sc.N, 3, 3)).astype(np.float32)
    D = 0.5 * (D + D.transpose(0, 2, 1))
    D[a.n:] = 0.0
    D_t = torch.tensor(D, device="cuda")

    def L_of(eps, tape=None):
        for k_, v_ in state.items():
            wp.copy(getattr(sc, k_), v_)
        dfc_t = (eps * D_t).contiguous()
        dfc = wp.from_torch(dfc_t, dtype=wp.mat33, requires_grad=tape is not None)
        for _ in range(a.grad_steps):
            sc.step(dfc)
        v = wp.to_torch(sc.v)
        return float(v[a.n:, 0].double().mean()), dfc

    # the taped value needs per-step buffers (in-place updates break the adjoint): a functional replay
    eps0, h = a.grad_eps, a.grad_h
    Lp, _ = L_of(eps0 + h)
    Lm, _ = L_of(eps0 - h)
    fd = (Lp - Lm) / (2 * h)
    ad = adjoint_dL(sc, state, D_t, eps0, a)
    print(f"[grad] dL/deps: tape {ad:.6e}  central FD (h={h:g}) {fd:.6e}  rel diff {abs(ad - fd) / max(abs(fd), 1e-30):.3e}",
          flush=True)
    return ad, fd


def adjoint_dL(sc, state, D_t, eps0, a):
    """The tape: per-step buffers (functional), the kernels' generated adjoints."""
    dev, dx, N, n = sc.dev, sc.dx, sc.N, a.n
    nx, ny, nz = sc.nxyz
    inv_dx = 1.0 / dx
    T = a.grad_steps
    Z = lambda dt: wp.zeros(N, dtype=dt, device=dev, requires_grad=True)       # noqa: E731
    xs = [wp.clone(state["x"], requires_grad=True)] + [Z(wp.vec3) for _ in range(T)]
    vs = [wp.clone(state["v"], requires_grad=True)] + [Z(wp.vec3) for _ in range(T)]
    Cs = [wp.clone(state["C"], requires_grad=True)] + [Z(wp.mat33) for _ in range(T)]
    Fs = [wp.clone(state["F"], requires_grad=True)] + [Z(wp.mat33) for _ in range(T)]
    Fb = [Z(wp.mat33) for _ in range(T + 1)]
    Fr = [Z(wp.mat33) for _ in range(T + 1)]
    Js = [wp.clone(state["J"], requires_grad=True)] + [wp.zeros(N, dtype=float, device=dev, requires_grad=True)
                                                       for _ in range(T)]
    Ps = [Z(wp.mat33) for _ in range(T)]
    ng = sc.ngrid
    G = lambda dt: [wp.zeros(ng, dtype=dt, device=dev, requires_grad=True) for _ in range(2)]   # noqa: E731
    gms = [G(float) for _ in range(T)]
    gmoms = [G(wp.vec3) for _ in range(T)]
    gvels = [G(wp.vec3) for _ in range(T)]
    # per step: the generated adjoint of the contact kernel replays its forward body, gate included, at backward time
    gvols = [[wp.zeros(ng, dtype=float, device=dev) for _ in range(2)] for _ in range(T)]
    dfc_t = (eps0 * D_t).contiguous().requires_grad_(False)
    dfc = wp.from_torch(dfc_t.clone(), dtype=wp.mat33, requires_grad=True)
    fext = wp.vec3(0.0, -a.g, 0.0)
    flag = wp.zeros(ng, dtype=int, device=dev)
    loss = wp.zeros(1, dtype=float, device=dev, requires_grad=True)

    S = lambda arr, b: arr[sc.sl[b].start:sc.sl[b].stop]                       # noqa: E731
    tape = wp.Tape()
    with tape:
        for t in range(T):
            wp.launch(K.k_stress_polar, dim=N, inputs=[Fs[t], dfc, sc.Fp, sc.lam, sc.mu, Ps[t]], device=dev)
            for b in range(2):
                wp.launch(K.k_p2g, dim=n, inputs=[S(xs[t], b), S(vs[t], b), S(Cs[t], b), S(Fs[t], b), S(dfc, b),
                                                  S(Ps[t], b), S(sc.m, b), S(sc.vol, b), sc.nbr0, S(sc.frag0, b), 0,
                                                  gms[t][b], gmoms[t][b], sc.gmin, dx, inv_dx, a.dt, 0.0, nx, ny, nz],
                          device=dev)
                gvol = gvols[t]
                wp.launch(k_p2g_volume, dim=n, inputs=[S(xs[t], b), S(Js[t], b), sc.vp, gvol[b], sc.gmin, inv_dx,
                                                       nx, ny, nz, dx], device=dev, record_tape=False)
            wp.launch(k_grid_op_contact, dim=ng,
                      inputs=[gms[t][0], gmoms[t][0], gms[t][1], gmoms[t][1], gvol[0], gvol[1], gvels[t][0],
                              gvels[t][1], a.dt, fext, float(sc.lo[1]), dx, nx, ny, nz, sc.floor_y, a.mu,
                              K.WALL_NODES, a.mu, dx ** 3, 1, flag, int(a.floor_gate)], device=dev)
            for b in range(2):
                wp.launch(K.k_g2p, dim=n, inputs=[S(xs[t], b), S(vs[t + 1], b), S(Cs[t + 1], b), S(Fs[t], b),
                                                  S(dfc, b), S(Fr[t + 1], b), gvels[t][b], sc.gmin, dx, inv_dx, a.dt,
                                                  nx, ny, nz], device=dev)
            wp.launch(K.k_volume_update_motion, dim=N, inputs=[Cs[t + 1], Js[t], Js[t + 1], a.dt], device=dev)
            wp.launch(K.k_update, dim=N, inputs=[xs[t], xs[t + 1], vs[t + 1], Fs[t], Fr[t + 1], Fb[t + 1], a.dt,
                                                 a.smoothing, sc.nbr0, sc.rest0, sc.frag0, 0, 0.0, sc.nbr0, 0,
                                                 sc.rest0], device=dev)
            wp.launch(K.k_volume_carry, dim=N, inputs=[Fb[t + 1], Js[t + 1], Fs[t + 1]], device=dev)
        wp.launch(k_loss, dim=N, inputs=[vs[T], n, N - n, loss], device=dev)
    tape.backward(loss=loss)
    g = wp.to_torch(dfc.grad).double()
    return float((g * D_t.double()).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000, help="particles per body")
    ap.add_argument("--seed", type=int, default=97)
    ap.add_argument("--mode", default="multi", choices=["multi", "multi_nogate", "single"])
    ap.add_argument("--v", type=float, default=0.65, help="each body's speed toward the other (wu/s)")
    ap.add_argument("--b", type=float, default=0.0, help="impact parameter in radii (0 = head-on)")
    ap.add_argument("--mu", type=float, default=0.0, help="Coulomb friction between the bodies (and the floor)")
    ap.add_argument("--gap", type=float, default=0.5, help="surface gap at the start (wu)")
    ap.add_argument("--g", type=float, default=0.0, help="gravity (wu/s^2)")
    ap.add_argument("--floor", action="store_true")
    ap.add_argument("--drop", type=float, default=0.0, help="the floor this far below the lowest particle (wu)")
    ap.add_argument("--floor_gate", type=int, default=0, help="1: the floor by the bodies' volume gate")
    ap.add_argument("--young", type=float, default=1.4e5)
    ap.add_argument("--poisson", type=float, default=0.2)
    ap.add_argument("--smoothing", type=float, default=0.955)
    ap.add_argument("--dt", type=float, default=1.0 / 240.0)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--out", default="d147_proto")
    ap.add_argument("--gradcheck", action="store_true")
    ap.add_argument("--grad_steps", type=int, default=40)
    ap.add_argument("--grad_eps", type=float, default=0.0)
    ap.add_argument("--grad_h", type=float, default=1e-3)
    a = ap.parse_args()
    wp.init()
    if a.gradcheck:
        gradcheck(a)
    else:
        run(a)


if __name__ == "__main__":
    main()
