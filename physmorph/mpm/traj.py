"""Differentiable MPM rollout via per-step state arrays (wp.Tape adjoint).

Each timestep reads state t and writes state t+1, so the tape retains every
intermediate for the reverse pass. Fresh Trajectory per forward => clean tape.
See docs/SPEC.md §4.2.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from . import kernels as K
from .state import MPMParams, make_state
from .step import gate_omega, nominal_support


def _id(N):
    return np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))


_ID_CACHE: dict = {}


def _id_dev(N: int, device: str):
    """Cached device identity (N,3,3): per-step F arrays are cloned from it on the device
    instead of being copied from a fresh numpy identity each time (2026-09-16 profile:
    host->device array construction was 40 % of a window)."""
    key = (N, str(device))
    a = _ID_CACHE.get(key)
    if a is None:
        a = wp.array(_id(N), dtype=wp.mat33, device=device)
        _ID_CACHE[key] = a
    return a


def compute_rest_volumes(x0, m, prm: MPMParams, device="cuda") -> np.ndarray:
    """Compute the reference particle volumes once, at the sampled source state.

    ``V_p0`` is material data, not rollout state: callers should keep this returned
    array and pass it to every subsequent :class:`Trajectory`.  The compatibility
    fallback in ``Trajectory(vol0=None)`` still computes from that trajectory's
    ``x0``, but the pipeline must not use the fallback after the source is deformed.
    """
    x0 = np.ascontiguousarray(x0, np.float32)
    if x0.ndim != 2 or x0.shape[1] != 3:
        raise ValueError(f"x0 must have shape (N,3), got {x0.shape}")
    # Reuse the canonical one-time estimator rather than duplicating its P2G
    # discretisation here.  Zero Lamé parameters are sufficient: dt=0 makes the
    # volume pass mass-only.
    from .step import compute_volumes
    state = make_state(x0, m, 0.0, 0.0, prm, device=device, requires_grad=False)
    compute_volumes(state, prm)
    vol0 = np.ascontiguousarray(state.vol.numpy(), np.float32)
    if not np.isfinite(vol0).all() or (vol0 < 0.0).any():
        raise RuntimeError("rest-volume estimation produced invalid Vp0")
    return vol0


_WARMED: set = set()          # devices whose kernels were launched once outside a capture


class Trajectory:
    def __init__(self, x0, m, lam, mu, prm: MPMParams, T: int,
                 Fp=None, v0=None, F0=None, C0=None, dFc=None, eta=None,
                 device="cuda", requires_grad=True, mat_grad=False, vol0=None,
                 Fg0=None, track_geom=False, bonds=None, persistent=False, layer=None, layer_u=None):
        x0 = np.ascontiguousarray(x0, np.float32)
        # PERSISTENT: the buffers are rolled out many times (line-search candidates); the
        # accumulated grid arrays are re-zeroed per step and the rollout can be recorded as
        # a CUDA graph (capture/run) — the same kernels with no Python launch overhead.
        self.persistent = bool(persistent)
        self.graph = None
        N = x0.shape[0]
        self.N, self.T, self.prm, self.device = N, T, prm, device
        rg = requires_grad

        def A(a, dt, g=False):
            return wp.array(np.ascontiguousarray(a), dtype=dt, device=device, requires_grad=g)

        def Z(dt, g=False):                    # device-side zeros: no host copy
            return wp.zeros(N, dtype=dt, device=device, requires_grad=g)

        def ID(g=False):                       # device-side identity clone
            return wp.clone(_id_dev(N, device), requires_grad=g)

        # CONTROL FIELD. Two modes, matching the two formulations:
        #   * a single array  -> ONE dFc shared by every step (the greedy/per-frame scheme)
        #   * a list of T arrays -> dFc[t], a control SEQUENCE, which is what the C++
        #     CompGraph optimises (one dFc per layer, never reset between layers).
        # `_dfc(t)` hides the difference from step().
        if dFc is None:
            self.dFc = A(np.zeros((N, 3, 3), np.float32), wp.mat33, rg)
            self.dFc_seq = None
        elif isinstance(dFc, (list, tuple)):
            assert len(dFc) == T, f"dFc sequence must have T={T} entries, got {len(dFc)}"
            self.dFc_seq = list(dFc)
            self.dFc = self.dFc_seq[0]          # volume precompute uses the t=0 control
        else:
            self.dFc = dFc
            self.dFc_seq = None
        # shared, non-differentiated
        m_a = np.broadcast_to(m, (N,)).astype(np.float32)
        self.m = A(m_a, wp.float32)

        # material: scalar / numpy -> constant array; a wp.array passes through UNCHANGED so the
        # torch bridge can hand in from_torch leaves (dL/d(lam,mu) flows back through the tape).
        def M(val, default):
            if isinstance(val, wp.array):
                return val
            if val is None:
                val = default
            a = np.full(N, float(val), np.float32) if np.isscalar(val) else val
            return A(a, wp.float32, mat_grad)

        self.lam = M(lam, 0.0)
        self.mu = M(mu, 0.0)
        self.eta = M(eta, 0.0)
        self.Fp = A(_id(N) if Fp is None else Fp, wp.mat33)
        if vol0 is None:
            vol_a = np.zeros(N, np.float32)
        else:
            vol_a = np.ascontiguousarray(vol0, np.float32)
            if vol_a.shape != (N,):
                raise ValueError(f"vol0 must have shape ({N},), got {vol_a.shape}")
            if not np.isfinite(vol_a).all() or (vol_a < 0.0).any():
                raise ValueError("vol0 must be finite and non-negative")
        self.vol = A(vol_a, wp.float32)
        # per-step trajectory
        F0a = _id(N) if F0 is None else F0
        v0a = np.zeros((N, 3), np.float32) if v0 is None else v0
        self.x = [A(x0, wp.vec3, rg) if t == 0 else Z(wp.vec3, rg) for t in range(T + 1)]
        self.v = [A(v0a, wp.vec3, rg) if t == 0 else Z(wp.vec3, rg) for t in range(T + 1)]
        # C[0] must be settable: the APIC affine field is part of the state. Dropping it when a
        # trajectory is restarted from a promoted state silently discards momentum content and
        # leaves an elastically loaded body frozen -> stored energy is re-released every restart.
        C0a = np.zeros((N, 3, 3), np.float32) if C0 is None else C0
        self.C = [A(C0a, wp.mat33, rg) if t == 0 else Z(wp.mat33, rg) for t in range(T + 1)]
        self.F = [A(F0a, wp.mat33, rg) if t == 0 else ID(rg) for t in range(T + 1)]
        self.Fraw = [ID(rg) for t in range(T + 1)]
        # GEOMETRIC deformation gradient (render kinematics; kernels.k_geom_update):
        # transported by the velocity gradient only, no control, no smoothing. Optional
        # so the physics-only paths pay nothing for it.
        self.track_geom = bool(track_geom)
        if self.track_geom:
            Fg0a = _id(N) if Fg0 is None else np.ascontiguousarray(Fg0, np.float32)
            if Fg0a.shape != (N, 3, 3):
                raise ValueError(f"Fg0 must have shape ({N},3,3), got {Fg0a.shape}")
            self.Fg = [A(Fg0a, wp.mat33, rg) if t == 0 else ID(rg) for t in range(T + 1)]
        else:
            self.Fg = None
        self.P = [Z(wp.mat33, rg) for t in range(T)]
        # GRID ARRAYS: the adjoint needs every step's grid (k_grid_op / k_g2p read them in
        # the backward pass), a forward-only rollout does not — step t+1 never reads step
        # t's grid, so a no-grad trajectory shares ONE set and re-zeroes it per step
        # (150k on a 233^3 grid: 7.4 GB -> 0.65 GB for the line-search trajectory).
        self.share_grid = not rg
        n_grid = 1 if self.share_grid else T
        gm_l = [wp.zeros(prm.ngrid, dtype=wp.float32, device=device, requires_grad=rg) for t in range(n_grid)]
        gmom_l = [wp.zeros(prm.ngrid, dtype=wp.vec3, device=device, requires_grad=rg) for t in range(n_grid)]
        gvel_l = [wp.zeros(prm.ngrid, dtype=wp.vec3, device=device, requires_grad=rg) for t in range(n_grid)]
        self.gm = [gm_l[t % n_grid] for t in range(T)]
        self.gmom = [gmom_l[t % n_grid] for t in range(T)]
        self.gvel = [gvel_l[t % n_grid] for t in range(T)]
        # SUPPORT-GATED APIC (kernels.k_support_gate; docs/thin_feature_transport.md §3):
        # omega_t[p] scales the affine term m*C in P2G at step t. Piecewise constant in x,
        # so it is computed OUTSIDE the tape per step and read by the adjoint as a constant.
        # n0 (nominal 3^3 count) is fixed by the caller (prm.gate_n0) so every window uses
        # the same gate; the fallback measures it on this trajectory's x0.
        self.omega1 = A(np.ones(N, np.float32), wp.float32)
        # MATERIAL RE-COUPLING of decoupled particles (kernels.k_p2g / k_update): bonds =
        # (nbr (N,K) int, rest (N,K) float) frozen for this rollout; the decoupling test is
        # the 3^3-cell count of the support-gate kernels (outside the tape, piecewise const).
        self.bonds = None
        self.bond_K = 0
        self.nbr0 = wp.zeros(1, dtype=wp.int32, device=device)
        self.rest0 = wp.zeros(1, dtype=wp.float32, device=device)
        self.ncount0 = wp.zeros(N, dtype=wp.float32, device=device)
        if bonds is not None:
            nbr, rest, frag = bonds                  # frag: (N,) 1.0 = fragment particle
            nbr = np.ascontiguousarray(nbr, np.int32)
            self.bond_K = int(nbr.shape[1])
            self.bond_nbr = wp.array(nbr.reshape(-1), dtype=wp.int32, device=device)
            self.bond_rest = wp.array(np.ascontiguousarray(rest, np.float32).reshape(-1), dtype=wp.float32, device=device)
            self.bond_frag = wp.array(np.ascontiguousarray(frag, np.float32), dtype=wp.float32, device=device)
            self.bonds = True
            # per-step decoupling test (docs/method.md 10.7): the 3^3-cell count of the current
            # state, outside the tape (piecewise constant), OR-ed with the commit-time mask
            self.cnt_b = wp.zeros(prm.ngrid, dtype=wp.int32, device=device)
            self.ncount_b = wp.zeros(N, dtype=wp.float32, device=device)
            self.omega_b = wp.array(np.ones(N, np.float32), dtype=wp.float32, device=device)
            self.frag_step = wp.zeros(N, dtype=wp.float32, device=device)
        # OUTER-LAYER RELAXATION (kernels.k_layer_resid / k_layer_project; docs/surface_gradient.md
        # §6): layer = (mask (N,), nrm (N,3), nbr (N,K), w (N,K), frac) frozen for this rollout.
        # k_update writes the advected positions into xu[t+1]; the projection writes x[t+1].
        self.layer = None
        if layer is not None:
            lmask, lnrm, lnbr, lw, lfrac = layer
            self.layer_K = int(np.asarray(lnbr).shape[1])
            self.layer_mask = A(np.ascontiguousarray(lmask, np.float32), wp.float32)
            self.layer_nrm = wp.array(np.ascontiguousarray(lnrm, np.float32), dtype=wp.vec3, device=device)
            self.layer_nbr = wp.array(np.ascontiguousarray(lnbr, np.int32).reshape(-1), dtype=wp.int32, device=device)
            self.layer_w = wp.array(np.ascontiguousarray(lw, np.float32).reshape(-1), dtype=wp.float32, device=device)
            self.layer_frac = float(lfrac)
            self.xu = [wp.zeros(N, dtype=wp.vec3, device=device, requires_grad=rg) for t in range(T + 1)]
            self.ld = [wp.zeros(N, dtype=wp.float32, device=device, requires_grad=rg) for t in range(T + 1)]
            # the position-mode control leaf u (N,): a warp view of the caller's tensor when
            # given (layer_u), else a zero buffer the eval path assigns into
            self.layer_u = layer_u if layer_u is not None else wp.zeros(N, dtype=wp.float32, device=device, requires_grad=rg)
            self.layer_frac_u = 1.0 / float(T)
            self.layer = True
        self.gate = bool(prm.gate_r_hi > prm.gate_r_lo)
        if self.gate:
            self.cnt = wp.zeros(prm.ngrid, dtype=wp.int32, device=device)
            self.ncount = A(np.zeros(N, np.float32), wp.float32)
            self.omega = [A(np.ones(N, np.float32), wp.float32) for t in range(T)]
            self.gate_n0 = float(prm.gate_n0) if prm.gate_n0 > 0 else nominal_support(x0, prm, device)
        if vol0 is None:
            # Backward-compatible single-rollout fallback.  Production callers
            # must compute Vp0 at source initialisation and reuse it explicitly.
            self._compute_volumes()

    def _compute_volumes(self):
        prm, dev, N = self.prm, self.device, self.N
        inv_dx, gmin = 1.0 / prm.dx, wp.vec3(*prm.grid_min)
        gm = wp.zeros(prm.ngrid, dtype=float, device=dev)
        gv = wp.zeros(prm.ngrid, dtype=wp.vec3, device=dev)
        P0 = wp.zeros(N, dtype=wp.mat33, device=dev)
        v0 = wp.zeros(N, dtype=wp.vec3, device=dev)
        C0 = wp.zeros(N, dtype=wp.mat33, device=dev)
        wp.launch(K.k_stress, dim=N, inputs=[self.F[0], self.dFc, self.Fp, self.lam, self.mu, P0], device=dev)
        wp.launch(K.k_p2g, dim=N, inputs=[self.x[0], v0, C0, self.F[0], self.dFc, P0, self.m, self.vol,
                  self.omega1, self.nbr0, self.ncount0, 0, gm, gv, gmin, prm.dx, inv_dx, 0.0, 0.0, prm.nx, prm.ny, prm.nz], device=dev)
        wp.launch(K.k_volume, dim=N, inputs=[self.x[0], self.m, gm, self.vol, gmin, prm.dx, inv_dx,
                  prm.nx, prm.ny, prm.nz], device=dev)

    def _omega(self, t: int):
        """Affine-transfer gate for step t (all ones unless the support gate is on)."""
        if not self.gate:
            return self.omega1
        gate_omega(self.x[t], self.prm, self.gate_n0, self.omega[t], self.ncount, self.cnt)
        return self.omega[t]

    def _bond_args(self, t: int):
        """(nbr, rest, frag, K) for step t; K = 0 (placeholders) unless bonds are attached.
        The decoupling flag is re-evaluated EVERY step from the current state (the 3^3-cell
        count, outside the tape) and OR-ed with the runner's commit-time fragment mask, so a
        particle that clears the fracture gap mid-window is bonded at once, not a window later."""
        if not self.bonds:
            return self.nbr0, self.rest0, self.ncount0, 0
        gate_omega(self.x[t], self.prm, 1.0, self.omega_b, self.ncount_b, self.cnt_b)
        wp.launch(K.k_frag_step, dim=self.N, inputs=[self.ncount_b, self.bond_frag, self.frag_step],
                  device=self.device)
        return self.bond_nbr, self.bond_rest, self.frag_step, self.bond_K

    def _dfc(self, t: int):
        """Control at step t: dFc[t] for a sequence, the shared field otherwise."""
        return self.dFc if self.dFc_seq is None else self.dFc_seq[t]

    def step(self, t: int):
        prm, dev, N = self.prm, self.device, self.N
        inv_dx, gmin, fext = 1.0 / prm.dx, wp.vec3(*prm.grid_min), wp.vec3(*prm.f_ext)
        dfc = self._dfc(t)
        bnb, brest, bnc, bK = self._bond_args(t)
        if self.share_grid or self.persistent:       # P2G accumulates: fresh grid per step
            self.gm[t].zero_()
            self.gmom[t].zero_()
        wp.launch(K.k_stress, dim=N, inputs=[self.F[t], dfc, self.Fp, self.lam, self.mu, self.P[t]], device=dev)
        wp.launch(K.k_p2g, dim=N, inputs=[self.x[t], self.v[t], self.C[t], self.F[t], dfc, self.P[t],
                  self.m, self.vol, self._omega(t), bnb, bnc, bK, self.gm[t], self.gmom[t], gmin, prm.dx, inv_dx,
                  prm.dt, prm.drag,
                  prm.nx, prm.ny, prm.nz], device=dev)
        wp.launch(K.k_grid_op, dim=prm.ngrid, inputs=[self.gm[t], self.gmom[t], self.gvel[t], prm.dt, fext,
                  prm.grid_min[1], prm.dx, prm.nx, prm.ny, prm.nz, prm.floor_y, prm.floor_friction,
                  K.WALL_NODES], device=dev)
        wp.launch(K.k_g2p, dim=N, inputs=[self.x[t], self.v[t + 1], self.C[t + 1], self.F[t], dfc,
                  self.Fraw[t + 1], self.gvel[t], self.eta, gmin, prm.dx, inv_dx, prm.dt, prm.nx, prm.ny, prm.nz,
                  prm.v_max, prm.eta_sym, prm.eta_mode], device=dev)
        x_next = self.xu[t + 1] if self.layer else self.x[t + 1]
        wp.launch(K.k_update, dim=N, inputs=[self.x[t], x_next, self.v[t + 1], self.F[t],
                  self.Fraw[t + 1], self.F[t + 1], prm.dt, prm.smoothing,
                  bnb, brest, bnc, bK, 1.0 / float(self.T)], device=dev)
        if self.layer:
            wp.launch(K.k_layer_resid, dim=N, inputs=[self.xu[t + 1], self.layer_mask, self.layer_nrm,
                      self.layer_nbr, self.layer_w, self.layer_K, self.ld[t + 1]], device=dev)
            wp.launch(K.k_layer_project, dim=N, inputs=[self.xu[t + 1], self.ld[t + 1], self.layer_mask,
                      self.layer_nrm, self.layer_nbr, self.layer_w, self.layer_K, self.layer_frac,
                      self.layer_u, self.layer_frac_u, self.x[t + 1]], device=dev)
        if self.track_geom:
            wp.launch(K.k_geom_update, dim=N, inputs=[self.C[t + 1], self.Fg[t], self.Fg[t + 1],
                      prm.dt], device=dev)

    def rollout(self):
        for t in range(self.T):
            self.step(t)
        return self.x[self.T], self.F[self.T]

    def capture(self) -> bool:
        """Record the rollout as a CUDA graph (persistent trajectories on a CUDA device).
        Every buffer the graph touches lives on this object, so a replay is exactly the
        rollout of the CURRENT contents of x[0], v[0], C[0], F[0], Fg[0], the material and
        the control sequence. Modules are warmed with one plain rollout first (module
        loading is not allowed inside a capture)."""
        if not self.persistent or not str(self.device).startswith("cuda"):
            return False
        key = str(self.device)
        if key not in _WARMED:
            self.rollout()
            wp.synchronize_device(self.device)
            _WARMED.add(key)
        with wp.ScopedCapture(device=self.device) as cap:
            self.rollout()
        self.graph = cap.graph
        return True

    def run(self):
        """Roll out: replay the captured graph when there is one, else launch the kernels."""
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.rollout()
        return self.x[self.T], self.F[self.T]
