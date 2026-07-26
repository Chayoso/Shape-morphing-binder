"""Differentiable MPM rollout via per-step state arrays (wp.Tape adjoint).

Each timestep reads state t and writes state t+1, so the tape retains every
intermediate for the reverse pass. Fresh Trajectory per forward => clean tape.
See docs/SPEC.md §4.2.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from . import kernels as K
from .state import MPMParams


def _id(N):
    return np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))


class Trajectory:
    def __init__(self, x0, m, lam, mu, prm: MPMParams, T: int,
                 Fp=None, v0=None, F0=None, C0=None, dFc=None, eta=None,
                 device="cuda", requires_grad=True, mat_grad=False):
        x0 = np.ascontiguousarray(x0, np.float32)
        N = x0.shape[0]
        self.N, self.T, self.prm, self.device = N, T, prm, device
        rg = requires_grad

        def A(a, dt, g=False):
            return wp.array(np.ascontiguousarray(a), dtype=dt, device=device, requires_grad=g)

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
        # material: optionally a DIFFERENTIABLE leaf (MatCast: dL/d(lam,mu) from rendered motion)
        self.lam = A(np.full(N, float(lam), np.float32) if np.isscalar(lam) else lam, wp.float32, mat_grad)
        self.mu = A(np.full(N, float(mu), np.float32) if np.isscalar(mu) else mu, wp.float32, mat_grad)
        self.eta = A(np.zeros(N, np.float32) if eta is None else
                     (np.full(N, float(eta), np.float32) if np.isscalar(eta) else eta), wp.float32, mat_grad)
        self.Fp = A(_id(N) if Fp is None else Fp, wp.mat33)
        self.vol = A(np.zeros(N, np.float32), wp.float32)
        # per-step trajectory
        F0a = _id(N) if F0 is None else F0
        v0a = np.zeros((N, 3), np.float32) if v0 is None else v0
        self.x = [A(x0 if t == 0 else np.zeros((N, 3), np.float32), wp.vec3, rg) for t in range(T + 1)]
        self.v = [A(v0a if t == 0 else np.zeros((N, 3), np.float32), wp.vec3, rg) for t in range(T + 1)]
        # C[0] must be settable: the APIC affine field is part of the state. Dropping it when a
        # trajectory is restarted from a promoted state silently discards momentum content and
        # leaves an elastically loaded body frozen -> stored energy is re-released every restart.
        C0a = np.zeros((N, 3, 3), np.float32) if C0 is None else C0
        self.C = [A(C0a if t == 0 else np.zeros((N, 3, 3), np.float32), wp.mat33, rg)
                  for t in range(T + 1)]
        self.F = [A(F0a if t == 0 else _id(N), wp.mat33, rg) for t in range(T + 1)]
        self.Fraw = [A(_id(N), wp.mat33, rg) for t in range(T + 1)]
        self.P = [A(np.zeros((N, 3, 3), np.float32), wp.mat33, rg) for t in range(T)]
        self.gm = [A(np.zeros(prm.ngrid, np.float32), wp.float32, rg) for t in range(T)]
        self.gmom = [A(np.zeros((prm.ngrid, 3), np.float32), wp.vec3, rg) for t in range(T)]
        self.gvel = [A(np.zeros((prm.ngrid, 3), np.float32), wp.vec3, rg) for t in range(T)]
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
                  gm, gv, gmin, prm.dx, inv_dx, 0.0, 0.0, prm.nx, prm.ny, prm.nz], device=dev)
        wp.launch(K.k_volume, dim=N, inputs=[self.x[0], self.m, gm, self.vol, gmin, prm.dx, inv_dx,
                  prm.nx, prm.ny, prm.nz], device=dev)

    def _dfc(self, t: int):
        """Control at step t: dFc[t] for a sequence, the shared field otherwise."""
        return self.dFc if self.dFc_seq is None else self.dFc_seq[t]

    def step(self, t: int):
        prm, dev, N = self.prm, self.device, self.N
        inv_dx, gmin, fext = 1.0 / prm.dx, wp.vec3(*prm.grid_min), wp.vec3(*prm.f_ext)
        dfc = self._dfc(t)
        wp.launch(K.k_stress, dim=N, inputs=[self.F[t], dfc, self.Fp, self.lam, self.mu, self.P[t]], device=dev)
        wp.launch(K.k_p2g, dim=N, inputs=[self.x[t], self.v[t], self.C[t], self.F[t], dfc, self.P[t],
                  self.m, self.vol, self.gm[t], self.gmom[t], gmin, prm.dx, inv_dx, prm.dt, prm.drag,
                  prm.nx, prm.ny, prm.nz], device=dev)
        wp.launch(K.k_grid_op, dim=prm.ngrid, inputs=[self.gm[t], self.gmom[t], self.gvel[t], prm.dt, fext,
                  prm.grid_min[1], prm.dx, prm.ny, prm.nz, prm.floor_y, prm.floor_friction], device=dev)
        wp.launch(K.k_g2p, dim=N, inputs=[self.x[t], self.v[t + 1], self.C[t + 1], self.F[t], dfc,
                  self.Fraw[t + 1], self.gvel[t], self.eta, gmin, prm.dx, inv_dx, prm.dt, prm.nx, prm.ny, prm.nz,
                  prm.v_max, prm.eta_sym, prm.eta_mode], device=dev)
        wp.launch(K.k_update, dim=N, inputs=[self.x[t], self.x[t + 1], self.v[t + 1], self.F[t],
                  self.Fraw[t + 1], self.F[t + 1], prm.dt, prm.smoothing], device=dev)

    def rollout(self):
        for t in range(self.T):
            self.step(t)
        return self.x[self.T], self.F[self.T]
