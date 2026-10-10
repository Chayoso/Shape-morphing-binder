"""MPM step orchestration. One frame = T x mpm_step. See docs/SPEC.md §3.3."""
from __future__ import annotations

import warp as wp

from . import kernels as K
from .state import MPMParams, MPMState


def _reset_grid(s: MPMState, prm: MPMParams):
    wp.launch(K.k_zero_scalar, dim=prm.ngrid, inputs=[s.grid_m], device=s.device)
    wp.launch(K.k_zero_vec, dim=prm.ngrid, inputs=[s.grid_v], device=s.device)


_NOBOND = {}


def _nobond(N: int, device: str):
    """(nbr, rest, ncount) placeholders for launches without material re-coupling (K=0)."""
    a = _NOBOND.get((N, device))
    if a is None:
        a = (wp.zeros(1, dtype=wp.int32, device=device), wp.zeros(1, dtype=wp.float32, device=device),
             wp.zeros(N, dtype=wp.float32, device=device))
        _NOBOND[(N, device)] = a
    return a


def cell_counts(x: wp.array, prm: MPMParams, ncount: wp.array, cnt: wp.array, record_tape: bool = False):
    """ncount: the particles in the 3^3 cells around each particle's own (kernels.k_neighbour_count) for positions x;
    cnt is int32 scratch of size ngrid (zeroed here). The material bonds' decoupling test (kernels.k_frag_step)."""
    gmin, inv_dx = wp.vec3(*prm.grid_min), 1.0 / prm.dx
    cnt.zero_()
    wp.launch(K.k_cell_count, dim=x.shape[0], inputs=[x, gmin, inv_dx, prm.nx, prm.ny, prm.nz, cnt],
              device=x.device, record_tape=record_tape)
    wp.launch(K.k_neighbour_count, dim=x.shape[0], inputs=[x, cnt, gmin, inv_dx, prm.nx, prm.ny, prm.nz, ncount],
              device=x.device, record_tape=record_tape)


def mpm_step(s: MPMState, prm: MPMParams):
    """One MLS-MPM timestep: stress -> P2G -> grid -> G2P -> update. eq (3')-(9)."""
    inv_dx = 1.0 / prm.dx
    gmin = wp.vec3(*prm.grid_min)
    fext = wp.vec3(*prm.f_ext)
    N, dev = s.N, s.device

    wp.launch(K.k_stress, dim=N, inputs=[s.F, s.dFc, s.Fp, s.lam, s.mu, s.P], device=dev)
    _reset_grid(s, prm)
    nb0 = _nobond(N, dev)
    wp.launch(K.k_p2g, dim=N,
              inputs=[s.x, s.v, s.C, s.F, s.dFc, s.P, s.m, s.vol, nb0[0], nb0[2], 0, s.grid_m, s.grid_v,
                      gmin, prm.dx, inv_dx, prm.dt, prm.drag, prm.nx, prm.ny, prm.nz],
              device=dev)
    wp.launch(K.k_grid_op, dim=prm.ngrid,
              inputs=[s.grid_m, s.grid_v, s.grid_v, prm.dt, fext,
                      prm.grid_min[1], prm.dx, prm.nx, prm.ny, prm.nz, prm.floor_y, prm.floor_friction,
                      K.WALL_NODES],
              device=dev)  # in-place ok (fwd only)
    wp.launch(K.k_g2p, dim=N,
              inputs=[s.x, s.v, s.C, s.F, s.dFc, s.F_new, s.grid_v, gmin, prm.dx, inv_dx, prm.dt, prm.nx, prm.ny, prm.nz],
              device=dev)
    wp.launch(K.k_update, dim=N,
              inputs=[s.x, s.x, s.v, s.F, s.F_new, s.F, prm.dt, prm.smoothing,
                      nb0[0], nb0[1], nb0[2], 0, 0.0, nb0[0], 0, nb0[1]], device=dev)
    if prm.floor_y > -1.0e8:                                # sharp particle-level floor (drop heroes)
        wp.launch(K.k_floor_clamp, dim=N,
                  inputs=[s.x, s.v, prm.floor_y, prm.floor_friction], device=dev)


def rollout(s: MPMState, prm: MPMParams, T: int):
    """Run T timesteps in place."""
    for _ in range(T):
        mpm_step(s, prm)


def compute_volumes(s: MPMState, prm: MPMParams):
    """One-time rest-volume estimate — eq (10). Call once after sampling."""
    inv_dx = 1.0 / prm.dx
    gmin = wp.vec3(*prm.grid_min)
    _reset_grid(s, prm)
    # mass-only P2G (dt=0, drag=0)
    wp.launch(K.k_stress, dim=s.N, inputs=[s.F, s.dFc, s.Fp, s.lam, s.mu, s.P], device=s.device)
    nb0 = _nobond(s.N, s.device)
    wp.launch(K.k_p2g, dim=s.N,
              inputs=[s.x, s.v, s.C, s.F, s.dFc, s.P, s.m, s.vol, nb0[0], nb0[2], 0, s.grid_m, s.grid_v,
                      gmin, prm.dx, inv_dx, 0.0, 0.0, prm.nx, prm.ny, prm.nz],
              device=s.device)
    wp.launch(K.k_volume, dim=s.N,
              inputs=[s.x, s.m, s.grid_m, s.vol, gmin, prm.dx, inv_dx, prm.nx, prm.ny, prm.nz],
              device=s.device)
