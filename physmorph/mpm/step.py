"""MPM step orchestration. One frame = T x mpm_step. See docs/SPEC.md §3.3."""
from __future__ import annotations

import warp as wp

from . import kernels as K
from .state import MPMParams, MPMState


def _reset_grid(s: MPMState, prm: MPMParams):
    wp.launch(K.k_zero_scalar, dim=prm.ngrid, inputs=[s.grid_m], device=s.device)
    wp.launch(K.k_zero_vec, dim=prm.ngrid, inputs=[s.grid_v], device=s.device)


def mpm_step(s: MPMState, prm: MPMParams):
    """One MLS-MPM timestep: stress -> P2G -> grid -> G2P -> update. eq (3')-(9)."""
    inv_dx = 1.0 / prm.dx
    gmin = wp.vec3(*prm.grid_min)
    fext = wp.vec3(*prm.f_ext)
    N, dev = s.N, s.device

    wp.launch(K.k_stress, dim=N, inputs=[s.F, s.dFc, s.Fp, s.lam, s.mu, s.P], device=dev)
    _reset_grid(s, prm)
    wp.launch(K.k_p2g, dim=N,
              inputs=[s.x, s.v, s.C, s.F, s.dFc, s.P, s.m, s.vol, s.grid_m, s.grid_v,
                      gmin, prm.dx, inv_dx, prm.dt, prm.drag, prm.nx, prm.ny, prm.nz],
              device=dev)
    wp.launch(K.k_grid_op, dim=prm.ngrid,
              inputs=[s.grid_m, s.grid_v, s.grid_v, prm.dt, fext,
                      prm.grid_min[1], prm.dx, prm.ny, prm.nz, prm.floor_y, prm.floor_friction],
              device=dev)  # in-place ok (fwd only)
    wp.launch(K.k_g2p, dim=N,
              inputs=[s.x, s.v, s.C, s.F, s.dFc, s.F_new, s.grid_v, s.eta,
                      gmin, prm.dx, inv_dx, prm.dt, prm.nx, prm.ny, prm.nz, prm.v_max,
                      prm.eta_sym, prm.eta_mode],
              device=dev)
    wp.launch(K.k_update, dim=N,
              inputs=[s.x, s.x, s.v, s.F, s.F_new, s.F, prm.dt, prm.smoothing], device=dev)
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
    wp.launch(K.k_p2g, dim=s.N,
              inputs=[s.x, s.v, s.C, s.F, s.dFc, s.P, s.m, s.vol, s.grid_m, s.grid_v,
                      gmin, prm.dx, inv_dx, 0.0, 0.0, prm.nx, prm.ny, prm.nz],
              device=s.device)
    wp.launch(K.k_volume, dim=s.N,
              inputs=[s.x, s.m, s.grid_m, s.vol, gmin, prm.dx, inv_dx, prm.nx, prm.ny, prm.nz],
              device=s.device)
