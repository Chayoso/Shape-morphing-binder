"""MPM state (SoA Warp arrays) + params. See docs/SPEC.md §2, §4.

Notation: `F` = total particle deformation gradient (integrated by G2P);
`Fp` = plastic rest-state; `dFc` = control increment. Effective elastic
deformation for stress (D1): F_e = (F + dFc) Fp^{-1}  — eq (3').
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass
class MPMParams:
    dx: float = 0.5
    dt: float = 1.0 / 240.0
    drag: float = 0.9
    smoothing: float = 0.955          # F-smoothing s — eq (9)
    v_max: float = 0.0                # G2P velocity clamp (0 = off); anti-scatter
    eta_sym: int = 0                  # 1 = OBJECTIVE viscosity (damp sym(C) only; spin preserved)
    eta_mode: int = 0                 # 1 = EXPONENTIAL damping exp(-dt*eta) (dt-consistent, no clamp);
                                      # 0 = legacy linear max(0,1-dt*eta) — kept for old-table repro
    f_ext: tuple = (0.0, 0.0, 0.0)
    grid_min: tuple = (-16.0, -16.0, -16.0)
    nx: int = 64
    ny: int = 64
    nz: int = 64
    floor_y: float = -1.0e9           # separating floor for drop tests (-inf = off)
    floor_friction: float = 0.0

    @property
    def ngrid(self) -> int:
        return self.nx * self.ny * self.nz


@dataclass
class MPMState:
    # particle arrays (N)
    x: wp.array; v: wp.array; C: wp.array
    F: wp.array; Fp: wp.array; dFc: wp.array
    P: wp.array; F_new: wp.array          # scratch
    m: wp.array; vol: wp.array
    lam: wp.array; mu: wp.array; eta: wp.array          # eta = viscosity (rubber<->honey)
    # grid arrays (ngrid)
    grid_m: wp.array; grid_v: wp.array
    N: int
    device: str

    def clone_kinematic(self) -> dict:
        """Snapshot x, v, C, F, Fp as numpy (for promotion / checkpointing)."""
        return {
            "x": self.x.numpy().copy(), "v": self.v.numpy().copy(),
            "C": self.C.numpy().copy(), "F": self.F.numpy().copy(),
            "Fp": self.Fp.numpy().copy(),
        }


def _mat_id(N: int) -> np.ndarray:
    return np.tile(np.eye(3, dtype=np.float32), (N, 1, 1))


def make_state(
    x: np.ndarray,
    m: np.ndarray,
    lam: float | np.ndarray,
    mu: float | np.ndarray,
    params: MPMParams,
    v: np.ndarray | None = None,
    F: np.ndarray | None = None,
    Fp: np.ndarray | None = None,
    eta: float | np.ndarray = 0.0,
    device: str = "cuda",
    requires_grad: bool = False,
) -> MPMState:
    """Allocate an MPMState from numpy particle data."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    N = x.shape[0]
    m = np.ascontiguousarray(np.broadcast_to(m, (N,)), dtype=np.float32)
    lam_a = np.full(N, float(lam), np.float32) if np.isscalar(lam) else np.ascontiguousarray(lam, np.float32)
    mu_a = np.full(N, float(mu), np.float32) if np.isscalar(mu) else np.ascontiguousarray(mu, np.float32)
    eta_a = np.full(N, float(eta), np.float32) if np.isscalar(eta) else np.ascontiguousarray(eta, np.float32)
    v = np.zeros((N, 3), np.float32) if v is None else np.ascontiguousarray(v, np.float32)
    F = _mat_id(N) if F is None else np.ascontiguousarray(F, np.float32)
    Fp = _mat_id(N) if Fp is None else np.ascontiguousarray(Fp, np.float32)

    def arr(a, dtype, rg=False):
        return wp.array(a, dtype=dtype, device=device, requires_grad=rg)

    return MPMState(
        x=arr(x, wp.vec3, requires_grad),
        v=arr(v, wp.vec3, requires_grad),
        C=arr(np.zeros((N, 3, 3), np.float32), wp.mat33, requires_grad),
        F=arr(F, wp.mat33, requires_grad),
        Fp=arr(Fp, wp.mat33, False),                       # plastic: not differentiated
        dFc=arr(np.zeros((N, 3, 3), np.float32), wp.mat33, requires_grad),
        P=arr(np.zeros((N, 3, 3), np.float32), wp.mat33, requires_grad),
        F_new=arr(_mat_id(N), wp.mat33, requires_grad),
        m=arr(m, wp.float32, False),
        vol=arr(np.zeros(N, np.float32), wp.float32, False),
        lam=arr(lam_a, wp.float32, False),
        mu=arr(mu_a, wp.float32, False),
        eta=arr(eta_a, wp.float32, False),
        grid_m=arr(np.zeros(params.ngrid, np.float32), wp.float32, requires_grad),
        grid_v=arr(np.zeros((params.ngrid, 3), np.float32), wp.vec3, requires_grad),
        N=N, device=device,
    )
