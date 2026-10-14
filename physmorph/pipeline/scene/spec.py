"""The scene's numbers and its initial state (D147): two spheres thrown toward each other on parabolas, colliding at the
apex with their centres offset vertically, then falling to a floor.

The numbers are dimensionless groups of the scene, stated once (not tuned): the impact speed over the sound speed
(v_rel / c = 0.1, a rubber ball's regime), the body's self-weight strain G = rho g H / E (F1's definition, H the
source's height; 0.05), the collision time (1 s, the bodies half-formed), the vertical offset of the centres at contact
(half a radius: friction spins the bodies), the launch height above the floor, Coulomb's coefficient (0.5, rubber on
rubber), the length (8 s) and each target's turn about the vertical at launch (the two bunnies face each other).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from ...mpm.constitutive import lame
from ...mpm.state import MPMParams


@dataclass
class SceneSpec:
    vrel_c: float = 0.1          # impact speed / sound speed
    G: float = 0.05              # rho g H / E
    t_contact: float = 1.0       # s, at the apex
    offset: float = 0.5          # the centres' vertical offset at contact, radii (body 0 above)
    lift: float = 1.0            # wu: the lowest particle above the floor at launch
    mu: float = 0.5              # Coulomb, between the bodies and on the floor
    seconds: float = 8.0         # the scene's length
    yaw: tuple = (0.0, 180.0)    # degrees about the vertical: each target's turn at launch
    floor: bool = True


def yaw_matrix(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])


def layout(spec: SceneSpec, sources: list, vps: list, young: float, poisson: float, mass_ref_n: int, dx: float,
           dt: float, T: int):
    """(x0 (N,3), v0 (N,3), R0 per body, prm of the world grid, a dict of the scene's derived numbers). sources: each
    body's source sample about its centroid in its target's orientation (the frame's reference); vps: each sample's
    volume per particle."""
    X0 = np.asarray(sources[0], np.float64)
    R = float(max(np.linalg.norm(np.asarray(s), axis=1).max() for s in sources))
    H = float(X0[:, 1].max() - X0[:, 1].min())
    rho = (mass_ref_n / len(X0)) / vps[0]
    lam, mu = lame(young, poisson)
    c = math.sqrt((lam + 2.0 * mu) / rho)
    g = spec.G * young / (rho * H)
    vh = 0.5 * spec.vrel_c * c
    vy = g * spec.t_contact
    b = spec.offset * R
    dc = math.sqrt(max((2.0 * R) ** 2 - b ** 2, 0.0))
    tc = spec.t_contact
    at_contact = [np.array([-0.5 * dc, 0.5 * b, 0.0]), np.array([0.5 * dc, -0.5 * b, 0.0])]
    vel = [np.array([vh, vy, 0.0]), np.array([-vh, vy, 0.0])]
    start = [p - v * tc - 0.5 * np.array([0.0, -g, 0.0]) * tc ** 2 for p, v in zip(at_contact, vel)]
    R0 = [yaw_matrix(a) for a in spec.yaw]
    x0 = np.concatenate([s @ r.T + c0 for s, r, c0 in zip(sources, R0, start)]).astype(np.float32)
    v0 = np.concatenate([np.tile(v, (len(s), 1)) for s, v in zip(sources, vel)]).astype(np.float32)
    floor_y = float(min(c0[1] - R for c0 in start) - spec.lift) if spec.floor else -1.0e9
    reach = vh * (spec.seconds - tc) + 2.0 * R + 4.0 * dx       # how far a body may travel back out and roll
    lo = np.array([start[0][0] - reach, (floor_y if spec.floor else min(c0[1] for c0 in start) - reach) - 4.0 * dx,
                   -(2.0 * R + 6.0 * dx)])
    hi = np.array([start[1][0] + reach, max(c0[1] for c0 in at_contact) + 2.0 * R + 6.0 * dx, 2.0 * R + 6.0 * dx])
    n = [int(math.ceil((hi[k] - lo[k]) / dx)) for k in range(3)]
    prm = MPMParams(dx=dx, dt=dt, drag=0.0, f_ext=(0.0, -g, 0.0), grid_min=tuple(float(v) for v in lo),
                    nx=n[0], ny=n[1], nz=n[2], floor_y=floor_y, floor_friction=spec.mu)
    windows = int(math.ceil(spec.seconds / (2 * T * dt)))
    info = {"R": R, "H": H, "rho": rho, "c": c, "g": g, "v_h": vh, "v_y": vy, "offset_wu": b,
            "centres_at_contact": [p.tolist() for p in at_contact], "centres_at_launch": [p.tolist() for p in start],
            "velocities": [v.tolist() for v in vel], "floor_y": floor_y, "grid": n, "windows": windows,
            "spec": dataclasses.asdict(spec)}
    return x0, v0, R0, prm, info
