"""The scene's numbers and its initial state (D147): two spheres thrown toward each other on parabolas, colliding at the
apex with their centres offset vertically, then falling to a floor.

The numbers are dimensionless groups of the scene, stated once (not tuned): the impact speed over the sound speed
(v_rel / c = 0.1, a rubber ball's regime), the body's self-weight strain G = rho g H / E (F1's definition, H the
source's height; 0.05), the collision time (1 s, the bodies half-formed), the vertical offset of the centres at contact
(half a radius: friction spins the bodies), the launch height above the floor, Coulomb's coefficient (0.5, rubber on
rubber), the length (8 s), each target's turn about the vertical at launch (the two bunnies face each other) and the
morph time (the user's redesign: the bodies meet mid-morph; scene/schedule.py).
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
    yaw: tuple = (90.0, -90.0)   # degrees about the vertical at launch: the bunny faces +z in its frame, so they face
                                 # each other along x (its tail at -z, its head forward; checked on the mesh)
    floor: bool = True
    bodies: int = 2              # 1: one body (V0 at rest without forces, V1 thrown onto the floor)
    frame: str = "body"          # "body": each body's Buckens frame; "fixed": the frame at launch (V0's world frame);
                                 # "standing" (B9b option 1, the user's choice of 2026-10-20; one body at rest on the
                                 # floor): the target's pose fixed in the world, standing on the floor, as the single-body
                                 # --floor runs' (scripts/scene_run.py prepares the body with --floor), velocities as they
                                 # are
    drag: float = 0.0            # V0 keeps the frozen drag (0.9) to compare with the frozen runs
    assim_free: bool = True      # the assimilation reads the controls' deformation without the loads (window.free_end_F)
    morph_time: float = 0.0      # s: each body's target on its way from the source over this time (scene/schedule.py,
                                 # the bodies meet mid-morph); 0: the target from the start
    g: float = 0.0               # wu/s^2: the gravity given as an acceleration, as the single-body runs' --f_ext (B9b:
                                 # B9's 0.65); 0: from G
    rest: bool = False           # one body at rest on the floor from the start, as the single-body --floor runs (B9b):
                                 # no throw (t_contact, vrel_c, offset and lift unused), the body about the origin and
                                 # the floor's plane half a sample pitch below its lowest particle, where the floor's
                                 # position constraint holds the centres (mpm/bodies.body_trajectory)


def yaw_matrix(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])


def layout(spec: SceneSpec, sources: list, vps: list, young: float, poisson: float, mass_ref_n: int, dx: float,
           dt: float, T: int, radius: float | None = None):
    """(x0 (N,3), v0 (N,3), R0 per body, prm of the world grid, a dict of the scene's derived numbers). sources: each
    body's source sample about its centroid in its target's orientation (the frame's reference); vps: each sample's
    volume per particle; radius: the bodies' largest radius as targets (the domain's reach)."""
    X0 = np.asarray(sources[0], np.float64)
    R = float(max(np.linalg.norm(np.asarray(s), axis=1).max() for s in sources))
    H = float(X0[:, 1].max() - X0[:, 1].min())
    rho = (mass_ref_n / len(X0)) / vps[0]
    lam, mu = lame(young, poisson)
    c = math.sqrt((lam + 2.0 * mu) / rho)
    g = spec.g if spec.g > 0.0 else spec.G * young / (rho * H)
    if spec.rest and spec.bodies != 1:
        raise ValueError("rest: one body at rest on the floor (B9b)")
    tc = 0.0 if spec.rest else spec.t_contact                  # at rest: no throw, the body about the origin
    vh = 0.0 if spec.rest else 0.5 * spec.vrel_c * c
    vy = g * tc
    b = spec.offset * R
    dc = math.sqrt(max((2.0 * R) ** 2 - b ** 2, 0.0))
    at_contact = [np.array([-0.5 * dc, 0.5 * b, 0.0]), np.array([0.5 * dc, -0.5 * b, 0.0])][:spec.bodies]
    vel = [np.array([vh, vy, 0.0]), np.array([-vh, vy, 0.0])][:spec.bodies]
    if spec.bodies == 1:                                       # one body: its apex at the origin
        at_contact = [np.zeros(3)]
    start = [p - v * tc - 0.5 * np.array([0.0, -g, 0.0]) * tc ** 2 for p, v in zip(at_contact, vel)]
    R0 = [yaw_matrix(a) for a in spec.yaw]
    x0 = np.concatenate([s @ r.T + c0 for s, r, c0 in zip(sources, R0, start)]).astype(np.float32)
    v0 = np.concatenate([np.tile(v, (len(s), 1)) for s, v in zip(sources, vel)]).astype(np.float32)
    floor_y = float(min(c0[1] - R for c0 in start) - spec.lift) if spec.floor else -1.0e9
    if spec.floor and spec.rest:          # the sample's surface (half a pitch below its lowest centre) on the floor
        floor_y = float(x0[:, 1].min()) - 0.5 * float(vps[0]) ** (1.0 / 3.0)
    # the domain from the scene's extents: a body (its largest radius as source or target, Rb, in any orientation)
    # stays within a horizontal reach of its launch and contact centres, travelling at most at the throw's horizontal
    # speed after the contact and rolling no faster (the same reach in z, where friction and tumbling may carry it);
    # vertically from the floor (the centres held half a pitch above it) to the apex; beyond that, the stencil's two
    # cells, the runner's two-cell margin and one cell
    Rb = max(R, float(radius or 0.0))
    reach = vh * (spec.seconds - tc) + Rb + 5.0 * dx
    xs = [c0[0] for c0 in start + at_contact]
    bottom = floor_y if spec.floor else min(c0[1] for c0 in start) - reach
    lo = np.array([min(xs) - reach, bottom - 5.0 * dx, -reach])
    hi = np.array([max(xs) + reach, max(c0[1] for c0 in at_contact) + Rb + 5.0 * dx, reach])
    n = [int(math.ceil((hi[k] - lo[k]) / dx)) for k in range(3)]
    prm = MPMParams(dx=dx, dt=dt, drag=spec.drag, f_ext=(0.0, -g, 0.0), grid_min=tuple(float(v) for v in lo),
                    nx=n[0], ny=n[1], nz=n[2], floor_y=floor_y, floor_friction=spec.mu)
    windows = int(math.ceil(spec.seconds / (2 * T * dt)))
    info = {"R": R, "H": H, "rho": rho, "c": c, "g": g, "G_of_g": g * rho * H / young, "v_h": vh, "v_y": vy,
            "offset_wu": b,
            "centres_at_contact": [p.tolist() for p in at_contact], "centres_at_launch": [p.tolist() for p in start],
            "velocities": [v.tolist() for v in vel], "floor_y": floor_y, "grid": n, "windows": windows,
            "spec": dataclasses.asdict(spec)}
    return x0, v0, R0, prm, info
