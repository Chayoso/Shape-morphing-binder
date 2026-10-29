"""Each body's loss raster in its own frame (D147 fix 1, the user's approval of 2026-10-16).

A body's target is built on the single-body preparation's domain (prepare.py: the cube [-h, h]^3, h the transport's
leash plus two MPM cells), and its loss raster covers that cube. In the scene a landing squashes a body and its rebound
stretches it along its own axes past that cube (the final 300k runs: body 0 to 5.76-7.06 wu against h = 5.48, the
CIC deposit losing mass from h - one loss cell = 5.28): the transport's mass check then returns inf ("escaped mass is
never concealed", the single-body rule, where leaving the raster is leaving the simulation) and the window lost its
controls (forensic, 2026-10-15). Here the raster covers where the body goes: at each window start the body's
uncontrolled continuation (the window's persistent rollout with zero controls, every step) is read in the body's frame,
and the raster is the smallest cube, grown from the preparation's by whole steps that keep its cell and its grid lines
(MPM and loss cells alike), that holds that extent with the preparation's own margin (two MPM cells) inside the CIC
deposit's last full cell. Nothing else is new: the grown target is built by the frozen build_target on the grown cube
with the base target's units, transport scale and relief, and a body that stays inside the preparation's cube keeps its
base target (the same object). Grown targets are kept per step count and reused."""
from __future__ import annotations

import dataclasses
import math

import torch
import warp as wp

from ..target import build_target


class BodyRaster:
    """prm: the body's preparation's MPM parameters (the cube [grid_min, grid_min + nx dx]^3), cfg: its run config (its
    loss_res: the loss cell is nx dx / loss_res)."""

    def __init__(self, prm, cfg):
        if not (prm.nx == prm.ny == prm.nz and len(set(prm.grid_min)) == 1):
            raise ValueError("the preparation's domain is a cube")
        self.prm, self.cfg = prm, cfg
        g = math.gcd(int(prm.nx), int(cfg.loss_res))
        self.um, self.ul = int(prm.nx) // g, int(cfg.loss_res) // g     # a step: um MPM cells = ul loss cells a side
        self.lo = float(prm.grid_min[0])
        self.ldx = float(prm.dx) * prm.nx / cfg.loss_res

    def bounds(self, k: int) -> tuple:
        """(lowest kept, highest kept) coordinate of the raster grown by k steps: the CIC deposit keeps a particle's
        whole mass from the cube's lower face to one loss cell below its upper face."""
        lo = self.lo - k * self.um * self.prm.dx
        hi = lo + (self.prm.nx + 2 * k * self.um) * self.prm.dx
        return lo, hi - self.ldx

    def need(self, ymin: float, ymax: float) -> int:
        """The fewest steps whose raster holds [ymin, ymax] with the preparation's margin, two MPM cells."""
        m = 2.0 * float(self.prm.dx)
        k = 0
        while True:
            lo, hi = self.bounds(k)
            if ymin - m >= lo and ymax + m <= hi:
                return k
            k += 1

    def grown(self, k: int):
        """(prm, cfg) of the raster grown by k steps (build_target reads the cube and loss_res)."""
        prm = dataclasses.replace(self.prm, grid_min=(self.lo - k * self.um * self.prm.dx,) * 3,
                                  nx=self.prm.nx + 2 * k * self.um, ny=self.prm.ny + 2 * k * self.um,
                                  nz=self.prm.nz + 2 * k * self.um)
        return prm, dataclasses.replace(self.cfg, loss_res=self.cfg.loss_res + 2 * k * self.ul)


def target_for(body, k: int):
    """The body's current target on the raster grown by k steps (k = 0: the base target itself)."""
    base = body.base
    if k == 0:
        return base
    if k not in body.packs:
        prm, cfg = body.raster.grown(k)
        with torch.no_grad():
            p = build_target(base.pts, prm, cfg, draws=base.draws)
        p.unit_ratio, p.unit_grad_ratio, p.ot_scale = base.unit_ratio, base.unit_grad_ratio, base.ot_scale
        p.relief = base.relief
        body.packs[k] = p
        print(f"[raster] body rows {body.start}-{body.stop}: the target on the raster grown by {k} step(s): "
              f"{cfg.loss_res}^3 loss cells of {p.ldx:.4f} wu over [{prm.grid_min[0]:.3f}, "
              f"{prm.grid_min[0] + prm.nx * prm.dx:.3f}]", flush=True)
    return body.packs[k]


def continuation_extent(win) -> list:
    """Each body's (lowest, highest) coordinate in its own frame over the window's uncontrolled continuation, every
    step from the start: the persistent no-grad rollout with the controls it was built with (zero)."""
    out = []
    with torch.no_grad():
        win.dc_buf.zero_()
        win.u_buf.zero_()
        win.tr.run()
        for body, tr in zip(win.bodies, win.tr.trajs):
            lo, hi = float("inf"), float("-inf")
            for t in range(win.T + 1):
                y = body.frame(wp.to_torch(tr.x[t]))[0]
                lo, hi = min(lo, float(y.min())), max(hi, float(y.max()))
            out.append((lo, hi))
    win.zero_rolled = True
    return out


def cover(win) -> list:
    """Set each body's target for the window (its base target, or the base on a grown raster when the uncontrolled
    continuation leaves the base raster); the views the objective reads take it. Returns the step count per body."""
    ext = continuation_extent(win)
    ks = []
    for b, (body, (lo, hi)) in enumerate(zip(win.bodies, ext)):
        cur = body.tgt
        if body.base is None or (cur is not body.base and all(cur is not p for p in body.packs.values())):
            body.base, body.packs = cur, {}                 # a new target (the schedule's, or the first window's)
        if body.raster is None:
            ks.append(0)
            continue
        k = body.raster.need(lo, hi)
        body.tgt = target_for(body, k)
        win.views[b].tgt = body.tgt
        ks.append(k)
    win.raster_k, win.raster_extent = ks, [[round(lo, 4), round(hi, 4)] for lo, hi in ext]
    return ks
