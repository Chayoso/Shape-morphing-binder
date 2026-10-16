"""Several bodies stepped together on one world grid (D147): each body a Trajectory on its own particles and grid field,
the pair's grid operation with the gated contact between the fields (mpm/contact.py), the rollout and its adjoint as
CUDA graphs, and the torch bridge of the scene.

Each body's Trajectory is the frozen single-body one (traj.py) on that body's particles: its stress, P2G (cell order,
hand adjoints), G2P, update, volume, bonds, minimum spacing and layer relaxation, every neighbour list local to the body.
Only the grid operation is shared: Bodies.step runs every body's part before the grid, then contact.k_grid_op_pair,
then every body's part after it. Outputs are concatenated in body order (body b holds the rows [start_b, stop_b)).
"""
from __future__ import annotations

import torch
import warp as wp

from . import contact as CT
from .function import RolloutSpec
from .state import MPMParams
from .traj import Trajectory, _release_idle_cuda_cache


class Bodies:
    """Two bodies' trajectories (same prm, same T) stepped together. vps: each body's sample volume per particle (the
    gate's node volume is vp J); mu: Coulomb's coefficient between the bodies and on the floor."""

    def __init__(self, trajs: list, vps: list, mu: float):
        if len(trajs) not in (1, 2):
            raise ValueError("one body, or a pair in contact")
        if any(tr.prm != trajs[0].prm or tr.T != trajs[0].T for tr in trajs):
            raise ValueError("the bodies share one grid and one step count")
        self.trajs, self.vps, self.mu = trajs, [float(v) for v in vps], float(mu)
        tr0 = trajs[0]
        self.prm, self.T, self.device = tr0.prm, tr0.T, tr0.device
        self.N = sum(tr.N for tr in trajs)
        self.starts = [0, trajs[0].N]
        # the gate's node volumes: per step on a tape (the contact kernel's generated adjoint replays the gate), one set
        # shared by the steps of a forward-only rollout
        n_sets = 1 if not tr0.requires_grad else self.T
        self.gvol = [[wp.zeros(self.prm.ngrid, dtype=float, device=self.device) for _ in trajs] for _ in range(n_sets)]
        self.ones = [wp.ones(tr.N, dtype=float, device=self.device) for tr in trajs]
        self.impulse = wp.zeros(1, dtype=wp.vec3, device=self.device)   # body 0's momentum change by the contact
        self.graph = None

    def step(self, t: int):
        prm, dev = self.prm, self.device
        inv_dx, gmin = 1.0 / prm.dx, wp.vec3(*prm.grid_min)
        for tr in self.trajs:
            tr._pre_grid(t)
        gv = self.gvol[t % len(self.gvol)]
        for b, tr in enumerate(self.trajs):
            gv[b].zero_()
            J = tr.J[t] if tr.carries else self.ones[b]
            wp.launch(CT.k_node_volume, dim=tr.N, inputs=[tr.x[t], J, self.vps[b], gv[b], gmin, prm.dx, inv_dx,
                                                         prm.nx, prm.ny, prm.nz], device=dev, record_tape=False)
        a = self.trajs[0]
        tail = [prm.dt, wp.vec3(*prm.f_ext), prm.grid_min[1], prm.dx, prm.nx, prm.ny, prm.nz, prm.floor_y, self.mu]
        if len(self.trajs) == 1:
            wp.launch(CT.k_grid_op_one, dim=prm.ngrid, inputs=[a.gm[t], a.gmom[t], gv[0], a.gvel[t]] + tail, device=dev)
        else:
            b = self.trajs[1]
            wp.launch(CT.k_grid_op_pair, dim=prm.ngrid, inputs=[a.gm[t], a.gmom[t], gv[0], a.gvel[t], b.gm[t],
                                                                b.gmom[t], gv[1], b.gvel[t]] + tail + [self.impulse],
                      device=dev)
        for tr in self.trajs:
            tr._post_grid(t)

    def rollout(self):
        for t in range(self.T):
            self.step(t)

    def capture(self) -> bool:
        """Record the rollout as a CUDA graph (persistent trajectories), as Trajectory.capture."""
        if not all(tr.persistent for tr in self.trajs) or not str(self.device).startswith("cuda"):
            return False
        self.rollout()                                  # module loading is not allowed inside a capture
        wp.synchronize_device(self.device)
        with wp.ScopedCapture(device=self.device) as cap:
            self.rollout()
        self.graph = cap.graph
        return True

    def run(self):
        _release_idle_cuda_cache(self.device)
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.rollout()

    # the stacked views the pipeline reads (body order)
    def cat(self, name: str, t: int) -> torch.Tensor:
        return torch.cat([wp.to_torch(getattr(tr, name)[t]) for tr in self.trajs])

    @property
    def layer_ug(self):
        """The bodies' u gates in body order (a copy, as a Warp array: what the window optimiser reads)."""
        return wp.from_torch(torch.cat([wp.to_torch(tr.layer_ug) for tr in self.trajs]).contiguous())


class FloorTrajectory(Trajectory):
    """A body's Trajectory over a floor: the step's last position update writes a buffer (xq), which
    contact.k_floor_position projects into x[t + 1], every centre at least floor_c high (D147, V1d)."""

    def __init__(self, *args, floor_c: float, **kw):
        super().__init__(*args, **kw)
        self.floor_c = float(floor_c)
        self.xq = [wp.zeros(self.N, dtype=wp.vec3, device=self.device, requires_grad=self.requires_grad)
                   for _ in range(self.T + 1)]

    def _x_out(self, t: int):
        return self.xq[t + 1]

    def _after_positions(self, t: int):
        wp.launch(CT.k_floor_position, dim=self.N, inputs=[self.xq[t + 1], self.floor_c, self.x[t + 1]],
                  device=self.device)


# ── the deformation's drag (D147 fix 4, the user's approval of 2026-10-16) ───────────────────────────────────────────
# The scene has no drag on the whole motion (--drag 0: a thrown body keeps its flight), so nothing damps a body's
# deformation either, while the single-body recipe damps every velocity at MPMParams.drag (0.9 / s) in P2G (m v (1 - dt
# drag)). Here each body's deformation velocity alone is damped at that rate, where the single-body drag acts: P2G reads
# v - dt rate (v - (v_c + w x r)), with v_c the body's mass-weighted mean velocity and w = I^-1 sum m r x v its
# least-squares angular velocity about its centre of mass c (r = x - c; the particles' part: APIC's affine C is left as
# the single-body drag leaves it). Sum m (v - v_c - w x r) = 0 and sum m r x (v - v_c - w x r) = L - I w = 0, so the
# body's linear and angular momentum are kept; the rigid motion (flight, spin, tumble) is not damped. On a tape the
# kernels' generated adjoints carry the gradient through the reductions.

@wp.kernel
def k_rigid_sums(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), m: wp.array(dtype=float),
                 sx: wp.array(dtype=wp.vec3), sv: wp.array(dtype=wp.vec3), sm: wp.array(dtype=float)):
    p = wp.tid()
    wp.atomic_add(sx, 0, m[p] * x[p])
    wp.atomic_add(sv, 0, m[p] * v[p])
    wp.atomic_add(sm, 0, m[p])


@wp.kernel
def k_rigid_moments(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), m: wp.array(dtype=float),
                    sx: wp.array(dtype=wp.vec3), sm: wp.array(dtype=float),
                    sL: wp.array(dtype=wp.vec3), sI: wp.array(dtype=wp.mat33)):
    p = wp.tid()
    r = x[p] - sx[0] / sm[0]
    wp.atomic_add(sL, 0, m[p] * wp.cross(r, v[p]))
    wp.atomic_add(sI, 0, m[p] * (wp.dot(r, r) * wp.identity(n=3, dtype=float) - wp.outer(r, r)))


@wp.kernel
def k_deformation_drag(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                       sx: wp.array(dtype=wp.vec3), sv: wp.array(dtype=wp.vec3), sm: wp.array(dtype=float),
                       sL: wp.array(dtype=wp.vec3), sI: wp.array(dtype=wp.mat33), kappa: float,
                       vd: wp.array(dtype=wp.vec3)):
    p = wp.tid()
    c = sx[0] / sm[0]
    w = wp.inverse(sI[0]) @ sL[0]
    vd[p] = v[p] - kappa * (v[p] - sv[0] / sm[0] - wp.cross(w, x[p] - c))


def deformation_drag_rate(prm: MPMParams) -> float:
    """The rate the scene's deformation is damped at: the single-body recipe's drag (MPMParams.drag's default), less
    the drag the scene's world already puts on every velocity (none in the scene; V0 kept the frozen 0.9 and so needs
    no more)."""
    return max(float(MPMParams.drag) - float(prm.drag), 0.0)


class _DeformationDrag:
    """A Trajectory whose P2G reads the body's velocities with their deformation part damped (the note above). The
    damped velocities of step t are their own arrays (vd[t]); P2G is the frozen one, handed vd[t] in place of v[t] for
    the step (self.v[t] is the step's input array only there), so its hand adjoint seeds vd[t].grad, which the
    damping kernels' generated adjoints carry to v[t] and x[t]."""

    def _init_drag(self, rate: float):
        self.drag_kappa = float(self.prm.dt) * float(rate)
        dev, rg, T = self.device, self.requires_grad, self.T
        Z1 = lambda dt: [wp.zeros(1, dtype=dt, device=dev, requires_grad=rg) for _ in range(T)]   # noqa: E731
        self.drag_sx, self.drag_sv, self.drag_sL = Z1(wp.vec3), Z1(wp.vec3), Z1(wp.vec3)
        self.drag_sm, self.drag_sI = Z1(float), Z1(wp.mat33)
        self.drag_vd = [wp.zeros(self.N, dtype=wp.vec3, device=dev, requires_grad=rg) for _ in range(T)]

    def _pre_grid(self, t: int):
        dev, N = self.device, self.N
        sx, sv, sm, sL, sI, vd = (self.drag_sx[t], self.drag_sv[t], self.drag_sm[t], self.drag_sL[t], self.drag_sI[t],
                                  self.drag_vd[t])
        for a in (sx, sv, sm, sL, sI):
            a.zero_()
        x, v = self.x[t], self.v[t]
        wp.launch(k_rigid_sums, dim=N, inputs=[x, v, self.m, sx, sv, sm], device=dev)
        wp.launch(k_rigid_moments, dim=N, inputs=[x, v, self.m, sx, sm, sL, sI], device=dev)
        wp.launch(k_deformation_drag, dim=N, inputs=[x, v, sx, sv, sm, sL, sI, self.drag_kappa, vd], device=dev)
        self.v[t] = vd
        try:
            super()._pre_grid(t)
        finally:
            self.v[t] = v


class DragTrajectory(_DeformationDrag, Trajectory):
    def __init__(self, *args, drag_rate: float, **kw):
        super().__init__(*args, **kw)
        self._init_drag(drag_rate)


class DragFloorTrajectory(_DeformationDrag, FloorTrajectory):
    def __init__(self, *args, drag_rate: float, **kw):
        super().__init__(*args, **kw)
        self._init_drag(drag_rate)


def body_trajectory(spec: RolloutSpec, dfc, layer_u=None, requires_grad=False, persistent=True,
                    vp: float | None = None, drag_rate: float | None = None) -> Trajectory:
    """One body's Trajectory from its spec (the frozen single-body construction of window/setup and function.py); over a
    floor (spec.prm.floor_y) with the sample's volume per particle vp, a FloorTrajectory whose centres stay half a
    pitch above the floor. drag_rate: the deformation's drag (1 / s; None: deformation_drag_rate(spec.prm), the
    single-body rate where the world has no drag; 0: none)."""
    bonds = ((spec.bond_nbr, spec.bond_rest, spec.bond_frag) if spec.bond_nbr is not None else None)
    kw = dict(Fp=spec.Fp, v0=spec.v0, F0=spec.F0, C0=spec.C0, dFc=dfc, device=spec.device, requires_grad=requires_grad,
              vol0=spec.vol0, Fg0=spec.Fg0, track_geom=spec.track_geom, bonds=bonds, persistent=persistent,
              layer=spec.layer, layer_u=layer_u, bond_history=spec.bond_history, control_steps=spec.control_steps,
              polar_adjoint=spec.polar_adjoint, spacing=spec.spacing, volume_exact=spec.volume_exact, J0=spec.J0)
    args = (spec.x0, spec.m, spec.lam, spec.mu, spec.prm, spec.T)
    rate = deformation_drag_rate(spec.prm) if drag_rate is None else float(drag_rate)
    floor = vp is not None and spec.prm.floor_y > -1.0e8
    if floor:
        kw["floor_c"] = spec.prm.floor_y + 0.5 * float(vp) ** (1.0 / 3.0)
    if rate > 0.0:
        return (DragFloorTrajectory if floor else DragTrajectory)(*args, drag_rate=rate, **kw)
    return (FloorTrajectory if floor else Trajectory)(*args, **kw)


_WARMED: set = set()


class BodiesAdjoint:
    """PersistentAdjoint (function.py) for two bodies: one tape over the pair's rollout, forward and adjoint captured
    as CUDA graphs; the control dc (T, N, 3, 3) and the u leaf (N,) are single tensors whose rows the bodies view; the
    seeds are the end state's (x, F, v), every step's velocity and every step's position (the scene's rigid velocity
    fit reads the positions, pipeline/scene/frame.py)."""

    def __init__(self, specs: list, vps: list, mu: float):
        dev, T = specs[0].device, specs[0].T
        _release_idle_cuda_cache(dev)
        self.Ns = [int(s.x0.shape[0]) for s in specs]
        self.N, self.T, self.dev = sum(self.Ns), T, dev
        self.cuda = str(dev).startswith("cuda")
        self.rows = [(sum(self.Ns[:b]), sum(self.Ns[:b + 1])) for b in range(len(specs))]
        self.dc = torch.zeros(T, self.N, 3, 3, device=dev)
        self.u = torch.zeros(self.N, device=dev)
        trajs, self.dc_wp, self.u_wp = [], [], []
        for (s, e), spec in zip(self.rows, specs):
            dcw = [wp.from_torch(self.dc[t, s:e], dtype=wp.mat33, requires_grad=True) for t in range(T)]
            uw = wp.from_torch(self.u[s:e], dtype=wp.float32, requires_grad=True) if spec.layer is not None else None
            trajs.append(body_trajectory(spec, dcw, uw, requires_grad=True, persistent=True, vp=vps[len(trajs)]))
            self.dc_wp.append(dcw)
            self.u_wp.append(uw)
        self.bodies = Bodies(trajs, vps, mu)
        self.sx = torch.zeros(T, self.N, 3, device=dev)              # seeds of x_1..x_T
        self.sv = torch.zeros(T, self.N, 3, device=dev)              # seeds of v_1..v_T
        self.sF = torch.zeros(self.N, 3, 3, device=dev)
        self.seeds = {}
        for (s, e), tr in zip(self.rows, trajs):
            for t in range(1, T + 1):
                self.seeds[tr.x[t]] = wp.from_torch(self.sx[t - 1, s:e], dtype=wp.vec3)
                self.seeds[tr.v[t]] = wp.from_torch(self.sv[t - 1, s:e], dtype=wp.vec3)
            self.seeds[tr.F[T]] = wp.from_torch(self.sF[s:e], dtype=wp.mat33)
        self.grad_arrays = []
        extra = [a for lst in self.dc_wp for a in lst] + [u for u in self.u_wp if u is not None]
        for tr in trajs:
            for val in list(vars(tr).values()) + [extra]:
                for a in (val if isinstance(val, (list, tuple)) else [val]):
                    if isinstance(a, wp.array) and a.grad is not None and all(a.grad is not g for g in self.grad_arrays):
                        self.grad_arrays.append(a.grad)
        self.tape = None
        self.g_fwd = self.g_bwd = None
        if self.cuda:
            if str(dev) not in _WARMED:                     # module load outside any capture
                self._record_forward()
                self._zero_grads()
                self.tape.backward(grads=self.seeds)
                wp.synchronize_device(dev)
                _WARMED.add(str(dev))
            with wp.ScopedCapture(device=dev) as cap:
                self._record_forward()
            self.g_fwd = cap.graph
            with wp.ScopedCapture(device=dev) as cap:
                self._zero_grads()
                self.tape.backward(grads=self.seeds)
            self.g_bwd = cap.graph

    def _zero_grads(self):
        for g in self.grad_arrays:
            g.zero_()

    def _record_forward(self):
        self.tape = wp.Tape()
        with self.tape:
            self.bodies.rollout()

    def forward(self):
        _release_idle_cuda_cache(self.dev)
        if self.g_fwd is not None:
            wp.capture_launch(self.g_fwd)
        else:
            self._record_forward()

    def backward(self):
        _release_idle_cuda_cache(self.dev)
        if self.g_bwd is not None:
            wp.capture_launch(self.g_bwd)
        else:
            self._zero_grads()
            self.tape.backward(grads=self.seeds)

    def apply(self, dFc_t: torch.Tensor, u_t: torch.Tensor | None = None):
        return _BodiesBridge.apply(dFc_t, u_t, self)


class _BodiesBridge(torch.autograd.Function):
    """(x_T, F_T, v_T, V, X) of the pair, rows in body order: V[t - 1] = v_t and X[t - 1] = x_t for t = 1..T."""

    @staticmethod
    def forward(ctx, dFc_t: torch.Tensor, u_t, adj: BodiesAdjoint):
        N, T = adj.N, adj.T
        adj.dc.copy_(dFc_t.detach().reshape(T, N, 3, 3))
        adj.u.copy_(u_t.detach() if u_t is not None else torch.zeros(N, device=adj.u.device))
        ctx.u_req = u_t is not None and u_t.requires_grad
        adj.forward()
        ctx.adj = adj
        b = adj.bodies
        V = torch.stack([b.cat("v", t) for t in range(1, T + 1)])
        X = torch.stack([b.cat("x", t) for t in range(1, T + 1)])
        FT = torch.cat([wp.to_torch(tr.F[T]).reshape(-1, 9) for tr in b.trajs])
        return X[-1].clone(), FT, V[-1].clone(), V, X

    @staticmethod
    def backward(ctx, gx, gF, gv, gV, gX):
        adj = ctx.adj
        N, T = adj.N, adj.T
        with torch.no_grad():
            adj.sx.copy_(gX) if gX is not None else adj.sx.zero_()
            if gx is not None:
                adj.sx[T - 1] += gx
            adj.sv.copy_(gV) if gV is not None else adj.sv.zero_()
            if gv is not None:
                adj.sv[T - 1] += gv
            adj.sF.copy_(gF.reshape(N, 3, 3)) if gF is not None else adj.sF.zero_()
            adj.backward()
            g = torch.stack([torch.cat([wp.to_torch(lst[t].grad).reshape(-1, 3, 3) for lst in adj.dc_wp])
                             for t in range(T)])
            g_u = (torch.cat([wp.to_torch(u.grad) for u in adj.u_wp]).clone()
                   if ctx.u_req and all(u is not None for u in adj.u_wp) else None)
        return g, g_u, None
