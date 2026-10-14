"""The split step's Python (D148, `--gap_split`): the arrays a Trajectory carries for it, its deposit, grid operation
and G2P per step with their hand adjoints on a hand tape (generated ones otherwise), and the record of the split.

gap = (plane_m (ngrid, 3), plane_n (ngrid, 3), q (N, 3) or None, vp[, sig]): the target's gap planes (gap_split.py),
the particles' side points, vp = 0 for variant A (vp > 0: variant B, the node volume's particle volume), sig = 0 for
hard sides (sig > 0: soft sides, D148c). near (one int a node): 1 where a gap node lies in the 4^3 stencil of a particle
whose base node it is; the particles without one skip the split's kernels (an exact speed-up: their shares are 0).
"""
from __future__ import annotations

import numpy as np
import torch
import warp as wp

from . import gap_g2p as GG
from . import gap_kernels as GKK

WARP_ADJOINT = True      # the split G2P's adjoint in cell order on CUDA (tests switch it to compare with the plain one)


def near_flags(pn, dims, device):
    """near[b] = 1 when a gap node lies in nodes b .. b + 3 on every axis (the stencil of base node b)."""
    import torch
    nx, ny, nz = dims
    t = torch.as_tensor(pn, device=str(device)) if not isinstance(pn, wp.array) else wp.to_torch(pn)
    m3 = (t.reshape(nx, ny, nz, 3).abs().sum(-1) > 0)
    near = torch.zeros_like(m3)
    for oi in range(4):
        for oj in range(4):
            for ok in range(4):
                near[:nx - oi, :ny - oj, :nz - ok] |= m3[oi:, oj:, ok:]
    return wp.from_torch(near.reshape(-1).int().contiguous(), dtype=wp.int32)


def attach(tr, gap, A, Z, n_grid: int, rg: bool):
    """Allocate the split step's arrays on Trajectory `tr`."""
    pm, pn, q, vp = gap[:4]
    prm, dev, T = tr.prm, tr.device, tr.T
    tr.gap_m, tr.gap_n = A(pm, wp.vec3), A(pn, wp.vec3)
    tr.gap_near = wp.clone(near_flags(wp.to_torch(tr.gap_n).reshape(-1, 3), (prm.nx, prm.ny, prm.nz), dev))
    tr.gap_q = A(q, wp.vec3) if q is not None else Z(wp.vec3)
    tr.gap_vp = float(vp)
    tr.gap_sig = float(gap[4]) if len(gap) > 4 else 0.0
    pieces = gap[5] if len(gap) > 5 else None         # D148c: (box0, boxd, boxi, labels, nc, rep[, clamp]) of gap_walls.py
    tr.gap_clamp = int(pieces[6]) if pieces is not None and len(pieces) > 6 else 0     # D148d: the nearest piece

    def ints(a, vec):
        t = torch.as_tensor(np.asarray(a) if isinstance(a, np.ndarray) else a, device=str(dev)).int()
        return wp.clone(wp.from_torch(t.reshape(-1, 3).contiguous() if vec else t.reshape(-1).contiguous(),
                                      dtype=wp.vec3i if vec else wp.int32))
    tr.gap_rep = wp.full(tr.N, -1, dtype=wp.int32, device=dev)
    if pieces is None:
        tr.gap_box0 = tr.gap_boxd = wp.zeros(prm.ngrid, dtype=wp.vec3i, device=dev)
        tr.gap_boxi = wp.full(prm.ngrid, wp.vec3i(-1, -1, -1), dtype=wp.vec3i, device=dev)
        tr.gap_boxlab, tr.gap_ncell = wp.zeros(1, dtype=wp.int32, device=dev), wp.vec3i(1, 1, 1)
    else:
        box0, boxd, boxi, labels, nc, rep = pieces[:6]
        tr.gap_box0, tr.gap_boxd, tr.gap_boxi = ints(box0, True), ints(boxd, True), ints(boxi, True)
        tr.gap_boxlab, tr.gap_ncell = ints(labels, False), wp.vec3i(*[int(v) for v in nc])
        if rep is not None:
            wp.to_torch(tr.gap_rep).copy_(torch.as_tensor(rep, device=str(dev)).int())
    g1 = [wp.zeros(prm.ngrid, dtype=wp.float32, device=dev, requires_grad=rg) for _ in range(n_grid)]
    p1 = [wp.zeros(prm.ngrid, dtype=wp.vec3, device=dev, requires_grad=rg) for _ in range(n_grid)]
    v1 = [wp.zeros(prm.ngrid, dtype=wp.vec3, device=dev, requires_grad=rg) for _ in range(n_grid)]
    tr.gm1 = [g1[t % n_grid] for t in range(T)]
    tr.gmom1 = [p1[t % n_grid] for t in range(T)]
    tr.gvel1 = [v1[t % n_grid] for t in range(T)]
    # variant B's node volume: one buffer a step on a tape (the grid kernel's generated adjoint replays its branch)
    nv = (T if rg else 1) if tr.gap_vp > 0 else 1
    tr.gap_vol = [wp.zeros(prm.ngrid, dtype=float, device=dev) for _ in range(nv)]
    tr.gap_ones = wp.ones(tr.N, dtype=float, device=dev) if tr.gap_vp > 0 else None
    tr.gap_stats = [wp.zeros(4, dtype=float, device=dev) for _ in range(T)]          # a record a step


def _geo(tr):
    prm = tr.prm
    return wp.vec3(*prm.grid_min), prm.dx, 1.0 / prm.dx


def p2g_side1(tr, t: int, dfc, bnb, bnc, bK: int, tape=None):
    """Step t's side-1 deposit (after the frozen P2G): on a hand tape its hand adjoint, else Warp's generated one."""
    from .adjoints import _dummy_mat
    prm, dev = tr.prm, tr.device
    gmin, dx, inv_dx = _geo(tr)
    if tr.share_grid or tr.persistent:
        tr.gm1[t].zero_()
        tr.gmom1[t].zero_()
    x, v, C, F, P, gm1, gp1 = tr.x[t], tr.v[t], tr.C[t], tr.F[t], tr.P[t], tr.gm1[t], tr.gmom1[t]
    head = [x, v, C, F, dfc, P, tr.m, tr.vol, bnb, bnc, bK, tr.gap_q, tr.gap_m, tr.gap_n, tr.gap_near, tr.gap_sig,
            tr.gap_box0, tr.gap_boxd, tr.gap_boxi, tr.gap_boxlab, tr.gap_ncell, tr.gap_rep, tr.gap_clamp]
    wp.launch(GKK.k_p2g_side1, dim=tr.N, inputs=head + [gm1, gp1, gmin, dx, inv_dx, prm.dt, prm.drag, prm.nx, prm.ny,
                                                        prm.nz], device=dev, record_tape=tape is None)
    if tape is None:
        return
    a_dfc = dfc.grad if dfc.grad is not None else _dummy_mat(dev)

    def backward():
        wp.launch(GKK.k_p2g_side1_adj, dim=tr.N, inputs=head + [gmin, dx, inv_dx, prm.dt, prm.drag, prm.nx, prm.ny,
                                                                 prm.nz, gm1.grad, gp1.grad, x.grad, v.grad, C.grad,
                                                                 F.grad, a_dfc, int(dfc.grad is not None), P.grad],
                  device=dev)

    tape.record_func(backward=backward, arrays=[a for a in (x, v, C, F, P, gm1, gp1, dfc) if a.grad is not None])


def grid_op(tr, t: int):
    """Step t's grid operation with the gap nodes' two fields."""
    from .contact import k_node_volume
    from .kernels import WALL_NODES
    p, dev = tr.prm, tr.device
    gv = tr.gap_vol[t % len(tr.gap_vol)]
    if tr.gap_vp > 0:                                   # variant B: the node's particle volume, off the tape
        gv.zero_()
        J = tr.J[t] if tr.carries else tr.gap_ones
        wp.launch(k_node_volume, dim=tr.N, inputs=[tr.x[t], J, tr.gap_vp, gv, wp.vec3(*p.grid_min), p.dx, 1.0 / p.dx,
                                                   p.nx, p.ny, p.nz], device=dev, record_tape=False)
    tr.gap_stats[t].zero_()
    wp.launch(GKK.k_grid_op_gap, dim=p.ngrid, inputs=[tr.gm[t], tr.gmom[t], tr.gm1[t], tr.gmom1[t], tr.gap_n, gv,
                                                      int(tr.gap_vp > 0), tr.gvel[t], tr.gvel1[t], p.dt,
                                                      wp.vec3(*p.f_ext), p.grid_min[1], p.dx, p.nx, p.ny, p.nz,
                                                      p.floor_y, p.floor_friction, WALL_NODES, tr.gap_stats[t]],
              device=dev)


def g2p(tr, t: int, dfc, tape):
    """Step t's G2P from each particle's side (the hand adjoint on a hand tape, else the generated one)."""
    from .adjoints import _dummy_mat
    prm, dev, N = tr.prm, tr.device, tr.N
    gmin, dx, inv_dx = _geo(tr)
    x, F, gv, gv1, v1, C1, Fn = tr.x[t], tr.F[t], tr.gvel[t], tr.gvel1[t], tr.v[t + 1], tr.C[t + 1], tr.Fraw[t + 1]
    sides = [tr.gap_q, tr.gap_m, tr.gap_n, tr.gap_near, tr.gap_sig, tr.gap_box0, tr.gap_boxd, tr.gap_boxi, tr.gap_boxlab,
             tr.gap_ncell, tr.gap_rep, tr.gap_clamp]
    wp.launch(GG.k_g2p_gap, dim=N, inputs=[x, v1, C1, F, dfc, Fn, gv, gv1] + sides + [gmin, dx, inv_dx, prm.dt, prm.nx,
                                                                                    prm.ny, prm.nz],
              device=dev, record_tape=tape is None)
    if tape is None:
        return
    a_dfc = dfc.grad if dfc.grad is not None else _dummy_mat(dev)

    def backward():
        tail = sides + [gmin, dx, inv_dx, prm.dt, prm.nx, prm.ny, prm.nz, v1.grad, C1.grad, Fn.grad, x.grad, F.grad,
                        a_dfc, int(dfc.grad is not None), gv.grad, gv1.grad]
        if tr.order is not None and WARP_ADJOINT:
            wp.launch_tiled(GG.k_g2p_adj_gap_warp, dim=[tr.order_blocks],
                            inputs=[tr.order, N, x, F, dfc, gv, gv1, C1] + tail, block_dim=GG.WARP_LANES, device=dev)
        else:
            wp.launch(GG.k_g2p_adj_gap, dim=N, inputs=[x, F, dfc, gv, gv1] + tail, device=dev)

    tape.record_func(backward=backward, arrays=[a for a in (x, F, gv, gv1, v1, C1, Fn, dfc) if a.grad is not None])


def record(tr) -> list:
    """The last rollout's split record over its steps: [split mass, separating mass, split node-steps, separating
    node-steps] (mass summed over steps)."""
    import torch
    return [float(v) for v in torch.stack([wp.to_torch(s) for s in tr.gap_stats]).sum(0).tolist()]
