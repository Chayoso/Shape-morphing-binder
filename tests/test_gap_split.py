"""D148: the two sides of a target gap on their own grid fields (`--gap_split`; mpm/gap_split.py, gap_kernels.py, gap_g2p.py, gap_step.py).

(1) The target's gap nodes: two slabs a narrow gap apart give gap nodes in the gap, outside the target, with the plane
    across it; one slab gives none.
(2) No split away from gaps: with no gap node, or gap nodes every particle is on one side of, the split step is the
    single field's, bit for bit on the CPU (the forward) and to rounding (the gradients).
(3) The hand adjoints of the side-1 deposit and the split G2P equal Warp's generated ones (sides mixed; hard and soft).
(4) Two slabs of one body pulled apart separate cleanly with the split and stay coupled on the single field; the
    momentum is kept.
(5) The sides from the transport on a synthetic gap: a block filling a target's gap sends each half to its own slab.
(6) The tape through a split matches central differences.
(7) Variant B: below one cell everywhere it is A, at or above one cell everywhere it is the single field.
(8) The split G2P's adjoint in cell order equals the per-particle one (hard and soft).
(9) Soft sides (D148c) at a vanishing blur are the hard ones.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
import warp as wp

from physmorph.mpm.function import PersistentAdjoint, RolloutSpec
from physmorph.mpm.gap_split import GapPlanes
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory, compute_rest_volumes
from test_runtime_exact import _tape_outputs
from test_volume_exact import _window_case


def _slabs(gap_cells: float, dx: float, h: float, seed=0, one=False):
    """Two slabs |y| >= gap / 2 (or one slab y >= gap / 2), 8 x 3 x 8 cells, jittered points at spacing h."""
    rng = np.random.default_rng(seed)
    g = np.arange(-4 * dx, 4 * dx, h) + 0.5 * h
    gy = np.arange(0.5 * gap_cells * dx, (0.5 * gap_cells + 3) * dx, h) + 0.5 * h
    top = np.stack(np.meshgrid(g, gy, g, indexing="ij"), -1).reshape(-1, 3)
    pts = top if one else np.concatenate([top, top * np.array([1.0, -1.0, 1.0])])
    return (pts + rng.uniform(-0.2 * h, 0.2 * h, pts.shape)).astype(np.float64)


def test_gap_planes_find_a_narrow_gap():
    dx, h, n = 1.0, 0.2, 16
    gmin = np.full(3, -8.0)
    gp = GapPlanes(_slabs(1.5, dx, h), dx, gmin, (n,) * 3, 0.5)
    ii = np.stack(np.nonzero(gp.mask.reshape((n,) * 3)), -1)
    assert len(ii) > 0
    xg = gmin + ii * dx
    inner = (np.abs(xg[:, 0]) <= 3) & (np.abs(xg[:, 2]) <= 3)          # away from the slabs' open ends
    assert inner.sum() > 0 and np.all(np.abs(xg[inner, 1]) < 0.75)     # outside the slabs: in the gap
    nn = gp.n[gp.mask][inner]
    assert np.all(np.abs(nn[:, 1]) > 0.95)                             # the plane across the gap
    assert np.all(np.abs(gp.m[gp.mask][inner][:, 1]) < 0.3)            # the medial plane
    assert GapPlanes(_slabs(1.5, dx, h, one=True), dx, gmin, (n,) * 3, 0.5).n_gap == 0


def _planes(prm, m, n):
    ng = prm.nx * prm.ny * prm.nz
    return np.tile(np.asarray(m, np.float32), (ng, 1)), np.tile(np.asarray(n, np.float32), (ng, 1))


@pytest.mark.parametrize("case", ["no_gap_node", "one_side"])
def test_no_split_away_from_gaps_is_the_single_field(case):
    spec, dc, u = _window_case("cpu")
    N = spec.x0.shape[0]
    if case == "no_gap_node":
        pm, pn = _planes(spec.prm, (0, 0, 0), (0, 0, 0))
    else:                                                             # gap nodes everywhere, every particle on side 0
        pm, pn = _planes(spec.prm, (0, 100.0, 0), (0, 1.0, 0))
    q = np.asarray(spec.x0, np.float32)
    frozen = _tape_outputs(dataclasses.replace(spec, track_geom=False), dc, u)
    split = _tape_outputs(dataclasses.replace(spec, track_geom=False, gap=(pm, pn, q, 0.0)), dc, u)
    for a, b in zip(frozen[:4], split[:4]):                           # x, F, v, every step's v: bit for bit
        assert torch.equal(a, b)
    for a, b in zip(frozen[4:], split[4:]):
        torch.testing.assert_close(b, a, rtol=1e-6, atol=1e-7 * float(a.abs().max()))


@pytest.mark.parametrize("sig", [0.0, 0.1])
def test_split_hand_adjoints_are_the_generated_ones(monkeypatch, sig):
    """The side-1 deposit's and the split G2P's hand adjoints (hard sides; soft sides at sig 0.1, D148c) against
    Warp's generated ones: the forward bit for bit, the gradients to rounding."""
    import physmorph.mpm.traj as TR
    spec, dc, u = _window_case("cpu")
    pm, pn = _planes(spec.prm, (0, 0, 0), (0, 1.0, 0))               # sides by the sign of y: mixed in every stencil
    spec = dataclasses.replace(spec, track_geom=False, gap=(pm, pn, np.asarray(spec.x0, np.float32), 0.0, sig))
    monkeypatch.setattr(TR, "HAND_ADJOINTS", False)
    gen = _tape_outputs(spec, dc, u)
    monkeypatch.setattr(TR, "HAND_ADJOINTS", True)
    hand = _tape_outputs(spec, dc, u)
    for a, b in zip(gen[:4], hand[:4]):
        assert torch.equal(a, b)
    for a, b in zip(gen[4:], hand[4:]):
        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-5 * float(a.abs().max()))
        assert float(a.abs().max()) > 0


def _two_slab_body(dev, T, split: bool, vp=0.0, speed=1.0, sig=0.0):
    """One body: two slabs touching at y = 0 (a block), the upper half moving up, the lower down."""
    dx, h = 0.5, 0.125
    prm = MPMParams(dx=dx, dt=1.0 / 240.0, drag=0.0, smoothing=0.955, grid_min=(-4.0,) * 3, nx=16, ny=16, nz=16)
    rng = np.random.default_rng(3)
    g = np.arange(-1.0, 1.0, h) + 0.5 * h
    x = np.stack(np.meshgrid(g, np.arange(-0.75, 0.75, h) + 0.5 * h, g, indexing="ij"), -1).reshape(-1, 3)
    x = (x + rng.uniform(-0.2 * h, 0.2 * h, x.shape)).astype(np.float32)
    up = x[:, 1] > 0
    v0 = np.zeros_like(x)
    v0[:, 1] = np.where(up, speed, -speed)
    vol0 = compute_rest_volumes(x, 1.0, prm, dev)
    gap = None
    if split:
        pm, pn = _planes(prm, (0, 0, 0), (0, 1.0, 0))
        gap = (pm, pn, x + v0, vp, sig)                               # the side point: where each half is going
    return RolloutSpec(x0=x, m=1.0, lam=4.0e4, mu=2.0e4, prm=prm, T=T, v0=v0, device=dev, vol0=vol0,
                       polar_adjoint=True, track_geom=False, volume_exact="carried", gap=gap), up


def _roll(spec, dc=None):
    T, N = spec.T, spec.x0.shape[0]
    seq = None
    if dc is not None:
        seq = [wp.from_torch(dc[t].contiguous(), dtype=wp.mat33) for t in range(T)]
    tr = Trajectory(spec.x0, spec.m, spec.lam, spec.mu, spec.prm, T, v0=spec.v0, dFc=seq, device=spec.device,
                    requires_grad=False, vol0=spec.vol0, polar_adjoint=True, volume_exact="carried", gap=spec.gap)
    tr.rollout()
    return tr


def test_two_slabs_of_one_body_separate():
    T = 60
    spec_s, up = _two_slab_body("cpu", T, True)
    spec_1, _ = _two_slab_body("cpu", T, False)
    tr_s, tr_1 = _roll(spec_s), _roll(spec_1)
    vs, v1 = wp.to_torch(tr_s.v[T]).numpy(), wp.to_torch(tr_1.v[T]).numpy()
    rel_s = vs[up, 1].mean() - vs[~up, 1].mean()
    rel_1 = v1[up, 1].mean() - v1[~up, 1].mean()
    assert rel_s > 1.8                                                # the halves keep parting at about 2
    assert abs(rel_1) < 0.5 * rel_s                                   # the single field holds them together
    xs = wp.to_torch(tr_s.x[T]).numpy()
    assert xs[up, 1].min() - xs[~up, 1].max() > 0.25                  # a clean gap between the halves
    assert np.all(np.sign(vs[:, 1]) == np.where(up, 1.0, -1.0))       # no particle left on the other side
    p0 = (spec_s.v0 * 1.0).sum(0)
    pT = vs.astype(np.float64).sum(0)
    assert np.allclose(pT, p0, atol=1e-3 * np.abs(spec_s.v0).sum())  # the momentum kept (rounding)
    rec = torch.stack([wp.to_torch(s) for s in tr_s.gap_stats]).sum(0)
    assert float(rec[2]) > 0 and float(rec[3]) > 0                   # split nodes, separating


@pytest.mark.parametrize("vp, same_as", [(1e-9, "A"), (1e9, "single")])
def test_variant_b_limits(vp, same_as):
    T = 20
    spec_b, _ = _two_slab_body("cpu", T, True, vp=vp)
    ref, _ = _two_slab_body("cpu", T, same_as == "A")
    xb, xr = wp.to_torch(_roll(spec_b).x[T]), wp.to_torch(_roll(ref).x[T])
    assert torch.equal(xb, xr)


def test_sides_from_the_transport_on_a_synthetic_gap():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from physmorph.losses.grid_ot import grid_transport_displacement
    from physmorph.pipeline.target import target_mass_grid
    dx, h, n = 1.0, 0.2, 16
    gmin = np.full(3, -8.0)
    tgt = _slabs(1.5, dx, h)
    gp = GapPlanes(tgt, dx, gmin, (n,) * 3, 0.5)
    rng = np.random.default_rng(1)                                    # the source: one block over both slabs and the gap
    x = rng.uniform([-4, -3.75, -4], [4, 3.75, 4], (len(tgt), 3)).astype(np.float32)
    xt = torch.tensor(x, device="cuda")
    m = torch.ones(len(x), device="cuda")
    lgmin = torch.tensor(gmin, device="cuda", dtype=torch.float32)
    tg = target_mass_grid(torch.tensor(tgt, dtype=torch.float32, device="cuda"), m, lgmin, dx, (n,) * 3)
    disp = grid_transport_displacement(xt, m, tg, lgmin, dx, (n,) * 3, eps=dx ** 2, iters=1600, tol=1e-3)
    q = (xt + disp).cpu().numpy().astype(np.float64)
    gnodes = np.nonzero(gp.mask)[0]
    ii = np.stack(np.unravel_index(gnodes, (n,) * 3), -1)
    xg = gmin + ii * dx
    near = np.abs(x[:, None, :] - xg[None]).max(2) < 2 * dx           # particle in the node's kernel reach
    p, k = np.nonzero(near)
    s_img = gp.side(q[p], gnodes[k])
    s_pos = gp.side(x[p].astype(np.float64), gnodes[k])
    assert (s_img == s_pos).mean() > 0.9                              # each half to its own slab
    # no particle alone on its side among a node's particles
    for g in np.unique(k):
        sel = k == g
        c = np.bincount(s_img[sel], minlength=2)
        assert c.min() == 0 or c.min() > 1


@pytest.mark.parametrize("sig", [0.0, 0.2])
def test_split_tape_matches_finite_differences(sig):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    T = 16
    spec, _ = _two_slab_body("cuda", T, True, speed=0.3, sig=sig)
    adj = PersistentAdjoint(spec)
    N = spec.x0.shape[0]
    D = torch.randn(T, N, 3, 3, generator=torch.Generator().manual_seed(5)).cuda() * 0.01
    W = torch.randn(N, 3, generator=torch.Generator().manual_seed(6)).cuda()

    def loss(eps, grad=False):
        dfc = (eps * D).requires_grad_(grad)
        xT, FT, vT, _, V = adj.apply(dfc, None)
        L = (xT * W).sum() + 0.1 * (V[-3] * W).sum() + 0.01 * FT.square().sum()
        if grad:
            (g,) = torch.autograd.grad(L, dfc)
            return float(L), float((g * D).sum())
        return float(L)

    _, an = loss(0.0, grad=True)
    h = 0.1
    fd = (loss(h) - loss(-h)) / (2 * h)
    assert abs(an - fd) <= 0.05 * abs(fd) + 1e-6, (an, fd)


@pytest.mark.parametrize("sig", [0.0, 0.1])
def test_split_g2p_adjoint_in_cell_order_is_the_plain_one(monkeypatch, sig):
    """(8) The split G2P's adjoint in the trajectory's cell order (CUDA, lanes summing per field) gives the per-particle
    adjoint's gradients to float rounding (the order of the grid's sums only)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    import physmorph.mpm.gap_step as GK
    spec, dc, u = _window_case("cuda")
    pm, pn = _planes(spec.prm, (0, 0, 0), (0, 1.0, 0))
    spec = dataclasses.replace(spec, track_geom=False, gap=(pm, pn, np.asarray(spec.x0, np.float32), 0.0, sig))
    monkeypatch.setattr(GK, "WARP_ADJOINT", False)
    plain = _tape_outputs(spec, dc, u)
    monkeypatch.setattr(GK, "WARP_ADJOINT", True)
    warp = _tape_outputs(spec, dc, u)
    for a, b in zip(plain[:4], warp[:4]):
        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-6 * max(1.0, float(a.abs().max())))
    for a, b in zip(plain[4:], warp[4:]):
        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-5 * float(a.abs().max()))
        assert float(a.abs().max()) > 0


def test_soft_sides_with_a_vanishing_blur_are_the_hard_ones():
    """(9) D148c's soft shares at a blur far below the particles' distances to the planes are the hard sides, bit for
    bit on the CPU (the blend s v1 + (1 - s) v0 and the weighted deposits reduce to the hard selection)."""
    spec, dc, u = _window_case("cpu")
    pm, pn = _planes(spec.prm, (0, 0, 0), (0, 1.0, 0))
    q = np.asarray(spec.x0, np.float32)
    hard = _tape_outputs(dataclasses.replace(spec, track_geom=False, gap=(pm, pn, q, 0.0, 0.0)), dc, u)
    soft = _tape_outputs(dataclasses.replace(spec, track_geom=False, gap=(pm, pn, q, 0.0, 1e-12)), dc, u)
    for a, b in zip(hard[:4], soft[:4]):
        assert torch.equal(a, b)
    for a, b in zip(hard[4:], soft[4:]):
        torch.testing.assert_close(b, a, rtol=1e-6, atol=1e-7 * float(a.abs().max()))


def test_separable_side_points_are_the_dense_ones():
    """(10) D148b's side points (grid_transport_displacement_separable: the potentials from the loss grid's separable
    solve) equal grid_transport_displacement's (the dense solve) on a grid small enough for both, to the solves'
    tolerance."""
    from physmorph.losses.grid_ot import (GridSinkhornLoss, grid_transport_displacement,
                                          grid_transport_displacement_separable)
    from physmorph.losses.volumetric import target_mass_grid
    dx, n = 1.0, 16
    tgt = torch.tensor(_slabs(1.5, dx, 0.25), dtype=torch.float32)
    rng = np.random.default_rng(2)
    x = torch.tensor(rng.uniform([-4, -3.75, -4], [4, 3.75, 4], (800, 3)), dtype=torch.float32)
    m = torch.ones(len(x))
    lg = torch.full((3,), -8.0)
    tg = target_mass_grid(tgt, torch.ones(len(tgt)), lg, dx, (n,) * 3)
    dense = grid_transport_displacement(x, m, tg, lg, dx, (n,) * 3, eps=dx ** 2, iters=1600, tol=1e-4)
    ot = GridSinkhornLoss(tg, lg, dx, (n,) * 3, eps=dx ** 2, iters=1600, tol=1e-4)
    sep = grid_transport_displacement_separable(x, m, ot)
    assert float((sep - dense).norm(dim=1).max()) < 0.02 * dx


def _pieces_of(pts, inner_fn):
    """For every gap node (margin 1), whether its two pieces are one inner and one outer wall (by inner_fn on the
    pieces' cell centres): (nodes, nodes whose pieces are not one of each)."""
    dx, n = 1.0, 16
    gp = GapPlanes(pts, dx, np.full(3, -8.0), (n,) * 3, 0.5, keep_boxes=True, margin=1)
    bad = 0
    for g, (a0, lab, ca, cb) in gp.boxes.items():
        kinds = []
        for c in (ca, cb):
            cc = gp.lo + (np.stack(np.nonzero(lab == c), -1) + a0 + 0.5) * gp.cell
            f = inner_fn(cc)
            kinds.append(1 if f.all() else (0 if not f.any() else -1))
        bad += int(sorted(kinds) != [0, 1])
    return len(gp.boxes), bad


def test_pieces_of_two_slabs_are_the_two_slabs():
    """(11) D148c: every gap node's two pieces are the two slabs, one each."""
    nodes, bad = _pieces_of(_slabs(1.5, 1.0, 0.2), lambda c: c[:, 1] < 0)
    assert nodes > 0 and bad == 0


def test_pieces_of_a_curved_gap_are_the_rod_and_the_shell():
    """(12) D148c: a gap bent round a cylinder (an inner rod, an outer shell 1.5 cells off) whose nodes' planes face
    every way: every gap node's two pieces are the rod and the shell (the one-cell ring keeps a clipped curved wall
    one piece: without it, nodes outside the shell saw its clipped rim as two pieces)."""
    rng = np.random.default_rng(4)
    p = rng.uniform(-6, 6, (200000, 3))
    r = np.hypot(p[:, 0], p[:, 1])
    pts = p[((r < 2.0) | ((r > 3.5) & (r < 5.5))) & (np.abs(p[:, 2]) < 4)]
    nodes, bad = _pieces_of(pts, lambda c: np.hypot(c[:, 0], c[:, 1]) < 2.75)
    assert nodes > 0 and bad == 0


def test_representative_cells_persist_in_the_gap():
    """(13) D148c: an image in an occupied cell takes that cell; in an empty cell the cell from before is kept; with
    none before, the nearest occupied cell."""
    from physmorph.mpm.gap_walls import rep_cells
    nc = (4, 1, 1)
    occ = torch.tensor([True, False, False, True])
    nearest = torch.tensor([0, 0, 3, 3])
    q = torch.tensor([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [2.5, 0.5, 0.5], [3.5, 0.5, 0.5]])
    prev = torch.tensor([-1, 3, -1, 0])
    rep = rep_cells(q, prev, np.zeros(3), 1.0, nc, occ, nearest)
    assert rep.tolist() == [0, 3, 3, 3]


def _walls_spec(spec, rep_of, normal=(0, 1.0, 0), q_shift=(0, 0, 0)):
    """Gap planes everywhere, every node's box one occupancy row of two cells: cell 0 piece 1 (side 0), cell 1 piece 2
    (side 1); the particles' representative cells by rep_of."""
    ng = spec.prm.nx * spec.prm.ny * spec.prm.nz
    pm, pn = _planes(spec.prm, (0, 0, 0), normal)
    box0 = np.zeros((ng, 3), np.int32)
    boxd = np.tile(np.asarray([2, 1, 1], np.int32), (ng, 1))
    boxi = np.tile(np.asarray([0, 1, 2], np.int32), (ng, 1))
    labels = np.asarray([1, 2], np.int32)
    rep = rep_of(np.asarray(spec.x0)).astype(np.int32)
    q = (np.asarray(spec.x0) + np.asarray(q_shift)).astype(np.float32)
    return dataclasses.replace(spec, track_geom=False, gap=(pm, pn, q, 0.0, 0.0, (box0, boxd, boxi, labels, (2, 1, 1), rep)))


def test_walls_hand_adjoints_are_the_generated_ones(monkeypatch):
    """(14) D148c's sides by the pieces through the deposit's and the G2P's hand adjoints against the generated ones."""
    import physmorph.mpm.traj as TR
    spec, dc, u = _window_case("cpu")
    spec = _walls_spec(spec, lambda x: (x[:, 1] < 0))           # the pieces against the plane: side 1 below
    monkeypatch.setattr(TR, "HAND_ADJOINTS", False)
    gen = _tape_outputs(spec, dc, u)
    monkeypatch.setattr(TR, "HAND_ADJOINTS", True)
    hand = _tape_outputs(spec, dc, u)
    for a, b in zip(gen[:4], hand[:4]):
        assert torch.equal(a, b)
    for a, b in zip(gen[4:], hand[4:]):
        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-5 * float(a.abs().max()))


def test_walls_with_one_wall_only_are_the_single_field():
    """(15) D148c with every particle in side 0's piece: the single field, bit for bit (the forward)."""
    spec, dc, u = _window_case("cpu")
    frozen = _tape_outputs(dataclasses.replace(spec, track_geom=False), dc, u)
    walls = _tape_outputs(_walls_spec(spec, lambda x: np.zeros(len(x))), dc, u)
    for a, b in zip(frozen[:4], walls[:4]):
        assert torch.equal(a, b)


def test_pieces_decide_where_they_hold_the_cell():
    """(16) D148c: where the representative cell lies in the node's box, the pieces decide: with the side points moved
    far to side 0 of every plane (normal -y) and the pieces putting the particles below y = 0 on side 1, the forward
    equals the plane test with the side points in place."""
    spec, dc, u = _window_case("cpu")
    by_pieces = _tape_outputs(_walls_spec(spec, lambda x: (x[:, 1] < 0), normal=(0, -1.0, 0), q_shift=(0, 10.0, 0)),
                              dc, u)
    pm, pn = _planes(spec.prm, (0, 0, 0), (0, -1.0, 0))
    by_plane = _tape_outputs(dataclasses.replace(spec, track_geom=False,
                                                 gap=(pm, pn, np.asarray(spec.x0, np.float32), 0.0)), dc, u)
    for a, b in zip(by_plane[:4], by_pieces[:4]):
        assert torch.equal(a, b)
