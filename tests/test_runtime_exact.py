"""The runtime phase after the freeze (tag freeze-2026-10-09): every speed-up in this file leaves the result as it was.

Each test runs the path before the change and the path after it on the same case and asks for the same numbers: bit
for bit on the CPU (Warp's CPU kernels are deterministic), and within the transfers' atomics on CUDA, where two
rollouts of one control already differ (the replay noise every window measures).

(1) The tape without the geometric deformation (RolloutSpec.track_geom False, the window's spec): no term of the
    settled path reads Fg, and leaving its kernels and buffers out changes no other output and no gradient.
(2) The relaxation's reference (window/layer.TargetRelief.at) looks up only the layer's particles on the target's
    surface: the same reference, bit for bit, as the lookup of the whole body.
(3) The hand-written adjoints of P2G, G2P, the update and the layer's projection (mpm/adjoints.py) give Warp's
    generated adjoints' gradients to float rounding (a different order of the same arithmetic; sums over particles
    accumulate with atomics in both).
(4) The Sinkhorn sweep's axis pass divides the cost by the temperature once per call: bit for bit.
(5) The window's exterior search reads the field in one Warp kernel: the tensor form's field to float rounding.
(6) D50 (ported by the user's approval of 2026-10-09): the replay pair takes the warm start's evaluation as its first
    rollout, unless the exterior's discs were looked for again after it.
(7) D53 (ported by the user's approval of 2026-10-09): the record's transport energy takes the commit's potentials.
(8) D138 batch 4: P2G and G2P's adjoint visit the particles in cell order and the lanes of a warp that share a
    stencil sum before their atomics (CUDA): the grid, the tape's outputs and the gradients equal the plain
    kernels' to float rounding (the order of the sums only), with fragment particles, invalid positions and lanes in
    several cells; G2P's adjoint in cell order reads the forward's new affine field (batch 5).
(9) D138 batch 5: the trajectory's minimum determinant step by step on its own buffers: bit for bit.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from physmorph.mpm.function import PersistentAdjoint
from test_volume_exact import _expand, _window_case


def _dev_or_skip(dev):
    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")


def _tape_outputs(spec, dc, u):
    """The persistent tape's outputs at (dc, u) and the gradients of a loss on x, F, v and every step's velocity
    (the settled objective's arguments), on both leaves."""
    adj = PersistentAdjoint(spec)
    dc, u = dc.clone().requires_grad_(True), u.clone().requires_grad_(True)
    xT, FT, vT, _, V = adj.apply(_expand(dc), u)
    w = torch.linspace(0.5, 1.5, 3, device=xT.device)
    L = (xT * w).pow(2).sum() * 1e2 + FT.pow(2).sum() + V.pow(2).sum() * 1e2 + (vT * w).sum()
    g_dc, g_u = torch.autograd.grad(L, (dc, u))
    return [t.detach() for t in (xT, FT, vT, V, g_dc, g_u)]


@pytest.mark.parametrize("dev", ["cpu", "cuda"])
@pytest.mark.parametrize("volume", ["off", "carried"])
def test_tape_without_the_geometric_deformation_is_the_same(dev, volume):
    _dev_or_skip(dev)
    spec, dc, u = _window_case(dev)
    if volume != "off":
        N = spec.x0.shape[0]
        J0 = (np.linalg.det(spec.F0) * np.random.default_rng(5).uniform(0.97, 1.03, N)).astype(np.float32)
        spec = dataclasses.replace(spec, volume_exact=volume, J0=J0)
    with_geom = _tape_outputs(dataclasses.replace(spec, track_geom=True), dc, u)
    without = _tape_outputs(dataclasses.replace(spec, track_geom=False), dc, u)
    for a, b in zip(with_geom, without):
        if dev == "cpu":
            assert torch.equal(a, b)
        else:
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6 * max(1.0, float(a.abs().max())))
    assert float(with_geom[4].abs().max()) > 0 and float(with_geom[5].abs().max()) > 0


def _relief_whole_body(relief, x, mask, nrm, nbr, w):
    """TargetRelief.at as frozen (tag freeze-2026-10-09): every particle looked up on the surface."""
    d, i = relief.tree.query(x, 1)
    i = i.reshape(-1)
    q, m = relief.points[i], relief.normals[i]
    foot = x - ((x - q) * m).sum(1, keepdim=True) * m
    res = (nrm * (foot - (w[..., None] * foot[nbr]).sum(1))).sum(1)
    near = (mask > 0.5) & (d.float().reshape(-1) < relief.reach)
    return torch.where(near, res - (w * res[nbr]).sum(1), torch.zeros((), device=x.device))


def test_relief_reference_from_the_layer_alone_is_the_same():
    _dev_or_skip("cuda")
    from physmorph.pipeline.window.layer import TargetRelief, layer_relax_data, layer_spacing
    rng = np.random.default_rng(11)
    g = np.arange(24, dtype=np.float32) * 0.1
    X = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    x = torch.tensor(X - X.mean(0) + rng.uniform(-0.03, 0.03, X.shape).astype(np.float32), device="cuda")
    # the target's surface: a sphere a little inside the block's corners, so that some layer particles are within
    # one spacing of it and some are not
    u = rng.normal(size=(20000, 3)).astype(np.float32)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    pts = torch.tensor(1.25 * u, device="cuda")
    nrm_s = torch.tensor(u, device="cuda")
    sp = layer_spacing(x)
    relief = TargetRelief(pts, nrm_s, sp)
    mask, nrm, nbr, w = layer_relax_data(x, sp, k=24, h_sp=2.0)
    got = relief.at(x, mask, nrm, nbr, w)
    ref = _relief_whole_body(relief, x, mask, nrm, nbr, w)
    assert torch.equal(got, ref)
    assert int((got != 0).sum()) > 50 and int(((mask > 0.5) & (got == 0)).sum()) > 50


@pytest.mark.parametrize("dev", ["cpu", "cuda"])
@pytest.mark.parametrize("volume", ["off", "carried"])
def test_hand_written_transfer_adjoints_match_the_generated_ones(dev, volume, monkeypatch):
    """(3) P2G's and G2P's hand-written adjoints (mpm/adjoints.py) against Warp's generated ones on the window case
    (bonds with two fragment particles, the layer and u, the minimum spacing, driven and released steps): the forward
    is the same kernel (bit for bit on the CPU), the gradients agree to float rounding."""
    _dev_or_skip(dev)
    import physmorph.mpm.traj as TR
    spec, dc, u = _window_case(dev)
    if volume != "off":
        N = spec.x0.shape[0]
        J0 = (np.linalg.det(spec.F0) * np.random.default_rng(5).uniform(0.97, 1.03, N)).astype(np.float32)
        spec = dataclasses.replace(spec, volume_exact=volume, J0=J0)
    spec = dataclasses.replace(spec, track_geom=False)
    monkeypatch.setattr(TR, "HAND_ADJOINTS", False)
    gen = _tape_outputs(spec, dc, u)
    monkeypatch.setattr(TR, "HAND_ADJOINTS", True)
    hand = _tape_outputs(spec, dc, u)
    for a, b in zip(gen[:4], hand[:4]):                       # the forward
        if dev == "cpu":
            assert torch.equal(a, b)
        else:
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6 * max(1.0, float(a.abs().max())))
    for a, b in zip(gen[4:], hand[4:]):                       # the gradients on dc and u
        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-5 * float(a.abs().max()))
        assert float(a.abs().max()) > 0


def _old_axis_kernel():
    """The frozen code's axis kernel (tag freeze-2026-10-09): cost / temperature divided per element."""
    import warp as wp

    @wp.kernel(enable_backward=False)
    def k(field: wp.array(dtype=float), cost: wp.array2d(dtype=float), temperature: wp.array(dtype=float),
          width: int, stride: int, result: wp.array(dtype=float)):
        i = wp.tid()
        coordinate = (i // stride) % width
        start = i - coordinate * stride
        maximum = float(-wp.inf)
        for j in range(width):
            value = field[start + j * stride] - cost[coordinate, j] / temperature[0]
            if wp.isnan(value):
                result[i] = value
                return
            maximum = wp.max(maximum, value)
        if not wp.isfinite(maximum):
            result[i] = maximum
            return
        total = float(0.)
        for j in range(width):
            value = field[start + j * stride] - cost[coordinate, j] / temperature[0]
            total += wp.exp(value - maximum)
        result[i] = maximum + wp.log(total)
    return k


def test_sinkhorn_axis_pass_with_the_cost_table_is_the_same():
    """(4) the Sinkhorn sweep's separable log-sum-exp with cost / temperature divided once per call: the frozen
    kernel's output bit for bit (empty nodes, three temperatures, every axis)."""
    _dev_or_skip("cuda")
    import warp as wp
    from physmorph.losses.grid_ot import GridSinkhornLoss
    torch.manual_seed(3)
    dims = (19, 23, 17)
    tgt = torch.rand(dims, device="cuda").reshape(-1) * (torch.rand(int(np.prod(dims)), device="cuda") > 0.5)
    loss = GridSinkhornLoss(tgt, torch.zeros(3, device="cuda"), 0.1, dims, eps=0.01, iters=400, tol=1e-3,
                            cuda_blocks=True)
    old = _old_axis_kernel()
    for T in (0.01, 0.37, 21.0):
        dual = torch.randn(tgt.numel(), device="cuda") * T * 3
        lw = (torch.rand(tgt.numel(), device="cuda") * (torch.rand(tgt.numel(), device="cuda") > 0.3)).log()
        with torch.no_grad():
            got = loss.transform(dual, lw, T)
            temp = torch.tensor([T], device="cuda")
            field = (dual / temp + lw).contiguous()
            stride = field.numel()
            for width, cost in zip(dims, loss.costs):
                stride //= width
                out = torch.empty_like(field)
                wp.launch(old, dim=field.numel(), inputs=[wp.from_torch(field), wp.from_torch(cost),
                                                          wp.from_torch(temp), width, stride],
                          outputs=[wp.from_torch(out)])
                field = out
            ref = -temp * field
        assert torch.equal(torch.nan_to_num(got, nan=7.0), torch.nan_to_num(ref, nan=7.0))


def test_device_field_is_the_tensor_field_to_rounding():
    """(5) the exterior's field in one Warp kernel (render/exterior_wp.py, the window's disc search) against the tensor
    form: the same values, gradients and weights to float rounding at lattice nodes in and around a ball, the same
    crossed cells but for nodes within rounding of the zero set, and the same discs."""
    _dev_or_skip("cuda")
    from physmorph.render.exterior import Lattice, ZhuBridson
    gen = torch.Generator().manual_seed(4)
    r = torch.arange(-8, 9, dtype=torch.float32)
    x = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    x = (x + .2 * (2 * torch.rand(x.shape, generator=gen) - 1))
    x = x[x.norm(dim=1) < 7.].cuda()
    ten, dev = ZhuBridson(x, 1.), ZhuBridson(x, 1., device_field=True)
    lat = Lattice(torch.zeros(3, device="cuda"), 30.)
    q = lat.at(torch.randint(45, 115, (20000, 3), device="cuda"), .4)
    ft, gt, st = ten(q)
    fd, gd, sd = dev(q)
    fin = torch.isfinite(ft)
    assert torch.equal(fin, torch.isfinite(fd)) and int(fin.sum()) > 1000
    assert float((ft[fin] - fd[fin]).abs().max()) < 1e-4
    assert float((gt[fin] - gd[fin]).abs().max()) < 1e-3
    torch.testing.assert_close(sd, st, rtol=1e-5, atol=1e-6)
    pt, nt, ct, _ = lat.discs(ten, .4, refine=False)
    pd, nd, cd, _ = lat.discs(dev, .4, refine=False)
    assert abs(ct - cd) <= 2 and abs(len(pt) - len(pd)) <= 2 and len(pt) > 500
    d = torch.cdist(pd, pt, compute_mode="donot_use_mm_for_euclid_dist").min(1).values
    assert float(d.max()) < 1e-4


@pytest.mark.parametrize("exterior", [False, True])
def test_replay_pair_takes_the_warm_start_evaluation(monkeypatch, exterior):
    """(6) D50 (tag settled-2026-10-03-d53, ported by the user's approval of 2026-10-09): a window with a warm start
    measures the replay noise with the warm start's evaluation of the kept control as the pair's first rollout and one
    more rollout, three evaluations at the start instead of four; when the exterior's discs were looked for again after
    that evaluation (at the other control), the pair is rolled out anew so that both of its rollouts read the same
    discs. The pair's measure is the same quantity: finite, of the replay noise's size."""
    _dev_or_skip("cuda")
    import physmorph.pipeline.window.solve as S
    from physmorph.mpm.state import MPMParams
    from physmorph.pipeline import PipelineConfig, run_pipeline
    rng = np.random.default_rng(11)
    src = rng.uniform(-1.5, 1.5, (300, 3)).astype(np.float32)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
    prm = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    cfg = PipelineConfig(T=4, iters=2, animations=3, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                         render_res=24, dt_res=32, patience=3, render_exterior=exterior)
    ev, rn = S.WindowOptimizer.eval, S.WindowOptimizer.replay_noise
    rows = []

    def eval_(self):
        self._n_eval = getattr(self, "_n_eval", 0) + 1
        return ev(self)

    def replay(self, start=None):
        n, builds = getattr(self, "_n_eval", 0), self.obj.ext_builds
        rel = rn(self, start)
        reuse = start is not None and start[1] == builds
        rows.append((start is not None, reuse, self._n_eval - n, rel, start[0] if start is not None else None))
        return rel

    monkeypatch.setattr(S.WindowOptimizer, "eval", eval_)
    monkeypatch.setattr(S.WindowOptimizer, "replay_noise", replay)
    run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    assert len(rows) >= 2 and (exterior or any(r[0] and r[1] for r in rows))
    for warm, reuse, n, rel, _ in rows:
        assert n == (1 if reuse else 2)                 # one rollout against the warm start's, else a pair anew
        assert np.isfinite(rel) and 0.0 <= rel < 1e-3
        if not exterior:
            assert reuse == warm                        # without the exterior every warm start is reused


def test_record_takes_the_commits_potentials(monkeypatch):
    """(7) D53 (tag settled-2026-10-03-d53, ported by the user's approval of 2026-10-09): the window's record reads the
    transport energy at the promoted state, which is the commit rollout's own end state unless a guard repaired it; it
    takes the potentials the commit solved there and solves nothing, and the energy is a fresh solve's to the
    rasterisation's own rounding."""
    _dev_or_skip("cuda")
    import physmorph.pipeline.run.runner as RN
    from physmorph.losses.grid_ot import GridSinkhornLoss
    from physmorph.mpm.state import MPMParams
    from physmorph.pipeline import PipelineConfig, run_pipeline
    rng = np.random.default_rng(11)
    src = rng.uniform(-1.5, 1.5, (300, 3)).astype(np.float32)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
    prm = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    cfg = PipelineConfig(T=4, iters=2, animations=3, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                         render_res=24, dt_res=32, patience=3, render_exterior=True)
    solve, rec = GridSinkhornLoss.solve, RN._record
    n = {"solves": 0}
    rows = []

    def counting(self, a, b):
        n["solves"] += 1
        return solve(self, a, b)

    def record(a, res, x, x_start, v, F, counts, commit, tgt_, cfg_, prm_, thin=None, J=None):
        before = n["solves"]
        out = rec(a, res, x, x_start, v, F, counts, commit, tgt_, cfg_, prm_, thin, J)
        solved = n["solves"] - before
        fresh = float(tgt_.grid_ot.state_energy(x, tgt_.m))
        rows.append((bool(counts["clamped"] or counts["nan_x"]), solved, out["transport_energy"], fresh))
        return out

    monkeypatch.setattr(GridSinkhornLoss, "solve", counting)
    monkeypatch.setattr(RN, "_record", record)
    run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    assert rows
    for repaired, solved, energy, fresh in rows:
        assert solved == (2 if repaired else 0)        # the cross and the self problem, only after a repair
        assert abs(energy - fresh) <= 1e-5 * max(abs(fresh), 1e-12)


def test_cell_ordered_p2g_is_the_plain_kernel_to_rounding():
    """(8) kernels.k_p2g_warp against k_p2g on a dense cloud (about 150 particles a cell, so a warp's lanes share a
    stencil and some sit in the next cell), with fragment particles on bonds and two invalid positions."""
    _dev_or_skip("cuda")
    import warp as wp
    import physmorph.mpm.kernels as K
    rng = np.random.default_rng(42)
    N, nx = 20000, 16
    x = rng.uniform(4.0, 9.0, (N, 3)).astype(np.float32)          # 5^3 cells of a 16^3 grid, dx 1
    x[17] = np.nan
    x[18] = 1.0e6
    A = lambda a, t: wp.array(a, dtype=t, device="cuda")          # noqa: E731
    v = A(rng.normal(0, .3, (N, 3)).astype(np.float32), wp.vec3)
    C = A(rng.normal(0, .2, (N, 3, 3)).astype(np.float32), wp.mat33)
    F = A(np.tile(np.eye(3, dtype=np.float32), (N, 1, 1)) + rng.normal(0, .05, (N, 3, 3)).astype(np.float32), wp.mat33)
    dfc = A(rng.normal(0, .02, (N, 3, 3)).astype(np.float32), wp.mat33)
    P = A(rng.normal(0, 1., (N, 3, 3)).astype(np.float32), wp.mat33)
    m = A(rng.uniform(.5, 1.5, N).astype(np.float32), float)
    vol = A(rng.uniform(.5, 1.5, N).astype(np.float32), float)
    K_ = 4
    nbr = A(rng.integers(0, N, N * K_).astype(np.int32), wp.int32)
    frag = A((rng.uniform(0, 1, N) < .02).astype(np.float32), float)
    order = np.argsort(np.floor(np.nan_to_num(x, nan=1e9)).astype(np.int64) @ np.array([20 * 20, 20, 1]), kind="stable")
    order = A(order.astype(np.int32), wp.int32)
    xa = A(x, wp.vec3)
    out = []
    for warp_order in (False, True):
        gm = wp.zeros(nx ** 3, dtype=float, device="cuda")
        gv = wp.zeros(nx ** 3, dtype=wp.vec3, device="cuda")
        args = [xa, v, C, F, dfc, P, m, vol, nbr, frag, K_, gm, gv, wp.vec3(0., 0., 0.), 1.0, 1.0, 1 / 240, 0.9,
                nx, nx, nx]
        if warp_order:
            wp.launch_tiled(K.k_p2g_warp, dim=[(N + K.WARP_LANES - 1) // K.WARP_LANES], inputs=[order, N] + args,
                            block_dim=K.WARP_LANES, device="cuda")
        else:
            wp.launch(K.k_p2g, dim=N, inputs=args, device="cuda")
        out.append((gm.numpy(), gv.numpy()))
    (m0, v0), (m1, v1) = out
    assert m0.max() > 100 and np.isfinite(m1).all() and np.isfinite(v1).all()
    np.testing.assert_allclose(m1, m0, rtol=1e-5, atol=1e-5 * m0.max())
    np.testing.assert_allclose(v1, v0, rtol=1e-5, atol=1e-5 * np.abs(v0).max())


@pytest.mark.parametrize("volume", ["off", "carried"])
def test_cell_ordered_transfers_keep_the_tape(volume, monkeypatch):
    """(8) The window case on CUDA through the production tape, cell order on and off: the forward and the gradients
    on dc and u to float rounding (the hand-written G2P adjoint in cell order, adjoints.k_g2p_adj_warp)."""
    _dev_or_skip("cuda")
    import physmorph.mpm.traj as TR
    spec, dc, u = _window_case("cuda")
    if volume != "off":
        N = spec.x0.shape[0]
        J0 = (np.linalg.det(spec.F0) * np.random.default_rng(5).uniform(0.97, 1.03, N)).astype(np.float32)
        spec = dataclasses.replace(spec, volume_exact=volume, J0=J0)
    spec = dataclasses.replace(spec, track_geom=False)
    monkeypatch.setattr(TR, "ORDERED_TRANSFERS", False)
    plain = _tape_outputs(spec, dc, u)
    monkeypatch.setattr(TR, "ORDERED_TRANSFERS", True)
    ordered = _tape_outputs(spec, dc, u)
    for a, b in zip(plain[:4], ordered[:4]):
        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-6 * max(1.0, float(a.abs().max())))
    for a, b in zip(plain[4:], ordered[4:]):
        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-5 * float(a.abs().max()))
        assert float(a.abs().max()) > 0


def test_trajectory_min_det_step_by_step_is_the_stacked_one():
    """(9) rollout._trajectory_min_det over the trajectory's own buffers, step by step: the same value bit for bit as the
    frozen code's one batched determinant over the stacked steps (tag freeze-2026-10-09b), volume off and carried."""
    _dev_or_skip("cuda")
    import types
    import warp as wp
    from physmorph.pipeline.window.rollout import _trajectory_min_det
    rng = np.random.default_rng(9)
    T, N = 6, 50000
    for carried in (False, True):
        Fs = [wp.array((np.eye(3) + rng.normal(0, .3, (N, 3, 3))).astype(np.float32), dtype=wp.mat33, device="cuda")
              for _ in range(T + 1)]
        Js = [wp.array(rng.uniform(.5, 1.5, N).astype(np.float32), dtype=float, device="cuda") for _ in range(T + 1)]
        tr = types.SimpleNamespace(F=Fs, J=Js, volume_exact=carried)
        win = types.SimpleNamespace(tr=tr, T=T, N=N)
        dc = torch.tensor(rng.normal(0, .1, (T, N, 9)).astype(np.float32), device="cuda")
        F_post = torch.stack([wp.to_torch(tr.F[t]).reshape(N, 3, 3) for t in range(1, T + 1)])
        F_pre = torch.stack([wp.to_torch(tr.F[t]).reshape(N, 3, 3) for t in range(T)])
        jt = torch.minimum(torch.linalg.det(F_post).min(), torch.linalg.det(F_pre + dc.view(T, N, 3, 3)).min())
        if carried:
            jt = torch.minimum(jt, torch.stack([wp.to_torch(tr.J[t]) for t in range(1, T + 1)]).min())
        assert _trajectory_min_det(win, dc) == float(jt)
