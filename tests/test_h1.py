import torch

from physmorph.losses.volumetric import d_h1, d_vol, rasterize_mass


def _two_blobs():
    g = torch.Generator().manual_seed(0)
    A = torch.tensor([10.0, 16.0, 16.0]) + torch.randn(200, 3, generator=g)
    B = torch.tensor([22.0, 16.0, 16.0]) + torch.randn(200, 3, generator=g)
    return A, B, g


def test_h1_zero_at_match_and_surplus_drains_toward_distant_deficit():
    dims, dx, gmin = (32, 32, 32), 1.0, torch.zeros(3)
    A, B, g = _two_blobs()
    tgt = torch.cat([A, B]); m = torch.ones(400)
    tg = rasterize_mass(tgt, m, gmin, dx, dims)
    # r == 0 -> the spectral part vanishes; only the (constant) self-energy remains
    L0 = float(d_h1(tgt, m, tg, gmin, dx, dims, self_correct=False))
    assert L0 < 1e-9
    # all mass piled at A: the deficit at B is 12 cells away
    cur = torch.cat([A, A + 0.3 * torch.randn(200, 3, generator=g)]).requires_grad_(True)
    L = d_h1(cur, m, tg, gmin, dx, dims)
    (gh,) = torch.autograd.grad(L, cur)
    assert torch.isfinite(gh).all()
    # the NET descent of the surplus blob points at B (+x); each particle's own
    # gradient is dominated by the blob's Coulomb self-repulsion (decompression), which
    # is the physics, so only the mean is tested here
    step = -gh.mean(0)
    assert step[0] > 0 and step[0] > 3 * abs(step[1]) and step[0] > 3 * abs(step[2])


def test_h1_self_force_is_exactly_removed():
    """A lone particle against an EMPTY target: the residual is its own CIC cloud, so
    the uncorrected loss is pure self-energy (a function of the sub-cell position with
    a minimum at the cell centre - the lattice attractor, REFUTE Opus F1). With the
    P3M self-energy subtraction the loss and its gradient are zero at every position."""
    dims, dx, gmin = (16, 16, 16), 1.0, torch.zeros(3)
    tg = torch.zeros(16 ** 3)
    m = torch.ones(1)
    g = torch.Generator().manual_seed(1)
    fr = torch.rand(6, 3, generator=g)
    for f in fr:
        x = (torch.tensor([[7.0, 8.0, 6.0]]) + f[None]).requires_grad_(True)
        Lraw = d_h1(x, m, tg, gmin, dx, dims, self_correct=False)
        (graw,) = torch.autograd.grad(Lraw, x)
        Lc = d_h1(x, m, tg, gmin, dx, dims)
        (gc,) = torch.autograd.grad(Lc, x)
        assert float(Lraw) > 0
        assert abs(float(Lc)) < 1e-6 * float(Lraw), (float(Lc), float(Lraw))
        assert float(gc.abs().max()) < 1e-4 * float(graw.abs().max() + 1e-12), (gc, graw)
    # the raw self-force is a real lattice attractor: it points toward the cell centre
    x = torch.tensor([[7.1, 8.1, 6.1]]).requires_grad_(True)
    (graw,) = torch.autograd.grad(d_h1(x, m, tg, gmin, dx, dims, self_correct=False), x)
    assert (-graw[0] > 0).all()                      # descent is toward (7.5, 8.5, 6.5)


def test_h1_is_nonlocal_where_d_vol_is_blind_at_any_subcell_offset():
    """A probe particle in EMPTY space between a surplus (A) and a deficit (B) has no
    local mismatch to descend. D_vol's gradient on it is its own CIC self-force (toward
    the nearest cell centre; zero only at the symmetric point - REFUTE Opus F5), while
    the self-corrected H^-1 gradient is the far field of A and B: pushed by the surplus,
    pulled by the deficit, coherently along +x at EVERY sub-cell offset."""
    dims, dx, gmin = (32, 32, 32), 1.0, torch.zeros(3)
    A, B, g = _two_blobs()
    tgt = torch.cat([A, B]); m = torch.ones(401)
    tg = rasterize_mass(tgt, torch.full((400,), 401.0 / 400.0), gmin, dx, dims)
    body = torch.cat([A, A + 0.3 * torch.randn(200, 3, generator=g)])
    offs = torch.tensor([[0.5, 0.5, 0.5], [0.52, 0.5, 0.5], [0.6, 0.5, 0.5], [0.9, 0.5, 0.5],
                         [0.5, 0.1, 0.8], [0.15, 0.85, 0.3]])
    for o in offs:
        probe = torch.tensor([[16.0, 16.0, 16.0]]) + o[None]
        cur = torch.cat([body, probe]).requires_grad_(True)
        (gh,) = torch.autograd.grad(d_h1(cur, m, tg, gmin, dx, dims), cur)
        p_h = -gh[-1]
        assert p_h[0] > 0 and p_h[0] > 5 * (abs(p_h[1]) + abs(p_h[2])), (o, p_h)
    # D_vol at an off-centre offset: the self-force toward the cell centre dominates and
    # points AWAY from B (-x) - not a signal about the deficit at all
    probe = torch.tensor([[16.9, 16.5, 16.5]])
    cur = torch.cat([body, probe]).requires_grad_(True)
    (gv,) = torch.autograd.grad(d_vol(cur, m, tg, gmin, dx, dims), cur)
    assert -gv[-1][0] < 0
