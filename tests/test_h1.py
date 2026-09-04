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
    assert float(d_h1(tgt, m, tg, gmin, dx, dims)) < 1e-9          # r == 0
    # all mass piled at A: the deficit at B is 12 cells away
    cur = torch.cat([A, A + 0.3 * torch.randn(200, 3, generator=g)]).requires_grad_(True)
    L = d_h1(cur, m, tg, gmin, dx, dims)
    assert float(L) > 0
    (gh,) = torch.autograd.grad(L, cur)
    assert torch.isfinite(gh).all()
    # the NET descent of the surplus blob points at B (+x); each particle's own
    # gradient is dominated by the blob's Coulomb self-repulsion (decompression), which
    # is the physics, so only the mean is tested here
    step = -gh.mean(0)
    assert step[0] > 0 and step[0] > 3 * abs(step[1]) and step[0] > 3 * abs(step[2])


def test_h1_is_nonlocal_where_d_vol_is_blind():
    """A probe particle in EMPTY space between a surplus (A) and a deficit (B) has no
    local mismatch to descend: D_vol's gradient on it is zero by symmetry (its own CIC
    footprint is symmetric), while the H^-1 gradient is the far field of A and B -
    pushed by the surplus, pulled by the deficit, coherently along +x."""
    dims, dx, gmin = (32, 32, 32), 1.0, torch.zeros(3)
    A, B, g = _two_blobs()
    tgt = torch.cat([A, B]); m = torch.ones(401)
    tg = rasterize_mass(tgt, torch.ones(400), gmin, dx, dims)
    probe = torch.tensor([[16.5, 16.5, 16.5]])            # cell centre: symmetric footprint
    cur = torch.cat([A, A + 0.3 * torch.randn(200, 3, generator=g), probe]).requires_grad_(True)
    (gh,) = torch.autograd.grad(d_h1(cur, m, tg, gmin, dx, dims), cur)
    (gv,) = torch.autograd.grad(d_vol(cur, m, tg, gmin, dx, dims), cur)
    p_h, p_v = -gh[-1], -gv[-1]
    assert p_h[0] > 0 and p_h[0] > 5 * (abs(p_h[1]) + abs(p_h[2])), p_h
    assert p_v.abs().max() < 1e-3 * p_h[0], (p_v, p_h)
