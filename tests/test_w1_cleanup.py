"""One-sided-W1 cleanup term (fringe tranche, rationale.md §7) — 3D loss-grid DT.

The 2D multi-view variant was falsified by forensics (visual hull hides interior
concavities); these tests pin the 3D mechanism's claims, including the Codex round's
counterexamples (target self-force, fixed-N sparsity invariance, clamp handoff)."""
import numpy as np
import torch

from physmorph.losses.volumetric import d_w1, target_dt_grid, target_mass_grid

DIMS = (24, 24, 24)
DX = 0.25
GMIN = torch.tensor([-3.0, -3.0, -3.0])


def _setup(n=800, seed=3):
    rng = np.random.default_rng(seed)
    t = torch.tensor(rng.uniform(-0.5, 0.5, (n, 3)).astype(np.float32))
    m = torch.ones(len(t))
    grid = target_mass_grid(t, m, GMIN, DX, DIMS)
    dt3 = target_dt_grid(grid, DX, DIMS, clamp=2.0 * 3.0)
    return t, m, grid, dt3


def test_no_target_self_force():
    """Codex finding 1/2 (ported to 3D): the loss AND its gradient must vanish on the
    target itself — support is built from the same CIC stencil the sampler gathers."""
    t, m, grid, dt3 = _setup()
    x = t.clone().requires_grad_(True)
    loss = d_w1(x, m, dt3, GMIN, DX, DIMS)
    assert float(loss) < 1e-8
    loss.backward()
    # rim particles at exact cell faces may see a sub-cell boundary subgradient; it must
    # be rare and bounded, never a bulk force
    gnorm = x.grad.norm(dim=1)
    assert float(gnorm.max()) < 2.0         # sum form: rim subgradient O(1), never a bulk force
    assert float((gnorm > 1e-6).float().mean()) < 0.05


def test_zero_inside_monotone_outside():
    t, m, grid, dt3 = _setup()
    far = torch.tensor([[2.0, 0.0, 0.0]])
    near = torch.tensor([[1.2, 0.0, 0.0]])
    one = torch.ones(1)
    l_far = float(d_w1(far, one, dt3, GMIN, DX, DIMS))
    l_near = float(d_w1(near, one, dt3, GMIN, DX, DIMS))
    assert l_far > l_near > 0


def test_sparsity_invariant_at_fixed_n():
    """Codex finding 14: fixed total N, lone stray vs 8 co-located strays — the
    per-particle pull must be identical (linear term, no saturation)."""
    t, m, grid, dt3 = _setup()
    P = torch.tensor([1.5, 0.0, 0.0])

    def stray_grad(n_stray):
        body = t[: len(t) - n_stray]
        x = torch.cat([body, P.repeat(n_stray, 1)]).clone().requires_grad_(True)
        d_w1(x, torch.ones(len(x)), dt3, GMIN, DX, DIMS).backward()
        return x.grad[len(body):].norm(dim=1)

    g1, g8 = stray_grad(1), stray_grad(8)
    assert float(g1[0]) > 1e-6
    assert abs(float(g1[0]) - float(g8.mean())) < 0.02 * float(g1[0])


def test_pull_points_toward_support():
    t, m, grid, dt3 = _setup()
    x = torch.tensor([[1.5, 0.8, 0.0]], requires_grad=True)
    d_w1(x, torch.ones(1), dt3, GMIN, DX, DIMS).backward()
    step = -x.grad[0]
    assert step[0] < 0 and step[1] < 0      # descent moves toward the clump at origin


def test_no_force_free_gap_inside_box():
    """Codex finding 4: with clamp=2*extent every point of the box interior must keep a
    nonzero DT gradient (no plateau between the DT clamp and the w_box leash)."""
    t, m, grid, dt3 = _setup()
    extent = 1.0                             # pretend box; grid spans to +-3
    corner = torch.tensor([[0.98, 0.98, 0.98]], requires_grad=True)   # inside box corner
    d_w1(corner, torch.ones(1), dt3, GMIN, DX, DIMS).backward()
    assert float(corner.grad.norm()) > 1e-6


def test_subcell_fringe_regime_has_gradient():
    """Opus finding 2: the production fringe lives 0.03-0.17*extent off the surface —
    on the fine target-fitted grid (cell ~0.019*extent) that band must be on a live
    DT slope, not in a CIC-dilation/trilinear dead zone."""
    rng = np.random.default_rng(5)
    extent = 3.0
    t = torch.tensor(rng.uniform(-1.5, 1.5, (4000, 3)).astype(np.float32))
    res = 160
    dx = 3.0 * extent / res
    gmin = torch.tensor([-1.5 * extent] * 3)
    grid = target_mass_grid(t, torch.ones(len(t)), gmin, dx, (res,) * 3)
    dt3 = target_dt_grid(grid, dx, (res,) * 3, clamp=2.0 * extent)
    for off in (0.2, 0.5):                    # world units off the +x face at 1.5
        x = torch.tensor([[1.5 + off, 0.0, 0.0]], requires_grad=True)
        d_w1(x, torch.ones(1), dt3, gmin, dx, (res,) * 3).backward()
        assert float(x.grad.norm()) > 0.1, f"dead zone at {off} world units"


def test_isolation_gate_complementarity():
    """§7.3/§7.4: the kNN-scale gate must silence BULK mass (dense at particle-spacing
    scale, wherever it sits — the dose-response protection) while a lone stray keeps
    the full pull. In-support false positives cost nothing (DT=0 there)."""
    from physmorph.losses.volumetric import isolation_gate
    rng = np.random.default_rng(9)
    body = torch.tensor(rng.uniform(-0.5, 0.5, (2000, 3)).astype(np.float32))
    clump = torch.tensor(rng.uniform(1.4, 1.6, (300, 3)).astype(np.float32))  # dense, off-target
    lone = torch.tensor([[2.5, 0.0, 0.0]])
    x = torch.cat([body, clump, lone])
    gate = isolation_gate(x)
    assert float(gate[2000:2300].mean()) < 0.1      # dense outside clump: silenced
    assert float(gate[-1]) > 0.9                    # lone stray: full pull
    assert float(gate[:2000].mean()) < 0.1          # bulk body: silenced
