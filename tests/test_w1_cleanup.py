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


def test_w1_budget_self_annealing():
    """§7.5: one scalar caps total pull mass. Early bulk-outside windows scale down
    (dose-response protection); late floater windows run at full per-particle pull."""
    from physmorph.losses.volumetric import w1_budget
    rng = np.random.default_rng(9)
    t = torch.tensor(rng.uniform(-0.5, 0.5, (2000, 3)).astype(np.float32))
    grid = target_mass_grid(t, torch.ones(len(t)), GMIN, DX, DIMS)
    dt3 = target_dt_grid(grid, DX, DIMS, clamp=6.0)
    # early: a third of the body far outside support
    outside = torch.tensor(rng.uniform(1.5, 2.5, (1000, 3)).astype(np.float32))
    x_early = torch.cat([t, outside])
    s_early = w1_budget(x_early, dt3, GMIN, DX, DIMS, budget_frac=0.01)
    assert s_early < 0.05                        # ~30/1000: heavily scaled down
    # late: a handful of floaters
    x_late = torch.cat([t, outside[:12]])
    s_late = w1_budget(x_late, dt3, GMIN, DX, DIMS, budget_frac=0.01)
    assert s_late == 1.0                         # full pull on the residue


def test_deficit_field_marks_underfill_and_saturates():
    """Hole-side W1 (§7.5): the deficit field must be nonzero near an under-covered
    target region, pull toward it, and vanish (None) once the body covers the target."""
    from physmorph.losses.volumetric import deficit_field
    rng = np.random.default_rng(11)
    t = torch.tensor(rng.uniform(-0.5, 0.5, (3000, 3)).astype(np.float32))
    body_half = t[t[:, 0] < 0.1]                 # body covers only the left part
    tm = target_mass_grid(t, torch.ones(len(t)), GMIN, DX, DIMS)
    df = deficit_field(body_half, torch.ones(len(body_half)), tm, GMIN, DX, DIMS)
    assert df is not None
    ddt, dmass = df
    assert dmass > 0
    # Opus F1 regression: EVERY deficit cell must lie in TRUE target support
    occ = (tm.reshape(DIMS) > 1e-6).numpy()
    dt_grid = ddt.reshape(DIMS).numpy()
    assert not ((dt_grid == 0) & ~occ).any(), "deficit mask leaked outside support"
    # a particle left of the deficit is pulled +x toward it
    x = torch.tensor([[-0.2, 0.0, 0.0]], requires_grad=True)
    d_w1(x, torch.ones(1), ddt, GMIN, DX, DIMS).backward()
    assert -x.grad[0][0] > 0                     # descent moves toward +x (the deficit)
    # full coverage -> no deficit
    assert deficit_field(t, torch.ones(len(t)), tm, GMIN, DX, DIMS) is None


def test_knn_gate_selectivity():
    """§7.6: the restored kNN gate silences dense mass (bulk AND dense off-target
    clumps) while a lone stray keeps the full pull."""
    from physmorph.losses.volumetric import isolation_gate
    rng = np.random.default_rng(9)
    body = torch.tensor(rng.uniform(-0.5, 0.5, (2000, 3)).astype(np.float32))
    clump = torch.tensor(rng.uniform(1.4, 1.6, (300, 3)).astype(np.float32))
    lone = torch.tensor([[2.5, 0.0, 0.0]])
    gate = isolation_gate(torch.cat([body, clump, lone]))
    assert float(gate[2000:2300].mean()) < 0.1 and float(gate[:2000].mean()) < 0.1
    assert float(gate[-1]) > 0.9


def test_state_ok_rejects_trajectory_inversion():
    """Guard v2: a candidate whose rollout inverted at ANY step is rejected even when
    the terminal state recovered (hero7/hero9: F_invert_steps=1 slipped through)."""
    from physmorph.pipeline.optimizer import _state_ok
    xT = torch.zeros(4, 3); FT = torch.eye(3).repeat(4, 1).reshape(4, 9)
    vT = torch.zeros(4, 3)
    assert _state_ok((xT, FT, vT, 0.5))
    assert not _state_ok((xT, FT, vT, -0.01))       # mid-trajectory inversion
    assert _state_ok((xT, FT, vT))                  # legacy 3-tuple still works


def test_nn_band_pull_and_berth():
    """Grid-free near-band W1 (§7.10): zero inside the berth (rim safe), constant pull
    toward the ASSIGNED target particle in the band, ineligible beyond far_k."""
    from physmorph.losses.volumetric import nn_band_assign, d_nn_band
    rng = np.random.default_rng(3)
    t = torch.tensor(rng.uniform(-0.5, 0.5, (2000, 3)).astype(np.float32))
    spacing = 0.03
    x0 = torch.tensor([[0.55, 0.0, 0.0],     # ~0.05 off the face: in band
                       [0.51, 0.0, 0.0],     # inside berth: rim
                       [0.9, 0.0, 0.0]])     # beyond far band
    idx, elig = nn_band_assign(x0, t, spacing, berth_k=1.5, far_k=4.5)
    assert float(elig[1]) == 0.0 and float(elig[2]) == 0.0 and float(elig[0]) == 1.0
    x = x0.clone().requires_grad_(True)
    d_nn_band(x, torch.ones(3), t, idx, elig, 1.5 * spacing).backward()
    g = x.grad
    assert float(g[1].norm()) == 0.0 and float(g[2].norm()) == 0.0
    assert -g[0][0] < 0                      # descent pulls the band particle toward -x
