"""One-sided W1 cleanup term on the target's 3-D distance transform, the isolation gate, the
near-band pull and the candidate validity check.

The 2D multi-view variant was falsified by forensics (visual hull hides interior
concavities); these tests pin the 3D mechanism's claims, including the Codex round's
counterexamples (target self-force, fixed-N sparsity invariance, clamp handoff)."""
import numpy as np
import pytest
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


def test_grid_gate_is_the_mpm_decoupling_test_and_knn_the_old_gate():
    """D126 (--spray_gate grid): the gate is the simulator's own decoupling test (k_frag_step's 3^3-cell count through
    the support-gate kernel, or the commit's fragment mask): a particle with no grid neighbour is isolated (1), a pair
    stretched to 0.8 cells but sharing cells with each other and nodes with the body is not (0), the body is not; and
    under "knn" the dispatcher is the old ramp bit for bit. The default is "knn"."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable (the commit's fragment mask labels on the device)")
    from physmorph.losses.volumetric import grid_isolation_gate, isolation_gate, spray_gate
    from physmorph.mpm.state import MPMParams
    from physmorph.pipeline.config import PipelineConfig
    assert PipelineConfig().spray_gate == "knn"
    prm = MPMParams(dx=0.5, dt=1.0 / 240.0, drag=0.0, smoothing=1.0, grid_min=(-6.0, -6.0, -6.0), nx=24, ny=24, nz=24)
    rng = np.random.default_rng(11)
    body = rng.uniform(-1.0, 1.0, (600, 3)).astype(np.float32)             # ~9 particles a cell
    pair = np.array([[1.15, 0.0, 0.0], [1.55, 0.0, 0.0]], np.float32)       # stretched (0.8 cells apart), both coupled
    lone = np.array([[4.5, 0.0, 0.0]], np.float32)                           # no other particle within its 3^3 cells
    x = torch.as_tensor(np.concatenate([body, pair, lone]), device="cuda")
    g = grid_isolation_gate(x, prm)
    assert g.shape == (603,) and g.dtype == x.dtype
    assert float(g[:600].max()) == 0.0 and float(g[600:602].max()) == 0.0
    assert float(g[602]) == 1.0
    # a particle alone in its 3^3 cells is isolated by the per-step test even where the commit mask still joins it
    near = torch.as_tensor(np.concatenate([body, [[2.45, 0.0, 0.0]]]).astype(np.float32), device="cuda")
    assert float(grid_isolation_gate(near, prm)[-1]) == 1.0
    assert torch.equal(spray_gate(x, "grid", 1.2, 1.8, prm=prm), g)
    assert torch.equal(spray_gate(x, "knn", 1.2, 1.8), isolation_gate(x, 1.2, 1.8))
    local = torch.ones(603, device="cuda")
    assert torch.equal(spray_gate(x, "knn", 1.3, 1.7, local=local), isolation_gate(x, 1.3, 1.7, local=local))


def test_state_ok_rejects_trajectory_inversion():
    """Guard v2: a candidate whose rollout inverted at ANY step is rejected even when
    the terminal state recovered (hero7/hero9: F_invert_steps=1 slipped through)."""
    from types import SimpleNamespace
    from physmorph.pipeline.window.rollout import state_ok
    xT = torch.zeros(4, 3); FT = torch.eye(3).repeat(4, 1).reshape(4, 9)
    vT = torch.zeros(4, 3)

    def e(jt, in_domain=True):
        return SimpleNamespace(xT=xT, FT=FT, vT=vT, jt=jt, in_domain=in_domain)
    assert state_ok(e(0.5))
    assert not state_ok(e(-0.01))                   # mid-trajectory inversion
    assert not state_ok(e(0.5, in_domain=False))    # a frame outside the domain


def test_nn_band_pull_and_berth():
    """Grid-free near-band W1 (§7.10): zero inside the berth (rim safe), constant pull
    toward the ASSIGNED target particle in the band, ineligible beyond far_k."""
    from physmorph.losses.volumetric import nn_band_assign, d_nn_band
    g = torch.linspace(-0.5, 0.5, 11)
    t = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    spacing = 0.1                                # exact grid spacing
    x0 = torch.tensor([[0.75, 0.0, 0.0],         # 0.25 off the face: in band
                       [0.60, 0.0, 0.0],         # 0.10 < berth 0.15: rim
                       [1.20, 0.0, 0.0]])        # 0.70 > far 0.45: DT-W1's job
    from physmorph import gpu
    idx, elig = nn_band_assign(x0, gpu.KNN(t), spacing, berth_k=1.5, far_k=4.5)
    assert float(elig[0]) == 1.0 and float(elig[1]) == 0.0 and float(elig[2]) == 0.0
    x = x0.clone().requires_grad_(True)
    d_nn_band(x, torch.ones(3), t, idx, elig, 1.5 * spacing).backward()
    g_ = x.grad
    assert float(g_[1].norm()) == 0.0 and float(g_[2].norm()) == 0.0
    assert -g_[0][0] < 0                         # descent pulls toward -x (the face)


def test_nn_band_current_form_shares_the_band():
    """The selection merit's form of the near band (current nearest target points) counts the same band as the
    frozen assignment: inside the berth nothing, in the band the distance beyond the berth, at or beyond the far
    edge nothing; without a far edge every particle beyond the berth counts."""
    from physmorph import gpu
    from physmorph.losses.volumetric import d_nn_band_current
    g = torch.linspace(-0.5, 0.5, 11)
    t = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    x = torch.tensor([[0.75, 0.0, 0.0],          # 0.25 off the face: in the band
                      [0.60, 0.0, 0.0],          # 0.10 < berth 0.15
                      [1.20, 0.0, 0.0]])         # 0.70 beyond the far edge 0.45
    knn, ones, berth = gpu.KNN(t), torch.ones(3), 0.15
    banded = float(d_nn_band_current(x, ones, t, ones, berth, knn, far=0.45))
    whole = float(d_nn_band_current(x, ones, t, ones, berth, knn))
    assert banded == pytest.approx(0.25 - berth, abs=1e-6)
    assert whole == pytest.approx((0.25 - berth) + (0.70 - berth), abs=1e-6)
