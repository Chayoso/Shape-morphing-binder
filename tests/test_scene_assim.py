"""D147: the scene's plastic assimilation reads the controls' deformation without the loads (scene/window.free_end_F):
for a body without loads it is the loaded rollout itself (the frozen rule, to the atomics' rounding); an impact's
compression is not made rest shape (CUDA). The assimilation acts between windows on the promoted state, outside every
tape, as the frozen rule does: no adjoint passes through it."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from physmorph import gpu
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import compute_rest_volumes
from physmorph.pipeline import PipelineConfig
from physmorph.pipeline.scene.window import Body, SceneWindow
from physmorph.pipeline.window.setup import StartState
from physmorph.plasticity import assimilate_elastic

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


def _ball(c, r=3.0, h=0.5, seed=0):
    rng = np.random.default_rng(seed)
    g = np.arange(-r, r + 1e-9, h)
    p = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    p = p[np.linalg.norm(p, axis=1) <= r]
    return (p + rng.uniform(-0.075, 0.075, p.shape) + np.asarray(c)).astype(np.float32)


def _window(xs, vs, prm, cfg):
    stub = SimpleNamespace(relief=None, unit_ratio=1.0, unit_grad_ratio=1.0)
    x = gpu.tensor(np.concatenate(xs))
    bodies, bonds, vols, s = [], [], [], 0
    for xb in xs:
        xt = gpu.tensor(xb)
        bodies.append(Body(start=s, stop=s + len(xb), Xq=xt - xt.mean(0), tgt=stub, vp=0.125, mass=1.0))
        coh = gpu.knn(xt, cfg.coh_k + 1)[1][:, 1:]
        bonds.append((coh, (xt[coh] - xt[:, None]).norm(dim=2), torch.zeros(len(xb), device="cuda")))
        vols.append(compute_rest_volumes(xt, 1.0, prm, "cuda"))
        s += len(xb)
    start = StartState(x=x, Fp=torch.eye(3, device="cuda").repeat(len(x), 1, 1), v=gpu.tensor(np.concatenate(vs)))
    return SceneWindow(start, prm, cfg, bodies, torch.cat(vols), bonds, mu=0.5)


def _loaded_end_F(win):
    win.tr.run()
    return torch.cat([wp.to_torch(tr.F[win.T]).reshape(-1, 3, 3) for tr in win.tr.trajs])


def test_without_loads_the_free_rollout_is_the_loaded_one():
    cfg = PipelineConfig(T=6, young=1.0e4, min_spacing=0.9, volume_exact="carried")
    prm = MPMParams(dx=1.0, dt=1.0 / 240.0, drag=0.0, grid_min=(-16.0, -10.0, -10.0), nx=32, ny=20, nz=20)
    x = _ball((0.0, 0.0, 0.0))
    win = _window([x], [np.zeros_like(x)], prm, cfg)
    leaf = 0.01 * torch.randn(cfg.T, len(x), 3, 3, generator=torch.Generator().manual_seed(3)).cuda()
    win.load(leaf, None)
    F_loaded = _loaded_end_F(win)
    F_free = win.free_end_F()
    assert float((F_loaded - F_free).abs().max()) < 1e-5
    Fp0 = torch.eye(3, device="cuda").repeat(len(x), 1, 1)
    a = assimilate_elastic(F_loaded, Fp0, eta=0.5)
    b = assimilate_elastic(F_free, Fp0, eta=0.5)
    assert float((a - b).abs().max()) < 1e-5


def test_an_impacts_compression_is_not_assimilated():
    """Two balls collide (no control, no gravity) over a window: the frozen rule makes the facing layers' compression
    rest shape (its det deficit about half the compression's, eta = 0.5), the controls' deformation without the loads
    leaves Fp the identity (a twentieth of that at most)."""
    cfg = PipelineConfig(T=20, young=1.0e4, min_spacing=0.9, volume_exact="carried")
    prm = MPMParams(dx=1.0, dt=1.0 / 240.0, drag=0.0, grid_min=(-16.0, -10.0, -10.0), nx=32, ny=20, nz=20)
    xs = [_ball((-3.0, 0.0, 0.0)), _ball((3.0, 0.0, 0.0), seed=1)]                       # touching at the start
    vs = [np.tile([3.7, 0.0, 0.0], (len(xs[0]), 1)).astype(np.float32),
          np.tile([-3.7, 0.0, 0.0], (len(xs[1]), 1)).astype(np.float32)]               # v_rel / c = 0.2
    win = _window(xs, vs, prm, cfg)
    win.load(torch.zeros(cfg.T, win.N, 3, 3, device="cuda"), None)
    F_loaded = _loaded_end_F(win)
    assert float(wp.to_torch(win.tr.impulse).norm()) > 0.0           # they collided
    Fp0 = torch.eye(3, device="cuda").repeat(win.N, 1, 1)
    frozen = assimilate_elastic(F_loaded, Fp0, eta=0.5, isochoric=False)
    free = assimilate_elastic(win.free_end_F(), Fp0, eta=0.5, isochoric=False)
    deficit = 1.0 - float(torch.linalg.det(F_loaded).min())          # the compression the loaded window made
    assert deficit > 0.01
    assert 1.0 - float(torch.linalg.det(frozen).min()) > 0.3 * deficit
    assert 1.0 - float(torch.linalg.det(free).min()) < 0.05 * deficit
    assert float((free - Fp0).abs().max()) < 0.05 * deficit + 1e-4


def test_the_uncontrolled_continuation_is_the_windows_own_zero_control_rollout():
    """B9b, assim_share: zero_end_F rolls the window's start out with zero control and zero u under the same loads
    (gravity, the floor with its position constraint): the window's own rollout of the zero control, to the rounding."""
    cfg = PipelineConfig(T=20, young=1.0e4, min_spacing=0.9, volume_exact="carried")
    x = _ball((0.0, 0.0, 0.0))
    prm = MPMParams(dx=1.0, dt=1.0 / 240.0, drag=0.0, f_ext=(0.0, -50.0, 0.0), grid_min=(-16.0, -10.0, -10.0),
                    nx=32, ny=20, nz=20, floor_y=float(x[:, 1].min()) - 0.5 * 0.125 ** (1.0 / 3.0))
    win = _window([x], [np.zeros_like(x)], prm, cfg)
    win.load(torch.zeros(cfg.T, len(x), 3, 3, device="cuda"), None)
    F_loaded = _loaded_end_F(win)
    assert float((F_loaded - torch.eye(3, device="cuda")).abs().max()) > 1e-5       # the loads deform it (40 steps,
                                                                                    # pressed into the floor)
    dev = float((F_loaded - torch.eye(3, device="cuda")).abs().max())
    assert float((F_loaded - win.zero_end_F()).abs().max()) < max(1e-5, 1e-3 * dev)        # the atomics' rounding


def test_the_controls_share_cancels_the_loads_and_their_rotation():
    """runner.share_F: with no control the share is the identity (nothing assimilated); a rotation the loads add to both
    ends cancels; for an unloaded start at rest (F_zero = I) it is the frozen rule's reading."""
    from physmorph.pipeline.scene.runner import share_F
    g = torch.Generator().manual_seed(4)
    n = 64
    Fp = torch.eye(3) + 0.05 * torch.randn(n, 3, 3, generator=g)
    F0 = (torch.eye(3) + 0.1 * torch.randn(n, 3, 3, generator=g)) @ Fp
    F1 = (torch.eye(3) + 0.1 * torch.randn(n, 3, 3, generator=g)) @ F0
    assert float((share_F(F0, F0, Fp) - Fp).abs().max()) < 1e-5            # F_e = I
    a = torch.tensor(0.7)
    Q = torch.tensor([[torch.cos(a), 0.0, torch.sin(a)], [0.0, 1.0, 0.0], [-torch.sin(a), 0.0, torch.cos(a)]])
    assert float((share_F(Q @ F1, Q @ F0, Fp) - share_F(F1, F0, Fp)).abs().max()) < 1e-4
    eye = torch.eye(3).repeat(n, 1, 1)
    assert float((share_F(F1, eye, eye) - F1).abs().max()) < 1e-5
    a1 = assimilate_elastic(share_F(F0, F0, Fp).cuda(), Fp.cuda(), eta=0.5)
    assert float((a1 - Fp.cuda()).abs().max()) < 1e-4


def test_the_loads_own_effect_reading_is_the_frozen_one_without_loads():
    """B9b, assim_load: with no gravity and no floor the zero-control continuations with and without the loads are the
    same rollout, so runner.load_free_F reads the committed end as the frozen rule does and the assimilated Fp is the
    frozen one, to the rollouts' rounding (a moving, controlled body; the share failed exactly here)."""
    from physmorph.pipeline.scene.runner import load_free_F
    cfg = PipelineConfig(T=20, young=1.0e4, min_spacing=0.9, volume_exact="carried")
    prm = MPMParams(dx=1.0, dt=1.0 / 240.0, drag=0.0, grid_min=(-16.0, -10.0, -10.0), nx=32, ny=20, nz=20)
    x = _ball((0.0, 0.0, 0.0))
    v = np.tile([0.4, -0.2, 0.1], (len(x), 1)).astype(np.float32)                 # a moving start
    win = _window([x], [v], prm, cfg)
    leaf = 0.01 * torch.randn(cfg.T, len(x), 3, 3, generator=torch.Generator().manual_seed(5)).cuda()
    win.load(leaf, None)
    F_ctrl = _loaded_end_F(win).clone()
    F_zl, F_zf = win.zero_end_F(), win.free_zero_end_F()
    assert float((F_zl - F_zf).abs().max()) < 1e-5
    assert float((load_free_F(F_ctrl, F_zl, F_zf) - F_ctrl).abs().max()) < 1e-4
    Fp0 = torch.eye(3, device="cuda").repeat(len(x), 1, 1)
    a = assimilate_elastic(load_free_F(F_ctrl, F_zl, F_zf), Fp0, eta=0.5)
    b = assimilate_elastic(F_ctrl, Fp0, eta=0.5)
    assert float((a - b).abs().max()) < 1e-4


def test_the_loads_own_effect_is_what_the_continuation_carries_under_them():
    """runner.load_free_F: without a load it is the committed end; the loaded continuation itself reads as the unloaded
    one (the weight's and the floor's strain taken off); under gravity on the floor the two continuations differ."""
    from physmorph.pipeline.scene.runner import load_free_F
    g = torch.Generator().manual_seed(6)
    n = 64
    F0 = torch.eye(3) + 0.1 * torch.randn(n, 3, 3, generator=g)
    F1 = torch.eye(3) + 0.1 * torch.randn(n, 3, 3, generator=g)
    Fc = torch.eye(3) + 0.1 * torch.randn(n, 3, 3, generator=g)
    assert float((load_free_F(Fc, F0, F0) - Fc).abs().max()) < 1e-5
    assert float((load_free_F(F0, F0, F1) - F1).abs().max()) < 1e-5
    cfg = PipelineConfig(T=20, young=1.0e4, min_spacing=0.9, volume_exact="carried")
    x = _ball((0.0, 0.0, 0.0))
    prm = MPMParams(dx=1.0, dt=1.0 / 240.0, drag=0.0, f_ext=(0.0, -50.0, 0.0), grid_min=(-16.0, -10.0, -10.0),
                    nx=32, ny=20, nz=20, floor_y=float(x[:, 1].min()) - 0.5 * 0.125 ** (1.0 / 3.0))
    win = _window([x], [np.zeros_like(x)], prm, cfg)
    F_zl, F_zf = win.zero_end_F(), win.free_zero_end_F()
    assert float((F_zl - F_zf).abs().max()) > 1e-5
