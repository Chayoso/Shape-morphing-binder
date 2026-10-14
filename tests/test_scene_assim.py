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
    rest shape (det Fp below 0.995), the controls' deformation without the loads leaves Fp the identity."""
    cfg = PipelineConfig(T=20, young=1.0e4, min_spacing=0.9, volume_exact="carried")
    prm = MPMParams(dx=1.0, dt=1.0 / 240.0, drag=0.0, grid_min=(-16.0, -10.0, -10.0), nx=32, ny=20, nz=20)
    xs = [_ball((-3.1, 0.0, 0.0)), _ball((3.1, 0.0, 0.0), seed=1)]
    vs = [np.tile([1.85, 0.0, 0.0], (len(xs[0]), 1)).astype(np.float32),
          np.tile([-1.85, 0.0, 0.0], (len(xs[1]), 1)).astype(np.float32)]
    win = _window(xs, vs, prm, cfg)
    win.load(torch.zeros(cfg.T, win.N, 3, 3, device="cuda"), None)
    F_loaded = _loaded_end_F(win)
    assert float(wp.to_torch(win.tr.impulse).norm()) > 0.0           # they collided
    Fp0 = torch.eye(3, device="cuda").repeat(win.N, 1, 1)
    frozen = assimilate_elastic(F_loaded, Fp0, eta=0.5, isochoric=False)
    free = assimilate_elastic(win.free_end_F(), Fp0, eta=0.5, isochoric=False)
    assert float(torch.linalg.det(frozen).min()) < 0.995
    assert float(torch.linalg.det(free).min()) > 0.999
    assert float((free - Fp0).abs().max()) < 5e-3
