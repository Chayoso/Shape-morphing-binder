"""D147 redesign: the morph's progress (physmorph/pipeline/scene/schedule.py) -- the ramp, each window's target on the
transport's displacement interpolation with the final target's units, the final target itself from s = 1, and the
scene run with the schedule (CUDA)."""
import numpy as np
import pytest
import torch

from physmorph.pipeline.scene.schedule import ramp

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


def test_ramp_is_a_smoothstep_from_zero_to_one():
    assert ramp(0.0, 2.0) == 0.0 and ramp(-1.0, 2.0) == 0.0
    assert ramp(2.0, 2.0) == 1.0 and ramp(5.0, 2.0) == 1.0
    assert abs(ramp(1.0, 2.0) - 0.5) < 1e-12
    s = [ramp(t, 2.0) for t in np.linspace(0.0, 2.0, 41)]
    assert all(b >= a for a, b in zip(s, s[1:]))
    assert ramp(0.05, 2.0) < 0.01                      # the target's speed starts at zero


def _bodies(n=600, seed=7):
    from physmorph import gpu
    from physmorph.mpm.state import MPMParams
    from physmorph.pipeline import PipelineConfig
    from physmorph.pipeline.scene.window import Body
    from physmorph.pipeline.target import build_target, calibrate_units

    rng = np.random.default_rng(seed)
    canon = MPMParams(dx=0.5, nx=12, ny=12, nz=12, grid_min=(-3.0, -3.0, -3.0))   # the loss cell 0.5 resolves the clouds
    cfg = PipelineConfig(T=3, iters=2, animations=2, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                         render_res=24, dt_res=32)
    bodies, xs, vs = [], [], []
    for b in range(2):
        src = rng.uniform(-1.5, 1.5, (n, 3)).astype(np.float32)
        src -= src.mean(0)
        tgt = (rng.uniform(-1.5, 1.5, (n, 3)) * np.array([1.6, 0.6, 1.0])).astype(np.float32)
        tgt -= tgt.mean(0)
        tp = build_target(tgt, canon, cfg)
        calibrate_units(tp, gpu.tensor(src), cfg)
        bodies.append(Body(start=b * n, stop=(b + 1) * n, Xq=gpu.tensor(src), tgt=tp, vp=27.0 / n,
                           mass=cfg.mass_ref_n / n, cfg=cfg))
        xs.append(src + np.array([(-1) ** (b + 1) * 2.0, 0.0, 0.0], np.float32))
        vs.append(np.tile([(-1) ** b * 0.5, 0.0, 0.0], (n, 1)).astype(np.float32))
    return bodies, canon, cfg, xs, vs


def test_each_window_targets_the_displacement_interpolation_with_the_final_units():
    from physmorph import metrics
    from physmorph.pipeline.scene.schedule import MorphSchedule

    bodies, canon, cfg, _, _ = _bodies()
    final = [bd.tgt for bd in bodies]
    sch = MorphSchedule(bodies, [canon, canon], 2.0)
    for b, bd in enumerate(bodies):
        X, D, Y = bd.Xq.cpu().numpy(), sch.D[b].cpu().numpy(), final[b].pts.cpu().numpy()
        assert D.shape == X.shape and np.isfinite(D).all()
        assert metrics.chamfer(X + D, Y) < 0.7 * metrics.chamfer(X, Y)      # the map carries the source to the target
        assert final[b].ot_scale is not None and final[b].ot_scale > 0
    s = sch.apply(bodies, 1.0)
    assert abs(s - 0.5) < 1e-12
    for b, bd in enumerate(bodies):
        assert bd.tgt is not final[b]
        assert torch.allclose(bd.tgt.pts, bd.Xq + 0.5 * sch.D[b])
        assert (bd.tgt.unit_ratio, bd.tgt.unit_grad_ratio, bd.tgt.ot_scale) == \
            (final[b].unit_ratio, final[b].unit_grad_ratio, final[b].ot_scale)
        assert bd.tgt.relief is None
    assert sch.apply(bodies, 2.5) == 1.0
    assert all(bd.tgt is f for bd, f in zip(bodies, final))


def test_scene_runs_with_the_schedule():
    """Two windows of the scene with the targets on the way: every window commits and records its progress."""
    from physmorph import gpu
    from physmorph.mpm.state import MPMParams
    from physmorph.pipeline.scene.runner import run_scene
    from physmorph.pipeline.scene.schedule import MorphSchedule

    bodies, canon, cfg, xs, vs = _bodies()
    sch = MorphSchedule(bodies, [canon, canon], 0.1)
    world = MPMParams(dx=1.0, drag=0.0, grid_min=(-16.0, -10.0, -10.0), nx=32, ny=20, nz=20)
    x0, v0 = gpu.tensor(np.concatenate(xs)), gpu.tensor(np.concatenate(vs))
    res = run_scene(bodies, x0, v0, world, cfg, windows=2, mu=0.5, log=lambda *_: None, schedule=sch)
    hist = res["history"]
    assert len(hist) == 2 and all("bodies" in h for h in hist)
    dt = 2 * cfg.T * world.dt
    assert [h["morph_s"] for h in hist] == [ramp(dt, 0.1), ramp(2 * dt, 0.1)]
    assert all(v == 0 for v in res["guards"].values())
