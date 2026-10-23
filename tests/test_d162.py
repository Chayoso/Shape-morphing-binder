"""D162: the render weight referenced to the render's own floor (render_weight_ref "floor") and the placement terms at
the loss cell (placement_at_cell); both default off, the frozen path unchanged."""
import math

import pytest
import torch

from physmorph.pipeline.config import PipelineConfig
from physmorph.pipeline.render_loss import floor_referenced_weight

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the device KD-tree")


def test_defaults_are_the_frozen_path():
    c = PipelineConfig()
    assert c.render_weight_ref == "residual" and c.placement_at_cell is False


def test_floor_needs_the_exterior_and_a_known_reference():
    with pytest.raises(ValueError):
        PipelineConfig(render_weight_ref="floor")                 # render_exterior off: no exterior render term
    with pytest.raises(ValueError):
        PipelineConfig(render_exterior=True, render_weight_ref="excess")
    PipelineConfig(render_exterior=True, render_weight_ref="floor")


def test_floor_referenced_weight():
    lam0 = 0.3
    assert floor_referenced_weight(lam0, 1.0, 0.0) == pytest.approx(lam0)            # no floor: the first weight
    assert floor_referenced_weight(lam0, 2e-3, 1e-3) == pytest.approx(0.15)          # half the term is excess
    assert floor_referenced_weight(lam0, 1e-3, 1e-3) == 0.0                          # at the floor: no push
    assert floor_referenced_weight(lam0, 5e-4, 1e-3) == 0.0                          # below it: none, never negative
    assert floor_referenced_weight(lam0, 0.0, 1e-3) == 0.0
    for L in (1.1e-3, 3e-3, 1e-1, 10.0):                                             # bounded by the first weight,
        w = floor_referenced_weight(lam0, L, 1e-3)                                    # growing with the excess
        assert 0.0 < w < lam0
    assert floor_referenced_weight(lam0, 3e-3, 1e-3) > floor_referenced_weight(lam0, 2e-3, 1e-3)


@cuda
def test_proximity_free_radius_is_one_loss_cell():
    from physmorph import gpu
    from physmorph.losses.support import SurfaceProximity
    g = torch.Generator().manual_seed(0)
    cloud = torch.rand(4000, 3, generator=g, dtype=torch.float64) * 2.0 - 1.0
    base = SurfaceProximity(cloud)
    assert base.floor == 0.5 * math.exp(-base.spacing ** 2 / (2 * base.h * base.h))   # the frozen floor, untouched
    r = 4.0 * base.spacing
    sp = SurfaceProximity(cloud, free_radius=r)
    assert sp.floor == pytest.approx(math.exp(-r ** 2 / (2 * sp.h * sp.h)))
    y = sp.y[:1].to(gpu.DEVICE)
    sp.y = y
    for f, covered in ((0.98, True), (1.05, False)):
        x = torch.cat([y + torch.tensor([[f * r, 0.0, 0.0]], device=y.device, dtype=y.dtype), y + 50.0], 0)
        assert (float(sp.penalty_per_point(x)[0]) == 0.0) == covered


@cuda
def test_isolation_gate_in_loss_cells():
    from physmorph import gpu
    from physmorph.losses.volumetric import isolation_gate
    g = torch.Generator().manual_seed(1)
    x = (torch.rand(5000, 3, generator=g) * 2.0 - 1.0).to(gpu.DEVICE)
    x = torch.cat([x, torch.tensor([[5.0, 5.0, 5.0]], device=x.device)], 0)          # one particle far out
    a = isolation_gate(x)                                                             # the frozen path
    b = isolation_gate(x, scale=None)
    assert torch.equal(a, b)
    c = isolation_gate(x, scale=1.0)                                                 # in units of 1.0 (a "loss cell")
    assert float(c[-1]) == 1.0 and float(c[:-1].max()) == 0.0                        # only the far particle is isolated


@cuda
def test_a_floor_run_records_its_reference_and_holds_lambda_to_it():
    """D162 (1) on the contracts' small cloud: every committed window's record carries the floor (one value a run, the
    pictures' own first sample read as the body), the render term it was read against, the excess, the first weight,
    and the window's weight is that first weight times the excess (times render_weight_scale)."""
    import numpy as np
    from physmorph.mpm.state import MPMParams
    from physmorph.pipeline import run_pipeline
    rng = np.random.default_rng(11)
    src = rng.uniform(-1.5, 1.5, (300, 3)).astype(np.float32)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
    prm = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    cfg = PipelineConfig(T=4, iters=2, animations=3, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                         render_res=24, dt_res=32, patience=3, render_exterior=True, render_weight_ref="floor")
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    rows = [r for r in res["history"] if r.get("frame_end") and not r.get("null_commit")]
    assert rows
    floors = {r["render_floor"] for r in rows}
    assert len(floors) == 1 and next(iter(floors)) > 0.0
    lam0 = {r["lam0"] for r in rows}
    assert len(lam0) == 1
    for r in rows:
        assert r["render_L"] > 0.0
        assert r["render_excess"] == pytest.approx(max(0.0, 1.0 - r["render_floor"] / r["render_L"]))
        assert r["lambda"] == pytest.approx(r["lam0"] * r["render_excess"] * cfg.render_weight_scale)


def test_shape_reference_needs_the_exterior():
    with pytest.raises(ValueError):
        PipelineConfig(render_weight_ref="shape")
    PipelineConfig(render_exterior=True, render_weight_ref="shape")


def _box_scene():
    import numpy as np
    import trimesh
    rng = np.random.default_rng(11)
    src = rng.uniform(-1.5, 1.5, (300, 3)).astype(np.float32)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
    box = trimesh.creation.box(extents=(3.9, 2.4, 3.0))
    return src, tgt, (np.asarray(box.vertices), np.asarray(box.faces)), 3.9 * 2.4 * 3.0


@cuda
def test_a_shape_run_records_b_and_holds_lambda_to_its_excess():
    """D162b on the contracts' small cloud (its target: uniform in a 3.9 x 2.4 x 3 box): every committed window records
    b, one b_floor a run, the excess max(0, 1 - b_floor / b), and the weight lambda_0 x that excess; the reading's
    seconds where the window started from a new state."""
    from physmorph.mpm.state import MPMParams
    from physmorph.pipeline import run_pipeline
    src, tgt, mesh, fill = _box_scene()
    prm = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    cfg = PipelineConfig(T=4, iters=2, animations=3, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                         render_res=24, dt_res=32, patience=3, render_exterior=True, render_weight_ref="shape")
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None, mesh=mesh, target_fill=fill)
    rows = [r for r in res["history"] if r.get("frame_end") and not r.get("null_commit")]
    assert rows
    assert len({r["render_b_floor"] for r in rows}) == 1 and rows[0]["render_b_floor"] > 0.0
    assert len({r["lam0"] for r in rows}) == 1
    for r in rows:
        assert r["render_b"] > 0.0 and r["render_b_se"] >= 0.0
        assert r["render_excess"] == pytest.approx(max(0.0, 1.0 - r["render_b_floor"] / r["render_b"]))
        assert r["lambda"] == pytest.approx(r["lam0"] * r["render_excess"] * cfg.render_weight_scale)
    assert rows[0]["t_shape"] is not None and rows[0]["t_shape"] > 0.0


@cuda
def test_a_shape_reading_repeats_and_ignores_the_particle_order():
    """The resample lattice and its jitter are fixed for the run: the same particles read the same b, in any order."""
    import numpy as np
    from physmorph.mpm.state import MPMParams
    from physmorph.pipeline.shape_read import ShapeRead
    from physmorph.pipeline.target import build_target
    from physmorph.pipeline.window.setup import domain_bounds
    _, tgt, mesh, fill = _box_scene()
    prm = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    cfg = PipelineConfig(T=4, iters=2, loss_res=12, render_views=2, render_elevs=(0.0, 0.5), render_res=24, dt_res=32,
                         render_exterior=True, render_weight_ref="shape")
    pack = build_target(tgt, prm, cfg)
    sr = ShapeRead(pack, cfg, tgt, mesh, fill, len(tgt), domain_bounds(prm), log=lambda *_: None)
    b1, _, n1 = sr.read(tgt)
    b2, _, n2 = sr.read(tgt[np.random.default_rng(0).permutation(len(tgt))])
    assert b1 == pytest.approx(sr.b_floor, rel=1e-3) and b2 == pytest.approx(b1, rel=1e-3) and n1 == n2 > 0   # (atomics)


def test_floor_pitch_needs_the_exterior():
    with pytest.raises(ValueError):
        PipelineConfig(render_weight_ref="floor_pitch")
    PipelineConfig(render_exterior=True, render_weight_ref="floor_pitch")


@cuda
def test_the_lowpassed_term_is_the_render_term_at_zero_width():
    """D162d: with no blur the low-passed reading is render_floor's value (the same pictures, the same measures), and a
    blur changes it."""
    from physmorph.mpm.state import MPMParams
    from physmorph.pipeline.target import build_target, render_floor, render_term_lowpass
    _, tgt, _, _ = _box_scene()
    prm = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    cfg = PipelineConfig(T=4, iters=2, loss_res=12, render_views=2, render_elevs=(0.0, 0.5), render_res=24, dt_res=32,
                         render_exterior=True, render_weight_ref="floor_pitch")
    pack = build_target(tgt, prm, cfg)
    x = tgt + 0.05
    assert render_term_lowpass(pack, cfg, x, sigma=1e-9) == pytest.approx(render_floor(pack, cfg, x), rel=1e-4)
    assert render_term_lowpass(pack, cfg, x) != pytest.approx(render_floor(pack, cfg, x), rel=1e-4)


@cuda
def test_a_floor_pitch_run_records_its_reference_and_holds_lambda_to_it():
    from physmorph.mpm.state import MPMParams
    from physmorph.pipeline import run_pipeline
    src, tgt, _, _ = _box_scene()
    prm = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    cfg = PipelineConfig(T=4, iters=2, animations=3, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                         render_res=24, dt_res=32, patience=3, render_exterior=True, render_weight_ref="floor_pitch")
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    rows = [r for r in res["history"] if r.get("frame_end") and not r.get("null_commit")]
    assert rows
    assert len({r["render_lp_floor"] for r in rows}) == 1 and rows[0]["render_lp_floor"] > 0.0
    for r in rows:
        assert r["render_lp"] > 0.0
        assert r["render_excess"] == pytest.approx(max(0.0, 1.0 - r["render_lp_floor"] / r["render_lp"]))
        assert r["lambda"] == pytest.approx(r["lam0"] * r["render_excess"] * cfg.render_weight_scale)
