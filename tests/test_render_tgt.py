"""D156's render target (--render_tgt; default off): the render's target pictures drawn from another mesh than the
target the transport and the body reach.

(1) off by default: no render target, the pictures from --tgt's samples (the frozen path);
(2) sampling.mesh.sample_in_frame puts a mesh's sample in the frame load_normalized put another mesh's cloud in, by the
    same operations: for load_normalized's own mesh it is load_normalized's sample bit for bit;
(3) prepare with render_tgt = tgt_path is the frozen path's prepare bit for bit (and its render sample is the target);
    with another render mesh the target and its frame are the frozen path's, and the render's samples are the render
    mesh's in that frame, at the target sample's number density;
(4) build_target with render_x: the pictures and their constants (box, pitch, lattice) from render_x, the transport,
    the cleanups and the W1 cleanup's grid from target_x; with render_x = target_x the frozen pack;
(5) every evaluation of the render terms in a run (the gradient, the line search and the commit rollout) reads the
    render target's pictures: they live in the run's one TargetPack.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest
import torch
import trimesh

from physmorph.sampling.mesh import _oriented, load_normalized, sample_in_frame


@pytest.fixture(autouse=True)
def _own_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PHYSMORPH_CACHE", str(tmp_path / "cache"))


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA (prepare and build_target measure on the GPU)")


def test_off_by_default(monkeypatch):
    from scripts.pipeline_run import parse_args
    monkeypatch.setattr(sys, "argv", ["pipeline_run.py"])
    assert parse_args().render_tgt is None
    from physmorph.prepare import prepare
    import inspect
    assert inspect.signature(prepare).parameters["render_tgt"].default is None


@pytest.mark.parametrize("match", [True, False])
def test_sample_in_frame_of_its_own_mesh_is_load_normalized_bit_for_bit(match):
    v_src = 61.3 if match else None
    frame, fill_a, fill_b = {}, {}, {}
    x, vol = load_normalized("assets/bunny.obj", 2500, 8, match_volume=v_src, sample="stratified",
                             return_volume=True, frame=frame, fill=fill_a)
    y, vol_y = sample_in_frame("assets/bunny.obj", frame, 2500, 8, fill=fill_b)
    assert y.dtype == np.float32 and np.array_equal(x, y)
    assert vol == vol_y and fill_a["volume"] == fill_b["volume"]
    assert frame["matched"] is match


def _hull_mesh(tmp_path) -> str:
    """A larger target in the bunny's oriented coordinates: its convex hull (orientation 'id' by its name)."""
    from physmorph.sampling.orientation import orient_name
    path = str(tmp_path / "bunny_cvx_r1.obj")
    trimesh.Trimesh(_oriented("assets/bunny.obj").convex_hull.vertices,
                    _oriented("assets/bunny.obj").convex_hull.faces).export(path)
    assert orient_name(path) == "id"
    return path


def _prep(tgt, render_tgt=None, n=3000):
    from physmorph.prepare import prepare
    return prepare("assets/isosphere.obj", tgt, n, 7, 26.0, 1.0, 0.3, log=lambda s: None, surface=n // 4, draws=3,
                   match_density=True, render_tgt=render_tgt)


def _same(a, b):
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(u, v) for u, v in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


def test_prepare_with_its_own_mesh_as_the_render_target_is_the_frozen_path():
    _cuda()
    off, on = _prep("assets/bunny.obj"), _prep("assets/bunny.obj", render_tgt="assets/bunny.obj")
    assert off.render_x is None and np.array_equal(on.render_x, off.tgt)
    for f in ("src", "tgt", "v_src", "v_tgt", "prm", "loss_res", "unit_ref_res", "nn_berth_k", "ppc", "tgt_surface",
              "tgt_draws", "density", "cell_shape"):
        assert _same(getattr(off, f), getattr(on, f)), f


def test_prepare_with_another_render_target_keeps_the_target_and_its_frame(tmp_path):
    _cuda()
    from scipy.spatial import cKDTree
    from physmorph.sampling.mesh import fill_volume_in_frame
    hull = _hull_mesh(tmp_path)
    ref, d = _prep(hull), _prep(hull, render_tgt="assets/bunny.obj")
    # the target is prepared exactly as without a render target: the hull's own sample, centred and scaled by itself
    for f in ("src", "tgt", "v_src", "v_tgt", "prm", "loss_res", "unit_ref_res", "nn_berth_k", "ppc", "tgt_surface",
              "density", "cell_shape"):
        assert _same(getattr(ref, f), getattr(d, f)), f
    # the render's samples: the bunny inside the hull, where it lies in the hull's frame (not centred by itself), at
    # the hull sample's number density (fewer particles: the bunny's fill volume is smaller)
    n_r = len(d.render_x)
    assert 0.6 * 3000 < n_r < 0.99 * 3000 and all(len(x) == n_r for x in d.tgt_draws) and len(d.tgt_draws) == 2
    th = cKDTree(d.tgt)
    sp = float(np.median(th.query(d.tgt, 2)[0][:, 1]))
    assert float(np.abs(d.tgt.astype(np.float64).mean(0)).max()) < 1e-4
    assert float(np.abs(d.render_x.astype(np.float64).mean(0)).max()) > 1e-3
    assert np.all(d.tgt.min(0) <= d.render_x.min(0) + sp) and np.all(d.tgt.max(0) >= d.render_x.max(0) - sp)
    assert float(np.quantile(th.query(d.render_x, 1)[0], 0.95)) <= 1.5 * sp
    sr = float(np.median(cKDTree(d.render_x).query(d.render_x, 2)[0][:, 1]))
    assert abs(sr / sp - 1.0) < 0.05                                   # the same spacing: the same number density
    # the frame: a bunny sample put in it by hand (the hull's frame, from load_normalized of the hull) is the render's
    _, v_src = load_normalized("assets/isosphere.obj", 3000, 7, return_volume=True, sample="stratified")
    frame = {}
    load_normalized(hull, 3000, 8, match_volume=v_src, sample="stratified", frame=frame)
    assert n_r == int(round(3000 * fill_volume_in_frame("assets/bunny.obj", frame, 3000)
                            / fill_volume_in_frame(hull, frame, 3000)))
    x, _ = sample_in_frame("assets/bunny.obj", frame, n_r, 8)
    assert np.array_equal(x, d.render_x)


def _clouds():
    rng = np.random.default_rng(4)
    a = rng.uniform(-1., 1., (300, 3)).astype(np.float32)
    b = (a * np.array([0.8, 1.0, 0.9], np.float32) + np.array([0.1, 0.0, -0.05], np.float32)).astype(np.float32)
    return a, b


def _cfg(**kw):
    from physmorph.pipeline.config import PipelineConfig
    return PipelineConfig(render_views=2, render_elevs=(0., 0.5), render_res=24, loss_res=16, dt_res=16,
                          render_exterior=True, **kw)


def _close(u, v):
    return all(torch.allclose(p, q, rtol=1e-5, atol=1e-6) for p, q in zip(u, v))


def test_build_target_draws_the_pictures_from_render_x():
    _cuda()
    from physmorph.mpm.state import MPMParams
    from physmorph.pipeline.target import build_target
    a, b = _clouds()
    prm = MPMParams(dx=.25, nx=32, ny=32, nz=32, grid_min=(-4., -4., -4.))
    cfg = _cfg()
    pa, pb, pab, paa = (build_target(a, prm, cfg), build_target(b, prm, cfg), build_target(a, prm, cfg, render_x=b),
                        build_target(a, prm, cfg, render_x=a))
    # the pictures and their constants: render_x's
    assert pab.extent == pb.extent and pab.pdx == pb.pdx and pab.pblur == pb.pblur
    assert pab.ext.pitch == pb.ext.pitch and pab.ext.h == pb.ext.h
    assert _close(pab.sils, pb.sils) and _close(pab.shade, pb.shade)
    assert _close(pab.ext.sils, pb.ext.sils) and _close(pab.ext.shade, pb.ext.shade)
    assert not _close(pab.ext.sils, pa.ext.sils)
    # everything else: target_x's
    assert _close([pab.grid], [pa.grid]) and torch.equal(pab.pts, pa.pts) and _close([pab.dt3], [pa.dt3])
    assert pab.dtdx == pa.dtdx and pab.nn_spacing == pytest.approx(pa.nn_spacing, rel=1e-6)
    assert pab.m_ref == pytest.approx(pa.m_ref, rel=1e-5) and pab.extent != pa.extent
    # render_x = target_x: the frozen pack
    assert paa.extent == pa.extent and paa.ext.pitch == pa.ext.pitch and _close([paa.dt3], [pa.dt3])
    assert _close(paa.ext.sils, pa.ext.sils) and _close(paa.ext.shade, pa.ext.shade) and _close(paa.sils, pa.sils)


def test_every_evaluation_of_the_render_terms_reads_the_render_targets_pictures(monkeypatch):
    """The gradient, the line search's candidates and the commit rollout (tr_eval) all score the render terms through
    Objective.render_terms on the run's one TargetPack; with render_x its pictures are render_x's."""
    _cuda()
    from physmorph.mpm.state import MPMParams
    from physmorph.pipeline import run_pipeline
    from physmorph.pipeline.target import build_target
    from physmorph.pipeline.window import objective as OB
    from physmorph.pipeline.window import solve as SO
    rng = np.random.default_rng(11)
    src = rng.uniform(-1.5, 1.5, (300, 3)).astype(np.float32)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
    rnd = (tgt * np.array([0.9, 1.0, 0.95], np.float32)).astype(np.float32)
    prm = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    from physmorph.pipeline.config import PipelineConfig
    cfg = PipelineConfig(T=4, iters=2, animations=2, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                         render_res=24, dt_res=32, patience=3, render_exterior=True)
    seen, state = [], {"commit": False}
    terms, commit = OB.Objective.render_terms, SO.commit_rollout

    def render_terms(self, xT):
        seen.append((state["commit"], self.tgt.ext.sils, self.tgt.extent))
        return terms(self, xT)

    def commit_rollout(*a, **k):
        state["commit"] = True
        try:
            return commit(*a, **k)
        finally:
            state["commit"] = False

    monkeypatch.setattr(OB.Objective, "render_terms", render_terms)
    monkeypatch.setattr(SO, "commit_rollout", commit_rollout)
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None, render_x=rnd)
    assert any(h.get("frame_end") for h in res["history"])
    assert any(c for c, _, _ in seen) and any(not c for c, _, _ in seen)    # the commit rollout and the others
    ref_r, ref_t = build_target(tgt, prm, cfg, render_x=rnd), build_target(tgt, prm, cfg)
    for _, sils, extent in seen:
        assert extent == ref_r.extent != ref_t.extent
        assert _close(sils, ref_r.ext.sils) and not _close(sils, ref_t.ext.sils)
