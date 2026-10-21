"""D155 (option A, `morph_time`, off by default; D155b `morph_u_final` and the exact pack without the particle pictures):
the window's target is the point of the geodesic between the source and
the target at the window's end time (physmorph/pipeline/run/morph.py). The ramp (CPU): zero at the start, one from the
ramp's end, monotone, flat at both ends. The run (CUDA): every window of the ramp reads a pack built from Z_s with the
run's calibrations and opens a selection epoch, the final pack from s = 1 on; the records carry s and the final target's
terms on the way; off, nothing of it reaches the run (no record, one epoch, the run's own pack)."""
import numpy as np
import pytest
import torch

from physmorph.pipeline.run.morph import ramp


def test_the_ramp_is_a_smoothstep_over_the_morph_time():
    assert ramp(0.0, 3.0) == 0.0 and ramp(-1.0, 3.0) == 0.0
    assert ramp(3.0, 3.0) == 1.0 and ramp(7.0, 3.0) == 1.0
    assert ramp(1.5, 3.0) == pytest.approx(0.5)
    s = [ramp(t, 3.0) for t in np.linspace(0.0, 3.0, 61)]
    assert all(b >= a for a, b in zip(s, s[1:]))                          # monotone
    assert ramp(0.03, 3.0) < 1e-3 and 1.0 - ramp(2.97, 3.0) < 1e-3          # flat at both ends


def test_the_ramp_length_is_the_bodys_elastic_period():
    from types import SimpleNamespace
    from scripts.pipeline_run import morph_time
    prep = SimpleNamespace(elastic_period=1.19)
    assert morph_time(SimpleNamespace(morph_ramp="off", morph_ref_n=0, n=40000), prep) == 0.0
    assert morph_time(SimpleNamespace(morph_ramp="elastic", morph_ref_n=0, n=40000), prep) == 1.19
    # the same body with 300k unit masses: the density 7.5x, the P-wave speed / sqrt(7.5), the period x sqrt(7.5)
    assert morph_time(SimpleNamespace(morph_ramp="elastic", morph_ref_n=300000, n=40000), prep) == pytest.approx(
        1.19 * 7.5 ** 0.5)


def test_the_switch_is_off_by_default_and_checked():
    from physmorph.pipeline import PipelineConfig
    assert PipelineConfig().morph_time == 0.0
    with pytest.raises(ValueError):
        PipelineConfig(morph_time=-1.0)
    with pytest.raises(ValueError):
        PipelineConfig(morph_time=1.0, baseline="xu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the pipeline runs on CUDA")
def test_the_ramp_windows_read_the_geodesic_and_the_end_reads_the_target(monkeypatch):
    from physmorph.mpm.state import MPMParams
    from physmorph.pipeline import PipelineConfig, run_pipeline
    from physmorph.pipeline.run import runner
    from physmorph.pipeline.window import solve
    seen = []
    init = solve.WindowOptimizer.__init__

    def recording(self, *a, **kw):
        init(self, *a, **kw)
        seen.append(self.tgt)

    monkeypatch.setattr(solve.WindowOptimizer, "__init__", recording)
    epochs = []
    new_epoch = runner.Selection.new_epoch

    def counting(self, count=True):
        epochs.append(count)
        return new_epoch(self, count)

    monkeypatch.setattr(runner.Selection, "new_epoch", counting)
    rng = np.random.default_rng(7)
    src = rng.uniform(-1.5, 1.5, (300, 3)).astype(np.float32)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
    draws = [(rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)]
    prm = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    base = dict(T=3, iters=2, animations=5, loss_res=12, render_views=2, render_elevs=(0.0, 0.5), render_res=24,
                dt_res=32, patience=5, reject_stop=0)
    window = 2 * base["T"] * prm.dt

    # off: one pack, one epoch (the constructor's), no D155 record
    seen.clear()
    epochs.clear()
    res = run_pipeline(src, tgt, prm, PipelineConfig(**base), log=lambda *_: None, draws=draws)
    assert seen and all(t is seen[0] for t in seen) and epochs == [False]
    assert not any("morph_s" in h or "final_TE" in h for h in res["history"])

    # on: a ramp over three and a half windows (s < 1 at the ends of windows 1-3, s = 1 from window 4)
    seen.clear()
    epochs.clear()
    t_m = 3.5 * window
    res = run_pipeline(src, tgt, prm, PipelineConfig(**base, morph_time=t_m), log=lambda *_: None, draws=draws)
    wins = [h for h in res["history"] if h.get("frame_end") and not h.get("null_commit")]
    assert wins and np.isfinite(res["frames"].x[-1]).all()
    s = [h["morph_s"] for h in wins]
    assert s[0] == pytest.approx(ramp(window, t_m)) and all(b >= a for a, b in zip(s, s[1:]))
    for h in wins:
        if h["morph_s"] < 1.0:
            assert {"final_TE", "final_sil", "final_pbr"} <= set(h) and np.isfinite(h["final_TE"])
            # the commit (its rollout, its selection merit, the record) read the window's in-between target
            assert h["transport_energy"] != h["final_TE"] and h["d_sil"] != h["final_sil"]
        else:
            assert "final_TE" not in h
    packs = []                                                              # the distinct packs, in the order read
    for t in seen:
        if not any(t is p for p in packs):
            packs.append(t)
    assert epochs[0] is False and epochs.count(True) == len(packs)        # a new epoch with every new target
    on_way = [p for p in packs if p.relief is None and p.draws is not None and len(p.draws) == 1
              and not isinstance(p.draws[0], np.ndarray)]
    assert on_way, "no window read an in-between target"
    if s[-1] >= 1.0:
        final = packs[-1]
        assert torch.equal(final.pts, torch.as_tensor(tgt, device=final.pts.device))   # the run's own sample
        assert not any(final is p for p in on_way)
        for p in on_way:                                                    # the run's calibrations on every pack
            assert p.ot_scale == final.ot_scale and p.unit_ratio == final.unit_ratio
            assert not torch.equal(p.pts, final.pts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the pipeline runs on CUDA")
def test_d155b_u_reads_the_final_target_on_the_ramp(monkeypatch):
    """morph_u_final: every pack on the way carries the final pack's gate (u reads its arrival at the final target), the
    final pack its own; without it the packs on the way have none."""
    from physmorph.mpm.state import MPMParams
    from physmorph.pipeline import PipelineConfig, run_pipeline
    from physmorph.pipeline.window import solve
    seen = []
    init = solve.WindowOptimizer.__init__

    def recording(self, *a, **kw):
        init(self, *a, **kw)
        seen.append(self.tgt)

    monkeypatch.setattr(solve.WindowOptimizer, "__init__", recording)
    rng = np.random.default_rng(7)
    src = rng.uniform(-1.5, 1.5, (300, 3)).astype(np.float32)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
    prm = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    base = dict(T=3, iters=2, animations=5, loss_res=12, render_views=2, render_elevs=(0.0, 0.5), render_res=24,
                dt_res=32, patience=5, reject_stop=0, morph_time=3.5 * 2 * 3 * prm.dt)
    for on in (False, True):
        seen.clear()
        run_pipeline(src, tgt, prm, PipelineConfig(**base, morph_u_final=on), log=lambda *_: None)
        final = seen[-1]
        on_way = [t for t in seen if t is not final]
        assert on_way and final.gate_ref is None
        for t in on_way:
            if on:
                ref = final.gate if final.gate is not None else (final.grid, final.ldx, final.ldims)
                assert t.gate_ref is not None and t.gate_ref[0] is ref[0] and t.gate_ref[1:] == ref[1:]
            else:
                assert t.gate_ref is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the pipeline runs on CUDA")
def test_d155b_a_pack_without_the_particle_pictures_is_the_same_pack():
    """build_target(particle_pictures=False) with the render terms on the exterior: the particle pictures are not drawn,
    and every other field the run reads is the same (to float rounding here, the splats' atomics being unordered; the
    bit-for-bit proof is the deterministic run with and without the skip, D155b)."""
    from physmorph.mpm.state import MPMParams
    from physmorph.pipeline import PipelineConfig
    from physmorph.pipeline.target import build_target
    rng = np.random.default_rng(3)
    p = rng.normal(size=(6000, 3))
    x = (p / np.linalg.norm(p, axis=1, keepdims=True) * rng.uniform(0, 1, (6000, 1)) ** (1 / 3) * 1.4).astype(np.float32)
    draw = (p[::-1] / np.linalg.norm(p[::-1], axis=1, keepdims=True) * rng.uniform(0, 1, (6000, 1)) ** (1 / 3)
            * 1.4).astype(np.float32)
    prm = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    cfg = PipelineConfig(loss_res=12, render_views=2, render_elevs=(0.0, 0.5), render_res=24, dt_res=32,
                         render_exterior=True)
    a = build_target(x, prm, cfg, draws=[draw])
    b = build_target(x, prm, cfg, draws=[draw], particle_pictures=False)
    assert a.sils is not None and b.sils is None and b.shade is None and b.ext is not None
    for u, v in zip(a.ext.sils + a.ext.shade, b.ext.sils + b.ext.shade):
        assert torch.allclose(u, v, rtol=1e-5, atol=1e-6)
    for k in ("grid", "dt3", "pts"):
        assert torch.allclose(getattr(a, k), getattr(b, k), rtol=1e-5, atol=1e-6)
    assert (a.extent, a.pdx, a.pblur, a.nn_spacing, a.m_ref, a.n_support) == (b.extent, b.pdx, b.pblur, b.nn_spacing,
                                                                              b.m_ref, b.n_support)
    assert a.ext.pitch == b.ext.pitch and a.ext.h == b.ext.h
