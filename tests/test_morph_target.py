"""D155 (option A, `morph_time`, off by default): the window's target is the point of the geodesic between the source and
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
