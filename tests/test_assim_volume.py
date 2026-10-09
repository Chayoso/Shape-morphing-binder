"""--assim_volume (D135): the per-commit plastic assimilation takes the elastic stretch's volume too.

With the motion's volume carried in F (D131, `--volume_exact carried`) the control can no longer pile volume into F, and
the isochoric assimilation leaves every volume change elastic: a volume the end state keeps has to be held by the control
(D131-D134). With the flag `assimilate_elastic(..., isochoric=False)` takes eta of it into Fp at every commit, so it becomes
the rest volume (det Fp); the band [assim_smin, assim_smax] bounds each principal stretch of Fp and so det Fp too.

(1) the flag is defined where F's volume is the one the stress reads (off, carried, smoothed) and refused with history /
    motion, whose stress reads (J / det F)^(1/3) F;
(2) in the pipeline: off = the isochoric rule (det Fp = 1); on = a plastic volume (det Fp != 1), recorded per window.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from physmorph.mpm.state import MPMParams


@pytest.mark.parametrize("vx", ["history", "motion"])
def test_refused_where_the_stress_reads_another_volume(vx):
    from physmorph.pipeline import PipelineConfig
    with pytest.raises(ValueError):
        PipelineConfig(volume_exact=vx, assim_volume=True)


@pytest.mark.parametrize("vx", ["off", "carried", "smoothed"])
def test_accepted_where_F_carries_the_stress_volume(vx):
    from physmorph.pipeline import PipelineConfig
    assert PipelineConfig(volume_exact=vx, assim_volume=True).assim_volume


@pytest.fixture(scope="module")
def clouds():
    rng = np.random.default_rng(7)
    src = rng.uniform(-1.5, 1.5, (300, 3)).astype(np.float32)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
    return src, tgt


@pytest.mark.parametrize("on", [False, True])
def test_the_rest_volume_takes_the_kept_volume_only_with_the_flag(clouds, on):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from physmorph.pipeline import PipelineConfig, run_pipeline
    prm = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    cfg = PipelineConfig(T=3, iters=2, animations=4, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                         render_res=24, dt_res=32, patience=10, c2f_event=False, volume_exact="carried",
                         assim_volume=on)
    res = run_pipeline(*clouds, prm, cfg, log=lambda *_: None)
    assert all(v == 0 for v in res["guards"].values())
    jp = np.linalg.det(np.asarray(res["Fp"]).reshape(-1, 3, 3))
    recs = [h for h in res["history"] if h.get("frame_end")]
    assert recs, "no committed window"
    if on:
        assert float(np.abs(jp - 1.0).max()) > 1e-3, float(np.abs(jp - 1.0).max())
        assert all({"Jp_min", "Jp_p50", "Jp_max"} <= set(h) for h in recs)
        assert float(jp.min()) >= cfg.assim_smin ** 3 and float(jp.max()) <= cfg.assim_smax ** 3
    else:
        assert float(np.abs(jp - 1.0).max()) < 1e-4, float(np.abs(jp - 1.0).max())
        assert not any("Jp_min" in h for h in recs)
