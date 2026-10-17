"""D154's diagnostic switch of the render weight by simulated time (run.runner.render_switch; default off).

Off (neither time set) the runner passes the configuration itself to every window, the frozen path. Set, a window
whose first step is before render_from_s or at or after render_until_s runs with the render weight's scale 0, the
twin's, and every other window with the configuration itself."""
import sys

import pytest

from physmorph.pipeline.config import PipelineConfig
from physmorph.pipeline.run.runner import render_switch
from scripts.pipeline_run import parse_args

DT = 1.0 / 240.0                                    # the runs' time step (prm.dt)


def test_off_by_default_is_the_configuration_itself(monkeypatch):
    cfg = PipelineConfig()
    assert cfg.render_from_s is None and cfg.render_until_s is None
    for step in (0, 1440, 4000):
        c, on = render_switch(cfg, step, DT)
        assert c is cfg and on is None
    monkeypatch.setattr(sys, "argv", ["pipeline_run.py"])
    a = parse_args()
    assert a.render_from_s is None and a.render_until_s is None


def test_until_turns_the_render_off_from_the_window_that_starts_at_t():
    cfg = PipelineConfig(render_until_s=6.0)
    for step in (0, 40 * 35):                       # window 35 starts at 5.83 s
        c, on = render_switch(cfg, step, DT)
        assert on and c is cfg
    for step in (40 * 36, 40 * 37):                 # window 36 starts at 6.00 s
        c, on = render_switch(cfg, step, DT)
        assert not on and c.render_weight_scale == 0.0
        assert c.render_until_s == 6.0 and c.lambda_auto == cfg.lambda_auto      # nothing else changes


def test_from_turns_the_render_on_from_the_window_that_starts_at_t():
    cfg = PipelineConfig(render_from_s=6.0, render_weight_scale=1.0)
    c, on = render_switch(cfg, 40 * 35, DT)
    assert not on and c.render_weight_scale == 0.0
    c, on = render_switch(cfg, 40 * 36, DT)
    assert on and c is cfg and c.render_weight_scale == 1.0


def test_the_times_are_checked():
    with pytest.raises(ValueError):
        PipelineConfig(render_from_s=-1.0)
    with pytest.raises(ValueError):
        PipelineConfig(render_until_s=float("nan"))
