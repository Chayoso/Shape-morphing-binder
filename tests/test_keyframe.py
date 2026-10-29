"""D175 picture keyframes (docs/design/keyframe_render_control.md): off by default, the storyboard's clock and segments,
the in-place swap of the pack, the config's preconditions. The bit-for-bit gate of the frozen path and the runs' term and
hold checks are the server gate (scripts/probes/paper/d175/208_d175_kf40.sh)."""
from types import SimpleNamespace

import pytest

from physmorph.pipeline.config import PipelineConfig
from physmorph.pipeline.run.storyboard import Storyboard, entry_of, frame_of


def test_off_by_default():
    c = PipelineConfig()
    assert c.keyframe_time == () and c.keyframe_geometry == "own"


def test_the_segment_boundary_falls_at_frame_240():
    assert frame_of(1.0, 20, 1.0 / 240.0) == 240
    # a commit (or a hold) adds 2T = 40 frames: windows 1..6 end at 40..240 and read the keyframe, window 7 the target
    ends = [len_frames - 1 + 40 for len_frames in range(1, 1 + 40 * 7, 40)]
    assert ends == [40, 80, 120, 160, 200, 240, 280]
    assert [entry_of(e, [240]) for e in ends] == [0] * 6 + [None]
    assert [entry_of(e, [80, 240]) for e in ends] == [0, 0, 1, 1, 1, 1, None]


def test_the_preconditions():
    PipelineConfig(keyframe_time=(1.0,), render_exterior=True, lambda_ema=0.0, render_split="u")
    for kw in (dict(lambda_ema=0.3), dict(render_exterior=False, lambda_ema=0.0),
               dict(lambda_ema=0.0, render_split="dFc"), dict(lambda_ema=0.0, render_split="u_rule"),
               dict(lambda_ema=0.0, render_weight_ref="floor")):
        with pytest.raises(ValueError):
            PipelineConfig(keyframe_time=(1.0,), **{"render_exterior": True, **kw})
    with pytest.raises(ValueError):
        PipelineConfig(keyframe_time=(1.0, 0.5), render_exterior=True, lambda_ema=0.0)
    with pytest.raises(ValueError):
        PipelineConfig(keyframe_geometry="none")


def _board(own=True):
    sb = Storyboard.__new__(Storyboard)
    sb.final_ext, sb.final_relief, sb.exts, sb.frames, sb.own, sb.cur = "E", "R", ["K"], [240], own, "unset"
    return sb


def test_the_pack_is_swapped_in_place_and_keeps_the_run_state():
    sb = _board()
    tgt = SimpleNamespace(ext="E", relief="R", geometry=True, keyframe=None, settled_step=0.5, u_adam=("m", "v"))
    assert sb.apply(tgt, 40) == (0, True)
    assert (tgt.ext, tgt.relief, tgt.geometry, tgt.keyframe) == ("K", None, False, 0)
    assert sb.apply(tgt, 240) == (0, False)
    assert sb.apply(tgt, 280) == (None, True)
    assert (tgt.ext, tgt.relief, tgt.geometry, tgt.keyframe) == ("E", "R", True, None)
    assert tgt.settled_step == 0.5 and tgt.u_adam == ("m", "v")


def test_the_naive_arm_keeps_the_final_geometry():
    sb = _board(own=False)
    tgt = SimpleNamespace(ext="E", relief="R", geometry=True, keyframe=None)
    sb.apply(tgt, 40)
    assert (tgt.ext, tgt.relief, tgt.geometry, tgt.keyframe) == ("K", "R", True, 0)
