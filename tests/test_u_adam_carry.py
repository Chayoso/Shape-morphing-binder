"""D146b (`u_adam_carry`): u's Adam moments carry from one committed window to the next, each coordinate's bias
correction counting its own age. The direction (CPU): at age 1 it is Adam's first step, a coordinate that never acted
gets zero, and a coordinate whose gradient turns from one window to the next moves less over the second window than
with fresh moments. The run (CUDA): the moments reach the next window, the switch off keeps them fresh, an outer reject
resets them."""
import numpy as np
import pytest
import torch

from physmorph.pipeline.window.solve import aged_adam_direction

B1, B2, EPS = 0.9, 0.999, 1e-3


def _adam(g_seq, m=None, v=None, age=None, t0=0, carry=True):
    """A window of Adam steps on one coordinate per entry (the line search's update, every trial accepted)."""
    m = torch.zeros_like(g_seq[0]) if m is None else m.clone()
    v = torch.zeros_like(g_seq[0]) if v is None else v.clone()
    age = torch.zeros_like(g_seq[0]) if age is None else age.clone()
    total = torch.zeros_like(g_seq[0])
    for k, g in enumerate(g_seq):
        m = B1 * m + (1 - B1) * g
        v = B2 * v + (1 - B2) * g * g
        if carry:
            d = aged_adam_direction(m, v, age + (g != 0).float(), B1, B2, EPS)
            age = age + (g != 0).float()
        else:
            t = t0 + k + 1
            d = (m / (1 - B1 ** t)) / ((v / (1 - B2 ** t)).sqrt() + EPS)
        total = total - d
    return total, m, v, age


def test_the_first_step_is_adams_and_an_idle_coordinate_does_not_move():
    g = torch.tensor([0.5, -2.0, 0.0])
    d = aged_adam_direction((1 - B1) * g, (1 - B2) * g * g, (g != 0).float(), B1, B2, EPS)
    want = g / (g.abs() + EPS)
    assert torch.allclose(d[:2], want[:2], rtol=1e-5)
    assert d[2] == 0.0 and torch.isfinite(d).all()


def test_a_coordinate_that_starts_late_gets_the_first_step():
    """A coordinate idle for a whole window (gate closed) and then given a gradient takes Adam's first step. A count
    shared with the coordinates that acted would misread its age: too small a step after a few iterations, up to
    3.2x too large after many (Kingma and Ba 2015, sec. 2.1: the sparse case)."""
    first = [torch.tensor([1.0, 0.0])] * 8
    _, m, v, age = _adam(first)
    step, *_ = _adam([torch.tensor([1.0, 1.0])], m, v, age)
    assert abs(float(step[1]) + 1.0 / (1.0 + EPS)) < 1e-5          # Adam's first step for the newcomer

    def shared(t):
        m1, v1 = (1 - B1) * 1.0, (1 - B2) * 1.0
        return (m1 / (1 - B1 ** t)) / (np.sqrt(v1 / (1 - B2 ** t)) + EPS)

    assert shared(9) < 0.6 and shared(5000) > 2.5


def test_a_turning_gradient_is_damped_and_a_steady_one_is_not():
    """Two windows of eight iterations: a coordinate whose gradient turns between them (the window-to-window reversal
    of D146) moves much less over the second window with carried moments than with fresh ones; a steady one moves
    as much."""
    w1 = [torch.tensor([1.0, 1.0])] * 8
    w2 = [torch.tensor([-1.0, 1.0])] * 8
    _, m, v, age = _adam(w1)
    carried, *_ = _adam(w2, m, v, age)
    fresh, *_ = _adam(w2, carry=False)
    assert abs(float(carried[0])) < 0.35 * abs(float(fresh[0]))   # the turning coordinate: damped
    assert abs(float(carried[1])) > 0.95 * abs(float(fresh[1]))   # the steady one: as before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the window runs on CUDA")
def test_the_moments_reach_the_next_window_and_a_reject_resets_them(monkeypatch):
    from physmorph.mpm.state import MPMParams
    from physmorph.pipeline import PipelineConfig, run_pipeline
    from physmorph.pipeline.run import runner
    from physmorph.pipeline.window import solve
    seen = []
    init = solve.WindowOptimizer.__init__

    def recording(self, *a, **kw):
        init(self, *a, **kw)
        seen.append((self.u_age is not None, self.tgt.u_adam is not None))

    monkeypatch.setattr(solve.WindowOptimizer, "__init__", recording)
    rng = np.random.default_rng(7)
    src = rng.uniform(-1.5, 1.5, (300, 3)).astype(np.float32)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
    prm = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    base = dict(T=3, iters=2, animations=3, loss_res=12, render_views=2, render_elevs=(0.0, 0.5), render_res=24,
                dt_res=32, patience=5, reject_stop=0)

    def committed(res, a):
        return any(h["animation"] == a and h.get("frame_end") and not h.get("null_commit") for h in res["history"])

    seen.clear()
    run_pipeline(src, tgt, prm, PipelineConfig(**base), log=lambda *_: None)
    assert seen and all(s == (False, False) for s in seen)       # off: fresh moments every window, the frozen path
    seen.clear()
    res = run_pipeline(src, tgt, prm, PipelineConfig(**base, u_adam_carry=True), log=lambda *_: None)
    assert seen[0] == (True, False)                               # the first window starts fresh
    if committed(res, 0) and len(seen) > 1:
        assert seen[1] == (True, True)                            # the next one starts from the carried state
    judge = runner.Selection.judge

    def reject_second(self, rec, components, disp):
        out = judge(self, rec, components, disp)
        return (True, False, False) if rec["animation"] == 1 else out

    monkeypatch.setattr(runner.Selection, "judge", reject_second)
    seen.clear()
    res = run_pipeline(src, tgt, prm, PipelineConfig(**base, u_adam_carry=True), log=lambda *_: None)
    if any(h["animation"] == 1 and h.get("outer_rejected") for h in res["history"]) and len(seen) > 2:
        assert seen[2] == (True, False)                           # an outer reject: the next window starts fresh
