"""Mass-ejection fix: the isolation counter, the escape-velocity hinge, and a pipeline smoke
with the veto on (warp CPU)."""
import numpy as np
import torch

from physmorph.pipeline.runner import _iso_count
from tests.test_render_controls_physics import clouds, prm  # noqa: F401  (module fixtures)


def test_iso_count_flags_only_the_lone_particle():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (500, 3)).astype(np.float32)          # spacing ~0.15
    x[0] = (5.0, 5.0, 5.0)
    assert _iso_count(x, 0.5) == 1
    assert _iso_count(x, 0.05) > 1                                  # tighter radius: many


def test_escape_hinge_is_zero_for_coherent_motion_and_positive_for_a_launched_particle():
    torch.manual_seed(0)
    N = 300
    x0 = torch.rand(N, 3)
    from scipy.spatial import cKDTree
    nbr = torch.as_tensor(cKDTree(x0.numpy()).query(x0.numpy(), k=9)[1][:, 1:])
    v = torch.tensor([0.3, 0.0, 0.1]).expand(N, 3).clone() + 0.01 * torch.randn(N, 3)
    def hinge(vT, esc_k=3.0):
        speed = vT.norm(dim=1); thr = (esc_k * speed.median()).clamp(min=1e-6)
        rel = (vT - vT[nbr].mean(1)).norm(dim=1)
        return torch.relu(rel - thr).pow(2).mean() / (thr * thr)
    assert float(hinge(v)) == 0.0
    v2 = v.clone(); v2[0] += torch.tensor([3.0, 0.0, 0.0])
    assert float(hinge(v2)) > 0.0


def test_pipeline_smoke_with_veto_and_hinge(prm, clouds):
    from tests.test_render_controls_physics import _cfg, _recs, run_pipeline
    src, tgt = clouds
    cfg = _cfg(lambda_auto=0.5, w_esc=1.0, eject_veto=True, outer_merit=True)
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    recs = _recs(res)
    assert recs and np.isfinite(recs[-1]["loss"])
    iso = [r["iso_count"] for r in res["history"] if "iso_count" in r and not r.get("outer_rejected")]
    # accepted commits never increase the isolated count above the window start
    for r in res["history"]:
        if "iso_count" in r and not r.get("outer_rejected"):
            assert r["iso_count"] <= r["iso_start"]


def test_continuity_rule_scale_and_smoke(prm, clouds):
    """The continuity limit is sp_i/(T dt) per particle (discretisation scale) and the pipeline
    runs with the feasibility check on; a synthetic launched particle violates the rule."""
    import torch
    from tests.test_render_controls_physics import _cfg, _recs, run_pipeline
    src, tgt = clouds
    cfg = _cfg(lambda_auto=0.5, continuity=True)
    res = run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    recs = _recs(res)
    assert recs and np.isfinite(recs[-1]["loss"])
    # rule check on synthetic data: coherent motion passes, one launched particle fails
    from scipy.spatial import cKDTree
    x0 = torch.as_tensor(src)
    nbr = torch.as_tensor(cKDTree(src).query(src, k=9)[1][:, 1:])
    sp = (x0[nbr] - x0[:, None, :]).norm(dim=2).mean(1)
    lim = sp / (cfg.T * prm.dt)
    v = torch.tensor([0.2, 0.0, 0.1]).expand(len(src), 3).clone()
    rel = (v - v[nbr].mean(1)).norm(dim=1)
    assert bool((rel <= lim).all())
    # a launch: relative speed of many spacings per window (the fixture's T=4 window is
    # 1/60 s, so the limit is ~60 x sp; real ejecta at T=20 run at 5-20x the limit)
    v[0] += torch.tensor([200.0, 0.0, 0.0])
    rel = (v - v[nbr].mean(1)).norm(dim=1)
    assert int((rel > lim).sum()) >= 1 and bool(rel[0] > lim[0])
