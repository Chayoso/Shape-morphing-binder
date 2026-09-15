"""Oscillation triage probe on synthetic archives (numpy only; each case < 2 s)."""
import json

import numpy as np
import pytest

from scripts.probes import oscillation_triage as ot

N, T, W = 200, 10, 30


def _run(kind, **kw):
    arrays, hist, prov, cfg = ot.make_synthetic_archive(kind, n=N, T=T, windows=W, **kw)
    return ot.triage(arrays, hist, prov, cfg), (arrays, hist, prov, cfg)


def test_stop_and_go_is_driver_c():
    rep, _ = _run("stopgo", cfl=0.1)
    d, s = rep["decision"], rep["speed"]
    assert s["window_locked"] and abs(s["period"] - T) <= 0.15 * T
    assert s["n_windows"] == W and s["sag_median"] > 0.5 and s["jump_median"] > 2
    assert s["kin_end_frac"] > 0.9
    assert d["driver_C"] and not d["driver_A"] and not d["driver_B"] and not d["cfl_violation"]
    assert d["visible"] and "C_control" in d["verdict"]
    c = rep["control"]                      # alternating commits -> reversals in history
    assert c["reversal_frac_lt_thr"] > 0.9 and c["n_accepted"] == W


def test_elastic_ringing_is_driver_b():
    rep, _ = _run("stiffness", cfl=0.4)
    d, s, k = rep["decision"], rep["speed"], rep["stiffness"]
    assert k["cfl"] == pytest.approx(0.4, rel=1e-6)
    # speed is |velocity|: the rectified mode shows at tau_e/2
    assert abs(s["period"] - k["tau_e_steps"] / 2) <= 0.25 * k["tau_e_steps"] / 2
    assert not s["window_locked"] and k["stiffness_ringing"]
    assert d["driver_B"] and not d["driver_C"] and not d["driver_A"]


def test_breathing_is_driver_a():
    rep, _ = _run("volume", cfl=0.1)
    d, v = rep["decision"], rep["volume"]
    assert v["n_samples"] == W + 1 and v["meanJ_p2p"] > 0.02
    assert abs(v["corr_J_speed"]) > 0.5      # speed peaks when J crosses its trend
    assert v["proxy_rel_p2p"] > 0.02 and v["corr_J_proxy"] > 0.9
    assert d["driver_A"] and not d["driver_C"] and not d["driver_B"]


def test_smooth_drift_has_no_driver_and_is_invisible():
    rep, _ = _run("drift", cfl=0.1)
    d, z = rep["decision"], rep["visibility"]
    assert d["drivers"] == [] and "no identifiable driver" in d["verdict"]
    assert not d["visible"] and z["frac_excursion_gt_half_sp"] <= 0.01
    assert z["frac_move_gt_half_sp"] <= 0.01 and z["n_tail_commits"] == W


def test_robustness_held_frames_missing_history_truncated_delivery():
    arrays, hist, prov, cfg = ot.make_synthetic_archive("stopgo", n=N, T=T, windows=W, held_every=4)
    assert len(arrays["frames"]) == 1 + W * T + W // 4
    rep = ot.triage(arrays, hist, prov, cfg)                 # history: held frames flagged
    assert rep["speed"]["n_held_steps"] == W // 4 and rep["speed"]["n_windows"] == W
    assert rep["control"]["n_held"] == W // 4 and rep["decision"]["driver_C"]
    rep2 = ot.triage(arrays, None, None, None, T=T)          # no json at all
    assert rep2["speed"]["n_windows"] == W and rep2["decision"]["driver_C"]
    assert not rep2["control"]["has_history"] and rep2["provenance"]["source"]["dt"] == "default"
    with pytest.raises(ValueError):
        ot.triage(arrays, None, None, None)                  # T unknowable
    arrays["deliver_n"] = np.int64(len(arrays["frames"]) * 6 // 10)   # best-commit truncation
    rep3 = ot.triage(arrays, hist, prov, cfg)
    assert 0 < rep3["speed"]["n_windows"] < W and rep3["visibility"]["n_tail_commits"] < W
    slim = {k: v for k, v in arrays.items() if k in ("frames",)}   # frames only
    rep4 = ot.triage(slim, None, None, None, T=T)
    assert rep4["volume"]["n_samples"] == 0 and rep4["decision"]["driver_A"] is False
    json.dumps(rep4)                                          # JSON-safe (no inf/nan)


def test_cli_roundtrip_through_saved_archive(tmp_path):
    arrays, hist, prov, cfg = ot.make_synthetic_archive("stopgo", n=N, T=T, windows=W)
    npz, js = ot.save_archive(arrays, hist, prov, cfg, str(tmp_path / "run"), arm="synthetic")
    assert ot._guess_json(npz) == js
    out = tmp_path / "triage.json"
    rep = ot.main(["--npz", npz, "--json", js, "--arm", "synthetic", "--out", str(out)])
    saved = json.loads(out.read_text())
    assert saved["decision"]["driver_C"] and saved["inputs"]["arm"] == "synthetic"
    assert saved["provenance"]["source"]["dt"] == "json.mpm" and saved["provenance"]["T"] == T
    assert saved["provenance"]["dt"] == pytest.approx(prov["mpm"]["dt"])
    assert rep["decision"] == saved["decision"]


def test_mid_window_turning_point_is_driver_c():
    """hyde06 2026-09-15 signature: speed V-shaped inside every window (0.47 -> 0.10 ->
    0.45), continuous across boundaries. sag ~ 0 and jump ~ 1 must not hide it."""
    import numpy as np
    from scripts.probes import oscillation_triage as OT
    rng = np.random.default_rng(5)
    N, T, W, dt = 200, 10, 30, 1.0 / 240
    base = rng.uniform(-1, 1, (N, 3)).astype(np.float32)
    frames = [base.copy()]
    x = base.copy()
    for w in range(W):
        for t in range(T):
            # velocity along +x decelerates through zero and re-accelerates the other
            # way inside every window (speed V-shaped, min mid-window, max at both ends;
            # the hyde06 signature: 0.47 -> 0.10 -> 0.45, continuous across the boundary)
            v = 0.5 * np.cos(np.pi * t / (T - 1))
            x = x + np.array([v * dt, 0, 0], np.float32) * (1.0 + 0.02 * rng.standard_normal())
            frames.append(x.copy())
    arrays = dict(frames=np.stack(frames), deliver_n=len(frames), src=base, tgt=base)
    rep = OT.triage(arrays, history=None, T=T, dt=dt, dx=0.5, young=1.4e5, poisson=0.2)
    assert rep["speed"]["window_locked"]
    assert rep["speed"]["modulation_median"] > 2.0
    assert rep["decision"]["driver_C"], rep["decision"]
