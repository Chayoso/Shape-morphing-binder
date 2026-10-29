"""d173_shares.py -- D173: the render's share of each control block's step, per window, read from a run's records.

Each optimisation window's first gradient records (window/solve.py _calibrate_lambda, every run since wip/d173 after
3c447081; split runs since 3c447081): g_share_b = lam_b |g_r,b| / (|g_p,b| + lam_b |g_r,b|) for b = dFc (the stress
increment, reached through the MPM adjoint) and u (the outer layer), g_r the render gradient AFTER PCGrad, lam_b the
weight that block's step takes (the run's on both; under --render_split u the held weight on u and the rule's on dFc).
The moments read: start = the first optimisation window; arrival = the first optimisation window whose transport energy
has made 90 % of its fall over the run (E <= E_last + 0.1 (E_first - E_last)); stop = the last optimisation window
(the settle windows after it optimise nothing). A run without the records (older code) reads None; the twin
(lambda = 0) reads 0 by construction."""
import json

import numpy as np


def opt_windows(run_json):
    a = next(iter(json.load(open(run_json))["arms"].values()))
    return [h for h in a["history"] if h.get("frame_end") and not h.get("settle") and not h.get("null_commit")]


def moments(opt):
    if not opt:
        return {}
    E = [h.get("transport_energy") for h in opt]
    i_arr = None
    if all(e is not None for e in E) and len(E) > 1:
        lim = E[-1] + 0.1 * (E[0] - E[-1])
        i_arr = next((i for i, e in enumerate(E) if e <= lim), len(E) - 1)
    return {"start": 0, "arrival": i_arr, "stop": len(opt) - 1}


def block_shares(run_json, twin=False):
    """dict(start/arrival/stop: dict(window, frame_end, u, dfc, lam_u, lam_dfc), mean_u, mean_dfc, recorded, series)."""
    opt = opt_windows(run_json)
    rec = any(h.get("g_share_dfc") is not None for h in opt)
    out = dict(recorded=rec or twin, windows=len(opt), series=[])
    for i, h in enumerate(opt):
        u, d = h.get("g_share_u"), h.get("g_share_dfc")
        if twin and u is None:
            u, d = 0.0, 0.0
        out["series"].append(dict(window=i, frame_end=h.get("frame_end"), u=u, dfc=d, lam_u=h.get("lam_block_u"),
                                  lam_dfc=h.get("lam_block_dfc"), transport=h.get("transport_energy")))
    for k, i in moments(opt).items():
        out[k] = None if i is None else out["series"][i]
    for b in ("u", "dfc"):
        v = [q[b] for q in out["series"] if q[b] is not None]
        out[f"mean_{b}"] = float(np.mean(v)) if v else None
    return out


def fmt3(s, b):
    """'start / arrival / stop' of block b ('u' or 'dfc')."""
    if not s or not s.get("recorded"):
        return "not recorded"
    f = lambda q: "-" if q is None or q.get(b) is None else f"{q[b]:.3f}"   # noqa: E731
    return f"{f(s.get('start'))} / {f(s.get('arrival'))} / {f(s.get('stop'))}"


HEAD = ("render share of u's step (start / arrival / stop)", "render share of dF_c's step, after PCGrad (start / arrival / stop)")
NOTE = ("Render shares: lam_b |g_r,b| / (|g_p,b| + lam_b |g_r,b|) at each optimisation window's first gradient, b = u (the "
        "outer layer) or dF_c (the stress increment, reached through the MPM adjoint), g_r after PCGrad, lam_b the weight the "
        "block's step takes (Rx: the held weight on both; Hu: held on u, the rule's per-window value on dF_c); start = the "
        "first optimisation window, arrival = the first whose transport energy has made 90 % of its fall, stop = the last "
        "(tmp/d173/d173_shares.py). The twin's are 0 by construction (lambda = 0).")
