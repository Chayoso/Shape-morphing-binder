"""replay_probe.py [pipeline_run args] -- D164 step 2, measurement only (run from a deployed repo's root): where does the
commit rollout's E differ from the accepted candidate's, and what does a replay of the committed control measure?

Per judged window it prints one [probe] JSON line with, for three evaluations of the committed control,
  a: the line search's evaluation of the accepted candidate (the value the check compares against),
  x: one extra evaluation of the same control just before the commit, under the commit's own protocol (eval_terms, the
     scored discs restored after it; on the tag's commit (--replay_noise start) a fresh transport solve and proximity
     search, under D164's amendment 1 the accepted candidate's potentials and lists, restored for the commit after it)
     -- the probe's addition,
  c: the commit rollout itself,
their terms (lv geometry energy, lk kinetic, lr render, lstab, cleanup) and E, the window's taped pairs, the start
pair, and whether the window ended by the iteration budget (no taped pair at the committed control).
The extra evaluation leaves the commit the state it would have read without the probe (discs and transport state)."""
import json
import os
import runpy
import sys

sys.path.insert(0, os.getcwd())
import physmorph  # noqa: F401,E402  (before torch: CuPy's CUDA 12 compiler must load first, physmorph/gpu.py)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import physmorph.pipeline.window.rollout as R  # noqa: E402
import physmorph.pipeline.window.solve as S  # noqa: E402

_last = {}
_eval = R.eval_terms


def eval_terms(win, obj, leaf, u):
    e = _eval(win, obj, leaf, u)
    _last["e"] = e
    return e


R.eval_terms = eval_terms
S.eval_terms = eval_terms


def terms(opt, e):
    with torch.no_grad():
        cl = float(opt.obj.cleanup(e.xT.detach()))
    return {"lv": float(e.lv), "lk": float(e.lk), "lr": float(e.lr), "lstab": float(e.lstab) if e.lstab is not None
            else None, "cleanup": cl, "E": opt.scalar(e)}


_ls = S.WindowOptimizer.line_search


def line_search(self, g, gn, cur, e):
    r = _ls(self, g, gn, cur, e)
    if r is not None:
        self._probe_a = terms(self, r[0])
        self._probe_a["E_ls"] = r[2]
    return r


S.WindowOptimizer.line_search = line_search
_commit = S.WindowOptimizer.commit
_roll = S.commit_rollout


def commit(self, hist, replay_rel, grad_converged, ls_exhausted, L_start, leaf0):
    probe = {"replay_rel": replay_rel, "taped": list(self.replays), "accepted": self.accepted,
             "iters_done": [r["iter"] for r in hist], "a": getattr(self, "_probe_a", None) if hist else None}

    def roll(win, obj, leaf, u, lam):
        if hist:
            scored = obj.discs
            ot = self.tgt.grid_ot                    # the commit's transport state as commit() set it (amendment 1:
            keep = (ot.repeat, ot._solved, ot._sup_idx) if ot is not None else None   # the accepted's, repeat)
            ex = eval_terms(win, obj, leaf, u)
            probe["x"] = terms(self, ex)
            obj.discs = scored                       # the commit reads the discs the candidate was scored with
            if keep is not None:                     # and the transport state it would have read without the probe
                ot.repeat, ot._solved, ot._sup_idx = keep
        c = _roll(win, obj, leaf, u, lam)
        probe["c"] = terms(self, _last["e"]) if hist else None
        probe["c_E_final"] = float(c.E_final)
        return c

    S.commit_rollout = roll
    try:
        res = _commit(self, hist, replay_rel, grad_converged, ls_exhausted, L_start, leaf0)
    finally:
        S.commit_rollout = _roll
    probe.update(null=res.stats.get("null_reason"), mismatch=res.stats.get("commit_mismatch"),
                 tol=res.stats.get("commit_tol"), start_verdict=res.stats.get("commit_replay_start"),
                 noise=res.stats.get("replay_noise"), res=res.stats.get("replay_res"))
    print("[probe] " + json.dumps(probe, default=float), flush=True)
    return res


S.WindowOptimizer.commit = commit
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
