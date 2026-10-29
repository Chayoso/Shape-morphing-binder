"""d174_stop.py RUN.json... -- D174: where the existing stop rule (run/selection.py: a window improves when its merit is below
best - tol |best|; the run stops after `patience` windows without an improvement) would have stopped a recorded run for
other values of cfg.tol (patience 5), replayed on the run's own merits (the recorded windows; the anneal / latch feedback a
different tol has on later windows is not replayed). tol 0.003 must reproduce the recorded stop."""
import json
import sys

TOLS = (0.003, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20)
PAT = 5


def replay(h, tol, pat=PAT):
    best, last, stale, kept = float("inf"), None, 0, 0
    for i, r in enumerate(h):
        if r.get("settle"):
            break
        a = r.get("animation", i)
        if last is not None and a - last > pat:                    # sel.plateau at the top of the loop
            return i, kept, "plateau"
        if r.get("grad_converged"):
            return i, kept, "converged"
        q = r.get("selection_merit")
        if r.get("null_commit") and not r.get("outer_rejected"):
            stale += 1
            if stale >= pat:
                return i + 1, kept, "null"
            continue
        if q is None:
            continue
        improved = last is None or q < best - tol * abs(best)
        if r.get("outer_rejected"):
            if not r.get("brake_reject") or r.get("replay"):
                stale += 1
            if stale >= pat or r.get("reject_streak", 0) >= 3:
                return i + 1, kept, "rejects"
            continue
        kept += 1
        if improved:
            best, last = q, a
        stale = 0 if improved else stale + 1
        if stale >= pat:
            return i + 1, kept, "patience"
    return len([r for r in h if not r.get("settle")]), kept, "end"


for p in sys.argv[1:]:
    a = list(json.load(open(p))["arms"].values())[0]
    h = a["history"]
    att = [r for r in h if not r.get("settle")]
    ta = [r.get("t_attempt") or 0.0 for r in att]
    out = []
    for tol in TOLS:
        n, k, why = replay(h, tol)
        out.append(f"{tol:g}: {n} att / {k} kept ({why}, {sum(ta[:n]):.0f} s)")
    print(f"{p.split('/')[-1]} [{len(att)} att]: " + " | ".join(out))
