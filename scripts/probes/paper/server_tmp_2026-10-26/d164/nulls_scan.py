"""nulls_scan.py DIR... -- D164 step 2 (measurement only): the tag's commit_replay nulls in existing run JSONs: the
mismatch (commit_E_final - E_accept) relative and in float32 ulps of E_accept, the start pair's replay_rel, the window
index, and how many would pass a tolerance of 10 ulps (the resolution floor alone)."""
import glob, json, sys
import numpy as np
rows, judged, runs = [], 0, 0
for d in sys.argv[1:]:
    for p in sorted(glob.glob(d + "/*.json")):
        try:
            js = json.load(open(p))
        except Exception:
            continue
        if "arms" not in js:
            continue
        runs += 1
        for arm in js["arms"].values():
            H = arm.get("history", [])
            for h in H:
                if "animation" in h and not h.get("grad_converged"):
                    judged += 1
                if h.get("null_reason") == "commit_replay" and h.get("E_accept") is not None:
                    Ea, Ec = h["E_accept"], h["commit_E_final"]
                    ulp = float(np.spacing(np.float32(abs(Ea))))
                    rows.append(dict(run=p.split("/")[-1], a=h["animation"], rel=(Ec - Ea) / abs(Ea), ulps=(Ec - Ea) / ulp,
                                     rr=h.get("replay_rel") or 0.0))
print(f"runs {runs} judged windows {judged} commit_replay nulls {len(rows)} ({100 * len(rows) / max(judged, 1):.2f} %)")
if rows:
    rel = np.array([r["rel"] for r in rows]); ul = np.array([r["ulps"] for r in rows]); rr = np.array([r["rr"] for r in rows])
    a = np.array([r["a"] for r in rows])
    q = lambda x: " ".join(f"{v:.3g}" for v in np.quantile(x, [0, .1, .25, .5, .75, .9, .99, 1]))
    print("mismatch rel  q0 q10 q25 q50 q75 q90 q99 max:", q(rel))
    print("mismatch ulps q0 q10 q25 q50 q75 q90 q99 max:", q(ul))
    print("start pair replay_rel: zero in", int((rr == 0).sum()), "of", len(rr), "; quantiles:", q(rr))
    print("window index: first window", int((a == 0).sum()), "; quantiles", q(a))
    for k in (10, 20, 50, 100):
        print(f"  nulls with mismatch <= {k} ulps: {int((ul <= k).sum())}")
