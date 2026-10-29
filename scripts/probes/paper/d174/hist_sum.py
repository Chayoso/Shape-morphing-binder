"""hist_sum.py RUN.json... -- D174: a run's attempts, kept / null / rejected windows, settle rows and their seconds."""
import json
import sys

for p in sys.argv[1:]:
    d = json.load(open(p))
    arm = list(d["arms"].values())[0]
    h = arm["history"]
    att = [r for r in h if not r.get("settle")]
    st = [r for r in h if r.get("settle")]
    kept = [r for r in att if not r.get("null_commit") and not r.get("grad_converged")]
    nulls = [r for r in att if r.get("null_commit") and not r.get("outer_rejected")]
    rej = [r for r in att if r.get("outer_rejected")]
    ta = [r.get("t_attempt") for r in h if r.get("t_attempt") is not None]
    last = h[-1].get("t_attempt") if h else None
    trials = sum(int(r.get("ls_trials") or 0) for r in att)
    iters = sum(int(r.get("iters") or 0) for r in kept)
    m = arm.get("metrics", {})
    print(f"{p.split('/')[-1]}: attempts {len(att)} (kept {len(kept)}, null {len(nulls)}, outer-rej {len(rej)}), "
          f"settle {len(st)}; sum t_attempt {sum(ta):.0f} s (last row {last if last is None else round(last)} s); "
          f"mean/attempt {sum(ta[:-1]) / max(1, len(ta) - 1):.2f} s; ls trials {trials}, accepted iters {iters}; "
          f"wall {m.get('wall_s') or m.get('minutes') or arm.get('minutes')}")
    ks = [k for k in m if 'min' in k or 'wall' in k or 'time' in k]
    print("   metrics time keys:", {k: m[k] for k in ks})
