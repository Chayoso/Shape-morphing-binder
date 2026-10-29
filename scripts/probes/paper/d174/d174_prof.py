"""d174_prof.py RUN (prefix of output/gpu/d174/runs/NAME) -- D174: where a run's wall time goes.
The log's stamped lines give the phases (prepare, thin set, target build, window loop, settle, end); the history's
--profile sections give the loop by section (inclusive and exclusive seconds, calls), the util CSV the GPU's busy share
per phase."""
import json
import sys
from collections import defaultdict

pre = sys.argv[1]
lines = []
for l in open(pre + ".log", errors="replace"):
    p = l.split(" ", 1)
    try:
        lines.append((float(p[0]), p[1].rstrip("\n") if len(p) > 1 else ""))
    except ValueError:
        pass
t0 = lines[0][0]


def first(s, after=0.0):
    for t, l in lines:
        if s in l and t - t0 >= after:
            return t - t0
    return None


marks = {"start": 0.0, "arm": first("===== ARM"), "thin": first("[v2run] thin set"), "loop": first("[v2] N="),
         "settle_end": first("windows with zero controls"),
         "metrics": first("[gates]"), "saved": first("saved "), "end": lines[-1][0] - t0}
last_anim = max((t - t0 for t, l in lines if "[v2] anim" in l or "converged at anim" in l), default=None)
print("phases (s from start):", {k: (round(v, 1) if v is not None else None) for k, v in marks.items()})
d = json.load(open(pre + ".json"))
arm = list(d["arms"].values())[0]
h = arm["history"]
att = [r for r in h if not r.get("settle")]
st = [r for r in h if r.get("settle")]
print(f"attempts {len(att)}: kept {sum(1 for r in att if not r.get('null_commit') and not r.get('grad_converged'))}, "
      f"null {sum(1 for r in att if r.get('null_commit') and not r.get('outer_rejected'))}, outer-rejected "
      f"{sum(1 for r in att if r.get('outer_rejected'))}; settle {len(st)}; run seconds {arm['seconds']:.1f}")
ta = [r["t_attempt"] for r in att if r.get("t_attempt") is not None]
print(f"t_attempt: sum {sum(ta):.1f} s over {len(ta)} rows, mean {sum(ta) / max(1, len(ta)):.2f} s "
      f"(the last row carries the settle: {h[-1].get('t_attempt')})")
for k in ("t_start", "t_grad", "t_ls", "t_commit"):
    v = [r.get(k) for r in att if r.get(k) is not None]
    print(f"  {k}: mean {sum(v) / max(1, len(v)):.3f} s ({len(v)} rows)")
tr = [int(r.get("ls_trials") or 0) for r in att]
it = [int(r.get("iters") or 0) for r in att]
eb = [int(r.get("ext_builds") or 0) for r in att]
print(f"ls trials {sum(tr)} ({sum(tr) / len(att):.2f} a window), accepted iterations {sum(it)}, exterior searches "
      f"{sum(eb)} ({sum(eb) / len(att):.2f} a window)")
T, N = defaultdict(float), defaultdict(int)
TS, NS = defaultdict(float), defaultdict(int)
for i, r in enumerate(h):
    tgtT, tgtN = (TS, NS) if (r is h[-1] and st) else (T, N)
    for src in (r.get("prof") or {}, r.get("prof_run") or {}):
        for k, v in (src.get("t") or {}).items():
            tgtT[k] += v
        for k, v in (src.get("n") or {}).items():
            tgtN[k] += v


def show(T, N, n_rows, label):
    tot = sum(v for k, v in T.items() if "/" not in k)
    print(f"\n== {label}: top-level sections {tot:.1f} s ({tot / max(1, n_rows):.2f} s a window over {n_rows})")
    excl = {}
    for k, v in T.items():
        kids = sum(v2 for k2, v2 in T.items() if k2.startswith(k + "/") and k2.count("/") == k.count("/") + 1)
        excl[k] = v - kids
    for k in sorted(T, key=lambda k: -T[k]):
        if T[k] >= 0.005 * tot:
            print(f"  {k:<44s} incl {T[k]:8.2f} s  excl {excl[k]:8.2f} s  calls {N.get(k, 0):6d}  "
                  f"{T[k] / max(1, N.get(k, 1)) * 1e3:8.1f} ms/call  {T[k] / tot * 100:5.1f} %")
    leaf = defaultdict(float)
    for k, v in excl.items():
        leaf[k.split("/")[-1]] += v
    print("  -- exclusive by leaf name:")
    for k in sorted(leaf, key=lambda k: -leaf[k])[:25]:
        print(f"     {k:<20s} {leaf[k]:8.2f} s {leaf[k] / tot * 100:5.1f} %  ({leaf[k] / max(1, n_rows):.3f} s a window)")
    for k in sorted(N):
        if k.split("/")[-1].startswith("ot_"):
            print(f"     count {k}: {N[k]}")


show(T, N, len(att), "attempts (the last row's sections, incl. the settle, apart)")
if st:
    show(TS, NS, len(st), "the last attempt + the settle windows")
# GPU utilisation per phase
try:
    import datetime
    util = []
    for l in open(pre + ".util.csv"):
        p = [s.strip() for s in l.split(",")]
        ts = datetime.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
        util.append((ts, float(p[1].rstrip(" %")), float(p[2].split()[0])))
    seg = [("prepare", 0.0, marks["arm"]), ("thin set", marks["arm"], marks["thin"]), ("target build", marks["thin"], marks["loop"]),
           ("window loop", marks["loop"], last_anim), ("settle", last_anim, marks["settle_end"]),
           ("end (metrics, save)", marks["settle_end"] or last_anim, marks["end"])]
    print("\nGPU busy (nvidia-smi utilization.gpu, 1 Hz) and phases:")
    for name, a, b in seg:
        if a is None or b is None:
            continue
        u = [x[1] for x in util if t0 + a <= x[0] < t0 + b]
        m = [x[2] for x in util if t0 + a <= x[0] < t0 + b]
        if u:
            print(f"  {name:<22s} {b - a:7.1f} s  util mean {sum(u) / len(u):5.1f} %  <50 %: "
                  f"{sum(1 for x in u if x < 50) / len(u) * 100:4.0f} % of samples  mem max {max(m):.0f} MiB")
except FileNotFoundError:
    pass
