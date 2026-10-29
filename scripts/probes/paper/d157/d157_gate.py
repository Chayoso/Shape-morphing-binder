"""d157_gate.py -- D157's gate: the reader's pooled 1 - IoU (1024 px, 18 training views, raw referee) against D154's
`exact_1024` on every state both read (40k: the same constants; 300k: D154 took its box from the independent sample).
Reported per mesh: watertight or fan-closed, the largest absolute difference."""
import glob
import json
from collections import defaultdict

D = "/data/relcfd/chayo/physmorph_v2/output/gpu"
mine = {}
for f in glob.glob(f"{D}/d157/read*.jsonl"):
    for ln in open(f):
        r = json.loads(ln)
        sv = r["sil_views"]["raw"]
        mine[(r["n"], r["mesh"], r["seed"], r["label"])] = (
            1 - sum(sv["inter_px"][:18]) / sum(sv["union_px"][:18]), r["ctx"]["loops_closed"])
theirs = {}
for f in glob.glob(f"{D}/d154/read/end_*.jsonl") + glob.glob(f"{D}/d154/read/g40*.jsonl"):
    for ln in open(f):
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if "exact_1024" not in r or r["raw"] not in ("end", "tgt") and not str(r["raw"]).isdigit():
            continue
        n = "40k" if r.get("ref") == "own" else "300k"
        lab = "own" if r["label"] == "own" else r["label"]
        if n == "300k" and r["raw"] != "end" and lab != "own":
            continue
        theirs.setdefault((n, r["mesh"], r["seed"], lab), r["exact_1024"]["one_minus_iou"])
by = defaultdict(list)
rel = defaultdict(list)
for k, (v, loops) in mine.items():
    if k in theirs:
        by[(k[0], k[1], loops > 0)].append(abs(v - theirs[k]))
        rel[k[0]].append(abs(v - theirs[k]) / theirs[k])
for (n, mesh, closed), d in sorted(by.items()):
    print(f"{n:4s} {mesh:12s} {'fan-closed' if closed else 'watertight'} states {len(d):2d} max |diff| {max(d):.2e}")
for n in ("40k", "300k"):
    w = [x for (nn, _, c), d in by.items() if nn == n and not c for x in d]
    o = [x for (nn, _, c), d in by.items() if nn == n and c for x in d]
    print(n, "watertight", len(w), "max", max(w) if w else None, "| fan-closed", len(o), "max", max(o) if o else None)
for n, v in rel.items():
    v = sorted(v)
    print(n, "relative |diff|: median %.2e, max %.2e, share above 1 %%: %d / %d" % (v[len(v) // 2], v[-1], sum(x > .01 for x in v), len(v)))
print("worst 300k:", sorted(((abs(v - theirs[k]) / theirs[k], k) for k, (v, _) in mine.items() if k in theirs and k[0] == "300k"), reverse=True)[:5])
