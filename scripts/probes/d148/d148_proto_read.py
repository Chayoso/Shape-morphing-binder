"""d148_proto_read.py DIR PREFIX BASE1,BASE2 ARM[,ARM...] -- D148's 40k go / no-go reading (pre-registered in D148):
the shell against its bulk at windows 1-9 (median, p90; d139_shear.py) of each arm against the baseline runs' range
widened by its width; the split record per committed window ([gap] lines: split node-steps, separating, the separating
share of the split nodes' mass); the particles apart from the body (d120_apart.py: windows 3-6, 3-10, 10-20); the end
(silhouette IoU, chamfer, thin, the last kinetic record), windows and seconds an attempt."""
import json
import re
import sys

import numpy as np

D, pre = sys.argv[1], sys.argv[2]
base, arms = sys.argv[3].split(","), sys.argv[4].split(",")


def jrows(p):
    try:
        return [json.loads(s) for s in open(p) if s.startswith("{")]
    except OSError:
        return []


def shear(tag):
    return {r["window"]: r for r in jrows(f"{D}/shear_{pre}_{tag}.log") if 1 <= r.get("window", 0) <= 9}


def apart(tag):
    rows = []
    try:
        for s in open(f"{D}/apart_{pre}_{tag}.log"):
            m = re.match(r"frame\s+\d+ raw\s+\d+ window\s+([\d.]+) apart\s+(\d+)", s)
            if m:
                rows.append((float(m.group(1)), int(m.group(2))))
    except OSError:
        return None
    w = np.array([r[0] for r in rows])
    a = np.array([r[1] for r in rows])
    pick = lambda lo, hi: int(a[(w >= lo) & (w <= hi)].max()) if ((w >= lo) & (w <= hi)).any() else -1  # noqa: E731
    return pick(3, 6), pick(3, 10), pick(10, 20)


def gap(tag):
    out = []
    try:
        for s in open(f"{D}/gap_{pre}_{tag}.log"):
            m = re.search(r"split node-steps (\d+), separating (\d+); the separating share of the split nodes' mass ([\d.]+)", s)
            if m:
                out.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
    except OSError:
        pass
    return out


def end(tag):
    try:
        a = next(iter(json.load(open(f"{D}/{pre}_{tag}.json"))["arms"].values()))
    except (OSError, StopIteration, KeyError):
        return None
    h = [r for r in a["history"] if r.get("frame_end") and not r.get("null_commit") and r["frame_end"] <= a["deliver_n"]]
    att = [r.get("t_attempt") for r in a["history"] if r.get("t_attempt")]
    m = a["metrics"]
    return dict(sil=m["sil_iou"], chamfer=m["chamfer"], thin=m.get("thin_uncovered"), kin=h[-1]["kin"], windows=len(h),
                attempts=len(a["history"]),
                s_att=float(np.mean(att)) if att else a["seconds"] / max(len(a["history"]), 1))


S = {t: shear(t) for t in base + arms}
print("== the shell against its bulk, windows 1-9 (pitches): baseline range widened by its width; arms")
for k in ("rel_med", "rel_p90"):
    for w in range(1, 10):
        vals = [S[t][w][k] for t in base if w in S[t]]
        if len(vals) < 2:
            continue
        lo, hi = min(vals), max(vals)
        wd = hi - lo
        line = f"  {k} w{w}: base {lo:.2f}-{hi:.2f} band [{lo - wd:.2f}, {hi + wd:.2f}]"
        for t in arms:
            if w in S[t]:
                v = S[t][w][k]
                line += f" | {t} {v:.2f}{' OUT' if not lo - wd <= v <= hi + wd else ''}"
        print(line)
print("== the bulk's own displacement (pitches), windows 1-9")
for t in base + arms:
    print(f"  {t}: " + " ".join(f"{S[t][w]['bulk_moved_med']:.2f}" for w in range(1, 10) if w in S[t]))
print("== the split record per committed window (split node-steps, separating, separating mass share)")
for t in arms:
    g = gap(t)
    print(f"  {t}: {len(g)} commits; " + "; ".join(f"w{i + 1} {a}/{b} {c:.3f}" for i, (a, b, c) in enumerate(g[:40])))
print("== the particles apart from the body: max over windows 3-6, 3-10, 10-20")
for t in base + arms:
    print(f"  {t}: {apart(t)}")
print("== the end")
for t in base + arms:
    e = end(t)
    print(f"  {t}: {e}")
