"""d154_switch.py OUT_DIR -- D154 (2)'s reading (CPU, read-only): the render switched off (Loff6) or on (Pon6) at 6 s
against B1's render arm L and twin P of the same seed (and, where run, the replicates L / P of seed 99).

Per measure, per raw frame r, lead(r) = (P(r) - X(r)) / P(r) with P = B1's twin (its kept frames' readings: the
yardstick t3_*.log, the display e3_*.log; the floor-free reads of output/gpu/d154/read; every window's record), against
B1's lead_LP(r) = (P(r) - L(r)) / P(r). Per committed window n after the switch (n = 1 is the window that starts at
6.0 s, its frames 1441-1480) the mean over its kept frames; the shares f_lost = 1 - lead_X / lead_LP (Loff6) and
f_built = lead_X / lead_LP (Pon6); the first window at which the share reaches 0.5, the share at 10 and at 20 windows.
Before the switch the run is a replicate of L (Loff6) or of P (Pon6): its relative difference from that B1 run over
5-6 s is the replicate spread. Writes switch.json (all series and the verdict per pre-registered rule) and prints."""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

B = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
B1, O = B / "d145t_b1", B / "d154"
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
T0, W = 1440, 40                                              # the switch at 6 s = step 1440; 40 steps a window


def jl(path):
    try:
        return [json.loads(s) for s in open(path) if s.startswith("{")]
    except OSError:
        return []


def t3(path):                                                 # the yardstick per kept frame (GPU logs of B1)
    return {int(r["state"]): {"yard_sil": r["exterior"]["sil"], "yard_pbr": r["exterior"]["pbr"]} for r in jl(path)}


def e3(path):                                                 # the display against the independent sample
    d = {}
    for r in jl(path):
        if str(r.get("state")).isdigit() and "to_target" in r:
            d[int(r["state"])] = {"disp_front": 1 - r["to_target"]["front"]["iou"],
                                  "disp_back": 1 - r["to_target"]["back"]["iou"],
                                  "disp_diff": r["to_target"]["front"]["difference"]}
    return d


EXACT = {"iou1024": lambda r: r["exact_1024"]["one_minus_iou"], "outline_sd": lambda r: r["exact_1024"]["outline_state_sd_a"],
         "outline_mesh_sd": lambda r: r["exact_1024"]["outline_mesh_sd_a"], "normal_cl": lambda r: r["dist"]["normal_closest_deg"],
         "shading_x": lambda r: r["exact_1024"]["shading_mse"], "dist_sd": lambda r: r["dist"]["state_to_mesh_signed_sd"],
         "yard_mesh_sil": lambda r: r["yard_mesh"]["sil"], "yard_mesh_pbr": lambda r: r["yard_mesh"]["pbr"]}


def mine(name, kind):                                         # my CPU reads: yard (yardstick) or exact (floor-free)
    d = {}
    files = glob.glob(str(O / f"read/sw_{kind}_{name}_*.jsonl"))
    if kind == "exact":                                       # B1's seed-97 frames read every 0.5 s (read/ref_K.jsonl)
        files += glob.glob(str(O / "read/ref_[0-9]*.jsonl"))
    for f in files:
        for r in jl(f):
            if not str(r["raw"]).isdigit() or r["npz"] != f"{name}_frames12.npz":
                continue
            v = {"yard_sil": r["yard_ind"]["sil"], "yard_pbr": r["yard_ind"]["pbr"]}
            if kind == "exact":
                v.update({k: g(r) for k, g in EXACT.items()})
            d.setdefault(int(r["raw"]), {}).update(v)
    return d


def records(path):                                            # the run's own render terms, every committed window
    try:
        J = json.loads(Path(path).read_text())
    except OSError:
        return {}
    hist = next(iter(J["arms"].values()))["history"]
    d = {}
    for h in hist:
        if h.get("frame_end") and not h.get("null_commit") and h.get("d_sil") is not None:
            d[int(h["frame_end"]) - 1] = {"rec_sil": h["d_sil"], "rec_pbr": h["d_pbr"], "lam": h.get("lambda"),
                                          "g_share": h.get("g_share"), "switch": h.get("render_switch_on")}
    return d


def series(kind, mesh, arm, seed):
    """{raw: {measure: value}} of one run: B1's (kind b1) or this entry's (kind d154)."""
    s = {}
    if kind == "b1":
        name = f"{mesh}300k_{arm}{seed}"
        ex = mine(name, "exact")
        if not ex:                                            # seed 99: B1's frames are gone; the replicate's (Q), if run
            ex = mine(f"{mesh}300k_{arm}_{seed}", "exact")
        for src in (t3(B1 / f"t3_{name}.log"), e3(B1 / f"e3_{name}.log"), records(B1 / f"{name}.json"),
                    {k: {m: v for m, v in x.items() if m not in ("yard_sil", "yard_pbr")} for k, x in ex.items()}):
            for r, v in src.items():
                s.setdefault(r, {}).update(v)
    else:
        name = f"{mesh}300k_{arm}_{seed}"
        for src in (mine(name, "yard"), e3(O / f"runs/e3_{name}.log"), records(O / f"runs/{name}.json"), mine(name, "exact")):
            for r, v in src.items():
                s.setdefault(r, {}).update(v)
    return s


def per_window(s, key, n):
    lo, hi = T0 + W * (n - 1), T0 + W * n                     # window n's frames: (lo, hi]
    v = [x[key] for r, x in s.items() if lo < r <= hi and x.get(key) is not None]
    return float(np.mean(v)) if v else None


MEAS = ["yard_sil", "yard_pbr", "rec_sil", "rec_pbr", "disp_front", "disp_back", "iou1024", "outline_sd", "outline_mesh_sd",
        "normal_cl", "shading_x", "dist_sd", "yard_mesh_sil", "yard_mesh_pbr"]
PRIMARY = ("yard_sil", "yard_pbr")
NS = list(range(-17, 61))                                     # windows from 3.2 s before the switch to 10 s after
res = {}
for mesh in ("bunny", "dragon"):
    for seed in (97, 99):
        L, P = series("b1", mesh, "L", seed), series("b1", mesh, "P", seed)
        for arm in ("Loff6", "Pon6"):
            X = series("d154", mesh, arm, seed)
            if not X:
                continue
            key = f"{mesh}_{seed}_{arm}"
            res[key] = {}
            for m in MEAS:
                rows = []
                for n in NS:
                    l, p, x = per_window(L, m, n), per_window(P, m, n), per_window(X, m, n)
                    if None in (l, p, x) or p == 0:
                        rows.append(dict(n=n, t=6 + n / 6, L=l, P=p, X=x))
                        continue
                    lead_lp, lead_x = (p - l) / p, (p - x) / p
                    share = (1 - lead_x / lead_lp if arm == "Loff6" else lead_x / lead_lp) if lead_lp != 0 else None
                    rows.append(dict(n=n, t=6 + n / 6, L=l, P=p, X=x, lead_LP=lead_lp, lead_X=lead_x, share=share))
                post = [r for r in rows if r["n"] >= 1 and r.get("share") is not None]
                first = next((r["n"] for r in post if r["share"] >= 0.5), None)
                at = lambda k: next((r["share"] for r in rows if r["n"] == k and r.get("share") is not None), None)  # noqa: E731
                ref = L if arm == "Loff6" else P                # the run's own B1 counterpart before the switch
                pre = [(per_window(X, m, n), per_window(ref, m, n)) for n in range(-5, 1)]
                pre = [abs(x - r) / r for x, r in pre if x is not None and r]
                end_x = [v[m] for r, v in sorted(X.items()) if v.get(m) is not None]
                end_l = [v[m] for r, v in sorted(L.items()) if v.get(m) is not None]
                end_p = [v[m] for r, v in sorted(P.items()) if v.get(m) is not None]
                res[key][m] = dict(rows=rows, first_half=first, share10=at(10), share20=at(20),
                                   share_mean_11_20=float(np.mean([r["share"] for r in post if 11 <= r["n"] <= 20]))
                                   if any(11 <= r["n"] <= 20 for r in post) else None,
                                   lead_LP_at_switch=next((r.get("lead_LP") for r in rows if r["n"] == 0), None),
                                   replicate_spread=float(np.mean(pre)) if pre else None,
                                   end=dict(X=end_x[-1] if end_x else None, L=end_l[-1] if end_l else None,
                                            P=end_p[-1] if end_p else None))
# the pre-registered verdict per arm: both primary measures, >= 3 of 4 mesh-seeds
verdict = {}
for arm in ("Loff6", "Pon6"):
    keys = [k for k in res if k.endswith(arm)]
    ok = [k for k in keys if all(res[k].get(m, {}).get("first_half") is not None and res[k][m]["first_half"] <= 10
                                 and (res[k][m].get("share20") or -1) >= 0.5 for m in PRIMARY)]
    no = [k for k in keys if all((res[k].get(m, {}).get("share10") if res[k].get(m, {}).get("share10") is not None else 1) < 0.25
                                 for m in PRIMARY)]
    verdict[arm] = dict(n=len(keys), confirm=ok, refute=no,
                        result="confirmed" if len(ok) >= 3 else "refuted" if len(no) >= 3 else "partial" if keys else None)
json.dump(dict(series=res, verdict=verdict), open(out / "switch.json", "w"))
for k, d in res.items():
    for m in MEAS:
        v = d.get(m)
        if not v:
            continue
        f = lambda x: "nan" if x is None else f"{x:.3f}"  # noqa: E731
        print(f"{k:20s} {m:14s} lead_LP@6 {f(v['lead_LP_at_switch'])} first>=0.5 {v['first_half']} share10 {f(v['share10'])} "
              f"share20 {f(v['share20'])} mean11-20 {f(v['share_mean_11_20'])} replicate {f(v['replicate_spread'])} "
              f"end X {v['end']['X']} L {v['end']['L']} P {v['end']['P']}")
print(json.dumps(verdict, indent=1))
