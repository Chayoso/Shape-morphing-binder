"""table.py RUNS_DIR TAG[,TAG...] TIMELINE_TXT PROGRESS_TXT JUDGE_TXT CUT_TXT CLEAN_TXT -- D147's final table (the
re-freeze's 300k scene, pre-registered 2026-10-15): per run the judge's verdict and items (progress at the first contact,
parted in the air, landed, end error, the early rise), the tumble, the contact and the parting, the cut, the material
beyond 3 spacings of its body before the cut (apart.py's pieces in RUNS_DIR/apart_<tag>.txt: every kept frame with
such a piece, its size and distance), the end metrics per body, and the render's share (lambda and g_share over the
morph's windows, t <= morph_time, and over the run; the windows with a zero-control commit). Prints one JSON line per
run, then a tab-separated table. Read-only, CPU."""
import json
import sys
from pathlib import Path

import numpy as np


def lines(path):
    out = {}
    for ln in open(path):
        ln = ln.strip()
        if ln.startswith("{") and '"tag"' in ln:
            try:
                j = json.loads(ln)
            except json.JSONDecodeError:
                continue
            out.setdefault(j["tag"], {}).update(j)
    return out


runs, tags = Path(sys.argv[1]), sys.argv[2].split(",")
tl, pg, jd, ct, cl = (lines(p) for p in sys.argv[3:8])
rows = []
for tag in tags:
    j = json.loads((runs / f"{tag}.json").read_text())
    h = [r for r in j["history"] if "bodies" in r]
    t_m = float(j["scene"]["spec"].get("morph_time", 0.0))
    t_cut = ct[tag]["t_cut"]
    far = []
    ap = runs / f"apart_{tag}.txt"
    if ap.exists():
        for ln in open(ap):
            if ln.startswith('{"t"'):
                r = json.loads(ln)
                if r["t"] <= t_cut + 1e-9 and r["to_body_sp"] > 3.0:
                    far.append((r["t"], r["body"], r["n"], r["to_body_sp"]))

    def mean(key, sel):
        v = [r[key] for r in sel if r.get(key) is not None]
        return round(float(np.mean(v)), 4) if v else None

    morph = [r for r in h if r["t"] <= t_m + 1e-9]
    out = {"tag": tag, "pass": jd[tag]["pass"], "progress": pg[tag]["progress_mean"],
           "progress_bodies": pg[tag]["progress_at_contact"], "t_first_contact": pg[tag]["t_first_contact"],
           "early_ratio": jd[tag]["early_ratio"], "end_error": round(jd[tag]["end_error"], 4),
           "tumble_deg": [round(t, 1) if t is not None else None for t in tl[tag]["tumble_deg"]],
           "landing_s": tl[tag]["landing_s"], "t_contact": tl[tag]["t_contact"], "t_part": tl[tag]["t_part"],
           "parted_in_air": tl[tag]["parted_in_air"], "t_cut": t_cut, "t_cut_spacing": ct[tag]["t_cut_spacing"],
           "t_cut_impulse": ct[tag]["t_cut_impulse"], "clean": cl[tag]["clean"] if tag in cl else None,
           "far_frames": len({f[0] for f in far}), "far_particle_frames": int(sum(f[2] for f in far)),
           "far_largest": max((f[2] for f in far), default=0), "far_max_sp": max((f[3] for f in far), default=0.0),
           "far": far, "metrics": [{k: round(v, 4) for k, v in m.items()} for m in j["metrics"]],
           "lambda_morph": mean("lambda", morph), "lambda_all": mean("lambda", h),
           "g_share_morph": mean("g_share", morph), "g_share_all": mean("g_share", h),
           "zero_commits": sum(1 for r in h if r.get("zero_commit")), "windows": len(h),
           "minutes": round(j["seconds"] / 60, 1)}
    rows.append(out)
    print(json.dumps(out), flush=True)
cols = ("tag", "pass", "progress", "end_error", "tumble_deg", "t_contact", "t_part", "landing_s", "t_cut", "clean",
        "far_frames", "far_particle_frames", "far_largest", "far_max_sp", "lambda_morph", "g_share_morph", "g_share_all",
        "zero_commits", "minutes")
print("\t".join(cols))
for r in rows:
    print("\t".join(str(r[c]) for c in cols))
