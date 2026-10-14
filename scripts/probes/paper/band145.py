"""band145.py ROWS_JSONL ARM_A ARM_B [MESH ...] -- D145: three-seed bands of tab145.py rows and the comparison of two arms.

Rows are grouped by mesh (the name's prefix before "300k_") and arm (the name's suffix without its seed digits, e.g.
L97 -> L). For each mesh and measure: ARM_A's and ARM_B's min / median / max over seeds, the relative change of A
against B (median of A / median of B - 1; for IoU the change of 1 - IoU), and whether the two three-run ranges
overlap (D145's reading: an arm differs from the other on a measure only when they do not overlap)."""
import json
import re
import sys

import numpy as np

rows = [json.loads(s) for s in open(sys.argv[1]) if s.startswith("{")]
A, B = sys.argv[2], sys.argv[3]
meshes = sys.argv[4:]
MEAS = [("yard_ext_sil", "low"), ("yard_ext_pbr", "low"), ("disp_front_1miou", "low"), ("disp_crop_1miou", "low"),
        ("disp_back_1miou", "low"), ("disp_front_diff", "low"), ("sil_own", "high"), ("chamfer_own", "low"),
        ("thin", "low"), ("kin_last", "low"), ("apart_w3_10_max", "low"), ("com_largest_a", "low"),
        ("ang_net_gross_mean", "low"), ("rotation_deg", "low"), ("windows", "info"), ("minutes", "info"),
        ("share_median", "info"), ("lam_first", "info")]
latest = {}
for r in rows:                                  # the latest row per run name wins
    latest[r["name"]] = r
groups = {}
for name, r in latest.items():
    m = re.match(r"(.+?)300k_([A-Za-z0-9]*?[A-Za-z])(\d+)$", name)
    if not m:
        continue
    groups.setdefault((m.group(1), m.group(2)), []).append(r)
out = []
for mesh in meshes or sorted({k[0] for k in groups}):
    ga, gb = groups.get((mesh, A), []), groups.get((mesh, B), [])
    print(f"== {mesh}: {A} n={len(ga)} ({', '.join(sorted(r['name'] for r in ga))}), {B} n={len(gb)}")
    for key, sense in MEAS:
        va = [r[key] for r in ga if r.get(key) is not None]
        vb = [r[key] for r in gb if r.get(key) is not None]
        if not va and not vb:
            continue
        rel = None
        if va and vb and sense != "info":
            ma, mb = float(np.median(va)), float(np.median(vb))
            if key == "sil_own":
                ma, mb = 1 - ma, 1 - mb
            rel = ma / mb - 1 if mb else None
        overlap = (bool(va) and bool(vb) and min(va) <= max(vb) and min(vb) <= max(va))
        f = lambda v: "-" if not v else f"{min(v):.4g} / {np.median(v):.4g} / {max(v):.4g}"
        print(f"  {key:20s} {A} {f(va):32s} {B} {f(vb):32s} "
              + ("" if rel is None else f"{A} vs {B} {rel * 100:+.1f} %") + ("" if not (va and vb) else
                                                                              ("  overlap" if overlap else "  APART")))
        out.append(dict(mesh=mesh, measure=key, A=A, B=B, a=va, b=vb, rel=rel, overlap=overlap))
json.dump(out, open(sys.argv[1].replace(".jsonl", f"_{A}_vs_{B}.json"), "w"))
