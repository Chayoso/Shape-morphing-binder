"""band145t.py ROWS_JSONL ARM_A ARM_B [MESH ...] [--seeds 97,99,101] -- D145 post-tag: band145.py's reading of two arms
with the post-tag measures (tab145t.py rows).

Rows are grouped by mesh (the name's prefix before "300k_") and arm (the name's suffix without its seed digits); for each
mesh and measure: ARM_A's and ARM_B's min / median / max over seeds, the relative change of A against B (median A /
median B - 1; for an IoU the change of 1 - IoU), and whether the ranges overlap (D145's reading: an arm differs from the
other on a measure only when they do not overlap). --seeds keeps only those seeds. Writes ROWS_A_vs_B.json beside."""
import json
import re
import sys

import numpy as np

args = sys.argv[1:]
seeds = None
if "--seeds" in args:
    i = args.index("--seeds")
    seeds = set(args[i + 1].split(","))
    del args[i:i + 2]
rows = [json.loads(s) for s in open(args[0]) if s.startswith("{")]
A, B = args[1], args[2]
meshes = args[3:]
IOU = {"sil_own", "sil_ind"}
MEAS = [("yard_ext_sil", "low"), ("yard_ext_pbr", "low"), ("disp_front_1miou", "low"), ("disp_crop_1miou", "low"),
        ("disp_back_1miou", "low"), ("sil_ind", "high"), ("chamfer_ind", "low"), ("sil_own", "high"),
        ("chamfer_own", "low"), ("thin", "low"), ("kin_last", "low"), ("kin_peak", "low"), ("apart_w3_10_max", "low"),
        ("vis_w3_10_side_blob", "low"), ("vis_run_side_blob", "low"), ("vis_run_blob", "low"),
        ("vis_end_side_blob", "low"), ("pc_end_pieces", "low"), ("pc_end_largest", "low"), ("pc_end_farthest_a", "low"),
        ("pc_run_largest", "low"), ("pc_run_farthest_a", "low"), ("pc_far_frames", "low"), ("pc_far_largest", "low"),
        ("pc_far_particles_max", "low"), ("com_max_fine_a", "low"), ("com_end_fine_a", "low"),
        ("rotation_deg", "low"), ("ang_net_gross_mean", "low"), ("rec_body_net_P", "low"),
        ("rec_spacing_net_P", "low"), ("rec_grid_net_P", "low"), ("rec_spacing_rot_deg", "low"),
        ("rec_body_rot_deg", "low"), ("Lstart_max", "low"), ("Lstart_median", "low"), ("Lgrid_sum", "low"),
        ("Lspace_sum", "low"), ("Lu_sum", "low"), ("move_last20_a", "low"), ("pulse_exc", "low"),
        ("pulse_net", "info"), ("pulse_ret", "low"), ("pulse_corr_d", "info"), ("pulse_corr_r", "info"),
        ("rev_3_7", "info"), ("rev_7_15", "info"), ("rev_15", "info"), ("windows", "info"),
        ("attempts", "info"), ("minutes", "info"), ("share_c_median", "info"), ("lam_c_first", "info"),
        ("gcos_w1_10", "info"), ("graw_cos_w1_10", "info")]
latest = {}
for r in rows:
    latest[r["name"]] = r
groups = {}
for name, r in latest.items():
    m = re.match(r"(.+?)300k_([A-Za-z0-9]*?[A-Za-z])(\d+)$", name)
    if not m or (seeds and m.group(3) not in seeds):
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
            if key in IOU:
                ma, mb = 1 - ma, 1 - mb
            rel = ma / mb - 1 if mb else None
        overlap = (bool(va) and bool(vb) and min(va) <= max(vb) and min(vb) <= max(va))
        f = lambda v: "-" if not v else f"{min(v):.4g} / {np.median(v):.4g} / {max(v):.4g}"
        print(f"  {key:20s} {A} {f(va):32s} {B} {f(vb):32s} "
              + ("" if rel is None else f"{A} vs {B} {rel * 100:+.1f} %") + ("" if not (va and vb) else
                                                                              ("  overlap" if overlap else "  APART")))
        out.append(dict(mesh=mesh, measure=key, A=A, B=B, a=va, b=vb, rel=rel, overlap=overlap,
                        a_names=sorted(r["name"] for r in ga if r.get(key) is not None),
                        b_names=sorted(r["name"] for r in gb if r.get(key) is not None)))
suffix = "" if not seeds else "_s" + "_".join(sorted(seeds))
json.dump(out, open(args[0].replace(".jsonl", f"_{A}_vs_{B}{suffix}.json"), "w"))
