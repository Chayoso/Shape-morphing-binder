"""d157_summary.py TABLE_JSON -- D157: a compact digest of table.json for the write-up."""
import json
import sys

import numpy as np

T = json.load(open(sys.argv[1]))
M = T["metrics"]
KEYS = ["run/sil_iou", "d154/pooled_iou_train1024_raw", "sil/zb/raw/1024/held/iou_err", "sil/zb/cor/1024/held/iou_err",
        "sil/zb/cor/1024/held/biou2pct_err", "sil/zb/raw/1024/held/biou2pct_err", "sil/zb/cor/1024/held/biou1a_err",
        "sil/zb/cor/1024/held/bfdavis_err", "sil/zb/cor/1024/held/bf05a_err", "sil/zb/cor/1024/held/bf1a_err",
        "sil/zb/raw/1024/held/bf1a_err", "sil/zb/cor/1024/held/chamfer_a", "sil/zb/raw/1024/held/chamfer_a",
        "sil/zb/cor/1024/held/hd95_a", "sil/zb/cor/1024/held/hd_a", "sil/zb/cor/1024/train/biou2pct_err",
        "sil/zb/cor/1024/far/biou2pct_err", "sil/zb/cor/256/held/biou2pct_err", "sil/zb/cor/512/held/biou2pct_err",
        "d3/zb/cor/f0.25", "d3/zb/cor/f0.5", "d3/zb/cor/f1", "d3/zb/cor/f2", "d3/zb/raw/f1", "d3/zb/raw/f2",
        "d3/zb/cor/nc", "d3/zb/raw/nc", "d3/zb/cor/chamfer_l1_a", "d3/zb/raw/chamfer_l1_a", "d3/zb/cor/thin_rec1",
        "d3/zb/raw/thin_rec1", "run/thin_uncovered", "d3/zbs/cor/f0.5", "d3/zbs/cor/nc",
        "img/cor/shaded/held/mse", "img/cor/shaded/held/ssim", "img/cor/shaded/held/lpips",
        "img/cor/normal/held/mse", "img/cor/normal/held/ssim", "img/cor/normal/held/lpips",
        "img/raw/shaded/held/mse", "img/raw/shaded/held/ssim", "img/raw/shaded/held/lpips",
        "img/raw/normal/held/mse", "img/raw/normal/held/ssim", "img/raw/normal/held/lpips",
        "img/cor/shaded/far/lpips", "img/cor/normal/far/lpips", "img/cor/angle/held/deg", "img/raw/angle/held/deg"]
for N in ("300k", "40k"):
    print(f"===== {N}")
    for k in KEYS:
        x = M.get(f"{N}:{k}")
        if not x:
            continue
        pm = list(x["per_mesh_r"].values())
        s = (f"{k:38s} arm {x['arm_med']:.4g} twin {x['twin_med']:.4g} own {x['own_med']:.4g} | r {100 * x['r_med']:+.1f} % "
             f"[{100 * x['r_q25']:+.1f},{100 * x['r_q75']:+.1f}] meshes {100 * min(pm):+.1f}..{100 * max(pm):+.1f} "
             f"(+{x['mesh_pos']}/-{x['mesh_neg']}) pairs {x['arm_better']}/{x['pairs']} p {x['wilcoxon_p']:.1e}"
             f"{' SEES ' + x['direction'] if x['sees'] else ''}")
        if "psnr_gain_db_med" in x:
            s += (f" | PSNR arm {x['psnr_arm_med']:.2f} twin {x['psnr_twin_med']:.2f} dB, gain {x['psnr_gain_db_med']:+.2f} "
                  f"[{x['psnr_gain_db_q25']:+.2f},{x['psnr_gain_db_q75']:+.2f}]")
        print(s)
    dec = [v for kk, v in M.items() if v["n"] == N and "holm_p_decision" in v]
    print(f"decision set: {sum(v['sees'] for v in dec)} of {len(dec)} see it:",
          ", ".join(v["metric"].split("/", 1)[1] for v in dec if v["sees"]))
    print("not:", ", ".join(v["metric"].split("/", 1)[1] for v in dec if not v["sees"]))
    allm = [v for v in M.values() if v["n"] == N]
    print(f"all metrics: {sum(v['sees'] for v in allm)} of {len(allm)} see it; losses:",
          [v["metric"] for v in allm if v["sees"] and v["direction"] == "loss"])
for N, R in T["region"].items():
    print(f"===== (4a) {N}")
    for nm in ("all", "bin0", "bin1", "bin2", "bin3", "bin4", "bin5", "rim", "headon", "crevice", "rim_minus_headon"):
        x = R.get(nm)
        if x is None:
            continue
        if nm == "rim_minus_headon":
            print(f"  {nm:16s} med {x['med']:+.4f} pos {x['pos']}/{x['pairs']} p {x['p']:.1e}")
        else:
            print(f"  {nm:16s} share {x['share_med']:.3f} arm {x['arm_med']:.3f} twin {x['twin_med']:.3f} own {x['own_med']:.3f} "
                  f"d {x['d_med']:+.4f} [{x['d_q25']:+.4f},{x['d_q75']:+.4f}] ({x['d_px_med']:+.4f} render px) "
                  f"outward {x['outward']}/{x['pairs']} p {x['p']:.1e}")
print("===== (4b)")
B = T["bias"]
for N in ("300k", "40k"):
    for res_kind in ("run", "half", "double"):
        bb = [b for b in B if b["n"] == N and ((b["res"] == b["run_res"]) if res_kind == "run" else
                                               (b["res"] == b["run_res"] // 2 if res_kind == "half" else b["res"] == 2 * b["run_res"]))]
        if not bb:
            continue
        arm, twin = ("L", "P") if N == "300k" else ("V", "W")
        f = lambda q: np.array([b[q] for b in bb if b.get(q) is not None], float)
        obs = f("obs_arm_minus_twin")
        print(f"{N} {res_kind} res ({len(bb)} mesh-seeds, meshes {sorted(set(b['mesh'] for b in bb))}): "
              f"bias own {np.median(f('own_bias')):+.3f} [{f('own_bias').min():+.3f},{f('own_bias').max():+.3f}] "
              f"arm {np.median(f(arm + '_bias')):+.3f} [{f(arm + '_bias').min():+.3f},{f(arm + '_bias').max():+.3f}] "
              f"twin {np.median(f(twin + '_bias')):+.3f} [{f(twin + '_bias').min():+.3f},{f(twin + '_bias').max():+.3f}] | "
              f"twin argmin11 {np.median(f(twin + '_argmin11')):+.3f} argmin21 {np.median(f(twin + '_argmin21')):+.3f} "
              f"[{f(twin + '_argmin21').min():+.3f},{f(twin + '_argmin21').max():+.3f}] | arm argmin21 "
              f"{np.median(f(arm + '_argmin21')):+.3f} argmin11 {np.median(f(arm + '_argmin11')):+.3f} | observed arm-twin "
              f"{np.median(obs) if len(obs) else float('nan'):+.4f} | edge mean {np.median(f('edge_mean_a')):.3f} single "
              f"{np.median(f('edge_single_a')):.3f} | px {np.median(f('px_a')):.3f} pitch | at_edge "
              f"{int(sum(f(s + '_at_edge').sum() for s in ('own', arm, twin)))}")
