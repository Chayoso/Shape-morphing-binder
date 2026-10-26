import glob, json, os
import numpy as np
G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
for n in ("d163/runs300/plq300k_H_97", "d163/runs300/plq300k_T_97"):
    js = json.load(open(f"{G}/{n}.json")); p = js["provenance"]; arm = list(js["arms"].values())[0]
    print(n, {k: p.get(k) for k in ("tgt", "render_tgt", "n", "seed", "lambda_ema", "render_weight_scale", "render_weight_ref",
                                    "appearance", "proximity_berth", "placement_at_cell")}, "windows", len(arm["history"]),
          "deliver", arm["deliver_n"])
    z = np.load(f"{G}/{n}_frames12.npz", allow_pickle=True); print("  keys", list(z.keys()), "frames", z["frames"].shape)
for pat in ("d161/runs40/*_lh_frames12.npz", "d161/runs40/*twin_9?_frames12.npz", "d160/runs40/bunny40k_band_[DP]9?_frames12.npz",
            "d162/runs40/coin40k_F_9?_frames12.npz", "d162/runs40/bunny40k_F_9?_frames12.npz", "d162/runs40/bunny40k_band_F_9?_frames12.npz",
            "d162b/runs40/*_S_9?_frames12.npz"):
    fs = sorted(glob.glob(f"{G}/{pat}"))
    print(pat, len(fs), [os.path.basename(f) for f in fs][:6])
    if fs:
        z = np.load(fs[0], allow_pickle=True); print("   frames", z["frames"].shape, "raws step", np.diff(z["raws"][:4]))
for f in sorted(glob.glob(f"{G}/d161/runs40/*_lh.json"))[:1]:
    p = json.load(open(f))["provenance"]; print(f, {k: p.get(k) for k in ("lambda_ema", "tgt", "render_tgt")})
