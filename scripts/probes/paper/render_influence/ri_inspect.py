"""ri_inspect.py -- render influence (2026-10-22): what the archives hold (keys, frame counts, history fields)."""
import json
import numpy as np
G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
for p in [f"{G}/d145t_b1/bunny300k_L97_end1.npz", f"{G}/d145t_b7/bunny300k_RG97_frames12.npz"]:
    z = np.load(p, allow_pickle=True, mmap_mode=None)
    print(p, {k: (z[k].shape if hasattr(z[k], "shape") else type(z[k])) for k in z.files})
for m in ("bunny", "dragon"):
    for tag, p in (("L97", f"{G}/d145t_b1/{m}300k_L97"), ("P97", f"{G}/d145t_b1/{m}300k_P97"), ("RG97", f"{G}/d145t_b7/{m}300k_RG97"), ("T97", f"{G}/d159/{m}300k_T97")):
        js = json.load(open(p + ".json"))
        arm = next(iter(js["arms"].values()))
        hist = [h for h in arm["history"] if "animation" in h]
        h0 = hist[0]
        print(m, tag, "windows", len(hist), "keys", sorted(h0.keys())[:80])
        print("   ", {k: h0.get(k) for k in ("animation", "frame_start", "frame_end", "t0", "t1", "lambda", "g_share", "move")})
        print("    metrics", {k: arm["metrics"].get(k) for k in ("tgt_nn_spacing", "chamfer", "sil_iou")})
