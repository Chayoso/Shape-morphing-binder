"""d155b_tail.py MESH -- D155b forensic (CPU, records): after the arrival (the tag: u's gate over half the layer; A3G: the
ramp's end, where its gate opens), per window: the transport energy, the own silhouette and shading, u's gate, the
merit's gain, the stop; and how the run stopped (patience or reject streak), the delivered window, its time after the
arrival. Seed 97, render arm and twin."""
import json
import sys

G = "/data/relcfd/chayo/physmorph_v2/output/gpu/d155/gate"
mesh = sys.argv[1]
for arm in ("V97", "A3GL97", "W97", "A3GP97"):
    n = f"{mesh}40k_{arm}"
    a = next(iter(json.load(open(f"{G}/{n}.json"))["arms"].values()))
    H = a["history"]
    dn = a["deliver_n"]
    acc = [h for h in H if h.get("frame_end") and not h.get("null_commit")]
    t_arr = next((h["frame_end"] / 240 for h in acc if (h.get("u_gate") or 0) >= 0.5), None)
    if "A3G" in arm:
        t_arr = max(t_arr or 0, next(h["frame_end"] / 240 for h in acc if (h.get("morph_s") or 0) >= 1.0))
    print(f"== {n}: arrival {t_arr:.2f} s, delivered {dn / 240:.2f} s ({dn / 240 - t_arr:.2f} s after), windows "
          f"{len(acc)}, attempts {len(H)}, rejected {sum(1 for h in H if h.get('outer_rejected'))}, null "
          f"{sum(1 for h in H if h.get('null_commit') and not h.get('outer_rejected'))}, last: "
          f"stale {H[-1].get('stale')} streak {H[-1].get('reject_streak')}")
    for h in acc:
        t = h["frame_end"] / 240
        if t < t_arr - 1e-9:
            continue
        k = round((t - t_arr) * 6)
        if k in (0, 1, 2, 4, 6, 9, 12, 18, 24, 30, 36, 42, 48) or h is acc[-1]:
            print(f"  +{t - t_arr:5.2f} s  TE {h['transport_energy']:.3e} sil {h['d_sil']:.3e} pbr {h['d_pbr']:.3e} "
                  f"u {h.get('u_gate') or 0:.2f} gain {h.get('outer_gain') or 0:+.4f} improved {h.get('improved')} "
                  f"kin {h['kin']:.2e} alpha {h.get('alpha_last') or 0:.4f}")
