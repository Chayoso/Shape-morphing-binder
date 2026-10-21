"""d155b_jcourse.py -- D155b: the course of the tracked volume J (the records' p01 / p99 / max at every committed window's
end) and the rest volume det Fp (p01 / p99), the tag against A3G, render arm; times in s, and for A3G the ramp's s."""
import json

O = "/data/relcfd/chayo/physmorph_v2/output/gpu/d155/gate"
for m in ("dragon", "homer", "bunny"):
    print(m)
    for arm in ("V97", "A3GL97", "A3L97"):
        a = next(iter(json.load(open(f"{O}/{m}40k_{arm}.json"))["arms"].values()))
        acc = [h for h in a["history"] if h.get("frame_end") and not h.get("null_commit")]
        rows = []
        for t_q in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 99):
            h = min(acc, key=lambda r: abs(r["frame_end"] / 240 - t_q))
            rows.append(f"{h['frame_end'] / 240:.1f}s"
                        + (f"(s={h['morph_s']:.2f})" if h.get("morph_s") is not None and h["morph_s"] < 1 else "")
                        + f" J {h.get('Jx_p01', 0):.3f}-{h.get('Jx_p99', 0):.3f} Fp {h.get('Jp_p01', 0):.3f}-{h.get('Jp_p99', 0):.3f}")
        print(f"  {arm:7s} " + " | ".join(dict.fromkeys(rows)))
