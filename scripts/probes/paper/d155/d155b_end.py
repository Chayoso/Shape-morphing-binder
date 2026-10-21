"""d155b_end.py -- D155b: the delivered window's own terms (transport energy, silhouette, shading), the run length and
the windows after the ramp (or after the tag's u gate opens) against the yardstick and the display, per run."""
import json

G = "/data/relcfd/chayo/physmorph_v2/output/gpu/d155/gate"
R = {}
for line in open(f"{G}/gate_all.jsonl"):
    r = json.loads(line)
    R[r["name"]] = r


def last(n):
    J = json.load(open(f"{G}/{n}.json"))
    a = next(iter(J["arms"].values()))
    h = [x for x in a["history"] if x.get("frame_end") and not x.get("null_commit") and x["frame_end"] <= a["deliver_n"]]
    return h[-1], len(h), a["deliver_n"] / 240.0


for m in ("bunny", "dragon", "teapot", "armadilo", "homer"):
    for arm in ("V97", "V99", "V101", "A3L97", "A3GL97", "W97", "W99", "W101", "A3P97", "A3GP97"):
        n = f"{m}40k_{arm}"
        if n not in R:
            continue
        h, w, T = last(n)
        r = R[n]
        e = r["end"]
        ug = r.get("u_gate_half_t")
        start = r["T_e"] if r["switched"] else (ug or 0.0)
        post = sum(1 for t in r["curves"]["t"] if t > start + 1e-9)
        print(f"{n:18s} win {w:3d} T {T:5.2f} TE {h['transport_energy']:.3e} own sil {h['d_sil']:.3e} pbr "
              f"{h['d_pbr']:.3e} | yard {e.get('yard_sil', 0):.3e} {e.get('yard_pbr', 0):.3e} disp "
              f"{e.get('display', 0):.4f} thin {e.get('thin', 0):.3f} | u gate half at {ug} s, windows after "
              f"{start:.2f} s: {post}")
    print()
