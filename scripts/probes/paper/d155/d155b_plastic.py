"""d155b_plastic.py -- D155b: the plastic and the volume state of the delivered window (records): det Fp quantiles, the
tracked volume J, the smallest det F along the trajectory, the control's largest increment; tag against A3G."""
import json

O = "/data/relcfd/chayo/physmorph_v2/output/gpu/d155/gate"
for m in ("bunny", "dragon", "teapot", "armadilo", "homer"):
    print(m)
    for arm in ("V97", "V99", "V101", "A3GL97", "W97", "A3GP97"):
        a = next(iter(json.load(open(f"{O}/{m}40k_{arm}.json"))["arms"].values()))
        h = [x for x in a["history"] if x.get("frame_end") and not x.get("null_commit")
             and x["frame_end"] <= a["deliver_n"]][-1]
        print(f"  {arm:7s} detFp p01 {h.get('Jp_p01', 0):.3f} p99 {h.get('Jp_p99', 0):.3f} max {h.get('Jp_max', 0):.2f} | "
              f"J p01 {h.get('Jx_p01', 0):.3f} p99 {h.get('Jx_p99', 0):.3f} | Jmin traj {h.get('Jmin_traj', 0):.3f} | "
              f"dfc max {h.get('dfc_absmax', 0):.4f}")
