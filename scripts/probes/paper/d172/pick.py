"""pick.py MODE ... -- D172's small decisions for post172.sh (CPU, no reading of its own):
  gal VERDICT                 the arms that hold 6/6 at seed 97
  scales VERDICT              the best 40k scale arm (share >= 0.40 and >= twin + 0.10 at both seeds; shape ok preferred)
  job191 "ARMS"               the seed-99 gallery job script for those arms
  job190 ARM VARIANT          the 300k scale job script (the twin and ARM)
  list VERDICT                one line per read run: KEY STATUS CONTENT_OK FULL TWIN
  stop RUN_JSON               the run's last optimised frame (raw = frame_end - 1)"""
import json
import sys

mode = sys.argv[1]
FL = {"Hu": "$HU", "Hf": "$HF", "Rx": "$RX"}
if mode == "gal":
    V = json.load(open(sys.argv[2]))["arms"]
    print(" ".join(a for a, v in V.items() if v.get("held") == v.get("n") == 6))
elif mode == "scales":
    V = json.load(open(sys.argv[2]))["arms"]
    best = None
    for a in ("Hu", "Hf", "Rx"):
        v = [V.get(f"scl40_{a}_{s}", {}) for s in (97, 99)]
        if all(q.get("share_end") is not None and q["share_end"] >= 0.40 and (q.get("above_twin") or 0) >= 0.10 for q in v):
            m = min(q["share_end"] for q in v) + (0.5 if all(q.get("shape_ok") for q in v) else 0.0)
            if best is None or m > best[1]:
                best = (a, m)
    print(best[0] if best else "")
elif mode == "job191":
    arms = sys.argv[2].split()
    L = ["#!/bin/bash", f"# 191_d172_gal99.sh -- queue job (agent D172, armed by post172.sh): seed 99 of the 40k gallery for {' '.join(arms)}",
         "# (6/6 at seed 97).", 'J=191_gal99; TITLE="D172: gallery seed 99"', "source /data/relcfd/chayo/physmorph_v2/tmp/d172/job_common.sh",
         'log "start"; PAR=4']
    for a in arms:
        L.append(f"for m in dragon armadilo homer bunny bob teapot; do runp gal_${{m}}_{a}_99 --tgt assets/$m.obj --n 40000 --seed 99 {FL[a]}; done")
    L.append('wait; log "done"')
    print("\n".join(L))
elif mode == "job190":
    arm, var = sys.argv[2], sys.argv[3]
    print("\n".join([
        "#!/bin/bash", f"# 190_d172_scales300.sh -- queue job (agent D172, armed by post172.sh): the stylized dragon scales (variant {var}) at",
        f"# 300k, seed 97: the twin and {arm} (the best 40k arm).", 'J=190_scales300; TITLE="D172: stylized dragon scales, 300k"',
        "source /data/relcfd/chayo/physmorph_v2/tmp/d172/job_common.sh", 'log "start"; PAR=2',
        f'S="--tgt $G/d170/assets/d170smooth_dragon.obj --render_tgt $A/d172S{var}300_dragon.obj --n 300000 --seed 97"',
        "runp scl300k_T_97 $S $TW", f"runp scl300k_{arm}_97 $S {FL[arm]}", 'wait; log "done"']))
elif mode == "list":
    for k, v in json.load(open(sys.argv[2]))["arms"].items():
        if v.get("status") not in (None, "INCOMPLETE") and "full" in v:
            print(k, v["status"], int(bool(v.get("content_ok"))), v["full"], v["twin"])
elif mode == "stop":
    a = next(iter(json.load(open(sys.argv[2]))["arms"].values()))
    print(max(h["frame_end"] for h in a["history"] if h.get("frame_end") and not h.get("settle")) - 1)
