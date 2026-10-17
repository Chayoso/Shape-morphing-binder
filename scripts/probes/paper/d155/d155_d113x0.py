"""d155_d113x0.py -- D155: D113's D105 courses re-read with X0 = the first committed record (as D155 on the tag),
so the tag and D105 compare on one convention; also D113's own source-X0 numbers for the size of the convention."""
import json
import sys

import numpy as np

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d155")
from d155_records import area, crossing, progress  # noqa: E402

G = "/data/relcfd/chayo/physmorph_v2/output/gpu/d113"
for m in ("bunny300k", "dragon300k"):
    for r in json.load(open(f"{G}/course_{m}_D105.json"))["runs"]:
        c = r["curves"]
        t = np.array(c["t"])
        for lab, sl in (("source X0", slice(0, None)), ("first record X0", slice(1, None))):
            tt, P, R = t[sl], np.array(c["P"])[sl], np.array(c["R"])[sl]
            o = {}
            for sc in ("lin", "log"):
                cP, cR = progress(P, sc == "log"), progress(R, sc == "log")
                sg, ab = area(tt, cP, cR)
                o[sc] = (crossing(tt, cP, .9), crossing(tt, cR, .9), ab, sg)
            print(f"{m} {r['label']:>6} {lab:>16}: lin P90 {o['lin'][0]:.3f} R90 {o['lin'][1]:.3f} A {o['lin'][2]:.3f} | "
                  f"log P90 {o['log'][0]:.3f} R90 {o['log'][1]:.3f} A {o['log'][2]:.3f} ({o['log'][3]:+.3f})")
