"""clean.py RUNS_DIR CUT_TXT TAG[,TAG...] -- D147: the droplet's reading as pre-registered (docs/experiments.md, D147,
the attributions): a run is clean when no piece apart from its body (apart.py's components, joined within two sample
spacings; RUNS_DIR/apart_<tag>.txt over 9-16 s) lies beyond 3 spacings of its body in any kept frame before the cut
(cut.py's rule on that run, from CUT_TXT). Prints per run the frames with a far piece before the cut, the largest far
piece (particles), its time range, the largest distance, and CLEAN or NOT. CPU."""
import json
import sys
from collections import defaultdict
from pathlib import Path

runs, cuts = Path(sys.argv[1]), {}
for ln in open(sys.argv[2]):
    if ln.startswith("{"):
        j = json.loads(ln)
        cuts[j["tag"]] = j["t_cut"]
for tag in sys.argv[3].split(","):
    far = defaultdict(int)
    dmax = 0.0
    for ln in open(runs / f"apart_{tag}.txt"):
        if not ln.startswith('{"t"'):
            continue
        r = json.loads(ln)
        if r["t"] <= cuts[tag] and r["to_body_sp"] > 3.0:
            far[(r["t"], r["body"])] += r["n"]
            dmax = max(dmax, r["to_body_sp"])
    ts = sorted(t for t, _ in far)
    print(json.dumps({"tag": tag, "t_cut": cuts[tag], "far_frames": len(far), "largest": max(far.values(), default=0),
                      "from": ts[0] if ts else None, "to": ts[-1] if ts else None, "max_sp": round(dmax, 2),
                      "clean": not far}))
