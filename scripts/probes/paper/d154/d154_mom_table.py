"""d154_mom_table.py OUT_JSON -- D154 (2): momentum of the switched runs (d154_momentum.py summaries) beside B1's L and P
of the same seed (D145's m_*.log summary lines, momentum_probe.py's definitions): the centre of mass's largest and end
displacement (pitches), the net over gross angular move (mean, last 20), the net rotation over the run (deg)."""
import glob
import json
import re
import sys
from pathlib import Path

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
rows = []
for f in sorted(glob.glob(str(G / "d154/read/sw_mom_*.jsonl"))):
    name = Path(f).stem.replace("sw_mom_", "")
    for s in open(f):
        r = json.loads(s)
        if r.get("summary"):
            rows.append(dict(run=name, src="d154", com_far_a=r["com_far_a"], com_end_a=r["com_end_a"],
                             ang_mean=r["ang_mean"], ang_last20=r["ang_last20"], rot_deg=r["rot_deg"]))
num = r"([0-9.eE+-]+)"
for mesh in ("bunny", "dragon"):
    for seed in (97, 99):
        for arm in "LP":
            p = G / f"d145t_b1/m_{mesh}300k_{arm}{seed}.log"
            if not p.exists():
                continue
            line = [s for s in open(p) if "kept frames |" in s][-1]
            rows.append(dict(run=f"{mesh}300k_{arm}{seed}", src="B1",
                             com_far_a=float(re.search(r"largest displacement " + num, line).group(1)),
                             com_end_a=float(re.search(r"at the end " + num, line).group(1)),
                             ang_mean=float(re.search(r"angular move: largest [0-9.eE+-]+, mean " + num, line).group(1)),
                             ang_last20=float(re.search(r"angular move: .*?last 20 " + num, line).group(1)),
                             rot_deg=float(re.search(r"net rotation over the run " + num, line).group(1))))
json.dump(rows, open(sys.argv[1], "w"), indent=1)
for r in rows:
    print(f"{r['run']:22s} {r['src']:5s} com far {r['com_far_a']:.2e} end {r['com_end_a']:.2e} a | ang mean {r['ang_mean']:.2e} "
          f"last20 {r['ang_last20']:.2e} | rotation {r['rot_deg']:.4f} deg")
