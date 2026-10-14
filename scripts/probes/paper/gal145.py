"""gal145.py [ARM_A ARM_B] -- D145 B3: the 40k gallery read from output/gpu/d145_b3 (gallery_ind.py's ind.log, the runs'
JSON and d120_apart.py's logs; CPU only). Default arms V (render) and W (twin); seeds 97 / 99 / 101.
Per mesh: each arm's independent-sample silhouette IoU at each seed (min / median / max), the sampling floor (own
target sample against the independent one), V - W per seed and its three-seed mean, whether all seeds agree in sign,
the gain as a share of the twin's gap to the floor ((V - W) / (floor - W)), chamfer, the last kinetic record, windows,
apart in windows 3-10. Summary: meshes with the three-seed mean V - W > 0, meshes where all seeds agree, the paired
Wilcoxon signed-rank test over the meshes' three-seed means (scipy), the median share of the floor gap.
For an ablation arm (one seed, 97): ARM_A ARM_B = e.g. U V reads U against V's three-seed range."""
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

D = Path("/data/relcfd/chayo/physmorph_v2/output/gpu/d145_b3")
A, Bm = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("V", "W")
SEEDS = (97, 99, 101)
MESHES = "A armadilo beast bimba bob bunny C cheburashka cow dragon fandisk heart homer maxplanck nefertiti ogre spot teapot V".split()
ind = {}
for s in open(D / "ind.log"):
    if s.startswith("{"):
        r = json.loads(s)
        ind[r["tag"]] = r


def extra(tag):
    out = {}
    try:
        arm = json.loads((D / f"{tag}.json").read_text())["arms"]["render_full_dt_iso_nn"]
        kin = [h.get("kin") for h in arm["history"] if h.get("kin") is not None]
        out.update(kin_last=kin[-1] if kin else None, windows=sum(1 for h in arm["history"] if h.get("outer_accepted")))
    except (OSError, KeyError, ValueError):
        pass
    try:
        m = re.search(r"windows 3-10: apart max (\d+)", (D / f"apart_{tag}.log").read_text())
        out["apart_w3_10"] = int(m.group(1)) if m else None
    except OSError:
        pass
    return out


rows, means, shares, agree = [], [], [], 0
for mesh in MESHES:
    a = {s: ind.get(f"{mesh}40k_{A}{s}") for s in SEEDS}
    b = {s: ind.get(f"{mesh}40k_{Bm}{s}") for s in SEEDS}
    av = [a[s]["ind"]["sil"] for s in SEEDS if a[s]]
    bv = [b[s]["ind"]["sil"] for s in SEEDS if b[s]]
    floor = np.mean([r["floor"]["sil"] for r in list(a.values()) + list(b.values()) if r]) if (av or bv) else None
    row = dict(mesh=mesh, A=av, B=bv, floor=floor)
    paired = [a[s]["ind"]["sil"] - b[s]["ind"]["sil"] for s in SEEDS if a[s] and b[s]]
    if paired:
        row["diff"] = paired
        row["diff_mean"] = float(np.mean(paired))
        row["agree"] = all(d > 0 for d in paired) or all(d < 0 for d in paired)
        if len(paired) == 3:
            means.append(row["diff_mean"])
            agree += row["agree"] and row["diff_mean"] > 0
            if floor is not None and floor - np.mean(bv) > 0:
                shares.append(row["diff_mean"] / (floor - np.mean(bv)))
                row["floor_gap_share"] = shares[-1]
    elif av and len(bv) == 0 and len(av) == 1:
        pass
    row["chamfer_A"] = [a[s]["ind"]["chamfer"] for s in SEEDS if a[s]]
    row["extra_A"] = {s: extra(f"{mesh}40k_{A}{s}") for s in SEEDS if a[s]}
    row["extra_B"] = {s: extra(f"{mesh}40k_{Bm}{s}") for s in SEEDS if b[s]}
    rows.append(row)
    f = lambda v: "-" if not v else f"{min(v):.4f}/{np.median(v):.4f}/{max(v):.4f}"
    print(f"{mesh:12s} {A} {f(av)}  {Bm} {f(bv)}  floor {floor if floor is None else round(floor, 4)}  "
          f"diff {[round(d, 4) for d in paired]}  share {row.get('floor_gap_share', float('nan')):.2f}", flush=True)
print(f"meshes with all three seeds: {len(means)}; mean {A}-{Bm} > 0 on {sum(m > 0 for m in means)}; "
      f"all seeds agree positive on {agree}; median floor-gap share {np.median(shares) if shares else float('nan'):.3f}")
if len(means) >= 6:
    print(f"Wilcoxon (paired, two-sided) over the meshes' three-seed means: {wilcoxon(means)}")
(D / f"gal_{A}_{Bm}.json").write_text(json.dumps(rows, default=float))
