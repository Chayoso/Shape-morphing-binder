"""d106_rows.py — D106's table: per mesh of the 40k gallery, the end state's silhouette IoU against an independent 40k
sample (gallery_ind.py) of D98's physics-only twin (E) and render arm (D) (D100) and of D105's (W, V), the differences
the criteria read (V − W, D − E, V − D, W − E), and from D106's records the angular momentum at the window starts (end,
largest, mean; V against W). The single-run spread: D100b's second runs (V_D2, C_D2) against the first."""
import json
from collections import defaultdict
import numpy as np

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
sil = defaultdict(dict)
for d in ("d100", "d106"):
    for line in open(f"{G}/{d}/ind.log"):
        if line.strip():
            r = json.loads(line)
            mesh, arm = r["tag"].rsplit("_", 1)
            sil[mesh][arm] = r["ind"]["sil"]


def angmom(tag):
    try:
        h = [r for r in json.load(open(f"{G}/d106/{tag}.json"))["arms"]["render_full_dt_iso_nn"]["history"]
             if r.get("frame_end") and not r.get("null_commit") and r.get("L_start") is not None]
    except FileNotFoundError:
        return None
    L = [float(np.linalg.norm(r["L_start"])) for r in h]
    return (L[-1], max(L), float(np.mean(L))) if L else None


print("spread (second run - first):", {m: round(sil[m].get("D2", np.nan) - sil[m].get("D", np.nan), 4) for m in ("V", "C")})
print(f"{'mesh':12s} {'E':>7s} {'D':>7s} {'W':>7s} {'V':>7s} | {'V-W':>7s} {'D-E':>7s} {'V-D':>7s} {'W-E':>7s} | angular momentum V/W - 1: end, largest, mean")
n = dict(vw=0, de=0, vd_ok=0, we_ok=0, vd_bad=0, we_bad=0, l_ok=0, l_n=0, meshes=0)
for mesh in sorted(sil):
    s = sil[mesh]
    if not all(k in s for k in "DEVW"):
        continue
    n["meshes"] += 1
    vw, de, vd, we = s["V"] - s["W"], s["D"] - s["E"], s["V"] - s["D"], s["W"] - s["E"]
    n["vw"] += vw > 0; n["de"] += de > 0
    n["vd_ok"] += vd > -.002; n["we_ok"] += we > -.002; n["vd_bad"] += vd < -.01; n["we_bad"] += we < -.01
    LV, LW = angmom(f"{mesh}_V"), angmom(f"{mesh}_W")
    lrel = "" if LV is None or LW is None else ", ".join(f"{100 * (v / w - 1):+.0f} %" for v, w in zip(LV, LW))
    if LV is not None and LW is not None:
        n["l_n"] += 1; n["l_ok"] += LV[2] <= LW[2]
    print(f"{mesh:12s} {s['E']:7.4f} {s['D']:7.4f} {s['W']:7.4f} {s['V']:7.4f} | {vw:+7.4f} {de:+7.4f} {vd:+7.4f} {we:+7.4f} | {lrel}")
print(f"meshes {n['meshes']}: V ahead of W on {n['vw']}, D ahead of E on {n['de']}; V not behind D by more than 0.002 on "
      f"{n['vd_ok']} (more than 0.01 on {n['vd_bad']}); W not behind E so on {n['we_ok']} ({n['we_bad']}); "
      f"V's mean angular momentum not above W's on {n['l_ok']} of {n['l_n']}")
