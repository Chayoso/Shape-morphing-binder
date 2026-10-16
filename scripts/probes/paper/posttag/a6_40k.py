"""a6_40k.py [OUT_MD] -- D145 post-tag, amendment 6 at 40k: each ablation arm (NRD, NUC, NSS; seed 97, d145t_a6) against
the tag's B3 render arm V (d145t_b3, seeds 97 / 99 / 101; CPU only, gallery_ind.py's ind.log and the runs' own files).

Per arm and mesh: the independent-sample silhouette IoU, V's min / median / max, where the arm lies (below / inside /
above V's three-run range: D145's reading of a one-run arm), arm - V97 (the pre-registered pairing) and arm - V median;
beside it the chamfer (independent), apart particles at windows 3-10 (d120_apart.py), visible side blobs at windows
3-10 and over the run (visible_apart.py), the last kinetic record and windows, each against V97. Summary per arm: the
counts below / inside / above, the median of arm - V median, the meshes where the arm's apart / side blobs exceed V97's."""
import json
import re
import sys
from pathlib import Path

import numpy as np

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
DA, DV = G / "d145t_a6", G / "d145t_b3"
SEEDS = (97, 99, 101)
MESHES = "A armadilo beast bimba bob bunny C cheburashka cow dragon fandisk heart homer maxplanck nefertiti ogre spot teapot V".split()


def ind(d):
    out = {}
    try:
        for s in open(d / "ind.log"):
            if s.startswith("{"):
                r = json.loads(s)
                out[r["tag"]] = r
    except OSError:
        pass
    return out


def extra(d, tag):
    out = {}
    try:
        arm = json.loads((d / f"{tag}.json").read_text())["arms"]["render_full_dt_iso_nn"]
        kin = [h.get("kin") for h in arm["history"] if h.get("kin") is not None]
        out.update(kin_last=kin[-1] if kin else None, windows=sum(1 for h in arm["history"] if h.get("outer_accepted")))
    except (OSError, KeyError, ValueError):
        pass
    try:
        m = re.search(r"windows 3-10: apart max (\d+)", (d / f"apart_{tag}.log").read_text())
        out["apart"] = int(m.group(1)) if m else None
    except OSError:
        pass
    for rng, fn in (("w3_10", f"vis_w3_10_{tag}.jsonl"), ("run", f"vis_full_{tag}.jsonl")):
        try:
            rr = [json.loads(s) for s in open(d / fn) if s.startswith("{")]
            rr = [x for x in rr if x.get("window") == "range"]
            if rr:
                out[f"sb_{rng}"] = rr[-1]["side_blob"]
        except OSError:
            pass
    return out


ia, iv = ind(DA), ind(DV)
lines, summary = [], {}
for arm in ("NRD", "NUC", "NSS"):
    lines += [f"### A6 at 40k: {arm} (seed 97) against the tag's V (seeds 97 / 99 / 101)", "",
              "| mesh | arm sil IoU ind | V min / med / max | position | arm - V97 | arm - V med | chamfer arm / V97 | "
              "apart w3-10 arm / V97 | side blobs w3-10 arm / V97 | side blobs run arm / V97 | end kin arm / V97 | "
              "windows arm / V97 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    cnt, dmed, more_apart, more_sb, n = {"below": 0, "inside": 0, "above": 0}, [], [], [], 0
    for mesh in MESHES:
        a = ia.get(f"{mesh}40k_{arm}97")
        v = {s: iv.get(f"{mesh}40k_V{s}") for s in SEEDS}
        vv = [v[s]["ind"]["sil"] for s in SEEDS if v[s]]
        if not a or not v[97]:
            lines.append(f"| {mesh} | {'-' if not a else round(a['ind']['sil'], 4)} | " +
                         ("-" if not vv else f"{min(vv):.4f} / {np.median(vv):.4f} / {max(vv):.4f}") +
                         " | pending | | | | | | | | |")
            continue
        x = a["ind"]["sil"]
        pos = "inside"
        if len(vv) == 3:
            pos = "below" if x < min(vv) else ("above" if x > max(vv) else "inside")
            cnt[pos] += 1
            dmed.append(x - float(np.median(vv)))
        else:
            pos = f"({len(vv)} V runs)"
        n += 1
        ea, ev = extra(DA, f"{mesh}40k_{arm}97"), extra(DV, f"{mesh}40k_V97")
        g = lambda k, f="{}": "-" if ea.get(k) is None or ev.get(k) is None else f"{f.format(ea[k])} / {f.format(ev[k])}"
        if ea.get("apart") is not None and ev.get("apart") is not None and ea["apart"] > ev["apart"]:
            more_apart.append(mesh)
        if ea.get("sb_w3_10") is not None and ev.get("sb_w3_10") is not None and ea["sb_w3_10"] > ev["sb_w3_10"]:
            more_sb.append(mesh)
        lines.append(f"| {mesh} | {x:.4f} | " + (f"{min(vv):.4f} / {np.median(vv):.4f} / {max(vv):.4f}" if vv else "-") +
                     f" | {pos} | {x - v[97]['ind']['sil']:+.4f} | " +
                     ("-" if len(vv) < 3 else f"{x - float(np.median(vv)):+.4f}") +
                     f" | {a['ind']['chamfer']:.4f} / {v[97]['ind']['chamfer']:.4f} | {g('apart')} | {g('sb_w3_10')} | "
                     f"{g('sb_run')} | {g('kin_last', '{:.2e}')} | {g('windows')} |")
    summary[arm] = dict(n=n, **cnt, median_minus_Vmed=float(np.median(dmed)) if dmed else None,
                        more_apart_than_V97=more_apart, more_side_blobs_w3_10_than_V97=more_sb)
    lines += ["", f"Summary {arm}: meshes read {n}; against V's three-run range below {cnt['below']}, inside "
              f"{cnt['inside']}, above {cnt['above']}; median arm - V median "
              + ("-" if not dmed else f"{np.median(dmed):+.4f}") + f"; apart w3-10 above V97 on {len(more_apart)}; "
              f"side blobs w3-10 above V97 on {len(more_sb)}.", ""]
text = "\n".join(lines)
print(text)
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(text + "\n")
    Path(sys.argv[1]).with_suffix(".json").write_text(json.dumps(summary, indent=1))
