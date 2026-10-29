"""ext_table145.py OUT_MD -- D145's methods table (2026-10-20): every method beside ours on the six 300k meshes, from the
files the readers wrote (CPU only, nothing recomputed here).

Sources (output/gpu = G):
  end quality   G/d145_ext/ext_end.jsonl (ext_end145.py: every method's end frame against the independent sample;
                silhouette IoU, chamfer, thin uncovered, the floor), all seeds
  volume        D145's one routine (vol145 / vol145n): G/d145_vol/vol.jsonl (Xu 97, Buonomo 1, ARAP, OT), Buonomo's
                repo results/vol145_nise.jsonl (NISE), baselines/b14/results/vol.jsonl (ISD, 4Deform), G/d145_ext/
                vol_ext.jsonl (ours on the tag, Buonomo 2-3, two B14 runs with an unfillable frame), B12's rows (LS, FC)
  topology      the converters' per-frame mesh records: baselines/b14/results/*_meshes.json (ISD, 4Deform: the largest
                component's genus, components, watertight), baselines/buonomo/results/*_meshes.json (Buonomo, NISE:
                components, Euler, watertight; genus of the closed frames = components - Euler / 2), baselines/arap/
                results/*_arap.json (ARAP: one mesh of the target's connectivity; self-intersecting pairs), B12's rows
                (LS, FC: genus, components, watertight, self-intersection)
  trajectory    the harness on one grid per mesh (the tag's L97 first): G/b13/harness (ours L97) and G/d145_ext/harness_*
                (every external method's first seed); pieces off the body (pieces145.py): G/d145t_evidence/
                pieces_full_d145t_b1.jsonl (ours L97) and G/d145_ext/pieces_ext.jsonl
  wall time     ours and Xu et al.: the run's own seconds; Buonomo et al. and NISE: the morph training and the
                reconstruction (status.txt; the SDF fits of seed 1 apart); ISD, 4Deform: training + evaluation
                (times.jsonl); ARAP: the CMCF map + the interpolation; LS, FC: the optimisation's net (B12); OT not timed."""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

B = Path("/data/relcfd/chayo/physmorph_v2")
G = B / "output/gpu"
MESHES = ["bunny", "dragon", "teapot", "bob", "armadilo", "homer"]
METHODS = [("ours", "ours (PhysMorph, tag)"), ("xu", "Xu et al. TVCG 2025"), ("buo", "Buonomo et al. SGP 2025"),
           ("nise", "NISE ICCV 2023"), ("isd", "ISD ICLR 2025"), ("fd", "4Deform CVPR 2025"),
           ("isd_r", "ISD, radial matching"), ("fd_r", "4Deform, radial matching"), ("arap", "ARAP"),
           ("ot", "OT displacement"), ("lsa", "Large Steps (a)"), ("lsb", "Large Steps (b)"),
           ("fca", "FlexiCubes (a)"), ("fcb", "FlexiCubes (b)")]


def jl(p):
    try:
        return [json.loads(s) for s in open(p) if s.startswith("{")]
    except OSError:
        return []


def method_of(label):
    if re.match(r"L\d+$", label):
        return "ours"
    if label.startswith("XU"):
        return "xu"
    if label.startswith("BUO"):
        return "buo"
    if label == "NISE":
        return "nise"
    if label.startswith("ISDr"):
        return "isd_r"
    if label.startswith("ISD"):
        return "isd"
    if label.startswith("FDr"):
        return "fd_r"
    if label.startswith("FD"):
        return "fd"
    if label == "ARAP":
        return "arap"
    if label == "OT":
        return "ot"
    return label[:3].lower()            # LSa97 -> lsa, FCb101 -> fcb


D = {}                                   # (mesh, method) -> dict of lists
def put(mesh, method, key, val):
    if val is None:
        return
    D.setdefault((mesh, method), {}).setdefault(key, []).append(val)


# --- end quality
for r in jl(G / "d145_ext/ext_end.jsonl"):
    m, k = r["mesh"], method_of(r["label"])
    for key in ("sil_ind", "chamfer_ind", "thin_ind", "floor", "floor_chamfer", "thin_floor"):
        put(m, k, key, r.get(key))
    put(m, k, "labels", r["label"])
# --- volume (min, max, end) per run
vol_rows = []
for r in jl(G / "d145_vol/vol.jsonl"):
    lab = {"xu_XU97": "XU97", "buo": "BUO", "arap": "ARAP", "ot": "OT"}.get(r["label"])
    if lab:
        vol_rows.append((r["mesh"], lab, r))
for r in jl(B / "baselines/buonomo/results/vol145_nise.jsonl"):
    vol_rows.append((r["mesh"], "NISE", r))
for r in jl(B / "baselines/b14/results/vol.jsonl"):
    vol_rows.append((r["mesh"], r["label"], r))
for r in jl(G / "d145_ext/vol_ext.jsonl"):
    lab = r["label"].split("_", 1)[1] if "_" in r["label"] else r["label"].upper()
    vol_rows.append((r["mesh"], lab, r))
for m, lab, r in vol_rows:
    k = method_of(lab)
    put(m, k, "vol_min", r["min"])
    put(m, k, "vol_max", r["max"])
    put(m, k, "vol_end", r["at"]["1"])
b12 = json.load(open(G / "d145_ext/b12_rows.json"))["rows"]
for row in b12:
    k = row["method"] + row["arm"]
    if row["method"] == "ours":
        continue
    for run in row["runs"]:
        m = row["mesh"]
        for key, src in (("vol_min", "vol_min"), ("vol_max", "vol_max"), ("vol_end", "vol_end"), ("genus_end", "genus_end"),
                         ("genus_max", "genus_max"), ("comp_end", "comp_end"), ("comp_max", "comp_max"),
                         ("multi_comp_frames", "frames_multi_comp"), ("watertight_frames", "watertight_frames"),
                         ("si_frames", "si_frames"), ("si_end", "si_end"), ("wall_s", "wall")):
            put(m, k, key, run.get(src))
# --- topology from the converters' mesh records
for f in glob.glob(str(B / "baselines/b14/results/*_meshes.json")):
    d = json.load(open(f))
    k = method_of(d["tag"])
    rows = [x for x in d["rows"] if x.get("V")]
    ge = [x.get("largest_area_comp", {}).get("genus") for x in rows]
    put(d["name"], k, "genus_end", ge[-1] if ge else None)
    put(d["name"], k, "genus_max", max((g for g in ge if g is not None), default=None))
    put(d["name"], k, "comp_end", rows[-1].get("components") if rows else None)
    put(d["name"], k, "comp_max", max((x.get("components") or 0 for x in rows), default=None))
    put(d["name"], k, "multi_comp_frames", sum(1 for x in rows if (x.get("components") or 0) > 1))
    put(d["name"], k, "watertight_frames", sum(1 for x in rows if x.get("watertight")))
    put(d["name"], k, "unsampled", len(d.get("unsampled", [])))
for f in glob.glob(str(B / "baselines/buonomo/results/*_meshes.json")):
    d = json.load(open(f))
    k = method_of(d["tag"])
    rows = d["rows"]
    gs = [((2 * x["components"] - x["euler"]) // 2) if x.get("watertight") and x.get("euler") is not None
          and x.get("components") else None for x in rows]
    put(d["name"], k, "genus_end", gs[-1])
    put(d["name"], k, "genus_max", max((g for g in gs if g is not None), default=None))
    put(d["name"], k, "comp_end", rows[-1].get("components"))
    put(d["name"], k, "comp_max", max((x.get("components") or 0 for x in rows), default=None))
    put(d["name"], k, "multi_comp_frames", sum(1 for x in rows if (x.get("components") or 0) > 1))
    put(d["name"], k, "watertight_frames", sum(1 for x in rows if x.get("watertight")))
for f in glob.glob(str(B / "baselines/arap/results/*_arap.json")):
    d = json.load(open(f))
    m = Path(f).name.split("_")[0]
    rows = d["frames"]["rows"]
    put(m, "arap", "genus_end", 0)
    put(m, "arap", "genus_max", 0)
    put(m, "arap", "comp_end", 1)
    put(m, "arap", "comp_max", 1)
    put(m, "arap", "multi_comp_frames", 0)
    put(m, "arap", "watertight_frames", sum(1 for x in rows if x.get("watertight")))
    put(m, "arap", "si_frames", sum(1 for x in rows if (x.get("self_intersecting_pairs") or 0) > 0))
    put(m, "arap", "si_end", rows[-1].get("self_intersecting_pairs"))
    put(m, "arap", "wall_s", d["cmcf"]["seconds"] + d["arap"]["seconds"])
# --- trajectory: harness and pieces
for m in MESHES:
    for r in jl(G / f"b13/harness/harness_{m}.jsonl") + jl(G / f"d145_ext/harness_{m}.jsonl"):
        if r["label"] in ("F97", "FM97"):
            continue
        k = method_of(r["label"])
        rd = r["read"]
        put(m, k, "p_end", r["p_end"])
        put(m, k, "p75", "yes" if rd.get("p0.75") else "no")
        for q in ("p0.25", "p0.5", "p0.75"):
            if rd.get(q):
                put(m, k, f"apart_{q}", rd[q]["apart"])
                put(m, k, f"occ_{q}", rd[q]["occupied_ratio"])
for r in jl(G / "d145t_evidence/pieces_full_d145t_b1.jsonl"):
    mm = re.match(r"(.+?)300k_(L97)$", r["run"])
    if mm:
        put(mm.group(1), "ours", "far_share", r.get("far_frames", 0) / r["frames_read"])
        put(mm.group(1), "ours", "far_ptcl", r.get("far_particles_max"))
        put(mm.group(1), "ours", "far_gap", r.get("run_farthest_a"))
        put(mm.group(1), "ours", "end_far", r.get("end_farthest_a"))
for r in [x for f in sorted(glob.glob(str(G / "d145_ext/pieces_ext*.jsonl"))) for x in jl(f)]:
    m, lab = r["run"].split(":")
    k = method_of(lab)
    put(m, k, "far_share", r.get("far_frames", 0) / r["frames_read"])
    put(m, k, "far_ptcl", r.get("far_particles_max"))
    put(m, k, "far_gap", r.get("run_farthest_a"))
    put(m, k, "end_far", r.get("end_farthest_a"))
    put(m, k, "collapsed", r.get("collapsed_frames", 0))
# --- wall time
for f in glob.glob(str(G / "d145t_b1/*300k_L*.json")):
    mm = re.match(r"(.+?)300k_L(\d+)\.json$", Path(f).name)
    if mm and mm.group(2) in ("97", "99", "101", "103", "105"):
        a = json.load(open(f))["arms"]
        put(mm.group(1), "ours", "wall_s", a[list(a)[0]]["seconds"])
for f in glob.glob(str(G / "d145_b4/*300k_XU*.json")):
    mm = re.match(r"(.+?)300k_XU(\d+)\.json$", Path(f).name)
    if mm:
        a = json.load(open(f))["arms"]
        put(mm.group(1), "xu", "wall_s", a[list(a)[0]]["seconds"])
st = open(B / "baselines/buonomo/status.txt").read().splitlines()
acc = {}
for s in st:
    mm = re.search(r"d145_b5/(\w+?)300k_(BUO\d?|NISE) (morph \S+|reconstruct) done (\d+) s", s)
    if mm:
        acc.setdefault((mm.group(1), mm.group(2)), {})[mm.group(3).split()[0]] = int(mm.group(4))
for (m, tag), v in acc.items():
    if "morph" in v:
        put(m, method_of(tag), "wall_s", v["morph"] + v.get("reconstruct", 0))
for r in jl(B / "baselines/b14/results/times.jsonl"):
    mm = re.match(r"(.+?)300k_(\w+)$", r["run"])
    put(mm.group(1), method_of(mm.group(2)), "wall_s", r["train_s"] + r["eval_s"])


def rng(v, fmt, scale=1.0):
    v = [x for x in (v or []) if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return "-"
    lo, hi = min(v) * scale, max(v) * scale
    if fmt == "d":
        lo, hi = int(round(lo)), int(round(hi))
    return format(lo, fmt) if format(lo, fmt) == format(hi, fmt) else f"{format(lo, fmt)}-{format(hi, fmt)}"


def wall(v):
    v = [x for x in (v or []) if x]
    if not v:
        return "-"
    lo, hi = min(v), max(v)
    f = (lambda s: f"{s / 60:.1f} min") if hi >= 120 else (lambda s: f"{s:.0f} s")
    return f(lo) if f(lo) == f(hi) else f"{f(lo)}-{f(hi)}"


L = ["# D145 methods table: every external method beside ours on the six 300k meshes", "",
     "End quality against the independent 300k sample (silhouette IoU, chamfer, thin uncovered of one thin set per mesh);",
     "ranges are min-max over the seeds / view-set seeds a method has (n). The floor is a perfect sample of the mesh.", ""]
# --- the master table: one row a method, across the meshes it ran on (per mesh the median over its seeds first)
TGT_GENUS = {"bunny": 0, "dragon": 15, "teapot": 1, "bob": 1, "armadilo": 0, "homer": 0}
med = lambda v: float(np.median([x for x in v if x is not None])) if v and any(x is not None for x in v) else None
L += ["## Master table (per mesh the median over seeds, then the range over the meshes the method ran on)", "",
      "| method | meshes | sil IoU ind | gap to the floor (1 - IoU over the floor's) | chamfer over the floor's | thin uncovered % | "
      "ours ahead on sil IoU (meshes, ranges apart) | end genus = target (mesh methods) | volume over the frames (one routine) | "
      "harness: p = 0.75 reached | apart at p 0.5 | frames with a piece >= 5 pitches off | wall |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for k, name in METHODS:
    ms = [m for m in MESHES if D.get((m, k), {}).get("sil_ind")]
    if not ms:
        continue
    g = lambda key: [med(D[(m, k)].get(key)) for m in ms]
    sil, ch, th = g("sil_ind"), g("chamfer_ind"), g("thin_ind")
    fl, flc = [med(D[(m, "ours")].get("floor")) for m in ms], [med(D[(m, "ours")].get("floor_chamfer")) for m in ms]
    gap = [(1 - s) / (1 - f) for s, f in zip(sil, fl) if s is not None and f]
    chr_ = [c / f for c, f in zip(ch, flc) if c is not None and f]
    ahead = 0
    for m in ms:
        a, b = D[(m, "ours")].get("sil_ind", []), D[(m, k)].get("sil_ind", [])
        if k != "ours" and a and b and min(a) > max(b):
            ahead += 1
    gen = "-"
    if k not in ("ours", "xu", "ot"):
        ok = [m for m in ms if D[(m, k)].get("genus_end") and med(D[(m, k)]["genus_end"]) == TGT_GENUS[m]]
        known = [m for m in ms if D[(m, k)].get("genus_end")]
        gen = f"{len(ok)} of {len(known)}" + ("" if len(known) == len(ms) else f" (end not closed on {len(ms) - len(known)})")
    vmin = [min(D[(m, k)].get("vol_min", [np.nan])) for m in ms]
    vmax = [max(D[(m, k)].get("vol_max", [np.nan])) for m in ms]
    vol = "-" if all(np.isnan(vmin)) else f"{np.nanmin(vmin):.2f}-{np.nanmax(vmax):.2f}"
    p75 = sum(1 for m in ms if "yes" in D[(m, k)].get("p75", []))
    hn = sum(1 for m in ms if D[(m, k)].get("p75"))
    ap = [med(D[(m, k)].get("apart_p0.5")) for m in ms]
    ap = [a for a in ap if a is not None]
    fs = [med(D[(m, k)].get("far_share")) for m in ms]
    fs = [x for x in fs if x is not None]
    wl = [w for m in ms for w in D[(m, k)].get("wall_s", [])]
    L.append(f"| {name} | {len(ms)} | {min(sil):.4f}-{max(sil):.4f} | {min(gap):.1f}-{max(gap):.1f} | "
             f"{min(chr_):.2f}-{max(chr_):.2f} | {min(th) * 100:.1f}-{max(th) * 100:.1f} | "
             f"{'-' if k == 'ours' else f'{ahead} of {len(ms)}'} | {gen} | {vol} | {'-' if not hn else f'{p75} of {hn}'} | "
             f"{'-' if not ap else f'{min(ap):.0f}-{max(ap):.0f}'} | "
             f"{'-' if not fs else f'{min(fs):.0%}-{max(fs):.0%}'} | {wall(wl)} |")
L.append("")
L += ["## A. The end and the cost", "",
      "| mesh | method | n | sil IoU ind | chamfer ind | thin uncovered % | genus end (largest comp.) | components end | wall |",
      "|---|---|---|---|---|---|---|---|---|"]
for m in MESHES:
    fl = D.get((m, "ours"), {})
    L.append(f"| {m} | *floor (a second sample)* | | {rng(fl.get('floor'), '.4f')} | {rng(fl.get('floor_chamfer'), '.4f')} | "
             f"{rng(fl.get('thin_floor'), '.1f', 100)} | {'0' if m in ('bunny', 'armadilo', 'homer') else {'dragon': '15', 'teapot': '1', 'bob': '1'}[m]} (target) | 1 | |")
    for k, name in METHODS:
        d = D.get((m, k))
        if not d or not d.get("sil_ind"):
            continue
        ge = "particles" if k in ("ours", "xu", "ot") else rng(d.get("genus_end"), "d")
        ce = "particles" if k in ("ours", "xu", "ot") else rng(d.get("comp_end"), "d")
        L.append(f"| {m} | {name} | {len(d['sil_ind'])} | {rng(d['sil_ind'], '.4f')} | {rng(d.get('chamfer_ind'), '.4f')} | "
                 f"{rng(d.get('thin_ind'), '.1f', 100)} | {ge} | {ce} | {wall(d.get('wall_s'))} |")
L += ["", "## B. The way there (21 frames for the baselines, every kept frame for ours)", "",
      "| mesh | method | genus max over frames | frames > 1 component | watertight frames | self-intersecting frames; end pairs | "
      "volume (one routine) min / max; end | harness p end | p = 0.75 reached | apart at p 0.25 / 0.5 / 0.75 | occupied at p 0.5 | "
      "frames with a piece >= 5 pitches off | most particles in such pieces | farthest piece (pitches) |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for m in MESHES:
    for k, name in METHODS:
        d = D.get((m, k))
        if not d or not d.get("sil_ind"):
            continue
        part = k in ("ours", "xu", "ot")
        implicit = k in ("buo", "nise", "isd", "fd", "isd_r", "fd_r")
        si = "-" if part else ("embedded (one level set)" if implicit else
                                f"{rng(d.get('si_frames'), 'd')}; {rng(d.get('si_end'), 'd')}")
        vol = "-" if not d.get("vol_min") else f"{rng(d['vol_min'], '.3f')} / {rng(d['vol_max'], '.3f')}; {rng(d['vol_end'], '.3f')}"
        ap = " / ".join(rng(d.get(f"apart_{q}"), "d") for q in ("p0.25", "p0.5", "p0.75"))
        L.append(f"| {m} | {name} | {'-' if part else rng(d.get('genus_max'), 'd')} | "
                 f"{'-' if part else rng(d.get('multi_comp_frames'), 'd')} | {'-' if part else rng(d.get('watertight_frames'), 'd')} | "
                 f"{si} | {vol} | {rng(d.get('p_end'), '.3f')} | {'/'.join(sorted(set(d.get('p75', [])))) or '-'} | {ap} | "
                 f"{rng(d.get('occ_p0.5'), '.3f')} | {rng(d.get('far_share'), '.0%')}"
                 + ("" if not any(d.get("collapsed", [])) else f" ({max(d['collapsed'])} collapsed, not read)")
                 + f" | {rng(d.get('far_ptcl'), 'd')} | "
                 f"{rng(d.get('far_gap'), '.1f')} |")
text = "\n".join(L) + "\n"
Path(sys.argv[1]).write_text(text)
json.dump({f"{m}|{k}": v for (m, k), v in D.items()}, open(Path(sys.argv[1]).with_suffix(".json"), "w"), default=float)
print(text)
