"""r4_table.py OUT_MD -- D145 reading 4: the tables of the measurement follow-ups (CPU; reads the readers' files only).

  (1) the yardstick (exterior silhouette and shading) and the display (1 - IoU front) at the end, every method, six
      meshes: r4/yard/t3end_* / e3end_* (the external methods) and d145t_b1/t3end_* / e3end_* (ours, L)
  (2) our display surface through the baselines' mesh readers: r4/topo.jsonl (end frames and the harness's frames)
  (3) one sampler: r4/disp_end.jsonl (our display mesh sampled as the baselines' meshes: silhouette IoU, chamfer, thin)
      and r4/fscore.jsonl (the F-score at 1 and 2 pitches, every method, sample against sample)
  (4) the harness's apart particles on the display surface: r4/display_*.jsonl and visible_apart's r4/vis_mid.jsonl
  and the updated master table: d145_ext/ext_table.json (reading 3) with the columns of (1) and (3) added."""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
X = G / "d145_ext/r4"
MESHES = ["bunny", "dragon", "teapot", "bob", "armadilo", "homer"]
NAMES = dict([("ours", "ours (PhysMorph, tag)"), ("ours_disp", "ours, display mesh resampled"), ("xu", "Xu et al. TVCG 2025"),
              ("buo", "Buonomo et al. SGP 2025"), ("nise", "NISE ICCV 2023"), ("isd", "ISD ICLR 2025"),
              ("fd", "4Deform CVPR 2025"), ("isd_r", "ISD, radial matching"), ("fd_r", "4Deform, radial matching"),
              ("arap", "ARAP"), ("ot", "OT displacement"), ("lsa", "Large Steps (a)"), ("lsb", "Large Steps (b)"),
              ("fca", "FlexiCubes (a)"), ("fcb", "FlexiCubes (b)")])


def method_of(label):
    for pat, k in ((r"L\d+$", "ours"), (r"XU", "xu"), (r"BUO", "buo"), (r"NISE", "nise"), (r"ISDr", "isd_r"),
                   (r"ISD", "isd"), (r"FDr", "fd_r"), (r"FD", "fd"), (r"ARAP", "arap"), (r"OT", "ot")):
        if re.match(pat, label):
            return k
    return label[:3].lower()


def jl(p):
    try:
        return [json.loads(s) for s in open(p) if s.startswith("{")]
    except OSError:
        return []


def last(path, pred):
    rows = [r for r in jl(path) if pred(r)]
    return rows[-1] if rows else None


D = {}
def put(m, k, key, v):
    if v is not None:
        D.setdefault((m, k), {}).setdefault(key, []).append(v)


# (1) exterior measures
srcs = [(f, "ext") for f in glob.glob(str(X / "yard/t3end_*.log"))] + \
       [(f, "ours") for f in glob.glob(str(G / "d145t_b1/t3end_*300k_L*.log"))]
for f, kind in srcs:
    m = re.match(r"t3end_(\w+?)300k_(\w+)\.log$", Path(f).name)
    if not m or (kind == "ours" and not re.match(r"L(97|99|101|103|105)$", m.group(2))):
        continue
    mesh, lab = m.group(1), m.group(2)
    k = method_of(lab)
    r = last(f, lambda r: "exterior" in r)
    put(mesh, k, "y_sil", r["exterior"]["sil"] if r else None)
    put(mesh, k, "y_pbr", r["exterior"]["pbr"] if r else None)
    if not r:
        put(mesh, k, "y_fail", lab)
    e = last(str(Path(f).parent / f"e3end_{mesh}300k_{lab}.log"),
             lambda r: r.get("state") not in ("target", "own") and "to_target" in r)
    if e is None and kind == "ext":           # a probe that failed here (out of memory): the pool's own end reading
        for pat, d in ((r"XU", "d145_b4"), (r"BUO", "d145_b5"), (r"(ISD|FD)", "d145_b14"), (r"ARAP", "d145_b11"),
                       (r"OT", "d145_b6")):
            if re.match(pat, lab):
                e = last(str(G / d / f"e3end_{mesh}300k_{lab}.log"),
                         lambda r: r.get("state") not in ("target", "own") and "to_target" in r)
                if e:
                    put(mesh, k, "d_from_pool", lab)
                break
        if e is None:
            put(mesh, k, "d_fail", lab)
    put(mesh, k, "d_front", 1 - e["to_target"]["front"]["iou"] if e else None)
    put(mesh, k, "d_back", 1 - e["to_target"]["back"]["iou"] if e else None)
    if kind == "ours":                        # the display's floor: the run's own target sample against IND
        o = last(str(Path(f).parent / f"e3end_{mesh}300k_{lab}.log"),
                 lambda r: r.get("state") == "own" and "to_target" in r)
        put(mesh, "floor", "d_front", 1 - o["to_target"]["front"]["iou"] if o else None)
        put(mesh, "floor", "d_back", 1 - o["to_target"]["back"]["iou"] if o else None)
# the yardstick's floor: the run's own target sample read by the same operator against IND (D152's d152_yard_floor.py)
for r in jl(G / "d152/yard_floor.jsonl"):
    if r["state"] == "own" and r["seed"] in (97, 99, 101, 103, 105):
        put(r["mesh"], "floor", "y_sil", r["exterior"]["sil"])
        put(r["mesh"], "floor", "y_pbr", r["exterior"]["pbr"])
# (3) one sampler and the F-score; FL* = a perfect sample's display mesh resampled (FLtgt the run's own target
# sample, a second sample of the mesh: the floor of this route; FLind the independent sample itself, scored against
# itself through the display mesh)
for r in jl(X / "disp_end.jsonl"):
    k = "ours_disp" if not r["label"].startswith("FL") else "disp_floor_" + r["label"][2:]
    for key in ("sil_ind", "chamfer_ind", "thin_ind", "floor_chamfer", "floor", "thin_floor"):
        put(r["mesh"], k, key, r.get(key))
for r in jl(X / "fscore.jsonl") + jl(X / "fscore_floor.jsonl"):
    k = r["method"] if r["method"] != "ours_disp_floor" else "disp_floor_" + r["label"][2:]
    for key in ("F1", "F2", "P1", "R1"):
        put(r["mesh"], k, key, r[key])
    put(r["mesh"], k, "F1_floor", r["floor"]["F1"])
    put(r["mesh"], k, "F2_floor", r["floor"]["F2"])
    if k == "ours":
        put(r["mesh"], "floor", "F1", r["floor"]["F1"])
        put(r["mesh"], "floor", "F2", r["floor"]["F2"])
prev = json.load(open(G / "d145_ext/ext_table.json"))
for m in MESHES:                              # the floor's particle measures, reading 3's (the run's own target sample)
    o = prev.get(f"{m}|ours", {})
    for key, src_key in (("sil_ind", "floor"), ("chamfer_ind", "floor_chamfer"), ("thin_ind", "thin_floor")):
        for v in o.get(src_key, []):
            put(m, "floor", key, v)


def rng(v, fmt, s=1.0):
    v = [x for x in (v or []) if x is not None and np.isfinite(x)]
    if not v:
        return "-"
    lo, hi = format(min(v) * s, fmt), format(max(v) * s, fmt)
    return lo if lo == hi else f"{lo}-{hi}"


med = lambda v: float(np.median(v)) if v else None
L = ["# D145 reading 4: the measurement follow-ups of the methods table", ""]
L += ["## (1) and (3): the end on the surface-resolving measures, every method (seed ranges)", "",
      "| mesh | method | yardstick silhouette e-4 | yardstick shading e-4 | display front 1 - IoU % | display back 1 - IoU % | "
      "F-score at 1 pitch | F-score at 2 pitches |", "|---|---|---|---|---|---|---|---|"]
for mesh in MESHES:
    fl = D.get((mesh, "floor"), {})
    L.append(f"| {mesh} | *floor (the run's own target sample)* | {rng(fl.get('y_sil'), '.3g', 1e4)} | "
             f"{rng(fl.get('y_pbr'), '.3g', 1e4)} | {rng(fl.get('d_front'), '.3g', 100)} | {rng(fl.get('d_back'), '.3g', 100)} | "
             f"{rng(fl.get('F1'), '.3f')} | {rng(fl.get('F2'), '.3f')} |")
    for k, name in NAMES.items():
        d = D.get((mesh, k))
        if not d:
            continue
        fail = "" if not d.get("y_fail") else f" ({len(d['y_fail'])} failed)"
        dnote = ("" if not d.get("d_fail") else f" ({len(d['d_fail'])} probe failed)") + \
                ("" if not d.get("d_from_pool") else f" ({len(d['d_from_pool'])} from the pool's reading)")
        L.append(f"| {mesh} | {name} | {rng(d.get('y_sil'), '.3g', 1e4)}{fail} | {rng(d.get('y_pbr'), '.3g', 1e4)} | "
                 f"{rng(d.get('d_front'), '.3g', 100)}{dnote} | {rng(d.get('d_back'), '.3g', 100)} | {rng(d.get('F1'), '.3f')} | "
                 f"{rng(d.get('F2'), '.3f')} |")
# ours' display mesh resampled, beside the particles
L += ["", "## (3) Our display mesh sampled as the baselines' meshes were (300k, `sample_volume_stratified`, seed 97)", "",
      "| mesh | ours (particles): sil IoU / chamfer / thin % | ours (display mesh resampled): sil IoU / chamfer / thin % / F1 | "
      "floor, particles (own target sample): sil IoU / chamfer / thin % / F1 | floor, display route (own target sample's "
      "display mesh resampled): sil IoU / chamfer / thin % / F1 | best external method's chamfer (median) |",
      "|---|---|---|---|---|---|"]
for mesh in MESHES:
    o, dsp = prev.get(f"{mesh}|ours", {}), D.get((mesh, "ours_disp"), {})
    fl, fd = D.get((mesh, "floor"), {}), D.get((mesh, "disp_floor_tgt"), {})
    best = min((med(prev[f"{mesh}|{k}"]["chamfer_ind"]), k) for k in ("isd", "arap", "lsb", "nise")
               if prev.get(f"{mesh}|{k}", {}).get("chamfer_ind"))
    four = lambda d: (f"{rng(d.get('sil_ind'), '.4f')} / {rng(d.get('chamfer_ind'), '.4f')} / "
                      f"{rng(d.get('thin_ind'), '.1f', 100)} / {rng(d.get('F1'), '.3f')}")
    L.append(f"| {mesh} | {rng(o.get('sil_ind'), '.4f')} / {rng(o.get('chamfer_ind'), '.4f')} / {rng(o.get('thin_ind'), '.1f', 100)} | "
             f"{four(dsp)} | {four(fl)} | {four(fd)} | {best[0]:.4f} ({NAMES[best[1]]}) |")
# (2) topology of the display mesh
L += ["", "## (2) Our display surface through the baselines' mesh readers", "",
      "| mesh | state | watertight | components (solid pieces / cavities) | particles at the other components | "
      "genus of the largest (sum over closed) | self-intersecting pairs | faces |", "|---|---|---|---|---|---|---|---|"]
TG = {"bunny": 0, "dragon": 15, "teapot": 1, "bob": 1, "armadilo": 0, "homer": 0}
disp = {(r["mesh"], r["label"], str(r["raw"])): r for f in glob.glob(str(X / "display_*.jsonl"))
        if "floor" not in f for r in jl(f)}
dflo = {(r["mesh"], r["label"], {-2: "ind", -1: "tgt"}.get(r["raw"], str(r["raw"]))): r
        for r in jl(X / "display_floor.jsonl")}
endraw = {}                                   # a state of r4_mesh.py's `end` spec is the largest raw of its mesh-label
for (m2, l2, r2) in disp:
    endraw[(m2, l2)] = max(endraw.get((m2, l2), -1), int(r2))
mid = {(m2, l2, r2) for (m2, l2, r2) in disp if int(r2) != endraw[(m2, l2)]}
STATE = {"ind": "floor: the independent sample", "tgt": "floor: the run's own target sample"}
rows = [(r, False) for r in jl(X / "topo.jsonl")] + [(r, True) for r in jl(X / "topo_floor.jsonl")]
for r, isfl in sorted(rows, key=lambda t: (MESHES.index(t[0]["tag"].split("_")[0]), t[1], t[0]["tag"])):
    mesh, lab, raw = re.match(r"(\w+?)_(L\d+)_(\w+)$", r["tag"]).groups()
    if isfl:
        dr = dflo.get((mesh, lab, raw), {})
        st = STATE.get(raw, f"{lab} raw {raw} (the most particles >= 5 pitches off the body)")
    else:
        dr = disp.get((mesh, lab, str(endraw.get((mesh, lab))) if raw == "end" else raw), {})
        st = f"{lab} end" if raw == "end" else f"{lab} raw {raw}"
    L.append(f"| {mesh} | {st} | {'yes' if r['watertight'] else 'no'} | {r['components']} "
             f"({dr.get('solid_pieces', '-')} / {dr.get('cavities', '-')}) | {dr.get('particles_off_main', '-')} | "
             f"{r['genus_largest']} ({r['genus_sum_closed']}; target {TG[mesh]}) | {r['self_intersecting_pairs']} | {r['F']} |")
# (2b) where the end's topology departs from the target's: extra / missing material against the run's own target
# sample (r4_extra.py) and the teapot's handle plane (r4_slice.py)
L += ["", "### Where the end departs from the target: material farther than 1 (2) pitches from the other sample "
      "(r4_extra.py; `ind` = the independent sample, the floor)", "",
      "| mesh | state | extra at 1 (2) pitches | missing at 1 (2) | the two largest extra clusters: particles, centre in the "
      "target's box, extent (pitches), deepest (pitches), nearest target surface |", "|---|---|---|---|---|"]
for r in sorted(jl(X / "extra.jsonl"), key=lambda r: (MESHES.index(r["mesh"]), r["label"] != "ind", r["label"])):
    cl = "; ".join(f"{c['size']} at ({', '.join(f'{v:.2f}' for v in c['centroid'])}), "
                   f"{' x '.join(f'{v:.0f}' for v in c['extent_a'])}, {c['max_dist_a']:.1f}, {c['near_target_surface']}"
                   for c in r["extra_clusters"][:2])
    L.append(f"| {r['mesh']} | {r['label']} | {r['extra']['1']} ({r['extra']['2']}) | {r['missing']['1']} "
             f"({r['missing']['2']}) | {cl} |")
sl = jl(X / "slice/teapot_x0.jsonl")
if sl:
    L += ["", "Teapot, the display's zero set in the handle's plane x = 0 (r4_slice.py): enclosed openings "
          + ", ".join(f"{r['label']} {r['enclosed_holes']}" + (f" ({r['holes'][0]['area_a2']} pitches^2 at y, z = "
                                                              f"{r['holes'][0]['centroid'][0]}, {r['holes'][0]['centroid'][1]})"
                                                              if r["holes"] else "") for r in sl) + "."]
# (4) the apart particles on the display surface; visible_apart.py's per-state counts from its log (its --jsonl is
# rewritten by each call): "<run> windows a-b, unique: apart N, hidden N, bump N, blob N, ..., side_blob N"
vis = {}
for s in open(X / "vis_mid.log"):
    mm = re.match(r"(\S+) windows \S+, unique: (.*)$", s.strip())
    if mm:
        vis[mm.group(1)] = {k: v for k, v in (kv.split(" ") for kv in mm.group(2).split(", "))}
L += ["", "## (4) The harness's apart particles at mid-progress, read on the display surface (L97)", "",
      "| mesh | frame (raw) | apart (harness link) | under the body's own display surface | in a separate solid display piece | "
      "solid pieces with particles (>= 10; >= 10 and >= 5 pitches off) | cavities | visible_apart: hidden / bump / blob (side blob) |",
      "|---|---|---|---|---|---|---|---|"]
items = [((m2, l2, r2), r, "") for (m2, l2, r2), r in disp.items() if (m2, l2, r2) in mid and l2 == "L97"]
items += [((m2, l2, r2), r, " (most far particles)" if r2 not in ("ind", "tgt") else f" ({STATE[r2]})")
          for (m2, l2, r2), r in dflo.items()]
for (mesh, lab, raw), r, note in sorted(items, key=lambda t: (MESHES.index(t[0][0]), t[2] != "",
                                                               int(t[0][2]) if t[0][2].isdigit() else -1)):
    v = vis.get(f"{mesh}_L97_{raw}")
    vs = "-" if not v else f"{v['hidden']} / {v['bump']} / {v['blob']} ({v['side_blob']})"
    L.append(f"| {mesh} | {raw}{note} | {r['apart']} | {r['apart_under_main']} | {r['apart_in_pieces']} | "
             f"{r['pieces_with_particles']} ({r['pieces_ge10']}; {r['pieces_ge10_gap5']}) | {r['cavities']} | {vs} |")
L += ["", "### visible_apart's kinds by the display component nearest each particle (r4_blob.py; main surface / internal "
      "cavity / separate solid piece)", "",
      "| mesh | raw | apart | hidden | bump | blob | blob's median distance to the display (pitches) |", "|---|---|---|---|---|---|---|"]
for r in sorted(jl(X / "blob.jsonl"), key=lambda r: (MESHES.index(r["mesh"]), r["raw"])):
    t3 = lambda k: f"{r[k]['main']} / {r[k]['cavity']} / {r[k]['piece']}"
    L.append(f"| {r['mesh']} | {r['raw']} | {r['apart']} | {t3('hidden')} | {t3('bump')} | {t3('blob')} | "
             f"{r['blob_dist_median_a']} |")
# the updated master table
L += ["", "## The updated master table (per mesh the median over seeds, then the range over meshes)", "",
      "| method | meshes | sil IoU ind | thin uncovered % | yardstick silhouette over the floor's | yardstick shading over "
      "the floor's | display front 1 - IoU over the floor's | F-score at 1 pitch | F-score at 2 pitches | chamfer over the "
      "floor's | end genus = target (median seed) | wall |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
order = ["floor", "disp_floor_tgt", "ours", "ours_disp", "xu", "buo", "nise", "isd", "fd", "isd_r", "fd_r", "arap", "ot",
         "lsa", "lsb", "fca", "fcb"]
NAMES["floor"] = "*floor: a second sample of the mesh (the run's own target sample)*"
NAMES["disp_floor_tgt"] = "*floor of the display route: that sample's display mesh resampled*"
gen = {}                                      # (mesh, method) -> the end genus per seed
for r in jl(X / "topo.jsonl"):
    if r["tag"].endswith("_end"):
        gen.setdefault((r["tag"].split("_")[0], "ours"), []).append(r["genus_largest"])
for r in jl(X / "topo_floor.jsonl"):
    if r["tag"].endswith("_tgt"):
        for k in ("floor", "disp_floor_tgt"):
            gen.setdefault((r["tag"].split("_")[0], k), []).append(r["genus_largest"])
for key, v in prev.items():
    m, k = key.split("|")
    if v.get("genus_end"):
        gen[(m, k)] = [g for g in v["genus_end"] if g is not None]
flo = lambda m, key: med(D.get((m, "floor"), {}).get(key))
for k in order:
    ms = [m for m in MESHES if prev.get(f"{m}|{k}", {}).get("sil_ind") or D.get((m, k), {}).get("sil_ind")]
    if not ms:
        continue
    src = lambda m: {**prev.get(f"{m}|{k}", {}), **D.get((m, k), {})}
    col = lambda key, s=1.0: [med(src(m).get(key)) * s for m in ms if src(m).get(key)]
    cr = [med(src(m)["chamfer_ind"]) / med(prev[f"{m}|ours"]["floor_chamfer"]) for m in ms if src(m).get("chamfer_ind")]
    wl = [w for m in ms for w in prev.get(f"{m}|{k}", {}).get("wall_s", [])]
    f = lambda v, fmt: "-" if not v else (format(min(v), fmt) if format(min(v), fmt) == format(max(v), fmt)
                                          else f"{format(min(v), fmt)}-{format(max(v), fmt)}")
    wtxt = "-" if not wl else (f"{min(wl) / 60:.1f}-{max(wl) / 60:.1f} min" if max(wl) >= 120 else f"{min(wl):.0f}-{max(wl):.0f} s")
    over = lambda key: [med(src(m)[key]) / flo(m, key) for m in ms if src(m).get(key) and flo(m, key)]
    gm = [(int(np.median(gen[(m, k)])) == TG[m]) for m in ms if gen.get((m, k))]
    gtxt = "-" if not gm else f"{sum(gm)} of {len(gm)}"
    L.append(f"| {NAMES[k]} | {len(ms)} | {f(col('sil_ind'), '.4f')} | {f(col('thin_ind', 100), '.1f')} | "
             f"{f(over('y_sil'), '.2f')} | {f(over('y_pbr'), '.2f')} | {f(over('d_front'), '.2f')} | {f(col('F1'), '.3f')} | "
             f"{f(col('F2'), '.3f')} | {f(cr, '.2f')} | {gtxt} | {wtxt} |")
text = "\n".join(L) + "\n"
Path(sys.argv[1]).write_text(text)
json.dump({f"{m}|{k}": v for (m, k), v in D.items()}, open(Path(sys.argv[1]).with_suffix(".json"), "w"), default=float)
print(text)
