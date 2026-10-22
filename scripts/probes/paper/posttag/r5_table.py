"""r5_table.py OUT_MD -- D145 reading 5: the tables (CPU; reads the readers' files only).

  (1) the cell-size runs (output/gpu/d145t_cell, LC97 / PC97) beside B1's L / P (seeds 97 / 99 / 101) of teapot, homer,
      dragon: topology (end genus, components, solid pieces with particles, the teapot's handle plane, the largest extra
      clusters), the display timeline (genus and pieces over 0-4 / 0-6 s, the dragon's 1.5-3.0 s), end quality (the
      yardstick, the display, silhouette IoU / chamfer / thin, F at 1 pitch, D157's Boundary IoU / F at 0.5 / PSNR
      against the corrected mesh), rest and cost (end kinetic energy, wall, windows, seconds per attempt, GPU memory),
      visible pieces (visible_apart over the run, apart over windows 3-10, far pieces);
  (2) D157's reader on the external methods (route pts: their end samples by our route, against the corrected mesh;
      route mesh: their own end meshes against the raw mesh) beside D157's rows for ours (L) and the own sample (the
      floor);
  (3) is r5_tail.py's own file."""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
X = G / "d145_ext/r5"
R4 = G / "d145_ext/r4"
B1, CELL = G / "d145t_b1", G / "d145t_cell"
TG = {"bunny": 0, "dragon": 15, "teapot": 1, "bob": 1, "armadilo": 0, "homer": 0}
MESHES = ["bunny", "dragon", "teapot", "bob", "armadilo", "homer"]


def jl(p):
    try:
        return [json.loads(s) for s in open(p) if s.startswith("{")]
    except OSError:
        return []


def last(path, pred):
    rows = [r for r in jl(path) if pred(r)]
    return rows[-1] if rows else None


def rng(v, fmt, s=1.0):
    v = [x * s for x in v if x is not None and np.isfinite(x)]
    if not v:
        return "-"
    lo, hi = format(min(v), fmt), format(max(v), fmt)
    return lo if lo == hi else f"{lo}-{hi}"


def status(d):
    out = {}
    for s in open(d / "status.txt"):
        m = re.match(r"(\S+) sim (\d+) wall (\d+) peak_mib (\d+)", s)
        if m:
            out[m.group(1)] = dict(rc=int(m.group(2)), wall=int(m.group(3)), peak=int(m.group(4)))
    return out


ST = {**status(B1), **status(CELL)}
D157 = {}
for f in glob.glob(str(G / "d157/read300_*.jsonl")):
    for r in jl(f):
        D157[(r["mesh"], r["seed"], r["label"])] = r


def d157_vals(r, kind="zb", ikind=None, ref="cor"):
    """(Boundary IoU error, F at 0.5, held-out PSNR, Chamfer-L1 (pitches), NC) of a D157-format row."""
    if r is None or "error" in r:
        return None
    try:
        b = r["sil"][kind][ref]["1024"]["held"]["biou2pct_err"]
        d3 = r["d3"][kind][ref]
        img = r["img"][ikind][ref] if ikind else r["img"][ref]
        return b, d3["f0.5"], float(np.mean(img["shaded"]["psnr"])), d3["chamfer_l1_a"], d3["nc"]
    except (KeyError, TypeError):
        return None


def run_record(d, name):
    try:
        js = json.load(open(d / f"{name}.json"))
    except OSError:
        return {}
    arm = js["arms"][next(iter(js["arms"]))]
    h = arm["history"]
    c = [x for x in h if x.get("outer_accepted")]
    t = [x.get("t_attempt") or 0.0 for x in h]
    return dict(windows=len(c), attempts=len(h), s_attempt=float(np.mean(t)) if t else None,
                t_end=c[-1]["frame_end"] / 240.0 if c else None, kin=c[-1].get("kin") if c else None,
                dx=js["provenance"].get("mpm", {}).get("dx"))


def end_reads(d, name):
    t = last(d / f"t3end_{name}.log", lambda r: "exterior" in r)
    e = last(d / f"e3end_{name}.log", lambda r: r.get("state") not in ("target", "own") and "to_target" in r)
    return dict(y_sil=t["exterior"]["sil"] if t else None, y_pbr=t["exterior"]["pbr"] if t else None,
                d_front=1 - e["to_target"]["front"]["iou"] if e else None,
                d_back=1 - e["to_target"]["back"]["iou"] if e else None)


def timeline(name):
    """[(t, genus of the largest, components, solid pieces with particles, cavities)] of r5_timeline.sh's frames."""
    topo = {r["tag"]: r for r in jl(X / f"topo_{name}.jsonl")}
    out, end = [], None
    for r in jl(X / f"display_{name}.jsonl"):
        mesh = r["mesh"]
        tag = f"{mesh}_{name}_{r['raw']}"
        tp = topo.get(tag) or {}
        item = (r["raw"] / 240.0, tp.get("genus_largest"), r["components"], r["pieces_with_particles"], r["cavities"],
                tp.get("watertight"), tp.get("self_intersecting_pairs"))
        out.append(item)
    for k, tp in topo.items():
        if k.endswith("_end"):
            end = tp
    return sorted(set(out)), end


L = ["# D145 reading 5", ""]
# ---------------------------------------------------------------------------------------------------------- (1)
L += ["## (1) The cell-size runs (`--cell_follows_n`, 3.3-pitch cell) beside B1 (the shape's 6.4-pitch cell)", ""]
ext3 = json.load(open(G / "d145_ext/ext_table.json"))
extc = {r["label"]: r for r in jl(X / "ext_cell.jsonl")}
fsc = {r["label"]: r for r in jl(X / "fscore_cell.jsonl")}
fsb = {(r["mesh"], r["label"]): r for r in jl(R4 / "fscore.jsonl") + jl(X / "fscore_b1P.jsonl") if r["method"] == "ours"}
extra = {(r["mesh"], r["label"]): r for r in jl(R4 / "extra.jsonl") + jl(X / "extra_b1homer.jsonl") + jl(X / "extra_cell.jsonl")}
r4topo = {r["tag"]: r for r in jl(R4 / "topo.jsonl")}
vis = {}
for d in (B1, CELL):
    for f in glob.glob(str(d / "vis_full_*.jsonl")):
        for r in jl(f):
            if r.get("window") == "range":
                vis[r["run"]] = r
pieces = {}
for f in [G / "d145t_evidence/pieces_full_d145t_b1.jsonl", X / "pieces_cell.jsonl"]:
    for r in jl(f):
        pieces[r["run"]] = r


def apart_line(d, name):
    try:
        s = [ln for ln in open(d / f"apart_{name}.log") if ln.startswith("spacing")]
        m = re.search(r"windows 3-10: apart max (\d+) median (\d+)", s[-1]) if s else None
        return f"{m.group(1)} / {m.group(2)}" if m else "-"
    except OSError:
        return "-"


L += ["| mesh | run | cell | end genus (target) | end components | genus over the timeline (max) | yardstick sil e-4 | "
      "shading e-4 | display front % | sil IoU / chamfer / thin % | F at 1 pitch | D157: BIoU err / F@0.5 / PSNR | "
      "end kinetic | wall min / windows / s per attempt / GPU GiB | visible blobs (side) over the run | apart w3-10 max / "
      "median | far-piece frames / most far particles |",
      "|" + "---|" * 17]
for mesh in ("teapot", "homer", "dragon"):
    for arm in ("L", "P"):
        for seed, cell in ((97, "shape"), (99, "shape"), (101, "shape"), (97, "N")):
            name = f"{mesh}300k_{arm}{seed}" if cell == "shape" else f"{mesh}300k_{arm}C97"
            d = B1 if cell == "shape" else CELL
            if not (d / f"{name}.json").exists() and name not in ST:
                continue
            er = end_reads(d, name)
            rec = run_record(d, name)
            tl, endtopo = timeline(name)
            if endtopo is None and cell == "shape":              # B1 seeds without a timeline: reading 4's end mesh
                endtopo = r4topo.get(f"{mesh}_{arm}{seed}_end")
            gmax = max((g for _, g, *_ in tl if g is not None), default=None)
            if cell == "N":
                e = extc.get(name)
                ext_s = "-" if not e else f"{e['sil_ind']:.4f} / {e['chamfer_ind']:.4f} / {100 * e['thin_ind']:.1f}"
                f1 = fsc.get(name, {}).get("F1")
                dv = None
                for r in jl(X / f"d157x_cell_{name}.jsonl"):
                    dv = d157_vals(r, "zb", "zbs")
            else:
                o = ext3.get(f"{mesh}|ours", {})
                lab = f"L{seed}"
                if arm == "L" and lab in o.get("labels", []):
                    i = o["labels"].index(lab)
                    ext_s = f"{o['sil_ind'][i]:.4f} / {o['chamfer_ind'][i]:.4f} / {100 * o['thin_ind'][i]:.1f}"
                else:
                    ext_s = "-"
                f1 = fsb.get((mesh, f"{arm}{seed}"), {}).get("F1")
                dv = d157_vals(D157.get((mesh, seed, arm)))
            st = ST.get(name, {})
            dtxt = "-" if not dv else f"{dv[0]:.4f} / {dv[1]:.3f} / {dv[2]:.2f}"
            v = vis.get(name)
            vtxt = "-" if not v else f"{v['blob']} ({v['side_blob']})"
            p = pieces.get(name)
            ptxt = "-" if not p else f"{p.get('far_frames', '-')} / {p.get('far_particles_max', '-')}"
            cost = (f"{st.get('wall', 0) / 60:.1f} / {rec.get('windows', '-')} / "
                    f"{rec.get('s_attempt') or 0:.1f} / {st.get('peak', 0) / 1024:.0f}") if st else "-"
            L.append(f"| {mesh} | {arm if cell == 'shape' else arm + 'C'}{seed} | {cell} | "
                     f"{endtopo['genus_largest'] if endtopo else '-'} ({TG[mesh]}) | {endtopo['components'] if endtopo else '-'} | "
                     f"{gmax if gmax is not None else '-'} | {rng([er['y_sil']], '.3g', 1e4)} | {rng([er['y_pbr']], '.3g', 1e4)} | "
                     f"{rng([er['d_front']], '.3g', 100)} | {ext_s} | {rng([f1], '.3f')} | {dtxt} | "
                     f"{rng([rec.get('kin')], '.2g')} | {cost} | {vtxt} | {apart_line(d, name)} | {ptxt} |")
# the handle plane and the extra clusters
L += ["", "### The teapot's handle plane (x = 0) and the largest extra-material clusters", ""]
for r in jl(R4 / "slice/teapot_x0.jsonl"):
    L.append(f"- B1 {r['label']}: enclosed openings {r['enclosed_holes']}" +
             (f", the largest {r['holes'][0]['area_a2']} pitches^2" if r["holes"] else ""))
for f in sorted(glob.glob(str(X / "slice_*.jsonl"))):
    for r in jl(f):
        L.append(f"- {r['label']}: enclosed openings {r['enclosed_holes']}" +
                 (f", the largest {r['holes'][0]['area_a2']} pitches^2 at y, z = {r['holes'][0]['centroid']}" if r["holes"] else ""))
L += ["", "| mesh | state | extra at 1 (2) pitches | missing at 1 (2) | the three largest extra clusters (particles, deepest pitches) |",
      "|---|---|---|---|---|"]
for (mesh, lab), r in sorted(extra.items(), key=lambda t: (t[0][0], t[0][1])):
    if mesh not in ("teapot", "homer", "dragon"):
        continue
    cl = "; ".join(f"{c['size']} ({c['max_dist_a']:.1f})" for c in r["extra_clusters"][:3])
    L.append(f"| {mesh} | {lab} | {r['extra']['1']} ({r['extra']['2']}) | {r['missing']['1']} ({r['missing']['2']}) | {cl} |")
# the timelines
L += ["", "### The display's timeline (t s: genus of the largest / components / solid pieces with particles / cavities)", ""]
for mesh in ("teapot", "homer", "dragon"):
    for name in (f"{mesh}300k_L97", f"{mesh}300k_P97", f"{mesh}300k_LC97", f"{mesh}300k_PC97"):
        tl, endtopo = timeline(name)
        if not tl:
            continue
        nonwt = sum(1 for x in tl if x[5] is False)
        si = sum(1 for x in tl if x[6] not in (0, None))
        txt = ", ".join(f"{t:.2f}: {g}/{c}/{p}/{cv}" for t, g, c, p, cv, _, _ in tl if t <= 6.01)
        L.append(f"- **{name}** ({len(tl)} frames; not watertight {nonwt}, with self-intersection {si}): {txt}")
# ---------------------------------------------------------------------------------------------------------- (2)
L += ["", "## (2) D157's reader on the external methods (median over seeds; per mesh)", "",
      "Route pts: the end's 300k sample by our route (Zhu-Bridson exterior / display) against the corrected mesh, as D157 "
      "reads ours. Route mesh: the method's own end mesh against the raw mesh (no sample anywhere). BIoU err = 1 - "
      "Boundary IoU (held-out views, 1024 px); F@0.5 = 3D F-score at half a pitch; PSNR = held-out studio images (dB); "
      "CD = Chamfer-L1 (pitches); NC = normal consistency.", "",
      "| mesh | method | route | n | BIoU err | F@0.5 | PSNR | CD | NC |", "|---|---|---|---|---|---|---|---|---|"]
NAMES = dict(L="ours (render arm)", P="ours, λ = 0 twin", own="a perfect sample (the own target sample)",
             XU="Xu et al.", BUO="Buonomo et al.", NISE="NISE", ISDn="ISD", FDn="4Deform", ISDr="ISD radial",
             FDr="4Deform radial", ARAP="ARAP", OT="OT", LSa="Large Steps (a)", LSb="Large Steps (b)", FCa="FlexiCubes (a)",
             FCb="FlexiCubes (b)")


def method_of(lab):
    m = re.match(r"(XU|BUO|NISE|ISDn|FDn|ISDr|FDr|ARAP|OT|LSa|LSb|FCa|FCb)", lab)
    return m.group(1) if m else lab


summary = {}
for mesh in MESHES:
    rows = {}
    for (m2, seed, lab), r in D157.items():
        if m2 == mesh and lab in ("L", "P", "own"):
            v = d157_vals(r)
            if v:
                rows.setdefault((lab, "pts"), []).append(v)
    for r in jl(X / f"d157x_ext_{mesh}.jsonl"):
        if r["label"] == "L":
            continue                                          # the replication row (equals D157's)
        v = d157_vals(r, "zb", "zbs", "cor") if r["route"] == "pts" else d157_vals(r, "mesh", "mesh", "raw")
        if v:
            rows.setdefault((method_of(r["label"]), r["route"]), []).append(v)
    for (k, route), vs in sorted(rows.items(), key=lambda t: (list(NAMES).index(t[0][0]) if t[0][0] in NAMES else 99, t[0][1])):
        a = np.array(vs)
        med = np.median(a, 0)
        summary[(mesh, k, route)] = dict(n=len(vs), lo=a.min(0).tolist(), hi=a.max(0).tolist(), med=med.tolist())
        L.append(f"| {mesh} | {NAMES.get(k, k)} | {route} | {len(vs)} | {rng(a[:, 0], '.4f')} | {rng(a[:, 1], '.3f')} | "
                 f"{rng(a[:, 2], '.2f')} | {rng(a[:, 3], '.3f')} | {rng(a[:, 4], '.4f')} |")
# who is ahead of ours (L's seed range) per mesh and measure
L += ["", "### Against ours' seed range (ahead = the method's range lies on the better side of ours'; level = overlap)", "",
      "| method | route | BIoU err: ahead / level / behind (meshes) | F@0.5 | PSNR |", "|---|---|---|---|---|"]
keys = sorted({(k, route) for (_, k, route) in summary if k != "L"})
for k, route in keys:
    cells = []
    for j, better_low in ((0, True), (1, False), (2, False)):
        ah = lv = bh = 0
        for mesh in MESHES:
            o, s = summary.get((mesh, "L", "pts")), summary.get((mesh, k, route))
            if not o or not s:
                continue
            if better_low:
                ahead, behind = s["hi"][j] < o["lo"][j], s["lo"][j] > o["hi"][j]
            else:
                ahead, behind = s["lo"][j] > o["hi"][j], s["hi"][j] < o["lo"][j]
            ah += ahead
            bh += behind
            lv += not (ahead or behind)
        cells.append(f"{ah} / {lv} / {bh}")
    L.append(f"| {NAMES.get(k, k)} | {route} | " + " | ".join(cells) + " |")
text = "\n".join(L) + "\n"
Path(sys.argv[1]).write_text(text)
json.dump({f"{m}|{k}|{r}": v for (m, k, r), v in summary.items()}, open(Path(sys.argv[1]).with_suffix(".json"), "w"))
print(text)
