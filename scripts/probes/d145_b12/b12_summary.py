"""B12 (D145) summary: one row per mesh x method x arm over the three view-set seeds (min-max), from the runs' run.json,
frames_check.json, B11's end check (endcheck.jsonl) and D145's volume routine (vol.jsonl); D145's harness on seed 97.
Ours beside it from D145's existing files only (nothing of ours is re-read here): the end check through the same
b11_endcheck.py (ours_endcheck.jsonl, B1's render arm L97 / L99 / L101), the volume routine (d145_vol, L97 bunny and dragon
only), the harness (d145_harness: the D135 deliverables SB / FB) and the run minutes (d145_evidence/tab_1015.jsonl).
The bunny's first batch (12:07-12:15 CDT, superseded by the amendment of 12:20 CDT) survives only as its first rows in
endcheck.jsonl / vol.jsonl (its run dirs were overwritten by the re-run); its end numbers are printed as a replicate.
Wall = the optimisation's net time (amendment (2): the per-iterate recording timed apart; every run of the measured set
has optimisation_net).
Writes output/gpu/d145_b12/b12_rows.json and prints markdown tables.  python3 b12_summary.py"""
import json
import os

B = "/data/relcfd/chayo/physmorph_v2"
B12 = f"{B}/baselines/b12"
O = f"{B}/output/gpu/d145_b12"
G = f"{B}/output/gpu"
MESHES = ("bunny", "dragon", "teapot", "bob", "armadilo", "homer")
SEEDS = (97, 99, 101)
NAMES = dict(ls="Large Steps", fc="FlexiCubes")
OURS_HARNESS = dict(bunny="SB", dragon="FB")


def jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(x) for x in open(path) if x.strip()]


end_rows = jsonl(f"{O}/endcheck.jsonl")
ends = {r["tag"]: r for r in end_rows}                          # the last row of a tag wins: the bunny's re-run
first_ends = {}
for r in end_rows:
    first_ends.setdefault(r["tag"], r)                          # the first row: the bunny's superseded first batch
vols = {(r["mesh"], r["label"]): r for r in jsonl(f"{O}/vol.jsonl")}
harn = {}
for m in MESHES:
    for r in jsonl(f"{O}/harness_{m}.jsonl"):
        harn[(m, r["label"])] = r


def rng(xs, fmt):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "-"
    lo, hi = min(xs), max(xs)
    return fmt.format(lo) if fmt.format(lo) == fmt.format(hi) else f"{fmt.format(lo)}-{fmt.format(hi)}"


def ints(fr, key):
    return [f[key] for f in fr if isinstance(f.get(key), int)]


rows = []
for m in MESHES:
    for method in ("ls", "fc"):
        for arm in ("a", "b"):
            per = []
            for s in SEEDS:
                label = f"{method.upper()}{arm}{s}"
                rd = f"{B12}/results/{method}/{m}_{arm}_s{s}"
                if not os.path.exists(f"{rd}/run.json"):
                    continue
                run = json.load(open(f"{rd}/run.json"))
                fc = json.load(open(f"{rd}/frames_check.json")) if os.path.exists(f"{rd}/frames_check.json") else None
                e = ends.get(f"{m}300k_{label}")
                v = vols.get((m, label))
                fr = fc["frames"] if fc else []
                gl = [f.get("genus_largest") for f in fr]
                mv = [f["vol_ratio_tgt_mesh"] for f in fr if f.get("vol_ratio_tgt_mesh") is not None]
                t10 = run.get("topology_every10") or []
                per.append(dict(
                    seed=s, label=label,
                    sil_ind=e["ind"]["sil"] if e else None, ch_ind=e["ind"]["chamfer"] if e else None,
                    sil_own=e["own"]["sil"] if e else None, floor=e["floor"]["sil"] if e else None,
                    floor_ch=e["floor"]["chamfer"] if e else None,
                    vol_min=v["min"] if v else None, vol_max=v["max"] if v else None, vol_end=v["at"]["1"] if v else None,
                    vol_t0=v["at"]["0"] if v else None,
                    meshvol_min=min(mv, default=None), meshvol_max=max(mv, default=None),
                    meshvol_end=fr[-1].get("vol_ratio_tgt_mesh") if fr else None,
                    genus_end=fr[-1].get("genus_largest") if fr else None,
                    genus_min=min([g for g in gl if g is not None], default=None),
                    genus_max=max([g for g in gl if g is not None], default=None),
                    genus_sum_end=fr[-1].get("genus_sum_closed") if fr else None,
                    genus_sum_max=max(ints(fr, "genus_sum_closed"), default=None),
                    frames_genus_unknown=sum(1 for g in gl if g is None),
                    comp_end=fr[-1].get("components") if fr else None,
                    comp_max=max([f.get("components", 0) for f in fr], default=None),
                    frames_multi_comp=sum(1 for f in fr if f.get("components", 0) > 1),
                    watertight_end=fr[-1].get("watertight") if fr else None,
                    watertight_frames=sum(1 for f in fr if f.get("watertight")),
                    si_end=fr[-1]["self_intersecting_pairs"] if fr and isinstance(fr[-1].get("self_intersecting_pairs"), int) else None,
                    si_frames=sum(1 for x in ints(fr, "self_intersecting_pairs") if x > 0),
                    si_max=max(ints(fr, "self_intersecting_pairs"), default=None),
                    si_failed=sum(1 for f in fr if isinstance(f.get("self_intersecting_pairs"), str)),
                    t10_genus_max=max([t["genus"] for t in t10 if isinstance(t.get("genus"), int)], default=None),
                    t10_comp_max=max([t["components"] for t in t10 if isinstance(t.get("components"), int)], default=None),
                    wall=run["wall"].get("optimisation_net", run["wall"]["optimisation"]), wall_gross=run["wall"]["optimisation"],
                    wall_total=run["wall"]["total"], recording_split="optimisation_net" in run["wall"],
                    steps=run.get("steps_total"), p2=run.get("params", {}).get("phase2_iterations"),
                    conv=run.get("converged_drop_last20", run.get("converged_drop_last100")),
                    loss_start=run["loss_start"], loss_end=run["loss_end"], harness=harn.get((m, label))))
            if per:
                rows.append(dict(mesh=m, method=method, arm=arm, runs=per))

# ours, from D145's existing files
ours_end = {r["tag"]: r for r in jsonl(f"{O}/ours_endcheck.jsonl")}
ours_vol = {r["mesh"]: r for r in jsonl(f"{G}/d145_vol/vol.jsonl") if r["label"] == "ours_L97"}
ours_tab = {r["name"]: r for r in jsonl(f"{G}/d145_evidence/tab_1015.jsonl") if r.get("dir") == "d145_b1"}
ours_h = {}
for m, lab in OURS_HARNESS.items():
    for r in jsonl(f"{G}/d145_harness/harness_{m}.jsonl"):
        if r["label"] == lab:
            ours_h[m] = r
ours = []
for m in MESHES:
    es = [ours_end.get(f"{m}300k_L{s}") for s in SEEDS]
    ts = [ours_tab.get(f"{m}300k_L{s}") for s in SEEDS]
    ours.append(dict(mesh=m, sil_ind=[e["ind"]["sil"] if e else None for e in es],
                     ch_ind=[e["ind"]["chamfer"] if e else None for e in es],
                     floor=[e["floor"]["sil"] if e else None for e in es], floor_ch=[e["floor"]["chamfer"] if e else None for e in es],
                     minutes=[t["minutes"] if t else None for t in ts], vol=ours_vol.get(m), harness=ours_h.get(m)))

json.dump(dict(rows=rows, ours=ours), open(f"{O}/b12_rows.json", "w"), indent=1)

print("## End and trajectory, min-max over view-set seeds 97 / 99 / 101 (ours: B1's render arm L97 / L99 / L101)\n")
print("| mesh | method (arm) | sil IoU ind. | chamfer ind. | floor sil / chamfer | volume routine: min / max over frames; end "
      "| mesh vol / target mesh: frames; end | genus largest: end (frames) | genus sum closed: end (max) "
      "| components end (max); multi-comp. frames | watertight end; frames /21 | self-int. frames /21 (max pairs); end pairs "
      "| wall s (net) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for m in MESHES:
    for r in [x for x in rows if x["mesh"] == m]:
        p = r["runs"]
        g = lambda k: [x[k] for x in p]  # noqa: E731
        t10 = ""
        if r["method"] == "fc":
            t10 = f"; every 10 it.: genus <= {rng(g('t10_genus_max'), '{:d}')}, comp. <= {rng(g('t10_comp_max'), '{:d}')}"
        print(f"| {m} | {NAMES[r['method']]} ({r['arm']}) | {rng(g('sil_ind'), '{:.4f}')} | {rng(g('ch_ind'), '{:.4f}')} | "
              f"{rng(g('floor'), '{:.4f}')} / {rng(g('floor_ch'), '{:.4f}')} | {rng(g('vol_min'), '{:.3f}')} / {rng(g('vol_max'), '{:.3f}')}; "
              f"{rng(g('vol_end'), '{:.3f}')} | {rng(g('meshvol_min'), '{:.3f}')} .. {rng(g('meshvol_max'), '{:.3f}')}; "
              f"{rng(g('meshvol_end'), '{:.3f}')} | {rng(g('genus_end'), '{:d}')} ({rng(g('genus_min'), '{:d}')} .. "
              f"{rng(g('genus_max'), '{:d}')}) | {rng(g('genus_sum_end'), '{:d}')} ({rng(g('genus_sum_max'), '{:d}')}) | "
              f"{rng(g('comp_end'), '{:d}')} ({rng(g('comp_max'), '{:d}')}); {rng(g('frames_multi_comp'), '{:d}')}{t10} | "
              f"{'/'.join('y' if x else 'n' for x in g('watertight_end'))}; {rng(g('watertight_frames'), '{:d}')} | "
              f"{rng(g('si_frames'), '{:d}')} ({rng(g('si_max'), '{:d}')}); end {rng(g('si_end'), '{:d}')} | {rng(g('wall'), '{:.0f}')} |")
    o = [x for x in ours if x["mesh"] == m][0]
    v = o["vol"]
    vs = f"{v['min']:.3f} / {v['max']:.3f}; {v['at']['1']:.3f} (L97)" if v else "pending (B1)"
    mins = [x for x in o["minutes"] if x is not None]
    print(f"| {m} | **ours** (L) | {rng(o['sil_ind'], '{:.4f}')} | {rng(o['ch_ind'], '{:.4f}')} | {rng(o['floor'], '{:.4f}')} / {rng(o['floor_ch'], '{:.4f}')} | "
          f"{vs} | (particles, no mesh) | - | - | - | - | - | {rng([x * 60 for x in mins], '{:.0f}')} "
          f"({rng(mins, '{:.1f}')} min) |")

print("\n## Large Steps convergence and FlexiCubes loss (min-max over seeds)\n")
print("| mesh | method (arm) | steps | phase-2 iterations | last-window drop | loss start -> end | failed SI checks |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    p = r["runs"]
    g = lambda k: [x[k] for x in p]  # noqa: E731
    print(f"| {r['mesh']} | {NAMES[r['method']]} ({r['arm']}) | {rng(g('steps'), '{:d}')} | {rng(g('p2'), '{:d}')} | "
          f"{rng(g('conv'), '{:.4f}')} | {rng(g('loss_start'), '{:.4f}')} -> {rng(g('loss_end'), '{:.4f}')} | "
          f"{sum(g('si_failed'))} |")

print("\n## D145 harness, seed 97 (progress p = 1 - S(x_t, I) / S(x_0, I); apart particles / occupied volume at p; "
      "n/r = p never reached)\n")
print("| mesh | run | p at the end | raw iterate at p 0.25 / 0.5 / 0.75 / end | apart at p 0.25 / 0.5 / 0.75 / end "
      "| occupied volume at p 0.25 / 0.5 / 0.75 / end |")
print("|---|---|---|---|---|---|")
KEYS = ("p0.25", "p0.5", "p0.75", "end")
for m in MESHES:
    cands = [(f"{NAMES[r['method']]} ({r['arm']})", r["runs"][0]["harness"]) for r in rows if r["mesh"] == m
             and r["runs"] and r["runs"][0]["seed"] == 97]
    if m in OURS_HARNESS:
        cands.append((f"**ours** ({OURS_HARNESS[m]}, D135)", ours_h.get(m)))
    for name, h in cands:
        if not h:
            print(f"| {m} | {name} | - | - | - | - |")
            continue
        rd = h["read"]
        f = lambda k, fmt: " / ".join(fmt.format(rd[x][k]) if rd.get(x) else "n/r" for x in KEYS)  # noqa: E731
        print(f"| {m} | {name} | {h['p_end']:.4f} | {f('raw', '{}')} | {f('apart', '{}')} | {f('occupied_ratio', '{:.3f}')} |")

print("\n## The bunny's superseded first batch (12:07-12:15 CDT) against its re-run (15:2x CDT): end, per seed\n")
print("| run | sil IoU ind. first / re-run | chamfer ind. first / re-run |")
print("|---|---|---|")
for method in ("LS", "FC"):
    for arm in ("a", "b"):
        for s in SEEDS:
            t = f"bunny300k_{method}{arm}{s}"
            a, b = first_ends.get(t), ends.get(t)
            if a is None or b is None or a is b:
                continue
            print(f"| {method}{arm}{s} | {a['ind']['sil']:.4f} / {b['ind']['sil']:.4f} | "
                  f"{a['ind']['chamfer']:.4f} / {b['ind']['chamfer']:.4f} |")
