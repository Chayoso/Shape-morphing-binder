"""B12 (D145) summary: one row per mesh x method x arm over the three view-set seeds (min-max), from the runs' run.json,
frames_check.json, B11's end check (endcheck.jsonl) and D145's volume routine (vol.jsonl); harness rows when present.
Writes output/gpu/d145_b12/b12_rows.json and prints a markdown table.  python3 b12_summary.py"""
import json
import os

B12 = "/data/relcfd/chayo/physmorph_v2/baselines/b12"
O = "/data/relcfd/chayo/physmorph_v2/output/gpu/d145_b12"
MESHES = ("bunny", "dragon", "teapot", "bob", "armadilo", "homer")
SEEDS = (97, 99, 101)
NAMES = dict(ls="Large Steps", fc="FlexiCubes")


def jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(x) for x in open(path) if x.strip()]


ends = {r["tag"]: r for r in jsonl(f"{O}/endcheck.jsonl")}
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
                per.append(dict(
                    seed=s, label=label,
                    sil_ind=e["ind"]["sil"] if e else None, ch_ind=e["ind"]["chamfer"] if e else None,
                    sil_own=e["own"]["sil"] if e else None, floor=e["floor"]["sil"] if e else None,
                    floor_ch=e["floor"]["chamfer"] if e else None,
                    vol_min=v["min"] if v else None, vol_max=v["max"] if v else None, vol_end=v["at"]["1"] if v else None,
                    genus_end=fr[-1].get("genus_largest") if fr else None,
                    genus_min=min([g for g in gl if g is not None], default=None),
                    genus_max=max([g for g in gl if g is not None], default=None),
                    frames_genus_unknown=sum(1 for g in gl if g is None),
                    comp_end=fr[-1].get("components") if fr else None,
                    comp_max=max([f.get("components", 0) for f in fr], default=None),
                    watertight_end=fr[-1].get("watertight") if fr else None,
                    watertight_frames=sum(1 for f in fr if f.get("watertight")),
                    si_end=fr[-1].get("self_intersecting_pairs") if fr else None,
                    si_frames=sum(1 for f in fr if isinstance(f.get("self_intersecting_pairs"), int) and f["self_intersecting_pairs"] > 0),
                    si_max=max([f["self_intersecting_pairs"] for f in fr if isinstance(f.get("self_intersecting_pairs"), int)], default=None),
                    meshvol_end=fr[-1].get("vol_ratio_tgt_mesh") if fr else None,
                    wall=run["wall"]["optimisation"], wall_total=run["wall"]["total"],
                    steps=run.get("steps_total"), p2=run.get("params", {}).get("phase2_iterations"),
                    conv=run.get("converged_drop_last20", run.get("converged_drop_last100")),
                    loss_end=run["loss_end"], harness=harn.get((m, label))))
            if per:
                rows.append(dict(mesh=m, method=method, arm=arm, runs=per))
json.dump(rows, open(f"{O}/b12_rows.json", "w"), indent=1)

print("| mesh | method, arm | sil IoU ind. | chamfer ind. | floor | volume (routine) min-max / end | genus largest: end (range over frames) | components end (max) | watertight end, frames | self-int. frames (max pairs) | wall s |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    p = r["runs"]
    g = lambda k: [x[k] for x in p]  # noqa: E731
    print(f"| {r['mesh']} | {NAMES[r['method']]} ({r['arm']}) | {rng(g('sil_ind'), '{:.4f}')} | {rng(g('ch_ind'), '{:.4f}')} | "
          f"{rng(g('floor'), '{:.4f}')} | {rng(g('vol_min'), '{:.3f}')} / {rng(g('vol_max'), '{:.3f}')}; end {rng(g('vol_end'), '{:.3f}')} | "
          f"{rng(g('genus_end'), '{:d}')} ({rng(g('genus_min'), '{:d}')} .. {rng(g('genus_max'), '{:d}')}) | "
          f"{rng(g('comp_end'), '{:d}')} ({rng(g('comp_max'), '{:d}')}) | {'/'.join(str(x) for x in g('watertight_end'))}, "
          f"{rng(g('watertight_frames'), '{:d}')}/21 | {rng(g('si_frames'), '{:d}')} ({rng(g('si_max'), '{:d}')}) | "
          f"{rng(g('wall'), '{:.0f}')} |")
