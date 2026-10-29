"""b13_md1017.py EVDIR -- B13's 300k tables (CPU only): per mesh the default arm L (B1's render arm on the tag) and the
force arms F (--baseline force, scale 1) and FM (force_mf, scale 1/4), seeds 97 / 99 / 101.

Reads (EVDIR = output/gpu/b13/ev1017): tab_all.jsonl (tab145.py + tab145t rows: the 36 force runs and B1's L rows of
d145t_evidence/tab_tx.jsonl), b13_rows.jsonl (b13_read300.py: the force records, seconds an attempt, the geometry
energy), the harness at matched progress (b13/harness/harness_MESH.jsonl, e3h_MESH_LABEL.log; seed 97), the volume
routine (b13/vol_b13.jsonl; seed 97), the visible pieces and the pieces over the run (seed 97 only: the 99 / 101
archives were removed after their readings, as pre-registered).

Cells: the median over the three seeds; for F / FM the change against L's median (for an IoU the change of 1 - IoU) and
* when the two three-run ranges do not overlap (D145's reading). Seed-97 tables: one run an arm, no range, "observed".
--detail prints min / median / max instead. Markdown on stdout."""
import json
import sys
from pathlib import Path

import numpy as np

E = Path(sys.argv[1])
D = E.parent
G = D.parent
DETAIL = "--detail" in sys.argv
MESHES = ["bunny", "dragon", "teapot", "bob", "armadilo", "homer"]
SEEDS = [97, 99, 101]
ARMS = ["L", "F", "FM"]
IOU = {"sil_ind", "sil_own"}


def jl(p):
    try:
        return [json.loads(s) for s in open(p) if s.startswith("{")]
    except OSError:
        return []


rows = {}
for r in jl(E / "tab_all.jsonl"):
    rows[r["name"]] = r
for r in jl(E / "b13_rows.jsonl"):
    rows.setdefault(r["name"], {}).update({k: v for k, v in r.items() if k not in rows.get(r["name"], {})})


def name(m, a, s):
    return f"{m}300k_{a}{s}"


def vals(m, a, k, seeds=SEEDS):
    out = []
    for s in seeds:
        r = rows.get(name(m, a, s))
        if r is not None and r.get(k) is not None:
            out.append(float(r[k]))
    return out


def fmt(v):
    if v is None:
        return "-"
    if v == 0:
        return "0"
    if float(v).is_integer() and abs(v) < 1e6:
        return str(int(v))
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 0.01:
        return f"{v:.4g}" if a < 1 else f"{v:.3g}"
    return f"{v:.2e}"


def change(k, va, vl):
    if not va or not vl:
        return None
    ma, ml = float(np.median(va)), float(np.median(vl))
    if k in IOU:
        ma, ml = 1 - ma, 1 - ml
    return None if ml == 0 else ma / ml - 1


def apart(va, vl):
    return bool(va) and bool(vl) and (min(va) > max(vl) or max(va) < min(vl))


def table(title, meas, seeds=SEEDS, note=None):
    print(f"### {title}\n")
    if note:
        print(note + "\n")
    print("| mesh | arm | " + " | ".join(lab for _, _, lab in meas) + " |")
    print("|---|---|" + "---|" * len(meas))
    for m in MESHES:
        for a in ARMS:
            cells = []
            n = 0
            for k, sc, _ in meas:
                v = vals(m, a, k, seeds)
                n = max(n, len(v))
                if not v:
                    cells.append("-")
                    continue
                if DETAIL:
                    c = f"{fmt(min(v) * sc)} / {fmt(float(np.median(v)) * sc)} / {fmt(max(v) * sc)}"
                else:
                    c = fmt(float(np.median(v)) * sc)
                if a != "L":
                    vl = vals(m, "L", k, seeds)
                    ch = change(k, v, vl)
                    star = "*" if (len(seeds) > 1 and apart(v, vl)) else ""
                    if ch is not None:
                        c += f" ({ch * 100:+.0f} %{star})"
                    elif star:
                        c += " (*)"
                cells.append(c)
            print(f"| {m} | {a} ({n}) | " + " | ".join(cells) + " |")
    print()


print("# B13 at 300k on freeze-2026-10-15: force control against dFc (read 2026-10-17)\n")
print("L = B1's render arm (dFc, the tag's default path); F = `--baseline force` (scale 1); FM = `--baseline force_mf`"
      " (scale 1/4). The force arms carry the same render term, u, objective and optimiser. Seeds 97 / 99 / 101; n per arm"
      " in brackets. Cells: median; for F / FM the change against L's median (an IoU: the change of 1 - IoU), * = the"
      " three-run ranges do not overlap (D145's reading). Units as labelled; a = the display pitch (0.708 x the target"
      " sample's median 8th-neighbour distance).\n")

table("End quality", [
    ("sil_ind", 1, "sil IoU ind"), ("chamfer_ind", 1, "chamfer ind"), ("yard_ext_sil", 1e4, "yard sil e-4"),
    ("yard_ext_pbr", 1e4, "yard shading e-4"), ("disp_front_1miou", 100, "display front 1-IoU %"),
    ("disp_back_1miou", 100, "display back 1-IoU %"), ("thin", 100, "thin %"), ("sil_own", 1, "own sil IoU"),
    ("chamfer_own", 1, "own chamfer")])
table("Momentum and rest (the probe on every kept frame; the records summed over committed windows)", [
    ("com_max_fine_a", 1, "CoM largest a"), ("com_end_fine_a", 1, "CoM end a"),
    ("rotation_deg", 1, "net rotation deg"), ("ang_net_gross_mean", 1e3, "ang net/gross e-3"),
    ("rec_body_net_P", 1, "record body net P"), ("rec_body_rot_deg", 1, "record body rot deg"),
    ("Lstart_max", 1, "L max"), ("Lgrid_sum", 1, "sum dL grid"), ("kin_last", 1e6, "end kin e-6"),
    ("kin_peak", 1, "kin peak")])
table("The momentum the control puts in (B13's force records, summed over committed windows; L: dFc enters as a "
      "stress, whose divergence has zero net force and torque, so no such record)", [
    ("force_dv_path", 1, "sum |dv_com| wu/s"), ("force_dv_net", 1, "|sum dv_com| wu/s"),
    ("force_dL_net", 1, "|sum dL| per mass"), ("force_gross", 1, "sum gross wu/s"),
    ("force_ratio_win_max", 1, "max window dv/gross"), ("force_zero_windows", 1, "windows dv = 0"),
    ("force_windows", 1, "windows"), ("force_absmax", 1, "leaf max")])
table("Floating material over all three seeds (apart particles; pieces off the body at the end, pieces145)", [
    ("apart_w3_10_max", 1, "apart w3-10 max"), ("apart_w3_10_median", 1, "apart w3-10 median"),
    ("pc_end_pieces", 1, "pieces end"), ("pc_end_particles", 1, "particles in pieces end"),
    ("pc_end_largest", 1, "largest piece end"), ("pc_end_farthest_a", 1, "farthest gap end a"),
    ("pc_end_ge10", 1, "end pieces >= 10"), ("pc_end_ge5a", 1, "end pieces >= 5 a")])
table("Floating material over the run, seed 97 (visible_apart, pieces145 every 4th kept frame; one run an arm, observed)", [
    ("vis_w3_10_side_blob", 1, "side blobs w3-10"), ("vis_run_side_blob", 1, "side blobs run"),
    ("vis_run_blob", 1, "blobs run"), ("vis_end_side_blob", 1, "side blobs last window"),
    ("pc_run_largest", 1, "largest piece run"), ("pc_run_farthest_a", 1, "farthest gap run a"),
    ("pc_far_frames", 1, "frames with a piece >= 5 a"), ("pc_far_largest", 1, "largest piece >= 5 a"),
    ("pc_far_particles_max", 1, "far particles a frame max"), ("pc_far10_frames", 1, "frames far piece >= 10"),
    ("pc_far_first_s", 1, "far first s"), ("pc_far_last_s", 1, "far last s")], seeds=[97])
table("Cost and the run", [
    ("windows", 1, "windows"), ("attempts", 1, "attempts"), ("minutes", 1, "minutes"), ("s_attempt", 1, "s / attempt"),
    ("energy_end", 1e5, "geometry energy end e-5"), ("share_c_median", 1, "g_share median"),
    ("lam_c_first", 1, "lambda first"), ("gcos_w1_10", 1, "cos(g_phys, g_render) w1-10")])

# the trajectory: the harness at matched progress (seed 97) and the volume routine
print("### Trajectory at matched progress, seed 97 (D145's harness: progress p on one grid against the independent sample;"
      " time = raw / 240 s; apart at 1.5 a; occupied volume against the sample's own; the display's apart discs / sets"
      " at those frames)\n")
print("| mesh | arm | t p=0.25 s | t p=0.5 s | t p=0.75 s | t end s | p end | apart p.25 / .5 / .75 / end | occupied p.25 / .5 / .75 / end | display apart discs p.25 / .5 / .75 / end | display sets p.25 / .5 / .75 / end |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for m in MESHES:
    hr = {r["label"]: r for r in jl(D / "harness" / f"harness_{m}.jsonl")}
    for a in ARMS:
        r = hr.get(f"{a}97")
        if not r:
            print(f"| {m} | {a} | - |")
            continue
        rd = r["read"]
        ks = ["p0.25", "p0.5", "p0.75", "end"]
        disp = {x.get("state"): x for x in jl(D / "harness" / f"e3h_{m}_{a}97.log")}
        da = " / ".join(fmt(disp.get(str(rd[k]["raw"]), {}).get("apart")) for k in ks)
        ds = " / ".join(fmt(disp.get(str(rd[k]["raw"]), {}).get("sets")) for k in ks)
        print(f"| {m} | {a} | " + " | ".join(f"{rd[k]['raw'] / 240:.3f}" for k in ks[:3]) + f" | {rd['end']['raw'] / 240:.2f}"
              f" | {r['p_end']:.5f} | " + " / ".join(str(rd[k]["apart"]) for k in ks) + " | "
              + " / ".join(f"{rd[k]['occupied_ratio']:.3f}" for k in ks) + f" | {da} | {ds} |")
print()
vol = {}
for r in jl(D / "vol_b13.jsonl"):
    vol[(r.get("mesh"), r.get("label"))] = r
if vol:
    print("### The volume routine, seed 97 (vol145.py: enclosed volume over the target's, same routine for every arm)\n")
    keys = None
    for (m, lab), r in vol.items():
        keys = [k for k in r if k not in ("mesh", "label", "frames", "path")]
        break
    print("| mesh | arm | " + " | ".join(keys) + " |")
    print("|---|---|" + "---|" * len(keys))
    for m in MESHES:
        for a in ARMS:
            r = vol.get((m, f"{a}97"))
            if r:
                print(f"| {m} | {a} | " + " | ".join(
                    json.dumps(r[k]) if not isinstance(r[k], float) else fmt(r[k]) for k in keys) + " |")
    print()
