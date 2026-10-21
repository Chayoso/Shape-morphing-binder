"""read_b9b.py STAGE RUNS_DIR [OUT_PREFIX] -- B9b's reading (pre-registered in docs/experiments.md, D145, "B9b"): the
one-body scene under gravity with the floor (STAGE 40k or 300k: bunny and dragon, render arm L at seeds 97 / 99 / 101 and
the twin P at 97) against B9's single-body runs on the tag and the no-gravity runs. CPU only; reads the files the chains
wrote next to each run:
  TAG.json (scene_run.py: metrics in the body's frame, history, seconds), TAG_track.json (tobody.py), ind40.log
  (gallery_ind.py, 40k) or endgen.jsonl (endgen145t.py, 300k) for the end and the at-B9-time frames in the body frame
  against the independent sample, t3end_TAG.log / e3end_TAG.log (300k: the yardstick and the display), m_TAG.log
  (momentum_probe.py), apart_TAG.txt (D147's apart.py).
The items (each with its pre-registered bar):
  reach     silhouette IoU against the independent sample in the body's frame >= the no-gravity runs' lowest seed - 0.01
            (40k: the tag's V, d145t_b3; 300k: the tag's L, d145t_b1, seeds 97 / 99 / 101; D147's E4 margin)
  height    the end centroid above the floor's centre plane against the target standing there within one pitch, and
            the body upright (tilt of its up axis below V0's 3 deg; option 1 (frame "standing"): a separate item on the
            shape's own tilt, tobody.shape_rotation, at most UPRIGHT below; the Buckens (material) tilt reported)
  rest      the last window's rigid speed (|v_c| and |w| R) below 0.01 wu/s and the lowest centre within one pitch of
            the plane (D147 V1's bar); the deformation's speed and the last second's particle move reported beside B9
  momentum  gravity is vertical and the floor frictionless, so P_x, P_z and L_y (about the centroid) stay at zero: the
            horizontal centre-of-mass drift (their visible effect) below one pitch over the run; L_y against the largest
            horizontal L the floor gives, P_xz against a window of gravity, and the turn about the vertical reported
  pieces    no serious piece (>= 10 particles >= 5 pitches from the body, D145) at the end; over the run reported
  render    the arm against its twin at seed 97 (reported)
  minutes   reported
GATE (40k only): every render arm reaches, ends at the target's height upright, has no serious piece at the end, and
commits every window to the scene's end."""
import json
import sys
from pathlib import Path

import numpy as np

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
E = G / "d145t_evidence"
ARMS = [("L", 97), ("L", 99), ("L", 101), ("P", 97)]
MESHES = ("bunny", "dragon")
B9T = {"40k": {"bunny": {"G2L97": 1280}, "dragon": {"G2L97": 1960}},                       # B9's last delivered frame
       "300k": {"bunny": {"G2L97": 1440, "G2L99": 280, "G2L101": 280, "G2P97": 280},
                "dragon": {"G2L97": 2280, "G2L99": 1920, "G2L101": 1880, "G2P97": 1800}}}
DT = 1.0 / 240.0


def jl(path):
    try:
        return [json.loads(s) for s in open(path) if s.strip().startswith("{")]
    except OSError:
        return []


def last_json(path, pred=lambda r: True):
    rows = [r for r in jl(path) if pred(r)]
    return rows[-1] if rows else None


def b9_refs(stage):
    """B9's single-body runs on the tag and the no-gravity references, per mesh: {arm_seed: row}."""
    out = {m: {"b9": {}, "nog": {}} for m in MESHES}
    if stage == "40k":
        ind = {r["tag"]: r for r in jl(G / "d145t_b9/ind.log")}
        nog = {r["tag"]: r for r in jl(G / "d145t_b3/ind.log")}
        for m in MESHES:
            t = f"{m}40k_G2L97"
            try:
                arm = json.loads((G / f"d145t_b9/{t}.json").read_text())["arms"]["render_full_dt_iso_nn"]
            except (OSError, KeyError, ValueError):
                arm = {}
            h = arm.get("history", [])
            com = [x["com"] for x in h if x.get("outer_accepted") and isinstance(x.get("com"), list)]
            if t in ind:
                out[m]["b9"]["G2L97"] = dict(sil_ind=ind[t]["ind"]["sil"], chamfer_ind=ind[t]["ind"]["chamfer"],
                                             thin=arm.get("metrics", {}).get("thin_uncovered"),
                                             windows=sum(1 for x in h if x.get("outer_accepted")),
                                             minutes=(arm.get("seconds") or 0) / 60, deliver_n=arm.get("deliver_n"),
                                             rec_com_y_end=com[-1][1] if com else None)
            for s in (97, 99, 101):
                k = f"{m}40k_V{s}"
                if k in nog:
                    out[m]["nog"][f"V{s}"] = dict(sil_ind=nog[k]["ind"]["sil"], chamfer_ind=nog[k]["ind"]["chamfer"])
    else:
        rows = {}
        for r in jl(E / "r2/tab_all.jsonl"):
            rows[r["name"]] = r
        for m in MESHES:
            for a in ("G2L97", "G2L99", "G2L101", "G2P97"):
                r = rows.get(f"{m}300k_{a}")
                if r and r.get("dir") == "d145t_b9":
                    out[m]["b9"][a] = r
            for s in (97, 99, 101):
                r = rows.get(f"{m}300k_L{s}")
                if r and r.get("dir") == "d145t_b1":
                    out[m]["nog"][f"L{s}"] = r
    return out


def pieces(path, t_end):
    rows = [r for r in jl(path) if "n" in r]
    serious = [r for r in rows if r["n"] >= 10 and r["to_body_sp"] >= 5.0]
    end = [r for r in rows if abs(r["t"] - t_end) < 1e-6]
    return dict(pieces_frames=len({r["t"] for r in rows}), serious_frames=len({r["t"] for r in serious}),
                serious_first_s=min((r["t"] for r in serious), default=None),
                serious_last_s=max((r["t"] for r in serious), default=None),
                serious_largest=max((r["n"] for r in serious), default=0),
                run_largest=max((r["n"] for r in rows), default=0),
                run_farthest_sp=max((r["to_body_sp"] for r in rows), default=0.0),
                end_pieces=len(end), end_particles=sum(r["n"] for r in end),
                end_farthest_sp=max((r["to_body_sp"] for r in end), default=0.0),
                end_serious=sum(1 for r in end if r["n"] >= 10 and r["to_body_sp"] >= 5.0))


def scene_row(stage, R, mesh, arm, seed):
    tag = f"{mesh}{stage}_{arm}{seed}"
    try:
        j = json.loads((R / f"{tag}.json").read_text())
        tr = json.loads((R / f"{tag}_track.json").read_text())
    except (OSError, ValueError):
        return None
    met, hist, info = j["metrics"][0], j["history"], j["scene"]
    ok = [h for h in hist if "bodies" in h]
    last = ok[-1]["bodies"][0]
    Rb = tr["R_body"]
    M = float(j["config"]["mass_ref_n"])
    g = float(info["g"])
    tw = 2 * j["config"]["T"] * j["mpm"]["dt"]
    end = tr["track"][-1]
    pairs = [p for p in tr["track"] if "move_sp" in p]
    lastsec = [p for p in pairs if p["t"] >= end["t"] - 1.0 - 1e-9]
    r = dict(name=tag, mesh=mesh, arm=arm, seed=seed, windows=len(ok), windows_planned=info["windows"],
             invalid=any(h.get("invalid") for h in hist), minutes=j["seconds"] / 60,
             s_per_window=j["seconds"] / max(len(ok), 1), t_end=end["t"], g=g,
             sil_own=met["sil_iou"], chamfer_own=met["chamfer"], thin=met.get("thin_uncovered"),
             dh_wu=end["dh_wu"], dh_sp=end["dh_sp"], lowest_sp=end["lowest_sp"], tilt_deg=end["tilt_deg"],
             turn_deg=end["turn_deg"], yaw_deg=end["yaw_deg"],
             drop_min_wu=min(p["dh_wu"] for p in tr["track"]),
             drop_min_t=min(tr["track"], key=lambda p: p["dh_wu"])["t"],
             drift_end_sp=end["drift_xz_sp"], drift_max_sp=max(p["drift_xz_sp"] for p in tr["track"]),
             yaw_max_deg=max(abs(p["yaw_deg"]) for p in tr["track"]),
             tilt_max_deg=max(p["tilt_deg"] for p in tr["track"]),
             shape_tilt_deg=end.get("shape_tilt_deg"),
             shape_tilt_max_deg=max((p["shape_tilt_deg"] for p in tr["track"] if "shape_tilt_deg" in p
                                     and p["t"] >= 0.5 * end["t"]), default=None),
             v_com_end=float(np.linalg.norm(last["v_com"])), omegaR_end=float(np.linalg.norm(last["omega"])) * Rb,
             v_def_rms_end=pairs[-1]["v_def_rms"] if pairs else None,
             move_lastsec_sp=float(np.mean([p["move_sp"] for p in lastsec])) if lastsec else None,
             Pxz_max=max(float(np.hypot(h["bodies"][0]["P"][0], h["bodies"][0]["P"][2])) for h in ok),
             Ly_max=max(abs(float(h["bodies"][0]["L"][1])) for h in ok),
             Lxz_max=max(float(np.hypot(h["bodies"][0]["L"][0], h["bodies"][0]["L"][2])) for h in ok), P_window=M * g * tw,
             Py_end=float(last["P"][1]), pitch=tr["pitch"],
             lam_median=float(np.median([h["lambda"] for h in ok if h.get("lambda") is not None] or [np.nan])),
             g_share_median=float(np.median([h["g_share"] for h in ok if h.get("g_share") is not None] or [np.nan])),
             d_sil_end=float(last["d_sil"]), d_pbr_end=float(last["d_pbr"]),
             d_sil=[float(h["bodies"][0]["d_sil"]) for h in ok],
             replay_retries=sum(len(h.get("replay_retries") or []) for h in ok),
             zero_commits=sum(1 for h in ok if h.get("zero_commit")))
    if stage == "40k":
        ind = {x["tag"]: x for x in jl(R / "ind40.log")}
    else:
        ind = {x["run"]: x for x in jl(R / "endgen.jsonl")}
    if tag in ind:
        r.update(sil_ind=ind[tag]["ind"]["sil"], chamfer_ind=ind[tag]["ind"]["chamfer"])
    for T, a in tr.get("at", {}).items():
        k = a["file"].replace(".npz", "")
        if k in ind:
            r.setdefault("at", {})[T] = dict(t=a["t"], sil_ind=ind[k]["ind"]["sil"], chamfer_ind=ind[k]["ind"]["chamfer"])
    m = R / f"m_{tag}.log"
    try:
        txt = m.read_text()
        r["move_last20_a"] = float(txt.split("last 20:")[-1].split("a")[0]) if "last 20:" in txt else None
    except OSError:
        pass
    if stage == "300k":
        t3 = last_json(R / f"t3end_{tag}.log", lambda x: "exterior" in x)
        if t3:
            r.update(yard_ext_sil=t3["exterior"]["sil"], yard_ext_pbr=t3["exterior"]["pbr"])
        e3 = last_json(R / f"e3end_{tag}.log", lambda x: "to_target" in x and x.get("state") not in ("target", "own"))
        if e3:
            r.update(disp_front_1miou=1.0 - e3["to_target"]["front"]["iou"])
    r.update(pieces(R / f"apart_{tag}.txt", end["t"]))
    return r


# the upright bar on the shape's tilt (tobody.shape_rotation; option 1's pre-registration, 2026-10-20): V0's 3 deg, or the
# single-body path's own reading of an upright end where larger (B9 G2L97 at 300k: dragon 0.75, bunny 3.10 deg)
UPRIGHT = {"bunny": 3.0, "dragon": 3.0}       # corrected 2026-10-21: B9 G2L97 converged reads 0.22 / 0.17 deg (the
                                              # 3.10 / 0.75 were the unconverged measure's), so V0's 3 deg for both


def verdicts(r, bar):
    v = {}
    v["reach"] = r.get("sil_ind") is not None and r["sil_ind"] >= bar
    if r.get("shape_tilt_deg") is not None:         # option 1: height and upright apart, upright on the shape's tilt
        v["height"] = abs(r["dh_sp"]) <= 1.0
        v["upright"] = r["shape_tilt_deg"] <= UPRIGHT.get(r["mesh"], 3.0)
    else:                                           # B9b as pre-registered: one item, the Buckens frame's tilt
        v["height"] = abs(r["dh_sp"]) <= 1.0 and r["tilt_deg"] < 3.0
    v["rest"] = r["v_com_end"] < 0.01 and r["omegaR_end"] < 0.01 and abs(r["lowest_sp"]) <= 1.0
    v["momentum"] = r["drift_max_sp"] < 1.0
    v["pieces"] = r["end_serious"] == 0
    v["whole"] = (not r["invalid"]) and r["windows"] == r["windows_planned"]
    return v


def fmt(v, spec=".4g"):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "NO"
    if isinstance(v, str):
        return v
    return format(v, spec)


def rng(vals, spec=".4f"):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    return f"{min(vals):{spec}} / {np.median(vals):{spec}} / {max(vals):{spec}}"


def overlap(a, b):
    a, b = [v for v in a if v is not None], [v for v in b if v is not None]
    return bool(a and b and min(a) <= max(b) and min(b) <= max(a))


def main():
    stage, R = sys.argv[1], Path(sys.argv[2])
    prefix = sys.argv[3] if len(sys.argv) > 3 else str(R / f"read_{stage}")
    refs = b9_refs(stage)
    rows, lines, gate_fail = [], [], []
    title = sys.argv[4] if len(sys.argv) > 4 else "one body under gravity with the floor, the scene path"
    lines += [f"# B9b {stage}: {title}", "",
              "Body-frame measures against the independent sample; the height against the target standing on the "
              "floor; pitches are the sample's (vp^(1/3)). L = render arm, P = render-off twin.", ""]
    for mesh in MESHES:
        nog = [x["sil_ind"] for x in refs[mesh]["nog"].values()]
        bar = (min(nog) - 0.01) if nog else None
        lines += [f"## {mesh} {stage}", "", f"Reach bar (no-gravity tag runs' lowest seed - 0.01): {fmt(bar, '.4f')} "
                  f"(no-gravity silhouette IoU ind {rng(nog)})", ""]
        hdr = ["run", "sil IoU ind", "chamfer ind", "thin %", "end height vs target (sp)", "material tilt deg",
               "shape tilt deg (2nd half max)",
               "lowest (sp)", "drop min (wu, s)", "|v_c| wu/s", "|w|R wu/s", "def. speed wu/s", "move last s (sp)",
               "drift max (sp)", "yaw max deg", "max P_xz / (M g window)", "max L_y / max L_xz", "serious frames / at end",
               "largest / farthest (sp)", "windows", "min", "lambda med", "g_share med"]
        if stage == "300k":
            hdr[4:4] = ["yard sil e-4", "yard shading e-4", "display front %"]
        lines += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        vt = ["| run | reach | height | upright | rest | momentum | pieces | whole run |", "|---|---|---|---|---|---|---|---|"]
        for arm, seed in ARMS:
            r = scene_row(stage, R, mesh, arm, seed)
            if r is None:
                lines.append(f"| {mesh}{stage}_{arm}{seed} | (missing) |")
                if arm == "L":
                    gate_fail.append(f"{mesh}{stage}_{arm}{seed} missing")
                continue
            r["bar"] = bar
            v = verdicts(r, bar if bar is not None else 1.0)
            r["verdicts"] = v
            rows.append(r)
            cells = [r["name"], fmt(r.get("sil_ind"), ".4f"), fmt(r.get("chamfer_ind"), ".5f"),
                     fmt(None if r["thin"] is None else 100 * r["thin"], ".2f"), fmt(r["dh_sp"], "+.2f"),
                     fmt(r["tilt_deg"], ".2f"),
                     "-" if r.get("shape_tilt_deg") is None else f"{r['shape_tilt_deg']:.2f} ({r['shape_tilt_max_deg']:.2f})",
                     fmt(r["lowest_sp"], "+.2f"),
                     f"{r['drop_min_wu']:+.3f}, {r['drop_min_t']:.2f}", fmt(r["v_com_end"], ".2e"),
                     fmt(r["omegaR_end"], ".2e"), fmt(r["v_def_rms_end"], ".2e"), fmt(r["move_lastsec_sp"], ".4f"),
                     fmt(r["drift_max_sp"], ".2f"), fmt(r["yaw_max_deg"], ".2f"),
                     fmt(r["Pxz_max"] / r["P_window"], ".2e"), fmt(r["Ly_max"] / max(r["Lxz_max"], 1e-30), ".2e"),
                     f"{r['serious_frames']} / {r['end_serious']}",
                     f"{r['run_largest']} / {r['run_farthest_sp']:.1f}", f"{r['windows']} / {r['windows_planned']}",
                     fmt(r["minutes"], ".1f"), fmt(r["lam_median"], ".3g"), fmt(r["g_share_median"], ".3f")]
            if stage == "300k":
                cells[4:4] = [fmt(None if r.get("yard_ext_sil") is None else 1e4 * r["yard_ext_sil"], ".2f"),
                              fmt(None if r.get("yard_ext_pbr") is None else 1e4 * r["yard_ext_pbr"], ".2f"),
                              fmt(None if r.get("disp_front_1miou") is None else 100 * r["disp_front_1miou"], ".3f")]
            lines.append("| " + " | ".join(cells) + " |")
            vt.append("| " + " | ".join([r["name"]] + [fmt(v.get(k, "-")) for k in ("reach", "height", "upright", "rest", "momentum",
                                                                             "pieces", "whole")]) + " |")
            if arm == "L":
                for k in ("reach", "height", "upright", "pieces", "whole"):
                    if k in v and not v[k]:
                        gate_fail.append(f"{r['name']} {k}")
        lines += ["", "Items by their bars:", ""] + vt + [""]
        L97 = next((r for r in rows if r["name"] == f"{mesh}{stage}_L97"), None)
        P97 = next((r for r in rows if r["name"] == f"{mesh}{stage}_P97"), None)
        if L97 and P97:
            n = min(len(L97["d_sil"]), len(P97["d_sil"]))
            ratio = np.array(L97["d_sil"][:n]) / np.maximum(np.array(P97["d_sil"][:n]), 1e-30)
            ahead = float((ratio < 1.0).mean()) if n else float("nan")
            lines += [f"Render lead at seed 97 (arm / twin): silhouette term at the end {L97['d_sil_end']:.3g} / "
                      f"{P97['d_sil_end']:.3g} (x{L97['d_sil_end'] / max(P97['d_sil_end'], 1e-30):.2f}), median over "
                      f"windows x{float(np.median(ratio)) if n else float('nan'):.2f}, the arm lower in "
                      f"{100 * ahead:.0f} % of the windows; "
                      f"shading term {L97['d_pbr_end']:.3g} / {P97['d_pbr_end']:.3g}; silhouette IoU ind "
                      f"{fmt(L97.get('sil_ind'), '.4f')} / {fmt(P97.get('sil_ind'), '.4f')}"
                      + ("" if stage != "300k" else
                         f"; yardstick silhouette {fmt(L97.get('yard_ext_sil'), '.3e')} / {fmt(P97.get('yard_ext_sil'), '.3e')}, "
                         f"shading {fmt(L97.get('yard_ext_pbr'), '.3e')} / {fmt(P97.get('yard_ext_pbr'), '.3e')}, display "
                         f"front 1-IoU {fmt(L97.get('disp_front_1miou'), '.4f')} / {fmt(P97.get('disp_front_1miou'), '.4f')}"), ""]
        # against B9's single-body runs
        b9 = refs[mesh]["b9"]
        sc = [r for r in rows if r["mesh"] == mesh and r["arm"] == "L"]
        lines += [f"Against B9 (single-body path, tag, floor + G 0.02) and the no-gravity runs ({mesh} {stage}):", "",
                  "| measure | scene L (min / median / max) | B9 G2L | B9 twin G2P97 | no gravity | scene vs B9 G2L |",
                  "|---|---|---|---|---|---|"]
        keys = [("sil_ind", ".4f"), ("chamfer_ind", ".5f"), ("thin", ".4f"), ("minutes", ".1f")]
        if stage == "300k":
            keys[3:3] = [("yard_ext_sil", ".3e"), ("yard_ext_pbr", ".3e"), ("disp_front_1miou", ".4f"),
                         ("move_last20_a", ".4f")]
        for k, sp in keys:
            a = [r.get(k) for r in sc]
            bl = [x.get(k) for n, x in b9.items() if n.startswith("G2L")]
            bp = [x.get(k) for n, x in b9.items() if n.startswith("G2P")]
            ng = [x.get(k) for x in refs[mesh]["nog"].values()]
            lines.append(f"| {k} | {rng(a, sp)} | {rng(bl, sp)} | {rng(bp, sp)} | {rng(ng, sp)} | "
                         + ("-" if not [x for x in a if x is not None] or not [x for x in bl if x is not None]
                            else "overlap" if overlap(a, bl) else "APART") + " |")
        lines.append("")
        bt = B9T[stage][mesh]
        at_rows = []
        for n, f in bt.items():
            s = int(n[3:])
            r = next((x for x in rows if x["mesh"] == mesh and x["arm"] == n[2] and x["seed"] == s), None)
            ref = b9.get(n, {})
            a = (r or {}).get("at", {}).get(f"{f * DT:g}")
            at_rows.append(f"| {n} at {f * DT:.2f} s | {fmt(ref.get('sil_ind'), '.4f')} | "
                           f"{fmt(None if a is None else a['sil_ind'], '.4f')} | {fmt(ref.get('chamfer_ind'), '.5f')} | "
                           f"{fmt(None if a is None else a['chamfer_ind'], '.5f')} |")
        lines += ["At B9's own end time (the scene's kept frame nearest it):", "",
                  "| B9 run, its end | B9 sil IoU ind | scene, same seed and arm | B9 chamfer ind | scene |",
                  "|---|---|---|---|---|"] + at_rows + [""]
        if stage == "300k":                 # B9's G2L97 read in its own Buckens frame (b9body.py): the frame's effect
            eg = {x["run"]: x for x in jl(R / "endgen.jsonl")}.get(f"b9_{mesh}300k_G2L97_body")
            bb = last_json(R / f"b9body_{mesh}.txt")
            w = b9.get("G2L97", {})
            if eg and bb:
                lines += [f"B9 G2L97 read as the scene reads its body: silhouette IoU ind {eg['ind']['sil']:.4f} in its "
                          f"Buckens frame against {fmt(w.get('sil_ind'), '.4f')} in the world against the lifted sample; "
                          f"chamfer {eg['ind']['chamfer']:.5f} against {fmt(w.get('chamfer_ind'), '.5f')}; its end turned "
                          f"{bb['turn_deg']:.2f} deg (tilt {bb['tilt_deg']:.2f}, about the vertical {bb['yaw_deg']:+.2f}), "
                          f"horizontal drift {bb['drift_xz_a']:.2f} pitches, height against the standing target "
                          f"{bb['dh_wu']:+.4f} wu, lowest centre {bb['lowest_a']:+.2f} pitches from the floor.", ""]
    if stage == "40k":
        lines.append("GATE 40k: " + ("PASS" if not gate_fail else "FAIL (" + "; ".join(gate_fail) + ")"))
    text = "\n".join(lines) + "\n"
    print(text)
    Path(prefix + ".md").write_text(text)
    with open(prefix + ".jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
