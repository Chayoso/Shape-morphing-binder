"""tab145t.py IN_JSONL OUT_JSONL [--relabel DIR=PREFIX ...] -- D145 post-tag: tab145.py's rows with the post-tag readings
added (CPU only; nothing recomputed, only read from the files next to each run and in output/gpu/d145t_evidence).

Added per run (missing pieces stay null):
  sil_ind, chamfer_ind, floor_sil   endgen145.py on the end frame (d145t_evidence/endgen_DIR.jsonl): silhouette IoU and
                                    chamfer against the independent 300k sample; the own-against-independent floor
  vis_*                             visible_apart.py: unique over the whole run (every 4th kept frame) and over windows
                                    3-10: apart / blob / side_blob; the last read window's blob and side_blob
  pc_*                              pieces145.py (d145t_evidence/pieces_DIR.jsonl, pieces_full_DIR.jsonl): pieces off
                                    the body at the end
                                    (count, largest, farthest gap in pitches, >= 10 particles, >= 5 pitches) and over the
                                    run (largest, farthest; the far pieces >= 5 pitches: frames, largest, particles)
  share_*, lam_*, gcos_w1_10        the record over committed windows: g_share median / min / max, lambda first / median,
                                    cos(g_phys, g_render) median over windows 1-10, after PCGrad (g_cos) and
                                    before it (g_raw_cos)
  kin_peak                          the largest kinetic record
  rec_*_net_P, rec_*_rot_deg        the record's per-update vectors summed over committed windows (D149's columns): net
                                    translation of the spacing push / grid / body / u / relaxation in pitches (the
                                    pieces' or visible_apart's pitch), net rotation of the push and the body in degrees
  Lstart_*, L*_sum                  angular momentum per unit mass: |L_start| largest / median over windows; sum over
                                    windows of |dL| by update (grid, spacing push, u, relaxation)
  rev_3_7, rev_7_15, rev_15         the record's window-to-window reversal cosine (all particles), median over committed
                                    windows ending in 3-7 s, 7-15 s and after 15 s (frame_end / 240; D146 (6))
  com_max_fine_a, com_end_fine_a    mom.sh's full-precision rows: centre-of-mass displacement, largest and at the end
  pulse_*                           d116_pulse.py on the last 30 % of kept frames (bunny, dragon): in-window normal
                                    excursion rms, net, returned share, the driven / released correlations
--relabel DIR=PREFIX renames that directory's runs MESH300k_ARMSEED -> MESH300k_PREFIXARMSEED (e.g. the pre-tag B1 as
OL / OP, so band145t.py can set them against the tag's L / P)."""
import json
import re
import sys
from pathlib import Path

import numpy as np

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
E = G / "d145t_evidence"


def jl(path):
    try:
        return [json.loads(s) for s in open(path) if s.startswith("{")]
    except OSError:
        return []


args = sys.argv[1:]
relabel = {}
while "--relabel" in args:
    i = args.index("--relabel")
    d, p = args[i + 1].split("=")
    relabel[d] = p
    del args[i:i + 2]
rows = jl(args[0])
endgen, pcs = {}, {}
pre_vis = {"full": jl(G / "d145_evidence/vis_full.jsonl"), "w3_10": jl(G / "d145_evidence/vis_w3_10.jsonl")}
out = []
for r in rows:
    d, name = r["dir"], r["name"]
    D = G / d
    if d not in endgen:
        endgen[d] = {e["run"].replace("_end1.npz", ""): e for e in jl(E / f"endgen_{d}.jsonl")}
        pcs[d] = {p["run"]: p for p in jl(E / f"pieces_{d}.jsonl")}          # end frames (end1)
        pcs[d].update({p["run"]: p for p in jl(E / f"pieces_full_{d}.jsonl")})  # kept frames (frames12), where kept
    e = endgen[d].get(name)
    if e:
        r.update(sil_ind=e["ind"]["sil"], chamfer_ind=e["ind"]["chamfer"], floor_sil=e["floor"])
    for rng, fname in (("full", f"vis_full_{name}.jsonl"), ("w3_10", f"vis_w3_10_{name}.jsonl")):
        v = jl(D / fname)
        if not v:                                   # the pre-tag readings sit in d145_evidence, one file for all runs
            v = [x for x in pre_vis[rng] if x.get("run") == name and f"/{d}/" in x.get("path", "")]
        if v and v[0].get("pitch"):
            r["_pitch"] = v[0]["pitch"]
        rr = [x for x in v if x.get("window") == "range"]
        if rr:
            tag = "run" if rng == "full" else "w3_10"
            r.update({f"vis_{tag}_{k}": rr[-1][k] for k in ("apart", "blob", "side_blob")})
        ww = [x for x in v if x.get("window") != "range"]
        if rng == "full" and ww:
            r.update(vis_end_blob=ww[-1]["blob"], vis_end_side_blob=ww[-1]["side_blob"], vis_end_window=ww[-1]["window"])
    p = pcs[d].get(name)
    if p:
        r.update(pc_end_pieces=p["end_pieces"], pc_end_particles=p["end_particles"], pc_end_largest=p["end_largest"],
                 pc_end_farthest_a=p["end_farthest_a"], pc_end_ge10=p["end_ge10"], pc_end_ge5a=p["end_ge5a"],
                 pc_end_listed=p["end_listed"])
        if p["frames_read"] > 1:
            r.update(pc_run_largest=p["run_largest"], pc_run_farthest_a=p["run_farthest_a"],
                     pc_run_largest_s=p["run_largest_s"], pc_run_farthest_s=p["run_farthest_s"])
            if "far_frames" in p:
                r.update(pc_far_frames=p["far_frames"], pc_far_largest=p["far_largest"],
                         pc_far_particles_max=p["far_particles_max"], pc_far10_frames=p["far10_frames"],
                         pc_far_first_s=p["far_first_s"], pc_far_last_s=p["far_last_s"])
    try:
        hist = json.loads((D / f"{name}.json").read_text())["arms"]["render_full_dt_iso_nn"]["history"]
        com = [h for h in hist if h.get("outer_accepted")]
        sh = sorted(h["g_share"] for h in com if h.get("g_share") is not None)
        lam = [h["lambda"] for h in com if h.get("lambda") is not None]
        gc = sorted(h["g_cos"] for h in com[1:11] if h.get("g_cos") is not None)
        gq = sorted(h["g_raw_cos"] for h in com[1:11] if h.get("g_raw_cos") is not None)
        kin = [h["kin"] for h in hist if h.get("kin") is not None]
        if sh:
            r.update(share_c_median=sh[len(sh) // 2], share_c_min=sh[0], share_c_max=sh[-1])
        if lam:
            r.update(lam_c_first=lam[0], lam_c_median=sorted(lam)[len(lam) // 2])
        if gc:
            r.update(gcos_w1_10=gc[len(gc) // 2])
        if gq:
            r.update(graw_cos_w1_10=gq[len(gq) // 2])
        if kin:
            r.update(kin_peak=max(kin))
        pitch = (p or {}).get("pitch") or r.get("_pitch")
        if pitch:                                   # the record's sums by update (D149's columns), in pitches
            def net(key):
                v = [h[key] for h in com if isinstance(h.get(key), list)]
                return float(np.linalg.norm(np.sum(v, 0))) if v else None
            for key, col in (("spacing_vcom", "rec_spacing_net_P"), ("grid_vcom", "rec_grid_net_P"),
                             ("body_vcom", "rec_body_net_P"), ("u_vcom", "rec_u_net_P"),
                             ("relax_vcom", "rec_relax_net_P")):
                v = net(key)
                if v is not None:
                    r[col] = v / pitch
            for key, col in (("spacing_vrot", "rec_spacing_rot_deg"), ("body_vrot", "rec_body_rot_deg")):
                v = net(key)
                if v is not None:
                    r[col] = float(np.degrees(v))
        Ls = [float(np.linalg.norm(h["L_start"])) for h in com if isinstance(h.get("L_start"), list)]
        if Ls:
            r.update(Lstart_max=max(Ls), Lstart_median=sorted(Ls)[len(Ls) // 2])
        for key, col in (("L_grid", "Lgrid_sum"), ("L_space", "Lspace_sum"), ("L_u", "Lu_sum"), ("L_relax", "Lrelax_sum")):
            v = [float(np.linalg.norm(h[key])) for h in com if isinstance(h.get(key), list)]
            if v:
                r[col] = sum(v)
        for lo, hi, col in ((3, 7, "rev_3_7"), (7, 15, "rev_7_15"), (15, 1e9, "rev_15")):
            v = sorted(h["reversal_cos"] for h in com if h.get("reversal_cos") is not None and h.get("frame_end")
                       and lo <= h["frame_end"] / 240.0 < hi)
            if v:
                r[col] = v[len(v) // 2]
    except (OSError, KeyError, ValueError):
        pass
    mrows = [x for x in jl(D / f"m_{name}.log") if "com" in x]
    if mrows:                                       # mom.sh's full-precision rows (pitches), every kept frame
        r.update(com_max_fine_a=max(x["com"] for x in mrows), com_end_fine_a=mrows[-1]["com"],
                 rot_end_fine_deg=mrows[-1].get("rot"))
    try:
        pl = [json.loads(s) for s in (D / f"pulse_{name}.log").read_text().splitlines() if s.startswith("{")]
        if pl:
            q = pl[-1]
            r.update(pulse_exc=q.get("excursion_rms_med"), pulse_net=q.get("net_rms_med"),
                     pulse_ret=q.get("returned_share_med"), pulse_corr_d=q.get("corr_driven_mid"),
                     pulse_corr_r=q.get("corr_released_mid"), pulse_windows=q.get("windows"))
    except OSError:
        pass
    if d in relabel:
        m = re.match(r"(.+?300k_)(.*)$", name)
        r["name"] = m.group(1) + relabel[d] + m.group(2)
    out.append(r)
with open(args[1], "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(f"{len(out)} rows -> {args[1]}")
