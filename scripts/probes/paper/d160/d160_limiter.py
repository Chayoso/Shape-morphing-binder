"""d160_limiter.py OUT_JSON NAME [NAME ...] -- D160 step 1, part A (CPU, read-only): what limits the colour after
arrival, from each run's window records (output/gpu/d160/runs40/NAME.json and read_colour.jsonl's timeline).

Arrival: the first committed window whose move (mean |x_end - x_start|) is at most 5 % of the run's largest.
After it, per run: the windows committed / rejected / null; lambda, |g_phys|, lambda |g_render| (= alpha_lam |g_phys|
at calibration) at arrival and at the end; the search length (alpha_last), the anneal, the line-search merit failures,
the per-window move (pitches); the outer decisions: for every window judged after arrival, the change of the merit's
physics part P = merit - lambda (sil + pbr) and of its render part R = lambda (sil + pbr) from the last accepted window
(both read with the judged window's lambda, as the selection rescores), and which part made a rejected window worse;
the colour term's (d_pbr) fall after arrival; the stop: patience (stale >= 5), reject_stop (3 in a row), or budget.
FPS 240; pitch from read_colour.jsonl."""
import json
import sys

import numpy as np

O = "/data/relcfd/chayo/physmorph_v2/output/gpu/d160"
col = {}
for s in open(f"{O}/read_colour.jsonl"):
    r = json.loads(s)
    col[r["name"]] = r
rows = []
for name in sys.argv[2:]:
    a = next(iter(json.load(open(f"{O}/runs40/{name}.json"))["arms"].values()))
    cfg = a["config"]
    H = a["history"]
    pitch = col[name]["pitch"]
    com = [h for h in H if h.get("frame_end") and not h.get("null_commit")]
    mv = np.array([h["move"] for h in com])
    ia = int(np.argmax(mv <= 0.05 * mv.max()))
    t_arr = com[ia]["frame_end"] / 240.0
    after = [h for h in H if h.get("animation", -1) >= com[ia]["animation"]]
    judged, prev = [], None
    for h in H:
        if h.get("selection_merit") is None or not np.isfinite(h.get("selection_merit") or np.nan):
            continue
        lam = h.get("lambda") or 0.0
        R = lam * ((h.get("d_render") or 0.0) + cfg["w_pbr"] * (h.get("d_pbr") or 0.0))
        P = h["selection_merit"] - R
        if prev is not None and h["animation"] >= com[ia]["animation"]:
            Rp = lam * ((prev.get("d_render") or 0.0) + cfg["w_pbr"] * (prev.get("d_pbr") or 0.0))
            Pp = prev["selection_merit"] - (prev.get("lambda") or 0.0) * ((prev.get("d_render") or 0.0)
                                                                          + cfg["w_pbr"] * (prev.get("d_pbr") or 0.0))
            acc = bool(h.get("outer_accepted", 1)) and not h.get("null_commit")
            judged.append(dict(a=h["animation"], accepted=acc, dP=(P - Pp) / max(abs(Pp + Rp), 1e-30),
                               dR=(R - Rp) / max(abs(Pp + Rp), 1e-30), latched=h.get("outer_gate_latched"),
                               gain=h.get("outer_gain"), d_pbr=h.get("d_pbr"), d_sil=h.get("d_render")))
        if bool(h.get("outer_accepted", 1)) and h.get("frame_end") and not h.get("null_commit"):
            prev = h
    rej = [j for j in judged if not j["accepted"]]
    last = H[-1]
    stale = max((h.get("stale") or 0) for h in H[-3:])
    streak = max((h.get("reject_streak") or 0) for h in H[-3:])
    stop = ("reject_stop" if streak >= cfg["reject_stop"] else "patience" if stale >= cfg["patience"]
            else "budget" if len(H) >= cfg["animations"] else "other")
    ca = [h for h in com if h["animation"] >= com[ia]["animation"]]
    tl = col[name].get("timeline") or []
    mis_arr = min(tl, key=lambda p: abs(p[0] - t_arr))[1] if tl else None
    row = dict(name=name, t_arrival=t_arr, windows=len(com), windows_after=len(ca),
               rejected_after=len(rej), null_after=sum(1 for h in after if h.get("null_commit") and h.get("outer_accepted") is None),
               lam_arr=com[ia].get("lambda"), lam_end=com[-1].get("lambda"),
               gphys_arr=com[ia].get("g_phys_norm"), gphys_end=com[-1].get("g_phys_norm"),
               grend_arr=com[ia].get("g_rend_norm"), grend_end=com[-1].get("g_rend_norm"),
               alpha_med_after=float(np.median([h["alpha_last"] for h in ca])),
               anneal_med_after=float(np.median([h["anneal"] for h in ca])),
               ls_fail_merit_after=int(sum(h.get("ls_fail_merit") or 0 for h in ca)),
               ls_trials_after=int(sum(h.get("ls_trials") or 0 for h in ca)),
               move_med_after_pitch=float(np.median([h["move"] for h in ca]) / pitch),
               pbr_arr=com[ia]["d_pbr"], pbr_end=com[-1]["d_pbr"], sil_arr=com[ia]["d_render"], sil_end=com[-1]["d_render"],
               render_share_merit_end=(com[-1]["lambda"] or 0) * (com[-1]["d_render"] + com[-1]["d_pbr"]) / com[-1]["selection_merit"],
               mis_arr=mis_arr, mis_end=col[name]["end"]["mis"],
               rejected=[dict(a=j["a"], dP=round(j["dP"], 4), dR=round(j["dR"], 4), worse_by="render" if j["dR"] > max(j["dP"], 0) else "physics")
                         for j in rej],
               stop=stop, stale_last=stale, reject_streak_last=streak, converged=a["converged"],
               delivered_window=a["truncation"]["best_animation"] if a["truncation"] else None)
    rows.append(row)
    print(json.dumps(row), flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
