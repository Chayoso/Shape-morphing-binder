"""d155_gate_read.py OUT_JSONL NAME=DIR [NAME=DIR ...] -- D155's 40k gate readings of a run from its JSON and its readers'
logs in DIR (CPU only, read-only): the same reading for the tag's runs (B3, or the tag's re-runs) and the switched runs.

Per run (the delivered windows; t = frame_end / 240 s):
  timing      the share of the summed per-window move (the record's mean |x_end - x_start|) after 1 s and after 3 s; the
              physics core Pc = ot_scale (TE + end drift) + released motion and the render term R = sil + shading against
              the FINAL target (the switched run's records final_TE / final_sil / final_pbr on the ramp, its own after),
              progress from the first record, A_lin / A_log (mean |c_P - c_R| over the run, signed > 0 physics ahead),
              P90 (s and run fraction)
  shared shape during 0 - T_e (T_e from the run's morph_time, else the tag's elastic period from the log): the raw
              cosine of the physics and render gradients (mean), the share of windows PCGrad acts in, the render share
              of the step (g_share) and of the merit lam R / (Pc + lam R)
  kinetic     the largest window-end kinetic energy and its time; the delivered window's (at rest)
  momentum    the centre of mass's largest drift from the source (wu), |L| at the window starts (largest, mean)
  apart       d120_apart.py's log (apart_NAME.log): the largest and median count in windows 3-6 and 3-12
  visible     visible_apart.py's per-window rows (vis_w3_10_NAME.jsonl): apart / blob, largest over windows 3-10
  end         the JSON metrics (own silhouette IoU, chamfer, thin uncovered); ind.log (gallery_ind: IoU, chamfer against
              the independent 40k sample); t3end_NAME.log (yardstick: exterior silhouette + shading) and e3end_NAME.log
              (display front 1 - IoU) if present
  minutes, windows"""
import json
import os
import re
import sys

import numpy as np

FPS = 240.0


def jl(p):
    try:
        return [json.loads(s) for s in open(p) if s.startswith("{")]
    except OSError:
        return []


def crossing(t, c, q):
    for i in range(1, len(t)):
        if c[i] >= q:
            a = (q - c[i - 1]) / max(c[i] - c[i - 1], 1e-12)
            return float(t[i - 1] + a * (t[i] - t[i - 1]))
    return float(t[-1])


def progress(x, log):
    x = np.asarray(x, float)
    if log:
        x = np.log(np.maximum(x, 1e-30))
    return (x[0] - x) / (x[0] - x[-1]) if x[0] != x[-1] else np.zeros_like(x)


def area(t, a, b):
    d = np.asarray(a) - np.asarray(b)
    dt = np.diff(t)
    signed = float(np.sum(0.5 * (d[1:] + d[:-1]) * dt))
    ab = 0.0
    for i in range(len(dt)):
        d0, d1 = d[i], d[i + 1]
        if d0 * d1 >= 0:
            ab += 0.5 * (abs(d0) + abs(d1)) * dt[i]
        else:
            s = abs(d0) / (abs(d0) + abs(d1))
            ab += 0.5 * (abs(d0) * s + abs(d1) * (1 - s)) * dt[i]
    span = float(t[-1] - t[0])
    return signed / span, ab / span


def read(name, d):
    J = json.load(open(f"{d}/{name}.json"))
    arm = next(iter(J["arms"].values()))
    cfg, dn, met = arm["config"], int(arm["deliver_n"]), arm["metrics"]
    log = open(f"{d}/{name}.log").read()
    m = re.search(r"\[disc\] .*elastic period=([0-9.]+) ms", log)
    t_e = float(cfg.get("morph_time") or 0.0) or (float(m.group(1)) / 1e3 if m else float("nan"))
    switched = float(cfg.get("morph_time") or 0.0) > 0
    s_ot = re.search(r"transport scale ([0-9.eE+-]+)", log)
    hist = [h for h in arm["history"] if h.get("frame_end") and not h.get("null_commit") and h["frame_end"] <= dn
            and h.get("transport_energy") is not None]
    t = np.array([h["frame_end"] / FPS for h in hist])
    f = np.array([h["frame_end"] / dn for h in hist])
    TE = np.array([h["final_TE"] if "final_TE" in h else h["transport_energy"] for h in hist])
    SIL = np.array([h["final_sil"] if "final_sil" in h else h["d_sil"] for h in hist])
    PBR = np.array([h["final_pbr"] if "final_pbr" in h else h["d_pbr"] for h in hist])
    w_pbr = float(cfg.get("w_pbr", 1.0))
    # one ot_scale for every run read here, 0.6 (the tag's runs do not log theirs; D155 measured that 0.3-1.2 move P90 by
    # <= 0.003 and A by <= 0.002); the switched run's own is recorded beside it
    ots = 0.6
    out_ots = float(s_ot.group(1)) if s_ot else None
    Pc = ots * (TE + np.array([h["stab_end"] for h in hist])) + np.array([h["stab"] for h in hist])
    R = SIL + w_pbr * PBR
    lam = np.array([h.get("lambda") or 0.0 for h in hist])
    move = np.array([h["move"] for h in hist])
    out = dict(name=name, switched=switched, T_e=t_e, windows=len(hist), deliver_n=dn, T=dn / FPS,
               minutes=arm["seconds"] / 60, ot_scale=ots, ot_scale_run=out_ots,
               move_after_1s=float(move[t > 1.0 + 1e-9].sum() / move.sum()),
               move_after_3s=float(move[t > 3.0 + 1e-9].sum() / move.sum()),
               move_after_Te=float(move[t > t_e + 1e-9].sum() / move.sum()))
    cm = np.cumsum(move) / move.sum()
    out["move90_s"] = float(t[np.searchsorted(cm, 0.9)])
    for sc in ("lin", "log"):
        cP, cR = progress(Pc, sc == "log"), progress(R, sc == "log")
        sg, ab = area(f, cP, cR)
        out[sc] = dict(A=ab, A_signed=sg, P90=crossing(f, cP, .9), R90=crossing(f, cR, .9),
                       P90_s=crossing(t, cP, .9), R90_s=crossing(t, cR, .9))
    ramp = t <= t_e + 1e-9
    rc = [h.get("g_raw_cos") for h, r in zip(hist, ramp) if r and h.get("g_raw_cos") is not None]
    pc = [abs((h.get("g_cos") or 0) - (h.get("g_raw_cos") or 0)) > 1e-12 for h, r in zip(hist, ramp) if r]
    gs = [h.get("g_share") for h, r in zip(hist, ramp) if r and h.get("g_share") is not None]
    out["ramp"] = dict(windows=int(ramp.sum()), raw_cos=float(np.mean(rc)) if rc else None,
                       pcgrad_share=float(np.mean(pc)) if pc else None, g_share=float(np.mean(gs)) if gs else None,
                       merit_share=float(np.mean((lam * R / (Pc + lam * R))[ramp])) if ramp.any() else None,
                       capped=int(sum(1 for h, r in zip(hist, ramp) if r and h.get("lambda_capped"))))
    late = ~ramp
    rc2 = [h.get("g_raw_cos") for h, r in zip(hist, late) if r and h.get("g_raw_cos") is not None]
    out["after_raw_cos"] = float(np.mean(rc2)) if rc2 else None
    kin = np.array([h["kin"] for h in hist])
    out.update(kin_peak=float(kin.max()), kin_peak_t=float(t[int(kin.argmax())]), kin_end=float(kin[-1]),
               kin_run_peak=float(max(h.get("kin_run") or 0 for h in hist)))
    com = np.array([h["com"] for h in hist])
    first = next(h for h in arm["history"] if h.get("L_start") is not None)
    out["com_drift"] = float(np.linalg.norm(com - com[0], axis=1).max())
    L = np.array([np.linalg.norm(h["L_start"]) for h in hist if h.get("L_start") is not None])
    out.update(L_max=float(L.max()) if len(L) else None, L_mean=float(L.mean()) if len(L) else None)
    del first
    # apart particles (d120_apart.py: "frame .. raw .. window W apart A sets S")
    rows = re.findall(r"window\s+([0-9.]+) apart\s+(\d+)", open(f"{d}/apart_{name}.log").read()) \
        if os.path.exists(f"{d}/apart_{name}.log") else []
    if rows:
        w = np.array([float(r[0]) for r in rows])
        a = np.array([int(r[1]) for r in rows])
        for lo, hi in ((3, 6), (3, 12)):
            m_ = (w >= lo) & (w <= hi)
            out[f"apart_w{lo}_{hi}"] = dict(max=int(a[m_].max()) if m_.any() else None,
                                            median=float(np.median(a[m_])) if m_.any() else None)
    vis = jl(f"{d}/vis_w3_10_{name}.jsonl")
    if vis:
        vw = [v for v in vis if isinstance(v.get("window"), (int, float)) and 3 <= v["window"] <= 10]
        out["visible_w3_10"] = dict(apart_max=max(v["apart"] for v in vw), blob_max=max(v["blob"] for v in vw),
                                    bump_max=max(v["bump"] for v in vw),
                                    side_blob_max=max(v.get("side_blob", 0) for v in vw)) if vw else None
    out["curves"] = dict(t=t.tolist(), move=move.tolist(), kin=kin.tolist(), raw_cos=[h.get("g_raw_cos") for h in hist],
                         merit_share=(lam * R / (Pc + lam * R)).tolist(), Pc=Pc.tolist(), R=R.tolist(),
                         u_gate=[h.get("u_gate") for h in hist], J=[h.get("Jx_mean") for h in hist])
    ug = [(tt, h.get("u_gate")) for tt, h in zip(t, hist) if h.get("u_gate") is not None]
    out["u_gate_1s"] = float(np.mean([u for tt, u in ug if tt <= 1.0 + 1e-9])) if ug else None
    out["u_gate_ramp"] = float(np.mean([u for tt, u in ug if tt <= t_e + 1e-9])) if ug else None
    out["u_gate_half_t"] = next((float(tt) for tt, u in ug if u >= 0.5), None)
    # D155c: the volume (records' tracked J) on the ramp and at the end; the targets' match and local density
    jm = [(tt, h.get("Jx_mean")) for tt, h in zip(t, hist) if h.get("Jx_mean") is not None]
    out["J_ramp_max"] = max((j for tt, j in jm if tt <= t_e + 1e-9), default=None)
    out["J_end"] = dict(mean=hist[-1].get("Jx_mean"), p01=hist[-1].get("Jx_p01"), p99=hist[-1].get("Jx_p99"))
    ks = [h["morph_k"] for h in hist if h.get("morph_k") is not None]
    vr = [h["morph_vol_ratio"] for h in hist if h.get("morph_vol_ratio") is not None]
    vq = [h["morph_vol_q"] for h in hist if h.get("morph_vol_q") is not None]
    out["morph"] = dict(k=[min(ks), max(ks)] if ks else None, vol_ratio=[min(vr), max(vr)] if vr else None,
                        vol_q_p05=min(q[0] for q in vq) if vq else None,
                        vol_q_p95=max(q[2] for q in vq) if vq else None, final_vol_q=None)
    mq = re.search(r"local volume quantiles p05 / p50 / p95 \[([^\]]*)\]", log)
    if mq:
        out["morph"]["final_vol_q"] = [float(v) for v in mq.group(1).split(",")]
    bs = [h.get("morph_build_s") for h in hist if (h.get("morph_build_s") or 0) > 0 and (h.get("morph_s") or 1) < 1]
    ta = [h.get("t_attempt") for h, r in zip(hist, ramp) if r and h.get("t_attempt")]
    out.update(build_s_mean=float(np.mean(bs)) if bs else None, ramp_attempt_s=float(np.mean(ta)) if ta else None)
    out["end"] = dict(sil_iou=met.get("sil_iou"), chamfer=met.get("chamfer"), thin=met.get("thin_uncovered"))
    for r in jl(f"{d}/ind.log"):
        if r.get("tag") == name:
            out["end"].update(ind_sil=r["ind"]["sil"], ind_chamfer=r["ind"]["chamfer"], floor_sil=r["floor"]["sil"])
    t3 = [r for r in jl(f"{d}/t3end_{name}.log") if r.get("exterior")]
    if t3:
        out["end"].update(yard_sil=t3[-1]["exterior"]["sil"], yard_pbr=t3[-1]["exterior"]["pbr"])
    e3 = [r for r in jl(f"{d}/e3end_{name}.log") if r.get("to_target") and str(r.get("state")) not in ("target", "own")]
    if e3:
        out["end"].update(display=1 - e3[-1]["to_target"]["front"]["iou"])
    return out


def main():
    with open(sys.argv[1], "w") as fo:
        for spec in sys.argv[2:]:
            name, d = spec.split("=", 1)
            try:
                r = read(name, d)
            except (OSError, StopIteration, KeyError, ValueError) as e:
                print(f"{name}: unreadable ({e!r})", flush=True)
                continue
            fo.write(json.dumps(r) + "\n")
            print(f"{name}: {r['windows']} windows {r['minutes']:.1f} min; move after 1 / 3 s {r['move_after_1s']:.3f} / "
                  f"{r['move_after_3s']:.3f}; A lin {r['lin']['A']:.3f} log {r['log']['A']:.3f}; P90 {r['lin']['P90_s']:.2f} s; "
                  f"kin peak {r['kin_peak']:.3g} at {r['kin_peak_t']:.2f} s, end {r['kin_end']:.2e}; ramp cos "
                  f"{r['ramp']['raw_cos']}; end {r['end']}", flush=True)


if __name__ == "__main__":
    main()
