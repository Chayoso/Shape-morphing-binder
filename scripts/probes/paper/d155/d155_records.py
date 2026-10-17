"""d155_records.py OUT_JSON -- D155 (CPU, read-only): D113's picture on the tag freeze-2026-10-15 from B1's
records and logs (and B7's RG97 records).

Per run, per committed window inside the delivered run:
  t   = frame_end / 240 s; f = frame_end / deliver_n (run fraction)
  Pc  = ot_scale (TE + stab_end) + stab (phys_core: geometry energy with proximity and end drift, released motion);
        ot_scale from D153's floor153 (bunny / dragon), else 0.6 with a sensitivity check (0.3 / 1.2)
  R   = d_sil + w_pbr d_pbr (the render term, unweighted; the twin records it, never optimised)
  path= 1 - |d|_k / |d|_0: the OT plan's mean |d| at each attempt's start (the log), |d|_0 at the source
Progress c = (X0 - X) / (X0 - X_end) with X0 = the FIRST committed record (the source terms are not recorded),
linear and log; crossings P50 / P90; A = the area between physics and render progress over the run, per unit run
(signed > 0: physics ahead). Thirds: g_share, lambda, render share of the merit lam R / (Pc + lam R).
Displacement: the m_ log's mean particle move per kept-frame pair (12 frames, pitches) summed per phase; the
record's move (mean |x_end - x_start| per window, wu -> pitches). Late: values at f = 1/3, t = 3 s, 6 s over the end.
Yardstick (t3_: exterior sil + pbr against the independent sample) and display (e3_: 1 - IoU front) per kept frame."""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.spatial import cKDTree

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
B1, B7 = f"{G}/d145t_b1", f"{G}/d145t_b7"
FPS = 240.0
OTS = {}
for _l in open(f"{G}/d153/floor153.jsonl"):
    _r = json.loads(_l)
    if _r.get("state") == "ind":
        _m = re.match(r"([a-z]+)300k_[A-Z]+(\d+)", _r["run"])
        OTS[(_m.group(1), int(_m.group(2)))] = float(_r["ot_scale"])


def jl(p):
    try:
        return [json.loads(s) for s in open(p) if s.startswith("{")]
    except (OSError, TypeError):
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
    """mean over the run of a - b (signed) and of |a - b|, both piecewise linear on the same knots"""
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


def at(t, x, q):
    return float(np.interp(q, t, x))


PITCH = {}


def pitch_of(mesh, seed):
    k = (mesh, seed)
    if k not in PITCH:
        tgt = np.asarray(np.load(f"{B1}/{mesh}300k_L{seed}_end1.npz")["tgt"], np.float64)
        PITCH[k] = 0.708 * float(np.median(cKDTree(tgt).query(tgt[::50], 9)[0][:, 8]))
    return PITCH[k]


def run(name, js, logp, mlog, t3, e3):
    m = re.match(r"([a-z]+)300k_([A-Z]+)(\d+)", name)
    mesh, arm_l, seed = m.group(1), m.group(2), int(m.group(3))
    J = json.load(open(js))
    arm = next(iter(J["arms"].values()))
    dn = int(arm["deliver_n"])
    w_pbr = float(arm["config"].get("w_pbr", 1.0))
    hist = arm["history"]
    dlines = [float(x) for x in re.findall(r"fixed grid transport: mean \|d\|=([0-9.eE+-]+) wu", open(logp).read())]
    if len(dlines) != len(hist):
        print(f"{name}: {len(dlines)} transport lines against {len(hist)} records", flush=True)
    s_known = OTS.get((mesh, seed))
    pit = pitch_of(mesh, seed)
    rows = []
    for i, h in enumerate(hist):
        if not h.get("frame_end") or h.get("null_commit") or h["frame_end"] > dn or h.get("transport_energy") is None:
            continue
        nxt = dlines[i + 1] if i + 1 < len(dlines) else None
        rows.append(dict(a=h["animation"], fe=h["frame_end"], TE=h["transport_energy"], se=h["stab_end"],
                         st=h["stab"], sil=h["d_sil"], pbr=h["d_pbr"], lam=h["lambda"], gs=h.get("g_share"),
                         gp=h.get("g_phys_norm"), gr=h.get("g_rend_norm"), kin=h["kin"], move=h["move"] / pit,
                         merit=h["selection_merit"], dnext=nxt, alpha=h.get("alpha_last")))
    t = np.array([r["fe"] / FPS for r in rows])
    f = np.array([r["fe"] / dn for r in rows])
    out = dict(name=name, mesh=mesh, arm=arm_l, seed=seed, deliver_n=dn, windows=len(rows), T=dn / FPS,
               minutes=arm["seconds"] / 60, pitch=pit, ot_scale=s_known, w_pbr=w_pbr, d0=dlines[0])
    s_use = s_known if s_known is not None else 0.6

    def core(s):
        return np.array([s * (r["TE"] + r["se"]) + r["st"] for r in rows])

    Pc = core(s_use)
    R = np.array([r["sil"] + w_pbr * r["pbr"] for r in rows])
    SIL = np.array([r["sil"] for r in rows])
    PBR = np.array([r["pbr"] for r in rows])
    TE = np.array([r["TE"] for r in rows])
    KIN = np.array([r["kin"] for r in rows])
    ST = np.array([r["st"] for r in rows])
    lam = np.array([r["lam"] or 0.0 for r in rows])
    cur = dict(t=t.tolist(), f=f.tolist(), Pc=Pc.tolist(), R=R.tolist(), SIL=SIL.tolist(), PBR=PBR.tolist(),
               TE=TE.tolist(), kin=KIN.tolist(), stab=ST.tolist(), lam=lam.tolist(), move=[r["move"] for r in rows],
               gs=[r["gs"] for r in rows], gp=[r["gp"] for r in rows], gr=[r["gr"] for r in rows],
               path=[1.0 - r["dnext"] / dlines[0] if r["dnext"] is not None else None for r in rows],
               alpha=[r["alpha"] for r in rows], merit=[r["merit"] for r in rows])
    for scale in ("lin", "log"):
        cP, cR = progress(Pc, scale == "log"), progress(R, scale == "log")
        cS, cB = progress(SIL, scale == "log"), progress(PBR, scale == "log")
        sg, ab = area(f, cP, cR)
        o = dict(P50=crossing(f, cP, .5), P90=crossing(f, cP, .9), R50=crossing(f, cR, .5), R90=crossing(f, cR, .9),
                 S90=crossing(f, cS, .9), B90=crossing(f, cB, .9), A_abs=ab, A_signed=sg)
        o.update({k + "_s": v * dn / FPS for k, v in list(o.items()) if k[0] in "PRSB"})
        for nm, x in (("Pc", Pc), ("R", R)):
            c = progress(x, scale == "log")
            o[f"{nm}_left_f33"] = 1 - at(f, c, 1 / 3)
            o[f"{nm}_left_3s"] = 1 - at(t, c, 3.0)
            o[f"{nm}_left_6s"] = 1 - at(t, c, 6.0)
        out[scale] = o
        cur["c" + scale] = dict(P=cP.tolist(), R=cR.tolist())
    sens = {}
    for s in (0.3, 0.6, 1.2):
        c = progress(core(s), False)
        cl = progress(core(s), True)
        sens[str(s)] = dict(P90=crossing(f, c, .9), A_lin=area(f, c, progress(R, False))[1],
                            A_log=area(f, cl, progress(R, True))[1], P90_log=crossing(f, cl, .9))
    out["ot_sens"] = sens
    pth = np.array([p if p is not None else np.nan for p in cur["path"]])
    ok = np.isfinite(pth)
    tt, ff = np.r_[0.0, t[ok]], np.r_[0.0, f[ok]]
    pp = np.r_[0.0, pth[ok]]
    out["path50_f"], out["path90_f"] = crossing(ff, pp, .5), crossing(ff, pp, .9)
    out["path50_s"], out["path90_s"] = crossing(tt, pp, .5), crossing(tt, pp, .9)
    out["path_at"] = {str(q): at(tt, pp, q) for q in (0.5, 1.0, 2.0, 3.0, 5.0, 6.0)}
    out["path_end"] = float(pp[-1])
    th = {}
    for lo, hi, nm in ((0, .1, "0-10"), (0, 1 / 3, "1st"), (1 / 3, 2 / 3, "2nd"), (2 / 3, 1.01, "3rd")):
        msk = ((f > lo) & (f <= hi)) if lo > 0 else (f <= hi)
        gs = [g for g, mm in zip(cur["gs"], msk) if mm and g is not None]
        share_m = [l * x / (p + l * x) for l, x, p, mm in zip(lam, R, Pc, msk) if mm]
        th[nm] = dict(windows=int(msk.sum()), g_share=float(np.mean(gs)) if gs else None,
                      lam=float(np.mean(lam[msk])) if msk.any() else None,
                      render_share_merit=float(np.mean(share_m)) if share_m else None,
                      move_share=float(np.sum(np.array(cur["move"])[msk]) / np.sum(cur["move"])))
    out["thirds"] = th
    mm = [r for r in jl(mlog) if isinstance(r.get("state"), int)]
    if mm:
        st = np.array([r["state"] for r in mm], float)
        mv = np.array([r["move"] for r in mm], float)
        keep = st <= dn
        st, mv = st[keep], mv[keep]
        tot = mv.sum()
        bins = [(0, 1), (1, 2), (2, 3), (3, 6), (6, 1e9)]
        out["move_bins_s"] = {f"{a}-{b if b < 1e8 else 'end'}": float(mv[(st / FPS > a) & (st / FPS <= b)].sum() / tot)
                              for a, b in bins}
        out["move_thirds"] = {nm: float(mv[(st / dn > lo) & (st / dn <= hi)].sum() / tot)
                              for nm, lo, hi in (("1st", 0, 1 / 3), ("2nd", 1 / 3, 2 / 3), ("3rd", 2 / 3, 1.01))}
        out["move_first10"] = float(mv[st / dn <= .1].sum() / tot)
        out["move_total_pitch"] = float(tot)
        cum = np.cumsum(mv) / tot
        out["move50_s"] = float(st[np.searchsorted(cum, .5)] / FPS)
        out["move90_s"] = float(st[np.searchsorted(cum, .9)] / FPS)
        out["move90_f"] = float(st[np.searchsorted(cum, .9)] / dn)
        out["move_late_rate"] = float(np.median(mv[st / dn > 2 / 3]))   # pitches per kept-frame pair, last third
        cur["kmove"] = dict(state=st.tolist(), move=mv.tolist())
    late = {}
    for nm, x in (("Pc", Pc), ("TE", TE), ("R", R), ("SIL", SIL), ("PBR", PBR), ("kin", KIN)):
        late[nm] = {"f33": at(f, x, 1 / 3) / x[-1], "3s": at(t, x, 3.0) / x[-1], "6s": at(t, x, 6.0) / x[-1],
                    "first_over_end": float(x[0] / x[-1])}
    out["late"] = late
    y = [r for r in jl(t3) if isinstance(r.get("state"), int) and r["state"] <= dn]
    if y:
        ys = np.array([r["state"] for r in y], float)
        Y = np.array([r["exterior"]["sil"] + r["exterior"]["pbr"] for r in y])
        YS = np.array([r["exterior"]["sil"] for r in y])
        cur["yard"] = dict(state=ys.tolist(), Y=Y.tolist(), YS=YS.tolist())
        for scale in ("lin", "log"):
            c = progress(Y, scale == "log")
            out["yard_" + scale] = dict(R50=crossing(ys / dn, c, .5), R90=crossing(ys / dn, c, .9),
                                        R90_s=crossing(ys / FPS, c, .9),
                                        left_f33=1 - at(ys / dn, c, 1 / 3), left_3s=1 - at(ys / FPS, c, 3.0),
                                        left_6s=1 - at(ys / FPS, c, 6.0))
        out["yard_late"] = {"f33": at(ys / dn, Y, 1 / 3) / Y[-1], "3s": at(ys / FPS, Y, 3.0) / Y[-1],
                            "6s": at(ys / FPS, Y, 6.0) / Y[-1], "src_over_end": float(Y[0] / Y[-1])}
    e = [dict(r, state=int(r["state"])) for r in jl(e3) if str(r.get("state")).isdigit() and int(r["state"]) <= dn and "to_target" in r]
    if e:
        es = np.array([r["state"] for r in e], float)
        D = np.array([1 - r["to_target"]["front"]["iou"] for r in e])
        cur["disp"] = dict(state=es.tolist(), D=D.tolist())
        out["disp_late"] = {"f33": at(es / dn, D, 1 / 3) / D[-1], "3s": at(es / FPS, D, 3.0) / D[-1],
                            "6s": at(es / FPS, D, 6.0) / D[-1], "src_over_end": float(D[0] / D[-1])}
        c = progress(D, False)
        out["disp_lin"] = dict(R90=crossing(es / dn, c, .9), R90_s=crossing(es / FPS, c, .9),
                               left_f33=1 - at(es / dn, c, 1 / 3))
    out["curves"] = cur
    return out


def main():
    res = []
    names = sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{B1}/*300k_[LP][0-9]*.json"))
    for n in names:
        t3, e3 = f"{B1}/t3_{n}.log", f"{B1}/e3_{n}.log"
        res.append(run(n, f"{B1}/{n}.json", f"{B1}/{n}.log", f"{B1}/m_{n}.log",
                       t3 if os.path.exists(t3) else None, e3 if os.path.exists(e3) else None))
        r = res[-1]
        print(f"{n}: windows {r['windows']} T {r['T']:.2f} s; lin P90 {r['lin']['P90']:.3f} R90 {r['lin']['R90']:.3f} "
              f"A {r['lin']['A_abs']:.3f}; log P90 {r['log']['P90']:.3f} R90 {r['log']['R90']:.3f} A "
              f"{r['log']['A_abs']:.3f} ({r['log']['A_signed']:+.3f}); path90 {r['path90_f']:.3f} "
              f"({r['path90_s']:.2f} s); move90 {r.get('move90_f', float('nan')):.3f} ({r.get('move90_s', float('nan')):.2f} s)",
              flush=True)
    for m in ("bunny", "dragon"):
        n = f"{m}300k_RG97"
        res.append(run(n, f"{B7}/{n}.json", f"{B7}/{n}.log", None, None, None))
    json.dump(res, open(sys.argv[1], "w"), default=float)


if __name__ == "__main__":
    main()
