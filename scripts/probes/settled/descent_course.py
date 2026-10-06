"""descent_course.py OUT_PREFIX LABEL=RUN_JSON,OWN_LOG,T3_LOG [...] — D113: does the physics finish first and the render
after, or do both fall together? Per run, the physics core and the render term per committed window, each normalised
to its own total drop from the source to the delivered end.

Physics core per window (the run's records): ot_scale (transport energy + end drift) + released motion; at the source
(OWN_LOG's state -1, own_terms_probe.py) ot_scale x transport energy (at rest). Render per window: the run's own
silhouette + shading on the exterior; at the source from OWN_LOG. Time: the window's last frame over the delivered
frames (the run fraction). Per term, the progress c(t) = (X0 - X(t)) / (X0 - X_end) on two scales: linear and log
(log X in place of X; the terms span two to four decades). Reported: the run fraction at which c first reaches 0.5
and 0.9, and the simultaneity numbers: A = the area between the physics and render progress curves over the run
(0: together; the signed area > 0: the physics ahead) and the gap of the two 90 % crossings. From OWN_LOG's kept frames
(if any): the physics geometry and the render per frame; from T3_LOG (the yardstick against an independent sample,
render_terms_probe.py; "-": none) the exterior silhouette + shading per frame; the arrival (particles within two
target spacings of the target, the mean path done). Writes OUT_PREFIX.json and OUT_PREFIX.png."""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows_of = lambda path: [json.loads(l) for l in open(path) if l.startswith("{")]  # noqa: E731


def crossing(t, c, q):
    """First run fraction at which the piecewise-linear progress reaches q."""
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
    """Integral over the run of a - b (both piecewise linear on the same knots): signed and absolute."""
    d = np.asarray(a) - np.asarray(b)
    dt = np.diff(t)
    signed = float(np.sum(0.5 * (d[1:] + d[:-1]) * dt))
    # the absolute value of a piecewise-linear function, exact per segment
    ab = 0.0
    for i in range(len(dt)):
        d0, d1 = d[i], d[i + 1]
        if d0 * d1 >= 0:
            ab += 0.5 * (abs(d0) + abs(d1)) * dt[i]
        else:
            s = abs(d0) / (abs(d0) + abs(d1))
            ab += 0.5 * (abs(d0) * s + abs(d1) * (1 - s)) * dt[i]
    return signed, float(ab)


def course(label, js, own, t3):
    arm = next(iter(json.load(open(js))["arms"].values()))
    dn = int(arm["deliver_n"])
    hist = [h for h in arm["history"] if h.get("frame_end") and not h.get("null_commit")
            and h["frame_end"] <= dn and h.get("transport_energy") is not None]
    orows = rows_of(own)
    head, src = orows[0], next(r for r in orows if r.get("state") == -1)
    s = float(head["ot_scale"])
    t = np.array([0.0] + [h["frame_end"] / dn for h in hist])
    P = [src["phys_geom"]] + [s * (h["transport_energy"] + h["stab_end"]) + h["stab"] for h in hist]
    TE = [src["TE"]] + [h["transport_energy"] for h in hist]
    R = [src["render"]] + [h["d_sil"] + h["d_pbr"] for h in hist]
    SIL = [src["sil"]] + [h["d_sil"] for h in hist]
    out = {"label": label, "windows": len(hist), "frames": dn, "minutes": arm["seconds"] / 60, "ot_scale": s,
           "P0": P[0], "P_end": P[-1], "R0": R[0], "R_end": R[-1], "TE0": TE[0], "TE_end": TE[-1]}
    curves = {"t": t.tolist(), "P": P, "R": R, "TE": TE, "SIL": SIL,
              "lambda": [None] + [h["lambda"] for h in hist], "g_share": [None] + [h.get("g_share") for h in hist],
              "g_phys": [None] + [h.get("g_phys_norm") for h in hist], "g_rend": [None] + [h.get("g_rend_norm") for h in hist]}
    for scale in ("lin", "log"):
        cP, cR = progress(P, scale == "log"), progress(R, scale == "log")
        cT, cS = progress(TE, scale == "log"), progress(SIL, scale == "log")
        sg, ab = area(t, cP, cR)
        out[scale] = {"P50": crossing(t, cP, .5), "P90": crossing(t, cP, .9), "R50": crossing(t, cR, .5),
                      "R90": crossing(t, cR, .9), "TE50": crossing(t, cT, .5), "TE90": crossing(t, cT, .9),
                      "SIL50": crossing(t, cS, .5), "SIL90": crossing(t, cS, .9),
                      "area_abs": ab, "area_signed": sg}
        out[scale]["gap90"] = out[scale]["R90"] - out[scale]["P90"]
        out[scale]["gap50"] = out[scale]["R50"] - out[scale]["P50"]
        curves["c" + scale] = {"P": cP.tolist(), "R": cR.tolist()}
    # the render's influence, by run thirds
    for name, key in (("g_share", "g_share"), ("lambda", "lambda")):
        v = [(tt, h.get(key)) for tt, h in zip(t[1:], hist) if h.get(key) is not None]
        out[name + "_thirds"] = [float(np.mean([x for tt, x in v if lo <= tt < hi] or [np.nan]))
                                 for lo, hi in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01))]
    # per frame: the probe's own terms and arrival, the yardstick
    fr = [r for r in orows if r.get("state", -1) >= 0]
    if fr:
        tf = np.array([r["state"] / dn for r in fr])
        out["frame"] = {"t": tf.tolist(), "phys_geom": [r["phys_geom"] for r in fr], "render": [r["render"] for r in fr],
                        "near2sp": [r["near2sp"] for r in fr], "path": [r["path"] for r in fr]}
        for q in (.5, .9):
            out[f"path{int(q * 100)}"] = crossing(np.r_[0., tf], np.r_[0., [r["path"] for r in fr]], q)
            out[f"near{int(q * 100)}"] = crossing(np.r_[0., tf], np.r_[src["near2sp"], [r["near2sp"] for r in fr]], q)
        for scale in ("lin", "log"):
            cP = progress([src["phys_geom"]] + out["frame"]["phys_geom"], scale == "log")
            cR = progress([src["render"]] + out["frame"]["render"], scale == "log")
            tt = np.r_[0., tf]
            sg, ab = area(tt, cP, cR)
            out["frame_" + scale] = {"P50": crossing(tt, cP, .5), "P90": crossing(tt, cP, .9),
                                     "R50": crossing(tt, cR, .5), "R90": crossing(tt, cR, .9),
                                     "area_abs": ab, "area_signed": sg}
    if t3 != "-":
        yr = [r for r in rows_of(t3) if isinstance(r.get("state"), int) and r["state"] <= dn]
        ty = np.array([r["state"] / dn for r in yr])
        Y = [r["exterior"]["sil"] + r["exterior"]["pbr"] for r in yr]
        out["yard"] = {"t": ty.tolist(), "render": Y}
        for scale in ("lin", "log"):
            cY = progress(Y, scale == "log")
            out["yard_" + scale] = {"R50": crossing(ty, cY, .5), "R90": crossing(ty, cY, .9)}
    out["curves"] = curves
    return out


def main():
    prefix, runs = sys.argv[1], []
    for spec in sys.argv[2:]:
        label, paths = spec.split("=", 1)
        js, own, t3 = paths.split(",")
        runs.append(course(label, js, own, t3))
    slim = [{k: v for k, v in r.items() if k not in ("curves", "frame", "yard")} for r in runs]
    json.dump({"runs": runs}, open(prefix + ".json", "w"))
    for r in slim:
        print(json.dumps(r))
    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    cols = ["k", "tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
    for r, c in zip(runs, cols):
        cv, lab = r["curves"], r["label"]
        t = np.array(cv["t"])
        ax[0, 0].plot(t, cv["clog"]["P"], "-", color=c, label=f"{lab} physics")
        ax[0, 0].plot(t, cv["clog"]["R"], "--", color=c, label=f"{lab} render")
        ax[0, 1].plot(t, cv["clin"]["P"], "-", color=c)
        ax[0, 1].plot(t, cv["clin"]["R"], "--", color=c)
        ax[0, 2].semilogy(t, np.array(cv["P"]) / cv["P"][0], "-", color=c)
        ax[0, 2].semilogy(t, np.array(cv["R"]) / cv["R"][0], "--", color=c)
        if "yard" in r:
            ax[0, 2].semilogy(r["yard"]["t"], np.array(r["yard"]["render"]) / r["yard"]["render"][0], ":", color=c)
        ax[1, 0].semilogy(t[1:], cv["lambda"][1:], "-", color=c, label=lab)
        ax[1, 1].plot(t[1:], cv["g_share"][1:], "-", color=c)
        if "frame" in r:
            ax[1, 2].plot(r["frame"]["t"], r["frame"]["path"], "-", color=c, label=f"{lab} path done")
            ax[1, 2].plot(r["frame"]["t"], r["frame"]["near2sp"], "--", color=c, label=f"{lab} within 2 spacings")
    ax[0, 0].set_title("progress, log scale (solid physics core, dashed render)")
    ax[0, 1].set_title("progress, linear scale")
    ax[0, 2].set_title("term / its source value (dotted: yardstick render)")
    ax[1, 0].set_title("render weight lambda")
    ax[1, 1].set_title("render share of the step g_share")
    ax[1, 2].set_title("arrival")
    for a in ax.flat:
        a.set_xlabel("run fraction (frames)")
        a.grid(alpha=.3)
    ax[0, 0].legend(fontsize=7)
    ax[1, 2].legend(fontsize=7)
    fig.suptitle(prefix.split("/")[-1])
    fig.tight_layout()
    fig.savefig(prefix + ".png", dpi=90)


if __name__ == "__main__":
    main()
