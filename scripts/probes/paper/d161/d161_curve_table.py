"""d161_curve_table.py OUT_MD OUT_PNG CURVE_JSONL [CURVE_JSONL ...] [--features FEATURES_JSON] -- D161: the response
curve's table and figure from d161_curve.py's rows (CPU, numpy / matplotlib).

Per (mesh, N, w, dt), in pitches of relief:
  pictures   the relief's signal in the target pictures: (L_r(X0; relief pictures) - L_r(X0; smooth)) / L_r(X0; smooth)
             (how much more the smooth arrived body pays against the relief pictures, over its own floor);
  render     ds_r: the render term's own minimiser along the relief mode (the relief the pictures ask the particles
             for), and its share ds_r / dt;
  physics    each term's derivative at ds = 1 pitch (its hold against one pitch of relief; > 0 resists) and the sum;
  totals     ds*: the minimiser of physics + lambda x render (lambda: the tag's late windows, and lambda_adapt);
  window     R: the share of a layer relief the relaxation leaves after one window (layer particles, and the discs); G:
             the relief one window of u = 1 pitch x P writes (layer, discs); u_win: the step the tag's late windows take
             (iters x alpha_last, pitches); A_ss = G u_win / (1 - R): the relief the relaxation lets u hold if every
             window pushes it at the full step;
  predicted  min(ds*_adapt, A_ss) and its share of dt; the blocker: the smallest of (pictures' ask ds_r, ds*, A_ss).
"""
import json
import sys

import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith("--")]
feat = None
if "--features" in sys.argv:
    feat = json.load(open(sys.argv[sys.argv.index("--features") + 1]))
    args = [a for a in args if a != sys.argv[sys.argv.index("--features") + 1]]
out_md, out_png, files = args[0], args[1], args[2:]
heads, rows = {}, []
for f in files:
    for ln in open(f):
        r = json.loads(ln)
        if r.get("head"):
            heads[(r["mesh"], r["N"])] = r
        elif "w" in r:
            rows.append(r)


def fmt(v, p=3):
    return "–" if v is None else (f"{v:.{p}g}" if isinstance(v, float) else str(v))


lines = ["# D161 response curve", ""]
for key, h in heads.items():
    lines.append(f"**{key[0]} {key[1] // 1000}k**: pitch {h['pitch']:.4f} wu, loss cell {h['loss_cell_pitch']:.2f} a, "
                 f"MPM dx {h['mpm_dx_pitch']:.2f} a, pixel {h['pixel_pitch']:.2f} a, field radius {h['ext_radius']} a, "
                 f"layer spacing {h['sp0_layer'] / h['pitch']:.2f} a, berth {h['berth_pitch']:.2f} a, ot_scale "
                 f"{h['ot_scale']:.4g}, unit_ratio {h['unit_ratio']:.4g}, lambda_tag {fmt(h['lam_tag'])}, u_win "
                 f"{fmt(h['u_window_tag_pitch'])} a")
lines += ["", "| mesh | N | w | dt | pictures | ds_r (share) | hold T / P / nn / sp at 1a | push render (tag) | ds* tag | "
          "ds* adapt | R layer / disc | G layer / disc | A_ss | predicted (share) | PCGrad kept |", "|" + "---|" * 15]
pred = {}
for r in sorted(rows, key=lambda r: (r["mesh"], r["N"], r["w"], r["dt"])):
    h = heads[(r["mesh"], r["N"])]
    floor = r["floor_smooth"]
    sig = (r["X0_vs_relief"] - floor) / max(floor, 1e-30)
    ds_r = r["render_only"]["ds_star"]
    hold = r["phys"]["1.0"]["der"]
    tot = r["totals"]
    st = tot.get("tag", {}).get("ds_star")
    sa = tot.get("adapt", {}).get("ds_star")
    w = r["window"]
    R_l, R_d = w["R_layer"]["A"] / h["pitch"], w["R_disc"]["A"] / h["pitch"]
    l0 = r["field"]["layer"]["A"] / h["pitch"]                         # the layer's relief at X(1) (the start of R)
    d0 = r["field"]["disc"]["A"] / h["pitch"]
    Rl, Rd = R_l / max(l0, 1e-9), R_d / max(d0, 1e-9)
    Gl, Gd = w["G_layer"]["A"] / h["pitch"], w["G_disc"]["A"] / h["pitch"]
    uw = h["u_window_tag_pitch"] or 0.0
    A_ss = Gl * uw / max(1 - Rl, 1e-3)
    p = min(x for x in (sa if sa is not None else np.inf, A_ss) if x is not None)
    pc = r.get("pcgrad")
    kept = None if not pc else pc["along_after"] / pc["along"] if pc["along"] else None
    pred[(r["mesh"], r["N"], round(r["w"], 1), r["dt"])] = dict(p=p, ds_r=ds_r, sa=sa, A_ss=A_ss, sig=sig, Rl=Rl, Gl=Gl)
    lines.append(f"| {r['mesh']} | {r['N'] // 1000}k | {r['w']:.1f} | {r['dt']} | {sig:+.3f} | {ds_r:.2f} ({ds_r / r['dt']:.2f}) | "
                 f"{hold['T']:.2e} / {hold['P']:.2e} / {hold['nn']:.2e} / {hold['sp']:.2e} | "
                 f"{fmt(tot.get('tag', {}).get('push_render'))} | {fmt(st)} | {fmt(sa)} | {Rl:.2f} / {Rd:.2f} | "
                 f"{Gl:.2f} / {Gd:.2f} | {A_ss:.2f} | {p:.2f} ({p / r['dt']:.2f}) | {fmt(kept)} |")
open(out_md, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
keys = sorted(heads)
fig, axes = plt.subplots(1, len(keys), figsize=(5.2 * len(keys), 4.4), squeeze=False)
for ax, key in zip(axes[0], keys):
    for dt, col in zip((0.25, 0.5, 1.0, 2.0), ("#8c8c8c", "#4a7ab5", "#d9822b", "#3a9a5b")):
        ks = sorted(k for k in pred if k[0] == key[0] and k[1] == key[1] and k[3] == dt)
        if not ks:
            continue
        ws = [k[2] for k in ks]
        ax.plot(ws, [pred[k]["ds_r"] / dt for k in ks], "--", color=col, lw=1)
        ax.plot(ws, [pred[k]["p"] / dt for k in ks], "-o", color=col, lw=2, ms=4, label=f"target {dt} a")
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("relief wavelength (pitches)")
    ax.set_ylabel("share of the target relief kept")
    ax.set_title(f"{key[0]} {key[1] // 1000}k: dashed = the pictures ask, solid = predicted")
    if feat:
        for fr in feat:
            for b in fr["bands"]:
                lo, hi = b["wavelength_300k" if key[1] > 100000 else "wavelength_40k"]
                amp = b["top_rms_300k" if key[1] > 100000 else "top_rms_40k"]
                ax.axvspan(lo, hi, ymin=0.0, ymax=0.04, color="#c44" if fr["mesh"] == "dragon" else "#888", alpha=0.25)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out_png, dpi=130)
