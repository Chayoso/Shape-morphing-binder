"""d158_table.py OUT_DIR [MESH ...] -- D158's reader (CPU): the tables of d158_fit.py's dumps (d158.jsonl, head.json),
and the readings' numbers: energy shares of h and of the relief h_rel held by one value an outer particle (W, the
display's weighting) and through the exterior field (F, the loop's path); band energies; the relief's render gain at
the four pictures and each part's share of it; the render gradient at h = 0; the static fits."""
import json

import numpy as np
import sys
from pathlib import Path

out = Path(sys.argv[1])
meshes = sys.argv[2:] or ["bunny", "dragon"]
BANDS = ("<2.7", "2.7-5.4", "5.4-10.8", "10.8-21.6", ">21.6")
SET = ("dense_ac", "dense_loop", "loop_ac", "loop_loop")
summary = {}


def L(t):
    return t[0] + t[1]                                             # silhouette + shading (w_pbr 1, the fit's)


for mesh in meshes:
    d = out / f"{mesh}300k_L97"
    head = json.load(open(d / "head.json"))
    rows = [json.loads(s) for s in open(d / "d158.jsonl")]
    print(f"\n=== {mesh} 300k L97: a {head['pitch_a']:.4f} a_c {head['pitch_ac']:.4f} (a_c/a {head['ac_over_a']:.3f}); "
          f"D115 pixel {head['pix_ac_over_a']:.3f} a ({head['res_ac']} px), loop pixel {head['pix_loop_over_a']:.3f} a "
          f"({head['res_loop']} px), ratio {head['pix_loop_over_a'] / head['pix_ac_over_a']:.2f}; discs {head['lattice_over_a']:.3f} a "
          f"(loop's {head['loop_lattice_over_a']:.3f} a); target draws {head['draws']} (tgt reproduced to {head['tgt_reproduced_maxdiff']:.2e}); "
          f"pictures dense vs loop: {head['target_pictures_diff']}")
    summary[mesh] = dict(head=head, rows=[])
    for r in rows:
        e, t = r["energy"], r["terms"]
        print(f"-- t {r['t_s']:.2f} s (kept {r['kept']}), arrived {r['arrived']:.3f}, layer {r['layer']}, lin check {r['lin_check_relerr']}, "
              f"F nbrs {r['field_nbrs_mean']}; h mean {r['h_mean_a']:+.3f} a rms {r['h_rms_a']:.3f} a, h_rel rms {r['h_rel_rms_a']:.3f} a")
        print(f"   energy held: h by W {e['h_uW']:.3f} F {e['h_uF']:.3f}; h_rel by W {e['rel_uW']:.3f} F {e['rel_uF']:.3f} "
              f"(rest {e['rel_restW']:.3f} / {e['rel_restF']:.3f}); lsq {json.dumps(r['lsq'])}")
        for nm, b in r["bands"].items():
            tot = sum(b)
            print(f"   bands {nm:7s} " + " ".join(f"{BANDS[i]}: {v:.2e} ({v / max(tot, 1e-30):.2f})" for i, v in enumerate(b)))
        srow = dict(t=r["t_s"], arrived=r["arrived"], h_mean=r["h_mean_a"], h_rms=r["h_rms_a"], rel_rms=r["h_rel_rms_a"],
                    eW=e["rel_uW"], eF=e["rel_uF"], ehW=e["h_uW"], ehF=e["h_uF"], bands_h=r["bands"]["h"], set={})
        dz = d / f"dump_{r['kept']:04d}.npz"
        if dz.exists():                                            # the least squares' own norm: every disc
            zz = np.load(dz)
            on = zz["arrived"] > 0.5
            al = {}
            for src, b in (("h", zz["h"]), ("rel", zz["h_rel"])):
                for nm in "WF":
                    pu = zz[f"{src}_u{nm}"] if src == "rel" else zz[f"h_u{nm}"]
                    pu = zz[f"rel_u{nm}"] if src == "rel" else pu
                    al[f"{src}_u{nm}"] = float((pu.astype(np.float64) ** 2).sum() / (b.astype(np.float64) ** 2).sum())
                    al[f"{src}_rest{nm}"] = float(((b - pu).astype(np.float64) ** 2).sum() / (b.astype(np.float64) ** 2).sum())
                    al[f"{src}_u{nm}_arr"] = float((pu[on].astype(np.float64) ** 2).sum() / (b[on].astype(np.float64) ** 2).sum())
                    al[f"{src}_rest{nm}_arr"] = float(((b - pu)[on].astype(np.float64) ** 2).sum() / (b[on].astype(np.float64) ** 2).sum())
            print("   energy over every disc (the projection's own norm): " + " ".join(f"{k} {v:.3f}" for k, v in al.items()))
            srow["all"] = al
        for s in SET:
            tt = t[s]
            L0, Lh, Ll = L(tt["zero"]), L(tt["h"]), L(tt["low"])
            G = Ll - Lh
            shW, shF = (Ll - L(tt["low+relW"])) / G if G else float("nan"), (Ll - L(tt["low+relF"])) / G if G else float("nan")
            g = r["grad"][s]
            print(f"   {s:10s} L0 {L0:.4e} L(h) {Lh:.4e} ({(L0 - Lh) / L0:+.3f}) L(uW) {L(tt['uW']):.4e} ({(L0 - L(tt['uW'])) / L0:+.3f}) "
                  f"L(restW) {(L0 - L(tt['restW'])) / L0:+.3f} L(uF) {(L0 - L(tt['uF'])) / L0:+.3f} L(restF) {(L0 - L(tt['restF'])) / L0:+.3f} | "
                  f"relief: L(low) {Ll:.4e} gain G/L(low) {G / Ll:+.4f}, share W {shW:+.3f} F {shF:+.3f} | "
                  f"sil {tt['zero'][0]:.3e}->{tt['h'][0]:.3e} pbr {tt['zero'][1]:.3e}->{tt['h'][1]:.3e} | "
                  f"grad cos h {g['cos_h']:+.3f} rel {g['cos_rel']:+.3f} uW {g['cos_uW']:+.3f} bands " + " ".join(f"{v:.2f}" for v in g["band_share"]))
            srow["set"][s] = dict(L0=L0, Lh=Lh, Llow=Ll, gain_h=(L0 - Lh) / L0, gain_uW=(L0 - L(tt["uW"])) / L0,
                                  gain_restW=(L0 - L(tt["restW"])) / L0, gain_uF=(L0 - L(tt["uF"])) / L0,
                                  gain_restF=(L0 - L(tt["restF"])) / L0, rel_gain=G / Ll, shW=shW, shF=shF,
                                  sil0=tt["zero"][0], silh=tt["h"][0], pbr0=tt["zero"][1], pbrh=tt["h"][1],
                                  cos_h=g["cos_h"], cos_rel=g["cos_rel"], grad_bands=g["band_share"],
                                  Lrel=L(tt["rel"]), silrel=tt["rel"][0], pbrrel=tt["rel"][1])
            print(f"   {s:10s} the relief alone on the run's own surface (h = 0 -> h_rel): total {(L(tt['rel']) - L0) / L0:+.3f}, "
                  f"silhouette {(tt['rel'][0] - tt['zero'][0]) / tt['zero'][0]:+.3f}, shading {(tt['rel'][1] - tt['zero'][1]) / tt['zero'][1]:+.3f}; "
                  f"the whole layer: total {(Lh - L0) / L0:+.3f}")
        if "static_fit" in r:
            srow["sfit"] = {}
            for s in SET:
                f = r["static_fit"][s]
                tot = sum(f["bands"])
                print(f"   static fit at {s:10s}: rms {f['rms_a']:.3f} a mean {f['mean_a']:+.3f} a, corr with D115's h {f['corr_h']:+.3f}, "
                      f"relief corr {f['corr_rel']:+.3f}, bands " + " ".join(f"{v / max(tot, 1e-30):.2f}" for v in f["bands"]) +
                      f", own terms {f['terms_own'][0]:.3e}/{f['terms_own'][1]:.3e}, at dense_ac {f['terms_dense_ac'][0]:.3e}/{f['terms_dense_ac'][1]:.3e}")
                srow["sfit"][s] = f
        summary[mesh]["rows"].append(srow)
json.dump(summary, open(out / "summary.json", "w"), indent=1)
print("\nwrote", out / "summary.json")
