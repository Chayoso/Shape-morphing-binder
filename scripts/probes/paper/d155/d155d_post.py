"""d155d_post.py -- D155d forensic (CPU, records only): what the body gets after it arrives. Per run (the tag's V seeds,
A3G L97, A3D L97 / L99 / L101): the arrival (D155b's definition: u's gate over half the layer; on the ramp not before s
= 1), the delivered end, the time and the committed windows between them, the step scale (the selection's anneal)
at the arrival and its mean after it, the share of windows after the arrival that improved the merit, the stop reason
(the log's last line of the selection), and the render's share of the merit after the arrival."""
import json
import re

O = "/data/relcfd/chayo/physmorph_v2/output/gpu/d155/gate"
FPS = 240.0


def read(name):
    a = next(iter(json.load(open(f"{O}/{name}.json"))["arms"].values()))
    dn = int(a["deliver_n"])
    acc = [h for h in a["history"] if h.get("frame_end") and not h.get("null_commit")]
    t_u = next((h["frame_end"] / FPS for h in acc if (h.get("u_gate") or 0) >= 0.5), 0.0)
    t_s = next((h["frame_end"] / FPS for h in acc if (h.get("morph_s") if h.get("morph_s") is not None else 1.0) >= 1.0), 0.0)
    ta = max(t_u, t_s)
    post = [h for h in acc if ta - 1e-9 < h["frame_end"] / FPS <= dn / FPS + 1e-9]
    allpost = [h for h in a["history"] if (h.get("frame_end") or 0) / FPS > ta - 1e-9 or h.get("null_commit")]
    at = min(acc, key=lambda h: abs(h["frame_end"] / FPS - ta))
    log = open(f"{O}/{name}.log").read()
    stop = re.findall(r"\[v2\] (anim \d+: [^\n]*early stop[^\n]*|converged at anim \d+[^\n]*|frozen after[^\n]*|delivery merit "
                      r"plateau[^\n]*)", log)
    lam = [(h.get("lambda") or 0.0) * ((h.get("d_render") or 0.0) + (h.get("d_pbr") or 0.0)) for h in post]
    share = [x / h["selection_merit"] for x, h in zip(lam, post) if h.get("selection_merit")]
    return dict(arrive=ta, end=dn / FPS, after=dn / FPS - ta, windows_after=len(post),
                anneal_at=at.get("anneal"), anneal_after=(sum(h.get("anneal") or 0 for h in post) / len(post)) if post else None,
                improved_after=(sum(h.get("improved") or 0 for h in post) / len(post)) if post else None,
                render_share_after=(sum(share) / len(share)) if share else None,
                total_windows=len(a["history"]), stop=stop[-1] if stop else "budget or none logged")


for m in ("bunny", "dragon", "teapot", "armadilo", "homer"):
    print(m)
    for arm in ("V97", "V99", "V101", "A3GL97", "A3DL97", "A3DL99", "A3DL101"):
        try:
            r = read(f"{m}40k_{arm}")
        except OSError:
            continue
        f = lambda x, nd=2: "-" if x is None else f"{x:.{nd}f}"  # noqa: E731
        print(f"  {arm:8s} arrive {r['arrive']:.2f} s, end {r['end']:.2f} s, after {r['after']:.2f} s in {r['windows_after']} "
              f"windows; anneal at arrival {f(r['anneal_at'])}, after {f(r['anneal_after'])}; improved after "
              f"{f(r['improved_after'])}; render share of the merit after {f(r['render_share_after'], 3)}; stop: {r['stop']}")
