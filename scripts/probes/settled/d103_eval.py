"""d103_eval.py LABEL=RUN_JSON[,DISPLAY_LOG] [...] — D103 on runs already made: the delivered window at the last render
weight (the code before) and at the epoch's largest (best_window now); at each: the run's own silhouette term and
transport energy, the kinetic energy over the ten windows up to it, the angular momentum at its end (the next
committed window's L_start, where the record has it), and with a display log (surface_layer_probe.py against the
independent sample) the display's mean over the last 20 kept frames up to its end."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np                                             # noqa: E402
from physmorph.pipeline.run.selection import best_window      # noqa: E402


def old_window(hist, n, tol, w_pbr=1.0):
    acc = [r for r in hist if r.get("frame_end") and not r.get("null_commit")
           and r.get("d_vol") is not None and np.isfinite(r.get("selection_merit", float("nan")))]
    epoch = max(r["selection_epoch"] for r in acc)
    acc = [r for r in acc if r["selection_epoch"] == epoch]
    lam = acc[-1].get("lambda") or 0.0
    merit = lambda r: r["selection_merit"] + (lam - (r.get("lambda") or 0.0)) * ((r.get("d_render") or 0.) + w_pbr * (r.get("d_pbr") or 0.))  # noqa: E731
    return int(min(acc, key=merit)["frame_end"])


def display(path, end):
    rows = [json.loads(l) for l in open(path) if l.startswith("{")]
    rows = [r for r in rows if str(r["state"]).isdigit() and "to_target" in r and int(r["state"]) <= end][-20:]
    m = lambda v, k: float(np.mean([(1 - r["to_target"][v][k]) if k == "iou" else r["to_target"][v][k] for r in rows]))  # noqa: E731
    return {f"{v}_{k}": m(v, k) for v in ("front", "crop", "back") for k in ("iou", "difference")}


for spec in sys.argv[1:]:
    name, rest = spec.split("=")
    run, disp = (rest.split(",") + [None])[:2]
    a = json.load(open(run))["arms"]["render_full_dt_iso_nn"]
    hist = a["history"]
    n_all = max(r["frame_end"] for r in hist if r.get("frame_end"))
    tol = a["config"].get("tol", 0.003) if isinstance(a.get("config"), dict) else 0.003
    ends = {"last weight": old_window(hist, n_all, tol), "largest weight": best_window(hist, n_all, tol)[0]}
    com = [r for r in hist if r.get("frame_end") and not r.get("null_commit")]
    print(f"== {name}: {len(com)} committed windows, frames {n_all}")
    for rule, end in ends.items():
        upto = [r for r in com if r["frame_end"] <= end]
        last = upto[-1]
        nxt = next((r for r in com if r["frame_end"] > end), None)
        Lend = float(np.linalg.norm(nxt["L_start"])) if nxt is not None and nxt.get("L_start") is not None else float("nan")
        kin = float(np.mean([r["kin"] for r in upto[-10:]]))
        line = (f"   {rule:14s}: window {last['animation']:3d} (frame {end}) | own silhouette {last['d_sil']:.3e}, shading {last['d_pbr']:.3e}, "
                f"transport {last['transport_energy']:.3e} | kinetic energy, last ten windows {kin:.2e} | |L| at the end {Lend:.2f}")
        if disp:
            d = display(disp, end)
            line += " | display 1-IoU / difference: " + "  ".join(f"{v} {d[v + '_iou']:.4f} / {d[v + '_difference']:.4f}" for v in ("front", "crop", "back"))
        print(line)
