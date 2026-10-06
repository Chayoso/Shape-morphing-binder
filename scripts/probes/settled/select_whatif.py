"""select_whatif.py RUN_JSON [...] — which window the delivered slice would end at if the selection scored every
window with another render weight (best_window's rule: one weight for all windows, the merit linear in it): the
last window's weight (the code), the largest weight of the run, and the weight at which the render part is half of
the merit at the window of the lowest silhouette. For each: the window, its silhouette term over the run's lowest,
its transport energy over the transport there."""
import json, sys
import numpy as np
out = []
for p in sys.argv[1:]:
    h = [r for r in json.load(open(p))["arms"]["render_full_dt_iso_nn"]["history"]
         if r.get("frame_end") and not r.get("null_commit") and r.get("selection_merit") is not None and r.get("d_sil") is not None]
    ep = max(r.get("selection_epoch", 0) for r in h)
    h = [r for r in h if r.get("selection_epoch", 0) == ep]
    lr = np.array([(r.get("d_render") or 0.) + (r.get("d_pbr") or 0.) for r in h])
    lam = np.array([r.get("lambda") or 0. for r in h])
    phys = np.array([r["selection_merit"] for r in h]) - lam * lr
    sil = np.array([r["d_sil"] for r in h]); tr = np.array([r["transport_energy"] for r in h])
    m = int(np.argmin(sil))
    w_half = phys[m] / max(lr[m], 1e-30)                          # render part = physics part at the best-silhouette window
    row = [p.split("/")[-1][:-5], len(h)]
    for w in (lam[-1], lam.max(), w_half):
        k = int(np.argmin(phys + w * lr))
        row += [h[k]["animation"], sil[k] / sil[m], tr[k] / tr[m]]
    out.append(row)
    print(f"{row[0]:16s} windows {row[1]:3d} | last weight: window {row[2]:3d} sil {row[3]:.2f} transport {row[4]:.2f} | "
          f"largest weight: window {row[5]:3d} sil {row[6]:.2f} transport {row[7]:.2f} | half-half weight: window {row[8]:3d} sil {row[9]:.2f} transport {row[10]:.2f}")
a = np.array([r[1:] for r in out], float)
for j, name in ((2, "last weight (the code)"), (5, "largest weight"), (8, "half-half weight")):
    print(f"{name:24s}: silhouette over the run's lowest, median {np.median(a[:, j]):.2f} (above 1.2 on {int((a[:, j] > 1.2).sum())}); "
          f"transport over the transport there, median {np.median(a[:, j + 1]):.2f}")
