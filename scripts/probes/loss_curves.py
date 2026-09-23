"""Loss and wall-clock curves for one or more runs: the per-commit records from <run>.json
(loss, D_vol, D_render/sil, lambda) against the commit index and against wall-clock time
taken from the live-viewer commit packets' mtimes (<live>/<run>_<arm>/commits/*). Pairs a
render run with its physics-only twin when both are given.
Usage: python loss_curves.py OUT_DIR out.png run1 [run2 ...]   (run = JSON stem, e.g. hr_bunny)
"""
import glob
import json
import os
import sys

import numpy as np

OUT, png = sys.argv[1], sys.argv[2]
runs = sys.argv[3:]


def load(run):
    j = json.load(open(os.path.join(OUT, run + ".json")))
    arms = j.get("arms", {})
    arm = next(iter(arms))
    res = arms[arm]
    hist = [h for h in res.get("history", []) if "loss" in h]
    # wall clock from live packets
    live = sorted(glob.glob(os.path.join(OUT, "live", f"{run}_{arm}", "commits", "*")), key=os.path.getmtime)
    t = np.array([os.path.getmtime(p) for p in live])
    t = t - t[0] if len(t) else t
    return arm, res, hist, t


rows = {}
for r in runs:
    arm, res, hist, t = load(r)
    a = np.array([h["animation"] for h in hist])
    L = np.array([h["loss"] for h in hist], float)
    dv = np.array([h.get("d_vol", np.nan) for h in hist], float)
    ds = np.array([h.get("d_sil", h.get("d_pbr", np.nan)) or np.nan for h in hist], float)
    rows[r] = dict(arm=arm, a=a, L=L, dv=dv, ds=ds, t=t, metrics={**res.get("metrics", {}), "seconds": res.get("seconds"), "deliver_n": res.get("deliver_n")})
    n = len(hist)
    print(f"{r:28s} arm={arm} commits={n} loss {L[0]:.4g} -> {L[-1]:.4g} (x{L[-1]/max(L[0],1e-12):.3f}) "
          f"D_vol {dv[0]:.4g} -> {dv[-1]:.4g} | wall {t[-1]/60 if len(t) else float('nan'):.1f} min for {len(t)} packets "
          f"({(t[-1]/max(len(t)-1,1)) if len(t) > 1 else float('nan'):.1f} s/commit) | seconds {res.get('seconds')} | chamfer {res.get('metrics', {}).get('chamfer')} silIoU {res.get('metrics', {}).get('sil_iou')}")

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for r, d in rows.items():
        ax[0].semilogy(d["a"], d["L"], label=r)
        ax[1].semilogy(d["a"], d["dv"], label=r)
        if len(d["t"]) >= 2:
            tt = d["t"][: len(d["a"])] if len(d["t"]) >= len(d["a"]) else np.linspace(0, d["t"][-1], len(d["a"]))
            ax[2].semilogy(tt / 60.0, d["L"][: len(tt)], label=r)
    ax[0].set_xlabel("commit"); ax[0].set_ylabel("window loss L"); ax[0].set_title("loss vs commit")
    ax[1].set_xlabel("commit"); ax[1].set_ylabel("D_vol"); ax[1].set_title("mass matching vs commit")
    ax[2].set_xlabel("wall-clock [min]"); ax[2].set_ylabel("window loss L"); ax[2].set_title("loss vs wall-clock (live packet mtimes)")
    for a_ in ax:
        a_.grid(alpha=0.3); a_.legend(fontsize=7)
    plt.tight_layout(); fig.savefig(png, dpi=110); print("saved", png)
except Exception as e:
    print("no figure:", e)
