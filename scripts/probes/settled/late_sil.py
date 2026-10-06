"""late_sil.py RUN_DIR — D100: do the late windows trade the silhouette for the transport? Per mesh and arm (O, D): the
committed windows, the delivered (best) window, the run's own silhouette term there against its lowest over the run,
and the transport energy there against at the window of that lowest silhouette."""
import glob, json, os, sys
import numpy as np
rows = []
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*_[OD].json"))):
    tag = os.path.basename(p)[:-5]
    a = json.load(open(p))["arms"]["render_full_dt_iso_nn"]
    h = [r for r in a["history"] if r.get("frame_end") and not r.get("null_commit") and r.get("d_sil") is not None]
    best = (a.get("truncation") or {}).get("best_animation", h[-1]["animation"])
    hb = next((r for r in h if r["animation"] == best), h[-1])
    hm = min(h, key=lambda r: r["d_sil"])
    rows.append((tag, len(h), best, hb["d_sil"] / hm["d_sil"], hm["animation"], hb["transport_energy"] / max(hm["transport_energy"], 1e-30)))
for arm in "OD":
    r = [x for x in rows if x[0].endswith("_" + arm)]
    ratio = np.array([x[3] for x in r])
    print(f"arm {arm}: {len(r)} meshes | commits median {np.median([x[1] for x in r]):.0f} | silhouette at the delivered window over its run minimum: "
          f"median {np.median(ratio):.2f}, above 1.2 on {int((ratio > 1.2).sum())}, above 1.5 on {int((ratio > 1.5).sum())}")
    for x in r:
        if x[3] > 1.2:
            print(f"   {x[0]:14s} commits {x[1]:3d} delivered {x[2]:3d} sil/min {x[3]:.2f} (min at window {x[4]}) transport there / at the min {x[5]:.2f}")
