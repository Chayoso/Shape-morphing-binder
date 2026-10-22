"""d161_windows.py OUT_MD RUN_PREFIX [RUN_PREFIX ...] -- D161: per committed window of each run, from its JSON record (no
recomputation): the time (frame_end / 240 s), lambda, the physics gradient norm |g_phys| and lambda x the projected
render gradient norm |g_render| (the rule holds lambda |g_r| = 0.5 |g_p|; g_share = the render's share of the step),
the step actually taken (alpha_last: Adam's accepted search length, the u step a coordinate takes an iteration, in
pitches; move: the mean displacement of the window, pitches), the line search's accepted / rejected iterations and
trials, the outer decision (accepted, improved, stale count) and the stop (actual_stop of the last window). The pitch
is read from the run's own target sample (0.708 x the median 8th-neighbour distance). Arrival: the first window whose
transport energy is below 2 x its final value."""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

out, prefixes = sys.argv[1], sys.argv[2:]
L = []
for pre in prefixes:
    js = json.load(open(pre + ".json"))
    arm = list(js["arms"].values())[0]
    z = np.load(pre + "_frames12.npz", allow_pickle=True)
    tgt = np.asarray(z["tgt"], np.float64)
    sub = tgt[np.random.default_rng(0).choice(len(tgt), 20000, replace=False)]
    a = 0.708 * float(np.median(cKDTree(sub).query(sub, 9)[0][:, 8])) * (20000 / len(tgt)) ** (1 / 3)
    H = arm["history"]
    com = [h for h in H if h.get("frame_end") and not h.get("null_commit")]
    te = np.array([h.get("transport_energy") or np.nan for h in com])
    arrive = int(np.argmax(te < 2 * te[-1])) if len(te) else 0
    name = pre.split("/")[-1]
    L += [f"## {name} ({len(H)} windows judged, {len(com)} committed; arrival at committed window {arrive}; "
          f"stop: {com[-1].get('actual_stop') if com else None}; converged: {arm.get('converged')})", "",
          "| win | t (s) | lambda | g_phys | lambda g_render | g_share | alpha_last (a) | move (a) | acc / rej it | ls trials | "
          "outer acc | improved | stale | transport E |", "|" + "---|" * 14]
    for i, h in enumerate(com):
        if i < arrive - 1 and i % 3:
            continue
        gp, gr, lam = h.get("g_phys_norm"), h.get("g_rend_norm"), h.get("lambda") or 0.0
        L.append(f"| {h['animation']} | {h['frame_end'] / 240:.2f} | {lam:.2e} | {gp if gp is None else f'{gp:.2e}'} | "
                 f"{'-' if gr is None else f'{lam * gr:.2e}'} | {h.get('g_share')} | {(h.get('alpha_last') or 0) / a:.3f} | "
                 f"{(h.get('move') or 0) / a:.4f} | {h.get('accepted')} / {h.get('rejected')} | {h.get('ls_trials')} | "
                 f"{h.get('outer_accepted')} | {h.get('improved')} | {h.get('stale')} | {h.get('transport_energy'):.2e} |")
    rej = [h for h in H if not (h.get("frame_end") and not h.get("null_commit"))]
    L += ["", f"windows not committed (null or rejected): {len(rej)}; their animations: {[h.get('animation') for h in rej][:20]}", ""]
open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
