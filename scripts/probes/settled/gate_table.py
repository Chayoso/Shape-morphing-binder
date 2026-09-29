"""gate_table.py DIR N TARGET [TARGET ...] — the S2 adoption-gate table: settled (st<N>_<t>) against the v3-grid-gs form
(v3<N>_<t>) for each target, both from DIR (archives, run JSONs, _start/_end stamps).
Per arm: silIoU, chamfer, det F min, wall (min) / accepted windows, stray particles (outside the largest connected
component at 1.5 end spacings), target surface farther than 1.5 target spacings from any particle, the outer layer's
window-to-window reversals (second-half share, longest streak), and the centre-of-mass drift (max over the morph, in
target spacings; source and target are centred, so momentum conservation keeps it at zero).
Gates (docs/experiments.md S2): silIoU >= v3 - 0.005, second-half reversals <= 25 %, COM drift <= 0.1 spacing; a
stall shows as a run that ends with a large target-surface gap and is read from the log, not here."""
import sys, os, json, numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from physmorph.sampling.orientation import orient_archive

D, N, targets = sys.argv[1], sys.argv[2], sys.argv[3:]


def outer(P, r):
    k = cKDTree(P); c = np.asarray(k.query_ball_point(P, r=2.0 * r, return_length=True, workers=-1))
    return c < 0.6 * np.median(c)


def arm(tag):
    npz = os.path.join(D, tag + "_render_full_dt_iso_nn.npz"); js = os.path.join(D, tag + ".json")
    if not (os.path.exists(npz) and os.path.exists(js)):
        return None
    z = np.load(npz, allow_pickle=True)
    frames, tgt, _, _ = orient_archive(z, npz)
    tgt = np.asarray(tgt, np.float32)
    rep = json.load(open(js))["arms"]["render_full_dt_iso_nn"]; m = rep["metrics"]; hist = rep["history"]
    dn = min(len(frames), int(z["deliver_n"])) if "deliver_n" in z.files else len(frames)
    x = np.asarray(frames[dn - 1], np.float32)
    sp_t = float(np.median(cKDTree(tgt).query(tgt, k=2, workers=-1)[0][:, 1]))
    sp_x = float(np.median(cKDTree(x).query(x, k=2, workers=-1)[0][:, 1]))
    # strays: particles outside the largest component at 1.5 end spacings
    pairs = cKDTree(x).query_pairs(1.5 * sp_x, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(x), len(x)))
    _, lab = connected_components(g, directed=False)
    strays = int(len(x) - np.bincount(lab).max())
    # target surface farther than 1.5 target spacings from any particle
    ot = outer(tgt, sp_t)
    gap = float((cKDTree(x).query(tgt[ot], workers=-1)[0] > 1.5 * sp_t).mean())
    # reversals of the outer layer between accepted windows
    ends = sorted({int(h["frame_end"]) - 1 for h in hist if h.get("frame_end") and not h.get("null_commit")})
    ends = [e for e in ends if 0 < e < dn]
    lay = np.nonzero(outer(x, sp_t))[0]
    prev_x = prev_d = None; corr = []
    for e in ends:
        xe = np.asarray(frames[e], np.float32)[lay]
        if prev_x is not None:
            d = xe - prev_x
            if prev_d is not None:
                corr.append(float((d * prev_d).sum() / max(np.linalg.norm(d) * np.linalg.norm(prev_d), 1e-30)))
            prev_d = d
        prev_x = xe
    corr = np.array(corr); half = corr[len(corr) // 2:]
    neg2 = float((half < 0).mean()) if len(half) else float("nan")
    streak = cur = 0
    for c in corr:
        cur = cur + 1 if c < 0 else 0; streak = max(streak, cur)
    # centre-of-mass drift
    com = np.array([np.asarray(frames[i], np.float64).mean(0) for i in range(0, dn, max(1, dn // 200))] + [x.astype(np.float64).mean(0)])
    drift = float(np.linalg.norm(com - com[0], axis=1).max()) / sp_t
    wall = None
    try:
        wall = (int(open(os.path.join(D, tag + "_end")).read()) - int(open(os.path.join(D, tag + "_start")).read())) / 60
    except Exception:
        pass
    return dict(sil=m.get("sil_iou"), ch=m.get("chamfer"), detF=m.get("detF_min"), wall=wall, win=len(ends),
                strays=strays, gap=gap, neg2=neg2, streak=streak, drift=drift)


def fmt(a):
    if a is None:
        return "  (pending)".ljust(78)
    w = f"{a['wall']:4.1f}" if a["wall"] is not None else "  - "
    return (f"{a['sil']:.4f} {a['ch']:.4f} {a['detF']:5.3f} {w}/{a['win']:<3d} {a['strays']:5d} {100 * a['gap']:5.1f}% "
            f"{100 * a['neg2']:4.0f}%/{a['streak']:<3d} {a['drift']:5.2f}")


print(f"{'target':12s} | {'settled: sil    chamf  detF  min/win strays gap   rev2/str COMsp':78s} | "
      f"{'v3 form: sil    chamf  detF  min/win strays gap   rev2/str COMsp':78s} | dSil    gates")
fails = []
for t in targets:
    s, v = arm(f"st{N}_{t}"), arm(f"v3{N}_{t}")
    verdict = "pending"
    if s is not None and v is not None:
        g = []
        if s["sil"] < v["sil"] - 0.005:
            g.append("sil")
        if not (s["neg2"] <= 0.25):
            g.append("reversal")
        if s["drift"] > 0.1:
            g.append("COM")
        verdict = "pass" if not g else "FAIL " + ",".join(g)
        if g:
            fails.append(t)
    d = f"{s['sil'] - v['sil']:+.4f}" if (s is not None and v is not None) else "   -   "
    print(f"{t:12s} | {fmt(s)} | {fmt(v)} | {d} {verdict}")
print(f"failing targets: {fails if fails else 'none'}")
