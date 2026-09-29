"""window_reversal.py NPZ [NPZ ...] — oscillation and surface churn from the delivered frames, the same for every run.
Window boundaries come from the run JSON's accepted history (frame_end), so runs with different frames per window
compare window by window. For the outer layer of the END cloud (count-based, the target spacing as radius):
  step_w   = median |x(end of window w) - x(end of window w-1)| over the layer, in target spacings
  corr_w   = cosine between the layer's displacement field of window w and of window w-1 (all layer particles
             stacked); negative = the surface reverses from one window to the next (breathing)
  normal_w = median |normal component| of the step (normal = away from the local 33-NN centroid of the end cloud)
Printed: the last 20 windows' medians, the share of negative-corr windows over the whole run and over its second
half, the longest run of consecutive negative windows, and the end-window step."""
import sys, json, numpy as np
from scipy.spatial import cKDTree
from physmorph.sampling.orientation import orient_archive
for path in sys.argv[1:]:
    z = np.load(path, allow_pickle=True)
    frames, tgt, _, orient = orient_archive(z, path)
    tgt = np.asarray(tgt, np.float32)
    rep = json.load(open(path.replace("_render_full_dt_iso_nn.npz", ".json")))
    hist = rep["arms"]["render_full_dt_iso_nn"]["history"]
    ends = sorted({int(h["frame_end"]) - 1 for h in hist if h.get("frame_end") and not h.get("null_commit")})
    dn = min(len(frames), int(z["deliver_n"])) if "deliver_n" in z.files else len(frames)
    ends = [e for e in ends if 0 < e < dn]
    sp_t = float(np.median(cKDTree(tgt).query(tgt, k=2, workers=-1)[0][:, 1]))
    XE = np.asarray(frames[dn - 1], np.float32)
    k = cKDTree(XE); cnt = np.asarray(k.query_ball_point(XE, r=2.0 * sp_t, return_length=True, workers=-1))
    lay = np.nonzero(cnt < 0.6 * np.median(cnt))[0]
    _, nb = k.query(XE[lay], k=33, workers=-1)
    nrm = XE[lay] - XE[nb].mean(1); nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    prev_x, prev_d, rows = None, None, []
    for e in ends:
        x = np.asarray(frames[e], np.float32)[lay]
        if prev_x is not None:
            d = x - prev_x
            step = float(np.median(np.linalg.norm(d, axis=1))) / sp_t
            nstep = float(np.median(np.abs((d * nrm).sum(1)))) / sp_t
            corr = (float((d * prev_d).sum() / max(np.linalg.norm(d) * np.linalg.norm(prev_d), 1e-30))
                    if prev_d is not None else np.nan)
            rows.append((e, step, nstep, corr))
            prev_d = d
        prev_x = x
    R = np.array(rows)
    c = R[:, 3]; valid = np.isfinite(c); neg = (c < 0) & valid
    half = np.arange(len(R)) >= len(R) // 2
    longest = cur = 0
    for v in neg:
        cur = cur + 1 if v else 0; longest = max(longest, cur)
    last = R[-20:]
    print(f"{path.split('/')[-1]}: {len(ends)} accepted windows, layer {len(lay)} particles, target spacing {sp_t:.4f} wu")
    print(f"   last 20 windows: step {np.median(last[:, 1]):.3f} sp (normal {np.median(last[:, 2]):.3f}), corr median "
          f"{np.nanmedian(last[:, 3]):+.2f}, negative {int(((last[:, 3] < 0)).sum())}/20")
    print(f"   whole run: negative-corr windows {int(neg.sum())}/{int(valid.sum())} ({100 * neg.sum() / max(valid.sum(), 1):.0f} %), "
          f"second half {int((neg & half).sum())}/{int((valid & half).sum())}, longest negative streak {longest}; "
          f"end-window step {R[-1, 1]:.4f} sp")
