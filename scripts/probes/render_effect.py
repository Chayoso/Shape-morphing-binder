"""Does the render gradient change the physics trajectory? Three twins of one target:
render-on (the gallery run), physics-only (lambda 0 from the start) and the INTERVENTION
(render on, switched off from window K). Same seed, same particles.

    render_effect.py <out_dir> <target> <render_prefix> <phys_prefix> <cut_prefix> <K> [--png plot.png]

Reports the divergence |x_cut - x_render| and |x_phys - x_render| per archived frame (mean
over particles, in particle spacings). The causal signature: the cut twin coincides with the
render twin (divergence ~ 0, only round-off) up to the archived frame of window K and
departs after it, while the physics-only twin differs from the first window on. Also the
end metrics of each twin and, from the run json, the per-window render share of the step
(g_share / lam) where recorded."""
import json
import sys
import numpy as np
from scipy.spatial import cKDTree

args = [a for a in sys.argv[1:] if not a.startswith("--")]
png = None
if "--png" in sys.argv:
    png = sys.argv[sys.argv.index("--png") + 1]
    if png in args:
        args.remove(png)
# optional identical-configuration control run (the run-to-run noise floor) and the render run it
# is compared against (default: the render prefix)
p_ctrl = p_ref = None
if "--ctrl" in sys.argv:
    p_ctrl = sys.argv[sys.argv.index("--ctrl") + 1]
    if p_ctrl in args:
        args.remove(p_ctrl)
if "--ctrl_ref" in sys.argv:
    p_ref = sys.argv[sys.argv.index("--ctrl_ref") + 1]
    if p_ref in args:
        args.remove(p_ref)
out, T, p_r, p_p, p_c, K = args[0], args[1], args[2], args[3], args[4], int(args[5])
ARM = "render_full_dt_iso_nn"


def load(p):
    z = np.load(f"{out}/{p}_{T}_{ARM}.npz", allow_pickle=True)
    fr = np.asarray(z["frames"], np.float32)
    dn = int(z["deliver_n"]) if "deliver_n" in z.files else len(fr)
    return fr[:dn], np.asarray(z["tgt"], np.float32)


fr_r, tgt = load(p_r); fr_p, _ = load(p_p); fr_c, _ = load(p_c)
tree = cKDTree(tgt)
sp = float(np.median(tree.query(tgt, k=2, workers=-1)[0][:, 1]))
m = min(len(fr_r), len(fr_p), len(fr_c))
ks = np.arange(0, m, max(1, m // 250))
d_c = np.array([np.linalg.norm(fr_c[k] - fr_r[k], axis=1).mean() for k in ks]) / sp
d_p = np.array([np.linalg.norm(fr_p[k] - fr_r[k], axis=1).mean() for k in ks]) / sp
d_ctrl = ks_c = None
if p_ctrl:
    fr_ctrl, _ = load(p_ctrl)
    fr_ref = load(p_ref)[0] if p_ref else fr_r
    mc = min(len(fr_ctrl), len(fr_ref))
    ks_c = ks[ks < mc]
    d_ctrl = np.array([np.linalg.norm(fr_ctrl[k] - fr_ref[k], axis=1).mean() for k in ks_c]) / sp


def stride_of(p):
    try:
        j = json.load(open(f"{out}/{p}_{T}.json"))
        return int(j.get("archive_stride", 8))
    except Exception:
        return 8


# archived frame index of window K: T=20 steps per window, one archived frame per `stride` steps (+1 per commit)
stride = stride_of(p_c)
frames_per_window = 20 // stride + 1
k_cut = K * frames_per_window
print(f"{T}: spacing {sp:.4f} wu; frames render/phys/cut = {len(fr_r)}/{len(fr_p)}/{len(fr_c)}; "
      f"window {K} ~ archived frame {k_cut}")
before = d_c[ks < k_cut]; after = d_c[ks >= k_cut]
print(f"cut twin vs render twin: mean divergence BEFORE window {K}: {before.mean() if len(before) else float('nan'):.4f} spacings "
      f"(max {before.max() if len(before) else float('nan'):.4f}); AFTER: {after.mean() if len(after) else float('nan'):.3f} (end {d_c[-1]:.3f})")
print(f"phys twin vs render twin: first-window divergence {d_p[1] if len(d_p) > 1 else d_p[0]:.3f}, "
      f"mid {d_p[len(d_p) // 2]:.3f}, end {d_p[-1]:.3f} spacings")
if d_ctrl is not None:
    b = d_ctrl[ks_c < k_cut]; a = d_ctrl[ks_c >= k_cut]
    print(f"control ({p_ctrl}, identical configuration) vs {p_ref or p_r}: BEFORE window {K}: "
          f"{b.mean() if len(b) else float('nan'):.4f} spacings (max {b.max() if len(b) else float('nan'):.4f}); "
          f"AFTER: {a.mean() if len(a) else float('nan'):.3f} (end {d_ctrl[-1]:.3f}) — the run-to-run noise floor")
for p in (p_r, p_p, p_c):
    try:
        j = json.load(open(f"{out}/{p}_{T}.json"))
        arms = j.get("arms", j)
        r = arms.get(ARM, arms) if isinstance(arms, dict) else arms
        met = r.get("metrics", r)
        h = r.get("history") or []
        shares = [e.get("g_share") for e in h if isinstance(e, dict) and e.get("g_share") is not None]
        lams = [e.get("lambda", e.get("lam")) for e in h if isinstance(e, dict) and e.get("lambda", e.get("lam")) is not None]
        cos = [e.get("g_cos") for e in h if isinstance(e, dict) and e.get("g_cos") is not None]
        if cos:
            print(f"{p:14s} accepted-step cosine with the render gradient: mean {np.mean(cos):.3f}, "
                  f"share of windows with cos > 0: {np.mean(np.array(cos) > 0) * 100:.0f} %")
        print(f"{p:14s} chamfer {met.get('chamfer', '?')} silIoU {met.get('sil_iou', met.get('silIoU', '?'))} "
              f"windows {len(h)}  render share of the step: mean {np.mean(shares):.3f} (n={len(shares)})" if shares else
              f"{p:14s} chamfer {met.get('chamfer', '?')} silIoU {met.get('sil_iou', met.get('silIoU', '?'))} windows {len(h)}  lam mean {np.mean(lams) if lams else float('nan'):.3g}")
    except Exception as e:
        print(f"{p}: json not readable ({e})")
if png:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    if d_ctrl is not None:
        ax.plot(ks_c, d_ctrl, color="0.5", label="identical configuration re-run (noise floor)")
    ax.plot(ks, d_p, label="physics-only twin")
    ax.plot(ks, d_c, label=f"render switched off at window {K}")
    ax.axvline(k_cut, color="k", ls="--", lw=0.8)
    ax.set_xlabel("archived frame"); ax.set_ylabel("mean |x - x_render| (particle spacings)"); ax.grid(alpha=.3); ax.legend()
    ax.set_title(f"{T} ({len(tgt) // 1000}k): divergence from the render-on twin")
    fig.tight_layout(); fig.savefig(png, dpi=110)
    print("saved", png)
