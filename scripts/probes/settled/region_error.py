"""region_error.py NPZ [NPZ ...] — the END state's accuracy against the target, by region, the same for every run.
Two-sided distances between the delivered end cloud x and the target cloud (both oriented as the probes do):
  cover   = target point  -> nearest end particle    (what the target asks for and the body does not reach)
  excess  = end particle  -> nearest target point    (where the body sits off the target)
and the same restricted to the OUTER LAYERS (surface to surface: the relief the eye reads):
  s_cover = target surface point -> nearest end SURFACE particle
  s_excess= end surface particle -> nearest target SURFACE point
Regions by height in the oriented frame: ears (y >= 1.96, the pure-ear base of ear_slab), upper (0.9 <= y < 1.96: the
head and the upper back), lower (y < 0.9). Medians / p95 / max in units of the TARGET spacing (nearest-neighbour)."""
import sys, numpy as np
from scipy.spatial import cKDTree
from physmorph.sampling.orientation import orient_archive
def outer(P, sp):
    k = cKDTree(P); cnt = np.asarray(k.query_ball_point(P, r=2.0 * sp, return_length=True, workers=-1))
    return cnt < 0.6 * np.median(cnt)
def stats(d, sp):
    return f"{np.median(d) / sp:5.2f} {np.percentile(d, 95) / sp:5.2f} {d.max() / sp:6.2f}" if len(d) else "   -     -      -  "
REG = [("ears  y>=1.96", lambda y: y >= 1.96), ("upper 0.9-1.96", lambda y: (y >= 0.9) & (y < 1.96)), ("lower y<0.9", lambda y: y < 0.9),
       ("all", lambda y: np.ones(len(y), bool))]
for path in sys.argv[1:]:
    z = np.load(path, allow_pickle=True)
    frames, tgt, _, orient = orient_archive(z, path)
    tgt = np.asarray(tgt, np.float32)
    dn = min(len(frames), int(z["deliver_n"])) if "deliver_n" in z.files else len(frames)
    x = np.asarray(frames[dn - 1], np.float32)
    sp_t = float(np.median(cKDTree(tgt).query(tgt, k=2, workers=-1)[0][:, 1]))
    sp_x = float(np.median(cKDTree(x).query(x, k=2, workers=-1)[0][:, 1]))
    ot, ox = outer(tgt, sp_t), outer(x, sp_t)      # one classification radius (the target spacing) for both clouds
    kx, kt = cKDTree(x), cKDTree(tgt); kxs, kts = cKDTree(x[ox]), cKDTree(tgt[ot])
    cover = kx.query(tgt, workers=-1)[0]; excess = kt.query(x, workers=-1)[0]
    s_cover = kxs.query(tgt[ot], workers=-1)[0]; s_excess = kts.query(x[ox], workers=-1)[0]
    s_any = kx.query(tgt[ot], workers=-1)[0]          # target surface -> nearest end particle of ANY depth (dents, under-fill)
    print(f"{path.split('/')[-1]} (orient {orient}): end frame {dn - 1}; target spacing {sp_t:.4f} wu, end spacing {sp_x:.4f}; "
          f"surface points target {int(ot.sum())}, end {int(ox.sum())}")
    print(f"   {'region':15s} | {'cover med p95 max':>19s} | {'excess med p95 max':>19s} | {'s_cover med p95 max':>19s} | {'s_excess med p95 max':>19s}   (target spacings)")
    for name, f in REG:
        mt, mx = f(tgt[:, 1]), f(x[:, 1]); mts, mxs = f(tgt[ot][:, 1]), f(x[ox][:, 1])
        print(f"   {name:15s} | {stats(cover[mt], sp_t):>19s} | {stats(excess[mx], sp_t):>19s} | {stats(s_cover[mts], sp_t):>19s} | {stats(s_excess[mxs], sp_t):>19s}")
        print(f"   {'':15s} | target surface -> nearest particle of any depth (med p95 max): {stats(s_any[mts], sp_t)}; beyond 1.5 sp: {float((s_any[mts] > 1.5 * sp_t).mean()) if mts.any() else float('nan'):.4f}")
    big = lambda d, m: float((d[m] > 2.0 * sp_t).mean()) if m.any() else float("nan")
    print("   share beyond 2 target spacings: " + ", ".join(
        f"{name.split()[0]} cover {big(cover, f(tgt[:, 1])):.4f} / s_cover {big(s_cover, f(tgt[ot][:, 1])):.4f} / s_excess {big(s_excess, f(x[ox][:, 1])):.4f}"
        for name, f in REG))
