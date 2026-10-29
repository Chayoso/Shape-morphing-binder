"""d175_video.py -- D175 deliverables in the studio look (D161 / D163's renderer: d163_render.render_rake, light-grey body,
grey gradient backdrop, the zbs display surface at the run's pitch). CPU (Embree via d157_lib). The camera is set from the
run's own target sample (the frames12 'tgt'), so any source / target pair frames correctly.

  sheet OUT.png KF.npz TWIN_NPZ FULL_NPZ T_K [W] [H]
        the keyframe at t_k against the bodies: rows = cameras (azimuth 0 / 90 deg, elevation 0.18), columns = the
        keyframe sample's display surface | dF_c + layer u (twin) at t_k | dF_c + layer u + render loss (full) at t_k
  sbs   OUT_DIR KF.npz TWIN_NPZ FULL_NPZ T_K STOP_RAW [W] [H] [AZ] [EL] [ORBIT] [WORKERS]
        twin | full side by side at every kept frame (an ended run held); the t_k frame held for 20 frames with the
        keyframe inset in both panels; then 20 held frames and ORBIT frames turning both end states. OUT_DIR/f%05d.jpg"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d161")
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d163")
import d157_lib as D7  # noqa: E402
from d163_render import display, label, render_rake  # noqa: E402

TWIN_L, FULL_L = "dF_c + layer u (twin)", "dF_c + layer u + render loss (full)"
_V = {}


def frame_of(npz):
    z = np.load(npz, allow_pickle=True)
    x_t = np.asarray(z["tgt"], np.float64)
    c = x_t.mean(0)
    rad = float(np.linalg.norm(x_t - c, axis=1).max())
    a = .708 * D7.median_kth_spacing(x_t, 8, 20000)
    return z, c, rad, a


def at(z, raw):
    raws = np.asarray(z["raws"])
    j = int(np.flatnonzero(raws == raw)[0]) if raw in raws else min(int(np.searchsorted(raws, raw)), len(raws) - 1)
    return np.asarray(z["frames"][j], np.float64), int(raws[j])


def sheet(argv):
    out, kf, twin, full, tk = argv[0], argv[1], argv[2], argv[3], int(argv[4])
    W = int(argv[5]) if len(argv) > 5 else 1200
    H = int(argv[6]) if len(argv) > 6 else 1000
    zt, c, rad, a = frame_of(twin)
    zf = np.load(full, allow_pickle=True)
    k = np.load(kf)
    xt, rt = at(zt, tk)
    xf, rf = at(zf, tk)
    surfs = [(display(np.asarray(k["samples"][0], np.float64), a), "keyframe P_k (its sample)", f"t_k = {tk / 240:.2f} s"),
             (display(xt, a), TWIN_L, f"t = {rt / 240:.2f} s"), (display(xf, a), FULL_L, f"t = {rf / 240:.2f} s")]
    dist = 3.6 * rad
    rows = []
    for view in ((0.0, 0.18), (np.pi / 2, 0.18)):
        rows.append(np.concatenate([label(render_rake(s, c, dist, view, W, H, 1.15 * rad / dist, rake=72.0), l, sub, scale=26)
                                    for s, l, sub in surfs], 1))
    from PIL import Image
    Image.fromarray(np.concatenate(rows, 0)).save(out)
    print("sheet done", out, flush=True)


def _ends(args):
    X, a = args
    s = display(X, a)
    return s.V, s.F, s.vn


def _frame(args):
    k, kind, payload = args
    g = _V
    W, H, view = g["W"], g["H"], g["view"]
    panels = []
    for side in ("twin", "full"):
        F, raws, lab = g[side]
        if kind in ("morph", "key"):
            r = payload
            ended = r > raws[-1]
            j = len(raws) - 1 if ended else int(np.searchsorted(raws, r))
            surf = g[f"end_{side}"] if ended else display(F[j], g["a"])
            vw = view
            sub = f"t = {r / 240.0:.2f} s"
            if kind == "key":
                sub += "  KEYFRAME"
            if side == "full" and r > g["stop"]:
                sub += "  settle: zero control"
            if ended:
                sub = f"t = {raws[-1] / 240.0:.2f} s (ended, held)"
        else:
            surf, vw = g[f"end_{side}"], (view[0] + payload, view[1])
            sub = f"end (t = {raws[-1] / 240.0:.2f} s), turned {np.degrees(payload):.0f} deg"
        img = render_rake(surf, g["look"], g["dist"], vw, W // 2, H, g["tan_h"], rake=72.0)
        if kind == "key" or (kind == "morph" and abs(payload - g["tk"]) <= 12):
            ins = g["inset"]
            img = img.copy()
            img[H - ins.shape[0] - 8:H - 8, W // 2 - ins.shape[1] - 8:W // 2 - 8] = ins
        panels.append(label(img, lab, sub, scale=26))
    from PIL import Image
    Image.fromarray(np.concatenate(panels, 1)).save(Path(g["out"]) / f"f{k:05d}.jpg", quality=93)
    return k


def sbs(argv):
    out, kf, twin, full, tk, stop = Path(argv[0]), argv[1], argv[2], argv[3], int(argv[4]), int(argv[5])
    opt = argv[6:] + [None] * 8
    W, H = int(opt[0] or 3840), int(opt[1] or 2160)
    az, el = float(opt[2] or 0.0), float(opt[3] or 0.18)
    orbit, workers = int(opt[4] or 160), int(opt[5] or 24)
    out.mkdir(parents=True, exist_ok=True)
    zt, c, rad, a = frame_of(twin)
    zf = np.load(full, allow_pickle=True)
    dist = 3.6 * rad
    k = np.load(kf)
    ks = display(np.asarray(k["samples"][0], np.float64), a)
    iw, ih = W // 8, H // 4
    inset = label(render_rake(ks, c, dist, (az, el), iw, ih, 1.15 * rad / dist, rake=72.0), "keyframe", scale=12)
    _V.update(W=W, H=H, view=(az, el), a=a, look=c, dist=dist, tan_h=1.15 * rad / dist, out=str(out), stop=stop, tk=tk,
              inset=inset, twin=(zt["frames"], np.asarray(zt["raws"]), TWIN_L),
              full=(zf["frames"], np.asarray(zf["raws"]), FULL_L))
    T = sorted(set(np.asarray(zt["raws"]).tolist()) | set(np.asarray(zf["raws"]).tolist()))
    from multiprocessing import get_context
    with get_context("fork").Pool(2) as pool:
        ends = pool.map(_ends, [(np.asarray(_V["twin"][0][-1]), a), (np.asarray(_V["full"][0][-1]), a)])
    _V["end_twin"], _V["end_full"] = (D7.Surf(V, F, vn=vn) for V, F, vn in ends)
    jobs, kk, key = [], 0, None
    for r in T:
        jobs.append((kk, "morph", r))
        kk += 1
        if r == tk:                                   # the keyframe held 20 frames: one render, copied
            jobs.append((kk, "key", r))
            key, kk = kk, kk + 20
    jobs += [(kk + i, "orbit", 0.0) for i in range(20)]
    jobs += [(kk + 20 + i, "orbit", 2 * np.pi * (i + 1) / orbit) for i in range(orbit)]
    todo = [j for j in jobs if not (out / f"f{j[0]:05d}.jpg").exists()]
    with get_context("fork").Pool(workers) as pool:
        for k in pool.imap_unordered(_frame, todo, chunksize=1):
            print("frame", k, flush=True)
    if key is not None:
        import shutil
        for i in range(1, 20):
            shutil.copyfile(out / f"f{key:05d}.jpg", out / f"f{key + i:05d}.jpg")
    print("frames done", kk + 20 + orbit, flush=True)


if __name__ == "__main__":
    {"sheet": sheet, "sbs": sbs}[sys.argv[1]](sys.argv[2:])
