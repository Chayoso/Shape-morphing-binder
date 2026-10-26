"""ri_endmap.py MESH OUT_DIR -- render influence (2026-10-22; CPU, measurement only): where on the END surface did the
render loss change the shape, against the run-to-run floor.

States (the tag freeze-2026-10-15, 300k, seed 97 unless said):
  twin  P97   "dF_c + layer u"                         d145t_b1/{m}300k_P97_end1.npz
  full  L97   "dF_c + layer u + render loss (full)"     d145t_b1/{m}300k_L97_end1.npz   (the delivered run, B1)
  full  RG97  the same arm and seed run again (D145 B7) d145t_b7/{m}300k_RG97_frames12.npz, last kept frame
  full  T97   the same arm and seed run again (D159)    d159/{m}300k_T97_end1.npz
  twin  P99 / P99' the twin at seed 99 run twice (B1 and D159's re-run): the twin's own run-to-run floor
Surface shown: the twin's display surface (zbs, marching cubes at a / 2, D157's field_mesh).
Value at each vertex v: delta(v) = f_P97(v) - f_full(v), f the exterior field the loss reads (Zhu-Bridson, radius 3,
offset 0.8 a): how far the full arm's surface lies OUTSIDE (+) or inside (-) the twin's there, in pitches.
Floor at v: sigma(v)^2 = the mean over the four re-run pairs (L97-RG97, L97-T97, RG97-T97, P99-P99') of their own
difference squared at v (the P99 pair read at v mapped into the seed-99 frame, lengths mapped back), averaged over the
surface within 2 pitches (Gaussian weights). A vertex is 'beyond noise' if |delta| > 2 sigma.
Exact-mesh check: each state's exterior point near v along the twin's normal, q = v - f(v) n, its signed distance
from the target mesh minus D157's offset c (the own sample's mean exterior offset): the full arm moved toward the mesh at
v if |g_full| < |g_twin|.
Outputs: OUT_DIR/endmap_{m}.json (numbers), endmap_{m}_vals.npz (V, F, delta, sigma; float32), and studio-look
renders with the map (ri_endrender.py reads the npz)."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d156")
import physmorph  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
import d157_lib as D7  # noqa: E402
from physmorph.render.exterior import ZhuBridson  # noqa: E402

torch.set_grad_enabled(False)
G = "/data/relcfd/chayo/physmorph_v2/output/gpu"


def last_frame(p):
    z = np.load(p, allow_pickle=True)
    return np.asarray(z["frames"][-1], np.float64), z


def frame_params(mesh, seed):
    m = np.load(f"{G}/d154/meshes/{mesh}_{seed}.npz")
    return float(m["scale"]), np.asarray(m["offset"], np.float64)


def to_seed(x97, mesh, seed):
    """seed-97 frame -> seed-`seed` frame (the inverse of d159_course.frame_map), and the length factor back."""
    s97, o97 = frame_params(mesh, 97)
    ss, os_ = frame_params(mesh, seed)
    return ((x97 / s97 + o97) - os_) * ss, s97 / ss


def field_at(x, a, P, chunk=400000):
    f = ZhuBridson(torch.as_tensor(x), a)
    out = np.empty(len(P))
    for i in range(0, len(P), chunk):
        out[i:i + chunk] = f(torch.as_tensor(P[i:i + chunk]), grad=False)[0].numpy()
    return out


def smooth_on(V, val, w, a, cell=1.5):
    """The area-weighted mean of `val` over the vertices in the 3 x 3 x 3 cells (side `cell` pitches) around each
    vertex's cell: a box about 4.5 pitches wide."""
    ijk = np.floor(V / (cell * a)).astype(np.int64)
    ijk -= ijk.min(0)
    P = int(ijk.max()) + 3
    key = (ijk[:, 0] * P + ijk[:, 1]) * P + ijk[:, 2]
    uk, inv = np.unique(key, return_inverse=True)
    sw = np.bincount(inv, weights=w)
    sv = np.bincount(inv, weights=w * val)
    cu = np.stack([uk // (P * P), (uk // P) % P, uk % P], 1)
    num, den = np.zeros(len(uk)), np.zeros(len(uk))
    for o in np.ndindex(3, 3, 3):
        c = cu + np.array(o) - 1
        kk = (c[:, 0] * P + c[:, 1]) * P + c[:, 2]
        pos = np.clip(np.searchsorted(uk, kk), 0, len(uk) - 1)
        hit = uk[pos] == kk
        num[hit] += sv[pos[hit]]
        den[hit] += sw[pos[hit]]
    return (num / np.maximum(den, 1e-30))[inv]


def main():
    mesh, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    xP, zP = last_frame(f"{G}/d145t_b1/{mesh}300k_P97_end1.npz")
    tgt = np.asarray(zP["tgt"], np.float64)
    a = .708 * D7.median_kth_spacing(tgt.astype(np.float32), 8, 20000)
    full = {"L97": last_frame(f"{G}/d145t_b1/{mesh}300k_L97_end1.npz")[0],
            "RG97": last_frame(f"{G}/d145t_b7/{mesh}300k_RG97_frames12.npz")[0],
            "T97": last_frame(f"{G}/d159/{mesh}300k_T97_end1.npz")[0]}
    # is the last kept frame the delivered end? (L97 has both archives)
    xLk = last_frame(f"{G}/d145t_b1/{mesh}300k_L97_frames12.npz")[0]
    end_vs_kept = float(np.abs(xLk - full["L97"]).max() / a)
    del xLk
    tw99 = {"P99": last_frame(f"{G}/d145t_b1/{mesh}300k_P99_end1.npz")[0],
            "P99r": last_frame(f"{G}/d159/{mesh}300k_P99_end1.npz")[0]}
    surf, info = D7.field_mesh("zbs", xP.astype(np.float32), a)
    V, F, n = surf.V, surf.F, surf.vn
    print(f"{mesh}: a {a:.5f}, display mesh {info}, end vs last kept frame (L97) {end_vs_kept:.3g} pitch, "
          f"{time.time() - t0:.0f} s", flush=True)
    f = {"P97": field_at(xP, a, V)}
    for k, x in full.items():
        f[k] = field_at(x, a, V)
    V99, lf = to_seed(V, mesh, 99)
    tgt99 = np.asarray(np.load(f"{G}/d145t_b1/{mesh}300k_P99_end1.npz")["tgt"], np.float64)
    a99 = .708 * D7.median_kth_spacing(tgt99.astype(np.float32), 8, 20000)
    for k, x in tw99.items():
        f[k] = field_at(x, a99, V99) * lf
    print(f"fields done {time.time() - t0:.0f} s", flush=True)
    ok = np.all(np.stack([np.isfinite(v) for v in f.values()]), 0)
    dl = {k: (f["P97"] - f[k]) / a for k in full}                     # + : the full arm's surface further out
    pairs = {"L97-RG97": f["L97"] - f["RG97"], "L97-T97": f["L97"] - f["T97"], "RG97-T97": f["RG97"] - f["T97"],
             "P99-P99r": f["P99"] - f["P99r"]}
    pairs = {k: v / a for k, v in pairs.items()}
    # vertex areas (a third of each face's)
    fa = surf.area
    va = np.zeros(len(V))
    for k in range(3):
        np.add.at(va, F[:, k], fa / 3)
    A = va[ok].sum()
    s2 = np.mean(np.stack([np.where(ok, v, 0.0) ** 2 for v in pairs.values()]), 0)
    sig = np.sqrt(smooth_on(V, s2, va * ok, a))
    delta = np.where(ok, dl["L97"], 0.0)
    beyond = ok & (np.abs(delta) > 2 * sig)
    # exact-mesh check
    m = np.load(f"{G}/d154/meshes/{mesh}_97.npz")
    t, _ = D7.close_holes(np.asarray(m["V"], np.float64), np.asarray(m["F"], np.int64))
    if t.volume < 0:
        t.invert()
    raw = D7.Surf(np.asarray(t.vertices), np.asarray(t.faces))
    c_a = None
    for p in sorted(Path(f"{G}/d157").glob("read300_*.jsonl")):
        for ln in open(p):
            r = json.loads(ln)
            if r.get("mesh") == mesh and r.get("seed") == 97 and "ctx" in r:
                c_a = r["ctx"]["c_a"]
                break
        if c_a is not None:
            break
    g = {}
    for k in ("P97", "L97", "RG97", "T97"):
        q = V - f[k][:, None] * n
        g[k] = raw.signed(q) / a - c_a                                    # pitches from the corrected referee
    toward = {}
    for k in full:
        closer = np.abs(g[k]) < np.abs(g["P97"])
        toward[k] = dict(all=float((va * closer)[ok].sum() / A),
                         beyond=float((va * closer)[beyond].sum() / max((va[beyond]).sum(), 1e-30)),
                         mean_abs_err_full_beyond=float((va * np.abs(g[k]))[beyond].sum() / max(va[beyond].sum(), 1e-30)),
                         mean_abs_err_twin_beyond=float((va * np.abs(g["P97"]))[beyond].sum() / max(va[beyond].sum(), 1e-30)))
    wrms = lambda v, msk=ok: float(np.sqrt((va * v ** 2)[msk].sum() / va[msk].sum()))   # noqa: E731
    corr = lambda u, v: float(np.corrcoef(u[ok], v[ok])[0, 1])                          # noqa: E731
    res = dict(mesh=mesh, pitch=a, c_a=c_a, display_mesh=info, end_vs_last_kept_pitch=end_vs_kept,
               valid_share=float(ok.mean()),
               delta_rms={k: wrms(v) for k, v in dl.items()},
               delta_p99_abs=float(np.percentile(np.abs(delta[ok]), 99)),
               pair_rms={k: wrms(v) for k, v in pairs.items()},
               sigma_median=float(np.median(sig[ok])),
               beyond_share=float(va[beyond].sum() / A),
               beyond_out_share=float(va[beyond & (delta > 0)].sum() / A),
               beyond_in_share=float(va[beyond & (delta < 0)].sum() / A),
               # the same rule applied to each re-run pair (noise against noise; sigma includes that pair): the share
               # the map would colour if the render did nothing
               beyond_share_pairs={k: float(va[ok & (np.abs(v) > 2 * sig)].sum() / A) for k, v in pairs.items()},
               beyond_share_pairs_loo={k: float(va[ok & (np.abs(v) > 2 * np.sqrt(smooth_on(
                   V, np.mean(np.stack([np.where(ok, u, 0.0) ** 2 for kk, u in pairs.items() if kk != k]), 0),
                   va * ok, a)))].sum() / A) for k, v in pairs.items()},
               corr_full_runs={"L97~RG97": corr(dl["L97"], dl["RG97"]), "L97~T97": corr(dl["L97"], dl["T97"]),
                               "RG97~T97": corr(dl["RG97"], dl["T97"])},
               corr_pairs_same={"L97-RG97~L97-T97": corr(pairs["L97-RG97"], pairs["L97-T97"])},
               mean_delta_three_rms=wrms((dl["L97"] + dl["RG97"] + dl["T97"]) / 3),
               toward_mesh=toward,
               err_corrected_referee_rms={k: wrms(v) for k, v in g.items()},
               seconds=time.time() - t0)
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(out / f"endmap_{mesh}.json", "w"), indent=1)
    np.savez_compressed(out / f"endmap_{mesh}_vals.npz", V=V.astype(np.float32), F=F.astype(np.int32),
                        n=n.astype(np.float32), delta=delta.astype(np.float32), sigma=sig.astype(np.float32),
                        ok=ok, center=tgt.mean(0), radius=float(np.linalg.norm(tgt - tgt.mean(0), axis=1).max()),
                        pitch=a)


if __name__ == "__main__":
    main()
