"""d173_stripread.py OUT_DIR N SEEDS [ARMS] -- D173: the stripes scene's runs read (one card): for each seed the twin
(output/gpu/d173/runs/strp{N}k_T_{s}) and the full arm(s) (strp{N}k_{ARM}_{s}, default Rxn = Rx with --colour_kernel nearest).
Per arm: colour r (D160's fast reader, OUT_DIR/read_colour.jsonl: (twin mis - mis) / (twin mis - floor)), the held-out
PSNR / SSIM of the end state's colour pictures against the target's (d173_texprobe.pictures / compare: the 46 held-out
views at the run's resolution, the disc colours by the loss's kernel and by the nearest particle; the ceiling an
independent sample painted by the scene), the edge widths (d173_edge2.py, OUT_DIR/edge.jsonl: each stripe edge's 10-90 %
rise, logistic, pitches; the mean of the two), Cohen's kappa of the outer layer's own paint against the target paint at
its place (0 = chance, 1 = the stripes), the render's share of each block's step (d173_shares.py), rest (v end / twin) and
shape (sil_iou, thin). Writes OUT_DIR/summary.md and OUT_DIR/verdict.json."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from physmorph import gpu  # noqa: E402,F401  (first: CuPy before torch)
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d173")
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d170")
import d173_texprobe as TP  # noqa: E402
import d173_shares as SH  # noqa: E402
import d170_read as R7  # noqa: E402
from physmorph.pipeline.appearance import paint, target_red  # noqa: E402

D, N, SEEDS = Path(sys.argv[1]), int(sys.argv[2]), [int(s) for s in sys.argv[3].split(",")]
ARMS = ["T"] + (sys.argv[4].split(",") if len(sys.argv) > 4 else ["Rxn"])
RUNS = Path("/data/relcfd/chayo/physmorph_v2/output/gpu/d173/runs")
TGT = "assets/bunny.obj"
NAMES = dict(T="twin (lambda = 0)", Rxn="Rx, nearest colour (full)", Rx="Rx (full)")
RED = np.array([0.9, 0.25, 0.2])


def kappa(a, b):
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / max(1 - pe, 1e-12)


def layer(X):
    """d173_edge.layer (the pipeline's outer-layer rule), copied: d173_edge parses its own argv on import."""
    from scipy.spatial import cKDTree
    sub = X[np.random.default_rng(0).choice(len(X), min(20000, len(X)), replace=False)]
    sp = float(np.median(cKDTree(sub).query(sub, 9)[0][:, 8])) * (len(sub) / len(X)) ** (1 / 3)
    _, nb = cKDTree(X).query(X, 33, workers=16)
    return np.linalg.norm(X - X[nb[:, 1:]].mean(1), axis=1) >= 0.5 * sp


def main():
    col = {}
    for x in open(D / "read_colour.jsonl"):
        if x.strip():
            r = json.loads(x)
            col[r["name"]] = r
    edge = {}
    for x in open(D / "edge.jsonl"):
        if x.strip():
            r = json.loads(x)
            edge[r["name"]] = r
    rows = ["| arm | seed | colour r | mis end | kappa of the outer layer (0 = chance) | PSNR / SSIM held (kernel) | PSNR / SSIM held (nn) | "
            "edge: particles / discs kernel / discs nn (pitches) | " + " | ".join(SH.HEAD) + " | v end / twin | sil_iou / twin | thin / twin |",
            "|" + "---|" * 13]
    V, ceil = {}, {}
    for s in SEEDS:
        name = lambda arm: f"strp{N // 1000}k_{arm}_{s}"                       # noqa: E731
        recs = {a: R7.run_rec(str(RUNS / name(a))) for a in ARMS}
        frame, x_t = TP.target_frame("assets/isosphere.obj", TGT, N, s)
        x_t = x_t.astype(np.float64)
        a = .708 * TP.D7.median_kth_spacing(x_t, 8, 20000)
        z0 = np.load(RUNS / f"{name('T')}_frames12.npz", allow_pickle=True)
        prm = json.loads(str(z0["appearance"]))
        assert np.abs(np.asarray(z0["tgt"], np.float64) - x_t).max() < 1e-5, "the run's target is not the frame's"
        res = int(round(96 * (N / 40000) ** (1 / 3)))
        extent = float(np.abs(x_t).max()) * 1.25
        lat = TP.Lattice(TP.gpu.tensor(x_t.mean(0).astype(np.float32)), 2.8 * float(np.linalg.norm(x_t - x_t.mean(0), axis=1).max()))
        X2 = np.asarray(TP.draws_in_frame(TGT, frame, N, [s + 2])[0], np.float64)
        pt = lambda X: paint(target_red(prm, X), prm).astype(np.float64)       # noqa: E731
        T = {nn: TP.pictures(x_t, pt(x_t), a, res, extent, lat, TP.D7.HELD, nn) for nn in (False, True)}
        ceil[s] = {("nn" if nn else "kernel"): TP.compare(TP.pictures(X2, pt(X2), a, res, extent, lat, TP.D7.HELD, nn), T[nn])
                   for nn in (False, True)}
        tw = col.get(name("T"))
        for arm in ARMS:
            p = RUNS / f"{name(arm)}_frames12.npz"
            r, t = recs[arm], recs["T"]
            if r is None or not p.exists():
                rows.append(f"| {NAMES.get(arm, arm)} | {s} | - |" + " - |" * 10)
                continue
            z = np.load(p, allow_pickle=True)
            X, alb = np.asarray(z["frames"][-1], np.float64), np.asarray(z["src_albedo"], np.float64)
            m = {("nn" if nn else "kernel"): TP.compare(TP.pictures(X, alb, a, res, extent, lat, TP.D7.HELD, nn), T[nn]) for nn in (False, True)}
            lay = layer(X)
            red = np.linalg.norm(alb[:, :3] - RED, axis=1) < 0.05
            kap = kappa(red[lay], target_red(prm, X[lay]))
            c = col.get(name(arm))
            rr = None
            if c is not None and tw is not None and arm != "T":
                rr = (tw["end"]["mis"] - c["end"]["mis"]) / max(tw["end"]["mis"] - c["floor"], 1e-12)
            e = edge.get(name(arm), {})
            ew = lambda k: R7.fmt((e.get(k) or {}).get("logistic"), ".2f")     # noqa: E731
            sh = SH.block_shares(str(RUNS / name(arm)) + ".json", twin=(arm == "T"))
            ratio = R7.vx(r, t)
            V[f"{arm}_{s}"] = dict(m, colour_r=rr, mis=None if c is None else c["end"]["mis"], kappa=kap, vx=ratio,
                                   sil=(r["sil"], t["sil"]), thin=(r["thin"], t["thin"]), windows=r["windows"], shares=sh,
                                   edge={k: (e.get(k) or {}).get("logistic") for k in ("particles", "discs_k3", "discs_nn")})
            rows.append(f"| {NAMES.get(arm, arm)} | {s} | {R7.fmt(rr, '.2f') if arm != 'T' else '0 (ref.)'} | "
                        f"{R7.fmt(None if c is None else c['end']['mis'])} | {kap:.3f} | {m['kernel']['psnr']:.2f} / {m['kernel']['ssim']:.3f} | "
                        f"{m['nn']['psnr']:.2f} / {m['nn']['ssim']:.3f} | {ew('particles')} / {ew('discs_k3')} / {ew('discs_nn')} | "
                        f"{SH.fmt3(sh, 'u')} | {SH.fmt3(sh, 'dfc')} | {R7.fmt(ratio, '.2f')}x | {r['sil']:.4f} / {t['sil']:.4f} | "
                        f"{r['thin']:.3f} / {t['thin']:.3f} |")
        et = edge.get(f"{name(ARMS[-1])}@tgt", {})
        rows.append(f"| target paint (its own sample) / ceiling (an independent sample painted) | {s} | - | "
                    f"{R7.fmt(None if tw is None else tw['floor'])} (floor) | 1 | {ceil[s]['kernel']['psnr']:.2f} / {ceil[s]['kernel']['ssim']:.3f} | "
                    f"{ceil[s]['nn']['psnr']:.2f} / {ceil[s]['nn']['ssim']:.3f} | "
                    f"{R7.fmt((et.get('particles') or {}).get('logistic'), '.2f')} / {R7.fmt((et.get('discs_k3') or {}).get('logistic'), '.2f')} / "
                    f"{R7.fmt((et.get('discs_nn') or {}).get('logistic'), '.2f')} | - | - | - | - | - |")
    parts = []
    for arm in ARMS[1:]:
        for s in SEEDS:
            k, kt = f"{arm}_{s}", f"T_{s}"
            if k not in V or kt not in V:
                continue
            v, t = V[k], V[kt]
            sg = R7.SHAPE_GATE
            shape_ok = v["sil"][0] >= v["sil"][1] - sg[0] and v["thin"][0] <= v["thin"][1] + sg[1]
            parts.append(f"{arm} s{s}: r {R7.fmt(v['colour_r'], '.2f')}, kappa {v['kappa']:.2f} (twin {t['kappa']:.2f}), PSNR "
                         f"{v['kernel']['psnr'] - t['kernel']['psnr']:+.2f} dB over the twin ({ceil[s]['kernel']['psnr'] - v['kernel']['psnr']:.2f} "
                         f"below the ceiling), particle edge {R7.fmt(v['edge']['particles'], '.2f')} pitches, rest "
                         f"{R7.fmt(v['vx'], '.2f')}x, shape {'held' if shape_ok else 'NOT held'}")
    verdict = "; ".join(parts)
    L = [f"# D173: the stripes scene at {N // 1000}k (three vertical stripes in the front view, the middle red; the source's "
         "top third red)", "", *rows, "",
         "The twin's own layout meets the stripes at chance (kappa near 0; d173_layout_check.py read 0.00-0.03 on the band "
         "scene's twins before any run), so whatever stripes the full arm shows the render laid. Colour r = (twin mis - mis) "
         "/ (twin mis - floor), D160's layout reading (1 = the floor of an independent sample, 0 = the twin). PSNR over the "
         "covered pixels, SSIM over the RGB image, mean of the 46 held-out views at the run's resolution. Edge widths: the "
         "logistic 10-90 % rise across each stripe edge, the mean of the two, pitches. " + SH.NOTE, "", "VERDICT: " + verdict]
    (D / "summary.md").write_text("\n".join(L) + "\n")
    json.dump(dict(arms=V, ceiling=ceil, verdict=verdict), open(D / "verdict.json", "w"), indent=1, default=float)
    print("VERDICT: " + verdict, flush=True)


if __name__ == "__main__":
    main()
