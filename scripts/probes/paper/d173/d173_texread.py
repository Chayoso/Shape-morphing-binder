"""d173_texread.py OUT_DIR N SEEDS -- D173 (3): the textured duck's runs read (one card): for each seed, the twin
(output/gpu/d173/runs/tex{N}k_T_{s}) and Rx (tex{N}k_Rx_{s}): the end state's colour pictures (each particle its own
archived albedo; the exterior discs, Zhu-Bridson radius 3, coloured by the kernel as the loss reads them and by the nearest
particle) against the target's pictures (the archive's target sample and its looked-up albedos, drawn the same way) in the
46 held-out views at the run's resolution: PSNR (covered pixels), SSIM (RGB); the ceiling: an independent perfect sample
painted by the texture (d173_texprobe.py's pictures); rest: the end v_mean against the twin's; shape: sil_iou / thin.
Writes OUT_DIR/summary.md (a table and a verdict) and OUT_DIR/verdict.json."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
from physmorph import gpu  # noqa: E402,F401  (first: CuPy before torch)
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d173")
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d170")
import d173_texprobe as TP  # noqa: E402
import d170_read as R7  # noqa: E402

D, N, SEEDS = Path(sys.argv[1]), int(sys.argv[2]), [int(s) for s in sys.argv[3].split(",")]
D.mkdir(parents=True, exist_ok=True)
RUNS = Path("/data/relcfd/chayo/physmorph_v2/output/gpu/d173/runs")
FULL, TWIN = R7.FULL, R7.TWIN


def main():
    rows = ["| arm | seed | PSNR held (kernel) | SSIM held (kernel) | PSNR held (nn) | SSIM held (nn) | v end / twin | sil_iou arm / twin | thin arm / twin |",
            "|" + "---|" * 9]
    V, ceil = {}, {}
    for s in SEEDS:
        recs = {a: R7.run_rec(str(RUNS / f"tex{N // 1000}k_{a}_{s}")) for a in ("T", "Rx")}
        frame, x_t = TP.target_frame("assets/isosphere.obj", TP.TGT, N, s)
        x_t = x_t.astype(np.float64)
        a = .708 * TP.D7.median_kth_spacing(x_t, 8, 20000)
        tex = TP.TextureMap(f"{TP.A}/d173bob_tex.obj", f"{TP.A}/d173bob_diffuse.png", frame, TP.TGT)
        res = int(round(96 * (N / 40000) ** (1 / 3)))
        extent = float(np.abs(x_t).max()) * 1.25
        lat = TP.Lattice(TP.gpu.tensor(x_t.mean(0).astype(np.float32)), 2.8 * float(np.linalg.norm(x_t - x_t.mean(0), axis=1).max()))
        X2 = np.asarray(TP.draws_in_frame(TP.TGT, frame, N, [s + 2])[0], np.float64)
        T = {nn: TP.pictures(x_t, tex.at(x_t), a, res, extent, lat, TP.D7.HELD, nn) for nn in (False, True)}
        ceil[s] = {("nn" if nn else "kernel"): TP.compare(TP.pictures(X2, tex.at(X2), a, res, extent, lat, TP.D7.HELD, nn), T[nn])
                   for nn in (False, True)}
        for arm in ("T", "Rx"):
            p = RUNS / f"tex{N // 1000}k_{arm}_{s}_frames12.npz"
            r, t = recs[arm], recs["T"]
            if r is None or not p.exists():
                rows.append(f"| {arm} | {s} | - | - | - | - | - | - | - |")
                continue
            z = np.load(p, allow_pickle=True)
            X, alb = np.asarray(z["frames"][-1], np.float64), np.asarray(z["src_albedo"], np.float64)
            m = {("nn" if nn else "kernel"): TP.compare(TP.pictures(X, alb, a, res, extent, lat, TP.D7.HELD, nn), T[nn]) for nn in (False, True)}
            ratio = R7.vx(r, t)
            V[f"{arm}_{s}"] = dict(m, vx=ratio, sil=(r["sil"], t["sil"]), thin=(r["thin"], t["thin"]), windows=r["windows"])
            rows.append(f"| {'twin' if arm == 'T' else 'Rx (full)'} | {s} | {m['kernel']['psnr']:.2f} | {m['kernel']['ssim']:.3f} | "
                        f"{m['nn']['psnr']:.2f} | {m['nn']['ssim']:.3f} | {R7.fmt(ratio, '.2f')}x | {r['sil']:.4f} / {t['sil']:.4f} | "
                        f"{r['thin']:.3f} / {t['thin']:.3f} |")
        rows.append(f"| ceiling (a perfect sample painted by the texture) | {s} | {ceil[s]['kernel']['psnr']:.2f} | {ceil[s]['kernel']['ssim']:.3f} | "
                    f"{ceil[s]['nn']['psnr']:.2f} | {ceil[s]['nn']['ssim']:.3f} | - | - | - |")
    gain = [V[f"Rx_{s}"]["kernel"]["psnr"] - V[f"T_{s}"]["kernel"]["psnr"] for s in SEEDS if f"Rx_{s}" in V and f"T_{s}" in V]
    gap = [ceil[s]["kernel"]["psnr"] - V[f"Rx_{s}"]["kernel"]["psnr"] for s in SEEDS if f"Rx_{s}" in V]
    verdict = (f"Rx over the twin: PSNR {', '.join(f'{g:+.2f}' for g in gain)} dB (held-out, the loss's kernel colours); "
               f"below the ceiling by {', '.join(f'{g:.2f}' for g in gap)} dB; rest {', '.join(R7.fmt(V[f'Rx_{s}']['vx'], '.2f') + 'x' for s in SEEDS if f'Rx_{s}' in V)}")
    L = [f"# D173 (3): the textured duck at {N // 1000}k (Keenan Crane's bob_diffuse.png), Rx vs the twin", "", *rows, "",
         f"Arms: Rx = {FULL}, the held weight on both blocks, Rx's settle; twin = {TWIN} (lambda = 0), the same paint. The source "
         "carries the texture projected radially from its centroid (each particle one of the texture's colours, laid out by a "
         "map the physics never reads); the target's pictures are its samples painted by the texture. PSNR over the covered "
         "pixels, SSIM over the RGB image, mean over the 46 held-out views (D157), at the run's resolution; kernel = the "
         "disc colours as the loss reads them (render/exterior.found_disc_albedo, radius 3 pitches), nn = each disc its "
         "nearest particle's colour (the identity).", "", "VERDICT: " + verdict]
    (D / "summary.md").write_text("\n".join(L) + "\n")
    json.dump(dict(arms=V, ceiling=ceil, verdict=verdict), open(D / "verdict.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
