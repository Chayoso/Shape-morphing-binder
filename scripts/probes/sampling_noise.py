"""Why the stratified cloud has no shot noise — the proof by measurement (docs/surface_gradient.md §9).

usage: sampling_noise.py --out <dir> [--n 40000] [--npz_rep <archive> --npz_strat <archive> --name bunny ...]

Theory (Gaussian blur K of width sigma, particle spacing p = (V/n)^(1/3), rho = 1/p^3):
  with replacement    : counts per fill voxel ~ Poisson  -> Var[rho_hat] = rho * int K^2
                        rel. std = (p/sigma)^(3/2) / sqrt(8 pi^(3/2))            = 8.2 % at sigma = 1.5 p
  stratified (1/voxel): counts exactly 1, only the sub-voxel jitter u ~ U(-p/2, p/2)^3 moves mass:
                        rho_hat(x) = sum_v K(x - x_v - u_v) ~ sum_v [K(x - x_v) - u_v . grad K]
                        Var = (p^2/12) * (1/p^3) * int |grad K|^2 = 1 / (64 pi^(3/2) p sigma^5)
                        rel. std = (p/sigma)^(5/2) / sqrt(64 pi^(3/2))           = 1.9 % at sigma = 1.5 p
  spectra: P_rep(k) = rho e^{-sigma^2 k^2} (white, then the blur);  P_strat(k) = rho (k^2 p^2 / 12) e^{-sigma^2 k^2}
           -> the ratio k^2 p^2 / 12 -> 0 at long wavelengths: a jitter is a dipole field, it has no monopole.
  level set: delta h = delta rho / |grad rho| at the half-space edge = rel.std * sigma * sqrt(pi)
Measured here on (A) a cube (exact geometry) and (B) the archived targets (--npz_rep / --npz_strat pairs).
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

SIG_SP = 1.5     # the renderer's / loss's blur in spacings
VOX_SP = 0.6     # density grid voxel in spacings (the renderer: 0.63 at 40k)


def spacing_of(x):
    sub = x[np.random.default_rng(0).choice(len(x), min(len(x), 20000), replace=False)]
    return float(np.median(cKDTree(sub).query(sub, k=9, workers=-1)[0][:, -1])) * (min(len(x), 20000) / len(x)) ** (1 / 3)


def density(x, lo, vox, dims, sig_vox):
    """CIC deposit + Gaussian blur, [x,y,z] indexing; unit mass per particle, per-voxel mass."""
    p = (x - lo) / vox - 0.5
    i0 = np.floor(p).astype(np.int64); f = p - i0
    rho = np.zeros(dims, np.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                i = i0 + np.array([dx, dy, dz])
                ok = (i >= 0).all(1) & (i < np.array(dims)).all(1)
                np.add.at(rho, (i[ok, 0], i[ok, 1], i[ok, 2]), w[ok])
    return ndimage.gaussian_filter(rho, sig_vox, mode="constant")


def counts_per_cell(x, lo, pitch):
    """Particles per cell of a lattice of the given pitch (the stratified fill's own lattice)."""
    ijk = np.floor((x - lo) / pitch).astype(np.int64)
    key = ijk[:, 0] * 1000003 + ijk[:, 1] * 1009 + ijk[:, 2]
    _, c = np.unique(key, return_counts=True)
    return c


def radial_spectrum(delta, vox):
    F = np.fft.fftn(delta); P = np.abs(F) ** 2 / delta.size
    kk = [np.fft.fftfreq(n, d=vox) * 2 * np.pi for n in delta.shape]
    K = np.sqrt(kk[0][:, None, None] ** 2 + kk[1][None, :, None] ** 2 + kk[2][None, None, :] ** 2)
    kmax = float(K.max()); bins = np.linspace(0, kmax, 40)
    idx = np.digitize(K.ravel(), bins) - 1
    Pr = np.bincount(idx, weights=P.ravel(), minlength=len(bins))[:len(bins) - 1]
    Nr = np.bincount(idx, minlength=len(bins))[:len(bins) - 1]
    kc = 0.5 * (bins[1:] + bins[:-1])
    ok = Nr > 0
    return kc[ok], Pr[ok] / Nr[ok]


def analyse_pair(name, x_rep, x_st, out, cube=False, L=None):
    sp = 0.5 * (spacing_of(x_rep) + spacing_of(x_st))
    vox = VOX_SP * sp; sig = SIG_SP * sp; sig_vox = sig / vox
    lo = np.minimum(x_rep.min(0), x_st.min(0)) - 4 * sig
    hi = np.maximum(x_rep.max(0), x_st.max(0)) + 4 * sig
    dims = tuple(int(np.ceil((hi[i] - lo[i]) / vox)) for i in range(3))
    res = {}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, (lab, x) in enumerate((("with replacement", x_rep), ("stratified", x_st))):
        rho = density(x, lo, vox, dims, sig_vox)
        rho_b = float(np.median(rho[rho > 0.5 * rho.max()]))          # bulk (interior) density
        # interior = deeper than 3 sigma from the body's boundary (rho > 0.5 bulk eroded by 3 sigma)
        body = rho > 0.5 * rho_b
        inner = ndimage.binary_erosion(body, iterations=int(np.ceil(3 * sig_vox)))
        rel = (rho[inner] - rho[inner].mean()) / rho[inner].mean()
        res[lab] = dict(spacing=sp, rel_std=float(rel.std()), n_inner=int(inner.sum()))
        # per-cell counts on the stratified pitch lattice
        pitch = sp if not cube else L / round(len(x) ** (1 / 3))
        c = counts_per_cell(x, lo, pitch)
        res[lab]["count_var_over_mean"] = float(c.var() / c.mean())
        ax = axes[row, 0]
        ax.hist(c, bins=np.arange(0, 8) - 0.5, color="#2f6f73" if row else "#a4552b", rwidth=0.8)
        ax.set_title(f"{lab}: particles per lattice cell (pitch = spacing)\nvar/mean = {c.var()/c.mean():.2f}  (Poisson = 1, one-per-cell = 0)", fontsize=9)
        ax.set_xlabel("particles in the cell"); ax.set_ylabel("cells")
        # slice through the middle of the body
        k = int(np.argmax(inner.sum((0, 1))))
        sl = rho[:, :, k] / rho_b
        ax = axes[row, 1]
        im = ax.imshow(sl.T, origin="lower", cmap="viridis", vmin=0.0, vmax=1.3)
        ax.set_title(f"{lab}: blurred density / bulk, mid slice (sigma = {SIG_SP} spacings)", fontsize=9); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax = axes[row, 2]
        ax.hist(rel, bins=80, color="#2f6f73" if row else "#a4552b")
        ax.set_title(f"{lab}: interior fluctuation (rho - mean)/mean over {int(inner.sum())} voxels\nstd = {rel.std()*100:.1f} %", fontsize=9)
        ax.set_xlabel("relative fluctuation"); ax.set_xlim(-0.3, 0.3)
        res[lab]["rho"] = rho; res[lab]["inner"] = inner; res[lab]["bulk"] = rho_b
    th_rep = (1 / SIG_SP) ** 1.5 / np.sqrt(8 * np.pi ** 1.5)
    th_st = (1 / SIG_SP) ** 2.5 / np.sqrt(64 * np.pi ** 1.5)
    fig.suptitle(f"{name}: shot noise of the blurred density — with replacement vs stratified  (theory: {th_rep*100:.1f} % vs {th_st*100:.1f} %)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, f"noise_{name}.png"), dpi=130); plt.close(fig)
    res["theory_rep"], res["theory_strat"] = float(th_rep), float(th_st)
    res["vox"], res["sig"] = vox, sig
    return res


def cube_spectrum(res, out, sp):
    """Radial power spectrum of the interior fluctuation of the cube (exact geometry)."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.6))
    curves = {}
    for lab, col in (("with replacement", "#a4552b"), ("stratified", "#2f6f73")):
        rho, inner = res[lab]["rho"], res[lab]["inner"]
        idx = np.argwhere(inner); lo_ = idx.min(0); hi_ = idx.max(0) + 1
        sub = rho[lo_[0]:hi_[0], lo_[1]:hi_[1], lo_[2]:hi_[2]]
        delta = (sub - sub.mean()) / sub.mean()
        k, P = radial_spectrum(delta, res["vox"])
        curves[lab] = (k, P)
        ax.loglog(k * sp, P, "o-", ms=3, color=col, label=f"{lab} (measured)")
    k = curves["with replacement"][0]
    sig, p = res["sig"], sp
    scale = curves["with replacement"][1][1] / np.exp(-sig ** 2 * k[1] ** 2)
    ax.loglog(k * sp, scale * np.exp(-sig ** 2 * k ** 2), "--", color="#a4552b", alpha=0.7, label="theory: rho e^(-sigma^2 k^2)  (white noise x blur)")
    ax.loglog(k * sp, scale * (k ** 2 * p ** 2 / 12) * np.exp(-sig ** 2 * k ** 2), "--", color="#2f6f73", alpha=0.7, label="theory: x k^2 p^2 / 12  (a dipole field: no monopole)")
    ax.axvline(2 * np.pi / 2.0, color="k", lw=0.8, ls=":"); ax.text(2 * np.pi / 2.0, ax.get_ylim()[0] * 1.5, " 2-spacing wavelength", fontsize=8)
    ax.set_xlabel("k x spacing"); ax.set_ylabel("power of (rho - mean)/mean"); ax.set_title("cube: radial power spectrum of the interior density fluctuation", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(out, "noise_spectrum.png"), dpi=130); plt.close(fig)


def main():
    a = sys.argv[1:]
    out = a[a.index("--out") + 1]; os.makedirs(out, exist_ok=True)
    n = int(a[a.index("--n") + 1]) if "--n" in a else 40000
    rng = np.random.default_rng(0)
    # ---- A: the cube, sampled the two ways the pipeline does ----
    L = 8.0
    fine = 110
    g = (np.arange(fine) + 0.5) * (L / fine)
    cen = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    idx = rng.integers(0, len(cen), n)
    x_rep = cen[idx] + rng.uniform(-0.5, 0.5, (n, 3)) * (L / fine)
    m = int(round(n ** (1 / 3)))                     # 34^3 = 39304 ~ n: one per cell
    g2 = (np.arange(m) + 0.5) * (L / m)
    lat = np.stack(np.meshgrid(g2, g2, g2, indexing="ij"), -1).reshape(-1, 3)
    x_st = lat + rng.uniform(-0.5, 0.5, lat.shape) * (L / m)
    res = analyse_pair("cube", x_rep.astype(np.float32), x_st.astype(np.float32), out, cube=True, L=L)
    sp = res["with replacement"]["spacing"]
    cube_spectrum(res, out, sp)
    print(f"cube: rel std with replacement {res['with replacement']['rel_std']*100:.2f} % (theory {res['theory_rep']*100:.2f}), "
          f"stratified {res['stratified']['rel_std']*100:.2f} % (theory {res['theory_strat']*100:.2f}); "
          f"count var/mean {res['with replacement']['count_var_over_mean']:.2f} vs {res['stratified']['count_var_over_mean']:.2f}")
    # ---- B: archived targets ----
    i = 0
    while "--npz_rep" in a[i:]:
        j = a.index("--npz_rep", i)
        name = a[a.index("--name", j) + 1]
        from physmorph.sampling.orientation import orient_archive
        zr = np.load(a[j + 1]); zs = np.load(a[a.index("--npz_strat", j) + 1])
        _, tr, _, _ = orient_archive(zr, a[j + 1]); _, ts, _, _ = orient_archive(zs, a[a.index("--npz_strat", j) + 1])
        r = analyse_pair(name, np.asarray(tr, np.float32), np.asarray(ts, np.float32), out)
        print(f"{name}: rel std with replacement {r['with replacement']['rel_std']*100:.2f} %, stratified {r['stratified']['rel_std']*100:.2f} %; "
              f"count var/mean {r['with replacement']['count_var_over_mean']:.2f} vs {r['stratified']['count_var_over_mean']:.2f}; "
              f"spacing {r['with replacement']['spacing']:.4f}")
        i = j + 1


if __name__ == "__main__":
    main()
