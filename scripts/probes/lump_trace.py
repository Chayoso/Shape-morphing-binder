"""The scale of the mid-morph lumps (2026-09-22): per sampled frame the plane-residual RMS of the outer
layer at THREE neighbourhood scales — the layer's own (h = 2 spacings, K = 24: the roughness the relaxation
holds), one MPM cell (h = dx / spacing, K scaled by h^2) and two cells — plus the transport speed of that
frame (mean |x_{i+1} - x_i| over the particles, in spacings per frame). A morph whose surface is rough at
2 spacings but smooth at the cell has particle-scale texture; one that is smooth at 2 spacings but lumpy at
the cell has transport lumps; roughness that follows the speed is a transport artefact, not a field's.

usage: lump_trace.py <npz> [stride] [--png out.png] [--cell_diag 26] [--fmax F]
"""
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", ".."))
from physmorph.render.surface_recon import layer_by_asymmetry, plane_residual  # noqa: E402


def opt(name, default, cast=float):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def main():
    args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and not sys.argv[i - 1].startswith("--")]
    path = args[0]
    stride = int(args[1]) if len(args) > 1 else 20
    png = opt("--png", None, str)
    cell_diag = opt("--cell_diag", 26.0)
    z = np.load(path)
    fr = z["frames"]; dn = int(z["deliver_n"]) if "deliver_n" in z.files else len(fr)
    fmax = min(dn, opt("--fmax", dn, int))
    x0 = np.asarray(fr[0], np.float32)
    sub = x0[np.random.default_rng(0).choice(len(x0), min(len(x0), 20000), replace=False)]
    sp = float(np.median(cKDTree(sub).query(sub, k=9, workers=-1)[0][:, -1])) * (min(len(x0), 20000) / len(x0)) ** (1 / 3)
    dx = float(np.linalg.norm(x0.max(0) - x0.min(0))) / cell_diag
    h_cell = dx / sp
    scales = [("2sp", 2.0, 24), ("cell", h_cell, int(round(24 * (h_cell / 2.0) ** 2))), ("2cell", 2 * h_cell, int(round(24 * (h_cell) ** 2)))]
    idx = list(range(0, fmax, stride))
    if idx[-1] != fmax - 1:
        idx.append(fmax - 1)
    rows = []
    for i in idx:
        x = np.asarray(fr[i], np.float32)
        m, n = layer_by_asymmetry(x, sp)
        r = [float(np.sqrt((plane_residual(x, m, n, sp, k=k, h_sp=h)[0] ** 2).mean())) for _, h, k in scales]
        v = float(np.linalg.norm(np.asarray(fr[min(i + 1, dn - 1)], np.float32) - x, axis=1).mean() / sp) if i + 1 < dn else 0.0
        vl = float(np.linalg.norm(np.asarray(fr[min(i + 1, dn - 1)], np.float32)[m] - x[m], axis=1).mean() / sp) if i + 1 < dn else 0.0
        rows.append((i, *r, float(m.mean()), v, vl))
        print(f"frame {i:5d}: rms " + "  ".join(f"{nm} {vv:.3f}" for (nm, _, _), vv in zip(scales, r))
              + f" sp  (layer {rows[-1][4] * 100:.1f} %, speed {v:.3f} sp/frame, layer {vl:.3f})")
    a = np.array(rows)
    print(f"spacing {sp:.4f} wu, cell {dx:.4f} wu = {h_cell:.2f} sp (K {scales[1][2]}, 2 cells K {scales[2][2]}); mean over the morph: "
          + ", ".join(f"{nm} {a[:, j + 1].mean():.3f}" for j, (nm, _, _) in enumerate(scales)) + " sp; end: "
          + ", ".join(f"{nm} {a[-1, j + 1]:.3f}" for j, (nm, _, _) in enumerate(scales))
          + f"; peak cell {a[:, 2].max():.3f} at frame {int(a[a[:, 2].argmax(), 0])}, peak 2cell {a[:, 3].max():.3f} at frame {int(a[a[:, 3].argmax(), 0])}"
          + f"; corr(rms 2sp, speed) {np.corrcoef(a[:-1, 1], a[:-1, 5])[0, 1]:.2f}, peak speed {a[:, 5].max():.3f} sp/frame at frame {int(a[a[:, 5].argmax(), 0])}")
    if png:
        np.savetxt(png.replace(".png", ".txt"), a, fmt="%.4f", header="frame rms_2sp rms_cell rms_2cell layer_frac speed_sp_per_frame layer_speed")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 3.4))
        ax.plot(a[:, 0], a[:, 1], label="plane residual RMS at 2 spacings (K 24)", color="#1F6F78")
        ax.plot(a[:, 0], a[:, 2], label=f"at one MPM cell = {h_cell:.1f} spacings (K {scales[1][2]})", color="#B5442E")
        ax.plot(a[:, 0], a[:, 3], label=f"at two cells = {2 * h_cell:.1f} spacings (K {scales[2][2]})", color="#8A6D1E")
        ax2 = ax.twinx(); ax2.plot(a[:, 0], a[:, 5], label="transport speed (sp / frame)", color="#5B6472", lw=0.9, ls="--")
        ax2.set_ylabel("sp / frame"); ax2.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("archived frame"); ax.set_ylabel("spacings"); ax.grid(alpha=.3); ax.legend(loc="upper left", fontsize=8)
        ax.set_title("outer-layer roughness during the morph: " + path.split("/")[-1].replace("_render_full_dt_iso_nn.npz", ""))
        fig.tight_layout(); fig.savefig(png, dpi=110)
        print("saved", png)


if __name__ == "__main__":
    main()
