"""d165_read.py OUT_DIR PATTERN_NPZ CEIL_NPZ LABEL:PREFIX [LABEL:PREFIX ...] -- D165's reading of the stamp runs (CPU).

Per run (PREFIX_end.npz of stamp.py): maps over the front word's box (D163's pattern: the word's raster, texel, origin
and face frame; stamp (x, z) = run (x, y)), in bins of a0 (the run's (volume per particle)^(1/3)):
  slab      the imprint: the depth of the slab's top particle in the bin below the slab's first top (y = 0), after the
            lift and the rest;
  plaque    how far the plaque's lowest particle reaches below its face plane (as loaded, after the settle, at the end);
  ceiling   the same for CEIL_NPZ (D163's ceiling: the relief mesh sampled at 300k in the run's frame), the most a
            300k plaque carries.
The letters' map L in [0, 1] (bilinear in the word's raster) at the bins' centres. Read: the word's contrast, the mean
over the letters' bins (L > 0.5) minus the mean over the bins away from them (L < 0.05), in wu and in a0; the
correlation of the map with L; the slab's contrast over the plaque's after the settle (the share printed).
Writes OUT_DIR/read.json, OUT_DIR/maps_<label>.npz, OUT_DIR/imprint_depth.png (the slab's maps on one scale, flipped
left-right so the word reads as seen from above), OUT_DIR/summary.md."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d161")
from d161_pattern import bilinear  # noqa: E402

out, pat_path, ceil_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
items = [s.split(":", 1) for s in sys.argv[4:]]
out.mkdir(parents=True, exist_ok=True)
pat = np.load(pat_path, allow_pickle=True)
c, fwd, up = pat["c_c"], pat["c_fwd"], pat["c_up"]
tex, texel, u0, v0 = pat["tex"], float(pat["texel"]), float(pat["u0"]), float(pat["v0"])
X0, X1 = c[0] + u0, c[0] + u0 + tex.shape[0] * texel          # the word's box (fwd = +x, up = +y of the run's frame)
Z0, Z1 = c[1] + v0, c[1] + v0 + tex.shape[1] * texel
R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def letters(xs, zs):
    p = np.stack([xs, zs, np.full_like(xs, c[2])], 1)
    q = p - c
    return bilinear(tex, (q @ fwd - u0) / texel, (q @ up - v0) / texel)


def column(P, h, mode):
    nx, nz = int(np.ceil((X1 - X0) / h)), int(np.ceil((Z1 - Z0) / h))
    i = np.floor((P[:, 0] - X0) / h).astype(np.int64)
    k = np.floor((P[:, 2] - Z0) / h).astype(np.int64)
    ok = (i >= 0) & (i < nx) & (k >= 0) & (k < nz)
    key = i[ok] * nz + k[ok]
    m = np.full(nx * nz, -np.inf if mode == "top" else np.inf)
    (np.maximum if mode == "top" else np.minimum).at(m, key, P[ok, 1])
    m[~np.isfinite(m)] = np.nan
    return m.reshape(nx, nz)


def contrast(depth, L):
    ok = np.isfinite(depth)
    inn, away = ok & (L > 0.5), ok & (L < 0.05)
    con = float(depth[inn].mean() - depth[away].mean()) if inn.any() and away.any() else float("nan")
    d, l = depth[ok], L[ok]
    corr = float(np.corrcoef(d, l)[0, 1]) if d.std() > 0 and l.std() > 0 else float("nan")
    return con, corr


rows, maps = [], {}
for label, prefix in items:
    z = np.load(prefix + "_end.npz", allow_pickle=True)
    info = json.loads(str(z["info"]))
    a0 = float(info["a0"])
    h = a0
    nx, nz = int(np.ceil((X1 - X0) / h)), int(np.ceil((Z1 - Z0) / h))
    xc = X0 + (np.arange(nx) + 0.5) * h
    zc = Z0 + (np.arange(nz) + 0.5) * h
    XX, ZZ = np.meshgrid(xc, zc, indexing="ij")
    L = letters(XX.ravel(), ZZ.ravel()).reshape(nx, nz)
    slab = -column(np.asarray(z["slab_end"], np.float64), h, "top")          # the imprint's depth below y = 0
    row = {"label": label, "kind": info["kind"], "a0": a0, "stage_end": info.get("stage_end"),
           "end_press": info.get("end_press"), "seconds": info.get("seconds")}
    con, corr = contrast(slab, L)
    row.update(slab_contrast=con, slab_contrast_a0=con / a0, slab_corr=corr,
               slab_depth_away=float(np.nanmean(np.where(L < 0.05, slab, np.nan))))
    for key in ("parents_load", "parents_settle", "parents_end"):
        P = np.asarray(z[key], np.float64)
        bottom = column(P, h, "bottom")
        prot = np.nanmedian(bottom) - bottom                                # how far below the face plane
        con_p, corr_p = contrast(prot, L)
        tag = key.split("_")[1]
        row.update({f"plaque_{tag}_contrast": con_p, f"plaque_{tag}_contrast_a0": con_p / a0, f"plaque_{tag}_corr": corr_p})
        maps[f"{label.split(' ')[0]}_plaque_{tag}"] = prot
    row["printed_share"] = con / row["plaque_settle_contrast"] if row["plaque_settle_contrast"] > 0 else float("nan")
    maps[f"{label.split(' ')[0]}_slab"] = slab
    maps["L"] = L
    rows.append(row)
    tag0 = label.split(" ")[0]
    np.savez(out / f"maps_{tag0}.npz", slab=slab, L=L, h=h, X0=X0, Z0=Z0,
             **{k[len(tag0) + 1:]: v for k, v in maps.items() if k.startswith(tag0 + "_plaque")})
    print(json.dumps(row), flush=True)

C = np.asarray(np.load(ceil_path, allow_pickle=True)["frames"][-1], np.float64) @ R.T
h = rows[0]["a0"] if rows else 0.0526
bottom = column(C, h, "bottom")
prot = np.nanmedian(bottom) - bottom
nx, nz = bottom.shape
XX, ZZ = np.meshgrid(X0 + (np.arange(nx) + 0.5) * h, Z0 + (np.arange(nz) + 0.5) * h, indexing="ij")
con_c, corr_c = contrast(prot, letters(XX.ravel(), ZZ.ravel()).reshape(nx, nz))
ceiling = {"label": "ceiling", "contrast": con_c, "contrast_a0": con_c / h, "corr": corr_c}
print(json.dumps(ceiling), flush=True)
(out / "read.json").write_text(json.dumps({"runs": rows, "ceiling": ceiling}, indent=1))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    slabs = [(r["label"], maps[f"{r['label'].split(' ')[0]}_slab"] / r["a0"]) for r in rows]
    vals = np.concatenate([m[np.isfinite(m)].ravel() for _, m in slabs]) if slabs else np.zeros(1)
    lo_, hi_ = np.quantile(vals, 0.01), np.quantile(vals, 0.99)
    fig, axes = plt.subplots(len(slabs), 1, figsize=(12, 2.4 * max(1, len(slabs))), squeeze=False)
    for ax, (label, m) in zip(axes[:, 0], slabs):
        # rows z (the run's y, the word's up), columns x: the plaque's front view, i.e. the top view mirrored
        im = ax.imshow(m.T, origin="lower", cmap="gray_r", vmin=lo_, vmax=hi_, extent=(X0, X1, Z0, Z1),
                       aspect="equal")
        ax.set_title(f"{label}: how deep the slab was pressed (in particle spacings), seen from above, mirrored to read")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes[:, 0].tolist(), shrink=0.8, label="depth below the slab's top (particle spacings)")
    fig.savefig(out / "imprint_depth.png", dpi=150, bbox_inches="tight")
except Exception as e:  # noqa: BLE001  (a figure; its failure must not cost the reading)
    print(json.dumps({"figure_failed": repr(e)}), flush=True)

lines = ["# D165 stamp test: reading (auto)", "",
         "| run | plaque word contrast loaded / settled / end (a0) | slab imprint contrast (a0) | imprint corr with the "
         "letters | printed share | press reached |", "|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['label']} | {r['plaque_load_contrast_a0']:.3f} / {r['plaque_settle_contrast_a0']:.3f} / "
                 f"{r['plaque_end_contrast_a0']:.3f} | {r['slab_contrast_a0']:.3f} | {r['slab_corr']:.3f} | "
                 f"{r['printed_share']:.2f} | {'yes' if r['end_press'] is not None else 'no'} |")
lines += ["", f"Ceiling (the relief mesh sampled at 300k): word contrast {ceiling['contrast_a0']:.3f} a0, corr "
          f"{ceiling['corr']:.3f}.", "",
          "Contrast: the mean over the letters' bins minus the mean away from them (bins of a0, the run's (volume per "
          "particle)^(1/3)); plaque: reach below its face plane; slab: depth of its top after the lift. Figure: "
          "imprint_depth.png; close-ups: imprint_*.png (d165_render.py)."]
(out / "summary.md").write_text("\n".join(lines) + "\n")
print("wrote", out / "summary.md")
