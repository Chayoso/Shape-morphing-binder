"""d107_bands.py DUMP_DIR FRAMES12_NPZ TARGET_OBJ OUT_DIR TITLE — D107: which gradient writes the shape detail, band by
band, from a run's per-window gradient dumps (`pipeline_run.py --grad_dump`, slimmed by tmp/d107_slim.py).

Per window, at the state the gradients are taken at (xT0, the window's end under its starting control), on the outer
layer (the pipeline's layer rule) within two pitches of the target mesh (fitted to the run's target sample as
exterior_offset_probe.py fits it, one million surface samples with their face normals): each field's component along
the particle's outward normal, in display pitches:
  offset  the signed distance to the mesh (> 0 outside; to the plane of the nearest surface sample);
  render  the render push, -lambda (dsil + dpbr) . n;             physics  the physics push, -dphys . n;
  r_dFc, p_dFc, r_u, p_u  what each channel alone moves the layer by in the window (the linear-response rollouts:
          the render or physics gradient on the stress control or on the layer control u, scaled to the window's step);
  opt     what the window's optimisation moved (the committed end state minus xT0);
  all     the whole window's move (the committed end state minus the window's start: dynamics, relaxation, controls).
Split into octave bands by differences of Gaussian means over the layer (sigma 0.5, 1, 2, 4, 8 pitches; only
neighbours whose layer normals agree, so a thin part's two faces are not mixed): nominal wavelengths 2.7, 5.4, 10.8,
21.6 pitches. Per band: the offset's rms; for each field its correlation with -offset ("toward": +1 when it undoes
the band's offset, 0 when blind to it), its rms, and for the moves the share of the band's offset they close
(-sum(f s) / sum(s s)). OUT_DIR/rows.txt (one JSON line per window) and OUT_DIR/bands.png."""
import glob, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import trimesh                                                 # noqa: E402
import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from physmorph.pipeline.window.layer import layer_by_asymmetry, layer_spacing  # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.sampling.mesh import load_mesh                  # noqa: E402
from physmorph.sampling.orientation import orient_name, rotation  # noqa: E402

dev = torch.device("cuda")
files = sorted(glob.glob(os.path.join(sys.argv[1], "win_*.npz")))
z12 = np.load(sys.argv[2], allow_pickle=True)
out, title = Path(sys.argv[4]), sys.argv[5]
out.mkdir(parents=True, exist_ok=True)
tgt = torch.as_tensor(np.asarray(z12["tgt"], np.float32), device=dev)
a = .708 * float(knn_self_torch(tgt, 9)[0][:, 8].median())    # the display pitch

# the target mesh, fitted to the target sample as exterior_offset_probe.py fits it
mesh = load_mesh(sys.argv[3])
mesh.merge_vertices()
o = orient_name(sys.argv[3])
if o != "id":
    mesh.vertices = np.asarray(mesh.vertices, np.float64) @ rotation(o).T
t_np = tgt.cpu().numpy().astype(np.float64)
vb = np.asarray(mesh.bounds, np.float64)
step = float(knn_self_torch(tgt, 7)[0][:, 6].median())
mesh.vertices = (np.asarray(mesh.vertices, np.float64) - vb.mean(0)) * float(np.mean((t_np.max(0) - t_np.min(0) + step) / (vb[1] - vb[0]))) \
    + 0.5 * (t_np.max(0) + t_np.min(0))
points, face = trimesh.sample.sample_surface(mesh, 1000000, seed=0)
surface = torch.as_tensor(np.asarray(points, np.float32), device=dev)
surface_normal = torch.as_tensor(np.asarray(mesh.face_normals[face], np.float32), device=dev)
surface_tree = gpu.KNN(surface)

SIGMAS = (.5, 1., 2., 4., 8.)
BANDS = ("2.7", "5.4", "10.8", "21.6")                         # nominal wavelengths of sigma 0.5-1, 1-2, 2-4, 4-8 pitches
FIELDS = ("render", "physics", "r_dFc", "p_dFc", "r_u", "p_u", "opt", "all")


def gauss_means(P, n, f):
    """(len(SIGMAS), K, M): the Gaussian means of the K fields f (K, M) over the layer points P, same-side neighbours."""
    M = len(P)
    acc = torch.zeros(len(SIGMAS), f.shape[0], M, device=dev, dtype=torch.float64)
    wsum = torch.zeros(len(SIGMAS), M, device=dev, dtype=torch.float64)
    for i0 in range(0, M, 1024):
        d2 = torch.cdist(P[i0:i0 + 1024], P).square()
        side = ((n[i0:i0 + 1024] @ n.T) > 0).double()
        for k, s in enumerate(SIGMAS):
            w = torch.exp(-.5 * d2 / (s * a) ** 2).double() * side
            acc[k, :, i0:i0 + 1024] = (w @ f.double().T).T
            wsum[k, i0:i0 + 1024] = w.sum(1)
    return acc / wsum[:, None]


rows = []
for wi, path in enumerate(files):
    z = np.load(path)
    if "xT_rend" not in z.files or "xT_final" not in z.files:
        continue
    t = lambda k: torch.as_tensor(z[k], device=dev).float()    # noqa: E731
    x = t("xT0")
    lay, nrm = layer_by_asymmetry(x, layer_spacing(x))
    li = torch.nonzero(lay).squeeze(1)                          # only the layer is looked up on the mesh (a tree query of
    _, at = surface_tree.query(x[li].contiguous(), 1)           #   every particle, most far inside, took 100 s a window)
    at = at.reshape(-1)
    s_all = torch.zeros(len(x), device=dev)
    s_all[li] = ((x[li] - surface[at]) * surface_normal[at]).sum(1) / a
    keep = lay & (s_all.abs() < 2.)
    if int(keep.sum()) < 200:
        continue
    P, n = x[keep], nrm[keep]
    along = lambda v: (v[keep] * n).sum(1) / a                 # noqa: E731
    lam = float(z["lam_r"])
    f = torch.stack((s_all[keep],
                     along(-lam * (t("gx_sil") + t("gx_pbr"))), along(-t("gx_phys")),
                     along(t("xT_rend") - t("xT_base")), along(t("xT_phys") - t("xT_base")),
                     along(t("xT_rend_u") - t("xT_base_u0")), along(t("xT_phys_u") - t("xT_base_u0")),
                     along(t("xT_final") - x), along(t("xT_final") - t("x0"))))
    G = gauss_means(P, n, f)
    row = dict(window=wi, layer_near=int(keep.sum()), lam=lam, g_share=float(z["g_share"]))
    for b, name in enumerate(BANDS):
        band = G[b] - G[b + 1]                                  # (K, M)
        s = band[0]
        ss = float((s * s).sum())
        row[f"off_{name}"] = float(s.square().mean().sqrt())
        for k, fld in enumerate(FIELDS, start=1):
            v = band[k]
            vv = float((v * v).sum())
            row[f"{fld}_toward_{name}"] = float((v * -s).sum() / max((vv * ss) ** .5, 1e-30))
            row[f"{fld}_rms_{name}"] = float(v.square().mean().sqrt())
            if fld in ("r_dFc", "p_dFc", "r_u", "p_u", "opt", "all"):
                row[f"{fld}_closes_{name}"] = float(-(v * s).sum() / max(ss, 1e-30))
    rows.append(row)
    print(json.dumps(row), flush=True)

with open(out / "rows.txt", "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")

# small multiples: one column per band; rows: the offset's rms, toward (pushes), toward (moves), the share each move closes
SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"render": "#2a78d6", "physics": "#eb6834", "r_dFc": "#2a78d6", "p_dFc": "#eb6834", "r_u": "#86b6ef", "p_u": "#f2b28a",
       "opt": "#3b8f5a", "all": "#0b0b0b"}
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GRID, "figure.facecolor": SURF, "axes.facecolor": SURF})
w = [r["window"] for r in rows]
fig, ax = plt.subplots(4, len(BANDS), figsize=(4.2 * len(BANDS), 11), sharex=True)
for c, name in enumerate(BANDS):
    ax[0, c].plot(w, [r[f"off_{name}"] for r in rows], color=INK, lw=2)
    ax[0, c].set_title(f"wavelength {name} pitches", color=INK)
    for fld, lbl in (("render", "render push"), ("physics", "physics push")):
        ax[1, c].plot(w, [r[f"{fld}_toward_{name}"] for r in rows], color=COL[fld], lw=2, label=lbl)
    for fld, lbl in (("r_dFc", "render alone (dFc)"), ("p_dFc", "physics alone (dFc)"), ("r_u", "render alone (u)"),
                     ("p_u", "physics alone (u)"), ("opt", "the optimisation's move"), ("all", "the window's whole move")):
        ax[2, c].plot(w, [r[f"{fld}_toward_{name}"] for r in rows], color=COL[fld], lw=2 if fld in ("opt", "all") else 1.5,
                      ls="-" if fld in ("r_dFc", "p_dFc", "opt", "all") else "--", label=lbl)
        ax[3, c].plot(w, [r[f"{fld}_closes_{name}"] for r in rows], color=COL[fld], lw=2 if fld in ("opt", "all") else 1.5,
                      ls="-" if fld in ("r_dFc", "p_dFc", "opt", "all") else "--", label=lbl)
    for rr in (1, 2, 3):
        ax[rr, c].axhline(0, color=GRID, lw=1)
    ax[3, c].set_xlabel("window")
ax[0, 0].set_ylabel("offset from the mesh, rms (pitches)")
ax[1, 0].set_ylabel("toward the mesh (pushes)")
ax[2, 0].set_ylabel("toward the mesh (moves)")
ax[3, 0].set_ylabel("share of the offset closed")
ax[1, 0].legend(frameon=False, fontsize=8)
ax[2, 0].legend(frameon=False, fontsize=7)
fig.suptitle(title, color=INK)
fig.tight_layout()
fig.savefig(out / "bands.png", dpi=130)
print("wrote", out, len(rows), "windows", flush=True)
