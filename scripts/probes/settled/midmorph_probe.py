"""midmorph_probe.py ARCHIVE_NPZ RUN_JSON TERMS_DIR — D36/D37: why the surface is sparse in the bulk morph, and where the
detached beads come from.

B1, per sampled frame, on the surface particles connected to the body (neighbourhood asymmetry of half a coverage
radius; single linkage at one layer spacing): the display's sparsity S / S_ref (support.surface_spacing against the
target's surface), and what makes it:
  the arrangement in the tangent plane: the covariance of the in-plane offsets of the 8 nearest in-plane neighbours has
    two lengths l1 >= l2 (sqrt of twice its eigenvalues), each against the same on the target's surface. One direction
    stretched: l1 up, l2 as the target's. The area diluted: both up.
  the material: the archived deformation gradient F gives the stretches s1 >= s2 of the material in the tangent plane
    (from (F F^T)^-1 restricted to it) and J = det F.
B2, the rendered particles that are in detached groups two layer spacings or more from the body at window 5's start,
traced back through the windows (term dump with the channel record): detached and on the layer or not, the depth they
had in the source, S / S_ref, det F, the stretch of their eight source bonds; the window in which each first appears
detached; and per window what moves them apart from their source neighbours, by channel (the relative displacement
along each bond: MPM advection, the u control, the rest)."""
import glob, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import torch.nn.functional as nnf                              # noqa: E402
from physmorph.pipeline.window.layer import detached_groups, layer_spacing  # noqa: E402
from physmorph.render.knn_gpu import knn_self_torch            # noqa: E402
from physmorph.render.studio import DensityNormals             # noqa: E402
from physmorph.render.support import live_support, surface_particles, surface_spacing  # noqa: E402

dev = torch.device("cuda")
z = np.load(sys.argv[1], allow_pickle=True)
run = json.load(open(sys.argv[2]))
cfg = run["arms"]["render_full_dt_iso_nn"]["config"]
frames, n_del = z["frames"], int(z["deliver_n"])
f_idx = [int(v) for v in z["F_sample_idx"]]
tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
td = knn_self_torch(tgt, 9)[0]
sp, cov_r = float(td[:, 1].median()), float(td[:, 8].median())
center = tgt.mean(0)
radius = float((tgt - center).norm(dim=1).max())
density_normals = DensityNormals(center, radius, sp, blur=3.0)
X = lambda raw: torch.as_tensor(np.asarray(frames[raw], np.float32), device=dev)  # noqa: E731


def prims(x):
    d, nb = knn_self_torch(x, 33)
    normals, mag = density_normals(x)
    strong = mag >= torch.quantile(mag[::max(1, len(x) // 100000)], .6)
    near = nb[:, 1:33]
    sn = strong[near]
    chosen = near[torch.arange(len(x), device=dev), sn.float().argmax(1)]
    normals = torch.where((~strong & sn.any(1))[:, None], normals[chosen], normals)
    for _ in range(2):
        normals = nnf.normalize(normals[nb].mean(1), dim=1, eps=1e-9)
    return d, nb, normals


def plane_lengths(x, nb, normals, ids):
    """(l1, l2) for the particles ids: the two lengths of the in-plane arrangement of their 8 nearest in-plane neighbours."""
    off = x[nb[ids, 1:]] - x[ids, None, :]
    n = normals[ids]
    p = off - (off * n[:, None, :]).sum(-1, keepdim=True) * n[:, None, :]
    near = p.norm(dim=-1).topk(8, dim=1, largest=False).indices
    q = p.gather(1, near[:, :, None].expand(-1, -1, 3)).double()
    ev = torch.linalg.eigvalsh(q.transpose(1, 2) @ q / 8)          # ascending: ~0 (the normal), then the plane's two
    return (2 * ev[:, 2]).clamp_min(0).sqrt().float(), (2 * ev[:, 1]).clamp_min(0).sqrt().float()


def plane_stretch(F, normals, ids):
    """(s1, s2, J): the material's stretches in the tangent plane from (F F^T)^-1 restricted to it, and det F."""
    Fd = F[ids].double()
    n = normals[ids].double()
    P = torch.eye(3, dtype=torch.float64, device=dev) - n[:, :, None] * n[:, None, :]
    M = P @ torch.linalg.inv(Fd @ Fd.transpose(1, 2)) @ P
    ev = torch.linalg.eigvalsh(M)                                   # ascending: ~0, then 1 / s1^2 <= 1 / s2^2
    return ev[:, 1].clamp_min(1e-12).rsqrt().float(), ev[:, 2].clamp_min(1e-12).rsqrt().float(), torch.linalg.det(Fd).float()


with torch.inference_mode():
    dT, nbT, nT = prims(tgt)
    surfT = torch.nonzero(surface_particles(tgt, nbT, cov_r)).squeeze(1)
    S_ref = float(surface_spacing(tgt, nbT, nT)[surfT].median())
    l1T, l2T = plane_lengths(tgt, nbT, nT, surfT)
    L1, L2 = float(l1T.median()), float(l2T.median())
med = lambda v: float(v.median()) if len(v) else float("nan")  # noqa: E731
print(f"N {len(tgt)}; target spacing {sp:.4f} wu; on the target's surface ({len(surfT)} particles): S_ref {S_ref / sp:.2f} sp, l1 {L1 / sp:.2f} sp, l2 {L2 / sp:.2f} sp, l1 / l2 median {med(l1T / l2T):.2f}")
print("\nB1. surface particles connected to the body, by the display's sparsity S / S_ref")
print("   raw (window) | class | particles | l1, l2 against the target surface's (median) | l1 / l2 | one direction (l1 >= 1.5, l2 < 1.2) %, diluted (l2 >= 1.2) % | "
      "material stretch in the plane s1, s2 (median), s1 s2 | det F | correlation of log S with log sqrt(s1 s2)")
for raw in [r for r in (80, 120, 160, 200, 240, 320, 480, f_idx[-1]) if r in f_idx]:
    with torch.inference_mode():
        x = X(raw)
        F = torch.as_tensor(np.asarray(z["F_samples"][f_idx.index(raw)], np.float32), device=dev)
        d, nb, normals = prims(x)
        group, _ = detached_groups(d, nb, layer_spacing(x))
        surf = surface_particles(x, nb, cov_r) & (group == 0)
        S = surface_spacing(x, nb, normals) / S_ref
        for name, m in (("all", surf), ("S >= 1.1", surf & (S >= 1.1)), ("S >= 1.5", surf & (S >= 1.5))):
            ids = torch.nonzero(m).squeeze(1)
            if len(ids) < 20:
                print(f"   {raw:5d} ({raw / 40:4.1f}) | {name:8s} | {len(ids)}")
                continue
            l1, l2 = plane_lengths(x, nb, normals, ids)
            s1, s2, J = plane_stretch(F, normals, ids)
            a, b = l1 / L1, l2 / L2
            cc = float(torch.corrcoef(torch.stack((S[ids].log(), (s1 * s2).clamp_min(1e-6).log() / 2)))[0, 1])
            print(f"   {raw:5d} ({raw / 40:4.1f}) | {name:8s} | {len(ids):6d} | {med(a):.2f}, {med(b):.2f} | {med(l1 / l2):.2f} | "
                  f"{100 * float(((a >= 1.5) & (b < 1.2)).float().mean()):3.0f}, {100 * float((b >= 1.2).float().mean()):3.0f} | "
                  f"{med(s1):.2f}, {med(s2):.2f}, {med(s1 * s2):.2f} | {med(J):.2f} | {cc:+.2f}")

files = sorted(glob.glob(os.path.join(sys.argv[3], "terms_*.npz")))
K0 = 5
with torch.inference_mode():
    x5 = X(40 * K0)
    d5, nb5, _ = prims(x5)
    spacing5 = layer_spacing(x5)
    g5, size5 = detached_groups(d5, nb5, spacing5)
    cand = torch.nonzero((g5 > 0) & (size5 > 1) & (live_support(d5, cov_r, sp) > 0)).squeeze(1)
    gap = torch.cdist(x5[cand], x5[g5 == 0]).min(1).values
    beads = cand[gap >= 2 * spacing5]
    src = X(0)
    c0 = src.mean(0)
    lattice = float(knn_self_torch(src, 7)[0][:, 6].median())
    depth = ((src - c0).norm(dim=1).max() - (src - c0).norm(dim=1)) / lattice
    bond = gpu.knn(src, 9)[1][:, 1:]
    rest = (src[bond] - src[:, None, :]).norm(dim=-1)
    surf5 = torch.nonzero(surface_particles(x5, nb5, cov_r) & (g5 == 0)).squeeze(1)
print(f"\nB2. the beads: {len(beads)} rendered particles in detached groups at least 2 layer spacings from the body at window {K0}'s start (of {len(cand)} rendered in detached groups)")
print(f"   depth in the source, lattice steps (median, share within 2 / within 4 / deeper than 8): beads {med(depth[beads]):.1f}, {100 * float((depth[beads] <= 2).float().mean()):.0f} / "
      f"{100 * float((depth[beads] <= 4).float().mean()):.0f} / {100 * float((depth[beads] > 8).float().mean()):.0f} %; the body's surface then {med(depth[surf5]):.1f}, "
      f"{100 * float((depth[surf5] <= 2).float().mean()):.0f} / {100 * float((depth[surf5] <= 4).float().mean()):.0f} / {100 * float((depth[surf5] > 8).float().mean()):.0f} %")
print("   window | detached % | on the layer % | S / S_ref (median) | det F at the window's start (median) | source-bond stretch mean, largest (median) | "
      "moved apart from the source neighbours this window, sp a bond: advection, u, rest | u's gate open % | bonds inside the bead set %")
onset = torch.full((len(beads),), -1, dtype=torch.long, device=dev)
inside = torch.zeros(len(src), dtype=torch.bool, device=dev); inside[beads] = True
for k in range(0, K0 + 1):
    f = np.load(files[k])
    with torch.inference_mode():
        x = torch.as_tensor(f["c_x0"], device=dev)
        ch = {c: torch.as_tensor(f["c_" + c], device=dev) for c in ("d_adv", "d_u", "d_rest")}
        lmask, ug, J0 = (torch.as_tensor(f["c_" + c], device=dev) for c in ("lmask", "ug", "J0"))
        d, nb, normals = prims(x)
        group, _ = detached_groups(d, nb, layer_spacing(x))
        S = surface_spacing(x, nb, normals) / S_ref
        det = group[beads] > 0
        onset = torch.where((onset < 0) & det, torch.full_like(onset, k), onset)
        e = x[bond[beads]] - x[beads, None, :]
        L = e.norm(dim=-1)
        eh = e / L[..., None].clamp_min(1e-9)
        apart = [float((((ch[c][bond[beads]] - ch[c][beads, None, :]) * eh).sum(-1)).mean()) / sp for c in ("d_adv", "d_u", "d_rest")]
        st = L / rest[beads]
        print(f"   {k:4d} | {100 * float(det.float().mean()):3.0f} | {100 * float((lmask[beads] > .5).float().mean()):3.0f} | {med(S[beads]):.2f} | {med(J0[beads]):.2f} | "
              f"{med(st.mean(1)):.2f}, {med(st.max(1).values):.2f} | " + " ".join(f"{v:+.2f}" for v in apart)
              + f" | {100 * float((ug[beads] > .5).float().mean()):3.0f} | {100 * float(inside[bond[beads]].float().mean()):3.0f}")
print("   first window in which a bead particle is detached: " + ", ".join(f"window {k}: {100 * float((onset == k).float().mean()):.0f} %" for k in range(0, K0 + 1)))
