"""Photoreal morph video: isosurface mesh of the particle density, rendered with Open3D's
Filament path (PBR material, image-based + sun lighting, soft shadows, ground plane).

    render_photoreal.py --npz run.npz --out out.mp4 [--res 900 --stride 3 --views 35,215]
                        [--still <frame> --out still.png]

The density grid is the one render_iso_video.py uses (trilinear splat of the particle
masses + a Gaussian of --blur particle spacings, iso = --iso x the source bulk density), so
the surface is the same object the isosurface videos show; the mesh is extracted with
marching cubes and smoothed (Taubin). Nothing is hidden: every mesh component is rendered
(--largest_only exists for illustration and is OFF by default) and a sidecar
<out>.components.txt records, per video frame, the number of isosurface components and the
number of isolated particles (8-NN distance > 3 x median), the per-frame QA the videos are
judged by (no floating particles, no particle-looking blobs).
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as Fn

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--res", type=int, default=900, help="pixels per view (square)")
ap.add_argument("--stride", type=int, default=3, help="archived frames per video frame")
ap.add_argument("--views", default="35,215", help="azimuths in degrees, side by side")
ap.add_argument("--elev", type=float, default=18.0)
ap.add_argument("--fps", type=int, default=20)
ap.add_argument("--hold", type=int, default=20, help="repeat the last frame this many times")
ap.add_argument("--grid", type=int, default=160)
ap.add_argument("--iso", default="auto",
                help="isosurface level as a fraction of the source bulk density, or 'auto' = the level at which a "
                     "filament two particles across (a 2x2 bundle, the thinnest continuum the particles can carry) "
                     "still renders: 2 s^2 / (pi sigma^2) with s the particle spacing and sigma the blur, capped at "
                     "0.5. At 0.5 a thin neck (teat, whisker) falls below the level and its bulb renders as a "
                     "detached ball although the material is connected at the particle scale (cow at 150k).")
ap.add_argument("--blur", type=float, default=1.5)
ap.add_argument("--smooth", type=int, default=12, help="Taubin smoothing iterations")
ap.add_argument("--fov", type=float, default=30.0, help="vertical field of view (degrees)")
ap.add_argument("--fill", type=float, default=0.78, help="fraction of the frame height the box spans")
ap.add_argument("--color", default="0.86,0.80,0.72", help="base colour (linear RGB)")
ap.add_argument("--rough", type=float, default=0.32)
ap.add_argument("--metal", type=float, default=0.0)
ap.add_argument("--ground", type=int, default=1, help="shadow-catching ground plane")
ap.add_argument("--target_ghost", type=float, default=0.0, help="alpha of a translucent target isosurface (0 = off)")
ap.add_argument("--largest_only", type=int, default=0, help="render only the largest mesh component (illustration only)")
ap.add_argument("--min_cells", type=float, default=1.0,
                help="drop isosurface components whose volume is below this many MPM cells (dx^3; dx = source "
                     "bbox diagonal / cell_diag): material the grid cannot resolve is not a continuum element. "
                     "0 = draw everything. Dropped components are counted in the sidecar.")
ap.add_argument("--cell_diag", type=float, default=26.0)
ap.add_argument("--bridge", type=int, default=1,
                help="draw particles the isosurface does not enclose but which link the body to another drawn "
                     "component as a filament one particle spacing thick (the rendered topology follows the "
                     "particle connectivity, not the threshold); 0 = off. Bridged components are counted in the sidecar.")
ap.add_argument("--surface", default="mc", choices=["mc", "poisson", "imls", "surfel"],
                help="surface from the density: 'mc' = marching cubes at the level (the gallery); 'poisson' = "
                     "screened Poisson reconstruction of the outer particle layer (density below the one-spacing "
                     "half-space value, normals from the density gradient; S2 of docs/experiments.md 2026-09-19); "
                     "'imls' = implicit moving-least-squares field of the same oriented surface particles on the "
                     "render grid, marching cubes at 0 (S3); 'surfel' = plane-pulled surfels triangulated in "
                     "their tangent planes, Laplacian remeshed (S4, the geometric part of 3D Gaussian Triangulation).")
ap.add_argument("--post", default="none", choices=["none", "bilateral"],
                help="mesh post-filter: 'bilateral' = bilateral normal filtering (Zheng 2011; S5)")
ap.add_argument("--pca_k", type=int, default=32, help="S1 neighbourhood size for the weighted PCA")
ap.add_argument("--pca_kr", type=float, default=4.0, help="S1 eigenvalue ratio clamp (Yu & Turk k_r)")
ap.add_argument("--pca_lam", type=float, default=0.9, help="S1 centre-smoothing weight (Yu & Turk lambda)")
ap.add_argument("--poisson_depth", type=int, default=9)
ap.add_argument("--imls_h", type=float, default=2.0, help="S3 kernel width in particle spacings")
ap.add_argument("--bilateral_iters", type=int, default=5)
ap.add_argument("--kernel", default="iso", choices=["iso", "aniso", "pca"],
                help="density kernel: 'iso' = CIC deposit + separable Gaussian blur of --blur spacings (grid-aligned, "
                     "isotropic); 'aniso' = every particle deposits its own Gaussian with covariance sigma0^2 F F^T "
                     "(the Gaussian rides the deformation, docs/method.md eq. 11; sigma0 = --sigma0 x rest spacing, "
                     "F from the archive's F samples). Surface bumps and 'visible particles' are hypothesised to be "
                     "the isotropic kernel failing to cover stretched material; see docs/experiments.md 2026-09-18.")
ap.add_argument("--sigma0", type=float, default=0.7, help="rest Gaussian size in particle spacings (aniso kernel)")
ap.add_argument("--F", default="geom", choices=["geom", "archive"],
                help="deformation gradient for the aniso kernel: 'geom' = the TOTAL deformation of each particle's "
                     "material patch, fitted per frame by least squares over its 12 rest neighbours (the plastic "
                     "flow included — what the patch actually looks like now); 'archive' = the archived physics F "
                     "(the ELASTIC part only after plastic assimilation, near identity).")
ap.add_argument("--knn", type=int, default=12, help="rest neighbours for the geometric F fit")
ap.add_argument("--still", type=int, default=-1, help="render only this archived frame to --out (png)")
ap.add_argument("--save_mesh", default="", help="with --still: also write the mesh (.ply) and a .json of its numbers")
ap.add_argument("--max_frames", type=int, default=0)
ap.add_argument("--label", default="")
a = ap.parse_args()

os.environ.setdefault("EGL_PLATFORM", "surfaceless")
import open3d as o3d  # noqa: E402
from skimage import measure  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
dev = "cuda" if torch.cuda.is_available() else "cpu"
z = np.load(a.npz, allow_pickle=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physmorph.sampling.orientation import orient_archive  # noqa: E402
from physmorph.render.surface_recon import (pca_kernels, surface_particles, layer_threshold, poisson_mesh,  # noqa: E402
                                            imls_grid, surfel_mesh, bilateral_normal_smooth)
frames_np, tgt_np, _src_np, _orient = orient_archive(z, a.npz)   # y-up (physmorph/sampling/orientation.json)
if _orient != "id":
    print(f"[photoreal] orientation {_orient} ({'from archive' if 'orient' in z.files else 'from the table, applied at render time'})", flush=True)
dn = int(z["deliver_n"]) if "deliver_n" in z.files else len(frames_np)
tgt_np = np.asarray(tgt_np, np.float32)
G = a.grid
tgt = torch.as_tensor(tgt_np, device=dev)
x0 = torch.as_tensor(np.asarray(frames_np[0], np.float32), device=dev)
N = x0.shape[0]

# ---- the box: every archived frame + the target, cube centred on the target ------------
lo = torch.minimum(torch.as_tensor(frames_np[:dn].reshape(-1, 3).min(0), device=dev), tgt.min(0).values)
hi = torch.maximum(torch.as_tensor(frames_np[:dn].reshape(-1, 3).max(0), device=dev), tgt.max(0).values)
ctr = 0.5 * (lo + hi)
half = float((hi - lo).max()) * 0.55
vox = 2 * half / G
n_sub = min(N, 20000)
sub = x0[torch.randperm(N, device=dev)[:n_sub]]
d8 = torch.cdist(sub, sub).topk(9, largest=False).values[:, -1]
spacing = float(d8.median()) * (n_sub / N) ** (1.0 / 3.0)
sig_vox = max(0.6, a.blur * spacing / vox)


def density(x):
    p = (x - (ctr - half)) / vox - 0.5
    i0 = torch.floor(p).long()
    f = p - i0.float()
    rho = torch.zeros(G * G * G, device=dev)
    for dz_ in (0, 1):
        for dy_ in (0, 1):
            for dx_ in (0, 1):
                w = ((f[:, 0] if dx_ else 1 - f[:, 0]) * (f[:, 1] if dy_ else 1 - f[:, 1]) * (f[:, 2] if dz_ else 1 - f[:, 2]))
                i = i0 + torch.tensor([dx_, dy_, dz_], device=dev)
                ok = ((i >= 0) & (i < G)).all(1)
                idx = (i[ok, 2] * G + i[ok, 1]) * G + i[ok, 0]
                rho.index_put_((idx,), w[ok], accumulate=True)
    rho = rho.view(1, 1, G, G, G)
    r = int(3 * sig_vox)
    k = torch.exp(-torch.arange(-r, r + 1, device=dev).float() ** 2 / (2 * sig_vox * sig_vox)); k = k / k.sum()
    rho = Fn.conv3d(rho, k.view(1, 1, 1, 1, -1), padding=(0, 0, r))
    rho = Fn.conv3d(rho, k.view(1, 1, 1, -1, 1), padding=(0, r, 0))
    rho = Fn.conv3d(rho, k.view(1, 1, -1, 1, 1), padding=(r, 0, 0))
    return rho[0, 0]                                       # (G,G,G) indexed [z,y,x]


from physmorph.render.covariance import select_archive_F  # noqa: E402
_F_cache = {}


_knn_rest = None


def geometric_F(x):
    """Total deformation gradient of each particle's material patch at the current frame: the
    least-squares map from its rest neighbour offsets (frame 0, k nearest) to the current ones,
    F = (sum d d0^T)(sum d0 d0^T)^-1 — the PhysGaussian kinematics of the patch, plastic flow
    included, which the archived elastic F cannot show."""
    global _knn_rest
    if _knn_rest is None:
        x0n = x0.detach().cpu().numpy().astype(np.float32)
        _, nb = cKDTree(x0n).query(x0n, k=a.knn + 1, workers=-1)
        _knn_rest = torch.as_tensor(nb[:, 1:], device=dev)
    d0 = x0[_knn_rest] - x0[:, None, :]                                   # (N,k,3) rest offsets
    d1 = x[_knn_rest] - x[:, None, :]                                     # (N,k,3) current offsets
    A = torch.einsum("nki,nkj->nij", d1, d0)                              # sum d d0^T
    B = torch.einsum("nki,nkj->nij", d0, d0) + 1e-6 * torch.eye(3, device=dev)[None] * float(spacing ** 2)
    return torch.linalg.solve(B.transpose(1, 2), A.transpose(1, 2)).transpose(1, 2)   # A B^-1


def frame_F(fi, x=None):
    """Deformation gradient at archived frame fi: geometric (default) or the archived elastic F."""
    if a.F == "geom" and x is not None:
        return geometric_F(x)
    F, _kind = select_archive_F(z, int(fi))
    key = id(F)
    if key not in _F_cache:
        _F_cache.clear()
        Fr = np.asarray(F, np.float32)
        if _orient != "id" and "orient" not in z.files:
            from physmorph.sampling.orientation import rotation
            R = rotation(_orient).astype(np.float32)
            Fr = np.einsum("ij,njk->nik", R, Fr)             # x' = R x  ->  F' = R F
        _F_cache[key] = torch.as_tensor(np.ascontiguousarray(Fr), device=dev)
    return _F_cache[key]


def density_aniso(x, F):
    """Sum of per-particle Gaussians N(x_p, sigma0^2 F_p F_p^T) on the voxel grid: the kernel is
    the particle's material patch carried by the deformation (FALSIFIED as a smoother,
    docs/experiments.md 2026-09-18 item 4; kept as an option)."""
    s0 = a.sigma0 * spacing
    F = torch.where(torch.isfinite(F).all(-1).all(-1)[:, None, None], F, torch.eye(3, device=dev)[None])
    # stretch saturation as in the objective's Gaussian forward model (cov_from_F sat): a
    # particle stretched beyond 3x rest keeps a 3x kernel, so a torn or inverted F cannot
    # paint a whole cell
    M = torch.einsum("nij,nkj->nik", F, F)
    Ms = torch.linalg.solve(torch.eye(3, device=dev)[None] + M / 9.0, M)
    M = 0.5 * (Ms + Ms.transpose(1, 2))
    cov = (s0 * s0) * M + (1e-3 * s0 * s0) * torch.eye(3, device=dev)[None]
    return density_from_cov(x, cov)


def density_pca(x):
    """S1, Yu & Turk (2013): kernel centres Laplacian-smoothed (lambda), covariances from the
    weighted PCA of each particle's neighbourhood (ratio clamp k_r), a sum of anisotropic
    Gaussians of the isotropic kernel's volume."""
    x_np = x.detach().cpu().numpy().astype(np.float32)
    cen, cov = pca_kernels(x_np, spacing, k=a.pca_k, kr=a.pca_kr, lam=a.pca_lam, sigma0=a.sigma0)
    return density_from_cov(torch.as_tensor(cen, device=dev), torch.as_tensor(cov, device=dev))


def density_from_cov(x, cov):
    """Sum of per-particle Gaussians N(x_p, cov_p) on the voxel grid, truncated at 3 sigma_max
    with a fixed stencil; unit mass per particle."""
    prec = torch.linalg.inv(cov)                                          # (N,3,3)
    det = torch.linalg.det(cov).clamp_min(1e-30)
    norm = 1.0 / ((2 * math.pi) ** 1.5 * torch.sqrt(det))                # unit mass per particle
    smax = torch.sqrt(cov.diagonal(dim1=1, dim2=2).sum(1))               # sigma_max <= sqrt(trace)
    r = int(min(6, max(1, math.ceil(3.0 * float(smax.max()) / vox))))
    offs = torch.stack(torch.meshgrid(*(torch.arange(-r, r + 1, device=dev),) * 3, indexing="ij"), -1).reshape(-1, 3)  # (K,3) dz,dy,dx? -> use xyz
    offs = offs[:, [2, 1, 0]].float()                                     # (K,3) in x,y,z
    p = (x - (ctr - half)) / vox - 0.5                                    # voxel-centre coordinates
    i0 = torch.round(p).long()
    rho = torch.zeros(G * G * G, device=dev)
    K = offs.shape[0]
    chunk = max(1, int(2e7 // K))
    for s in range(0, x.shape[0], chunk):
        pe = p[s:s + chunk]; ie = i0[s:s + chunk]; Pe = prec[s:s + chunk]; ne = norm[s:s + chunk]
        idx = ie[:, None, :] + offs[None].long()                          # (n,K,3) voxel indices (x,y,z)
        d = (idx.float() - pe[:, None, :]) * vox                          # world offset voxel centre - particle
        q = torch.einsum("nki,nij,nkj->nk", d, Pe, d)                      # Mahalanobis^2
        w = ne[:, None] * torch.exp(-0.5 * q) * (vox ** 3)                # mass fraction per voxel
        ok = ((idx >= 0) & (idx < G)).all(-1)
        lin = (idx[..., 2] * G + idx[..., 1]) * G + idx[..., 0]
        rho.index_put_((lin[ok],), w[ok], accumulate=True)
    return rho.view(G, G, G)


rho0 = density(x0) if a.kernel == "iso" else (density_pca(x0) if a.kernel == "pca" else density_aniso(x0, frame_F(0, x0)))
occ = rho0[rho0 > 0]
rho_bulk = float(occ.median()) if occ.numel() else 1.0
if str(a.iso).lower() == "auto":
    # a 2x2 bundle of particles (spacing s) blurred by a 3D Gaussian sigma has a line density
    # 4/s^2 per unit length -> peak 4 / (s^2 2 pi sigma^2) particles per volume; bulk = 1/s^3
    sig_wu = sig_vox * vox
    iso_frac = min(0.5, 2.0 * spacing ** 2 / (np.pi * sig_wu ** 2))
    print(f"[photoreal] iso auto: spacing {spacing:.4f} wu, blur sigma {sig_wu:.4f} wu -> iso {iso_frac:.3f} x bulk "
          f"(two-particle filament level; single particles peak at {spacing ** 3 / ((2 * np.pi) ** 1.5 * sig_wu ** 3):.3f})",
          flush=True)
else:
    iso_frac = float(a.iso)
iso = iso_frac * rho_bulk
origin = (ctr - half).cpu().numpy() + 0.5 * vox           # world position of voxel centre (0,0,0)
# the outer particle layer for --surface poisson|imls|surfel: density below the half-space value one
# spacing deep for THIS kernel's width (blur sigma for the isotropic kernel, sigma0 for pca/aniso)
kernel_sigma_sp = (sig_vox * vox / spacing) if a.kernel == "iso" else a.sigma0
layer_thr = layer_threshold(rho_bulk, kernel_sigma_sp)
if a.surface != "mc":
    print(f"[photoreal] surface {a.surface}: outer layer = density <= {layer_thr / rho_bulk:.3f} x bulk "
          f"(kernel sigma {kernel_sigma_sp:.2f} spacings)", flush=True)


cell_wu = float(np.linalg.norm(np.asarray(frames_np[0], np.float32).max(0) - np.asarray(frames_np[0], np.float32).min(0))) / a.cell_diag
min_vol = a.min_cells * cell_wu ** 3
_bb = np.asarray(frames_np[0], np.float32).max(0) - np.asarray(frames_np[0], np.float32).min(0)
ppc = a.min_cells * N * cell_wu ** 3 / (float(np.prod(_bb)) * 0.5236)   # particles per cell (docs/method.md 10.9)
print(f"[photoreal] deliverable rule: a component is drawn iff it holds >= {ppc:.0f} particles ({a.min_cells:g} cell of "
      f"material at ppc {ppc / a.min_cells:.0f}) and encloses >= {min_vol:.4f} wu^3; interior cavities removed; "
      f"sub-filament necks bridged", flush=True)


def _segment_mesh(p, q, r):
    """A cylinder of radius r from p to q (world), as an Open3D mesh."""
    d = q - p; L = float(np.linalg.norm(d))
    if L < 1e-9:
        return None
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=r, height=L, resolution=8, split=1)
    z = np.array([0.0, 0.0, 1.0]); u = d / L
    v = np.cross(z, u); s = float(np.linalg.norm(v)); c_ = float(np.dot(z, u))
    if s < 1e-9:
        R = np.eye(3) if c_ > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c_) / (s * s))
    cyl.rotate(R, center=(0, 0, 0))
    cyl.translate((p + q) / 2.0)
    return cyl


def filament_bridges(x_np, rho, drawn_labels_needed=2):
    """Particles the isosurface does not enclose but which link the body to another enclosed
    component are drawn as a filament one particle spacing thick (docs/method.md 10.10): a
    feature thinner than a two-particle bundle (the cow's teat: a 72-particle bulb on the
    target tied to the udder by a single-particle thread) is connected material at the
    particle scale, and the rendered topology must follow the particles, not the threshold.
    Returns (filament mesh or None, number of enclosed components bridged to the body)."""
    from scipy import ndimage
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    mask = rho >= iso
    lab, n_lab = ndimage.label(mask)
    if n_lab < 2:
        return None, 0
    counts = np.bincount(lab.ravel(), minlength=n_lab + 1)
    drawn = np.zeros(n_lab + 1, bool)
    drawn[1:] = counts[1:] * vox ** 3 >= min_vol                # the volume rule, on voxels
    if drawn.sum() < drawn_labels_needed:
        return None, 0
    body = int(np.argmax(counts[1:]) + 1)
    p = (x_np - (ctr - half).cpu().numpy()) / vox
    ijk = np.clip(np.rint(p).astype(np.int64), 0, G - 1)
    plab = lab[ijk[:, 2], ijk[:, 1], ijk[:, 0]]                  # voxel component of each particle
    free = np.where(plab == 0)[0]
    anch = np.where(drawn[plab])[0]
    if len(free) == 0 or len(anch) == 0:
        return None, 0
    r = 2.5 * spacing
    kf = cKDTree(x_np[free]); ka = cKDTree(x_np[anch])
    ff = np.array(list(kf.query_pairs(r)), dtype=np.int64).reshape(-1, 2)
    fa = kf.query_ball_tree(ka, r)
    # graph nodes: free particles (0..nf-1) then one super-node per drawn label
    nf = len(free); sup = {l: nf + i for i, l in enumerate(np.where(drawn)[0])}
    rows_, cols_ = [ff[:, 0], ff[:, 1]], [ff[:, 1], ff[:, 0]]
    fa_r, fa_c = [], []
    for i, nb in enumerate(fa):
        for j in nb:
            fa_r.append(i); fa_c.append(sup[int(plab[anch[j]])])
    rows_.append(np.array(fa_r, np.int64)); cols_.append(np.array(fa_c, np.int64))
    nn_ = nf + len(sup)
    rr = np.concatenate(rows_); cc = np.concatenate(cols_)
    if len(rr) == 0:
        return None, 0
    gph = coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(nn_, nn_))
    _, comp = connected_components(gph, directed=False)
    root_body = comp[sup[body]]
    others = [l for l in sup if l != body and comp[sup[l]] == root_body]
    if not others:
        return None, 0
    bridge = np.where(comp[:nf] == root_body)[0]                 # free particles on a path to the body
    if len(bridge) == 0:
        return None, len(others)
    bset = set(bridge.tolist())
    rad = 0.55 * spacing
    fil = o3d.geometry.TriangleMesh()
    for i, j in ff:
        if i in bset or j in bset:
            seg = _segment_mesh(x_np[free[i]], x_np[free[j]], rad)
            if seg is not None:
                fil += seg
    for i in bridge:
        for j in fa[i]:
            seg = _segment_mesh(x_np[free[i]], x_np[anch[j]], rad)
            if seg is not None:
                fil += seg
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=rad, resolution=6)
        sph.translate(x_np[free[i]]); fil += sph
    fil.compute_vertex_normals()
    return fil, len(others)


def mesh_of(x, fi=None):
    """Isosurface mesh (Open3D) of the cloud x; returns (mesh, n_components_raw, n_dropped, n_bridged, n_cavities)."""
    if a.kernel == "aniso" and fi is not None:
        rho_t = density_aniso(x, frame_F(fi, x))           # [z,y,x]
    elif a.kernel == "pca":
        rho_t = density_pca(x)
    else:
        rho_t = density(x)
    rho = rho_t.cpu().numpy()
    if not (float(np.nanmax(rho)) > iso):
        print(f"[photoreal] frame {fi}: nothing above the level (rho max {float(np.nanmax(rho)):.3g}, iso {iso:.3g}, "
              f"nan voxels {int(np.isnan(rho).sum())})", flush=True)
        return None, 0, 0, 0, 0
    if a.surface == "mc":
        v, f, _, _ = measure.marching_cubes(rho, level=iso, spacing=(vox, vox, vox))
        v = v[:, ::-1] + origin                            # (z,y,x) -> (x,y,z) world
    else:
        # S2 / S3: the oriented SURFACE particles (within 1.5 voxels of the level set, normals from
        # the density gradient) define the surface; the density keeps its role for the level, the
        # component mass rule and the bridges
        pts, nrm = surface_particles(x, rho_t, ctr, half, vox, layer_thr)
        if fi is None or fi == 0:
            print(f"[photoreal] surface {a.surface}: {len(pts)} outer-layer particles of {len(x)}", flush=True)
        if a.surface in ("poisson", "surfel"):
            pm = poisson_mesh(pts, nrm, depth=a.poisson_depth) if a.surface == "poisson" else surfel_mesh(pts, nrm)
            v = np.asarray(pm.vertices, np.float32)
            f = np.asarray(pm.triangles)[:, ::-1].astype(np.int64)   # the code below re-reverses
            if len(v) == 0 or len(f) == 0:
                print(f"[photoreal] frame {fi}: {a.surface} produced no triangles", flush=True)
                return None, 0, 0, 0, 0
        else:
            fld = imls_grid(pts, nrm, origin, vox, G, h=a.imls_h * spacing)
            v, f, _, _ = measure.marching_cubes(-fld, level=0.0, spacing=(vox, vox, vox))   # inside positive, as rho
            v = v[:, ::-1] + origin
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v.astype(np.float64)),
                                  o3d.utility.Vector3iVector(f[:, ::-1].astype(np.int32)))
    comp = np.asarray(m.cluster_connected_triangles()[0])
    n_comp = int(comp.max()) + 1 if len(comp) else 0
    n_drop = n_cav = 0
    if n_comp > 1 and (a.largest_only or a.min_cells > 0):
        keep = np.ones(len(comp), bool)
        if a.largest_only:
            keep = comp == int(np.bincount(comp).argmax())
        else:
            # component volumes from the signed tetra sum (marching-cubes surfaces are closed).
            # The SIGN separates outer pieces from interior cavities: a closed surface around a
            # void inside the material has its normals facing the void, i.e. the sign opposite
            # to the body's (bunny at 150k: a 1.3–1.8-cell hollow inside the ear counted as a
            # "second drawn piece" in 37 frames while nothing floats). Cavities are removed
            # from the mesh (they are invisible inside the body anyway) and counted apart.
            vv = v.astype(np.float64); ff = f[:, ::-1]
            tet = np.einsum("ij,ij->i", vv[ff[:, 0]], np.cross(vv[ff[:, 1]], vv[ff[:, 2]])) / 6.0
            svol = np.bincount(comp, weights=tet, minlength=n_comp)
            body_sign = np.sign(svol[int(np.argmax(np.abs(svol)))])
            cavity = (np.sign(svol) == -body_sign) & (svol != 0)
            # "material the grid does not resolve" is measured in MASS, not in isosurface volume:
            # the blurred surface of a compressed 30–80-particle chunk can enclose more than
            # dx^3 at the filament level and still be well under one cell of particles (150k C:
            # balls drawn in 53 frames while the grid probe found >= 1 cell in 6). A component
            # is a continuum element iff at least ppc = N dx^3 / V particles sit inside it.
            from scipy import ndimage
            vlab, _ = ndimage.label(rho >= iso)
            xp = x.detach().cpu().numpy()
            pv = (xp - (ctr - half).cpu().numpy()) / vox
            pijk = np.clip(np.rint(pv).astype(np.int64), 0, G - 1)
            plab_ = vlab[pijk[:, 2], pijk[:, 1], pijk[:, 0]]
            vcount = np.bincount(plab_, minlength=int(vlab.max()) + 1)
            # one representative vertex per mesh component -> its voxel label
            first_tri = np.full(n_comp, -1, np.int64)
            first_tri[comp[::-1]] = np.arange(len(comp))[::-1]
            rep = vv[ff[first_tri, 0]]
            rv = np.clip(np.rint((rep - (ctr - half).cpu().numpy()) / vox).astype(np.int64), 0, G - 1)
            # a surface vertex sits on the level: probe one voxel inward along the component's normal-free
            # guess (its centroid direction) — take the max label over the vertex voxel and its 26 neighbours
            mass = np.zeros(n_comp)
            for ci in range(n_comp):
                zz, yy, xx = rv[ci, 2], rv[ci, 1], rv[ci, 0]
                nb = vlab[max(zz - 1, 0):zz + 2, max(yy - 1, 0):yy + 2, max(xx - 1, 0):xx + 2]
                labs = np.unique(nb[nb > 0])
                mass[ci] = vcount[labs].max() if len(labs) else 0.0
            small = ((mass < ppc) | (np.abs(svol) < min_vol)) & ~cavity
            n_cav = int(cavity.sum())
            keep = ~(small | cavity)[comp]
        n_drop = n_comp - n_cav - int(len(np.unique(comp[keep]))) if keep.any() else n_comp - n_cav
        if not keep.all():
            m.remove_triangles_by_mask(~keep)
            m.remove_unreferenced_vertices()
    if a.post == "bilateral":
        m = bilateral_normal_smooth(m, iters=a.bilateral_iters)
    if a.smooth > 0:
        m = m.filter_smooth_taubin(number_of_iterations=a.smooth)
    m.compute_vertex_normals()
    n_bridge = 0
    if a.bridge and n_comp - n_drop - n_cav > 1:
        fil, n_bridge = filament_bridges(x.detach().cpu().numpy().astype(np.float64), rho)
        if fil is not None:
            m += fil
    return m, n_comp, n_drop, n_bridge, n_cav


def bumpiness(m):
    """Mean absolute dihedral angle (degrees) over the mesh's interior edges: 0 for a plane, small
    for a smooth closed surface, large for a bumpy one. The H1 measure (docs/experiments.md
    2026-09-18 item 4)."""
    if m is None or len(m.triangles) == 0:
        return float("nan")
    m.compute_triangle_normals()
    f = np.asarray(m.triangles); nrm = np.asarray(m.triangle_normals)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0)
    e = np.sort(e, axis=1)
    key = e[:, 0].astype(np.int64) * (f.max() + 1) + e[:, 1]
    tri = np.tile(np.arange(len(f)), 3)
    order = np.argsort(key, kind="stable"); key = key[order]; tri = tri[order]
    same = key[1:] == key[:-1]
    a_, b_ = tri[:-1][same], tri[1:][same]
    cosd = np.clip((nrm[a_] * nrm[b_]).sum(1), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosd)).mean()) if len(cosd) else float("nan")


def isolated_count(x_np):
    d = cKDTree(x_np).query(x_np, k=9, workers=-1)[0][:, -1]
    return int((d > 3.0 * np.median(d)).sum())


# ---- scene ------------------------------------------------------------------------------
W = a.res
views = [float(s) for s in a.views.split(",")]
rend = o3d.visualization.rendering.OffscreenRenderer(W, W)
scene = rend.scene
scene.set_background([0.94, 0.94, 0.935, 1.0])
scene.set_lighting(o3d.visualization.rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, (0.35, -0.85, -0.4))
scene.scene.enable_indirect_light(True)
scene.scene.set_indirect_light_intensity(38000.0)
scene.scene.enable_sun_light(True)
scene.scene.set_sun_light((0.35, -0.85, -0.4), (1.0, 0.98, 0.94), 85000.0)
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultLit"
mat.base_color = [float(c) for c in a.color.split(",")] + [1.0]
mat.base_roughness = a.rough
mat.base_metallic = a.metal
mat.base_reflectance = 0.5
ghost = o3d.visualization.rendering.MaterialRecord()
ghost.shader = "defaultLitTransparency"
ghost.base_color = [0.45, 0.55, 0.75, a.target_ghost]
ghost.base_roughness = 0.6
gmat = o3d.visualization.rendering.MaterialRecord()
gmat.shader = "defaultLit"
gmat.base_color = [0.97, 0.97, 0.965, 1.0]
gmat.base_roughness = 0.9
floor_y = float(min(tgt_np[:, 1].min(), frames_np[:dn:max(1, dn // 40)][..., 1].min())) - 0.02 * half
if a.ground:
    ground = o3d.geometry.TriangleMesh.create_box(40 * half, 0.02 * half, 40 * half)
    ground.translate([-20 * half + float(ctr[0]), floor_y - 0.02 * half, -20 * half + float(ctr[2])])
    ground.compute_vertex_normals()
    scene.add_geometry("ground", ground, gmat)
if a.target_ghost > 0:
    tm, _, _, _, _ = mesh_of(tgt)
    if tm is not None:
        scene.add_geometry("target", tm, ghost)
c = ctr.cpu().numpy()
# the camera distance that makes the bounding cube span `fill` of the frame height
dist = half / (a.fill * math.tan(math.radians(a.fov) / 2.0))


def render_views(m):
    imgs = []
    if m is not None:
        scene.add_geometry("body", m, mat)
    for az in views:
        el = math.radians(a.elev); az_r = math.radians(az)
        eye = c + dist * np.array([math.cos(el) * math.sin(az_r), math.sin(el), math.cos(el) * math.cos(az_r)])
        rend.setup_camera(a.fov, c.tolist(), eye.tolist(), [0.0, 1.0, 0.0])
        imgs.append(np.asarray(rend.render_to_image()))
    if m is not None:
        scene.remove_geometry("body")
    return np.concatenate(imgs, axis=1)


def label(img, text):
    if not text:
        return img
    try:
        from PIL import Image, ImageDraw, ImageFont
        im = Image.fromarray(img)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", max(14, img.shape[0] // 40))
        except Exception:
            font = ImageFont.load_default()
        ImageDraw.Draw(im).text((14, 10), text, fill=(50, 50, 50), font=font)
        return np.asarray(im)
    except Exception:
        return img


if a.still >= 0 or a.still == -2:
    fr = tgt if a.still == -2 else torch.as_tensor(np.asarray(frames_np[a.still], np.float32), device=dev)
    # --still -2 renders the TARGET cloud through the same pipeline: the floor of the bumpiness
    # measure for this discretisation
    m, n_comp, n_drop, n_bridge, n_cav = mesh_of(fr, a.still if a.still >= 0 else None)
    bump = bumpiness(m)
    print(f"[photoreal] still {a.still}: kernel {a.kernel} surface {a.surface} post {a.post}, "
          f"bumpiness (mean |dihedral|) {bump:.2f} deg, triangles {len(m.triangles) if m is not None else 0}, "
          f"components {n_comp} dropped {n_drop} cavities {n_cav} bridged {n_bridge}", flush=True)
    img = label(render_views(m), f"{a.label} frame {a.still}  {a.kernel}/{a.surface}/{a.post}  bump {bump:.1f} deg  "
                                 f"components {n_comp} (dropped {n_drop}, cavities {n_cav}, bridged {n_bridge})")
    o3d.io.write_image(a.out, o3d.geometry.Image(np.ascontiguousarray(img)))
    if a.save_mesh and m is not None:
        # the mesh + the discretisation it was made at, for scripts/probes/surface_gt.py
        import json
        o3d.io.write_triangle_mesh(a.save_mesh, m, write_ascii=False, compressed=True)
        with open(os.path.splitext(a.save_mesh)[0] + ".json", "w") as fh:
            json.dump({"npz": a.npz, "frame": a.still, "kernel": a.kernel, "surface": a.surface, "post": a.post,
                       "vox": vox, "spacing": spacing, "iso_frac": iso_frac, "layer_thr_frac": layer_thr / rho_bulk,
                       "bump": bump, "triangles": int(len(m.triangles)), "components": n_comp, "dropped": n_drop,
                       "cavities": n_cav, "bridged": n_bridge}, fh, indent=1)
        print(f"saved {a.save_mesh}")
    print(f"saved {a.out} (components {n_comp}, sub-cell dropped {n_drop})")
    sys.exit(0)

idx = list(range(0, dn, a.stride))
if idx[-1] != dn - 1:
    idx.append(dn - 1)
if a.max_frames > 0:
    idx = idx[: a.max_frames]
tmp = tempfile.mkdtemp(prefix="photoreal_")
qa = []
for k, i in enumerate(idx):
    x_np = np.asarray(frames_np[i], np.float32)
    m, n_comp, n_drop, n_bridge, n_cav = mesh_of(torch.as_tensor(x_np, device=dev), i)
    n_iso = isolated_count(x_np)
    qa.append((i, n_comp, n_iso, n_drop, n_bridge, n_cav))
    img = label(render_views(m), f"{a.label}  frame {i}/{dn - 1}")
    o3d.io.write_image(os.path.join(tmp, f"f{k:05d}.png"), o3d.geometry.Image(np.ascontiguousarray(img)))
    if k % 25 == 0:
        print(f"[photoreal] frame {k + 1}/{len(idx)} (archived {i}) components {n_comp} dropped {n_drop} isolated {n_iso}", flush=True)
n = len(idx)
for h in range(a.hold):
    os.link(os.path.join(tmp, f"f{n - 1:05d}.png"), os.path.join(tmp, f"f{n + h:05d}.png"))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(tmp, "f%05d.png"),
                "-movflags", "faststart", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", a.out], check=True)
with open(a.out + ".components.txt", "w") as fh:
    fh.write("archived_frame isosurface_components isolated_particles subcell_components_dropped components_bridged_to_body interior_cavities\n")
    for i, n_comp, n_iso, n_drop, n_bridge, n_cav in qa:
        fh.write(f"{i} {n_comp} {n_iso} {n_drop} {n_bridge} {n_cav}\n")
    comps = np.array([q[1] for q in qa]); isos = np.array([q[2] for q in qa]); drops = np.array([q[3] for q in qa])
    bridges = np.array([q[4] for q in qa]); cavs = np.array([q[5] for q in qa])
    drawn = comps - drops - cavs
    fh.write(f"# interior cavities (closed surfaces with the sign opposite to the body, removed, not pieces): "
             f"{(cavs > 0).sum()} frames (max {cavs.max()})\n")
    fh.write(f"# filament bridges (particle connectivity): {(bridges > 0).sum()} frames with a drawn component tied to the "
             f"body by particles the isosurface does not enclose (max {bridges.max()}); drawn components>1 AND not bridged "
             f"in {((drawn > 1) & (bridges < drawn - 1)).sum()} frames\n")
    fh.write(f"# iso {iso_frac:.3f} x bulk ({'auto: two-particle filament level' if str(a.iso).lower() == 'auto' else 'fixed'}), "
             f"blur {a.blur} spacings, grid {a.grid}\n")
    fh.write(f"# frames {len(qa)}  raw components>1 in {(comps > 1).sum()} frames (max {comps.max()})  "
             f"drawn components>1 in {(drawn > 1).sum()} frames (max {drawn.max()})  "
             f"sub-cell components dropped in {(drops > 0).sum()} frames (cell {cell_wu:.3f} wu, min {a.min_cells:g} cells)  "
             f"isolated particles max {isos.max()} (frame {idx[int(isos.argmax())]})\n")
for fpath in os.listdir(tmp):
    os.remove(os.path.join(tmp, fpath))
os.rmdir(tmp)
print(f"saved {a.out} ({n} frames + {a.hold} hold; raw components>1 in {(comps > 1).sum()}/{len(qa)} frames, "
      f"drawn components>1 in {(drawn > 1).sum()}, max isolated particles {isos.max()})")
