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
import copy
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
ap.add_argument("--track", action="store_true",
                help="surface tracking (2026-09-22, docs/method.md 10.15): the drawn mesh is advected with the particles its "
                     "vertices are bound to and pulled toward the fresh reconstruction at --track_alpha per frame; re-meshed "
                     "only when the drawn topology (pieces, bridges, cavities) changes or the drift exceeds --track_tol "
                     "spacings. Removes the frame-to-frame re-fit jitter of an independent reconstruction per frame.")
ap.add_argument("--track_alpha", type=float, default=0.3, help="per-frame pull of the tracked vertices toward the fresh surface")
ap.add_argument("--track_tol", type=float, default=1.0, help="re-mesh when the mean drift exceeds this many spacings")
ap.add_argument("--track_k", type=int, default=8, help="particles a vertex is bound to (Gaussian weights of one spacing)")
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
ap.add_argument("--poisson_depth", type=int, default=0, help="0 = octree depth from --poisson_cell")
ap.add_argument("--poisson_cell", type=float, default=1.0,
                help="finest Poisson octree cell in particle spacings (depth = ceil(log2(extent / cell))). The "
                     "quadratic B-spline spans three cells, so a feature thinner than 3 cells has its two sides "
                     "cancel (cow legs at cell 1.0); the thinnest drawn continuum is two particles across, hence "
                     "cell = 2/3 spacing keeps it (docs/experiments.md 2026-09-19 run 5)")
ap.add_argument("--poisson_trim", type=float, default=0.0,
                help="remove Poisson vertices farther than this many blur sigmas from every layer surfel; 0 = no "
                     "trim: the surface stays CLOSED, so every component has a mass and the mass/cavity rules of "
                     "10.10 apply as to marching cubes (a trim opens the surface into flaps that pass the mass rule "
                     "through the body's voxel label: cow video, drawn pieces > 1 in 143 of 292 frames at 3 sigma)")
ap.add_argument("--pca_sigma", type=float, default=0.0,
                help="S1 kernel size in spacings (the anisotropic kernel keeps this isotropic volume); 0 = --blur, "
                     "the baseline kernel's size, so only the SHAPE of the kernel differs from the gallery")
ap.add_argument("--imls_h", type=float, default=2.0, help="S3 kernel width in particle spacings")
ap.add_argument("--bulk", default="particle", choices=["particle", "voxel"],
                help="the bulk density the level is a fraction of: median over the particles (correct) or over "
                     "the occupied voxels (the gallery up to v8: biased low by the blur's halo)")
ap.add_argument("--layer", default="grad", choices=["grad", "density"],
                help="outer-layer rule for --surface poisson|imls|surfel: 'grad' = relative density gradient "
                     "|grad rho|/rho above the half-space value one spacing deep (invariant to the local density; "
                     "a stretched region of a morph frame is not all 'layer'); 'density' = density below the "
                     "half-space value one spacing deep (the first reading)")
ap.add_argument("--pull", type=int, default=1,
                help="surface poisson|imls|surfel: plane-pulling iterations of the outer layer (weighted PCA "
                     "normals, positions projected onto the local plane); 0 = raw positions and gradient normals")
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
from physmorph.render.surface_recon import (pca_kernels, surface_particles, surface_particles_grad,  # noqa: E402
                                            layer_threshold, layer_threshold_grad, oriented_layer, poisson_mesh, exterior_surfels,
                                            imls_grid, surfel_mesh, bilateral_normal_smooth, trilinear)
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
    cen, cov = pca_kernels(x_np, spacing, k=a.pca_k, kr=a.pca_kr, lam=a.pca_lam, sigma0=pca_sigma_sp)
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


pca_sigma_sp = a.pca_sigma if a.pca_sigma > 0 else a.blur
rho0 = density(x0) if a.kernel == "iso" else (density_pca(x0) if a.kernel == "pca" else density_aniso(x0, frame_F(0, x0)))
occ = rho0[rho0 > 0]
bulk_voxel = float(occ.median()) if occ.numel() else 1.0
# the bulk = the density a typical PARTICLE sees (median over the particles). The median over
# occupied VOXELS (the gallery up to v8) is pulled down by the blur's halo — 3 sigma = 4.5
# spacings of sub-bulk voxels around the whole body — so the "0.283 x bulk" level was in fact
# ~0.14 of the interior density and every surface sat 1.6–1.9 spacings outside the true one
# (docs/experiments.md 2026-09-19, surface_gt on bunny/dragon/cow)
bulk_particle = float(trilinear(x0, rho0, ctr, half, vox).median())
rho_bulk = bulk_particle if a.bulk == "particle" else bulk_voxel
print(f"[photoreal] bulk density: particle median {bulk_particle:.4g}, occupied-voxel median {bulk_voxel:.4g} "
      f"({bulk_voxel / bulk_particle:.3f} of it); using {a.bulk}", flush=True)
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
kernel_sigma_sp = (sig_vox * vox / spacing) if a.kernel == "iso" else (pca_sigma_sp if a.kernel == "pca" else a.sigma0)
layer_thr = layer_threshold(rho_bulk, kernel_sigma_sp)
layer_gthr = layer_threshold_grad(kernel_sigma_sp, spacing)
if a.surface != "mc":
    print(f"[photoreal] surface {a.surface}: outer layer = " +
          (f"|grad rho| / rho >= {layer_gthr * spacing:.3f} per spacing" if a.layer == "grad"
           else f"density <= {layer_thr / rho_bulk:.3f} x bulk") + f" (kernel sigma {kernel_sigma_sp:.2f} spacings)", flush=True)


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


def levelset_particle_labels(x_np, rho):
    """Per-particle label of the level-set component enclosing it (0 = none), the drawn mask over
    labels (the volume rule on voxels) and the body label — the marching-cubes surface's notion
    of 'enclosed'."""
    from scipy import ndimage
    mask = rho >= iso
    lab, n_lab = ndimage.label(mask)
    counts = np.bincount(lab.ravel(), minlength=n_lab + 1)
    drawn = np.zeros(n_lab + 1, bool)
    drawn[1:] = counts[1:] * vox ** 3 >= min_vol                # the volume rule, on voxels
    body = int(np.argmax(counts[1:]) + 1) if n_lab else 0
    p = (x_np - (ctr - half).cpu().numpy()) / vox
    ijk = np.clip(np.rint(p).astype(np.int64), 0, G - 1)
    plab = lab[ijk[:, 2], ijk[:, 1], ijk[:, 0]]                  # voxel component of each particle
    return plab, drawn, body


def mesh_particle_labels(m, x_np):
    """Per-particle label of the DRAWN mesh component enclosing it (0 = none): occupancy of the
    closed mesh (Open3D ray casting) and the component of the nearest triangle. The Poisson
    surface is the true boundary, tighter than the blurred level set, so the level set can join
    two pieces a one-spacing neck separates on the drawn surface (cow video: drawn pieces > 1 in
    141 frames while the voxel labels saw one body and drew no bridge). The bridge rule must
    read the surface that is drawn."""
    comp = np.asarray(m.cluster_connected_triangles()[0])
    n_lab = int(comp.max()) + 1 if len(comp) else 0
    if n_lab == 0:
        return np.zeros(len(x_np), np.int64), np.zeros(1, bool), 0
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    q = o3d.core.Tensor(np.asarray(x_np, np.float32))
    inside = sc.compute_occupancy(q).numpy() > 0.5
    cp = sc.compute_closest_points(q)
    pid = cp["primitive_ids"].numpy().astype(np.int64)
    dist = np.linalg.norm(cp["points"].numpy() - np.asarray(x_np, np.float32), axis=1)
    # the surface passes THROUGH the outer particle layer (the surfels it was fitted to), so a
    # particle within one spacing of the drawn surface is a surface particle of that component,
    # not free material: without this tolerance half the layer is "free" and the bridge rule
    # draws it all (cow 420: 31 M triangles of filament)
    enclosed = inside | (dist <= spacing)
    plab = np.where(enclosed, comp[np.clip(pid, 0, len(comp) - 1)] + 1, 0).astype(np.int64)
    counts = np.bincount(plab, minlength=n_lab + 1)
    drawn = np.ones(n_lab + 1, bool); drawn[0] = False           # every kept component is drawn
    body = int(np.argmax(counts[1:]) + 1)
    return plab, drawn, body


def filament_bridges(x_np, plab, drawn, body, drawn_labels_needed=2):
    """Particles the drawn surface does not enclose but which link the body to another enclosed
    component are drawn as a filament one particle spacing thick (docs/method.md 10.10): a
    feature thinner than a two-particle bundle (the cow's teat: a 72-particle bulb on the
    target tied to the udder by a single-particle thread) is connected material at the
    particle scale, and the rendered topology must follow the particles, not the threshold.
    plab/drawn/body from levelset_particle_labels (marching cubes) or mesh_particle_labels
    (the Poisson surface). Returns (filament mesh or None, number of enclosed components
    bridged to the body)."""
    from scipy.sparse import coo_matrix
    if drawn.sum() < drawn_labels_needed or body == 0:
        return None, 0
    from scipy.sparse.csgraph import dijkstra
    free = np.where(plab == 0)[0]
    anch = np.where(drawn[plab])[0]
    if len(free) == 0 or len(anch) == 0:
        return None, 0
    # link radius = the MPM cell (the continuum's own resolution; scripts/probes/grid_fragments.py calls
    # material a fragment only when it shares no dilated cell with the body), never below the
    # 2.5-spacing particle-thread radius. With the surface at the true boundary the expansion-phase
    # leader clusters (>= a cell of particles, 3-4 spacings off the body) were "drawn, unbridged"
    # pieces at 2.5 spacings (bunny 45-66, cow 24-72, dragon 210-285) while the grid probe counts 0
    # fragments there: they sit within one cell of the body.
    r = max(2.5 * spacing, cell_wu)
    kf = cKDTree(x_np[free]); ka = cKDTree(x_np[anch])
    ff = np.array(list(kf.query_pairs(r)), dtype=np.int64).reshape(-1, 2)
    fa = kf.query_ball_tree(ka, r)
    # graph nodes: free particles (0..nf-1) then one super-node per drawn label; edge weights =
    # distances, so the path drawn is the SHORTEST particle chain from the body to the piece.
    # (The earlier form drew every free particle in the connected cluster: with the surface at
    # the true boundary the expansion-phase spray outside the body is free, and bunny frame 63
    # became 383 M triangles of filament. The rule is "the particles that link", not "every
    # particle that touches the chain".)
    nf = len(free); sup = {l: nf + i for i, l in enumerate(np.where(drawn)[0])}
    w_ff = np.linalg.norm(x_np[free[ff[:, 0]]] - x_np[free[ff[:, 1]]], axis=1) if len(ff) else np.zeros(0)
    rows_, cols_, wts_ = [ff[:, 0], ff[:, 1]], [ff[:, 1], ff[:, 0]], [w_ff, w_ff]
    fa_r, fa_c, fa_w = [], [], []
    edge_anchor = {}                                           # (free i, label) -> nearest anchor particle
    for i, nb in enumerate(fa):
        for j in nb:
            l = int(plab[anch[j]]); d = float(np.linalg.norm(x_np[free[i]] - x_np[anch[j]]))
            key = (i, l)
            if key not in edge_anchor or d < edge_anchor[key][1]:
                edge_anchor[key] = (j, d)
    for (i, l), (j, d) in edge_anchor.items():
        fa_r.append(i); fa_c.append(sup[l]); fa_w.append(d)
    rows_.append(np.array(fa_r, np.int64)); cols_.append(np.array(fa_c, np.int64)); wts_.append(np.array(fa_w))
    # DIRECT contact between two drawn components (2026-09-22, g40 cow frames 267/270/438/480): when the
    # surface breaks across a thin feature while the particles continue, the connecting particles lie
    # within the one-spacing tolerance of both caps and are all "enclosed" — no free particle to walk
    # through, yet the pieces are within one cell at the particle level (a single connected component
    # at r; the grid probe counts no fragment). Edge between the two super-nodes = the nearest anchor
    # pair within r; the filament drawn is that pair.
    direct = {}
    aa = ka.query_pairs(r, output_type="ndarray")
    if len(aa):
        la, lb = plab[anch[aa[:, 0]]], plab[anch[aa[:, 1]]]
        dd = np.linalg.norm(x_np[anch[aa[:, 0]]] - x_np[anch[aa[:, 1]]], axis=1)
        for i0, i1, l0, l1, d in zip(aa[:, 0], aa[:, 1], la, lb, dd):
            if l0 == l1:
                continue
            key = (int(min(l0, l1)), int(max(l0, l1)))
            if key not in direct or d < direct[key][2]:
                direct[key] = (int(i0), int(i1), float(d))
    if direct:
        rows_.append(np.array([sup[k[0]] for k in direct], np.int64)); cols_.append(np.array([sup[k[1]] for k in direct], np.int64))
        wts_.append(np.array([v[2] for v in direct.values()]))
    nn_ = nf + len(sup)
    rr = np.concatenate(rows_); cc = np.concatenate(cols_); ww = np.concatenate(wts_) + 1e-9
    if len(rr) == 0:
        return None, 0
    gph = coo_matrix((ww, (rr, cc)), shape=(nn_, nn_)).tocsr()
    dist_, pred = dijkstra(gph, directed=False, indices=[sup[body]], return_predecessors=True)
    dist_, pred = dist_[0], pred[0]
    others = [l for l in sup if l != body and np.isfinite(dist_[sup[l]])]
    if not others:
        return None, 0
    rad = 0.55 * spacing
    fil = o3d.geometry.TriangleMesh()
    drawn_nodes = set()
    for l in others:
        path = [sup[l]]
        while path[-1] != sup[body] and pred[path[-1]] >= 0:
            path.append(int(pred[path[-1]]))
        if path[-1] != sup[body]:
            continue
        sup_lab = {v: k for k, v in sup.items()}
        # the segments along the path: free-free, free-super (the anchor of the edge used), super-super
        # (the direct anchor pair)
        for u_, v_ in zip(path[:-1], path[1:]):
            if u_ < nf and v_ < nf:
                ends = (x_np[free[u_]], x_np[free[v_]])
            elif u_ < nf:
                j = edge_anchor.get((u_, sup_lab[v_]))
                ends = (x_np[free[u_]], x_np[anch[j[0]]]) if j else None
            elif v_ < nf:
                j = edge_anchor.get((v_, sup_lab[u_]))
                ends = (x_np[anch[j[0]]], x_np[free[v_]]) if j else None
            else:
                key = (min(sup_lab[u_], sup_lab[v_]), max(sup_lab[u_], sup_lab[v_]))
                dp = direct.get(key)
                ends = (x_np[anch[dp[0]]], x_np[anch[dp[1]]]) if dp else None
            if ends is None:
                continue
            seg = _segment_mesh(ends[0], ends[1], rad)
            if seg is not None:
                fil += seg
        for n_ in path:
            if n_ < nf and n_ not in drawn_nodes:
                drawn_nodes.add(n_)
                sph = o3d.geometry.TriangleMesh.create_sphere(radius=rad, resolution=6)
                sph.translate(x_np[free[n_]]); fil += sph
    if len(fil.triangles) == 0:
        return None, len(others)
    fil.compute_vertex_normals()
    return fil, len(others)


FALLBACK_FRAMES = []   # frames whose Poisson reconstruction crashed twice and fell back to the level set


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
        if a.layer == "grad":
            pts, nrm = surface_particles_grad(x, rho_t, ctr, half, vox, layer_gthr)
        else:
            pts, nrm = surface_particles(x, rho_t, ctr, half, vox, layer_thr)
        # the exterior test (docs/surface_gradient.md 14): surfels with material on their outward side are
        # interior density steps, not surface; they would make Poisson draw an interior sheet
        pts, nrm, n_interior = exterior_surfels(pts, nrm, x.detach().cpu().numpy(), spacing)
        if a.pull > 0:
            pts, nrm = oriented_layer(pts, nrm, spacing, pull_iters=a.pull)
        if fi is None or fi == 0:
            print(f"[photoreal] surface {a.surface}: {n_interior} interior surfels dropped by the exterior test", flush=True)
            print(f"[photoreal] surface {a.surface}: {len(pts)} outer-layer particles of {len(x)}"
                  f"{' (plane-pulled, PCA normals)' if a.pull > 0 else ' (raw, gradient normals)'}", flush=True)
        if a.surface in ("poisson", "surfel"):
            pm = (poisson_mesh(pts, nrm, spacing, depth=a.poisson_depth, cell_sp=a.poisson_cell,
                               max_dist_sp=a.poisson_trim * kernel_sigma_sp, vox=vox)
                  if a.surface == "poisson" else surfel_mesh(pts, nrm))
            if pm is None:
                # the isolated Poisson child crashed twice (Open3D 0.19 segfaults now and then; a race,
                # not a frame): this frame falls back to the level set and the sidecar records it
                FALLBACK_FRAMES.append(fi)
                print(f"[photoreal] frame {fi}: Poisson failed twice -> marching cubes for this frame", flush=True)
                v, f, _, _ = measure.marching_cubes(rho, level=iso, spacing=(vox, vox, vox))
                v = v[:, ::-1] + origin
            else:
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
            if a.surface == "mc":
                for ci in range(n_comp):
                    zz, yy, xx = rv[ci, 2], rv[ci, 1], rv[ci, 0]
                    nb = vlab[max(zz - 1, 0):zz + 2, max(yy - 1, 0):yy + 2, max(xx - 1, 0):xx + 2]
                    labs = np.unique(nb[nb > 0])
                    mass[ci] = vcount[labs].max() if len(labs) else 0.0
                # the BODY is the component holding the most particles (ties: the larger enclosed volume)
                body = int(np.lexsort((np.abs(svol), mass))[-1])
                body_sign = np.sign(svol[body]) if svol[body] != 0 else 1.0
                cavity = (np.sign(svol) == -body_sign) & (svol != 0)
            else:
                # a reconstructed surface is not a level set: a piece next to the body would inherit the
                # body's voxel label (and its mass) and the body's own signed volume is not a safe sign
                # reference (bunny frame 63: the body classed as the cavity of a spray blob and removed).
                # The mass of a closed component is the number of particles it ENCLOSES (ray-casting
                # occupancy, queried over the particles in its bounding box); the body is the component
                # with the most; a light component whose centroid the body encloses is a cavity.
                cents = np.zeros((n_comp, 3))
                scenes = []
                for ci in range(n_comp):
                    tri = ff[comp == ci]
                    sub = o3d.t.geometry.TriangleMesh(o3d.core.Tensor(vv.astype(np.float32)),
                                                      o3d.core.Tensor(tri.astype(np.int32)))
                    sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(sub); scenes.append(sc)
                    pv_ = vv[np.unique(tri)]
                    cents[ci] = pv_.mean(0)
                    lo_, hi_ = pv_.min(0) - vox, pv_.max(0) + vox
                    inbox = np.where(((xp >= lo_) & (xp <= hi_)).all(1))[0]
                    if len(inbox):
                        occ_ = sc.compute_occupancy(o3d.core.Tensor(xp[inbox].astype(np.float32))).numpy()
                        mass[ci] = float((occ_ > 0.5).sum())
                body = int(np.lexsort((np.abs(svol), mass))[-1])
                inside_body = scenes[body].compute_occupancy(o3d.core.Tensor(cents.astype(np.float32))).numpy() > 0.5
                # every component the body encloses is interior — a void (light) or a closed sheet the
                # layer rule drew around a density step INSIDE the material (heavy: spot frame 114, a
                # second "drawn piece" nobody can see). Neither is a piece; both are removed and counted
                # in the cavity column.
                cavity = inside_body.copy()
                cavity[body] = False
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
        x_np64 = x.detach().cpu().numpy().astype(np.float64)
        plab, drawn, body_lab = (levelset_particle_labels(x_np64, rho) if a.surface == "mc"
                                 else mesh_particle_labels(m, x_np64))
        fil, n_bridge = filament_bridges(x_np64, plab, drawn, body_lab)
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


# ---- surface tracking (docs/method.md 10.15) ---------------------------------------------
def advect_vertices(V, x_prev, x_cur, k, h):
    """Move mesh vertices with the material: each vertex is bound to its k nearest particles of the
    PREVIOUS frame (Gaussian weights of width h = one spacing) and moves by their weighted mean
    displacement to the current frame. Particles that are neighbours at one frame are neighbours at
    the next, so the binding is renewed every frame."""
    kd = cKDTree(x_prev)
    d, j = kd.query(V, k=k, workers=-1)
    w = np.exp(-(d / h) ** 2)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
    return V + (w[:, :, None] * (x_cur[j] - x_prev[j])).sum(1)


def closest_on(mesh, P):
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    r = sc.compute_closest_points(o3d.core.Tensor(np.asarray(P, np.float32)))
    return r["points"].numpy().astype(np.float64)


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
                       "pull": a.pull, "layer": a.layer, "poisson_cell": a.poisson_cell, "poisson_trim": a.poisson_trim, "pca_sigma": pca_sigma_sp, "label": a.label, "bulk": a.bulk, "bulk_voxel_over_particle": bulk_voxel / bulk_particle,
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
trk = None            # (tracked mesh, particles at its last frame, drawn topology)
prev_fresh = None     # (fresh mesh, particles) of the previous video frame — the re-fit jitter reference
for k, i in enumerate(idx):
    x_np = np.asarray(frames_np[i], np.float32)
    m, n_comp, n_drop, n_bridge, n_cav = mesh_of(torch.as_tensor(x_np, device=dev), i)
    n_iso = isolated_count(x_np)
    x64 = x_np.astype(np.float64)
    jitter = drift = float("nan"); remeshed = 0
    if m is not None and prev_fresh is not None and prev_fresh[0] is not None and len(m.triangles) > 0:
        # the re-fit jitter of an independent reconstruction per frame: the previous fresh mesh carried
        # along with the material against the current fresh mesh (spacings)
        Vp = advect_vertices(np.asarray(prev_fresh[0].vertices), prev_fresh[1], x64, a.track_k, spacing)
        jitter = float(np.linalg.norm(closest_on(m, Vp) - Vp, axis=1).mean() / spacing)
    prev_fresh = (m, x64)
    draw_m = m
    if a.track and m is not None and len(m.triangles) > 0:
        topo = (n_comp - n_drop - n_cav, n_bridge, n_cav)
        if trk is None:
            trk = (copy.deepcopy(m), x64, topo); remeshed = 1
        else:
            V = advect_vertices(np.asarray(trk[0].vertices), trk[1], x64, a.track_k, spacing)
            P = closest_on(m, V)
            drift = float(np.linalg.norm(P - V, axis=1).mean() / spacing)
            if topo != trk[2] or drift > a.track_tol:
                trk = (copy.deepcopy(m), x64, topo); remeshed = 1
            else:
                tm = trk[0]
                tm.vertices = o3d.utility.Vector3dVector(V + a.track_alpha * (P - V))
                tm.compute_vertex_normals()
                trk = (tm, x64, topo)
        draw_m = trk[0]
    qa.append((i, n_comp, n_iso, n_drop, n_bridge, n_cav, jitter, drift, remeshed))
    img = label(render_views(draw_m), f"{a.label}  frame {i}/{dn - 1}")
    o3d.io.write_image(os.path.join(tmp, f"f{k:05d}.png"), o3d.geometry.Image(np.ascontiguousarray(img)))
    if k % 25 == 0:
        print(f"[photoreal] frame {k + 1}/{len(idx)} (archived {i}) components {n_comp} dropped {n_drop} isolated {n_iso}", flush=True)
n = len(idx)
for h in range(a.hold):
    os.link(os.path.join(tmp, f"f{n - 1:05d}.png"), os.path.join(tmp, f"f{n + h:05d}.png"))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(tmp, "f%05d.png"),
                "-movflags", "faststart", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", a.out], check=True)
with open(a.out + ".components.txt", "w") as fh:
    fh.write("archived_frame isosurface_components isolated_particles subcell_components_dropped components_bridged_to_body interior_cavities refit_jitter_sp track_drift_sp remeshed\n")
    for i, n_comp, n_iso, n_drop, n_bridge, n_cav, jit, drf, rm in qa:
        fh.write(f"{i} {n_comp} {n_iso} {n_drop} {n_bridge} {n_cav} {jit:.4f} {drf:.4f} {rm}\n")
    comps = np.array([q[1] for q in qa]); isos = np.array([q[2] for q in qa]); drops = np.array([q[3] for q in qa])
    bridges = np.array([q[4] for q in qa]); cavs = np.array([q[5] for q in qa])
    jits = np.array([q[6] for q in qa], float); drfs = np.array([q[7] for q in qa], float); rms = np.array([q[8] for q in qa])
    drawn = comps - drops - cavs
    fh.write(f"# re-fit jitter of the independent per-frame reconstruction (previous fresh mesh carried with the material vs "
             f"the current fresh mesh, spacings): mean {np.nanmean(jits):.3f}, p90 {np.nanpercentile(jits, 90):.3f}; "
             f"tracking {'ON' if a.track else 'off'}"
             + (f": re-meshed {int(rms.sum())} frames (topology change or drift > {a.track_tol} sp), tracked drift before the pull "
                f"mean {np.nanmean(drfs):.3f} sp, pull alpha {a.track_alpha}, k {a.track_k}" if a.track else "") + "\n")
    fh.write(f"# interior cavities (closed surfaces with the sign opposite to the body, removed, not pieces): "
             f"{(cavs > 0).sum()} frames (max {cavs.max()})\n")
    fh.write(f"# filament bridges (particle connectivity): {(bridges > 0).sum()} frames with a drawn component tied to the "
             f"body by particles the isosurface does not enclose (max {bridges.max()}); drawn components>1 AND not bridged "
             f"in {((drawn > 1) & (bridges < drawn - 1)).sum()} frames\n")
    fh.write(f"# surface {a.surface}" + (f" (outer layer by {a.layer}, pull {a.pull}, octree cell {a.poisson_cell} spacing, "
             f"trim {a.poisson_trim} sigma; docs/method.md 10.12); Poisson fallback to the level set in "
             f"{len(FALLBACK_FRAMES)} frames {FALLBACK_FRAMES[:20]}" if a.surface == "poisson" else "") +
             f"; bulk = {a.bulk} median (voxel/particle {bulk_voxel / bulk_particle:.3f})\n")
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
