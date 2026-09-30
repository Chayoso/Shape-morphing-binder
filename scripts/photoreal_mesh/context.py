"""Run context of the photoreal renderer, built once per run: the archive (frames and target in the
y-up orientation), the render box (a cube over every archived frame and the target) and its voxel
grid, the particle spacing, the density level (iso as a fraction of the bulk density), the outer-layer
thresholds of the reconstructed surfaces and the deliverable rule (MPM cell, particles per cell). The
other modules read what they need from the returned namespace; it also holds the caches and the lock
the prefetch threads share."""
import threading
from types import SimpleNamespace

import numpy as np
import torch

from physmorph.sampling.orientation import orient_archive
from physmorph.render.surface_recon import layer_threshold, layer_threshold_grad, trilinear
from photoreal_mesh.density import density, density_aniso, density_pca, frame_F


def build_context(a):
    """Load --npz and derive everything the frames share; prints the run header (orientation, bulk
    density, iso level, outer layer, deliverable rule). `a` = the parsed command line."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(a.npz, allow_pickle=True)
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
    pca_sigma_sp = a.pca_sigma if a.pca_sigma > 0 else a.blur
    ctx = SimpleNamespace(
        a=a, dev=dev, z=z, orient=_orient, frames_np=frames_np, tgt_np=tgt_np, dn=dn, tgt=tgt, x0=x0, N=N,
        G=G, ctr=ctr, half=half, vox=vox, spacing=spacing, sig_vox=sig_vox, pca_sigma_sp=pca_sigma_sp,
        knn_rest=None,          # rest-frame k nearest neighbours of every particle (geometric_F), built on first use
        F_cache={},             # the archived F of the current sample, oriented, on the device (frame_F)
        fallback_frames=[],     # frames whose Poisson reconstruction crashed twice and fell back to the level set
        layer_cache={},         # archived frame -> (surfel points, normals) of its outer layer (the surfel memory's input)
        layer_lock=threading.Lock())
    rho0 = density(ctx, x0) if a.kernel == "iso" else (density_pca(ctx, x0) if a.kernel == "pca" else density_aniso(ctx, x0, frame_F(ctx, 0, x0)))
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
    ctx.__dict__.update(bulk_voxel=bulk_voxel, bulk_particle=bulk_particle, rho_bulk=rho_bulk, iso_frac=iso_frac, iso=iso,
                        origin=origin, kernel_sigma_sp=kernel_sigma_sp, layer_thr=layer_thr, layer_gthr=layer_gthr,
                        cell_wu=cell_wu, min_vol=min_vol, ppc=ppc)
    return ctx
