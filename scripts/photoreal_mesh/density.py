"""Density grids of the photoreal renderer on the G^3 voxel grid of the render box ([z,y,x] indexing,
unit mass per particle): the isotropic kernel (CIC deposit + separable Gaussian blur, the grid
render_iso_video.py uses), the anisotropic kernel carried by the deformation gradient (geometric or
archived F) and the Yu & Turk PCA kernels (S1). Every function reads the run state from the context
namespace of photoreal_mesh.context."""
import math

import numpy as np
import torch
import torch.nn.functional as Fn
from scipy.spatial import cKDTree

from physmorph.render.covariance import select_archive_F
from physmorph.render.surface_recon import pca_kernels


def density(ctx, x):
    """Isotropic kernel: trilinear (CIC) deposit of the particles, then a separable Gaussian blur of
    sig_vox voxels (--blur particle spacings)."""
    G, dev, ctr, half, vox, sig_vox = ctx.G, ctx.dev, ctx.ctr, ctx.half, ctx.vox, ctx.sig_vox
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


def geometric_F(ctx, x):
    """Total deformation gradient of each particle's material patch at the current frame: the
    least-squares map from its rest neighbour offsets (frame 0, k nearest) to the current ones,
    F = (sum d d0^T)(sum d0 d0^T)^-1 — the PhysGaussian kinematics of the patch, plastic flow
    included, which the archived elastic F cannot show."""
    a, dev, x0, spacing = ctx.a, ctx.dev, ctx.x0, ctx.spacing
    if ctx.knn_rest is None:
        x0n = x0.detach().cpu().numpy().astype(np.float32)
        _, nb = cKDTree(x0n).query(x0n, k=a.knn + 1, workers=-1)
        ctx.knn_rest = torch.as_tensor(nb[:, 1:], device=dev)
    _knn_rest = ctx.knn_rest
    d0 = x0[_knn_rest] - x0[:, None, :]                                   # (N,k,3) rest offsets
    d1 = x[_knn_rest] - x[:, None, :]                                     # (N,k,3) current offsets
    A = torch.einsum("nki,nkj->nij", d1, d0)                              # sum d d0^T
    B = torch.einsum("nki,nkj->nij", d0, d0) + 1e-6 * torch.eye(3, device=dev)[None] * float(spacing ** 2)
    return torch.linalg.solve(B.transpose(1, 2), A.transpose(1, 2)).transpose(1, 2)   # A B^-1


def frame_F(ctx, fi, x=None):
    """Deformation gradient at archived frame fi: geometric (default) or the archived elastic F."""
    a, dev, z, _orient, _F_cache = ctx.a, ctx.dev, ctx.z, ctx.orient, ctx.F_cache
    if a.F == "geom" and x is not None:
        return geometric_F(ctx, x)
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


def density_aniso(ctx, x, F):
    """Sum of per-particle Gaussians N(x_p, sigma0^2 F_p F_p^T) on the voxel grid: the kernel is
    the particle's material patch carried by the deformation (FALSIFIED as a smoother,
    docs/experiments.md 2026-09-18 item 4; kept as an option)."""
    a, dev, spacing = ctx.a, ctx.dev, ctx.spacing
    s0 = a.sigma0 * spacing
    F = torch.where(torch.isfinite(F).all(-1).all(-1)[:, None, None], F, torch.eye(3, device=dev)[None])
    # stretch saturation as in the objective's Gaussian forward model (cov_from_F sat): a
    # particle stretched beyond 3x rest keeps a 3x kernel, so a torn or inverted F cannot
    # paint a whole cell
    M = torch.einsum("nij,nkj->nik", F, F)
    Ms = torch.linalg.solve(torch.eye(3, device=dev)[None] + M / 9.0, M)
    M = 0.5 * (Ms + Ms.transpose(1, 2))
    cov = (s0 * s0) * M + (1e-3 * s0 * s0) * torch.eye(3, device=dev)[None]
    return density_from_cov(ctx, x, cov)


def density_pca(ctx, x):
    """S1, Yu & Turk (2013): kernel centres Laplacian-smoothed (lambda), covariances from the
    weighted PCA of each particle's neighbourhood (ratio clamp k_r), a sum of anisotropic
    Gaussians of the isotropic kernel's volume."""
    a, dev, spacing, pca_sigma_sp = ctx.a, ctx.dev, ctx.spacing, ctx.pca_sigma_sp
    x_np = x.detach().cpu().numpy().astype(np.float32)
    cen, cov = pca_kernels(x_np, spacing, k=a.pca_k, kr=a.pca_kr, lam=a.pca_lam, sigma0=pca_sigma_sp)
    return density_from_cov(ctx, torch.as_tensor(cen, device=dev), torch.as_tensor(cov, device=dev))


def density_from_cov(ctx, x, cov):
    """Sum of per-particle Gaussians N(x_p, cov_p) on the voxel grid, truncated at 3 sigma_max
    with a fixed stencil; unit mass per particle."""
    G, dev, ctr, half, vox = ctx.G, ctx.dev, ctx.ctr, ctx.half, ctx.vox
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
