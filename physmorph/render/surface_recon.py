"""Smooth surface reconstruction from a particle cloud — the candidates of docs/experiments.md
2026-09-19 (S1 Yu & Turk anisotropic kernels, S2 screened Poisson, S3 IMLS, S4 surfel
triangulation, S5 bilateral normal filtering). Every function takes numpy/torch arrays and returns plain data; the renderer
(scripts/render_photoreal.py) wires them behind --kernel / --surface / --post.

References: Yu & Turk, "Reconstructing surfaces of particle-based fluids using anisotropic
kernels", TOG 2013 (weighted PCA covariance, eigenvalue ratio clamp k_r = 4, centre smoothing
lambda = 0.9); Kazhdan & Hoppe, "Screened Poisson surface reconstruction", TOG 2013 (Open3D);
Öztireli, Guennebaud, Gross, "Feature preserving point set surfaces based on non-linear kernel
regression", CGF 2009 (RIMLS); Zheng et al., "Bilateral normal filtering for mesh denoising",
TVCG 2011.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial import cKDTree


# ---- S1: Yu & Turk ---------------------------------------------------------------------------
def pca_kernels(x_np: np.ndarray, spacing: float, k: int = 32, kr: float = 4.0, lam: float = 0.9,
                sigma0: float = 0.7, n_eps: int = 25, kn: float = 0.5, chunk: int = 20000):
    """Per-particle anisotropic kernels of Yu & Turk (2013).

    Weighted PCA over the k nearest neighbours with the cubic weight w(r) = 1 - (r/R)^3,
    R = the k-th neighbour distance (their r_i = 2h). Covariance eigenvalues sigma_1 >= sigma_2
    >= sigma_3 are clamped so sigma_k >= sigma_1 / kr (kr = 4), scaled so that the kernel keeps
    the isotropic volume sigma0^3 (their k_s rescale), and a particle with fewer than n_eps
    neighbours in range keeps an isotropic kernel of kn * sigma0. The kernel CENTRES are the
    Laplacian-smoothed positions x_bar = (1 - lam) x + lam * sum w x_j / sum w (lam = 0.9) —
    the step that removes the sampling noise of the centres before any kernel is placed.
    Returns (centres (N,3) float32, cov (N,3,3) float32)."""
    N = len(x_np)
    kd = cKDTree(x_np)
    d, nb = kd.query(x_np, k=k + 1, workers=-1)
    d, nb = d[:, 1:], nb[:, 1:]
    R = d[:, -1:]                                              # per-particle radius = k-th neighbour
    w = np.clip(1.0 - (d / np.maximum(R, 1e-9)) ** 3, 0.0, None)        # (N,k)
    wsum = w.sum(1, keepdims=True) + 1e-12
    xw = (w[:, :, None] * x_np[nb]).sum(1) / wsum                      # weighted mean
    cen = (1.0 - lam) * x_np + lam * xw                                # smoothed centres
    cov = np.zeros((N, 3, 3), np.float64)
    s0 = sigma0 * spacing
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        dxn = x_np[nb[s:e]] - xw[s:e, None, :]                         # (n,k,3)
        C = np.einsum("nk,nki,nkj->nij", w[s:e], dxn, dxn) / wsum[s:e, :, None]
        evals, evecs = np.linalg.eigh(C)                               # ascending
        evals = evals[:, ::-1]; evecs = evecs[:, :, ::-1]              # descending
        s1 = np.maximum(evals[:, :1], 1e-12)
        ev = np.maximum(evals, s1 / kr)                                # ratio clamp
        # rescale to the isotropic volume: prod(sigma) = s0^3 (their k_s normalisation)
        sig = np.sqrt(np.maximum(ev, 1e-12))
        scale = s0 / np.cbrt(np.prod(sig, axis=1, keepdims=True))
        sig = sig * scale
        Ci = np.einsum("nij,nj,nkj->nik", evecs, sig ** 2, evecs)
        few = (w[s:e] > 0).sum(1) < n_eps
        Ci[few] = (kn * s0) ** 2 * np.eye(3)[None]
        cov[s:e] = Ci
    return cen.astype(np.float32), cov.astype(np.float32)


# ---- surface particles + normals from the blurred density ---------------------------------------
def trilinear(x: torch.Tensor, rho: torch.Tensor, ctr, half: float, vox: float, with_grad: bool = False):
    """rho ([z,y,x] voxel grid, voxel centre (0,0,0) at ctr - half + vox/2) sampled trilinearly at
    the points x; with_grad also returns the central-difference gradient (x,y,z) there."""
    G = rho.shape[0]
    if with_grad:
        gz, gy, gx = torch.gradient(rho, spacing=(vox, vox, vox))
        grad = torch.stack([gx, gy, gz], -1)                              # (G,G,G,3) in x,y,z
    p = (x - (ctr - half)) / vox - 0.5
    i0 = torch.floor(p).long(); f = p - i0.float()
    val = torch.zeros(len(x), device=x.device)
    gv = torch.zeros(len(x), 3, device=x.device) if with_grad else None
    for dz_ in (0, 1):
        for dy_ in (0, 1):
            for dx_ in (0, 1):
                wgt = ((f[:, 0] if dx_ else 1 - f[:, 0]) * (f[:, 1] if dy_ else 1 - f[:, 1]) * (f[:, 2] if dz_ else 1 - f[:, 2]))
                i = (i0 + torch.tensor([dx_, dy_, dz_], device=x.device)).clamp(0, G - 1)
                val = val + wgt * rho[i[:, 2], i[:, 1], i[:, 0]]
                if with_grad:
                    gv = gv + wgt[:, None] * grad[i[:, 2], i[:, 1], i[:, 0]]
    return (val, gv) if with_grad else val


def surface_particles(x: torch.Tensor, rho: torch.Tensor, ctr, half: float, vox: float, thr: float):
    """The outer particle layer: particles whose smoothed density is below `thr`, with outward
    normals = -grad rho / |grad rho| sampled trilinearly. Under the half-space model the
    density at depth d (spacings) is bulk * Phi(d / sigma); the caller sets thr = bulk *
    Phi(1 / sigma_sp) — the value one spacing deep, between the first layer (depth 0.5) and
    the second (1.5) — so thr derives from the kernel width, not from a tuning. `bulk` must be
    the density a typical PARTICLE sees (the median over particles), not the median over
    occupied voxels: the blur's halo outside the body pulls the voxel median far below the
    interior value (bunny 150k: 434 particles selected instead of a layer).
    Returns (points (M,3) float32, normals (M,3) float32) on the CPU."""
    val, gv = trilinear(x, rho, ctr, half, vox, with_grad=True)
    gn = gv.norm(dim=1).clamp_min(1e-12)
    sel = (val <= thr) & (gn > 1e-9)
    n = -gv[sel] / gn[sel, None]                                          # outward = down the density
    return x[sel].detach().cpu().numpy().astype(np.float32), n.detach().cpu().numpy().astype(np.float32)


def surface_particles_grad(x: torch.Tensor, rho: torch.Tensor, ctr, half: float, vox: float, g_thr: float):
    """The outer particle layer by the RELATIVE density gradient |grad rho| / rho >= g_thr (per
    world unit). Invariant to the local density amplitude, which the density threshold is not:
    on a morph frame a stretched region sits at 0.6 x bulk throughout and the density rule
    takes all of it as 'layer' (several particles thick, garbage normals, holes in the
    Poisson surface), while its relative gradient is ~0 inside and large only at its edge.
    Under the half-space model |grad rho| / rho = phi(d / sigma) / (sigma Phi(d / sigma)); the
    caller sets g_thr at depth d = 1 spacing (layer_threshold_grad)."""
    val, gv = trilinear(x, rho, ctr, half, vox, with_grad=True)
    gn = gv.norm(dim=1).clamp_min(1e-12)
    sel = (gn / val.clamp_min(1e-12) >= g_thr) & (gn > 1e-9)
    n = -gv[sel] / gn[sel, None]
    return x[sel].detach().cpu().numpy().astype(np.float32), n.detach().cpu().numpy().astype(np.float32)


def oriented_layer(points: np.ndarray, ref_normals: np.ndarray, spacing: float, k: int = 24,
                   h_sp: float = 2.0, pull_iters: int = 1):
    """Denoised surfels from the outer layer: per particle a weighted PCA plane over its k
    nearest layer neighbours (Gaussian weights of width h_sp spacings — the layer's thickness),
    the normal = the plane normal oriented by the density-gradient reference, the position
    pulled onto the plane (the plane-pulling constraint of 3DGT, one MLS projection per
    iteration). The gradient normals of a blurred shot-noise density carry ~20 deg of noise
    each and the positions half a fill pitch; the plane fit over ~k particles removes most of
    both. Returns (pulled points (M,3) float32, normals (M,3) float32)."""
    P = points.astype(np.float64).copy()
    R = ref_normals.astype(np.float64)
    h = h_sp * spacing
    # a particle with no layer neighbour within ~3h is not a surface: an isolated particle (the
    # morph sheds a few, 1-3 wu from the body) passes the gradient rule, its weights underflow,
    # and the "plane" it would be pulled onto is undefined
    kd = cKDTree(P)
    d1 = kd.query(P, k=2, workers=-1)[0][:, 1] if len(P) > 1 else np.full(len(P), np.inf)
    keep = d1 <= 3.0 * h
    P, R = P[keep], R[keep]
    if len(P) <= k:
        return P.astype(np.float32), R.astype(np.float32)
    for _ in range(max(1, pull_iters)):
        kd = cKDTree(P)
        d, nb = kd.query(P, k=k + 1, workers=-1)
        d, nb = d[:, 1:], nb[:, 1:]
        # neighbours on the SAME side only (normal agreement): in a sheet two or three particles
        # thick the k nearest span both faces, the weighted centroid is the mid-plane and both faces
        # get pulled onto it — coincident surfels of opposite normal, a ragged Poisson surface (C at
        # 40k). The gradient reference normal separates the faces.
        w = np.exp(-(d / h) ** 2) * np.clip((R[nb] * R[:, None, :]).sum(-1), 0.0, None)
        wsum = w.sum(1, keepdims=True) + 1e-12
        c = (w[:, :, None] * P[nb]).sum(1) / wsum
        dxn = P[nb] - c[:, None, :]
        C = np.einsum("nk,nki,nkj->nij", w, dxn, dxn) / wsum[:, :, None]
        evals, evecs = np.linalg.eigh(C)                                   # ascending
        n = evecs[:, :, 0]                                                 # smallest variance = the normal
        flip = (n * R).sum(1) < 0
        n[flip] = -n[flip]
        P = P - ((P - c) * n).sum(1, keepdims=True) * n
        R = n
    return P.astype(np.float32), R.astype(np.float32)


def layer_by_asymmetry(x_np: np.ndarray, spacing: float, k: int = 32, thr_sp: float = 0.5):
    """Outer layer without a density grid: the offset of a particle from the centroid of its k
    nearest neighbours, in spacings, is ~0 inside and ~ (0.5 + ...) at the surface (SPH surface
    detection). thr_sp = 0.5 spacing = the depth of the first layer under the half-space model.
    Returns (mask (N,) bool, outward normal estimate (N,3) = -(centroid offset) normalised)."""
    kd = cKDTree(x_np)
    d, nb = kd.query(x_np, k=k + 1, workers=-1)
    c = x_np[nb[:, 1:]].mean(1)
    off = x_np - c
    n = np.linalg.norm(off, axis=1)
    mask = n >= thr_sp * spacing
    normal = off / (n[:, None] + 1e-12)
    return mask, normal.astype(np.float32)


def plane_residual(x_np: np.ndarray, mask: np.ndarray, ref_normals: np.ndarray, spacing: float,
                   k: int = 24, h_sp: float = 2.0):
    """Per layer particle, the signed distance to the same-side weighted PCA plane of its k layer
    neighbours (the plane pulling of oriented_layer, measured instead of applied). The RMS over
    the layer is the bump amplitude of the particle surface at the 2-spacing scale — no mesh, no
    kernel, resolution-free. Returns (residual (M,) in spacings, plane normals (M,3))."""
    P = x_np[mask].astype(np.float64); R = ref_normals[mask].astype(np.float64)
    if len(P) <= k:
        return np.zeros(len(P)), R.astype(np.float32)
    h = h_sp * spacing
    kd = cKDTree(P)
    d, nb = kd.query(P, k=k + 1, workers=-1)
    d, nb = d[:, 1:], nb[:, 1:]
    w = np.exp(-(d / h) ** 2) * np.clip((R[nb] * R[:, None, :]).sum(-1), 0.0, None)
    wsum = w.sum(1, keepdims=True) + 1e-12
    c = (w[:, :, None] * P[nb]).sum(1) / wsum
    dxn = P[nb] - c[:, None, :]
    C = np.einsum("nk,nki,nkj->nij", w, dxn, dxn) / wsum[:, :, None]
    evals, evecs = np.linalg.eigh(C)
    n = evecs[:, :, 0]
    flip = (n * R).sum(1) < 0
    n[flip] = -n[flip]
    res = ((P - c) * n).sum(1) / spacing
    return res, n.astype(np.float32)


def layer_relax_data(x0: np.ndarray, spacing: float, k: int = 24, h_sp: float = 2.0, thr_sp: float = 0.5):
    """Frozen per-window data of the outer-layer relaxation force (kernels.k_layer_resid /
    k_layer_force): (mask (N,) float, nrm (N,3), nbr (N,k) int, w (N,k)). Layer by the
    neighbourhood asymmetry; normals from the asymmetry offset; neighbours = the k nearest
    LAYER particles with Gaussian weights of h_sp spacings times the normal agreement (same
    side only). Rows of non-layer particles hold zeros."""
    N = len(x0)
    mask, nrm = layer_by_asymmetry(x0, spacing, thr_sp=thr_sp)
    idx = np.where(mask)[0]
    nbr = np.zeros((N, k), np.int32); w = np.zeros((N, k), np.float32)
    if len(idx) > k:
        P = x0[idx].astype(np.float64); R = nrm[idx].astype(np.float64)
        kd = cKDTree(P)
        d, nb = kd.query(P, k=k + 1, workers=-1)
        d, nb = d[:, 1:], nb[:, 1:]
        ww = np.exp(-(d / (h_sp * spacing)) ** 2) * np.clip((R[nb] * R[:, None, :]).sum(-1), 0.0, None)
        ww = ww / np.maximum(ww.sum(1, keepdims=True), 1e-12)     # rows sum to 1: no division in the kernels
        nbr[idx] = idx[nb]
        w[idx] = ww.astype(np.float32)
    return mask.astype(np.float32), nrm.astype(np.float32), nbr, w


def target_surface_normals(x_np: np.ndarray, spacing: float, k: int = 24, h_sp: float = 2.0):
    """G1 (docs/surface_gradient.md §4): per-particle normals and surface weights of a cloud from
    its RECONSTRUCTED surface — outer layer by the asymmetry rule, plane-pulled surfels, screened
    Poisson (isolated), then for every particle the normal of the nearest triangle and the weight
    exp(-(dist / spacing)^2) (1 on the surface, ~0 two spacings inside). The shading target
    rendered from these carries the sampled cloud's shot noise no longer. Returns
    (normals (N,3) float32, weights (N,) float32)."""
    import open3d as o3d
    mask, ref = layer_by_asymmetry(x_np, spacing)
    if mask.sum() <= k + 1:
        return ref.astype(np.float32), mask.astype(np.float32)
    pts, nrm = oriented_layer(x_np[mask], ref[mask], spacing, k=k, h_sp=h_sp)
    mesh = poisson_mesh(pts, nrm, spacing, max_dist_sp=0.0) if len(pts) > k + 1 else None
    if mesh is None or len(mesh.triangles) == 0:
        return ref.astype(np.float32), mask.astype(np.float32)
    mesh.compute_triangle_normals()
    tn = np.asarray(mesh.triangle_normals)
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    r = sc.compute_closest_points(o3d.core.Tensor(np.asarray(x_np, np.float32)))
    pid = r["primitive_ids"].numpy().astype(np.int64)
    dist = np.linalg.norm(r["points"].numpy() - np.asarray(x_np, np.float32), axis=1)
    n = tn[np.clip(pid, 0, len(tn) - 1)]
    # orient outward: agree with the asymmetry reference where that is defined
    flip = (n * ref).sum(1) < 0
    n = n.copy(); n[flip & mask] = -n[flip & mask]
    w = np.exp(-(dist / spacing) ** 2)
    return n.astype(np.float32), w.astype(np.float32)


def layer_threshold(bulk: float, sigma_sp: float) -> float:
    """bulk * Phi(1 / sigma_sp): the half-space density one spacing deep for a Gaussian kernel of
    sigma_sp spacings (1.5 -> 0.748 bulk; 0.7 -> 0.923 bulk)."""
    return float(bulk) * 0.5 * (1.0 + math.erf((1.0 / sigma_sp) / math.sqrt(2.0)))


def layer_threshold_grad(sigma_sp: float, spacing: float) -> float:
    """phi(1/sigma) / (sigma Phi(1/sigma)) per spacing, i.e. the relative density gradient of a
    half-space one spacing deep (sigma 1.5 -> 0.285 / spacing), returned per world unit."""
    u = 1.0 / sigma_sp
    phi = math.exp(-0.5 * u * u) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + math.erf(u / math.sqrt(2.0)))
    return phi / (sigma_sp * Phi) / spacing


# ---- S4: surfel triangulation, the geometric part of 3D Gaussian Triangulation -------------------
def surfel_mesh(points: np.ndarray, normals: np.ndarray, k: int = 16, cos_min: float = 0.9,
                pull_iters: int = 2, laplace_iters: int = 3):
    """Oriented surfels -> triangles without an implicit. (1) Plane pulling: each point is moved
    onto the weighted local plane of its k neighbours (one MLS projection, `pull_iters` times) —
    the constraint sum |n_i^T (mu_i - p_i)| of 3DGT (arXiv 2607.10690) solved directly.
    (2) Tangent-plane angular-greedy triangulation: for each point its neighbours with normal
    agreement > cos_min are sorted by angle in the tangent plane and consecutive pairs become
    triangles (a triangle is kept once, keyed by its sorted vertex ids). (3) Laplacian remeshing:
    `laplace_iters` tangential Laplacian steps over the mesh's vertex adjacency. Returns an
    Open3D mesh (not guaranteed watertight: a triangle soup consistent with the surfels)."""
    import open3d as o3d
    P = points.astype(np.float64).copy(); Nn = normals.astype(np.float64).copy()
    Nn /= np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-12
    kd = cKDTree(P)
    d, nb = kd.query(P, k=k + 1, workers=-1)
    d, nb = d[:, 1:], nb[:, 1:]
    h = float(np.median(d[:, -1]))
    for _ in range(pull_iters):
        w = np.exp(-(d / h) ** 2) * np.clip((Nn[nb] * Nn[:, None, :]).sum(-1), 0.0, None)
        wsum = w.sum(1, keepdims=True) + 1e-12
        c = (w[:, :, None] * P[nb]).sum(1) / wsum                         # weighted neighbour centroid
        n_avg = (w[:, :, None] * Nn[nb]).sum(1) / wsum
        n_avg /= np.linalg.norm(n_avg, axis=1, keepdims=True) + 1e-12
        P = P - ((P - c) * n_avg).sum(1, keepdims=True) * n_avg           # project onto the local plane
        Nn = n_avg
        kd = cKDTree(P); d, nb = kd.query(P, k=k + 1, workers=-1); d, nb = d[:, 1:], nb[:, 1:]
    # tangent frames
    a_ = np.where(np.abs(Nn[:, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    u = np.cross(Nn, a_); u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
    v = np.cross(Nn, u)
    rel = P[nb] - P[:, None, :]                                           # (N,k,3)
    ok = (Nn[nb] * Nn[:, None, :]).sum(-1) > cos_min
    ang = np.arctan2((rel * v[:, None, :]).sum(-1), (rel * u[:, None, :]).sum(-1))
    ang = np.where(ok, ang, np.nan)
    order = np.argsort(np.where(np.isnan(ang), 1e9, ang), axis=1)
    tris = []
    Npts = len(P)
    idx = np.arange(Npts)
    for j in range(k):
        j2 = (j + 1) % k
        a1 = np.take_along_axis(nb, order[:, j:j + 1], 1)[:, 0]
        a2 = np.take_along_axis(nb, order[:, j2:j2 + 1], 1)[:, 0]
        v1 = np.take_along_axis(ang, order[:, j:j + 1], 1)[:, 0]
        v2 = np.take_along_axis(ang, order[:, j2:j2 + 1], 1)[:, 0]
        good = ~np.isnan(v1) & ~np.isnan(v2) & (a1 != a2)
        gap = (v2 - v1) % (2 * np.pi)
        good &= gap < np.pi                                                 # no triangle across a boundary gap
        tris.append(np.stack([idx[good], a1[good], a2[good]], 1))
    T = np.concatenate(tris, 0) if tris else np.zeros((0, 3), np.int64)
    if len(T) == 0:
        return o3d.geometry.TriangleMesh()
    key = np.sort(T, axis=1)
    _, first = np.unique(key, axis=0, return_index=True)
    T = T[first]
    # orient consistently with the surfel normals
    fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    flip = (fn * Nn[T[:, 0]]).sum(1) < 0
    T[flip] = T[flip][:, [0, 2, 1]]
    # tangential Laplacian remeshing
    from scipy import sparse
    for _ in range(laplace_iters):
        e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], 0)
        A = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(Npts, Npts))
        A = ((A + A.T) > 0).astype(np.float64).tocsr()
        deg = np.asarray(A.sum(1)).ravel()
        has = deg > 0
        lap = np.zeros_like(P); lap[has] = (A @ P)[has] / deg[has, None] - P[has]
        lap -= (lap * Nn).sum(1, keepdims=True) * Nn                        # tangential only
        P = P + 0.5 * lap
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(P), o3d.utility.Vector3iVector(T.astype(np.int32)))
    m.remove_degenerate_triangles(); m.remove_unreferenced_vertices()
    m.compute_vertex_normals()
    return m


# ---- S2: screened Poisson ---------------------------------------------------------------------
def poisson_mesh(points: np.ndarray, normals: np.ndarray, spacing: float, depth: int = 0,
                 cell_sp: float = 1.0, max_dist_sp: float = 2.0, vox: float = 0.0):
    """Open3D screened Poisson reconstruction of the oriented outer layer.

    depth = 0 picks the octree depth from the discretisation: the finest cell equals the
    layer's in-plane sample spacing (cell_sp = 1 spacing; a cell holds about one surfel, and
    the quadratic B-spline basis spanning three cells averages the layer's normal-direction
    noise — the half-pitch jitter of the fill — instead of following it):
    depth = ceil(log2(extent / (cell_sp * spacing))). A cell of two spacings (tried first)
    leaves the surface too far from the samples and the trim below removes it. Vertices
    farther than max_dist_sp spacings from every layer particle are removed: Poisson closes
    every gap with a hallucinated envelope, and no particle supports a surface there. The
    caller passes the reach of the density kernel (3 sigma of the blur, 4.5 spacings): a trim
    at 2 spacings cut holes wherever a morph frame's layer was locally sparse (dragon frame
    400, 13 components)."""
    import open3d as o3d
    from .poisson_worker import poisson_isolated
    ext = float(np.max(points.max(0) - points.min(0)))
    if depth <= 0:
        depth = int(math.ceil(math.log2(max(ext / (cell_sp * spacing), 2.0))))
    res = poisson_isolated(points, normals, depth)
    if res is None:
        return None
    v, f = res
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v.astype(np.float64)), o3d.utility.Vector3iVector(f))
    if len(v) and max_dist_sp > 0:
        d = cKDTree(points).query(v, k=1, workers=-1)[0]
        mesh.remove_vertices_by_mask(d > max_dist_sp * spacing)
    mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
    # the octree cell sets the triangle size (about one spacing); the level-set mesh's is the render
    # voxel. Loop-subdivide until the triangles are no coarser than the voxel, so that a 40k cloud
    # (spacing 2.6 voxels) does not render as facets: iterations = ceil(log2(cell / vox)).
    if vox > 0 and len(mesh.triangles):
        n_sub = int(math.ceil(math.log2(max(cell_sp * spacing / vox, 1.0))))
        if n_sub > 0:
            mesh = mesh.subdivide_loop(number_of_iterations=n_sub)
    return mesh


# ---- S3: IMLS implicit on the render grid ----------------------------------------------------
def imls_grid(points: np.ndarray, normals: np.ndarray, origin: np.ndarray, vox: float, G: int,
              h: float, k: int = 24, chunk: int = 200000):
    """Implicit moving-least-squares field f(x) = sum w_j n_j.(x - p_j) / sum w_j with
    w_j = exp(-|x - p_j|^2 / h^2) over the k nearest surface points, evaluated at every voxel
    centre within 2h of a surface point; elsewhere +-inf sign by nearest-normal test. Returns
    the field as a (G,G,G) [z,y,x] float32 array with the surface at f = 0 (inside negative)."""
    kd = cKDTree(points)
    zz, yy, xx = np.meshgrid(np.arange(G), np.arange(G), np.arange(G), indexing="ij")
    q = np.stack([xx, yy, zz], -1).reshape(-1, 3).astype(np.float32) * vox + origin[None, :]
    f = np.full(len(q), np.nan, np.float32)
    for s in range(0, len(q), chunk):
        qe = q[s:s + chunk]
        d, nb = kd.query(qe, k=k, workers=-1)
        near = d[:, 0] <= 2.0 * h
        w = np.exp(-(d / h) ** 2)
        diff = qe[:, None, :] - points[nb]                                # (n,k,3)
        dot = (diff * normals[nb]).sum(-1)                                # (n,k)
        val = (w * dot).sum(1) / (w.sum(1) + 1e-12)
        # far from every sample: the sign of the nearest sample's plane, magnitude = distance
        far = ~near
        val[far] = np.sign(dot[far, 0]) * d[far, 0]
        f[s:s + chunk] = val
    return f.reshape(G, G, G)


# ---- S5: bilateral normal filtering (Zheng et al. 2011) ------------------------------------------
def bilateral_normal_smooth(mesh, iters: int = 5, sigma_s: float | None = None, sigma_n: float = 0.35,
                            vertex_iters: int = 10):
    """Filter face normals with weights = area * exp(-|c_i - c_j|^2 / 2 sigma_s^2) *
    exp(-|n_i - n_j|^2 / 2 sigma_n^2) over face neighbours (shared vertices), then move the
    vertices so that the faces agree with the filtered normals (Sun et al. update). Features
    (normal jumps larger than sigma_n) are preserved; the sampling ripple is removed."""
    import open3d as o3d
    from scipy import sparse
    v = np.asarray(mesh.vertices, np.float64); f = np.asarray(mesh.triangles)
    nF = len(f)
    # face adjacency through shared vertices
    vf = sparse.coo_matrix((np.ones(3 * nF), (f.ravel(), np.repeat(np.arange(nF), 3))), shape=(len(v), nF)).tocsr()
    A = (vf.T @ vf).tocoo()
    A = A.tocsr(); A.setdiag(0); A.eliminate_zeros(); A = A.tocoo()
    fi_, fj_ = A.row, A.col
    for _ in range(iters):
        e1 = v[f[:, 1]] - v[f[:, 0]]; e2 = v[f[:, 2]] - v[f[:, 0]]
        nrm = np.cross(e1, e2); area = 0.5 * np.linalg.norm(nrm, axis=1) + 1e-12
        nrm = nrm / (2 * area[:, None])
        cen = v[f].mean(1)
        if sigma_s is None:
            sigma_s = float(np.linalg.norm(cen[fi_] - cen[fj_], axis=1).mean())
        ws = np.exp(-((cen[fi_] - cen[fj_]) ** 2).sum(1) / (2 * sigma_s ** 2))
        wn = np.exp(-((nrm[fi_] - nrm[fj_]) ** 2).sum(1) / (2 * sigma_n ** 2))
        wgt = area[fj_] * ws * wn
        acc = np.zeros_like(nrm); acc_w = np.zeros(nF)
        np.add.at(acc, fi_, wgt[:, None] * nrm[fj_]); np.add.at(acc_w, fi_, wgt)
        nf = acc + area[:, None] * nrm                                     # include self
        nf = nf / (np.linalg.norm(nf, axis=1, keepdims=True) + 1e-12)
        # vertex update: v_i += (1/|F_i|) sum_f n_f n_f^T (c_f - v_i)
        for _v in range(vertex_iters):
            cen = v[f].mean(1)
            disp = np.zeros_like(v); cnt = np.zeros(len(v))
            for c in range(3):
                vi = f[:, c]
                dvec = cen - v[vi]
                proj = (dvec * nf).sum(1, keepdims=True) * nf
                np.add.at(disp, vi, proj); np.add.at(cnt, vi, 1.0)
            v = v + disp / np.maximum(cnt, 1)[:, None]
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f.astype(np.int32)))
    m.compute_vertex_normals()
    return m


def layer_grad_weights(x0: np.ndarray, mask: np.ndarray, nrm: np.ndarray, nbr: np.ndarray, w: np.ndarray,
                       spacing: float) -> np.ndarray:
    """P3 (docs/final_plan.md 2; kernels.k_layer_F): per layer particle the least-squares weights
    g (N,K,3) of the TANGENTIAL gradient over its frozen same-side neighbourhood, so that for a
    field d given on the layer, grad_t d(p) = sum_a (d_a - d_p) (x) g_a: r_a = x_a - x_p projected
    onto the tangent plane, M = sum_a w_a r_a r_a^T plus the normal ridge s n n^T that makes M
    invertible (s = the tangential scale trace(M)/2; the tangential block is the 2-D inverse), and
    g_a = w_a M^{-1} r_a. The normal derivative is not observable on a one-particle-thick sheet;
    k_layer_F takes it as d_p / depth with depth = one spacing (the layer thickness under the
    asymmetry rule). Rows off the layer are zero."""
    N, K = nbr.shape
    g = np.zeros((N, K, 3), np.float32)
    idx = np.where(mask > 0.5)[0]
    if len(idx) == 0:
        return g
    P = np.asarray(x0, np.float64)
    n = np.asarray(nrm, np.float64)[idx]
    r = P[nbr[idx]] - P[idx][:, None, :]                                   # (M,K,3)
    r = r - (r * n[:, None, :]).sum(-1, keepdims=True) * n[:, None, :]     # tangential part
    ww = np.asarray(w, np.float64)[idx]
    M = np.einsum("mk,mki,mkj->mij", ww, r, r)
    s = np.trace(M, axis1=1, axis2=2) / 2.0 + 1e-30
    ridge = s[:, None, None] * (np.einsum("mi,mj->mij", n, n) + 1e-6 * np.eye(3)[None])
    Minv = np.linalg.inv(M + ridge)
    g[idx] = (ww[:, :, None] * np.einsum("mij,mkj->mki", Minv, r)).astype(np.float32)
    return g


def layer_u_gate(x0: np.ndarray, tgt: np.ndarray, mask: np.ndarray, spacing: float,
                 sigma_sp: float = 1.5, nsig: float = 2.0, k: int = 256):
    """P2 (docs/final_plan.md 2; docs/surface_gradient.md 12): WHERE the u channel may act. Per
    layer particle the particle-scale density residual at the window start,
        r_p = (rho_morph(x_p) - rho_target(x_p)) / rho_ref,
    Gaussian kernel of sigma = sigma_sp spacings (the renderer's and the level set's surface
    scale), rho_ref = the target's bulk value; the gate is |r_p| > nsig * floor with floor = the
    shot-noise floor of a Poisson-process cloud at that width, (p/sigma)^{3/2} / sqrt(8 pi^{3/2}),
    p = the volumetric spacing = spacing / 1.24 (surface_gradient.md 9; the morph's cloud is
    disordered) and nsig standard deviations. A surface half a spacing off the target's crosses
    the 2-sigma gate (d rho / dn = 0.27 bulk per spacing at sigma = 1.5); a surface on the target
    and a saturated smooth region stay below it. Both densities are truncated at the same k
    nearest points so the truncation cancels in r. Returns (gate (N,) float32: 1 where u may act,
    the active share of the layer)."""
    N = len(x0)
    gate = np.zeros(N, np.float32)
    idx = np.where(np.asarray(mask) > 0.5)[0]
    if len(idx) == 0 or tgt is None or len(tgt) < k or N < k:
        return gate, 0.0
    sig = float(sigma_sp) * float(spacing)
    x0 = np.asarray(x0, np.float64)
    tgt = np.asarray(tgt, np.float64)
    kx, kt = cKDTree(x0), cKDTree(tgt)
    P = x0[idx]
    dm, _ = kx.query(P, k=k, workers=-1)
    dt, _ = kt.query(P, k=k, workers=-1)
    rho_m = np.exp(-0.5 * (dm / sig) ** 2).sum(1)
    rho_t = np.exp(-0.5 * (dt / sig) ** 2).sum(1)
    sub = tgt[np.random.default_rng(0).choice(len(tgt), min(len(tgt), 4000), replace=False)]
    dtt, _ = kt.query(sub, k=k, workers=-1)
    rho_ref = float(np.median(np.exp(-0.5 * (dtt / sig) ** 2).sum(1)))
    p_vol = float(spacing) / 1.24
    floor = (p_vol / sig) ** 1.5 / math.sqrt(8.0 * math.pi ** 1.5)
    r = (rho_m - rho_t) / max(rho_ref, 1e-12)
    on = np.abs(r) > float(nsig) * floor
    gate[idx[on]] = 1.0
    return gate, float(on.mean())


def exterior_surfels(points: np.ndarray, normals: np.ndarray, x_all: np.ndarray, spacing: float,
                     r_sp: float = 2.0, t_sp: float = 0.75, frac: float = 0.1):
    """The EXTERIOR test of the outer layer (docs/surface_gradient.md 14; method.md 10.12): a surfel
    is a surface only if the space on its outward side is empty of particles. The layer rule
    (|grad rho| / rho) also fires at a density step INSIDE the material (a 40 %-density pocket of
    the target, a compressed region of a morph), and the surfels it produces there make Poisson
    draw an interior sheet. Per surfel, count the particles in the outward cap of the 2-spacing
    ball beyond t_sp = 0.75 spacing (2.5 x the layer's own roughness: a true surface's ~12 layer
    neighbours in the cap footprint put 0.07 particles there on average); the cap holds cap_vol /
    p_vol^3 particles at bulk density (7.8 sp^3 x 1.9 = 14.8 at r 2, t 0.75) and 5.9 at the faintest
    pocket seen (40 % of bulk, bunny 14). The threshold frac = 1/10 of the bulk cap (1.5, i.e. two or
    more particles) separates the two Poisson counts: a true surfel is dropped with probability
    0.2 %, a 40 %-pocket surfel kept with 2 %. Returns (points, normals, n_dropped)."""
    if len(points) == 0:
        return points, normals, 0
    P = np.asarray(points, np.float64); Nn = np.asarray(normals, np.float64)
    r = float(r_sp) * spacing; t = float(t_sp) * spacing
    h = r - t
    cap_vol = math.pi * h * h * (3.0 * r - h) / 3.0
    p_vol = spacing / 1.24
    thr = float(frac) * cap_vol / p_vol ** 3
    kd = cKDTree(np.asarray(x_all, np.float64))
    lists = kd.query_ball_point(P, r, workers=-1)
    X = np.asarray(x_all, np.float64)
    cnt = np.zeros(len(P), np.int64)
    for i, l in enumerate(lists):
        if len(l) == 0:
            continue
        d = X[l] - P[i]
        cnt[i] = int(((d @ Nn[i]) > t).sum())
    keep = cnt <= thr
    return (np.asarray(points)[keep], np.asarray(normals)[keep], int((~keep).sum()))
