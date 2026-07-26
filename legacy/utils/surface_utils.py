"""
Surface-aware utilities for volumetric morphing.

This module provides a minimal fixed-surface extraction pipeline:
- reconstruct a proxy surface from the source particle cloud on a voxel grid
- map particles near that reconstructed surface to a persistent surface mask

The mask is then reused for surface-only rendering and correction while the
full particle set remains the volumetric state for MPM.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import trimesh


def _estimate_graph_normals(x_src: np.ndarray, nbr: np.ndarray) -> np.ndarray:
    """Estimate source-space normals once on the frozen shell graph."""
    n = x_src.shape[0]
    normals = np.zeros_like(x_src, dtype=np.float32)
    for i in range(n):
        nn = np.asarray(nbr[i], dtype=np.int32).reshape(-1)
        pts = x_src[np.concatenate(([i], nn), axis=0)]
        pts = pts - pts.mean(axis=0, keepdims=True)
        cov = pts.T @ pts
        eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
        nrm = eigvecs[:, int(np.argmin(eigvals))].astype(np.float32)
        normals[i] = nrm
    center = x_src.mean(axis=0, keepdims=True)
    flip = ((x_src - center) * normals).sum(axis=1) < 0
    normals[flip] *= -1.0
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-8)
    return normals.astype(np.float32)


def _build_anchor_patches(x_src: np.ndarray, target_patches: int) -> tuple[np.ndarray, np.ndarray]:
    """Assign each source-shell node to a fixed source-space patch."""
    n = x_src.shape[0]
    k = min(max(int(target_patches), 1), n)
    if k <= 1:
        return np.zeros((n,), dtype=np.int32), np.zeros((1,), dtype=np.int32)

    anchors = np.zeros((k,), dtype=np.int32)
    anchors[0] = int(np.argmax(np.linalg.norm(x_src - x_src.mean(axis=0, keepdims=True), axis=1)))
    min_dist2 = np.sum((x_src - x_src[anchors[0]]) ** 2, axis=1)
    for i in range(1, k):
        anchors[i] = int(np.argmax(min_dist2))
        cand_dist2 = np.sum((x_src - x_src[anchors[i]]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, cand_dist2)

    _, nearest = cKDTree(x_src[anchors]).query(x_src, k=1)
    return np.asarray(nearest, dtype=np.int32), anchors


def load_triangle_mesh(mesh_path: str):
    """Load a triangle mesh with minimal processing."""
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    if hasattr(mesh, 'dump'):
        mesh = mesh.dump(concatenate=True)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if verts.ndim != 2 or verts.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Invalid triangle mesh: {mesh_path}")
    return verts, faces


def _mesh_vertex_adjacency(num_verts: int, faces: np.ndarray) -> list[np.ndarray]:
    adj = [set() for _ in range(int(num_verts))]
    for tri in np.asarray(faces, dtype=np.int32):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    return [np.asarray(sorted(v), dtype=np.int32) for v in adj]


def _mesh_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(verts, dtype=np.float32)
    tri = np.asarray(faces, dtype=np.int32)
    face_n = np.cross(
        verts[tri[:, 1]] - verts[tri[:, 0]],
        verts[tri[:, 2]] - verts[tri[:, 0]],
    ).astype(np.float32)
    for c in range(3):
        np.add.at(normals, tri[:, c], face_n)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-8)
    return normals.astype(np.float32)


def _anchor_hop_matrix(anchor_verts: np.ndarray, v_adj: list[np.ndarray]) -> np.ndarray:
    """Pairwise hop distances between patch anchor vertices."""
    anchor_verts = np.asarray(anchor_verts, dtype=np.int32).reshape(-1)
    k = int(anchor_verts.shape[0])
    inf = np.iinfo(np.int32).max // 4
    hops = np.full((k, k), inf, dtype=np.int32)
    if k == 0:
        return hops

    v_to_anchor = {}
    for i, v in enumerate(anchor_verts.tolist()):
        v_to_anchor.setdefault(int(v), []).append(i)

    for src_idx, src_v in enumerate(anchor_verts.tolist()):
        dist = np.full((len(v_adj),), -1, dtype=np.int32)
        queue = [int(src_v)]
        dist[int(src_v)] = 0
        head = 0
        unresolved = set(range(k))
        while head < len(queue) and unresolved:
            v = queue[head]
            head += 1
            dv = int(dist[v])
            for aidx in v_to_anchor.get(v, []):
                hops[src_idx, aidx] = dv
                if aidx in unresolved:
                    unresolved.remove(aidx)
            for nb in v_adj[v]:
                nb = int(nb)
                if dist[nb] < 0:
                    dist[nb] = dv + 1
                    queue.append(nb)
        hops[src_idx, src_idx] = 0
    return hops


def _build_far_partners_from_patches(
    x_src: np.ndarray,
    patch_ids: np.ndarray,
    patch_anchor_idx: np.ndarray,
    particle_to_vert: np.ndarray,
    mesh_faces: np.ndarray,
    partner_k: int,
    patch_min_hops: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sparse nonlocal partner sets:
    particles pair with source-geodesically far patches, choosing the closest
    source-space candidates from those far patches.
    """
    n = int(x_src.shape[0])
    partner_k = max(int(partner_k), 0)
    if n == 0 or partner_k <= 0:
        return np.full((n, 0), -1, dtype=np.int32), np.zeros((n, 0), dtype=np.float32)

    patch_ids = np.asarray(patch_ids, dtype=np.int32).reshape(-1)
    patch_anchor_idx = np.asarray(patch_anchor_idx, dtype=np.int32).reshape(-1)
    particle_to_vert = np.asarray(particle_to_vert, dtype=np.int32).reshape(-1)
    num_patches = int(patch_anchor_idx.shape[0])
    if num_patches <= 1:
        return np.full((n, partner_k), -1, dtype=np.int32), np.zeros((n, partner_k), dtype=np.float32)

    mesh_faces = np.asarray(mesh_faces, dtype=np.int32)
    v_adj = _mesh_vertex_adjacency(int(mesh_faces.max()) + 1, mesh_faces)
    anchor_verts = particle_to_vert[np.clip(patch_anchor_idx, 0, max(n - 1, 0))]
    patch_hops = _anchor_hop_matrix(anchor_verts, v_adj)

    patch_members = [np.flatnonzero(patch_ids == pid).astype(np.int32) for pid in range(num_patches)]
    partner_idx = np.full((n, partner_k), -1, dtype=np.int32)
    partner_src_dist = np.zeros((n, partner_k), dtype=np.float32)

    for pid in range(num_patches):
        src_idx = patch_members[pid]
        if src_idx.size == 0:
            continue

        far_patch_mask = patch_hops[pid] >= int(patch_min_hops)
        far_patch_mask[pid] = False
        far_patches = np.flatnonzero(far_patch_mask)
        if far_patches.size == 0:
            continue

        cand_idx = np.concatenate([patch_members[j] for j in far_patches if patch_members[j].size > 0], axis=0)
        if cand_idx.size == 0:
            continue

        tree = cKDTree(x_src[cand_idx])
        k_eff = min(partner_k, cand_idx.size)
        dist, nn = tree.query(x_src[src_idx], k=k_eff)
        dist = np.asarray(dist, dtype=np.float32)
        nn = np.asarray(nn, dtype=np.int32)
        if dist.ndim == 1:
            dist = dist[:, None]
            nn = nn[:, None]
        partner_idx[src_idx, :k_eff] = cand_idx[nn]
        partner_src_dist[src_idx, :k_eff] = dist

    return partner_idx, partner_src_dist


def _collect_particle_graph_from_mesh(
    x_src: np.ndarray,
    k: int,
    sigma_scale: float,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    max_hops: int,
    hop_weight: float,
):
    """Build particle neighbors from source-mesh connectivity instead of Euclidean kNN."""
    n = x_src.shape[0]
    nv = int(mesh_vertices.shape[0])
    v_adj = _mesh_vertex_adjacency(nv, mesh_faces)
    v_normals = _mesh_vertex_normals(mesh_vertices, mesh_faces)
    v_tree = cKDTree(mesh_vertices)
    _, particle_to_vert = v_tree.query(x_src, k=1)
    particle_to_vert = np.asarray(particle_to_vert, dtype=np.int32).reshape(-1)

    particles_by_vert = [[] for _ in range(nv)]
    for pid, vid in enumerate(particle_to_vert.tolist()):
        particles_by_vert[int(vid)].append(int(pid))

    euclid_tree = cKDTree(x_src)
    k_eff = max(int(k), 1)
    nbr = np.zeros((n, k_eff), dtype=np.int32)
    dist = np.zeros((n, k_eff), dtype=np.float32)
    hop_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def vertex_particle_candidates(v0: int):
        if v0 in hop_cache:
            return hop_cache[v0]
        visited = {int(v0)}
        frontier = [int(v0)]
        cand_particles: list[int] = []
        cand_hops: list[int] = []
        hop = 0
        while frontier and hop <= max(int(max_hops), 0):
            next_frontier = []
            for vid in frontier:
                plist = particles_by_vert[vid]
                if plist:
                    cand_particles.extend(plist)
                    cand_hops.extend([hop] * len(plist))
                for nb in v_adj[vid]:
                    nb = int(nb)
                    if nb not in visited:
                        visited.add(nb)
                        next_frontier.append(nb)
            if len(cand_particles) >= max(4 * k_eff, k_eff + 1):
                break
            frontier = next_frontier
            hop += 1
        hop_cache[v0] = (
            np.asarray(cand_particles, dtype=np.int32),
            np.asarray(cand_hops, dtype=np.int32),
        )
        return hop_cache[v0]

    # Fallback Euclidean neighbors only if topology candidates are too sparse.
    k_fallback = min(k_eff + 1, n)
    fallback_dist, fallback_idx = euclid_tree.query(x_src, k=k_fallback)
    fallback_dist = np.asarray(fallback_dist, dtype=np.float32)
    fallback_idx = np.asarray(fallback_idx, dtype=np.int32)
    if fallback_idx.ndim == 1:
        fallback_idx = fallback_idx[:, None]
        fallback_dist = fallback_dist[:, None]

    valid_fallback = fallback_dist[:, 1:] > 1e-8 if fallback_idx.shape[1] > 1 else np.zeros((n, 0), dtype=bool)
    src_spacing = float(np.median(fallback_dist[:, 1:][valid_fallback])) if np.any(valid_fallback) else 1.0
    sigma = max(src_spacing * float(sigma_scale), 1e-6)

    for i in range(n):
        vid = int(particle_to_vert[i])
        cand, hops = vertex_particle_candidates(vid)
        if cand.size > 0:
            keep = cand != i
            cand = cand[keep]
            hops = hops[keep]
        if cand.size < k_eff:
            fb_idx = fallback_idx[i, 1:] if fallback_idx.shape[1] > 1 else np.zeros((0,), dtype=np.int32)
            fb_dist = fallback_dist[i, 1:] if fallback_dist.shape[1] > 1 else np.zeros((0,), dtype=np.float32)
            seen = set(cand.tolist())
            for pid, pd in zip(fb_idx.tolist(), fb_dist.tolist()):
                if pid == i or pid in seen:
                    continue
                cand = np.concatenate([cand, np.asarray([pid], dtype=np.int32)], axis=0)
                hops = np.concatenate([hops, np.asarray([max_hops + 1], dtype=np.int32)], axis=0)
                seen.add(pid)
                if cand.size >= k_eff:
                    break
        if cand.size == 0:
            nbr[i] = i
            continue
        d = np.linalg.norm(x_src[cand] - x_src[i], axis=1).astype(np.float32)
        score = d / max(src_spacing, 1e-6) + float(hop_weight) * hops.astype(np.float32)
        order = np.argsort(score)[:k_eff]
        sel = cand[order]
        sel_d = d[order]
        if sel.shape[0] < k_eff:
            pad_idx = np.full((k_eff - sel.shape[0],), int(sel[-1]), dtype=np.int32)
            pad_d = np.full((k_eff - sel.shape[0],), float(sel_d[-1]), dtype=np.float32)
            sel = np.concatenate([sel, pad_idx], axis=0)
            sel_d = np.concatenate([sel_d, pad_d], axis=0)
        nbr[i] = sel
        dist[i] = sel_d

    weights = np.exp(-(dist ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    normals = v_normals[particle_to_vert]
    return nbr, weights, normals.astype(np.float32), src_spacing, particle_to_vert


def reconstruct_surface_from_particles(
    x: np.ndarray,
    resolution: int = 64,
    padding: float = 2.0,
    sigma: float = 1.5,
    level_ratio: float = 0.02,
):
    """
    Reconstruct a proxy surface from a particle cloud on a regular voxel grid.

    Returns:
        dict with keys:
          - verts: (V, 3) reconstructed surface vertices or None
          - faces: (F, 3) faces or None
          - origin: (3,)
          - spacing: float
          - level: float
          - density_max: float
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 3 or x.shape[0] == 0:
        raise ValueError("x must be (N,3) and non-empty")

    mins = x.min(axis=0) - padding
    maxs = x.max(axis=0) + padding
    spacing = float((maxs - mins).max() / max(int(resolution), 2))
    spacing = max(spacing, 1e-6)

    idx = np.clip(((x - mins) / spacing).astype(np.int32), 0, resolution - 1)
    density = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    for i in range(x.shape[0]):
        ix, iy, iz = idx[i]
        density[ix, iy, iz] += 1.0

    if sigma > 0:
        density = gaussian_filter(density, sigma=float(sigma))

    density_max = float(density.max())
    level = float(density_max * level_ratio)
    if density_max <= 1e-8 or level <= 0.0:
        return {
            'verts': None,
            'faces': None,
            'origin': mins.astype(np.float32),
            'spacing': spacing,
            'level': level,
            'density_max': density_max,
        }

    try:
        verts, faces, _, _ = marching_cubes(density, level=level)
        verts = verts * spacing + mins
    except Exception:
        verts, faces = None, None

    return {
        'verts': None if verts is None else verts.astype(np.float32),
        'faces': None if faces is None else faces.astype(np.int32),
        'origin': mins.astype(np.float32),
        'spacing': spacing,
        'level': level,
        'density_max': density_max,
    }


def build_fixed_surface_mask(
    x: np.ndarray,
    resolution: int = 64,
    padding: float = 2.0,
    sigma: float = 1.5,
    level_ratio: float = 0.02,
    threshold_mult: float = 1.75,
    min_surface_frac: float = 0.08,
    max_surface_frac: float = 0.40,
):
    """
    Build a persistent surface mask from a volumetric particle cloud.

    Steps:
      1. reconstruct a surface proxy on a 3D grid
      2. assign each particle a distance to the reconstructed surface
      3. keep particles close to that surface
    """
    x = np.asarray(x, dtype=np.float32)
    recon = reconstruct_surface_from_particles(
        x,
        resolution=resolution,
        padding=padding,
        sigma=sigma,
        level_ratio=level_ratio,
    )
    verts = recon['verts']
    spacing = float(recon['spacing'])
    N = x.shape[0]

    if verts is None or len(verts) == 0:
        mask = np.ones((N,), dtype=bool)
        recon.update({
            'surface_mask': mask,
            'surface_distance_threshold': 0.0,
            'surface_fraction': 1.0,
            'surface_distance_mean': 0.0,
        })
        return recon

    dist, _ = cKDTree(verts).query(x, k=1)
    dist = np.asarray(dist, dtype=np.float32)

    base_thr = float(threshold_mult * spacing)
    thr = base_thr
    mask = dist <= thr
    frac = float(mask.mean())

    if frac < min_surface_frac:
        thr = max(thr, float(np.quantile(dist, min_surface_frac)))
    if frac > max_surface_frac:
        thr = min(thr, float(np.quantile(dist, max_surface_frac)))

    mask = dist <= thr
    frac = float(mask.mean())

    recon.update({
        'surface_mask': mask.astype(bool),
        'surface_distance_threshold': float(thr),
        'surface_fraction': frac,
        'surface_distance_mean': float(dist.mean()),
    })
    return recon


def build_frozen_surface_graph(
    x: np.ndarray,
    particle_mask=None,
    k: int = 16,
    sigma_scale: float = 1.0,
    num_patches: int = 96,
    mesh_vertices: np.ndarray | None = None,
    mesh_faces: np.ndarray | None = None,
    max_hops: int = 4,
    hop_weight: float = 0.75,
    separation_enabled: bool = False,
    separation_partner_k: int = 4,
    separation_patch_min_hops: int = 6,
):
    """
    Build a fixed shell graph from the source/initial particle layout.

    The graph is defined once in source space and reused later so smoothing does
    not reconnect features based on changing Euclidean proximity.
    """
    x = np.asarray(x, dtype=np.float32)
    if particle_mask is None:
        idx = np.arange(x.shape[0], dtype=np.int32)
    else:
        mask = np.asarray(particle_mask, dtype=bool).reshape(-1)
        idx = np.flatnonzero(mask).astype(np.int32)

    if idx.size == 0:
        return None

    x_src = np.ascontiguousarray(x[idx])
    n = x_src.shape[0]
    if n == 1:
        return {
            'indices': idx,
            'source_positions': x_src,
            'neighbors': np.zeros((1, 1), dtype=np.int32),
            'weights': np.ones((1, 1), dtype=np.float32),
            'source_spacing': 0.0,
        }

    if mesh_vertices is not None and mesh_faces is not None and len(mesh_faces) > 0:
        nbr, weights, normals, src_spacing, particle_to_vert = _collect_particle_graph_from_mesh(
            x_src,
            k=max(int(k), 1),
            sigma_scale=float(sigma_scale),
            mesh_vertices=np.asarray(mesh_vertices, dtype=np.float32),
            mesh_faces=np.asarray(mesh_faces, dtype=np.int32),
            max_hops=int(max_hops),
            hop_weight=float(hop_weight),
        )
        graph_mode = 'mesh_geodesic'
    else:
        k_eff = min(max(int(k), 1) + 1, n)
        dist, nbr = cKDTree(x_src).query(x_src, k=k_eff)
        nbr = nbr[:, 1:] if nbr.ndim == 2 and nbr.shape[1] > 1 else nbr[:, :1]
        dist = dist[:, 1:] if dist.ndim == 2 and dist.shape[1] > 1 else dist[:, :1]
        dist = np.asarray(dist, dtype=np.float32)
        nbr = np.asarray(nbr, dtype=np.int32)

        valid = dist > 1e-8
        src_spacing = float(np.median(dist[valid])) if np.any(valid) else 1.0
        sigma = max(src_spacing * float(sigma_scale), 1e-6)
        weights = np.exp(-(dist ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
        normals = _estimate_graph_normals(x_src, nbr)
        particle_to_vert = None
        graph_mode = 'euclidean_knn'
    patch_ids, patch_anchor_idx = _build_anchor_patches(x_src, num_patches)
    sep_partner_idx = None
    sep_partner_src_dist = None
    if (
        bool(separation_enabled)
        and particle_to_vert is not None
        and mesh_faces is not None
        and len(mesh_faces) > 0
    ):
        sep_partner_idx, sep_partner_src_dist = _build_far_partners_from_patches(
            x_src,
            patch_ids,
            patch_anchor_idx,
            particle_to_vert,
            mesh_faces,
            partner_k=int(separation_partner_k),
            patch_min_hops=int(separation_patch_min_hops),
        )

    return {
        'indices': idx,
        'source_positions': x_src,
        'neighbors': nbr,
        'weights': weights,
        'source_normals': normals,
        'source_spacing': src_spacing,
        'patch_ids': patch_ids,
        'patch_anchor_idx': patch_anchor_idx,
        'num_patches': int(np.max(patch_ids) + 1) if patch_ids.size else 0,
        'particle_to_vert': particle_to_vert,
        'graph_mode': graph_mode,
        'sep_partner_idx': sep_partner_idx,
        'sep_partner_src_dist': sep_partner_src_dist,
    }


def apply_displacement_surface_proxy(
    x: np.ndarray,
    surface_graph,
    strength: float = 0.75,
    diffusion_iters: int = 4,
    bilateral: bool = False,
    bilateral_sigma_scale: float = 1.5,
    normal_sigma: float = 0.35,
    separation_strength: float = 0.0,
    separation_margin_scale: float = 0.65,
    separation_min_margin_scale: float = 2.0,
    separation_max_step_scale: float = 0.35,
):
    """
    Smooth a shell proxy by filtering displacement on a frozen shell graph.

    This keeps the render branch from observing the raw shell clumps directly
    while avoiding absolute-position fairing that can collapse thin structures.
    """
    if surface_graph is None:
        return np.asarray(x, dtype=np.float32), {
            'proxy_applied': 0,
            'proxy_strength': 0.0,
            'proxy_diffusion_iters': 0,
            'proxy_separation_strength': 0.0,
            'proxy_separation_active_count': 0,
            'proxy_separation_active_frac': 0.0,
            'proxy_shift_mean': 0.0,
            'proxy_shift_max': 0.0,
        }

    x_cur = np.asarray(x, dtype=np.float32)
    x_src = np.asarray(surface_graph['source_positions'], dtype=np.float32)
    nbr = np.asarray(surface_graph['neighbors'], dtype=np.int32)
    w_base = np.asarray(surface_graph['weights'], dtype=np.float32)
    n = x_src.shape[0]
    if x_cur.shape[0] != n or n == 0:
        return x_cur, {
            'proxy_applied': 0,
            'proxy_strength': 0.0,
            'proxy_diffusion_iters': 0,
            'proxy_separation_strength': 0.0,
            'proxy_separation_active_count': 0,
            'proxy_separation_active_frac': 0.0,
            'proxy_shift_mean': 0.0,
            'proxy_shift_max': 0.0,
        }

    strength = float(np.clip(strength, 0.0, 1.0))
    diffusion_iters = max(int(diffusion_iters), 0)
    separation_strength = max(float(separation_strength), 0.0)
    if (strength <= 0.0 or diffusion_iters <= 0) and separation_strength <= 0.0:
        return x_cur, {
            'proxy_applied': 0,
            'proxy_strength': strength,
            'proxy_diffusion_iters': diffusion_iters,
            'proxy_separation_strength': separation_strength,
            'proxy_separation_active_count': 0,
            'proxy_separation_active_frac': 0.0,
            'proxy_shift_mean': 0.0,
            'proxy_shift_max': 0.0,
        }

    u = (x_cur - x_src).astype(np.float32)
    src_spacing = max(float(surface_graph.get('source_spacing', 1.0)), 1e-6)
    src_normals = np.asarray(surface_graph.get('source_normals'), dtype=np.float32)

    if diffusion_iters > 0 and strength > 0.0:
        for _ in range(diffusion_iters):
            w = w_base
            if bilateral:
                du = u[nbr] - u[:, None, :]
                sigma_u = max(src_spacing * float(bilateral_sigma_scale), 1e-6)
                w = w * np.exp(-(du ** 2).sum(axis=2) / (2.0 * sigma_u * sigma_u)).astype(np.float32)
                if src_normals.shape[0] == n:
                    cos_sim = np.clip((src_normals[:, None, :] * src_normals[nbr]).sum(axis=2), -1.0, 1.0)
                    sigma_n = max(float(normal_sigma), 1e-3)
                    w = w * np.exp(-((1.0 - cos_sim) ** 2) / (2.0 * sigma_n * sigma_n)).astype(np.float32)
                w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-8)
            u_avg = (w[:, :, None] * u[nbr]).sum(axis=1).astype(np.float32)
            u = ((1.0 - strength) * u + strength * u_avg).astype(np.float32)

    z = (x_src + u).astype(np.float32)
    sep_active_count = 0
    sep_partner_idx = surface_graph.get('sep_partner_idx', None)
    sep_partner_src_dist = surface_graph.get('sep_partner_src_dist', None)
    if separation_strength > 0.0 and sep_partner_idx is not None and sep_partner_src_dist is not None:
        sep_partner_idx = np.asarray(sep_partner_idx, dtype=np.int32)
        sep_partner_src_dist = np.asarray(sep_partner_src_dist, dtype=np.float32)
        if sep_partner_idx.shape[0] == n and sep_partner_src_dist.shape == sep_partner_idx.shape:
            valid = sep_partner_idx >= 0
            if np.any(valid):
                partner_safe = np.clip(sep_partner_idx, 0, max(n - 1, 0))
                z_partner = z[partner_safe]
                dz = z[:, None, :] - z_partner
                dist = np.linalg.norm(dz, axis=2)
                src_dir = x_src[:, None, :] - x_src[partner_safe]
                src_dir_norm = np.linalg.norm(src_dir, axis=2, keepdims=True)
                src_dir = src_dir / np.maximum(src_dir_norm, 1e-8)
                cur_dir = dz / np.maximum(dist[:, :, None], 1e-8)
                cur_dir = np.where(dist[:, :, None] > 1e-6, cur_dir, src_dir)

                margin = np.maximum(
                    sep_partner_src_dist * float(separation_margin_scale),
                    float(separation_min_margin_scale) * src_spacing,
                )
                active = valid & (dist < margin)
                sep_active_count = int(active.sum())
                if sep_active_count > 0:
                    push = (float(separation_strength) * (margin - dist))[:, :, None] * cur_dir
                    push *= active[:, :, None].astype(np.float32)
                    delta = np.zeros_like(z, dtype=np.float32)
                    ii, jj = np.nonzero(active)
                    pair_push = 0.5 * push[ii, jj]
                    np.add.at(delta, ii, pair_push)
                    np.add.at(delta, partner_safe[ii, jj], -pair_push)
                    max_step = float(separation_max_step_scale) * src_spacing
                    if max_step > 0.0:
                        delta_norm = np.linalg.norm(delta, axis=1, keepdims=True)
                        delta *= np.minimum(1.0, max_step / np.maximum(delta_norm, 1e-8))
                    z = (z + delta).astype(np.float32)
    proxy_shift = np.linalg.norm(z - x_cur, axis=1)
    return z, {
        'proxy_applied': 1,
        'proxy_strength': float(strength),
        'proxy_diffusion_iters': int(diffusion_iters),
        'proxy_separation_strength': float(separation_strength),
        'proxy_separation_active_count': int(sep_active_count),
        'proxy_separation_active_frac': float(sep_active_count / max(int(np.sum(sep_partner_idx >= 0)) if sep_partner_idx is not None else 1, 1)),
        'proxy_shift_mean': float(proxy_shift.mean()),
        'proxy_shift_max': float(proxy_shift.max()),
    }


__all__ = [
    'reconstruct_surface_from_particles',
    'build_fixed_surface_mask',
    'build_frozen_surface_graph',
    'apply_displacement_surface_proxy',
    'load_triangle_mesh',
]
