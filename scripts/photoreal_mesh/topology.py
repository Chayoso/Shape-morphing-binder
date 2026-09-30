"""Drawn topology follows the particles, not the threshold (docs/method.md 10.10): per-particle labels
of the enclosing level-set or drawn-mesh component, and the filament bridges drawn one particle
spacing thick along the shortest particle chain from the body to a piece the surface separated."""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


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


def levelset_particle_labels(ctx, x_np, rho):
    """Per-particle label of the level-set component enclosing it (0 = none), the drawn mask over
    labels (the volume rule on voxels) and the body label — the marching-cubes surface's notion
    of 'enclosed'."""
    G, ctr, half, vox, iso, min_vol = ctx.G, ctx.ctr, ctx.half, ctx.vox, ctx.iso, ctx.min_vol
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


def mesh_particle_labels(ctx, m, x_np):
    """Per-particle label of the DRAWN mesh component enclosing it (0 = none): occupancy of the
    closed mesh (Open3D ray casting) and the component of the nearest triangle. The Poisson
    surface is the true boundary, tighter than the blurred level set, so the level set can join
    two pieces a one-spacing neck separates on the drawn surface (cow video: drawn pieces > 1 in
    141 frames while the voxel labels saw one body and drew no bridge). The bridge rule must
    read the surface that is drawn."""
    spacing = ctx.spacing
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


def filament_bridges(ctx, x_np, plab, drawn, body, drawn_labels_needed=2):
    """Particles the drawn surface does not enclose but which link the body to another enclosed
    component are drawn as a filament one particle spacing thick (docs/method.md 10.10): a
    feature thinner than a two-particle bundle (the cow's teat: a 72-particle bulb on the
    target tied to the udder by a single-particle thread) is connected material at the
    particle scale, and the rendered topology must follow the particles, not the threshold.
    plab/drawn/body from levelset_particle_labels (marching cubes) or mesh_particle_labels
    (the Poisson surface). Returns (filament mesh or None, number of enclosed components
    bridged to the body)."""
    spacing, cell_wu = ctx.spacing, ctx.cell_wu
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
