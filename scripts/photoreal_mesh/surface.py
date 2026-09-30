"""The per-frame surface of the photoreal renderer (mesh_of): the density grid, then marching cubes at
the level ('mc') or a surface reconstructed from the oriented outer particle layer ('poisson', 'imls',
'surfel'), the component rules (mass, volume, cavities), the optional bilateral / Taubin smoothing and
the filament bridges; and the outer-layer surfel memory (--surfel_memory) that carries the previous
frames' surfels with the material."""
import numpy as np
import open3d as o3d
import torch
from skimage import measure
from scipy.spatial import cKDTree

from physmorph.render.surface_recon import (surface_particles, surface_particles_grad, oriented_layer,
                                            poisson_mesh, exterior_surfels, imls_grid, surfel_mesh,
                                            bilateral_normal_smooth)
from photoreal_mesh.density import density, density_aniso, density_pca, frame_F
from photoreal_mesh.topology import filament_bridges, levelset_particle_labels, mesh_particle_labels
from photoreal_mesh.tracking import advect_vertices


def _layer_raw(ctx, i):
    """The outer-layer surfels of archived frame i (density -> layer -> exterior test -> optional pull),
    cached; the surfel memory reads the previous frames through this."""
    a, dev, frames_np, spacing = ctx.a, ctx.dev, ctx.frames_np, ctx.spacing
    ctr, half, vox, layer_thr, layer_gthr = ctx.ctr, ctx.half, ctx.vox, ctx.layer_thr, ctx.layer_gthr
    with ctx.layer_lock:
        if i in ctx.layer_cache:
            return ctx.layer_cache[i]
    x = torch.as_tensor(np.asarray(frames_np[i], np.float32), device=dev)
    rho_t = density_aniso(ctx, x, frame_F(ctx, i, x)) if a.kernel == "aniso" else (density_pca(ctx, x) if a.kernel == "pca" else density(ctx, x))
    if a.layer == "grad":
        pts, nrm = surface_particles_grad(x, rho_t, ctr, half, vox, layer_gthr)
    else:
        pts, nrm = surface_particles(x, rho_t, ctr, half, vox, layer_thr)
    pts, nrm, _ = exterior_surfels(pts, nrm, x.detach().cpu().numpy(), spacing)
    if a.pull > 0:
        pts, nrm = oriented_layer(pts, nrm, spacing, pull_iters=a.pull)
    with ctx.layer_lock:
        ctx.layer_cache[i] = (pts, nrm)
    return pts, nrm


def surfel_memory_union(ctx, fi, pts, nrm, x_np):
    """The current frame's surfels plus the previous K-1 video frames' surfels carried to this frame with the
    material, kept only where a particle is still within one spacing, downsampled at half a spacing (normals
    averaged): a box filter of one control window over the surface, so a neck the reconstruction loses for a
    few frames stays, the re-fit jitter averages out, and the mesh is still fresh every frame."""
    a, frames_np, spacing = ctx.a, ctx.frames_np, ctx.spacing
    with ctx.layer_lock:
        ctx.layer_cache[fi] = (pts, nrm)
    x_cur = np.asarray(x_np, np.float64)
    kd_cur = cKDTree(x_cur)
    ups, unr = [np.asarray(pts, np.float64)], [np.asarray(nrm, np.float64)]
    for m in range(1, a.surfel_memory):
        j = fi - m * a.stride
        if j < 0:
            break
        pj, nj = _layer_raw(ctx, j)
        if len(pj) == 0:
            continue
        xj = np.asarray(frames_np[j], np.float64)
        pj = np.asarray(pj, np.float64); nj = np.asarray(nj, np.float64)
        pa = advect_vertices(pj, xj, x_cur, a.track_k, spacing)
        tip = advect_vertices(pj + spacing * nj, xj, x_cur, a.track_k, spacing)
        na = tip - pa
        na /= np.maximum(np.linalg.norm(na, axis=1, keepdims=True), 1e-9)
        d, _ = kd_cur.query(pa, k=1, workers=-1)
        keep = d <= spacing                                   # the material is still there
        ups.append(pa[keep]); unr.append(na[keep])
    P = np.concatenate(ups); Nn = np.concatenate(unr)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.normals = o3d.utility.Vector3dVector(Nn)
    pc = pc.voxel_down_sample(0.5 * spacing)
    out_p = np.asarray(pc.points, np.float32); out_n = np.asarray(pc.normals, np.float32)
    out_n /= np.maximum(np.linalg.norm(out_n, axis=1, keepdims=True), 1e-9)
    return out_p, out_n, len(P)


def mesh_of(ctx, x, fi=None):
    """Isosurface mesh (Open3D) of the cloud x; returns (mesh, n_components_raw, n_dropped, n_bridged, n_cavities)."""
    a, G, ctr, half, vox, origin, spacing = ctx.a, ctx.G, ctx.ctr, ctx.half, ctx.vox, ctx.origin, ctx.spacing
    iso, layer_thr, layer_gthr, kernel_sigma_sp = ctx.iso, ctx.layer_thr, ctx.layer_gthr, ctx.kernel_sigma_sp
    ppc, min_vol = ctx.ppc, ctx.min_vol
    if a.kernel == "aniso" and fi is not None:
        rho_t = density_aniso(ctx, x, frame_F(ctx, fi, x))           # [z,y,x]
    elif a.kernel == "pca":
        rho_t = density_pca(ctx, x)
    else:
        rho_t = density(ctx, x)
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
        if a.surfel_memory > 0 and fi is not None and fi >= 0:
            n_own = len(pts)
            pts, nrm, n_union = surfel_memory_union(ctx, fi, pts, nrm, x.detach().cpu().numpy())
            if fi == a.stride * (a.surfel_memory - 1):
                print(f"[photoreal] surfel memory: {a.surfel_memory} frames; frame {fi}: {n_own} own surfels, "
                      f"{n_union} in the union, {len(pts)} after the half-spacing downsample", flush=True)
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
                ctx.fallback_frames.append(fi)
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
        plab, drawn, body_lab = (levelset_particle_labels(ctx, x_np64, rho) if a.surface == "mc"
                                 else mesh_particle_labels(ctx, m, x_np64))
        fil, n_bridge = filament_bridges(ctx, x_np64, plab, drawn, body_lab)
        if fil is not None:
            m += fil
    return m, n_comp, n_drop, n_bridge, n_cav
