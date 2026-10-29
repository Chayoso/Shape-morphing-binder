"""hull_carve.py OUT_DIR N RES MESH [MESH ...] -- D156: the visual hull of a target mesh from the render's 18 training
silhouettes, at the pictures' resolution, written as a mesh the pipeline can take as its target.

Run from a repo copy with the frozen sampler (repo_d156; CPU only, open3d / skimage). For each mesh:
- the frame: the mesh's own run frame on the frozen path (prepare.prepare at N, seed 97: the source sample, isosphere
  seed 97, and the mesh normalised by load_normalized at seed 98, matched to the source's volume); the orthographic
  box (the pictures' half-extent) 1.25 x the mesh sample's largest |x| (target.build_target); the 18 views
  make_views(6, (0, +-0.5)). (The D156 runs put the true mesh in the HULL's frame, centred on the hull's sample: the
  hull is a solid in the mesh's coordinates, so only the sub-pixel placement of the pixel lattice and the box's
  size, within about 1 %, differ from the run's pictures);
- the silhouettes: the exact mesh ray-cast (open3d) at the pipeline's own pixel nodes, x = -box + i (2 box / RES),
  i = 0 .. RES - 1 along the view's right and up (losses/silhouette.py's lattice: a point at a node lands on that pixel
  alone), a pixel in the silhouette when its ray hits the mesh (also through a non-watertight mesh's holes: any face);
- the carve: each view's silhouette as a signed distance in pixels (+ inside, the zero half-way between an inside and
  an outside node), read bilinearly at a point's projection; the hull's field is the minimum over the 18 views (a
  point is in the hull iff its projection lies in every silhouette); sampled on a grid at the pictures' pitch
  (2 box / RES) over the box plus one node of margin, marching cubes at zero (skimage), outward faces;
- the output: OUT_DIR/MESH_hull18_rRES.obj in the mesh's oriented (y-up) coordinates, so the pipeline's
  orientation table reads it as 'id' (checked): `--tgt OUT_DIR/MESH_hull18_rRES.obj --render_tgt assets/MESH.obj` is
  the hull prepared as a target is today, with the true mesh's pictures in its frame (prepare.prepare); one JSON line per mesh: the box, pixel, volumes (the pipeline's 110^3 fill of both, in world
  units), the hull's excess over the mesh, the share of the mesh's volume outside the hull (the carve's
  discretisation: the mesh must lie inside), the excess region's depth in pitches and the centroid offset of the hull
  from the mesh (the body's centre of mass stays at the hull's, the controls carry no net force).
No constant of the carve is new: the views, the box, the resolution and the pixel lattice are the pipeline's own."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
import open3d as o3d                                                                   # noqa: E402
import trimesh                                                                         # noqa: E402
from scipy.ndimage import distance_transform_edt, map_coordinates                     # noqa: E402
from scipy.spatial import cKDTree                                                      # noqa: E402
from skimage.measure import marching_cubes                                            # noqa: E402

from physmorph.pipeline.config import PipelineConfig                                   # noqa: E402
from physmorph.pipeline.render_loss import make_views                                  # noqa: E402
from physmorph.sampling.mesh import _oriented, filled_volume, load_normalized          # noqa: E402
from physmorph.sampling.orientation import orient_name                                 # noqa: E402

NT = int(os.environ.get("D156_THREADS", "6"))
out_dir, N, RES, meshes = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
out_dir.mkdir(parents=True, exist_ok=True)
cfg = PipelineConfig()
views = make_views(cfg.render_views, cfg.render_elevs)
SEED = 97


def basis(theta, phi):
    right = np.array([np.cos(theta), 0.0, -np.sin(theta)])
    up = np.array([-np.sin(phi) * np.sin(theta), np.cos(phi), -np.sin(phi) * np.cos(theta)])
    return right, up, np.cross(right, up)                    # d: toward the camera (render_loss's l_dir)


def silhouette(scene, box, res, theta, phi):
    """(res, res) bool: [i, j] = the ray through the pixel node (-box + i px along right, -box + j px along up) hits."""
    px = 2.0 * box / res
    c = -box + np.arange(res) * px
    cu, cv = np.meshgrid(c, c, indexing="ij")
    r, u, d = basis(theta, phi)
    o = cu[..., None] * r + cv[..., None] * u + 3.0 * box * d
    rays = np.concatenate([o.reshape(-1, 3), np.broadcast_to(-d, (res * res, 3))], 1).astype(np.float32)
    t = scene.cast_rays(o3d.core.Tensor(rays), nthreads=NT)["t_hit"].numpy()
    return np.isfinite(t).reshape(res, res)


def inside_sd(mask):
    """Signed distance of each pixel node to the silhouette's outline, in pixels: > 0 inside, the zero half-way
    between an inside node and an outside one."""
    return np.where(mask, distance_transform_edt(mask) - 0.5, 0.5 - distance_transform_edt(~mask))


def hull_field(pts, sds, box, res):
    """min over views of the bilinear silhouette distance at each point's projection (pixels); -inf off an image."""
    px = 2.0 * box / res
    f = np.full(len(pts), np.inf)
    for (th, ph), sd in zip(views, sds):
        r, u, _ = basis(th, ph)
        a, b = (pts @ r + box) / px, (pts @ u + box) / px          # node coordinates (losses/silhouette.py's rel)
        v = map_coordinates(sd, [a, b], order=1, mode="constant", cval=-1e9)
        f = np.minimum(f, v)
    return f


for mesh in meshes:
    t0 = time.time()
    path = f"assets/{mesh}.obj"
    _, v_src = load_normalized("assets/isosphere.obj", N, SEED, return_volume=True, sample="stratified")
    frame = {}
    tgt, v_tgt = load_normalized(path, N, SEED + 1, match_volume=v_src, sample="stratified", return_volume=True,
                                 frame=frame)
    box = float(np.abs(tgt).max()) * 1.25
    px = 2.0 * box / RES
    pitch = 0.708 * float(np.median(cKDTree(tgt).query(tgt, 9)[0][:, 8]))   # the field's pitch (target.build_target)
    m = _oriented(path)
    Vw = (np.asarray(m.vertices, np.float64) - frame["offset"]) * frame["scale"]
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(Vw.astype(np.float32)), o3d.core.Tensor(np.asarray(m.faces, np.uint32)))
    sils = [silhouette(scene, box, RES, th, ph) for th, ph in views]
    sds = [inside_sd(s) for s in sils]
    # the carve grid: the pictures' pitch over the box, one node of margin beyond it (the hull lies inside the box)
    g = -box - px + px * np.arange(RES + 3)
    G = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    f = hull_field(G, sds, box, RES).reshape(len(g), len(g), len(g))
    f[[0, -1], :, :] = f[:, [0, -1], :] = f[:, :, [0, -1]] = -1.0          # closed: outside on the margin
    verts, faces, _, _ = marching_cubes(f, level=0.0, spacing=(px, px, px), allow_degenerate=False)
    verts = verts + g[0]
    hw = trimesh.Trimesh(verts, faces, process=True)
    hw.update_faces(hw.nondegenerate_faces())                 # zero-area faces (edges shared by four faces)
    hw.merge_vertices()
    if not hw.is_watertight:                                  # a removed degenerate face can leave a one-face hole
        trimesh.repair.fill_holes(hw)
    assert hw.is_watertight, f"{mesh}: the hull is not watertight"
    if hw.volume < 0:
        hw.invert()
    # back to the mesh's oriented coordinates: world = (mesh - offset) x scale
    hm = trimesh.Trimesh(hw.vertices / frame["scale"] + frame["offset"], hw.faces, process=False)
    out = out_dir / f"{mesh}_hull18_r{RES}.obj"
    hm.export(str(out))
    assert orient_name(str(out)) == "id", out          # the pipeline must not rotate the hull again
    # measures (world units; pitch of the run's own target sample)
    vol_mesh = filled_volume(m) * frame["scale"] ** 3
    vol_hull = filled_volume(hm) * frame["scale"] ** 3
    mesh_out = float((hull_field(tgt.astype(np.float64), sds, box, RES) < 0).mean())  # the mesh's sample off the hull
    sp_mesh, _ = trimesh.sample.sample_surface(trimesh.Trimesh(Vw, np.asarray(m.faces), process=False), 100000, seed=0)
    f_surf = hull_field(np.asarray(sp_mesh, np.float64), sds, box, RES)  # pixels, < 0 = outside the hull
    hscene = o3d.t.geometry.RaycastingScene()                  # the hull's volume: uniform points of its box inside it
    hscene.add_triangles(o3d.core.Tensor(hw.vertices.astype(np.float32)), o3d.core.Tensor(hw.faces.astype(np.uint32)))
    rng = np.random.default_rng(0)
    lo, hi = hw.bounds
    cand = (lo + (hi - lo) * rng.random((400000, 3))).astype(np.float32)
    hs = cand[hscene.compute_occupancy(o3d.core.Tensor(cand), nthreads=NT).numpy() > 0.5].astype(np.float64)
    occ = scene.compute_occupancy(o3d.core.Tensor(hs.astype(np.float32)), nthreads=NT, nsamples=5).numpy() > 0.5
    ex = hs[~occ]
    dex = scene.compute_distance(o3d.core.Tensor(ex.astype(np.float32)), nthreads=NT).numpy() / pitch if len(ex) else np.zeros(0)
    q = (lambda p: float(np.quantile(dex, p)) if len(dex) else None)
    row = dict(mesh=mesh, n=N, res=RES, seed=SEED, box=box, px=px, pitch=pitch, px_over_pitch=px / pitch,
               out=str(out), hull_faces=len(hw.faces), hull_watertight=bool(hw.is_watertight),
               vol_mesh_fill=vol_mesh, vol_hull_fill=vol_hull, hull_over_mesh=vol_hull / vol_mesh,
               hull_mesh_volume_exact=float(hw.volume), excess_share_of_hull=float(len(ex) / max(len(hs), 1)),
               mesh_sample_outside_hull=mesh_out, mesh_surface_outside_hull=float((f_surf < 0).mean()),
               mesh_surface_outside_hull_over_half_px=float((f_surf < -0.5).mean()),
               mesh_surface_hull_px_p001=float(np.quantile(f_surf, 0.001)),
               centroid_offset_pitch=float(np.linalg.norm(hs.mean(0) - tgt.astype(np.float64).mean(0)) / pitch) if len(hs) else None,
               excess_depth_pitch=dict(p50=q(.5), p90=q(.9), p99=q(.99), max=float(dex.max()) if len(dex) else None,
                                       share_over_1=float((dex > 1).mean()) if len(dex) else None,
                                       share_over_2=float((dex > 2).mean()) if len(dex) else None),
               sil_pixels=[int(s.sum()) for s in sils], seconds=time.time() - t0)
    print(json.dumps(row), flush=True)
    with open(out_dir / f"hulls_r{RES}.jsonl", "a") as fo:
        fo.write(json.dumps(row) + "\n")
