"""Visualization-only surface reconstruction render from a saved checkpoint."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree
from scipy import ndimage as ndi

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import open3d as o3d
import pyrender
import trimesh
from skimage.measure import marching_cubes

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run import setup_cameras, setup_renderer, compute_sigma0
from tools.render_checkpoint import _build_render_mask, _resolve_config
from utils.io_utils import save_image_png
from renderer import compute_shading


def _median_nn_spacing(x: np.ndarray, sample_size: int = 20000) -> float:
    n = int(x.shape[0])
    if n < 2:
        return 1e-3
    if n > sample_size:
        rng = np.random.default_rng(42)
        sel = rng.choice(n, size=sample_size, replace=False)
        xs = x[sel]
    else:
        xs = x
    dd, _ = cKDTree(xs).query(xs, k=2)
    nn = np.asarray(dd[:, 1], dtype=np.float32)
    nn = nn[np.isfinite(nn) & (nn > 1e-8)]
    if nn.size == 0:
        return 1e-3
    return float(np.median(nn))


def _reconstruct_poisson_mesh(
    x: np.ndarray,
    normal_knn: int,
    orient_knn: int,
    poisson_depth: int,
    density_quantile: float,
    voxel_scale: float,
    taubin_iters: int,
) -> tuple[trimesh.Trimesh, dict]:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 3 or x.shape[0] < 64:
        raise ValueError("Need at least 64 3D points for surface reconstruction")

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(x.astype(np.float64)))
    base_spacing = _median_nn_spacing(x)
    voxel_size = max(base_spacing * float(voxel_scale), 1e-5)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=max(int(normal_knn), 8)))
    try:
        pcd.orient_normals_consistent_tangent_plane(max(int(orient_knn), 8))
    except Exception:
        pass

    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(poisson_depth), scale=1.1, linear_fit=False
    )
    densities = np.asarray(densities, dtype=np.float32)
    if densities.size > 0 and 0.0 < density_quantile < 1.0:
        thr = float(np.quantile(densities, density_quantile))
        keep = densities >= thr
        mesh_o3d.remove_vertices_by_mask(~keep)

    bbox = pcd.get_axis_aligned_bounding_box()
    ext = np.asarray(bbox.get_extent(), dtype=np.float64)
    minb = np.asarray(bbox.min_bound, dtype=np.float64).copy()
    maxb = np.asarray(bbox.max_bound, dtype=np.float64).copy()
    ctr = 0.5 * (minb + maxb)
    half = 0.5 * (maxb - minb) * 1.05
    minb = ctr - half - 0.05 * ext
    maxb = ctr + half + 0.05 * ext
    crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=minb, max_bound=maxb)
    mesh_o3d = mesh_o3d.crop(crop_box)
    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_unreferenced_vertices()
    mesh_o3d.compute_vertex_normals()

    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh_o3d.vertices, dtype=np.float32),
        faces=np.asarray(mesh_o3d.triangles, dtype=np.int32),
        process=False,
    )
    if taubin_iters > 0 and len(mesh.vertices) > 0 and len(mesh.faces) > 0:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=int(taubin_iters))

    mesh.remove_unreferenced_vertices()
    if hasattr(mesh, "update_faces") and hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    if hasattr(mesh, "nondegenerate_faces") and hasattr(mesh, "update_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if hasattr(mesh, "fix_normals"):
        mesh.fix_normals()

    meta = {
        "input_points": int(x.shape[0]),
        "voxel_size": float(voxel_size),
        "downsampled_points": int(len(pcd.points)),
        "poisson_depth": int(poisson_depth),
        "density_quantile": float(density_quantile),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "taubin_iters": int(taubin_iters),
    }
    return mesh, meta


def _reconstruct_voxel_mesh(
    x: np.ndarray,
    resolution: int,
    padding: float,
    sigma: float,
    level_ratio: float,
    closing_iters: int,
    taubin_iters: int,
) -> tuple[trimesh.Trimesh, dict]:
    x = np.asarray(x, dtype=np.float32)
    mins = x.min(axis=0) - float(padding)
    maxs = x.max(axis=0) + float(padding)
    res = int(resolution)
    spacing = float((maxs - mins).max() / max(res, 2))
    spacing = max(spacing, 1e-6)

    idx = np.clip(((x - mins) / spacing).astype(np.int32), 0, res - 1)
    density = np.zeros((res, res, res), dtype=np.float32)
    np.add.at(density, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    if sigma > 0:
        density = ndi.gaussian_filter(density, sigma=float(sigma))

    level = float(density.max() * float(level_ratio))
    occ = density >= level
    if closing_iters > 0:
        occ = ndi.binary_closing(occ, structure=np.ones((3, 3, 3), dtype=bool), iterations=int(closing_iters))
        occ = ndi.binary_fill_holes(occ)

    verts, faces, _, _ = marching_cubes(occ.astype(np.float32), level=0.5)
    verts = verts * spacing + mins
    mesh = trimesh.Trimesh(vertices=verts.astype(np.float32), faces=faces.astype(np.int32), process=False)
    if taubin_iters > 0 and len(mesh.vertices) > 0 and len(mesh.faces) > 0:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=int(taubin_iters))
    mesh.remove_unreferenced_vertices()
    if hasattr(mesh, "update_faces") and hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    if hasattr(mesh, "nondegenerate_faces") and hasattr(mesh, "update_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if hasattr(mesh, "fix_normals"):
        mesh.fix_normals()

    meta = {
        "input_points": int(x.shape[0]),
        "resolution": int(resolution),
        "spacing": float(spacing),
        "sigma": float(sigma),
        "level_ratio": float(level_ratio),
        "closing_iters": int(closing_iters),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "taubin_iters": int(taubin_iters),
    }
    return mesh, meta


def _camera_pose_from_lookat(cam_cfg: dict) -> tuple[np.ndarray, list[float]]:
    lookat = cam_cfg.get("lookat", {})
    eye = np.array(lookat.get("eye", [20.0, -25.0, 12.5]), dtype=np.float64)
    target = np.array(lookat.get("target", [0.0, 0.0, 0.0]), dtype=np.float64)
    up = np.array(lookat.get("up", [0.0, 0.0, 1.0]), dtype=np.float64)

    forward = target - eye
    forward /= max(np.linalg.norm(forward), 1e-8)
    right = np.cross(forward, up)
    right /= max(np.linalg.norm(right), 1e-8)
    true_up = np.cross(right, forward)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose, eye.tolist()


def _render_mesh(mesh: trimesh.Trimesh, cam_cfg: dict, color: list[float]) -> tuple[np.ndarray, np.ndarray]:
    w = int(cam_cfg.get("width", 1920))
    h = int(cam_cfg.get("height", 1080))
    fx = float(cam_cfg.get("fx", 712.5))
    fy = float(cam_cfg.get("fy", 712.5))
    cx = float(cam_cfg.get("cx", w / 2.0))
    cy = float(cam_cfg.get("cy", h / 2.0))

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[float(color[0]), float(color[1]), float(color[2]), 1.0],
        metallicFactor=0.05,
        roughnessFactor=0.85,
        alphaMode="OPAQUE",
    )
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=[0.18, 0.18, 0.18, 1.0])
    scene.add(pr_mesh)

    pose, _ = _camera_pose_from_lookat(cam_cfg)
    camera = pyrender.IntrinsicsCamera(
        fx=fx, fy=fy, cx=cx, cy=cy,
        znear=float(cam_cfg.get("znear", 0.01)),
        zfar=float(cam_cfg.get("zfar", 100.0)),
    )
    scene.add(camera, pose=pose)

    for intensity, offset in [(4.0, np.array([0.0, 0.0, 0.0])), (1.5, np.array([3.0, -2.0, 2.0]))]:
        light_pose = pose.copy()
        light_pose[:3, 3] += offset
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=float(intensity))
        scene.add(light, pose=light_pose)

    renderer = pyrender.OffscreenRenderer(w, h)
    color_rgba, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    alpha = (np.asarray(depth) > 0).astype(np.float32)
    rgb = np.asarray(color_rgba, dtype=np.uint8)[..., :3]
    # EGL/pyrender can return a valid alpha/depth image with washed-out RGB.
    # For visualization-only output, synthesize a shaded color image from alpha+depth.
    if alpha.max() > 0 and int(rgb.min()) == 255 and int(rgb.max()) == 255:
        base = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
        d = np.asarray(depth, dtype=np.float32)
        valid = alpha > 0
        shade = np.ones_like(d, dtype=np.float32)
        if np.any(valid):
            dv = d[valid]
            d0 = float(dv.min())
            d1 = float(dv.max())
            if d1 > d0 + 1e-8:
                z = (d - d0) / (d1 - d0)
                shade = 0.55 + 0.45 * (1.0 - z)
        rgb_f = np.ones((*alpha.shape, 3), dtype=np.float32)
        rgb_f[...] = 1.0
        rgb_f[valid] = np.clip(base * shade[..., None], 0.0, 1.0)[valid]
        rgb = (rgb_f * 255.0).astype(np.uint8)
    return rgb, alpha


def _interpolate_frames(surface_pts: np.ndarray, x_ref: np.ndarray, F_ref: np.ndarray, k: int = 8) -> np.ndarray:
    tree = cKDTree(x_ref)
    dd, idx = tree.query(surface_pts, k=min(max(int(k), 1), len(x_ref)))
    if np.ndim(dd) == 1:
        dd = dd[:, None]
        idx = idx[:, None]
    w = 1.0 / np.maximum(np.asarray(dd, dtype=np.float32), 1e-8)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-8)
    out = np.zeros((len(surface_pts), 3, 3), dtype=np.float32)
    for j in range(w.shape[1]):
        out += w[:, j:j + 1, None] * F_ref[np.asarray(idx[:, j], dtype=np.int32)]
    return out


def _render_gaussian_surface(
    surface_pts: np.ndarray,
    surface_normals: np.ndarray,
    F_surface: np.ndarray,
    cam_cfg: dict,
    render_cfg: dict,
    color: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    sigma0 = compute_sigma0(surface_pts, float(render_cfg.get("sigma0_scale", 0.7)))
    eye3 = np.eye(3, dtype=np.float32)[None]
    cov = np.matmul(np.matmul(F_surface, (sigma0 ** 2) * eye3), np.transpose(F_surface, (0, 2, 1))).astype(np.float32)
    cov += 1e-6 * eye3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_t = torch.from_numpy(surface_pts).float().to(device)
    cov_t = torch.from_numpy(cov).float().to(device)
    opacity = torch.full((len(surface_pts), 1), float(render_cfg.get("opacity", 0.95)), device=device)

    lookat = cam_cfg.get("lookat", {})
    eye = np.asarray(lookat.get("eye", [20.0, -25.0, 12.5]), dtype=np.float32)
    rgb = torch.from_numpy(
        compute_shading(
            surface_pts,
            surface_normals,
            camera_pos=eye,
            light_cfg=render_cfg.get("lighting", {}),
            albedo_color=color,
            model="phong",
        )
    ).float().to(device)

    renderer, _ = setup_renderer(cam_cfg, render_cfg, training_mode=False)
    pred = renderer.render(x_t, cov_t, rgb=rgb, opacity=opacity, prefer_cov_precomp=True, return_torch=False)
    rgb_out = np.asarray(pred.get("image"), dtype=np.float32)
    alpha = np.asarray(pred.get("alpha"), dtype=np.float32)
    return rgb_out, alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--experiment", type=str, default=None)
    ap.add_argument("--ep", type=int, default=-1)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--primary-only", action="store_true")
    ap.add_argument("--normal-knn", type=int, default=24)
    ap.add_argument("--orient-knn", type=int, default=48)
    ap.add_argument("--poisson-depth", type=int, default=9)
    ap.add_argument("--density-quantile", type=float, default=0.03)
    ap.add_argument("--voxel-scale", type=float, default=1.5)
    ap.add_argument("--taubin-iters", type=int, default=3)
    ap.add_argument("--method", type=str, default="poisson", choices=["poisson", "voxel"])
    ap.add_argument("--render-mode", type=str, default="mesh", choices=["mesh", "gaussian"])
    ap.add_argument("--voxel-resolution", type=int, default=160)
    ap.add_argument("--voxel-padding", type=float, default=2.0)
    ap.add_argument("--voxel-sigma", type=float, default=1.2)
    ap.add_argument("--voxel-level-ratio", type=float, default=0.015)
    ap.add_argument("--voxel-closing-iters", type=int, default=2)
    ap.add_argument("--gaussian-knn", type=int, default=8)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg = _resolve_config(cfg, args.experiment)

    out_dir = Path(cfg.get("output_dir", "output"))
    ckpt_dir = out_dir / "checkpoints"
    if args.ep >= 0:
        ckpt_path = ckpt_dir / f"ckpt_ep{args.ep:03d}.npz"
    else:
        ckpts = sorted(ckpt_dir.glob("ckpt_ep*.npz"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        ckpt_path = ckpts[-1]
    ep_num = int(ckpt_path.stem.split("ep")[1])

    print(f"Reconstructing surface from: {ckpt_path}")
    ckpt = np.load(ckpt_path)
    x_full = np.asarray(ckpt["positions"], dtype=np.float32)
    render_mask, render_meta = _build_render_mask(x_full, cfg)
    x = np.ascontiguousarray(x_full[render_mask])

    if args.method == "poisson":
        mesh, recon_meta = _reconstruct_poisson_mesh(
            x=x,
            normal_knn=args.normal_knn,
            orient_knn=args.orient_knn,
            poisson_depth=args.poisson_depth,
            density_quantile=args.density_quantile,
            voxel_scale=args.voxel_scale,
            taubin_iters=args.taubin_iters,
        )
        suffix = (
            f"_surfpoisson_d{args.poisson_depth}"
            f"_q{int(round(args.density_quantile * 100)):02d}"
            f"_v{args.voxel_scale:.2f}"
            f"_t{args.taubin_iters}"
        )
    else:
        mesh, recon_meta = _reconstruct_voxel_mesh(
            x=x,
            resolution=args.voxel_resolution,
            padding=args.voxel_padding,
            sigma=args.voxel_sigma,
            level_ratio=args.voxel_level_ratio,
            closing_iters=args.voxel_closing_iters,
            taubin_iters=args.taubin_iters,
        )
        suffix = (
            f"_surfvoxel_r{args.voxel_resolution}"
            f"_lr{int(round(args.voxel_level_ratio * 1000)):03d}"
            f"_c{args.voxel_closing_iters}"
            f"_t{args.taubin_iters}"
        )
    render_out = Path(args.out) if args.out else out_dir / f"surface_recon_ep{ep_num:03d}{suffix}"
    render_out.mkdir(parents=True, exist_ok=True)
    mesh.export(render_out / "surface_proxy.ply")

    point_msg = f"points={recon_meta['input_points']:,}"
    if 'downsampled_points' in recon_meta:
        point_msg += f"->{recon_meta['downsampled_points']:,}"
    print(
        f"  render_mask={render_meta['render_mask_frac']:.3f}, "
        f"{point_msg}, "
        f"mesh={recon_meta['vertices']:,} verts / {recon_meta['faces']:,} faces"
    )

    rcfg = cfg.get("render", {})
    color = rcfg.get("particle_color", [0.27, 0.51, 0.71])
    cam_cfg = cfg.get("camera", {})
    multi_cfg = cfg.get("multi_view", {})
    cams, cam_eyes, labels, _ = setup_cameras(cam_cfg, multi_cfg)
    if args.primary_only:
        cams = cams[:1]
        cam_eyes = cam_eyes[:1]
        labels = labels[:1]

    surface_pts = np.asarray(mesh.vertices, dtype=np.float32)
    surface_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    if surface_normals.shape != surface_pts.shape or not np.all(np.isfinite(surface_normals)):
        surface_normals = trimesh.geometry.mean_vertex_normals(len(mesh.vertices), mesh.faces, mesh.face_normals).astype(np.float32)

    F_surface = None
    if args.render_mode == "gaussian":
        F_elastic_full = np.asarray(ckpt["F_elastic"], dtype=np.float32)
        Fp_full = np.asarray(ckpt["Fp"], dtype=np.float32) if "Fp" in ckpt.files else np.tile(np.eye(3, dtype=np.float32), (len(x_full), 1, 1))
        F_ref = np.matmul(F_elastic_full[render_mask], Fp_full[render_mask]).astype(np.float32)
        F_surface = _interpolate_frames(surface_pts, x, F_ref, k=args.gaussian_knn)

    for v, (cam, label, eye) in enumerate(zip(cams, labels, cam_eyes)):
        if args.render_mode == "gaussian":
            cam = dict(cam)
            cam["lookat"] = dict(cam.get("lookat", {}))
            cam["lookat"]["eye"] = np.asarray(eye, dtype=np.float32).tolist()
            rgb, alpha = _render_gaussian_surface(surface_pts, surface_normals, F_surface, cam, rcfg, color=color)
            save_image_png(render_out / f"view{v:02d}_{label}_rgb.png", rgb)
        else:
            rgb, alpha = _render_mesh(mesh, cam, color=color)
            save_image_png(render_out / f"view{v:02d}_{label}_rgb.png", rgb.astype(np.float32) / 255.0)
        save_image_png(render_out / f"view{v:02d}_{label}_alpha.png", alpha)
        if v == 0:
            if args.render_mode == "gaussian":
                save_image_png(render_out / "render.png", rgb)
            else:
                save_image_png(render_out / "render.png", rgb.astype(np.float32) / 255.0)

    with open(render_out / "meta.txt", "w", encoding="utf-8") as f:
        f.write(f"checkpoint={ckpt_path}\n")
        f.write(f"render_mask_frac={render_meta['render_mask_frac']:.6f}\n")
        for k, v in recon_meta.items():
            f.write(f"{k}={v}\n")
        f.write(f"render_mode={args.render_mode}\n")
        if args.render_mode == "gaussian":
            f.write(f"gaussian_knn={args.gaussian_knn}\n")

    print(f"saved={render_out}")


if __name__ == "__main__":
    main()
