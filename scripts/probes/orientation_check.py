"""Which way is up for each asset?  Renders every asset mesh under candidate rotations
(identity, x-90, x+90, z-90, z+90) from two azimuths with the photoreal scene, one PNG per asset,
so the up-axis table (assets/orientation.json) can be written by looking, once.

    orientation_check.py <out_dir> <asset> [<asset> ...]
"""
import math
import os
import sys

os.environ.setdefault("EGL_PLATFORM", "surfaceless")
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image, ImageDraw

OUT = sys.argv[1]
assets = sys.argv[2:]
ROT = {
    "id": np.eye(3),
    "x-90": trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0])[:3, :3],
    "x+90": trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])[:3, :3],
    "z-90": trimesh.transformations.rotation_matrix(-math.pi / 2, [0, 0, 1])[:3, :3],
    "z+90": trimesh.transformations.rotation_matrix(math.pi / 2, [0, 0, 1])[:3, :3],
}
W = 300
rend = o3d.visualization.rendering.OffscreenRenderer(W, W)
scene = rend.scene
scene.set_background([0.94, 0.94, 0.935, 1.0])
scene.set_lighting(o3d.visualization.rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, (0.35, -0.85, -0.4))
scene.scene.enable_sun_light(True)
scene.scene.set_sun_light((0.35, -0.85, -0.4), (1.0, 0.98, 0.94), 85000.0)
scene.scene.enable_indirect_light(True)
scene.scene.set_indirect_light_intensity(38000.0)
mat = o3d.visualization.rendering.MaterialRecord(); mat.shader = "defaultLit"
mat.base_color = [0.86, 0.80, 0.72, 1.0]; mat.base_roughness = 0.32
gmat = o3d.visualization.rendering.MaterialRecord(); gmat.shader = "defaultLit"
gmat.base_color = [0.97, 0.97, 0.965, 1.0]; gmat.base_roughness = 0.9

for A in assets:
    m = trimesh.load(f"assets/{A}.obj", force="mesh")
    v = np.asarray(m.vertices, np.float64); v = v - v.mean(0); v = v * (8.0 / np.linalg.norm(v.max(0) - v.min(0)))
    tiles = []
    for name, R in ROT.items():
        vr = v @ R.T
        om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vr), o3d.utility.Vector3iVector(np.asarray(m.faces, np.int32)))
        om.compute_vertex_normals()
        lo, hi = vr.min(0), vr.max(0); c = (lo + hi) / 2; half = float((hi - lo).max()) * 0.55
        ground = o3d.geometry.TriangleMesh.create_box(40 * half, 0.02 * half, 40 * half)
        ground.translate([-20 * half + c[0], lo[1] - 0.04 * half, -20 * half + c[2]]); ground.compute_vertex_normals()
        row = []
        for az in (35.0, 215.0):
            scene.clear_geometry()
            scene.add_geometry("ground", ground, gmat); scene.add_geometry("body", om, mat)
            el = math.radians(18.0); azr = math.radians(az)
            dist = half / (0.78 * math.tan(math.radians(30.0) / 2))
            eye = c + dist * np.array([math.cos(el) * math.sin(azr), math.sin(el), math.cos(el) * math.cos(azr)])
            rend.setup_camera(30.0, c.tolist(), eye.tolist(), [0.0, 1.0, 0.0])
            row.append(np.asarray(rend.render_to_image()))
        tile = Image.fromarray(np.concatenate(row, axis=1))
        ImageDraw.Draw(tile).text((8, 6), f"{A} {name}", fill=(40, 40, 40))
        tiles.append(np.asarray(tile))
    img = Image.fromarray(np.concatenate(tiles, axis=0))
    img.save(os.path.join(OUT, f"orient_{A}.png"))
    print("saved", A)
