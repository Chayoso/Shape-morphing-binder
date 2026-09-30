"""The Open3D Filament scene of the photoreal renderer: offscreen renderer, soft-shadow sun + image-based
lighting, the PBR material of the body, the shadow-catching ground plane, the optional translucent
target ghost, the camera ring of --views; render_views draws one mesh from every view, label stamps
the caption."""
import math
from types import SimpleNamespace

import numpy as np
import open3d as o3d

from photoreal_mesh.surface import mesh_of


def build_scene(ctx):
    """The renderer, lights, materials, ground and camera distance for this run (the box of ctx)."""
    a, ctr, half, tgt, tgt_np, frames_np, dn = ctx.a, ctx.ctr, ctx.half, ctx.tgt, ctx.tgt_np, ctx.frames_np, ctx.dn
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
        tm, _, _, _, _ = mesh_of(ctx, tgt)
        if tm is not None:
            scene.add_geometry("target", tm, ghost)
    c = ctr.cpu().numpy()
    # the camera distance that makes the bounding cube span `fill` of the frame height
    dist = half / (a.fill * math.tan(math.radians(a.fov) / 2.0))
    return SimpleNamespace(a=a, rend=rend, scene=scene, mat=mat, ghost=ghost, gmat=gmat, views=views, c=c, dist=dist)


def render_views(view, m):
    """The mesh m (None = empty scene) from every azimuth of --views, side by side."""
    a, rend, scene, mat, views, c, dist = view.a, view.rend, view.scene, view.mat, view.views, view.c, view.dist
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
    """The caption text stamped at the top left of the image (unchanged if PIL is missing)."""
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
