"""Photoreal morph video: isosurface mesh of the particle density, rendered with Open3D's
Filament path (PBR material, image-based + sun lighting, soft shadows, ground plane).

    render_photoreal.py --npz run.npz --out out.mp4 [--res 900 --stride 3 --views 35,215]
                        [--still <frame> --out still.png]

The density grid is the one render_iso_video.py uses (trilinear splat of the particle
masses + a Gaussian of --blur particle spacings, iso = --iso x the source bulk density), so
the surface is the same object the isosurface videos show; the mesh is extracted with
marching cubes and smoothed (Taubin). Nothing is hidden: every mesh component is rendered
(--largest_only exists for illustration and is OFF by default) and a sidecar
<out>.components.txt records, per video frame, the number of isosurface components and the
number of isolated particles (8-NN distance > 3 x median), the per-frame QA the videos are
judged by (no floating particles, no particle-looking blobs).

The code lives in scripts/photoreal_mesh/: args (the command line), context (archive, render box,
spacing, density level, layer thresholds), density (the kernels), topology (particle labels, filament
bridges), surface (mesh_of, the surfel memory), measures (bumpiness, isolated particles), tracking
(vertex advection), scene (Open3D renderer, lights, camera) and video (the still and video drivers).
This file is the entry point: it parses the command line, builds the run context and the scene, and
renders the still or the video.
"""
import os
import sys

import numpy as np  # noqa: F401  (numpy and torch load before open3d, as they always have)
import torch  # noqa: F401

from photoreal_mesh.args import build_parser

a = build_parser().parse_args()

os.environ.setdefault("EGL_PLATFORM", "surfaceless")
import open3d as o3d  # noqa: E402

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from photoreal_mesh.context import build_context  # noqa: E402
from photoreal_mesh.scene import build_scene  # noqa: E402
from photoreal_mesh.video import render_still, render_video  # noqa: E402

ctx = build_context(a)          # archive, render box, spacing, density level, layer thresholds, caches
view = build_scene(ctx)         # Open3D renderer, lights, materials, ground (+ target ghost), camera
if a.still >= 0 or a.still == -2:
    render_still(ctx, view)
    sys.exit(0)
render_video(ctx, view)
