"""Modules of scripts/render_photoreal.py, the mesh-based photoreal morph renderer (marching cubes,
screened Poisson, IMLS or surfels of the particle density, rendered with Open3D's Filament path):

    args      the command line (every option and its documentation)
    context   the run context: archive, render box, particle spacing, density level, layer thresholds
    density   the density grids (isotropic CIC + blur, F-carried anisotropic, Yu & Turk PCA kernels)
    topology  per-particle component labels and the filament bridges (drawn topology = particles)
    surface   the per-frame surface (mesh_of) and the outer-layer surfel memory
    measures  bumpiness and the isolated-particle count (the per-frame QA)
    tracking  vertex advection with the material and closest points on a mesh
    scene     the Open3D scene, the camera and the per-view render
    video     the still and the video drivers, the sidecar of per-frame numbers

The functions read the run state from the namespace context.build_context returns. render_photoreal.py
imports them after it has set EGL_PLATFORM for the headless renderer.
"""
