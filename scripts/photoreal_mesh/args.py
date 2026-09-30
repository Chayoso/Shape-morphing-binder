"""Command line of scripts/render_photoreal.py: every option of the mesh photoreal renderer with the
help text that documents it (the method notes live in the help strings)."""
import argparse


def build_parser():
    """The argument parser of render_photoreal.py (options, defaults and help as they have always been)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=900, help="pixels per view (square)")
    ap.add_argument("--stride", type=int, default=3, help="archived frames per video frame")
    ap.add_argument("--views", default="35,215", help="azimuths in degrees, side by side")
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--hold", type=int, default=20, help="repeat the last frame this many times")
    ap.add_argument("--track", action="store_true",
                    help="surface tracking (2026-09-22, docs/method.md 10.15): the drawn mesh is advected with the particles its "
                         "vertices are bound to and pulled toward the fresh reconstruction at --track_alpha per frame; re-meshed "
                         "only when the drawn topology (pieces, bridges, cavities) changes or the drift exceeds --track_tol "
                         "spacings. Removes the frame-to-frame re-fit jitter of an independent reconstruction per frame.")
    ap.add_argument("--track_alpha", type=float, default=0.3, help="per-frame pull of the tracked vertices toward the fresh surface")
    ap.add_argument("--track_tol", type=float, default=1.0, help="re-mesh when the mean drift exceeds this many spacings")
    ap.add_argument("--track_stretch", type=float, default=0.0,
                    help="re-mesh when the tracked mesh's 90th-percentile edge exceeds this multiple of the fresh mesh's "
                         "median edge (2 = the Nyquist factor: stretched triangles cannot carry the fresh detail); 0 = off")
    ap.add_argument("--track_keep", action="store_true",
                    help="at a drift / periodic re-mesh keep the tracked triangles that have no fresh counterpart (a neck the "
                         "reconstruction lost stays a tube); a particle-confirmed topology change still re-meshes fully")
    ap.add_argument("--track_k", type=int, default=8, help="particles a vertex is bound to (Gaussian weights of one spacing)")
    ap.add_argument("--surfel_memory", type=int, default=0,
                    help="temporal coherence WITHOUT a tracked mesh (2026-09-23): the outer-layer surfels of the previous "
                         "K-1 video frames, carried to the current frame with the material (the same kNN advection as "
                         "--track) and kept only where material still is (within one spacing of a particle), join the "
                         "current surfels before the Poisson solve, downsampled at half a spacing; every frame is a fresh "
                         "mesh (no stretched triangles, no re-mesh pops). K = frames of one control window (T / stride); "
                         "0 = off")
    ap.add_argument("--prefetch", type=int, default=4,
                    help="reconstruct this many upcoming frames in parallel threads (the Poisson solve runs in a child "
                         "process per frame, so the reconstruction overlaps across frames and with the rendering); 0 = serial")
    ap.add_argument("--track_every", type=int, default=60, help="re-mesh at least every this many video frames (bounds the "
                                                                  "stretching of the advected tessellation); 0 = never forced")
    ap.add_argument("--grid", type=int, default=160)
    ap.add_argument("--iso", default="auto",
                    help="isosurface level as a fraction of the source bulk density, or 'auto' = the level at which a "
                         "filament two particles across (a 2x2 bundle, the thinnest continuum the particles can carry) "
                         "still renders: 2 s^2 / (pi sigma^2) with s the particle spacing and sigma the blur, capped at "
                         "0.5. At 0.5 a thin neck (teat, whisker) falls below the level and its bulb renders as a "
                         "detached ball although the material is connected at the particle scale (cow at 150k).")
    ap.add_argument("--blur", type=float, default=1.5)
    ap.add_argument("--smooth", type=int, default=12, help="Taubin smoothing iterations")
    ap.add_argument("--fov", type=float, default=30.0, help="vertical field of view (degrees)")
    ap.add_argument("--fill", type=float, default=0.78, help="fraction of the frame height the box spans")
    ap.add_argument("--color", default="0.86,0.80,0.72", help="base colour (linear RGB)")
    ap.add_argument("--rough", type=float, default=0.32)
    ap.add_argument("--metal", type=float, default=0.0)
    ap.add_argument("--ground", type=int, default=1, help="shadow-catching ground plane")
    ap.add_argument("--target_ghost", type=float, default=0.0, help="alpha of a translucent target isosurface (0 = off)")
    ap.add_argument("--largest_only", type=int, default=0, help="render only the largest mesh component (illustration only)")
    ap.add_argument("--min_cells", type=float, default=1.0,
                    help="drop isosurface components whose volume is below this many MPM cells (dx^3; dx = source "
                         "bbox diagonal / cell_diag): material the grid cannot resolve is not a continuum element. "
                         "0 = draw everything. Dropped components are counted in the sidecar.")
    ap.add_argument("--cell_diag", type=float, default=26.0)
    ap.add_argument("--bridge", type=int, default=1,
                    help="draw particles the isosurface does not enclose but which link the body to another drawn "
                         "component as a filament one particle spacing thick (the rendered topology follows the "
                         "particle connectivity, not the threshold); 0 = off. Bridged components are counted in the sidecar.")
    ap.add_argument("--surface", default="mc", choices=["mc", "poisson", "imls", "surfel"],
                    help="surface from the density: 'mc' = marching cubes at the level (the gallery); 'poisson' = "
                         "screened Poisson reconstruction of the outer particle layer (density below the one-spacing "
                         "half-space value, normals from the density gradient; S2 of docs/experiments.md 2026-09-19); "
                         "'imls' = implicit moving-least-squares field of the same oriented surface particles on the "
                         "render grid, marching cubes at 0 (S3); 'surfel' = plane-pulled surfels triangulated in "
                         "their tangent planes, Laplacian remeshed (S4, the geometric part of 3D Gaussian Triangulation).")
    ap.add_argument("--post", default="none", choices=["none", "bilateral"],
                    help="mesh post-filter: 'bilateral' = bilateral normal filtering (Zheng 2011; S5)")
    ap.add_argument("--pca_k", type=int, default=32, help="S1 neighbourhood size for the weighted PCA")
    ap.add_argument("--pca_kr", type=float, default=4.0, help="S1 eigenvalue ratio clamp (Yu & Turk k_r)")
    ap.add_argument("--pca_lam", type=float, default=0.9, help="S1 centre-smoothing weight (Yu & Turk lambda)")
    ap.add_argument("--poisson_depth", type=int, default=0, help="0 = octree depth from --poisson_cell")
    ap.add_argument("--poisson_cell", type=float, default=1.0,
                    help="finest Poisson octree cell in particle spacings (depth = ceil(log2(extent / cell))). The "
                         "quadratic B-spline spans three cells, so a feature thinner than 3 cells has its two sides "
                         "cancel (cow legs at cell 1.0); the thinnest drawn continuum is two particles across, hence "
                         "cell = 2/3 spacing keeps it (docs/experiments.md 2026-09-19 run 5)")
    ap.add_argument("--poisson_trim", type=float, default=0.0,
                    help="remove Poisson vertices farther than this many blur sigmas from every layer surfel; 0 = no "
                         "trim: the surface stays CLOSED, so every component has a mass and the mass/cavity rules of "
                         "10.10 apply as to marching cubes (a trim opens the surface into flaps that pass the mass rule "
                         "through the body's voxel label: cow video, drawn pieces > 1 in 143 of 292 frames at 3 sigma)")
    ap.add_argument("--pca_sigma", type=float, default=0.0,
                    help="S1 kernel size in spacings (the anisotropic kernel keeps this isotropic volume); 0 = --blur, "
                         "the baseline kernel's size, so only the SHAPE of the kernel differs from the gallery")
    ap.add_argument("--imls_h", type=float, default=2.0, help="S3 kernel width in particle spacings")
    ap.add_argument("--bulk", default="particle", choices=["particle", "voxel"],
                    help="the bulk density the level is a fraction of: median over the particles (correct) or over "
                         "the occupied voxels (the gallery up to v8: biased low by the blur's halo)")
    ap.add_argument("--layer", default="grad", choices=["grad", "density"],
                    help="outer-layer rule for --surface poisson|imls|surfel: 'grad' = relative density gradient "
                         "|grad rho|/rho above the half-space value one spacing deep (invariant to the local density; "
                         "a stretched region of a morph frame is not all 'layer'); 'density' = density below the "
                         "half-space value one spacing deep (the first reading)")
    ap.add_argument("--pull", type=int, default=1,
                    help="surface poisson|imls|surfel: plane-pulling iterations of the outer layer (weighted PCA "
                         "normals, positions projected onto the local plane); 0 = raw positions and gradient normals")
    ap.add_argument("--bilateral_iters", type=int, default=5)
    ap.add_argument("--kernel", default="iso", choices=["iso", "aniso", "pca"],
                    help="density kernel: 'iso' = CIC deposit + separable Gaussian blur of --blur spacings (grid-aligned, "
                         "isotropic); 'aniso' = every particle deposits its own Gaussian with covariance sigma0^2 F F^T "
                         "(the Gaussian rides the deformation, docs/method.md eq. 11; sigma0 = --sigma0 x rest spacing, "
                         "F from the archive's F samples). Surface bumps and 'visible particles' are hypothesised to be "
                         "the isotropic kernel failing to cover stretched material; see docs/experiments.md 2026-09-18.")
    ap.add_argument("--sigma0", type=float, default=0.7, help="rest Gaussian size in particle spacings (aniso kernel)")
    ap.add_argument("--F", default="geom", choices=["geom", "archive"],
                    help="deformation gradient for the aniso kernel: 'geom' = the TOTAL deformation of each particle's "
                         "material patch, fitted per frame by least squares over its 12 rest neighbours (the plastic "
                         "flow included — what the patch actually looks like now); 'archive' = the archived physics F "
                         "(the ELASTIC part only after plastic assimilation, near identity).")
    ap.add_argument("--knn", type=int, default=12, help="rest neighbours for the geometric F fit")
    ap.add_argument("--still", type=int, default=-1, help="render only this archived frame to --out (png)")
    ap.add_argument("--save_mesh", default="", help="with --still: also write the mesh (.ply) and a .json of its numbers")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--label", default="")
    return ap
