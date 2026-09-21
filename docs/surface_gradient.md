# The render gradient and the surface: what it sees, why it does not flatten, how it could — and how the triangle-mesh methods get their surfaces (2026-09-19)

Companion to `docs/method.md` §10.12 (the outer-layer Poisson surface) and
`docs/experiments.md` 2026-09-19. Written after the 40k check, on the user's two questions:
the smoothing kills detail — should the render gradient not have captured it? and: give
the render gradient the job of flattening; and lay out how the existing triangle-mesh
methods produce their surfaces.

## 1. What the render channel sees today (facts from the code)

The flagship arm `render_full_dt_iso_nn` carries two image-like terms and one grid term:

| term | code | what it is | what it resolves |
|---|---|---|---|
| D_render (the render channel) | `pipeline/render_loss.py d_render` | multi-view SOFT SILHOUETTE: 6 azimuths × 3 elevations, `alpha = 1 − exp(−k·coverage)` with CIC coverage, k = 1.5, asymmetric MSE (hole weight 2, spray 1), at `render_res` 64 (c2f raises it mid-run) | the OUTLINE of the body in each view. Pixel = 2·extent/64 ≈ 0.14–0.16 wu, i.e. 1.6–1.9 spacings at 150k, 1.1–1.2 at 40k. A bump on the FACING surface changes no silhouette; only bumps on the occluding contour are visible, and those are blurred by the pixel. |
| D_pbr (`w_pbr` = 1.0 in the recipe: ON, inside the balanced render scalar; the gates read the pure silhouette) | `render_loss.py d_pbr`, `field_normals`, `shaded_view` | headlight-Lambertian shading of the particles with normals `n = −∇ρ/|∇ρ|` from the CIC density on the LOSS grid (dx = 3.6 spacings), surface weight `|∇ρ|`, soft front-bias visibility, splatted at the render pixel; L2 against the TARGET CLOUD shaded the same way. v8 bunny: d_pbr ≈ 0.02 against d_render ≈ 0.17 | facing-surface orientation at the loss-CELL scale: a normal from a density averaged over 3.6 spacings cannot register a bump of 0.6-spacing amplitude and 3-spacing wavelength at all. And its reference is the target cloud shaded with its own shot noise. |
| D_vol / ot_pace (the physics channel) | `losses/volumetric.py`, `optimizer.py phys_loss` | density image on the loss grid; under `--loss_units density` the loss grid FOLLOWS the MPM cell: v8 provenance `loss_res` 37, cell dx = diag/26 = 0.308 wu = 3.6 spacings at 150k (2.4 at 40k) | mass distribution at the MPM cell scale — one cell is wider than a whole bump wavelength |
| Gaussian loss (opt-in, `use_gauss_loss`, off) | `pipeline/gauss_loss.py` | the viewer's 3DGS rasteriser with Σ = σ0² F Fᵀ against pre-rendered target images | appearance at the pixel; target images from the target cloud's Gaussians (same noise) |

So the render gradient of the production recipe is a silhouette gradient plus a shading
gradient. The silhouette cannot see a 0.6-spacing bump on a facing surface. The shading
term looks at the facing surface, but through normals of a density averaged over the MPM
cell (3.6 spacings) — below that it registers nothing — and its reference image is the
target cloud shaded with its own shot noise. This is the precise sense in which "the
render gradient should flatten it" is not happening: the term exists, its normals are too
coarse to see the bumps, and where it does see something its target is the noise.

## 2. Where the bumps come from and what is observable

Measured on the target clouds (2026-09-19 readings):

- The target cloud is a voxel fill at pitch 0.8–1.3 spacings with 0.6–2 particles per fill
  voxel drawn with replacement — a Poisson-random subsample. Blurred at 1.5 spacings the
  relative shot noise is 1/√53 ≈ 14 %; the level set moves by ~0.6 spacing with a
  3-spacing correlation length. The source sphere is sampled the same way, so the morph
  carries the same texture from frame 0.
- Below the spacing (scales, fur) nothing exists in the cloud; no gradient can supply it.
- In the 1–3 spacing band signal and noise coincide. The level set shows the noise as
  "detail" (its normal error against the true mesh equals the Poisson surface's, 12.7–15.1°
  vs 12.9–16.9°); a reconstruction that averages the band removes both: roughness at two
  spacings, dragon 150k true 15.4° → level set 12.3° → Poisson 9.8°; bunny 40k 10.6° →
  13.2° → 5.6°.
- The loss cell is the MPM cell, 3.6 spacings at 150k. The loss-cell ladder (loss_res
  64/96/128 in legacy units, 2026-09-18) did not move the lump amplitude: the bumps live
  below every cell that was tried, and where the grid does resolve them it is fitting the
  target's noise as much as its features.

A render gradient can flatten only what it observes. Today it observes bumps neither on
the facing surface (silhouette), nor through the shading normals (cell 3.6 spacings), nor
in the density (same cell). The bump band, 1–3 spacings, is invisible to all three.

## 3. How the image-based methods get smooth surfaces (the survey the user asked for)

Every one of them optimises a photometric loss, and NONE of them gets its surface from the
photometric gradient alone. The smoothness comes from geometric regularisers that tie each
primitive to a locally fitted plane or to the normal of the rendered depth, plus a meshing
step that averages. The photometric gradient at the primitive scale is as noisy as ours.

| method | primitive | what makes it a surface (regularisers, exact) | meshing / export | manifold? | geometry quality | transfers to particles-without-images as |
|---|---|---|---|---|---|---|
| 3DGS (Kerbl 2023) | ellipsoids | none | post hoc (Poisson/TSDF) | — | poor | — |
| SuGaR (Guédon & Lepetit 2023, arXiv 2311.12775) | 3D Gaussians pulled onto a surface | density d(p) = Σ α_g exp(−½(p−μ_g)ᵀΣ_g⁻¹(p−μ_g)); ideal density from the closest Gaussian d̄(p) = exp(−⟨p−μ_g*, n_g*⟩²/(2 s_g*²)); SDF f(p) = ±s_g*√(−2 log d(p)), R = mean_p |f̂(p) − f(p)|; normal term R_norm = mean ‖∇f/‖∇f‖ − n_g*‖² | points sampled on depth maps where d(p + t v) = λ = 0.3 (linear interpolation along ±3σ), normals ∇d/‖∇d‖; screened Poisson depth 10; then Gaussians bound to triangles, joint refinement | yes (Poisson) | minutes vs hours; PSNR kept | the level set of a Gaussian density + Poisson = our S2 with depth-map samples instead of the particle layer; the SDF regulariser = "a Gaussian's density must be a thin shell" |
| 2DGS (Huang 2024, arXiv 2403.17888) | oriented planar disks, ray–splat intersection | depth distortion L_d = Σ_{i,j} ω_i ω_j |z_i − z_j| (α = 1000 bounded / 100 unbounded) — squeezes the splats along the ray; normal consistency L_n = Σ_i ω_i (1 − n_iᵀ N), N = ∇_x p_s × ∇_y p_s / ‖·‖ from the rendered depth's finite differences (β = 0.05) | TSDF fusion of median depth (Open3D, voxel 0.004, trunc 0.02) | yes (TSDF) | DTU Chamfer 0.83; without normal consistency 1.24, without depth distortion 0.88 | the two regularisers ARE the "render gradient flattens" mechanism: the primitive normal is pulled to the normal of what is rendered |
| Gaussian Surfels (Dai 2024, arXiv 2404.17774) | Gaussians with z-scale 0 (the normal's derivative vanishes) | self-supervised depth–normal consistency L_c = 1 − Ñ · N(V(D̃)) (λ_c 0 → 0.1), normal prior 0.04(1 − Ñ·N̂) + 0.005 L1(∇Ñ, 0), opacity loss exp(−(o−0.5)²/0.05) | volumetric cutting on a 512³ grid (weighted opacity, prune < 1), screened Poisson depth 10 | yes | DTU CD 0.882; 1.243 without L_c | same as 2DGS: the rendered depth defines the normal the primitives must agree with |
| Triangle Splatting (Held 2025, arXiv 2505.19175) | triangles as differentiable splats, window I(p) = ReLU(φ(p)/φ(s))^σ (φ signed distance to the edges, s the incenter, σ annealed: small = hard) | L = (1−λ)L1 + λL_DSSIM + β1 L_o + β2 L_d + β3 L_n + β4 L_s with the size term L_s = 2/‖(v1−v0)×(v2−v0)‖ (larger triangles); SfM init, +30 % every 500 it, prune unseen | triangle SOUP — "constructing a connected mesh still requires post-processing" | no | "sharper transitions than Gaussians"; no CD reported | the normal/distortion terms again; the window is a rasteriser, not a surface prior |
| Triangle Splatting+ (2025, arXiv 2509.25122) | triangles with SHARED vertices T_m = (i,j,k); vertex gradients accumulate over incident triangles | σ annealed 1.0 → 1e-4, opacity floor annealed to 1 (opaque), L_o, L_n | semi-connected mesh: 1.5 triangles per vertex, 80 % of triangles touch another | no | rendering metrics only | connectivity through shared vertices; no geometry claim |
| 2D Triangle Splatting (2025, arXiv 2506.18575) | 2D triangles, opacity o_j = O·exp(−e_j^{2γ}/2), e_j = 1 − 3 min(barycentric) | compactness γ annealed 1 → 50 over the last 10k it; L_n = mean(1 − n_j·n_j′); depth distortion L_d = ΣΣΣ o T o T (d − d)² | GLB triangle soup "with possible overlaps and discontinuities", not watertight | no | DTU CD 0.570 with 265k triangles | the normal-consistency + depth-distortion pair, once more |
| 3D Gaussian Triangulation (ECCV 2026, arXiv 2607.10690) | Gaussian surfels from posed RGB-D | plane pulling L_plane = Σ |n_iᵀ(μ_i − p_i)| (centres onto the local plane), L_n = 1 − |n_gᵀ n_i| (n_g = shortest axis) | tangent-plane triangulation: adaptive radius from the k = 4 NN mean, mutual visibility on the tangent plane, ⟨n_i, n_j⟩ > 0.9, polar-angle sort, adjacent pairs, subtended angle > 10°; Laplacian remeshing (3 × split/collapse/flip + centroid relocation); frozen historical regions | "seamless stitching" claimed, no manifold guarantee | accuracy / completion ratio < 5 cm on Replica; no CD | the plane pulling is what our `oriented_layer` does; their triangulation is what our S4 attempt failed to reproduce (overlapping fans) |
| SplatSurf (Visual Computer 2026) | triangle soup optimised from a trained 3DGS | edge/normal regularisers to a manifold | mesh | yes | — | mesh-from-splats post hoc |
| Yu & Turk 2013 (particle fluids) | anisotropic kernels from weighted PCA, centres Laplacian-smoothed (λ = 0.9), ratio clamp k_r = 4 | none — the kernel is the smoother | marching cubes of the kernel sum | yes | — | tested as S1: same 13° roughness (the centre smoothing is the useful part; the anisotropy is not) |
| screened Poisson (Kazhdan & Hoppe 2013) | oriented points | the screening term (point fidelity) vs the octree depth | indicator level set | yes | — | S2 (adopted), cell = one spacing |
| RIMLS (Öztireli 2009) | oriented points | robust MLS weights | implicit | yes | — | S3, falsified here (thin sheets cancel) |

Reading across the rows: the surface of every image-based method is made by (i) a term
that pulls each primitive's position/normal toward a locally consistent plane — 2DGS L_n,
Gaussian Surfels L_c, 3DGT L_plane, SuGaR R — and (ii) an averaging mesher (Poisson at
depth 10, TSDF at a voxel). The ablations are unambiguous: without (i) the Chamfer error
grows by 40–50 % (2DGS 0.83 → 1.24, Gaussian Surfels 0.88 → 1.24). The triangle-splatting
family adds connectivity but not smoothness; their outputs are soups. So the honest answer
to "should the render gradient capture it": in this literature it does not either. The
photometric gradient places the primitives; a geometric consistency term flattens them;
the mesher averages what is left.

## 4. Giving the render gradient the flattening role here — five mechanisms

All five are stated with their constants derived from the discretisation. None is
implemented yet; the order is the order of expected leverage.

**G1 — Shading against a denoised target.** D_pbr is already on; change its TARGET: shade
the target from its reconstructed surface (the Poisson mesh of the target cloud, normals
from the mesh) instead of from the noisy density, while the morph is shaded from its own
field normals as now — and compute those normals on a grid at the render pixel
(render_res 64: 1.6–1.9 spacings), not on the MPM cell (3.6 spacings), so that a
3-spacing bump is at least at the edge of what the normals resolve. The gradient then
pushes the morph's outer layer to shade like a smooth surface, i.e. against the bumps,
with no new term and no change to the adjoint (D_pbr already differentiates through
`field_normals`; only its grid and its target change). Constants: the render pixel for
the normal grid; the layer rule for the target's surface. Risk: the front-bias visibility
is approximate; the term stays inside the existing λ balance. This is the cheapest test
of the user's premise and should go first among G1–G3. It cannot reach below the pixel:
the 0.8–1.3-spacing fill pitch stays invisible without G5.

**G2 — Normal consistency on the outer layer (the 2DGS / Gaussian Surfels / 3DGT term,
without images).** For the outer-layer particles (the |∇ρ|/ρ rule of §10.12), penalise
Σ_i w_i (1 − n_iᵀ N_i) where n_i is the particle's PCA-plane normal over its same-side
neighbours (2-spacing Gaussian weights, as in `oriented_layer`) and N_i the density-field
normal at the particle (the loss grid's −∇ρ/|∇ρ|), plus the plane-pulling residual
Σ_i |n_iᵀ(x_i − c_i)| (3DGT's L_plane with c_i the weighted neighbour centroid). Both are
differentiable in x; the MPM adjoint pulls them back to the controls like any other
terminal term. This is exactly what the literature's ablations say matters most. Constants:
the layer threshold (0.285/spacing), the PCA kernel (2 spacings = the layer thickness);
the weight goes through the λ balance against the physics gradient norm.

**G3 — Depth distortion on the layer (2DGS L_d).** In the silhouette rasteriser, per pixel,
penalise the depth spread of the outer-layer particles that cover it: Σ_pixel Σ_{i,j} w_i w_j
|z_i − z_j|. It thins the layer along the ray — the shot-noise thickness of the surface.
Constant: none beyond the layer rule.

**G4 — Sobolev preconditioning of the render gradient (§6, `grid_smooth.py`).** The raw
∂D/∂x_T is Jacobi-like and high-frequency; a screened-diffusion metric spreads the pull
over the correlation length of the bumps (3 spacings → κ from that length). Already exists
as the experimental `render_gs` arm; it dropped the F component and worsened the shape
criteria, so it is a search-direction transform to re-test only for the surface texture,
not a candidate for the shape.

**G5 — Remove the noise at the source: stratified or Poisson-disk sampling of source and
target** (one jittered particle per fill voxel instead of drawing with replacement). This
is the one change that alters what the gradient can observe: the target's density and
shading images lose their 14 % shot noise, the loss blur and the Poisson cell can go down
to the sampling scale, and G1–G3 get clean targets. It is a discretisation change (all
runs), not a renderer change; the v8 runs stay as they are.

Pre-registered acid test for any of them: the morph's own outer layer, measured with
`scripts/probes/surface_gt.py`'s roughness at two spacings on the RAW particle surface
(the Poisson mesh at cell = one spacing, no subdivision) must drop toward the target
cloud's floor on the same probe, with the gallery QA (drawn pieces, bridges, fragments)
unchanged and the silhouette IoU within the run-to-run spread. G5 is tested first and
without a run: resample the target clouds without noise and rerun the surface acid test at
a 0.5-spacing Poisson cell — if the dragon's roughness climbs toward the true 15.4° without
fragments, the sampling is the lever and G1–G3 are worth their runs.

## 6. The gradient-stage analysis at 40k, and the protocol it dictates (2026-09-19, 17:00–18:30)

`--grad_dump` (pipeline/optimizer.py) writes, per window, the terminal covectors of the
physics, silhouette and shading channels on the particles, their control gradients after
the MPM adjoint, and the end state of the window under each channel's control gradient
alone (scaled to the accepted step's control norm); `scripts/probes/grad_stage.py` reads
them. 40k bunny, the recipe, 8 windows (3 minutes):

| stage | physics | silhouette | shading |
|---|---|---|---|
| 1 covector norm | 1.8e-3 | 8.8e-3 | 9.2e-4 |
| 1 share on the outer layer (8.5 % of particles) | 0.32 | 0.99 | 0.86 |
| 1 of that, along the normal | 0.49 | 0.73 | 0.61 |
| 1 of that, ROUGH at 2 spacings (unexplained by the neighbourhood mean) | 0.32 | 0.65 | 0.67 |
| 2 control gradient: neighbour correlation at 1 / 2 / 4 / 8 spacings | .96 / .87 / .61 / .18 | .96 / .86 / .61 / .23 | .95 / .84 / .52 / .11 |
| 3 response |dx| on the layer / interior (spacings) | 0.65 / 0.18 | 0.62 / 0.10 | 0.54 / 0.10 |
| 3 rough share of the layer's normal displacement | 0.10 | 0.16 | 0.17 |
| 3 outer-layer plane-residual RMS at the window's end (spacings; no step 0.474) | 0.458 | 0.503 | 0.505 |

The accepted composite step ends at 0.451; over the eight windows the RMS goes 0.49 → 0.45
and stays there. Reading: (1) the render covector IS a surface signal, and two thirds of it
is bump-band noise — uncorrelated between neighbours two spacings apart — because its
target images are the target cloud's own shot noise; (2) the pull-back through P2G/G2P
gives every channel's control gradient the GRID's correlation length (0.86 at two
spacings, 0.6 at four, 0.2 at eight; cell = 3.6 spacings) whatever the covector's
roughness was — the bump band is projected out of the control; (3) followed alone, the
render channels ROUGHEN the outer layer (+0.03 spacing) and the physics channel smooths it
(−0.016): the accepted step's small smoothing comes from the physics side. So the render
gradient cannot flatten the surface through the control stress, and adding a smoother
loss would not change that: the actuator has no sub-cell modes.

The protocol, first form — an external force on the outer-layer particles (the user's
suggestion): a critically damped spring toward the rough part of the local-plane residual,
τ = one window. On a slab (tests/test_layer_relax.py) it relaxed a half-spacing bump by
2.4 % in a window. The reason is the discretisation itself: a force on one particle
accelerates the momentum of its cell, which P2G/G2P average over the ~50 particles of the
cell, so the particle keeps only its share; sub-cell RELATIVE motion is not a momentum
mode. The material bonds of 10.7 already had to be a position projection for the same
reason. Second form, adopted: a per-step POSITION projection (kernels `k_layer_resid`,
`k_layer_project`; `--layer_relax`), x_p ← x_p − (1/T)(d_p − d̄_p) n_p on the outer layer
(the asymmetry rule, 0.5 spacing), d_p the residual to the same-side weighted PCA plane of
its 24 layer neighbours (Gaussian weights of two spacings, normalised on the host — a
division by the loop-accumulated weight sum inside the kernel broke the Warp adjoint,
gradients 1e23), d̄_p the neighbourhood mean of the residual. What the neighbours share
(curvature, features) cancels in d − d̄; what they do not (the sampling noise) is removed
over one window, (1 − 1/T)^T = e⁻¹ per window, as the bonds re-join over one window. Both
kernels are on the tape; directional finite differences through the extended bridge match
(tests). Frozen per window like the control basis. Acid test: the outer-layer plane-
residual RMS of the morph (0.45 spacings at 40k without it) and the Poisson / marching-
cubes roughness of the rendered frames, with the gallery QA and the silhouette IoU
unchanged.

First readings (40k bunny, 8 windows, `--grad_dump`): with the projection at 1/T the
outer-layer plane-residual RMS sits at 0.35–0.38 spacings against 0.45–0.47 without it
(−22 %), the rough share of every channel's response drops from 0.10–0.17 to 0.06, and
each window still ends slightly above where it started (0.350 → 0.365): the physics
regenerates roughness as fast as one window relaxes it — the equilibrium of the two
rates. A HARD per-step constraint (fraction 1) is FALSIFIED: it diverged within two
windows (residual 1.6e7, jitter 6.8, silhouette IoU 0.69) — the rough operator I − W
built from neighbours' residuals on their own planes is not a contraction when applied
fully, only when applied at 1/T. The fraction is therefore the window relaxation, as
derived, and the equilibrium roughness is what it buys.

Full 40k runs with the projection (`lrx_bunny`, `lrx_dragon`, the recipe + `--layer_relax`;
`scripts/probes/layer_rms.py`, stride 40 frames), outer-layer plane-residual RMS in spacings:

| | mean over the morph | end frame | target cloud (the sampling floor) |
|---|---|---|---|
| bunny, recipe (`lr64_bunny`) | 0.442 | 0.457 | 0.339 |
| bunny, + layer relaxation | **0.290** | **0.270** | 0.339 |
| dragon, recipe (`lr64_dragon`) | 0.403 | 0.371 | 0.325 |
| dragon, + layer relaxation | **0.279** | **0.261** | 0.325 |

The morph's surface ends SMOOTHER than the sampled target it is chasing — the relaxation
accumulates over the 66–80 windows as the morph slows and regenerates less. End metrics:
silhouette IoU 0.962 → 0.957 (bunny), 0.964 → 0.952 (dragon); chamfer 0.1204 → 0.1206,
0.1222 → 0.1237 (unchanged); det F min 0.68 → 0.72, 0.66 → 0.71 (less compression); wall
12.5 → 10.1 min, 22.5 → 11.8 min (fewer windows to the stop). The IoU cost is 0.5–1.2
points on 40k: the constraint holds the outer layer to its local plane, so the silhouette
term's noisy pull on individual particles no longer buys IoU at the pixel — which was the
noise. Recorded, not hidden.

## 7. Letting the render gradient reach below the cell: G1 + a position-mode control channel (2026-09-19, 19:40–)

User's question: how to make the render gradient see below the cell — is the signal too
high-resolution? Answer from §6: the signal is not the limit (65 % of the layer covector
is sub-cell), the ACTUATOR is (the stress control has no sub-cell modes) and the
REFERENCE is (the shading target carries the target cloud's noise). Two structural
changes, implemented together:

**G1 — a denoised shading reference** (`--pbr_denoised`; `render/surface_recon.py
target_surface_normals`, `pipeline/runner.py build_target`, `render_loss.shade_targets`
with precomputed normals). The target's outer layer (asymmetry rule) is plane-pulled and
Poisson-reconstructed once at setup; every target particle takes the normal of its nearest
triangle and a surface weight exp(−(dist/spacing)²), and the shading images are rendered
from those — no shot noise in the reference. The morph is shaded from `field_normals` on a
grid at the RENDER PIXEL (2·extent/render_res ≈ 1.7 spacings at 40k) with a Gaussian blur
of 1.5 spacings (the renderer's own density), instead of the MPM-cell loss grid (3.6
spacings): the normals of the drawn surface, differentiable in x.

**A — a position-mode control channel** (`--layer_ctrl`; `k_layer_project` term
`+ (u_p/T) n_p`, the extended and persistent bridges carry `u` as a leaf, the optimiser
holds it as a second Adam leaf next to dFc, clipped to one spacing per window and never
warm-started — a window's u is consumed by that window). Its adjoint is the identity times
the physics response of the remaining steps, so the render covector's sub-cell content
reaches it unfiltered. Together with the relaxation (frac 1/T, the same kernel) the outer
layer has two position-mode inputs: the self-referential smoothing of §6 and the
target-referenced correction of the render channel. Tests: u = 0 reproduces the plain
rollout; dL/du matches directional central differences; interior particles receive no
gradient; the reconstructed normals of a sampled sphere are radial on the shell with
weights ~1 there and ~0 in the core.

Runs at 40k (bunny, dragon; the recipe + `--layer_relax` +): `g1_*` = G1, `g1a_*` = G1 + A.
Acid test as in §6: outer-layer plane-residual RMS over the morph, the rendered
roughness (marching cubes / Poisson at the mid frame), silhouette IoU and chamfer, the
per-frame QA of the videos, and — the discriminator between detail and noise — the END
frame's Poisson surface against the TRUE target mesh (`surface_gt.py --gt_all`): roughness
that comes with a LOWER normal error is structure, roughness with a higher one is noise.

**Bunny 40k (recipe / + relax / + relax + G1 / + relax + G1 + A):**

| | silIoU | chamfer | det F min | layer RMS (morph / end, sp) | Poisson rough, mid frame | end frame vs true mesh: d_abs / n_dev / rough |
|---|---|---|---|---|---|---|
| lr64 (recipe) | 0.9623 | 0.1204 | 0.680 | 0.442 / 0.457 | — (13.4 MC) | 1.05 / 21.7° / 5.90° |
| lrx (+ relax) | 0.9567 | 0.1206 | 0.717 | 0.290 / 0.270 | 5.60° | 1.01 / 22.6° / 5.91° |
| g1 (+ G1) | 0.9568 | 0.1203 | 0.716 | 0.288 / 0.232 | 5.92° | 1.02 / 22.3° / 5.85° |
| g1a (+ G1 + A) | **0.9614** | **0.1196** | **0.758** | 0.363 / 0.329 | 6.19° | **1.00 / 21.9°** / 6.24° |
| target cloud itself | | | | 0.339 | 5.56° | 0.92 / 18.8° / 5.55° |

Reading: G1 alone changes nothing the metrics see (its normals at the pixel are still a
1.5-spacing blur; the reference is cleaner but the actuator is the same control stress).
The position channel A recovers the IoU the relaxation had cost (0.9567 → 0.9614, against
0.9623 without either), improves the chamfer and the compression floor, and puts the end
frame closest to the true surface (1.00 spacings) with the LOWEST normal error of the four
runs (21.9°) while its Poisson roughness is 0.35° higher — the extra roughness is
structure the channel pulled in from the target, not noise; the plane-residual RMS
(0.363, above the relaxation's 0.29 but still below the recipe's 0.44 and near the
target's own 0.339) says the same. In the stills the g1a bunny has fuller ears and a
rounder body than g1's over-thinned ones. What A cannot do: the target's shading is
rendered at the pixel (1.7 spacings) through the same 1.5-spacing blur, so the detail it
recovers is pixel-scale, not sub-pixel — the G5 (sampling) limit stands.

**Dragon 40k (same four arms):**

| | silIoU | chamfer | det F min | layer RMS (morph / end) | Poisson rough, mid frame (comps) | end frame vs true mesh: d_abs / n_dev / rough |
|---|---|---|---|---|---|---|
| lr64 (recipe) | 0.9642 | 0.1222 | 0.663 | 0.403 / 0.371 | 7.8° | 0.40 / 25.1° / 7.89° |
| lrx (+ relax) | 0.9519 | 0.1237 | 0.705 | 0.279 / 0.261 | 7.2° (7) | 0.39 / 24.6° / 7.82° |
| g1 (+ G1) | 0.9493 | 0.1249 | 0.743 | 0.265 / 0.239 | 7.0° (7) | 0.39 / 24.8° / 7.46° |
| g1a (+ G1 + A) | 0.9594 | 0.1235 | **0.778** | 0.341 / 0.263 | 8.5° (10) | 0.40 / 24.9° / 8.59° |
| target cloud itself | | | | 0.325 | 7.8° | 0.27 / 19.9° / 7.8° |

Same pattern as the bunny on the global metrics: the relaxation costs 1.2 IoU points, A
gives 0.75 of them back (0.9519 → 0.9594, against 0.9642), the compression floor rises
monotonically (0.66 → 0.78). On the surface itself the dragon is less kind to A than the
bunny was: the end frame's normal error against the true mesh is the same for all four
arms (24.6–25.1°, the mesh-scale morph error dominating), so the +1.1° of Poisson
roughness A adds (7.5 → 8.6°) is not shown to be structure here — the dragon's scales sit
at the render blur and the channel's pull at the pixel cannot resolve them; what it pulls
is the pixel-scale outline. Honest summary across both targets: A buys shape fidelity
(IoU, chamfer, det F) at a small roughness cost over the relaxation alone; the relaxation
buys smoothness at an IoU cost; neither reaches below the render pixel, which is the
sampling limit (G5).

**Decision proposed to the user (19:10):** recipe += `--layer_relax --pbr_denoised
--layer_ctrl` (smooth particle surface, shape within 0.3–0.5 IoU points of the old recipe,
the best chamfer and compression floor), Poisson as the deliverable surface; for the
smoothest possible surface at a 1-point IoU cost, `--layer_relax` alone. The 150k gallery
then needs NEW runs (the projection and the channel are forward-model changes), not a
re-render. Accepted (19:20), with two conditions: the whole pipeline is verified at 40k
before any 150k run, and the ambiguous readings are resolved.

**The ambiguity resolved: is A's extra roughness structure or noise?** The mean normal
error was the wrong instrument (dominated by the mesh-scale morph error, equal across
arms). Two band-limited measures at the two-spacing scale (`surface_gt.py --gt_all`,
`detail_analysis`): `hp_res` = RMS of the high-passed signed distance of the end frame's
Poisson surface to the true surface (bumps that FOLLOW the true surface leave it
unchanged; bumps that do not raise it); `dcorr` = correlation of the high-passed recon
normal field with the high-passed true normal field at the closest points (the target's
own detail is positively correlated, noise is not).

| end frame | bunny hp_res (sp) / dcorr | dragon hp_res / dcorr |
|---|---|---|
| recipe (lr64) | 0.236 / +0.28 | 0.203 / +0.28 |
| + relax (lrx) | 0.201 / +0.27 | 0.199 / +0.32 |
| + G1 (g1) | 0.199 / +0.27 | 0.197 / +0.32 |
| + G1 + A (g1a) | **0.185** / +0.26 | **0.195** / **+0.33** |
| target cloud's Poisson (the floor) | 0.201 / +0.37 | 0.181 / +0.44 |

Verdict: (1) A's bumps are STRUCTURE — g1a has the lowest residual to the true surface on
both targets (its Poisson roughness is higher AND its residual is lower, which only a
surface that follows the truth can do); on the dragon it is also the most correlated
with the true detail (+0.33 against +0.28 for the recipe), on the bunny the correlation
is flat within 0.02 (the bunny has little true detail at two spacings: 10.6° against the
dragon's 21.7°). (2) The relaxation removes NOISE, not detail: hp_res 0.236 → 0.201 with
dcorr unchanged (bunny) or up (dragon). (3) The relaxation's IoU cost is at the
silhouette pixel, not in the shape: chamfer (0.1204 → 0.1206) and the end frame's distance
to the true surface (1.05 → 1.01) do not move. (4) The recipe's higher residual (0.236) is
the sampling noise the physics carried to the end; below the target's own floor (0.201)
nothing can go without G5.

**The gradient at each stage under the new recipe (2026-09-21, 09:30; `gs40new_bunny`,
8 windows, `--grad_dump` extended to the u channel; `grad_stage.py` stages 2b / 3b).**

| stage | physics | silhouette | shading (G1) |
|---|---|---|---|
| 1 covector: layer share / normal / ROUGH at 2 sp | 0.29 / 0.51 / 0.29 | 0.97 / 0.73 / 0.53 | 0.86 / 0.57 / 0.55 (old recipe 0.67) |
| 2 through the STRESS control: corr at 2 / 4 / 8 sp | .87 / .62 / .19 | .89 / .65 / .26 | .85 / .55 / .15 |
| 2b through the u CHANNEL: corr at 2 / 4 / 8 sp; rough share | .69 / .37 / .13; 0.18 | **.46 / .21 / .08; 0.35** | .45 / .18 / .06; 0.39 |
| 2b u-gradient norm | 6.4e-4 | 6.4e-3 | 5.3e-4 |
| 3b response through u alone: |dx| layer / interior (sp); rough share; layer RMS end (base 0.429) | 0.33 / 0.003; 0.20; 0.474 | 0.19 / 0.001; 0.22; 0.470 | 0.20 / 0.001; 0.26; 0.454 |
| 3 response through the stress control alone: layer RMS end (base 0.498) | 0.499 | 0.534 | 0.542 |

The accepted u per window: RMS 0.585 spacings over the 3 399 layer particles, 19 % of
them AT the one-spacing clip, neighbour correlation at 2 sp 0.31. λ 0.05–0.14 (old recipe
0.09–0.28: the u-gradient raised the render channel's norm and the balancer lowered λ),
g_share 0.33–0.52.

Reading. (1) The render covector on the particles is what it was: 97 % on the outer
layer, half of it rough; G1 lowered the shading covector's rough share from 0.67 to 0.55
(the reference is clean, but the morph's own normals on the finer pixel grid are noisier
than on the cell grid — the two effects nearly cancel). (2) Through the stress control
nothing changed: the grid's low-pass (0.89 at two spacings) is the same. (3) Through u
the render gradient KEEPS half of its two-spacing content (0.46 against 0.89) and a fifth
at four spacings — this is the sub-cell path that did not exist, and the render channel
owns it (its u-gradient is 10× the physics channel's before λ). (4) What it does with it,
alone: moves the layer 0.19 spacing per window and the interior not at all, and RAISES
the layer's plane residual by 0.04 spacing (0.429 → 0.47) — in the expansion phase the
silhouette pull at the pixel is half rough, and the relaxation removes that half next
step; the accepted u sits at the clip on a fifth of the layer, so in this phase the
channel is doing pixel-scale TRANSPORT (the outline toward the target) at capacity, not
sub-cell correction. The eight-window RMS with A (0.43–0.51) is above the relaxation-only
run's (0.35) and level with the old recipe's (0.47); over the full morph the end-frame
analysis above says the surplus is structure. So the two position inputs fight in the
expansion phase and agree at the end. What the numbers point at, if anything is to be
changed: the rough part of u (what its 2-spacing neighbourhood mean does not explain, 35 %
of the gradient) is exactly what the relaxation undoes one step later — projecting the
u-gradient onto the layer's smooth subspace with the same W (h = 2 spacings, no new
constant) would give the channel the pixel-scale structure and none of the noise, and
free the clip for the transport it is being used for. Not run; recorded as the next test.

**Was the covector always half noise? (user, 2026-09-21).** Yes, and worse: the rough
share of the layer's normal component at two spacings, 8-window means — v8 recipe
silhouette 0.65 / shading 0.67; + relaxation 0.52 / 0.56; the new recipe 0.53 / 0.55
(the 40 % of fig. 2 is window 3 alone). Two sources add: the target images' shot noise
(what G1 removed from the shading reference) and the CIC silhouette rasteriser itself —
a particle's gradient is the derivative of its own 2×2-pixel kernel footprint, so
neighbours at different sub-pixel offsets get different, even opposite, pulls; that part
is recipe-independent. It was invisible before because the stress path's grid low-pass
hid it (fig. 4: every recipe's stress-path curve coincides); the u channel is the first
actuator that lets it reach the surface, which is why the projection below is needed.

**The projection test (`--layer_ctrl_smooth`, commit 878afd0).** The u leaf's Adam step
is replaced by W·step, W the relaxation's own normalised same-side neighbour weights
(h = 2 spacings): a search-direction transform, the balancer and the physics untouched.
Runs `ps40_bunny`, `ps40_dragon` (the recipe + the projection) against `p40_*`; the
8-window dump `gs40smooth_bunny` for the stage analysis.

Results (10:30). On the channel it does what it was designed to do: the accepted u goes
from RMS 0.585 to 0.397 spacings, from 19 % to 8.5 % of the layer at the clip, from a
2-spacing correlation of 0.31 to 0.71; through u alone the render channels no longer
roughen the layer (window-end RMS 0.436 = the u = 0 base, against +0.04 before; shading
even −0.006). The u-gradients themselves are unchanged (the projection acts on the step).
The 8-window silhouette IoU falls 0.911 → 0.902: with the clip no longer saturated the
expansion-phase transport through u is slower.

| full 40k runs | silIoU | chamfer | det F min | layer RMS morph / end | Poisson rough mid | end frame hp_res / dcorr |
|---|---|---|---|---|---|---|
| bunny, recipe (`p40`) | 0.9628 | 0.1196 | 0.766 | 0.355 / 0.276 | 6.2° | 0.177 / +0.28 |
| bunny, + projection (`ps40`) | 0.9585 | 0.1205 | 0.767 | **0.330** / 0.265 | 6.2° | 0.199 / +0.24 |
| dragon, recipe | 0.9555 | 0.1235 | 0.736 | 0.338 / 0.265 | 8.5° | 0.200 / +0.32 |
| dragon, + projection | 0.9534 | 0.1239 | 0.756 | **0.308** / 0.288 | **7.8°** | 0.197 / +0.31 |

Verdict: FALSIFIED as an addition to the recipe. The morph-mean layer RMS drops 7–9 % and
the dragon's mid-frame Poisson roughness 8.5 → 7.8°, but the bunny's end frame loses
detail against the true mesh (high-passed residual 0.177 → 0.199, detail correlation
+0.28 → +0.24, about two twin-spreads), the IoU and chamfer move by the spread, and the
early morph is slower. The reason is structural: the relaxation IS this same W applied to
the state one step later, so projecting the step as well is redundant — it takes the
channel's 2-spacing structure away together with its noise, while the relaxation alone
takes only what the neighbours do not share. `--layer_ctrl_smooth` stays available; it
is not in `RECIPE`.

**The run-to-run spread of the IoU comparisons.** The verification runs `p40_bunny` /
`p40_dragon` are the same configuration as `g1a_*` (the new recipe through
`hyde06_env.sh`): silIoU 0.9628 vs 0.9614 (bunny), 0.9555 vs 0.9594 (dragon); chamfer
0.1196 vs 0.1196, 0.1235 vs 0.1235; det F min 0.766 vs 0.758, 0.736 vs 0.778. The IoU
spread of identical runs is therefore ±0.4 points (CUDA atomics, 2026-09-18 controls
agreed). Read against it: the relaxation's cost (−0.6 bunny, −1.2 dragon) is 1.5–3× the
spread; A's recovery (+0.5, +0.75) is 1–2×; the new recipe against the v8 recipe (+0.05
bunny, −0.9 dragon) is within 2×. Chamfer and det F, which do not move between the
twins, carry the shape conclusion.

## 8. The three-way check at 40k (2026-09-21; pre-registered before the readings)

User: check all three. All at 40k, 150k excluded.

- **G5 — stratified sampling** (`--sampler stratified`; `sampling/mesh.py
  sample_volume_stratified`): one jittered particle per fill voxel, the fill resolution
  chosen by bisection so the fill holds ≥ n voxels (bunny 40k: 44³ = 42 759 voxels, pitch
  0.33 wu), the surplus dropped without replacement; source and target alike. Adopt iff
  the covector's rough share falls AND the end frame's high-passed residual falls / detail
  correlation rises beyond the twin spread (0.01 / 0.02), with IoU and chamfer within the
  spread. Runs `g5_bunny`, `g5_dragon`; dump `gs40g5_bunny`.
- **The quadratic B-spline splat** (`--sil_kernel quad`; `losses/silhouette.py
  splat_terms`, `set_kernel`): the silhouette and shading rasterisers splat with the
  3×3 quadratic B-spline instead of the 2×2 CIC — the lowest-order kernel whose
  derivative is continuous, so a particle's pull no longer flips sign across pixel edges;
  targets and morph alike; no constant. Adopt iff the silhouette covector's rough share
  falls, IoU and chamfer stay within the spread and the end-frame residual / correlation
  do not worsen. Runs `k40_bunny`, `k40_dragon` (after the thin-feature batch); dump
  `gs40quad_bunny`.
- **The thin-feature targets** (bob, beast, C) through the new recipe (`t40_*`) and the
  v8 recipe (`t40v8_*`): the relaxation and the u channel need same-side neighbours,
  which a two-particle sheet may not have. Pass iff the QA columns (grid fragments ≥ 1
  cell, end fragments, re-attachments, drawn pieces > 1 in the Poisson videos) are 0 and
  the silhouette IoU is within the spread of the v8 twin.

**Stage-1/2b readings (8-window bunny dumps, 11:05):**

| | new recipe | + G5 stratified | + quad splat |
|---|---|---|---|
| silhouette covector: norm / rough share at 2 sp | 9.2e-3 / 0.53 | 9.2e-3 / 0.52 | **6.2e-3 / 0.33** |
| shading covector: rough share | 0.55 | 0.54 | **0.42** |
| silhouette u-gradient: rough share / corr at 2 sp | 0.35 / 0.46 | 0.37 / 0.45 | **0.22 / 0.61** |
| accepted u: RMS (sp) / at the clip / corr 2 sp | 0.585 / 19 % / 0.31 | 0.588 / 0 % / 0.40 | 0.568 / 21 % / 0.35 |
| 8-window silIoU | 0.9105 | 0.9097 | 0.9181 |

G5 changes NOTHING in the covector: the target cloud's shot noise is not where the rough
half comes from (the stratified cloud still has its ±½-pitch jitter, and the rasteriser
does not care which sample it splats). The quad splat removes 37 % of the silhouette
covector's rough share (0.53 → 0.33) and 24 % of the shading's, shrinks the silhouette
gradient's norm by a third (the discontinuous CIC derivative was carrying sign flips as
magnitude), and the u channel receives a far more coherent signal (rough 0.35 → 0.22,
correlation at two spacings 0.46 → 0.61). The recipe-independent half of the roughness
was the rasteriser, as hypothesised; what is left (0.33) is the quadratic kernel's own
second-derivative discontinuity plus genuine sub-two-spacing structure. Full runs: next.

**G5 full runs (`g5_bunny`, `g5_dragon`; 11:05–11:20).** Against the recipe twins `p40_*`
(spacings, degrees; the "target" rows are each run's own sampled target through the
Poisson surface — the floor the morph is chasing):

| | silIoU | chamfer | det F | layer RMS morph / end (target floor) | Poisson rough mid | END FRAME vs the true mesh: d_abs / d_95 / n_dev / rough / hp_res / dcorr |
|---|---|---|---|---|---|---|
| bunny, recipe | 0.9628 | 0.1196 | 0.766 | 0.355 / 0.276 (0.339) | 6.2° | 0.99 / 6.13 / 21.8° / 6.42 / 0.177 / +0.28 |
| bunny, + G5 | **0.9670** | **0.1163** | 0.752 | **0.267** / 0.295 (0.294) | **5.4°** | **0.43 / 1.12 / 17.6° / 5.42 / 0.159 / +0.36** |
| bunny targets (recipe → G5) | | | | | | 0.92 → 0.26 / 6.0 → 0.66 / 18.8 → 14.5° / 5.55 → 5.00 / 0.201 → 0.174 / +0.37 → +0.38 |
| dragon, recipe | 0.9555 | 0.1235 | 0.736 | 0.338 / 0.265 (0.325) | 8.5° | 0.40 / 1.25 / 25.1° / 8.00 / 0.200 / +0.32 |
| dragon, + G5 | **0.9613** | **0.1188** | **0.767** | **0.293** / 0.249 (0.310) | **8.1°** | **0.36 / 1.10 / 23.6° / 8.12 / 0.185 / +0.34** |
| dragon targets (recipe → G5) | | | | | | 0.27 → 0.24 / 0.67 → 0.74 / 19.9 → 19.7° / 7.78 → 7.17 / 0.181 → 0.171 / +0.44 → +0.49 |

Verdict: ADOPTED. The pre-registered criteria are met on both targets beyond the twin
spread: the end frame's high-passed residual to the true surface falls (0.177 → 0.159,
0.200 → 0.185), its detail correlation rises (+0.28 → +0.36, +0.32 → +0.34), the mean
normal error falls (21.8 → 17.6°, 25.1 → 23.6°), IoU rises 0.4–0.6 and chamfer falls 3–4 %
(each against its own target), the outer layer is 13–25 % smoother over the morph and its
Poisson surface 0.4–0.8° smoother at mid-morph. Two honest notes. (1) Part of the bunny's
absolute-distance gain is the TARGET's: the stratified fill at 44³ (pitch 0.33 wu) does
not build the spurious closed floor under the open-base bunny.obj that the 110³ fill did
(target Poisson d_95 6.0 → 0.66), so the bunny's d_abs / d_95 columns compare a morph to a
better target, and the morph-minus-target gap (0.07 → 0.17 spacings) is not an
improvement; the dragon, a closed mesh, shows the morph's own gain (gap 0.13 → 0.12,
n_dev gap 5.2 → 3.9°). The band-limited columns (hp_res, dcorr) and the normal error are
the ones that carry the verdict. (2) The covector's rough share did NOT move (stage A):
G5 improves what the gradient is chasing (a target with 25 % less layer noise, 0.339 →
0.294) and what the morph starts from (the source sphere, the same sampler), not the
gradient's own texture — that is the quad splat's job. The morph runs 2.3× longer (23 min:
the stratified source/target take more windows to the stop; the run-time budget is not a
criterion here).

**Thin-feature targets (`t40_*` new recipe vs `t40v8_*` v8 recipe; 11:25):**

| | silIoU | chamfer | hole | det F | wall | layer RMS morph / end (target floor) |
|---|---|---|---|---|---|---|
| bob v8 → new | 0.9734 → **0.9779** | 0.1172 → 0.1169 | 0 → 0 | 0.809 → 0.832 | 23 → 26 min | 0.542 → **0.430** / 0.505 → 0.491 (0.472) |
| beast v8 → new | 0.9376 → 0.9386 | 0.1212 → 0.1239 | 2.17 → 2.01 % | 0.755 → 0.723 | 45 → 39 min | 0.493 → **0.370** / 0.395 → 0.283 (0.353) |
| C v8 → new | 0.8371 → **0.9134** | 0.1909 → **0.1503** | 0.41 → 0.09 % | 0.813 → 0.841 | 9.6 → 10.3 min | 0.522 → **0.448** / 0.594 → 0.431 (0.313) |

No thin-feature target got worse: the ring (bob) and the C gain IoU (+0.45, +7.6 points),
the beast is within the spread; the outer layer is 20–25 % smoother on all three, the
beast's end layer below its target's floor. Both C runs stop after 15 windows (301
frames) — the same early stop on both recipes, so the C comparison is at the stop, where
the new recipe is far ahead. QA (no renderer): grid fragments ≥ 1 cell 0 frames on all
six runs; end fragments beast 2 → 0, others 0; re-attachments 0 (no net); stray census
at the end, particles beyond 1 wu: bob 1 → 0 (max 1.33 → 0.14 wu), beast 4 → 0 (1.42 →
0.47), C 104 → 29 (2.59 → 1.78). The new recipe is cleaner on every column. Poisson
video pieces: stage C2, below.

**Quad full runs (`k40_*` vs `p40_*`), end metrics:** bunny silIoU 0.9628 → 0.9658,
chamfer 0.1196 → 0.1191, det F 0.766 → 0.736; dragon 0.9555 → 0.9582, 0.1235 → 0.1229,
0.736 → 0.762 — at the spread's edge, in the good direction. **The combination
(`gq_*` = G5 + quad):** bunny 0.9658 / 0.1165 / 0.766, dragon 0.9649 / 0.1187 / 0.752 (10
min each: the combination reaches its stop fastest).

**Quad and the combination on the surface (12:00):**

| | layer RMS morph / end | Poisson rough mid | end frame vs the true mesh: d_abs / d_95 / n_dev / rough / hp_res / dcorr | frames to the stop |
|---|---|---|---|---|
| bunny recipe (`p40`) | 0.355 / 0.276 | 6.2° | 0.99 / 6.13 / 21.8° / 6.42 / 0.177 / +0.28 | 1504 |
| bunny + quad (`k40`) | 0.374 / 0.341 | 6.3° | 0.99 / 6.12 / 21.7° / 6.45 / **0.212** / +0.28 | 1542 |
| bunny + G5 (`g5`) | 0.267 / 0.295 | 5.4° | 0.43 / 1.12 / 17.6° / 5.42 / 0.159 / +0.36 | 1143 |
| bunny + G5 + quad (`gq`) | 0.269 / 0.252 | 5.8° | 0.66 / **4.22** / 18.8° / 5.60 / **0.258** / +0.32 | 882 |
| dragon recipe | 0.338 / 0.265 | 8.5° | 0.40 / 1.25 / 25.1° / 8.00 / 0.200 / +0.32 | 1965 |
| dragon + quad | 0.340 / 0.256 | 8.1° | 0.40 / 1.21 / 24.5° / 7.84 / 0.194 / +0.32 | 1746 |
| dragon + G5 | 0.293 / 0.249 | 8.1° | 0.36 / 1.10 / 23.6° / 8.12 / 0.185 / +0.34 | 1122 |
| dragon + G5 + quad | 0.307 / 0.241 | 8.0° | 0.36 / 1.13 / 23.8° / 8.17 / 0.194 / +0.34 | 921 |

Verdict on the quad splat: FALSIFIED as a recipe addition. It does what it was built to do
to the gradient (stage A: the silhouette covector's rough share 0.53 → 0.33) and the
shape metrics move a spread in the good direction, but the surface it produces is not
better: the bunny's end frame is ROUGHER in the band against the true mesh (hp_res 0.177 →
0.212, 3.5 spreads) with a higher outer-layer RMS (0.355 → 0.374, end 0.276 → 0.341); the
dragon is within the spread. On top of G5 it is worse still — the bunny stops early (882
frames) with an unfinished base (d_95 1.12 → 4.22) and hp_res 0.159 → 0.258. The
mechanism, as far as the numbers show it: the CIC kernel's sign-flipping per-particle
pulls cancel inside a cell and across the relaxation, so they cost nothing on the surface;
the quad kernel's coherent pull drives the u channel harder at the pixel scale (its
u-gradient correlation at two spacings 0.46 → 0.61, u at the clip 21 %) and the layer
sits farther from its relaxed state, while the smoother, smaller silhouette gradient
(norm −⅓, the balancer restores the share) yields smaller window gains and an earlier
three-reject stop. A smoother covector was not what the surface needed. `--sil_kernel
quad` stays available; not in the recipe.

**Decision (12:05).** RECIPE += `--sampler stratified` (G5). The v8 recipe with the outer-
layer relaxation, the denoised shading reference, the position channel and stratified
sampling of source and target; the deliverable surface Poisson. The quad splat and the
smooth-subspace projection are recorded as falsified additions; the thin-feature targets
pass every QA column and gain IoU under the recipe (their Poisson video pieces: stage C2).

## 9. Why the stratified cloud has no shot noise — the proof (2026-09-21; `scripts/probes/sampling_noise.py`)

Setting: a normalised Gaussian blur K of width σ, n particles in volume V, volumetric
spacing p = (V/n)^{1/3}, intensity ρ = 1/p³, the blurred density ρ̂(x) = Σ_i K(x − x_i).
Note the unit: the project's "spacing" is the median 8-NN distance, which for a Poisson
process is (6/π)^{1/3} = 1.24 p, so the renderer's σ = 1.5 spacings is 1.86 p.

*With replacement* (the v8 sampler: n draws from a fine fill lattice, jittered) is a
Poisson process: the count in any region has variance equal to its mean, and by Campbell's
theorem Var[ρ̂] = ρ ∫K² = ρ (4πσ²)^{−3/2}, so the relative fluctuation is
(p/σ)^{3/2} / √(8π^{3/2}) = 5.9 % at σ = 1.86 p. Its spectrum is white times the blur:
P(k) = ρ e^{−σ²k²}.

*Stratified* (one particle per fill voxel, jittered by u ~ U(−p/2, p/2)³): the count per
voxel is exactly one — the monopole term is gone — and only the jitter moves mass:
ρ̂(x) = Σ_v K(x − x_v − u_v) ≈ Σ_v [K(x − x_v) − u_v·∇K(x − x_v)], a DIPOLE field with
Var[ρ̂] = (p²/12)(1/p³)∫|∇K|² = 1/(64 π^{3/2} p σ⁵), relative fluctuation
(p/σ)^{5/2} / √(64π^{3/2}) = 1.1 % at σ = 1.86 p — 5.2× lower — and a spectrum
P(k) = ρ (k²p²/12) e^{−σ²k²} that vanishes at long wavelengths: the ratio to the Poisson
spectrum is k²p²/12. A jitter is a displacement, and a displacement field's density
perturbation is a divergence: it has no k = 0 component. The level-set displacement,
δh = δρ/|∇ρ| = rel·σ√π at the half-space edge, is 0.19 p vs 0.04 p.

Measured (CIC deposit at 0.6 spacings + the 1.5-spacing blur, interior voxels deeper than
3σ; the cube with the two pipeline samplers, the bunny / dragon target clouds of the v8 and
G5 archives): interior relative fluctuation cube 5.5 % → 1.1 % (theory 5.9 / 1.1), bunny
8.1 → 2.9 %, dragon 5.2 → 2.2 %; index of dispersion of the count in 1.5-spacing spheres
(Poisson = 1) cube 1.01 → 0.26, bunny 1.30 → 0.31, dragon 1.06 → 0.28 — the replacement
clouds ARE Poisson (the bunny even over-dispersed: the 110³ fill re-draws voxels); the
cube's spectrum follows the two theory curves (the stratified one down to a 1 %-power
measurement floor at k·spacing < 1: window leakage and the lattice–grid beat). The
stratified targets' extra fluctuation over the cube's 1.1 % is the 6.5 % of fill voxels
left empty to hit n exactly (a Poisson term, √0.065 × 5.9 % = 1.5 %) plus boundary
effects. On the surface: the target level set's band-limited residual against the true
mesh falls 0.185 → 0.137 (bunny) and 0.168 → 0.156 (dragon); the Poisson surface's 0.200 →
0.174 and 0.181 → 0.171 with the detail correlation up. Page: artifact "층화 샘플링의
잡음 증명".

## 10. Does the render channel change the physics under the current recipe? The factorial (2026-09-21; pre-registered before the readings)

The user's question (13:30): can the evidence so far be read as "rendering meaningfully
affects the physics"? What IS established (docs/experiments.md 2026-09-16/17/18): under
the v3 recipe at 40k the render channel adds 1.4–4.6 silIoU points and lowers the physics
term itself (dragon D_vol 0.0259 → 0.0206); under v6/v7 at 150k the render channel is
~35 % of every accepted control update at a cosine of 0.02–0.08 to the physics gradient
(a deterministic per-window measurement), and the outcome sits 4–10 run-to-run spreads
above the physics-only and cut twins on bunny, bob and dragon; the trajectory half
(the cut twin's divergence after the intervention) is 1.4–1.7× the control's own — present
but modest, and buried under chaos on the dragon. Two targets (bob ring, V) were worse
with the channel under v3. What is NOT established:

- (a) any of it under the CURRENT recipe (relaxation + G1 + u + stratified): no λ = 0 twin
  has been run since the actuator changed;
- (b) the attribution: through the stress control the render gradient is grid low-passed
  (§6–§7: correlation 0.87–0.89 at two spacings, the same as the physics gradient), so the
  physics path acts at the cell scale (3.6 spacings at 40k) and the sub-cell path is u — a
  per-window position projection that bypasses P2G/G2P (kinematic, not physics);
- (c) the physics STATE with and without the channel (D_vol, det F, windows to the stop);
- (d) the trajectory half at 40k, where the chaos is milder than at 150k.

**Design.** 40k, seed 1, stratified sampling (identical particles across the six runs of a
target), bunny / dragon / bob, `scripts/ops/factorial40.sh`:

| cell | run | flags |
|---|---|---|
| render on, u on | `fx_11` | RECIPE |
| the same again | `fx_11c` | RECIPE (identical configuration: the run-to-run spread) |
| render off, u on | `fx_01` | RECIPE `--lambda_auto 0` |
| render on, u off | `fx_10` | RECIPE without `--layer_ctrl` |
| render off, u off | `fx_00` | RECIPE without `--layer_ctrl`, `--lambda_auto 0` |
| intervention | `fx_cut` | RECIPE `--render_until K`, K = a third of the run (20 / 20 / 8 windows) |

Readings per run: end metrics (silIoU, chamfer, det F min), json telemetry (`fx_summary.py`:
windows, g_share over windows 1–20 and all, λ, cosine, D_vol first / last window), outer-
layer RMS (morph mean / end), QA columns (grid fragments, end fragments, re-attachments,
stray census), end frame vs the true mesh (Poisson; d_abs / d_95 / n_dev / rough / hp_res /
dcorr), and the divergence from `fx_11` of the cut twin against the control
(`render_effect.py --ctrl fx_11c`). The spread is `fx_11` vs `fx_11c` (with `g5_*` as a
third sample on bunny / dragon); prior spread from the p40 twins: IoU ±0.4, hp_res 0.01,
dcorr 0.02.

**Hypotheses and what falsifies them.**

- H-A (outcome): `fx_11` − `fx_01` exceeds the spread in silIoU (prediction +1–5 points,
  as under v3) and chamfer (−1–3 %). Falsified if within the spread — then the channel
  does not change the outcome under the current recipe.
- H-B (physics state): with the channel, D_vol at the end is LOWER (as v3 dragon) and the
  run reaches its stop in fewer windows; g_share 0.33–0.52 in every render-on run (the
  deterministic proof that the render gradient is in the control update). Falsified if
  D_vol is higher or equal within the spread.
- H-C (attribution): with u OFF, `fx_10` − `fx_00` keeps the outline gain (silIoU beyond
  the spread) but leaves the surface detail (hp_res, dcorr, n_dev) within the spread — the
  physics path is cell-scale. The render effect that needs u, (`fx_11` − `fx_01`) −
  (`fx_10` − `fx_00`), is where any detail effect lives. u's own effect `fx_11` − `fx_10`:
  hp_res / dcorr improve as in §7, IoU within the spread. Falsified if `fx_10` − `fx_00`
  moves hp_res / dcorr beyond the spread (the physics path DOES carry detail) or if the
  u-dependent part is nil.
- H-D (trajectory): the cut twin's divergence from `fx_11` after K exceeds the control's
  by more than 1.5× (150k bob / bunny: 1.4–1.7×). Falsified if indistinguishable.

Verdict rule: "the render channel meaningfully changes the physics" is supported only if
H-A and H-B hold on at least two of three targets; the surface-detail claim is supported
only if the u-dependent part of H-C is beyond the spread — otherwise the honest statement
is "render controls the outline through the physics at the cell scale; the sub-cell surface
is the kinematic channel plus the Poisson surface".

## 5. Sources

Triangle Splatting arXiv 2505.19175; Triangle Splatting+ 2509.25122; 2D Triangle
Splatting 2506.18575; Incremental Online Scene Reconstruction by 3D Gaussian Triangulation
(ECCV 2026) 2607.10690; SuGaR 2311.12775; 2D Gaussian Splatting 2403.17888; High-quality
Surface Reconstruction using Gaussian Surfels 2404.17774; Yu & Turk, TOG 2013; Kazhdan &
Hoppe, TOG 2013; Öztireli, Guennebaud, Gross, CGF 2009; SplatSurf, The Visual Computer 2026.
