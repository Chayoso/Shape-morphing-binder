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

**The run-to-run spread of the IoU comparisons.** The verification runs `p40_bunny` /
`p40_dragon` are the same configuration as `g1a_*` (the new recipe through
`hyde06_env.sh`): silIoU 0.9628 vs 0.9614 (bunny), 0.9555 vs 0.9594 (dragon); chamfer
0.1196 vs 0.1196, 0.1235 vs 0.1235; det F min 0.766 vs 0.758, 0.736 vs 0.778. The IoU
spread of identical runs is therefore ±0.4 points (CUDA atomics, 2026-09-18 controls
agreed). Read against it: the relaxation's cost (−0.6 bunny, −1.2 dragon) is 1.5–3× the
spread; A's recovery (+0.5, +0.75) is 1–2×; the new recipe against the v8 recipe (+0.05
bunny, −0.9 dragon) is within 2×. Chamfer and det F, which do not move between the
twins, carry the shape conclusion.

## 5. Sources

Triangle Splatting arXiv 2505.19175; Triangle Splatting+ 2509.25122; 2D Triangle
Splatting 2506.18575; Incremental Online Scene Reconstruction by 3D Gaussian Triangulation
(ECCV 2026) 2607.10690; SuGaR 2311.12775; 2D Gaussian Splatting 2403.17888; High-quality
Surface Reconstruction using Gaussian Surfels 2404.17774; Yu & Turk, TOG 2013; Kazhdan &
Hoppe, TOG 2013; Öztireli, Guennebaud, Gross, CGF 2009; SplatSurf, The Visual Computer 2026.
