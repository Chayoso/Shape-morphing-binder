# Surface-Aware Volumetric Pivot
## Current Implemented Pipeline and Validation Notes

This document replaces the earlier speculative pivot note.

It now describes the **current implemented surface-aware pipeline** in the codebase, including:
- fixed surface extraction from volumetric particles
- split correction-shell vs render-shell masks
- shell-biased volumetric sampling
- surface-aware `dFc` / control-space guidance
- current validation branches and what they are showing

The central design is:

> **physics still evolves the full volumetric state, but rendering supervises a surface subset and injects guidance back into physics through `dFc`, not by directly overwriting the final positions.**

This is the version that should be treated as the current main exploratory branch.

---

# 1. Design Goal

The original decoupled field-correction pipeline improved global alignment, but it had two persistent issues:
- render supervision was too diffuse over the full particle set
- thin-structure bifurcation, especially bunny ears, still collapsed into a one-lobe solution

The surface-aware pivot changes the coupling structure:
- **volume** remains the physical carrier
- **surface** becomes the observation manifold
- **3DGS-style differentiable rendering** acts as a surface sensor
- **control-space guidance through `dFc`** becomes the main route for physics-side correction

This is no longer a pure field-position correction method.

---

# 2. Current System Identity

The current implementation should be described as:

> **surface-aware render-guided volumetric morphing with control-space guidance**

Role split:
- **MPM volume particles**: full physical state, mass carrier, and topology-like evolution
- **surface subset**: observation manifold used by rendering loss
- **renderer**: visibility-aware differentiable sensor producing alpha/depth gradients
- **`dFc` guidance**: mechanism that feeds render-derived information back into physics rollout

The key change from earlier stages of the project is that the main exploratory branch is now:
- not `surface gradient -> field -> position overwrite`
- but `surface gradient -> smoothed control penalty -> Adam updates dFc inside physics`

---

# 3. State Spaces

Let:
- `x in R^(3N)` be the full particle positions
- `S_corr subset {1..N}` be the fixed correction shell
- `S_render subset {1..N}` be the thicker render shell
- `x_render = P_render x` be the rendered particle subset
- `F, Fp` be the deformation and plastic state tensors
- `dFc` be the control-space deformation increment optimized in the physics loop
- `R(.)` be the differentiable renderer

The important distinction is:
- `S_corr` is thin and used for correction/control focus
- `S_render` is thicker and used for rendering stability and denser visual support

So the observation is:

```text
y_pred = R(P_render x)
L_obs  = L_alpha + w_depth L_depth + optional view aggregation penalties
```

The correction/control signal is derived from the rendered subset, but it is ultimately consumed by the physics optimizer through `dFc`.

---

# 4. Current Pipeline

## Stage 0. Shell-Biased Volumetric Sampling

Before simulation starts, the source and target point clouds are initialized with **nonuniform volumetric sampling**:
- dense shell near the surface
- coarse interior bulk

This is implemented in:
- [GeometryLoading.cpp](/home/chayo/Desktop/Shape-morphing-binder/DiffMPMLib3D/GeometryLoading.cpp)
- [GeometryLoading.h](/home/chayo/Desktop/Shape-morphing-binder/DiffMPMLib3D/GeometryLoading.h)
- [bind.cpp](/home/chayo/Desktop/Shape-morphing-binder/bind/bind.cpp)
- [physics_utils.py](/home/chayo/Desktop/Shape-morphing-binder/utils/physics_utils.py)

Current default shell-biased setup:
- `surface_points_per_cell_cuberoot = 6`
- `interior_points_per_cell_cuberoot = 4`
- `shell_thickness_cells = 1.5`

This yields:
- denser render support on thin structures
- a lighter interior carrier
- per-particle mass/volume adjusted to the local sampling density

This is not pure surface-only sampling.
It is **dense shell + coarse interior** sampling.

## Stage 1. Fixed Surface Reconstruction

After particle initialization, a fixed source surface mask is computed once using a voxel reconstruction.

Implemented in:
- [surface_utils.py](/home/chayo/Desktop/Shape-morphing-binder/utils/surface_utils.py)
- [run.py](/home/chayo/Desktop/Shape-morphing-binder/run.py)

Current default:
- `recon_resolution = 64`
- one-time reconstruction
- no dynamic re-extraction during optimization

Two masks are derived:
- `surface_mask`: correction shell
- `render_surface_mask`: render shell

The mask artifact is saved as:
- `surface_recon.npz`

This split is important:
- the correction shell remains relatively thin
- the render shell is intentionally thicker to avoid holes and sparse-shell artifacts

## Stage 2. Full Volumetric Physics Rollout

Physics still runs on the **full particle set**.

This remains a volumetric MPM pipeline:
- full particle state
- full stress/deformation update
- full grid transfer

The surface-aware pivot does **not** discard interior physics.

Instead:
- interior particles provide volumetric support and physical coherence
- shell particles dominate observation

## Stage 3. Surface-Aware Rendering

Rendering no longer consumes the full particle set.

Instead:
- only `render_surface_mask` is passed into the differentiable renderer
- target observations are generated as `alpha + depth`

Implemented in:
- [training_loop.py](/home/chayo/Desktop/Shape-morphing-binder/utils/training_loop.py)
- [rendering_utils.py](/home/chayo/Desktop/Shape-morphing-binder/utils/rendering_utils.py)

Current observation terms:
- BCE alpha
- IoU alpha
- depth
- optional multi-view hard-max / top-k terms

This changes the role of the renderer:
- it is not a state representation
- it is a differentiable surface sensor

## Stage 4. Surface-Aware Gradient Collection

Observation gradients are computed only for the rendered shell subset.

These gradients are:
- view-aware
- visibility-aware
- restricted to the current render shell

They are then scattered back to full particle indexing for downstream use.

Earlier branches used these gradients mainly for field-position correction.
The current main branch uses them to build a **control guidance penalty**.

## Stage 5. Control-Space Guidance Through `dFc`

This is the current core mechanism.

Implemented in:
- [control_guidance.py](/home/chayo/Desktop/Shape-morphing-binder/utils/control_guidance.py)
- [run.py](/home/chayo/Desktop/Shape-morphing-binder/run.py)

The pipeline is:

1. take the previous episode's surface-aware observation gradient
2. smooth and diffuse it over local neighbors
3. convert it into:
   - `dL/dx` penalty
   - `dL/dF` penalty
4. pass those penalties into the differentiable physics backward pass
5. let Adam optimize `dFc` inside the physics rollout

So the current main path is:

```text
surface render gradient
-> smoothed control penalty
-> injected (dL/dx, dL/dF)
-> Adam updates dFc
-> next physics rollout follows a render-guided trajectory
```

This is the main conceptual shift:
- the renderer no longer only corrects final positions
- it now biases the internal physics control space

## Stage 6. Multi-View Failure Emphasis

Because bunny ear failure is strongly view-dependent, the current pipeline supports:
- hard-max view weighting
- top-k view weighting
- ear-focus regional boost

Implemented in:
- [training_loop.py](/home/chayo/Desktop/Shape-morphing-binder/utils/training_loop.py)
- [run.py](/home/chayo/Desktop/Shape-morphing-binder/run.py)

Current exploratory ear branch enables:
- `mv_hardmax_w = 1.0`
- `mv_topk_w = 0.5`
- `mv_topk_k = 2`
- `ear_focus_boost = 1.0`

The goal is to stop the optimizer from averaging away the missing-ear view.

---

# 5. Why We Split Correction Shell and Render Shell

The earlier surface-aware branch used essentially one thin shell for both:
- rendering
- observation
- correction

That caused:
- visible holes
- shell sparsity artifacts
- weak depth support

The current pipeline separates them:

## Correction Shell

Thin shell, used for:
- control/correction focus
- ear-region boosting
- stable, sparse support for guidance

## Render Shell

Thicker shell, used for:
- denser splatting support
- better alpha continuity
- stronger depth consistency

Current typical fractions in shell-biased runs:
- correction shell: about `10%`
- render shell: about `18%`

This has already improved visual density substantially.

---

# 6. Why `dFc` Is the Main Route

The earlier position-correction branches improved alignment, but they mainly acted as a residual overwrite on `x`.

That is useful for:
- small geometric cleanup
- local residual correction

But bunny ear bifurcation behaves more like a **trajectory selection problem** than a final-position cleanup problem.

In this project, topology-like change is already possible in physics:
- hole-like structures appeared in earlier physics-only runs
- surface reconstruction can change connectivity as volume density evolves

Therefore the missing ears are not best treated as a last-step positional cleanup.
They should be treated as a **physics-space bifurcation problem**.

That is why the main surface-aware branch now targets:
- `dFc`
- not just `x_corrected`

Short version:

> **If the ears must emerge through the physics trajectory, render guidance must enter the control space, not only the final positions.**

---

# 7. Current Implemented Config Families

## 7.1 Surface-Aware Baselines

- [bunny_surface_aware.yaml](/home/chayo/Desktop/Shape-morphing-binder/configs/bunny_surface_aware.yaml)
- [bunny_surface_aware_debug.yaml](/home/chayo/Desktop/Shape-morphing-binder/configs/bunny_surface_aware_debug.yaml)

These validate:
- fixed surface extraction
- alpha+depth target generation
- render-shell rendering
- basic surface-aware observation path

## 7.2 dFc-Guided Surface-Aware Runs

- [bunny_surface_dfc_guided.yaml](/home/chayo/Desktop/Shape-morphing-binder/configs/bunny_surface_dfc_guided.yaml)
- [bunny_surface_dfc_guided_thick.yaml](/home/chayo/Desktop/Shape-morphing-binder/configs/bunny_surface_dfc_guided_thick.yaml)

These validate:
- control-space guidance through `dFc`
- thicker render shell
- improved alpha/depth stability

## 7.3 Shell-Biased Volumetric Runs

- [bunny_surface_dfc_guided_shellbiased_30.yaml](/home/chayo/Desktop/Shape-morphing-binder/configs/bunny_surface_dfc_guided_shellbiased_30.yaml)

This is the first serious full-length surface-aware volumetric branch:
- shell-biased sampling
- fixed surface-aware rendering
- control guidance through `dFc`
- 30 episodes

## 7.4 Ear-Focus Branch

- [bunny_surface_dfc_guided_shellbiased_earfocus_30.yaml](/home/chayo/Desktop/Shape-morphing-binder/configs/bunny_surface_dfc_guided_shellbiased_earfocus_30.yaml)

This is the current thin-structure rescue branch:
- shell-biased volumetric sampling
- surface-aware rendering
- `dFc` guidance
- reduced smoothing
- worst-view-aware aggregation
- explicit ear-region focus boost

This is currently the most aggressive attempt to force the optimizer out of the one-lobe basin.

---

# 8. Metrics and Diagnostics

The current pipeline should always be monitored with:

## Observation

- `loss_total_mv`
- `loss_total_obj_mv`
- `loss_hardmax_mv`
- `loss_topk_mv`
- `worst_view_bce`
- `loss_depth_mv`

## Physics / Control

- `loss_physics`
- `dFc_mean`
- `dFc_max`
- `J_min`
- `J_max`

## Surface-Usage Sanity

- `surface_particle_frac`
- `render_particle_frac`
- `render_surface_particle_frac`
- `control_guidance_active_frac`
- `control_guidance_focus_frac`

## Visualization

Always inspect:
- `render.png`
- `viz/epXXX_views.png`
- gradient heatmaps
- `dFc` norm overlays

For bunny ears, the most important failure signal is:
- one or two views remain catastrophically bad
- while average loss keeps decreasing

That means the optimizer is still taking the smooth one-lobe basin.

---

# 9. What Has Already Been Validated

## Confirmed Improvements

### Surface-aware rendering works

- fixed 64-grid surface reconstruction is stable
- alpha+depth target generation works
- rendering only the surface subset is viable

### `dFc` guidance is better than position-only surface correction

Short runs showed:
- lower `loss_total_mv`
- lower `loss_depth_mv`
- lower `loss_physics`

relative to the earlier surface-aware position branch

### Thick render shell helps

Splitting correction shell vs render shell:
- reduced visible holes
- improved early depth loss
- made surface rendering less sparse

### Shell-biased volumetric sampling helps

Compared to the thick-shell baseline, shell-biased runs showed:
- better global multi-view convergence
- better depth convergence
- better physics loss

This indicates the remaining issue is not just sparse shell rendering anymore.

## What Is Still Not Solved

Even with:
- surface-aware rendering
- thicker render shell
- shell-biased sampling
- `dFc` guidance

the main shell-biased run still converged to:
- a smoother global bunny
- but **not a clean two-ear split**

This means the remaining bottleneck is now:
- not particle scarcity alone
- not shell visibility alone
- but insufficient high-frequency bifurcation signal

That is why the current focus has shifted to:
- hard-max / top-k view emphasis
- ear-region boost
- less diffused control guidance

---

# 10. Current Working Interpretation

At this point the correct interpretation is:

> **Surface-aware volumetric morphing is working as a global alignment mechanism, but bunny ear emergence is still a control-space bifurcation problem.**

More concretely:
- shell-biased sampling improved support
- surface-aware rendering improved signal quality
- `dFc` guidance improved trajectory-level correction
- but the optimizer still prefers a smooth one-lobe local minimum unless the missing-ear views are explicitly emphasized

This is why the current main branch is the ear-focus shell-biased run, not the earlier field-position branch.

---

# 11. Recommended Next-Step Rule

The current decision logic should be:

1. keep the shell-biased volumetric + surface-aware + `dFc` pipeline as the main branch
2. use the ear-focus branch to test whether explicit thin-structure emphasis can break the one-lobe basin
3. if that succeeds, this becomes the hero pipeline
4. if not, keep shell-biased `dFc` guidance as the stable method and document bunny ear bifurcation as an open thin-structure limitation

In other words:

- the old position-correction path is now a validated baseline
- the current main bet is `surface-aware dFc guidance`
- the current rescue strategy is `ear-focus + hard-max/top-k`

---

# 12. Short Summary

The currently implemented surface-aware pipeline is:

1. initialize a shell-biased volumetric particle set
2. reconstruct a fixed 64-grid surface mask once
3. derive a thin correction shell and a thicker render shell
4. run full volumetric MPM physics
5. render only the thicker surface shell using alpha+depth supervision
6. compute surface-aware gradients
7. convert those gradients into smoothed `dL/dx` and `dL/dF` penalties
8. inject them into the differentiable physics optimizer so that Adam updates `dFc`
9. monitor multi-view, depth, and `J` stability
10. optionally bias the signal with hard-max / top-k weighting and ear focus

This is the pipeline that should be treated as current truth in the codebase.
