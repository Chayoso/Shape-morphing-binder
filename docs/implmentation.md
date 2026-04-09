# Thin-Structure Failure in Silhouette-Heavy Physics-Guided Shape Optimization
## Implementation, Validation, and Paper Direction for a Premier-Tier Submission

## Status Update (2026-04-09)

The codebase now includes the following implemented mechanisms:
- decoupled Stage A/B/C pipeline
- worst-view-aware multi-view aggregation (`hard-max`, `top-k`)
- global coarse field correction with manual projection `g_u = W^T g_x`
- local residual refinement with active-mask-restricted support
- immediate mass-loss gate
- newly added **delayed local refinement schedule**
- newly added **bilateral paired ROI support** for local refinement

What is already validated:
- physics-only is clearly weaker than decoupled stronger correction
- hard-max reduces worst-view failure
- local residual refinement improves bunny-like thin-structure cases
- sharper splats and stronger edge losses do not help

What is still under validation today:
- whether **bilateral + delayed local refinement (BDLR)** materially improves bunny ear separation over one-sided or early local refinement

What is not yet part of the final method:
- depth supervision
- normal supervision
- two-stage structural gate

Current recommendation:
- treat **BDLR** as the last major algorithmic candidate
- keep depth as a weak optional follow-up only if BDLR alone is insufficient
- avoid expanding the method into a kitchen-sink loss stack

---

# Code-Aligned Implementation Specification

This section summarizes the **actual implemented system** in code, not the aspirational design.

## A. Core system identity

The current framework is best described as:

> **physics-guided particle morphing with a decoupled render-guided correction stage**

It is **not** a fully joint end-to-end optimizer and it is **not** a pure image-driven reconstruction system.

The implementation is centered around:
- `run.py`
- `utils/training_loop.py`
- `utils/deformation_field.py`

## B. Pipeline stages

### B.1 Stage A: physics rollout

Entry point:
- `run_single()` in `run.py`
- `run_episode()` in `utils/training_loop.py`

Behavior:
- initialize / advance the particle state using the existing differentiable MPM stack
- optimize the physics objective first
- extract the post-rollout particle state:
  - positions `x`
  - deformation gradient `F_e`
  - plastic buffer `Fp`

Important implementation detail:
- the physics stage is the **primary trajectory generator**
- render supervision is **not** injected directly into the physics optimizer in the main decoupled mode

### B.2 Observation branch

Entry point:
- `compute_multiview_gradients()` in `utils/training_loop.py`

Behavior:
- render the rollout state from multiple views with the differentiable Gaussian renderer
- compute per-view observation losses
- aggregate them into a multi-view objective
- backpropagate only through the renderer to obtain particle-space gradients `dL/dx`

Current observation terms:
- alpha BCE
- alpha IoU
- optional DT
- optional edge
- optional RGB

Current multi-view aggregation modes:
- mean
- weighted mean
- hard-max penalty
- top-k penalty

Current paper-facing default:
- `mv_hardmax_w > 0`
- `mv_topk_w = 0` in the cleanest main setting

### B.3 Stage B: field-space correction

Entry point:
- `render_correction()` in `run.py`
- `CoarseDeformationField` in `utils/deformation_field.py`

Behavior:
- construct a coarse grid field over the current particle state
- precompute trilinear interpolation weights once per correction cycle
- optimize field displacements instead of directly updating particles

Forward model:
- `x' = x + W u`

Backward model:
- `g_u = W^T g_x`

This is implemented explicitly in:
- `forward()` in `CoarseDeformationField`
- `backward()` in `CoarseDeformationField`

Regularization currently implemented in field space:
- smoothness loss on neighboring grid nodes
- L2 penalty on field displacement
- trust-region style displacement clipping

### B.4 Stage B-local: local residual refinement

Entry point:
- `_select_local_refine_region()` in `run.py`

Behavior:
- after the global coarse correction, optionally allocate a local refinement field
- select a high-error upper-body / thin-structure ROI from particle gradient magnitude
- optionally restrict the field with an `active_mask`

Current implemented local-refine controls:
- `grid_res`
- `num_iters`
- `lr`
- `w_smooth`
- `w_l2`
- `max_disp`
- `z_min_frac`
- `score_percentile`
- `bbox_padding`
- `active_margin`
- `max_extent_frac`
- `min_extent_frac`

Current implemented scheduling / symmetry options:
- delayed start via `start_ep` or `start_frac`
- bilateral paired ROI support
- mirror-symmetry projection in local field space

Mirror projection implementation:
- `project_mirror_symmetry()` in `utils/deformation_field.py`

Important interpretation:
- local refine is currently an **extension branch for thin-structure rescue**
- it should not be presented as the only thing making the method work

### B.5 Stage C: acceptance / commit

Entry point:
- `immediate_mass_loss_gate()` in `run.py`
- `apply_accepted_correction()` in `run.py`

Behavior:
- evaluate whether the corrected state causes an unacceptable jump in immediate end-layer mass loss
- accept or reject the correction before committing it back into the promoted simulation state

Current implementation facts:
- this is an **immediate consistency gate**
- it is **not** rollout-aware validation
- accepted states can optionally:
  - reset kinematics
  - absorb total deformation into the point cloud
  - zero / reset external plastic carryover

## C. Data and state flow

### C.1 State variables

Main state variables carried through the loop:
- particle positions `x`
- elastic / total deformation `F_e`
- plastic-like residual state `Fp`
- local field parameters `u`

### C.2 Gradient flow

The actual gradient path is:

1. rollout current particle state
2. render current state
3. compute particle-space observation gradient `dL/dx`
4. project gradient into field parameters with `W^T`
5. update field
6. apply corrected positions back to simulation only if accepted

This means the method is:
- differentiable through the render branch
- explicit / manual through the field projection
- decoupled from full simulation autograd

## D. Rendering stack

Current renderer role:
- generate multi-view alpha supervision
- optionally provide RGB / shading supervision
- provide observation gradients

Current practical status:
- alpha-centered supervision is the main reliable signal
- RGB exists but is not the main driver of results
- depth / normal are not yet part of the final validated method

Important caveat:
- any run where the renderer fails to initialize is **invalid** for method comparison
- such runs should not be used in tables or figures

## E. Current valid experimental branches

### E.1 Main valid story

The strongest validated method story is:
- physics-only baseline
- decoupled stronger correction
- hard-max multi-view aggregation

This already supports the core paper claim.

### E.2 Thin-structure branch

The currently valid thin-structure branch is:
- local residual refinement
- top-gradient ROI variants
- bilateral ROI variants
- delayed local refine variants

Best local candidate so far:
- `topgrad14_tight`

Interpretation:
- these variants improve the bunny ear case slightly
- they do not yet change the main quantitative story as strongly as decoupling + hard-max

### E.3 Invalid / provisional branch

The strong forced-symmetry bunny rescue branch is currently **provisional only**.

Reason:
- at least one earlier `bunny_force_sym` batch executed with renderer initialization failure
- those outputs are not valid for scientific comparison

Therefore:
- forced-symmetry rescue may still be useful for debugging / hero-case exploration
- but it is **not** part of the current validated method

## F. Current output and diagnostics

Per-episode logs currently include:
- `loss_physics`
- `alpha_mse`
- `loss_total_mv`
- `loss_total_obj_mv`
- `loss_bce_mv`
- `loss_iou_mv`
- `loss_dt_mv`
- `worst_view_bce`
- correction metrics
- local refine activity metrics
- field / motion diagnostics

PNG / visualization support currently exists for:
- primary render
- primary alpha
- worst-view alpha
- per-view visualizations
- point-cloud / gradient overlays

Video assembly:
- `tools/make_run_video.py`

Motion-check batch:
- `configs/bunny_motion_check.yaml`

## G. What is scientifically supported right now

The following claims are currently supported by code and runs:
- decoupled correction clearly improves multi-view observation loss over physics-only
- worst-view-aware aggregation reduces asymmetric failure better than equal-weight mean
- field-space correction is stable and practical
- local thin-structure refinement provides incremental gains

The following claims are **not yet** strongly supported:
- full symmetry restoration
- depth / normal as core validated contributions
- rollout-aware physical consistency guarantee
- local thin-structure branch as a universally dominant variant across shapes

## H. Graphics / optimization diagnosis

From a graphics and optimization perspective, the current system behaves like this:

- physics provides the low-frequency, physically coherent backbone
- rendering provides geometric residuals that correct visible mismatch
- the field parameterization keeps correction spatially coherent
- hard-view aggregation focuses the optimizer on stubborn views
- thin-structure failure is mainly a **capacity placement and basin selection** problem

This is why:
- decoupling helped a lot
- hard-max helped failure cases
- simply sharpening the image loss did not
- purely adding more loss terms is unlikely to be the clean final answer

## I. Recommended locked framing for writing

For the paper, the safest specification of the implemented framework is:

> PhysMorph-GS performs physics-guided particle morphing with a decoupled render-guided correction stage in deformation-field space. A differentiable renderer supplies particle-space observation gradients, which are manually projected into field parameters for coherent correction. The corrected state is committed only after an immediate consistency check.

This wording matches the current codebase and should remain the default description unless the implementation changes materially.

---

# Paper-Ready Method Summary

## Method Overview

PhysMorph-GS is a **physics-guided particle morphing framework with decoupled render-guided correction**.

At a high level, each episode consists of three stages:

1. **Physics rollout**
   - the differentiable MPM optimizer advances the particle state toward the target under physical constraints
   - this stage is responsible for globally coherent and physically plausible motion

2. **Render-guided field correction**
   - the current particle state is rendered from multiple views
   - observation loss is evaluated in image space
   - the resulting particle-space gradient is projected into a deformation field
   - the field is optimized and then reapplied to particles

3. **Consistency check and commit**
   - the corrected state is committed only if it does not significantly degrade immediate physical consistency

This structure makes rendering feedback usable **without directly destabilizing the physics optimizer**.

## Mathematical view

Let:
- `x_t` be the current particle positions
- `Phi(.)` be one episode of physics rollout
- `u` be deformation-field parameters
- `W` be the trilinear interpolation operator
- `R(.)` be the differentiable renderer
- `L_obs` be the multi-view observation loss

Then the framework can be written as:

```text
x_phys = Phi(x_t)
x_corr = x_phys + W u*
u* = argmin_u L_obs(R(x_phys + W u)) + Omega(u)
x_{t+1} = Gate(x_phys, x_corr)
```

The corresponding gradient mapping is:

```text
g_x = dL_obs / d(x_phys + W u)
g_u = W^T g_x
```

This manual projection is the key mechanism that makes field-space correction practical.

## Algorithm Box

```text
Algorithm 1: PhysMorph-GS

Input:
  initial particle state x_0
  physical simulator Phi
  differentiable renderer R
  target multi-view observations

for each episode t do
  1. Physics rollout:
       x_phys <- Phi(x_t)

  2. Multi-view rendering:
       render x_phys from V views
       compute observation loss L_obs
       compute particle-space gradient g_x

  3. Global field correction:
       initialize coarse deformation field u
       project gradient to field space: g_u = W^T g_x
       update u with smoothness/L2 regularization
       obtain corrected state x_global = x_phys + W u

  4. Optional local residual refinement:
       select high-error ROI
       optimize a local field within the ROI
       produce x_local

  5. Acceptance:
       if immediate consistency gate passes:
           commit corrected state
       else:
           keep x_phys

  6. Advance episode:
       x_{t+1} <- accepted state
end for
```

## What each component is responsible for

### Physics rollout
- global shape evolution
- physical plausibility
- material coherence

### Field-space correction
- observation-driven geometric refinement
- coherent non-local deformation
- stable alternative to particle-wise direct updates

### Hard-view multi-view aggregation
- prevents average-view optimization from ignoring stubborn views
- reduces asymmetric local minima

### Local residual refinement
- allocates additional capacity to thin or high-error regions
- acts as a targeted extension, not the main engine of the method

### Immediate gate
- filters unsafe corrected states cheaply
- serves as practical consistency control rather than full physical validation

## Final implementation stance

The current codebase should be described as:

> a decoupled physics-plus-render optimization framework in which physics generates the coarse physically plausible trajectory, and rendering supplies field-space residual corrections for geometric alignment.

That is the most accurate and defensible description for writing.

# 0. Executive Summary

## Problem setting
We currently have a physics-guided shape optimization pipeline with a decoupled correction stage:
- **Stage A**: physics rollout
- **Stage B**: observation-driven deformation field correction
- **Stage C**: immediate acceptance gate

The current system already establishes several strong empirical findings:
- physics-only is weaker than decoupled stronger correction
- hard-max view aggregation reduces worst-view failure
- local residual refinement improves bunny-like cases
- sharper splats and stronger edge losses do **not** help
- the remaining bottleneck is **not simply stronger supervision**, but how to allocate capacity and route gradients for **thin structures**

## Core diagnosis
The failure case is not a generic shape mismatch. It is a **thin-structure local minimum problem**:
- silhouette supervision underconstrains front/back and left/right separation
- thin protrusions such as bunny ears require **structural separation**, not just boundary agreement
- current field projection and local ROI selection may amplify asymmetric updates
- early local refinement likely locks optimization into a suboptimal basin

## Main recommendation
The highest-value next step is **not** to broadly expand the method.  
Instead, the method should be refined in a focused and defensible way:

### Recommended final algorithmic upgrade
1. **Keep the overall pipeline unchanged**
2. **Add depth as a weak auxiliary or control signal, not as a major new training objective**
3. **Introduce bilateral paired ROI refinement**
4. **Delay local residual refinement until after global alignment stabilizes**
5. **Optionally soften the gate into a two-stage structural acceptance criterion**

## Research stance
For a premier-tier paper, the strongest position is:
- avoid looking like the method only wins by stacking more losses
- show that thin-structure failure is a **capacity placement + optimization schedule** problem
- propose a targeted algorithmic fix that is minimal, principled, and well-validated

---

# 1. Current Pipeline and What It Already Proves

## 1.1 Current structure
The present pipeline already has a good paper backbone:

### Target generation
- multi-view cameras are created in `run.py`
- target alpha / distance-transform style maps are created in `rendering_utils.py`
- supervision is currently alpha-centered

### Stage A: Physics rollout
- the physics optimizer evolves the state forward
- rollout provides the physically plausible baseline trajectory

### Observation loss / gradient
- observation gradients are computed via differentiable rendering
- multi-view aggregation supports mean / weighted / hard-max / top-k
- alpha loss already includes BCE + IoU + DT + optional edge

### Stage B: Field correction
- global coarse field handles large-scale correction
- local residual field refines ROI regions
- field projection uses the standard projected gradient form:
\[
g_u = W^\top g_x
\]

### Stage C: Acceptance gate
- updates are accepted only if they pass an immediate mass-loss gate
- this is a consistency filter, not a rollout-aware structural validator

## 1.2 What is already known empirically
This is important because it narrows the true research question.

### Already established
- decoupled correction is valuable
- hard-max aggregation helps failure cases
- local residual refine matters
- stronger edge/contour sharpening is not the answer
- the remaining issue is concentrated in thin, separated, high-curvature structures

This means the paper should **not** be framed as “more losses help.”  
It should be framed as:

> Even with a strong decoupled physics + rendering correction pipeline, thin structures remain a systematic failure mode because the issue lies in gradient routing, local capacity allocation, and optimization timing.

That is a much stronger research story.

---

# 2. Why Simply Adding Depth May Not Solve the Problem

---

## 2.1 The key question
The right question is **not**:
> “Can depth be added?”

The right question is:
> “Does depth provide a genuinely new optimization signal after projection into the current correction space?”

If physics rollout already aligns depth-like volumetric structure reasonably well, then adding depth may produce only marginal gains.

## 2.2 Two likely scenarios

### Case A: depth gradient is mostly aligned with alpha gradient
Then depth is not new information, only stronger weighting.

Formally:
\[
\nabla_x L_{\text{depth}} \approx c \nabla_x L_{\alpha}
\]

In that case, depth is mostly scaling a direction the optimizer already follows.

### Case B: depth contains useful structure, but projection destroys it
Even if depth carries useful thin-structure cues, they may be attenuated after:
- projection into deformation basis
- active-mask restriction
- asymmetric ROI selection
- early local overfitting
- acceptance gate rejection

Then the bottleneck is **not supervision**, but **optimization geometry**.

## 2.3 Research implication
Depth is still worth using, but likely in one of these forms:
- weak auxiliary supervision
- ROI proposal signal
- view selection score
- gate refinement cue
- diagnosis tool

Depth should **not** be presented as the main conceptual fix unless it clearly changes the failure mode.

---

# 3. True Failure Mode: Thin Structure = Capacity Placement + Optimization Schedule

---

## 3.1 Why silhouette-heavy supervision fails
Silhouette constraints are inherently ambiguous:
- they match occupancy at the boundary
- they do not strongly couple hidden separation
- they underconstrain bilateral part separation

For bunny ears, the target is not just “one high protrusion” but “two laterally separated thin protrusions.”  
Silhouette alone allows incorrect but locally stable explanations.

## 3.2 Why thin structures are harder than bulk structures
Thin structures amplify all of the following:
- basis smoothness mismatch
- view ambiguity
- local asymmetry
- gradient dilution
- gate conservativeness
- premature local commitment

Thus, the actual optimization problem is:
- not just finding the shape
- but allocating enough **correction capacity** to the structurally fragile region
- at the **right time**
- under a **stable and symmetric update rule**

## 3.3 The most likely concrete failure in the current system
A very plausible current failure sequence is:

1. physics rollout gets close globally
2. one ear gets slightly larger error signal
3. ROI selection activates only that side
4. local residual refines the wrong asymmetric basin
5. acceptance gate preserves only immediately safe moves
6. the optimization never escapes into the two-ear basin

This is a much richer and more publishable explanation than “alpha is weak.”

---

# 4. Recommended Algorithmic Direction

---

## 4.1 Design principle
For a premier-tier paper, the final algorithmic addition should be:
- minimal
- interpretable
- targeted at the actual failure mode
- strong enough to produce measurable gains
- small enough to preserve a clean method story

## 4.2 Final recommended method
The most defensible algorithmic improvement is:

### **Bilateral Delayed Local Refinement (BDLR)**  
with optional weak depth assistance.

This should be the main upgrade.

---

## 4.3 Component 1: Bilateral paired ROI refinement

### Motivation
Current ROI selection based on high-error seeds can activate only one side of a symmetric thin structure.

For bunny ears, this is dangerous because:
- one side receives extra capacity
- the other side remains frozen or under-modeled
- asymmetry gets amplified

### Proposed change
When a high-error ROI is selected, automatically generate its mirrored counterpart in object-canonical coordinates.

### Mechanism
1. compute residual map in image or projected object space
2. identify candidate thin-structure ROI
3. map ROI to canonical object coordinates
4. reflect ROI across the known bilateral axis
5. allocate local refinement capacity to **both** regions

### Result
This does **not** force exact symmetry.  
It forces **balanced opportunity for refinement**, which is much more defensible.

### Why this is strong
This addresses the actual bottleneck:
- not loss weakness
- but asymmetric capacity placement

That is a real algorithmic contribution.

---

## 4.4 Component 2: Delayed local residual refinement

### Motivation
Early local refinement is dangerous because the global structure is still unstable.  
Noisy multi-view residuals can pull the optimization into an early wrong basin.

### Proposed schedule
Use local residual only after coarse global alignment stabilizes.

### Example schedule
- Epochs 1–10: global coarse only
- Epochs 11–20: enable bilateral local refine
- Epochs 21–40: continue local refine with hard-max / top-k emphasis

### Adaptive alternative
Instead of fixed epochs, trigger local refinement when:
- global alpha IoU passes threshold
- coarse correction norm decreases below threshold
- view disagreement stabilizes

### Why this matters
This improves optimization trajectory rather than just objective design.  
That is exactly the kind of subtle but important contribution premier-tier reviewers appreciate.

---

## 4.5 Component 3: Depth as a control signal, not a major loss

### Motivation
Depth may still be valuable, but most likely as a **structural signal** rather than a dominant objective.

### Recommended uses
#### ROI proposal
Use:
\[
R = w_\alpha |\Delta \alpha| + w_d |\Delta D|
\]
to better localize structurally ambiguous thin regions.

#### View scoring
When using hard-max or top-k, prefer views with strong thin-structure depth mismatch.

#### Gate assistance
Use depth improvement inside candidate thin regions to help determine whether a local correction is structurally promising.

### Optional weak auxiliary loss
A low-weight depth term can be included:
\[
L = L_\alpha + \lambda_d L_{\text{depth}}
\]
but only with small \(\lambda_d\).  
This should be positioned as stabilization or tie-breaking, not the conceptual centerpiece.

### Recommendation
Use depth, but do not build the paper around “we added depth.”

---

## 4.6 Component 4: Optional two-stage gate
This is optional but potentially valuable.

### Current limitation
The immediate gate may reject corrections that are temporarily imperfect but move the state toward a better structural basin.

### Proposed replacement
#### Stage 1: safety filter
Reject catastrophic updates:
- severe mass loss
- instability spikes
- invalid deformations

#### Stage 2: structural promise filter
Allow updates that improve:
- ROI-local alpha consistency
- thin-structure depth ordering
- short-horizon surrogate rollout score

### Rationale
Thin-structure corrections often require accepting moves that are not immediately best under the original metric.

### Risk
This adds complexity.  
Only include if it clearly helps and does not destabilize the story.

---

# 5. What Not to Do

For a strong paper, avoiding the wrong expansions is just as important.

## 5.1 Do not keep sweeping edge losses
Already observed to be weak or harmful.

## 5.2 Do not broaden into RGB / shading unless unavoidable
This changes the story from structural shape correction into appearance supervision.

## 5.3 Do not add many priors at once
If symmetry prior, depth, normals, stronger edge, more cameras, and gate changes all appear together, the paper becomes unconvincing.

## 5.4 Do not turn this into a kitchen-sink method
Premier-tier reviewers usually punish methods that look like:
- loss stacking
- engineering accumulation
- case-specific tricks without diagnosis

The method should remain tight.

---

# 6. Validation Plan: What Must Be Proven

For a premier-tier target, the experiments need to answer **why** the method works, not just whether metrics improve.

---

## 6.1 Central empirical claim
The main empirical claim should be:

> Thin-structure failure in silhouette-heavy physics-guided optimization is primarily caused by asymmetric capacity placement and premature local refinement, and can be alleviated by bilateral delayed local refinement with weak structural cues.

This is a strong and publishable claim.

---

## 6.2 Minimum experiment matrix

### Baselines
1. Physics-only
2. Physics + decoupled global correction
3. Physics + global + local correction
4. Physics + global + local + hard-max
5. Proposed BDLR
6. Proposed BDLR + weak depth assistance

### Optional diagnostic variants
7. Early local refine
8. One-sided ROI refine
9. Bilateral ROI without delay
10. Delay without bilateral pairing

These ablations are critical.  
They isolate the mechanism.

---

## 6.3 Shapes to include
At least three groups:

### Group A: easy / bulky shapes
- smooth, globally coherent objects
- where baseline already does reasonably well

Purpose:
- show your method does not harm easy cases

### Group B: thin protrusion shapes
- bunny-like ears
- horns
- antennae
- branched parts

Purpose:
- show where the method matters most

### Group C: asymmetric thin structures
- objects with a single thin protrusion
- or asymmetric thin branches

Purpose:
- show the method is not limited to symmetry toy cases

This is important because otherwise reviewers may say the method is just rescuing bunny.

---

## 6.4 Metrics
Use both standard and failure-specific metrics.

### Global metrics
- silhouette IoU
- alpha BCE / DT loss
- final 3D Chamfer / distance metric
- physics consistency score
- worst-view metric

### Thin-structure-specific metrics
This is very important.

Possible options:
- ear-region IoU
- thin-part centerline distance
- skeleton distance
- connected-component agreement
- part separation score
- minimum neck thickness error
- bilateral correspondence error

At least one thin-structure-specific metric is strongly recommended.

Otherwise the main contribution is hard to quantify.

---

## 6.5 Diagnostics reviewers will love
These matter a lot.

### A. Gradient agreement analysis
Compare:
- \(\nabla_x L_\alpha\)
- \(\nabla_x L_{\text{depth}}\)
- projected field gradients

Report:
- cosine similarity
- norm ratios
- spatial concentration on thin regions

This directly supports your claim that the issue is not just adding more losses.

### B. ROI activation analysis
Visualize:
- one-sided ROI
- bilateral ROI
- correction support maps
- epoch-wise local refine activation

This makes the capacity argument concrete.

### C. Basin / trajectory analysis
Show optimization trajectories:
- early local refine falls into asymmetric local minimum
- delayed bilateral refine reaches better separated structure

Even one strong qualitative plot of this can be very persuasive.

### D. View sensitivity
Compare mean vs hard-max vs top-k vs proposed view scoring.  
This supports the thin-structure failure argument.

---

# 7. Implementation Plan

---

## 7.1 Priority order
This is the recommended implementation order.

### Priority 1
**Bilateral ROI generation**
- pair selected ROI with mirrored ROI
- equalize refinement capacity across both sides

### Priority 2
**Delayed local refine schedule**
- turn off local residual early
- enable after coarse alignment stabilizes

### Priority 3
**Weak depth-assisted ROI / view scoring**
- use depth disagreement for structural localization
- not necessarily for heavy loss weighting

### Priority 4
**Optional weak depth loss**
- only if it measurably helps
- keep weight low

### Priority 5
**Optional gate refinement**
- only if current gate is clearly blocking structurally good updates

---

## 7.2 Minimal viable implementation
If time is tight, the minimum viable strong version is:

1. bilateral ROI pairing
2. delayed local refine
3. hero bunny run
4. multi-shape batch with fixed settings

This is likely the best risk-reward tradeoff.

---

## 7.3 Pseudocode sketch

```python
# Stage A: physics rollout
state_T = physics_rollout(state_0, params)

# Observation residuals
obs = render_multiview(state_T)
alpha_residual = compute_alpha_residual(obs, target_alpha)
depth_residual = compute_depth_residual(obs, target_depth)

# Global correction always available
g_global = project_global(alpha_residual, maybe_depth_residual)

# Local refinement only after delay
if epoch >= local_refine_start:
    roi_seed = propose_roi(alpha_residual, depth_residual)
    roi_pair = mirror_roi(roi_seed, canonical_symmetry_axis)
    local_mask = build_union_mask(roi_seed, roi_pair)
    g_local = project_local(obs_grad, local_mask)
else:
    g_local = 0

g_total = combine(g_global, g_local, aggregation_mode)

candidate_update = apply_field_update(state_T, g_total)

if passes_safety_gate(candidate_update):
    if passes_structural_gate(candidate_update, thin_region_metrics):
        accept(candidate_update)
    else:
        reject(candidate_update)
else:
    reject(candidate_update)

    8. Paper Framing for a Premier-Tier Venue
8.1 The wrong framing

Do not frame the paper as:

“we added depth”
“we added symmetry”
“we used more losses”
“we tuned ROI better”

That sounds incremental.

8.2 The right framing

Frame the work as:

In physics-guided inverse shape optimization with silhouette-heavy observation losses, thin structures constitute a distinct failure mode not resolved by stronger supervision alone. We show that the core bottleneck lies in asymmetric correction capacity and premature local refinement. We propose a minimal structural refinement strategy that delays local updates and allocates refinement bilaterally, leading to improved recovery of thin separated structures while preserving global physical plausibility.

This is much stronger.

8.3 Suggested contribution bullets

A good final paper could claim:

Failure analysis: Identify thin-structure recovery as a systematic failure mode of silhouette-heavy decoupled physics-guided optimization.
Mechanistic diagnosis: Show that the failure arises from asymmetric local capacity allocation and premature local refinement, not simply insufficient loss strength.
Method: Propose bilateral delayed local refinement, optionally guided by weak structural cues such as depth.
Validation: Demonstrate improved recovery on thin-structure benchmarks with strong ablations, trajectory analysis, and thin-part-specific metrics.

These are publication-quality claims.

9. Experimental Section Structure

A recommended experiments section:

9.1 Setup
pipeline details
render settings
view aggregation modes
local refine schedule
gate definition
datasets / shapes
9.2 Main comparison
physics-only
decoupled baseline
proposed method
9.3 Thin-structure failure analysis
bunny and other thin-structure objects
qualitative comparisons
failure taxonomy
9.4 Ablation
bilateral vs one-sided ROI
early vs delayed local refine
no depth vs weak depth-assisted
mean vs hard-max vs top-k
optional gate variant
9.5 Diagnostic analysis
gradient cosine
ROI maps
trajectory comparison
part-specific metrics
9.6 Runtime / complexity

Keep this honest and clean.
If the added method is lightweight, emphasize that.

10. What Reviewers Are Likely to Ask

Prepare for these.

Q1. Why is depth not the main method?

Answer:

because the issue is not merely missing geometric information
gradient diagnosis shows the bottleneck lies in routing and local capacity allocation
depth is useful, but secondary
Q2. Is bilateral refinement just a bunny-specific trick?

Answer:

no, it is a general strategy for preventing asymmetric local over-commitment in thin-part recovery
validated on both symmetric and asymmetric thin structures
the paired ROI allocation is a refinement scheduling mechanism, not a semantic object prior
Q3. Why not use stronger geometric losses?

Answer:

stronger edge and contour terms do not address structural separation
the issue is not sharper silhouette matching but escaping local minima in thin-part formation
Q4. Is the improvement just from more parameters?

Answer:

ablations with early local refine and one-sided local refine show that capacity alone is insufficient
timing and bilateral placement are the key factors

These answers should be supported experimentally.

11. Concrete Final Recommendation
Best practical path

If the goal is a strong paper without over-expanding the method, the best path is:

Final method to implement
keep the overall decoupled physics + correction pipeline
add bilateral paired ROI
add delayed local residual refinement
optionally add weak depth-assisted ROI / view scoring
optionally add low-weight depth auxiliary loss
only refine the gate if clearly necessary
Best validation plan
hero bunny run at full budget
several thin-structure shapes
several non-thin controls
strong ablations that isolate bilateral pairing and delay
thin-structure-specific metrics
gradient / trajectory diagnostics
Best paper story

This is not a paper about “adding more losses.”
It is a paper about:

understanding and fixing thin-structure failure in silhouette-heavy physics-guided optimization through better local refinement timing and capacity allocation.

That is the cleanest and strongest research direction.

12. Final One-Paragraph Takeaway

The current bottleneck is best understood as a thin-structure optimization failure, not a simple supervision deficit. Because physics rollout already captures much of the coarse volumetric behavior, adding depth alone is unlikely to be a decisive solution. The more principled and impactful fix is to modify how and when correction capacity is allocated: local refinement should be delayed until global alignment stabilizes, and thin-structure ROIs should be activated bilaterally to prevent asymmetric basin capture. Weak depth cues can still be useful, but mainly as structural guidance for ROI selection, view scoring, or stabilization rather than as the central methodological contribution. For a premier-tier submission, the strongest paper will be one that clearly diagnoses this failure mode, introduces a minimal but targeted algorithmic fix, and validates it with rigorous ablations, trajectory analysis, and thin-structure-specific metrics.
