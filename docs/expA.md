# Experiment Plan After Exp B: Surface-Aware Gradient Coherence Study and Revised Implementation Roadmap

## Goal

The current evidence suggests that the main problem is **not only gradient scaling**, and **not only direct \(F\)-space mismatch**, but also that:

> **raw render gradients are too noisy when backprojected to the full volumetric particle set.**

Therefore, the next step is to test whether render guidance becomes more meaningful when restricted to **surface-aware / visually relevant particle subsets**.

This document summarizes:

1. **what experiment to run next**,  
2. **how to implement it**,  
3. **what values and plots to focus on**,  
4. **how the outcome should affect the next implementation decision**.

---

# 1. Current Situation

## Already established

### Exp 1: fixed-norm injection
- removed the physics-dependent magnitude collapse,
- but still produced a **small net-negative effect**,
- therefore scaling alone is **not** the main fix.

### Exp 4: raw terminal-state render gradient visualization
- raw particle-wise render gradients are highly incoherent,
- strong spikes appear only on a few particles,
- neighboring particles have nearly unrelated directions.

### Exp B: smoothing study
Best result:

- `clip_p95 + knn_k32`
- coherence improved by `3.5×`
- but still only reached:

\[
\text{coherence} = 0.0878
\]

which is still below the noisy-regime threshold (`0.1`).

Also, clipping suppresses spikes but also collapses the usable gradient magnitude.

## Interpretation

This strongly suggests:

> **simple smoothing is not enough**.

The likely deeper issue is that the render gradient is being interpreted over the **wrong support**:

- full volumetric particle set,
- instead of the visually relevant subset that actually contributes to the silhouette.

---

# 2. Main Hypothesis for the Next Experiment

The next hypothesis to test is:

> Render gradients may still be meaningful, but only on **surface-aware or visually contributing particles**, not on the full volumetric particle set.

If this is true, then the next method should build target-state guidance only from that subset.

If this is false, then even a surface-aware particle-level target may be too noisy, and the next step should be a coarser guidance representation.

---

# 3. Next Experiment: Exp A (Surface-Aware Subset Analysis)

## Main question

Does gradient coherence improve when the analysis is restricted to particles that are actually relevant to rendering?

---

## 3.1 Subsets to test

I should test at least the following three subsets.

### Subset A: visible particles
Particles that actually contribute to the current rendered alpha support for the selected view.

### Subset B: high-contribution particles
Particles with large contribution to alpha / projected support.  
For example:
- top \(p\%\) by alpha contribution,
- top \(p\%\) by visibility weight,
- top \(p\%\) by projected influence.

### Subset C: surface-shell particles
Particles near the object surface, estimated from geometry or neighborhood structure.  
For example:
- low local density,
- shell identified from neighbor count,
- outer layer based on distance-to-core heuristic.

---

## 3.2 Optional subsets if easy to implement

These are useful but not mandatory for the first pass.

### Subset D: boundary-relevant particles
Particles whose projections lie near the silhouette mismatch region or boundary band.

### Subset E: union/intersection sets
For example:
- visible ∩ shell
- high-contribution ∩ shell

These may produce cleaner guidance than any single subset.

---

# 4. What to Compute for Each Subset

For each subset, compute the following.

---

## 4.1 Raw coherence

Compute spatial coherence of the raw render gradient restricted to that subset.

### Why
This tells me whether the full-volume incoherence is caused by irrelevant particles.

---

## 4.2 Smoothed coherence

Apply the same smoothing variants used in Exp B, but only on the selected subset.

Recommended variants:

- raw
- `clip_p95`
- `knn_k8`
- `knn_k32`
- `clip_p95 + knn_k8`
- `clip_p95 + knn_k32`

### Why
A subset that is already more meaningful may become usable after modest smoothing, even if the full volume does not.

---

## 4.3 Spike ratio

Measure something like:

\[
\text{spike ratio} = \frac{\max \|g_i\|}{\text{mean} \|g_i\|}
\]

or equivalent.

### Why
This tells me whether the subset reduces the problem of extreme sparse spikes.

---

## 4.4 Gradient norm distribution

Plot or log:

- per-particle gradient norm histogram,
- mean norm,
- median norm,
- max norm,
- p95 norm.

### Why
I need to know whether smoothing improves coherence only by killing the signal, or whether it preserves usable magnitude.

---

## 4.5 Vector field visualization

For each important subset, visualize:

- particle positions,
- render gradient vectors,
- target silhouette overlay if possible.

### Why
Numeric coherence alone is not enough.  
A field with coherence slightly below `0.1` might still be qualitatively meaningful, while a field slightly above `0.1` may still be unusable.

---

# 5. What Values to Focus On

These are the key numbers to focus on.

---

## 5.1 Spatial coherence

This is the primary diagnostic.

### Interpretation
- `< 0.1` → noisy regime
- around `0.1` → borderline
- `> 0.1` → potentially usable
- clearly higher than full-volume result → strong evidence that support selection matters

### Main question
Does any subset cross or approach the usable regime after smoothing?

---

## 5.2 Spike ratio

This is the second most important diagnostic.

### Interpretation
- very large spike ratio → guidance dominated by a few pathological particles
- moderate spike ratio with improved coherence → much more promising

### Main question
Does the subset reduce the spike domination problem?

---

## 5.3 Mean vs max gradient norm

This is important because clipping can improve coherence while also killing the signal.

### Main question
Is coherence improvement accompanied by total signal collapse, or does usable magnitude remain?

---

## 5.4 Qualitative global directionality

This may be even more important than one numeric metric.

### Main question
Do the vectors actually point toward missing silhouette support regions (e.g. ear-like regions), or do they still look random after subset restriction?

---

# 6. Recommended Experimental Procedure

---

## Step 1. Full-volume baseline (already done)
Keep these for reference:

- full raw coherence,
- full smoothed coherence,
- full spike ratio,
- full vector field visualization.

This is the baseline that all subset experiments should be compared against.

---

## Step 2. Visible subset analysis
For visible particles only:

- compute raw coherence,
- compute smoothed coherence,
- compute spike ratio,
- visualize raw and smoothed fields.

### What I want to know
Does visibility restriction already recover meaningful structure?

---

## Step 3. High-contribution subset analysis
For top-contribution particles:

- repeat the same measurements.

Suggested variants:
- top 5%
- top 10%
- top 20%

### What I want to know
Does restricting to particles that matter most for alpha support improve guidance quality?

---

## Step 4. Surface-shell subset analysis
For shell particles:

- repeat the same measurements.

### What I want to know
Does a geometry-defined surface subset behave better than a purely visibility-defined subset?

---

## Step 5. Compare subsets
Make a summary table like:

| subset | variant | coherence | improvement vs full raw | spike ratio | mean norm | max norm |
|---|---:|---:|---:|---:|---:|---:|

### What I want to know
Which subset gives the best tradeoff between:
- higher coherence,
- lower spike ratio,
- non-vanishing gradient magnitude.

---

# 7. Decision Rules After Exp A

After Exp A, I should decide the next implementation path using the following logic.

---

## Case 1. Surface-aware subset gives clearly better coherence
For example:
- coherence approaches or exceeds `0.1`,
- spike ratio drops substantially,
- vector field becomes interpretable.

### Conclusion
Proceed with a **surface-aware revised Option A**:
- build guidance only on that subset,
- smooth it,
- construct \(x_{\text{target}}\),
- inject only \(dL/dx\)-based attractor.

This is the best-case outcome.

---

## Case 2. Surface-aware subset helps somewhat, but still not enough
For example:
- coherence improves but remains weak,
- vectors are less random but still noisy.

### Conclusion
Proceed with a **coarse-guidance version** of Option A:
- subset restriction,
- then stronger smoothing / coarse projection,
- then target construction.

Examples:
- graph smoothing,
- low-frequency basis,
- voxel/grid accumulation.

---

## Case 3. Surface-aware subset still looks bad
For example:
- coherence remains very low,
- vectors remain random,
- spike problems persist.

### Conclusion
Particle-level target construction from alpha gradients is probably too noisy even on surface-aware supports.

Then the next method should not be particle-target-based.

Instead, move to:
- image-space distance-transform attractor,
- coarse support-deficit field,
- control lattice / low-frequency field,
- or another structured guidance representation.

---

# 8. Revised Implementation Plan Depending on Exp A

---

## If Exp A is positive: revised Option A (surface-aware particle target)

### Step 1
Compute raw terminal render gradient:

\[
g_x^{\text{raw}} = \frac{\partial L_{\text{render}}}{\partial x_T}
\]

### Step 2
Mask to the best-performing subset:

\[
g_x^{\text{subset}} = \text{MaskSubset}(g_x^{\text{raw}})
\]

### Step 3
Apply smoothing / clipping on the subset:

\[
g_x^{\text{guide}} = \text{SmoothClip}(g_x^{\text{subset}})
\]

### Step 4
Construct the target:

\[
x_{\text{target}} = x_T - \eta \hat g_x^{\text{guide}}
\]

### Step 5
Inject only the attractor gradient in \(x\)-space:

\[
g_{\text{attr}} = 2\lambda_{\text{attr}}(x_T - x_{\text{target}})
\]

with:

- \(dL/dF = 0\)

---

## If Exp A is weak: revised Option A (coarse guidance field)

Instead of building the target directly per particle:

### Step 1
Accumulate render gradient onto a coarser field:
- grid,
- voxel lattice,
- low-frequency basis,
- graph-reduced representation.

### Step 2
Smooth in that coarse space.

### Step 3
Map the coarse field back to particles.

### Step 4
Construct \(x_{\text{target}}\) from the coarse guide.

This would be the next escalation if subset restriction alone is not enough.

---

# 9. What Not to Focus On

At this stage, I should **not** spend time on:

- further direct \(F\)-space injection tuning,
- larger fixed-norm alpha sweeps,
- raw particle-level Option A without subset selection,
- complicated heuristics before checking subset coherence.

The biggest uncertainty right now is not a hyperparameter.  
It is whether the usable signal exists on a meaningful visual support subset.

---

# 10. Recommended Deliverables From Exp A

After running Exp A, I should prepare:

### Numeric summary
A table with:
- subset,
- smoothing variant,
- coherence,
- improvement factor,
- spike ratio,
- mean / max norm.

### Visual summary
At least:
- full-volume raw field,
- best full-volume smoothed field,
- best subset raw field,
- best subset smoothed field.

### Short interpretation
A 3–5 sentence conclusion answering:

1. Does support-aware selection help?
2. Which subset works best?
3. Is particle-level target construction still viable?
4. Should the next step be surface-aware Option A or coarse-field Option A?

---

# 11. Paper-Oriented Interpretation

If Exp A works, the story becomes much stronger:

> Raw render gradients are incoherent when interpreted over the full volumetric particle set, because silhouette supervision constrains only partial visible support. Restricting guidance to visually relevant surface-support particles substantially improves coherence, enabling the construction of a meaningful target-state attractor.

If Exp A does not work, that is still useful:

> Even after support-aware restriction, particle-wise render gradients remain too noisy for direct target construction, motivating a coarser guidance representation rather than raw particle-level terminal guidance.

So Exp A is a high-value experiment in either case.

---

# 12. Short Final Summary

The next experiment should test whether render gradients become usable when restricted to surface-aware or visually relevant particle subsets.

The most important things to focus on are:

1. **spatial coherence**,  
2. **spike ratio**,  
3. **mean vs max gradient magnitude**,  
4. **qualitative directionality toward missing silhouette support**.

If a subset gives meaningfully better coherence, then implement a **surface-aware revised Option A**.  
If not, move to a **coarse guidance representation** instead of raw particle-level target construction.

---

# One-Sentence Summary

Run Exp A by evaluating render-gradient coherence on visible, high-contribution, and surface-shell particle subsets, compare raw and smoothed versions, and use coherence + spike ratio + qualitative field structure to decide whether the next implementation should be a surface-aware particle-level Option A or a coarser guidance-field version.