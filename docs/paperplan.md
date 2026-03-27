# Roadmap Before and Toward Option A (Target-State Proximal Guidance)

## Goal

The goal is to move toward **Option A** as the main method, where rendering supervision is converted into a **target-state preference**, and physics is responsible for finding a feasible trajectory toward that target.

However, before implementing Option A fully, I should first run a few diagnostic experiments to confirm **why direct injection fails** and **why a target-state formulation is justified**.

So the roadmap should be:

1. **run a small set of diagnostic experiments**,  
2. **implement the smallest viable version of Option A**,  
3. **evaluate whether Option A actually resolves the observed failure modes**.

---

# Phase 1. Diagnostic Experiments Before Option A

These experiments are not the final method. Their role is to establish:

- whether the failure is mostly due to scaling,
- whether the failure is mostly due to the choice of control space,
- whether render gradients are more meaningful in \(x\)-space than in \(F\)-space,
- and whether a target-state formulation is likely to help.

---

## Experiment 1. Fixed-norm direct injection

### Purpose
Test whether the current failure is partly caused by the fact that render injection is scaled by the physics gradient norm.

### Current bad rule
\[
g_{\text{inject}}
\propto
\|g_{\text{phys}}\|
\cdot
\frac{g_r}{\|g_r\|}
\]

### Test rule
\[
g_{\text{inject}} =
\alpha
\cdot
\frac{g_r}{\|g_r\|+\epsilon}
\]

where \(\alpha\) is a fixed injection strength.

### What to measure
- render loss decrease,
- silhouette improvement,
- physics loss convergence,
- stability,
- whether render guidance remains active late in optimization.

### Interpretation
- If this helps somewhat, then the previous scaling rule was indeed suppressing render supervision too much.
- If this still fails or becomes unstable, then the deeper issue is likely the control space itself rather than scaling alone.

### Why this experiment matters
This is important because it helps separate:

- **scaling failure**, from
- **representation/control-space failure**.

---

## Experiment 2. Compare gradient geometry in \(F\)-space vs \(x\)-space

### Purpose
Test whether render gradients are fundamentally more meaningful in \(x\)-space than in \(F\)-space.

### What to compute
- \(\cos(g_{\text{phys}}^F, g_{\text{render}}^F)\)
- \(\cos(g_{\text{phys}}^x, g_{\text{render}}^x)\)

Also inspect:
- gradient norms,
- spatial smoothness,
- spatial coherence of the render gradient field.

### What to visualize
- particle-wise \(dL_{\text{render}}/dx\),
- whether vectors point toward missing silhouette support regions,
- whether the field looks like global attraction or just local noise.

### Interpretation
- If \(x\)-space gradients are more coherent than \(F\)-space gradients, then this strongly supports Option A.
- If both are noisy, then Option A may still need smoothing or a coarse target representation.
- If only \(F\)-space looks bad, then the main issue is likely the injection space rather than alpha supervision alone.

### Why this experiment matters
This is probably the most important diagnostic experiment, because it tests the main hypothesis behind Option A:

> direct local deformation injection is the wrong space for silhouette guidance.

---

## Experiment 3. Quick force-space sanity check

### Purpose
Test whether render supervision behaves better in a higher-level control space than in direct \(F\)-space injection.

### Why this matters
This is not meant to become the final method. It is a fast way to check whether:

> lifting render supervision into a more physically meaningful control space helps.

### What to test
Use render guidance in force space and compare against direct \(F\)-space injection.

### What to measure
- silhouette improvement,
- physics convergence,
- stability,
- qualitative shape improvement,
- whether missing regions such as ear-like structures are easier to form.

### Interpretation
- If force-space behaves better than \(F\)-space, this is strong evidence that control-space choice is the core issue.
- If force-space still fails, then Option A may need smoothing, coarse targets, or stronger structural changes.

### Why this experiment matters
Even if force-space is not the final paper method, it is valuable supporting evidence that render supervision should not be applied directly in local deformation space.

---

## Experiment 4. Final-state render gradient inspection

### Purpose
Test whether the render gradient is more interpretable when viewed only at the terminal state \(x_T\).

### What to compute
\[
g_x = \frac{\partial L_{\text{render}}}{\partial x_T}
\]

### What to check
- whether the final-state render gradient points toward missing silhouette regions,
- whether it gives a plausible "mass should move here" signal,
- whether it is smoother and easier to interpret than full-trajectory gradients.

### Interpretation
- If the final-state gradient is meaningful, then the simplest Option A formulation is well justified.
- If it is too noisy, then Option A should likely include smoothing or a coarse target representation.

### Why this experiment matters
This directly tests whether a terminal-state proximal formulation is viable.

---

# What Phase 1 Should Establish

By the end of Phase 1, I want to know:

1. whether the failure is partly caused by scaling,
2. whether the deeper problem is the control space,
3. whether render gradients are more meaningful in \(x\)-space than in \(F\)-space,
4. whether a terminal-state target is a reasonable representation for render guidance.

If the evidence points in that direction, then Option A is no longer just a design guess. It becomes a justified next method.

---

# Phase 2. Implement the Smallest Viable Option A

The first version of Option A should be as simple and interpretable as possible.

The recommended first version is:

> **terminal-state proximal attractor guidance**

This means rendering supervision defines a preferred terminal configuration, and physics solves for a feasible trajectory toward it.

---

## Step 1. Compute render gradient in terminal-state position space

At the final simulation state \(x_T\), compute:

\[
g_x = \frac{\partial L_{\text{render}}}{\partial x_T}
\]

This is the render-driven shape guidance signal.

---

## Step 2. Convert that gradient into a pseudo target state

Define:

\[
x_{\text{target}} = x_T - \eta \,\hat g_x
\]

where \(\hat g_x\) is a normalized or clipped version of the gradient.

For example:

\[
\hat g_x = \frac{g_x}{\|g_x\|+\epsilon}
\]

or more safely:

\[
\hat g_x = \text{Clip}\left(\frac{g_x}{\|g_x\|+\epsilon}, \tau\right)
\]

### Why
This creates a small, local target proposal saying:

> “If the final state moved slightly in this direction, the silhouette would improve.”

The target is not meant to be a hard solution. It is only a preferred terminal state.

---

## Step 3. Add an attractor loss to the physics objective

Define:

\[
L_{\text{total}}
=
L_{\text{phys}}
+
\lambda_{\text{attr}}
\|x_T - x_{\text{target}}\|^2
\]

### Interpretation
- the render objective says **where to go**,  
- physics decides **how to get there**.

This avoids direct gradient mixing in low-level deformation space.

---

# Recommended Stabilization for the First Option A Version

The first version should not use the raw gradient unmodified. It should be stabilized.

---

## A. Target smoothing

Because \(dL_{\text{render}}/dx_T\) may be noisy, smooth the target signal before forming \(x_{\text{target}}\).

Possible options:

- neighbor averaging,
- graph Laplacian smoothing,
- Gaussian smoothing,
- low-frequency basis projection.

### Why
Silhouette gradients often contain local artifacts. Smoothing makes the target more physically realizable.

---

## B. Step clipping

Use a bounded target step:

\[
x_{\text{target}} = x_T - \text{clip}(\eta g_x, \tau)
\]

### Why
Without clipping, the pseudo target may become too aggressive and make the attractor physically unreasonable.

---

## C. Late-stage activation

Do not necessarily turn on the attractor from the beginning.

Instead, use a schedule such as:

- early stage: physics only,
- middle stage: weak attractor,
- late stage: stronger attractor.

### Why
This keeps early physical convergence stable and lets render guidance act mostly when silhouette refinement becomes relevant.

---

# Phase 3. What I Need to Verify After Implementing Option A

After implementing Option A, I should verify not only whether it improves the silhouette, but whether it resolves the actual failure modes of direct injection.

---

## Check 1. Is Option A more stable than direct injection?

### Compare against
- no render guidance,
- physics-scaled direct injection,
- fixed-norm direct injection,
- optional force-space heuristic,
- Option A.

### What to evaluate
- stability,
- physics convergence,
- render convergence,
- final silhouette metrics,
- qualitative deformation behavior.

### Why
The claim is not just that Option A works, but that it is more suitable than direct gradient injection.

---

## Check 2. Does Option A bypass the misalignment problem?

Direct injection fails partly because:

\[
\cos(g_{\text{phys}}, g_{\text{render}}) \approx 0
\]

Option A does not need to make those gradients directly compatible. Instead, it should let physics move toward a preferred target state.

### What to measure
- target tracking error \(\|x_T - x_{\text{target}}\|\),
- whether the attractor term actually improves render metrics,
- whether physics can follow the target without destabilization.

### Why
This checks whether Option A is successfully replacing direct incompatible injection with indirect feasible guidance.

---

## Check 3. Does Option A actually produce better shape-level correction?

This is especially important in missing-structure cases such as ear formation.

### What to inspect
- whether mass moves toward missing silhouette regions,
- whether the final shape covers more of the target support,
- whether the method avoids only compressing the source shape without extending it appropriately.

### Why
Option A is motivated by the need for global shape correction, not just local deformation.

---

## Check 4. Is physical plausibility preserved?

Option A should not improve silhouette at the cost of destroying physical behavior.

### What to track
- volume change,
- deformation singular values,
- stress magnitude,
- temporal smoothness,
- convergence speed,
- HC vs PO behavior.

### Why
A better silhouette is not enough if the physical solution becomes unrealistic.

---

## Check 5. How sensitive is Option A to target construction?

The Option A formulation includes several choices that may affect behavior:

- target step size \(\eta\),
- attractor weight \(\lambda_{\text{attr}}\),
- clipping threshold \(\tau\),
- smoothing strength,
- activation schedule.

### What to do
Run sensitivity sweeps over these hyperparameters.

### Why
This is important both for debugging and for a future paper-quality ablation.

---

# Paper-Oriented Ablation Map

If the goal is to eventually write this up as a paper method, the experimental logic should be organized clearly.

---

## Ablation A. Why direct injection fails

### Show
- physics-scaled render injection vanishes late,
- fixed-norm injection keeps signal alive but still struggles,
- severe misalignment in \(F\)-space,
- instability or weak effect under direct local injection.

### Message
Direct injection into local deformation space is structurally mismatched with silhouette supervision.

---

## Ablation B. Why control space matters

### Show
- compare \(F\)-space, \(x\)-space, and optional force-space behavior,
- visualize render gradient fields,
- show that higher-level spaces produce more coherent guidance.

### Message
Render supervision becomes more meaningful when lifted into a more global and physically interpretable control space.

---

## Ablation C. Why target-state proximal guidance works

### Show
- improved silhouette refinement,
- better stability,
- less direct conflict with physics,
- better final shape quality,
- preserved physical plausibility.

### Message
Separating **where to go** from **how to get there** resolves the conflict between render and physics objectives.

---

# Recommended Execution Order

## Step 1. Diagnostic experiments
Run:

1. fixed-norm direct injection,
2. \(x\)-space render gradient visualization,
3. force-space quick sanity check,
4. final-state render gradient inspection.

---

## Step 2. Minimal Option A implementation
Implement:

1. compute \(g_x = \partial L_{\text{render}}/\partial x_T\),
2. define \(x_{\text{target}} = x_T - \eta \hat g_x\),
3. optimize:

\[
L_{\text{total}}
=
L_{\text{phys}}
+
\lambda_{\text{attr}} \|x_T - x_{\text{target}}\|^2
\]

with:
- smoothing,
- clipping,
- late-stage activation.

---

## Step 3. Validation
Compare:

- no render guidance,
- physics-scaled direct injection,
- fixed-norm direct injection,
- optional force-space,
- Option A.

Evaluate:

- silhouette metrics,
- physics metrics,
- convergence,
- qualitative shape formation,
- stability.

---

# Short Final Summary

Before committing to Option A, I should first confirm experimentally that:

- fixing scaling alone is not enough,
- render gradients are more coherent in \(x\)-space than in \(F\)-space,
- and higher-level control spaces behave better than direct local deformation injection.

Then I should implement the smallest Option A:

- terminal-state render gradient,
- pseudo target state,
- proximal attractor loss.

Finally, I should verify that Option A improves silhouette refinement while preserving physical plausibility and avoiding the direct gradient conflict that breaks the current method.

---

# One-Sentence Summary

The right roadmap is: **first diagnose whether the failure comes from scaling and local control-space mismatch, then implement a minimal terminal-state proximal Option A, and finally verify that it replaces unstable direct gradient injection with stable and physically feasible shape guidance.**