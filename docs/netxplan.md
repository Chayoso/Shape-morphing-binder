# Why the Current Render Injection Strategy Is Failing, and What to Try Next

## 1. Executive Summary

The current failure is not just a matter of weak render loss. The deeper issue is that the **render gradient is being injected in a way that is both too small and poorly aligned with the physics objective**.

There are three main problems:

1. **The injected render gradient magnitude is too small** because it is scaled by the physics gradient norm.
2. **The render gradient direction is nearly orthogonal to the physics gradient direction**, so even when injected, it does not help physics converge.
3. **As physics converges, the render injection automatically vanishes**, which is the opposite of what is needed in the late-stage silhouette refinement regime.

So the current strategy effectively makes render supervision a weak, shrinking, and mostly misaligned perturbation to the physics optimization.

---

## 2. Observed Problems

### 2.1 Injected render gradient is too small

Current behavior:

- `phys_ctrl_norm ≈ 2~3`
- `render_grad_norm ≈ 617`
- effective scaling:

\[
\lambda_{\text{eff}} = 0.1 \times \frac{2.7}{617} \approx 0.00044
\]

This gives:

\[
\text{inject\_F} \approx 0.27
\]

which is only around **10% of the physics gradient norm**, and in practice may be even less effective because the directions are not aligned.

### Interpretation

The render signal is being shrunk too aggressively before it reaches the physics variables.

---

### 2.2 Render gradient is almost orthogonal to physics gradient

Observed cosine similarity:

\[
\cos(g_{\text{phys}}, g_{\text{render}}) \approx -0.001 \sim -0.003
\]

This is essentially zero.

### Interpretation

This means the render gradient is not pointing in the same direction as the physics objective. So injecting render gradients into the physical update does **not** act like a useful assist signal. Instead, it acts more like a side perturbation.

In other words:

- physics wants to move along one deformation pathway,
- rendering wants to move along a different shape pathway,
- the two are almost unrelated in the current variable space.

---

### 2.3 As physics converges, render injection also vanishes

The current scaling rule makes the injected render gradient proportional to the physics control norm.

That means:

\[
\text{inject scale} \propto \|g_{\text{phys}}\|
\]

So when physics converges and:

\[
\|g_{\text{phys}}\| \to 0,
\]

the render injection also goes to zero.

### Why this is bad

This is exactly the opposite of what is desirable.

Late in optimization:

- physics may already be near a local minimum,
- the remaining task may be silhouette refinement,
- render supervision should become **more important**, not less.

But under the current rule, render guidance disappears precisely when it is needed most.

---

### 2.4 HC physics converges more slowly than PO

Observed behavior suggests that HC physics converges more slowly (e.g. `+12%`, `+4%` slower), and the injected render gradient is not helping.

### Interpretation

The injected render signal is likely not aligned with the direction that reduces the physical objective. So instead of helping, it slightly interferes with the normal physical convergence trajectory.

---

## 3. Root Cause

The root cause is that the current scaling rule is conceptually wrong:

\[
g_{\text{inject}}
\propto
\alpha_{\text{balance}}
\cdot
\frac{\|g_{\text{phys}}\|}{\|g_{\text{render}}\|}
\cdot g_{\text{render}}
\]

This assumes that render guidance should be expressed as a fraction of the current physics gradient magnitude.

But that causes two structural failures:

1. **Render supervision is subordinated to physics**.
2. **Render supervision vanishes as physics converges**.

So the current system is not truly doing joint optimization. It is doing:

> physics optimization with a weak render perturbation that is only allowed to exist while physics gradients are large.

That is not suitable for silhouette-driven late-stage refinement.

---

## 4. Deeper Interpretation: This Is Not Just a Scale Problem

This is not only a numerical scaling issue. It is also a **representation mismatch** problem.

The render objective is trying to impose a shape-level change, for example:

- move mass toward the bunny ears,
- expand silhouette support in missing regions,
- create nonlocal geometric structure.

But the physics update operates in a lower-level variable space, such as:

- local deformation,
- stress response,
- particle motion under dynamics,
- physically stable deformation pathways.

So the two objectives are not merely scaled differently — they often prefer **different trajectories**.

A good intuitive example is:

- rendering wants particles to move directly toward the ear region,
- physics wants to first compress or redistribute the sphere in a physically stable way before forming ear-like structures.

This means the conflict is not accidental noise. It is a real mismatch between:

- **what the silhouette wants**, and
- **what the local physics variables can naturally realize**.

---

## 5. Why Simple Heuristic Tuning Is Not Enough

It is tempting to try one of these:

- just increase `alpha_balance`,
- just increase the injected render norm,
- just keep only positive cosine components,
- just tune lambda manually.

These may help partially, but they do not solve the fundamental problem.

### Why not

If the render gradient is mostly orthogonal to the physics gradient, then:

- scaling it up may destabilize physics,
- projecting it entirely onto physics may remove almost all of its useful shape information,
- filtering by cosine sign may be safe but may also destroy the very signal needed to form missing structures like ears.

So the problem is not just "how much render gradient to inject", but also:

> **in what variable space should render guidance act?**

This is the core issue.

---

## 6. What Should Change

---

## 6.1 Do not scale render injection by physics gradient norm

This is the first thing to change.

### Current bad idea

\[
g_{\text{inject}}
\propto
\|g_{\text{phys}}\|
\cdot
\frac{g_{\text{render}}}{\|g_{\text{render}}\|}
\]

### Better idea

Keep the render signal alive independently of physics convergence:

\[
g_{\text{inject}} =
\alpha
\cdot
\frac{g_{\text{render}}}{\|g_{\text{render}}\| + \epsilon}
\]

possibly with clipping or trust-region control.

### Why

This prevents render supervision from disappearing just because the physics term is near convergence.

---

## 6.2 Separate aligned and orthogonal render components

Let:

\[
g_r = g_{\text{render}}, \qquad g_p = g_{\text{phys}}
\]

Then decompose the render gradient into:

### aligned component

\[
g_r^{\parallel}
=
\text{proj}_{g_p}(g_r)
=
\frac{g_r^\top g_p}{\|g_p\|^2} g_p
\]

### orthogonal component

\[
g_r^{\perp} = g_r - g_r^{\parallel}
\]

### Why this helps

This gives a cleaner way to reason about what the render gradient is trying to do.

- \(g_r^{\parallel}\): compatible with current physics descent direction
- \(g_r^{\perp}\): asks for changes that physics is not currently pursuing

This is much more informative than a single cosine number.

---

## 6.3 Do not assume the orthogonal component is useless

It may be tempting to discard:

\[
g_r^{\perp}
\]

entirely.

That is safe, but it may also destroy the only signal that points toward missing silhouette structures.

For example:

- the aligned component may be tiny,
- the orthogonal component may contain the information needed to grow ear-like regions,
- projecting everything onto physics may make the optimization stable but incapable of improving shape.

So the orthogonal component should not necessarily be discarded. It may need to be handled differently.

---

## 6.4 Move render supervision to a coarser control space

This is likely the most important conceptual fix.

The current approach injects render gradients directly into low-level deformation variables such as \(F\) or particle-level physical states.

That is likely too low-level for silhouette supervision.

A silhouette mismatch such as "missing bunny ears" is a **global shape discrepancy**, not a local constitutive correction.

So render supervision should act on a more controllable and more global variable space, for example:

- coarse deformation latent,
- low-frequency displacement field,
- rest-shape offset,
- control force field,
- attractor field,
- template warp parameters.

### Why

The render objective is trying to express:

> "mass should be redistributed into this missing region"

but direct local \(F\)-injection is forcing that signal into a variable space that only supports local deformation behavior.

That mismatch is likely the reason the gradients become orthogonal and ineffective.

---

## 6.5 Consider alternating or proximal optimization instead of direct gradient injection

Another promising direction is to stop injecting render gradients directly into every physics update.

Instead, use one of these strategies:

### Alternating scheme

1. take several physics-driven steps,
2. take a render-driven coarse alignment step,
3. relax again under physics.

### Proximal / target scheme

Use render gradients to define a preferred target state:

\[
x_{\text{render-target}}
\]

and then solve:

\[
L = L_{\text{phys}} + \lambda \|x - x_{\text{render-target}}\|^2
\]

### Why

This lets render supervision propose where the shape should move, while physics determines how to get there in a feasible way.

This is often more stable than mixing incompatible gradients directly.

---

## 7. Recommended Next Experiments

---

## Experiment 1: Remove physics-dependent scaling

Replace the current rule with a fixed-norm render injection:

\[
g_{\text{inject}} =
\alpha
\cdot
\frac{g_r}{\|g_r\|+\epsilon}
\]

with a small \(\alpha\).

### Goal
Check whether render supervision remains active even when physics gradients become small.

---

## Experiment 2: Use aligned-only injection

Inject only the component of render gradient aligned with physics:

\[
g_{\text{inject}} =
\alpha
\cdot
\frac{g_r^{\parallel}}{\|g_r^{\parallel}\|+\epsilon}
\]

or gate it by cosine sign.

### Goal
Check whether render guidance helps only when compatible with the physics descent direction.

### Caveat
This may be stable but overly conservative.

---

## Experiment 3: Use aligned + weak orthogonal injection

Use:

\[
g_{\text{inject}} =
\alpha_{\parallel}
\cdot
\frac{g_r^{\parallel}}{\|g_r^{\parallel}\|+\epsilon}
+
\alpha_{\perp}
\cdot
\frac{g_r^{\perp}}{\|g_r^{\perp}\|+\epsilon}
\]

with:

\[
\alpha_{\perp} \ll \alpha_{\parallel}
\]

### Goal
Preserve some nonlocal shape information without letting the orthogonal component dominate.

---

## Experiment 4: Activate render injection only in later stages

Use a staged schedule:

- early stage: physics only,
- middle stage: aligned render injection,
- late stage: stronger render refinement.

### Goal
Avoid destabilizing early physical convergence while still allowing render-driven silhouette correction later.

---

## Experiment 5: Move render supervision to a coarse variable space

Instead of injecting directly into \(F\) or local particle variables, inject render guidance into one of:

- coarse displacement basis,
- low-frequency latent deformation,
- rest-shape offsets,
- target attractor fields.

### Goal
Test whether the render objective becomes more geometrically meaningful when applied in a more global and controllable space.

---

## Experiment 6: Compare full, aligned-only, and no-orthogonal injection

Evaluate at least these three conditions:

1. full render injection,
2. aligned-only render injection,
3. aligned + weak orthogonal injection.

Measure:

- silhouette improvement,
- physical loss,
- convergence rate,
- deformation stability,
- final shape plausibility.

### Goal
Determine whether the orthogonal render component is useful information or mostly destructive noise.

---

## 8. Practical Recommendation

If I had to choose the most realistic next path, it would be:

### First
Decouple render injection magnitude from the physics gradient norm.

### Second
Decompose render gradients into aligned and orthogonal parts, and test them separately.

### Third
Move render supervision away from direct low-level \(F\)-injection and toward a coarser, more global control space.

### Fourth
If direct joint gradient mixing remains unstable, switch to an alternating or proximal scheme.

---

## 9. Final Conclusion

The current render injection strategy is failing for a structural reason:

- the render signal is scaled relative to the physics gradient,
- the render direction is mostly orthogonal to the physics direction,
- and the signal disappears exactly when silhouette refinement should become important.

So this is not just a bad hyperparameter choice. It is a sign that:

> **the current variable space and injection rule are not well matched to the type of geometric guidance that alpha silhouettes provide.**

The likely fix is not to keep tuning the same formula harder, but to change the way render supervision is represented and applied.

---

## 10. One-Sentence Summary

The main issue is that render guidance is currently too weak, too misaligned, and too tied to physics convergence, so the next step is to keep render supervision alive independently, separate aligned and orthogonal components, and move silhouette-driven updates into a coarser and more physically meaningful control space.