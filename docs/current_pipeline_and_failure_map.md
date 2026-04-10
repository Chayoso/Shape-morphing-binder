# Current Pipeline and Failure Map
## Practical Summary of the Current Branch, What Failed, and What Still Matters

## Status

This note summarizes the **actual current exploratory branch** as of `2026-04-10`.

It is meant to answer four questions in one place:
- what the current pipeline actually is
- what we tried and why it failed
- what the current evidence says
- what the remaining tuning space is

This document is intentionally more operational than [paper_framing.md](/home/chayo/Desktop/Shape-morphing-binder/docs/paper_framing.md) and more failure-oriented than [surface_aware_volumetric_pivot.md](/home/chayo/Desktop/Shape-morphing-binder/docs/surface_aware_volumetric_pivot.md).

---

# 1. Current Pipeline Identity

The current branch is best described as:

> **surface-aware render-guided volumetric morphing with control-space guidance and early-phase render curriculum**

The important design split is:
- **full volumetric particles** still carry the physical state
- **a surface/render shell** is used as the observation manifold
- **multi-view differentiable rendering** provides alpha and depth supervision
- **render-derived gradients are injected back into physics through `dFc` guidance**
- **early render supervision is weakened/coarsened so thin-structure seeds are not immediately suppressed**

This is not:
- pure field overwrite
- pure image-driven reconstruction
- pure surface-only simulation
- full joint autograd through the entire physics stack

The practical structure is:

```text
full-volume physics rollout
-> render only a surface shell
-> compute multi-view observation loss
-> convert previous render gradients into smoothed control penalties
-> update dFc inside physics
-> repeat
```

---

# 2. Current Pipeline, Step by Step

## 2.1 Volumetric state

The simulator still runs on the full particle set:
- positions `x`
- deformation gradient `F`
- plastic-like state `Fp`
- control increment `dFc`

Physics remains volumetric MPM-style evolution.

## 2.2 Surface-aware observation

Rendering does not use all particles equally.

We first extract a fixed source shell and split it into:
- `surface_mask`: thinner correction shell
- `render_surface_mask`: thicker shell used for rendering support

This is important because:
- a very thin shell gives sparse, hole-prone renders
- a slightly thicker render shell gives more stable visibility support

## 2.3 Observation loss

Current observation terms are:
- BCE alpha
- IoU alpha
- depth
- optional hard-max / top-k multi-view emphasis

Current important implementation detail:
- losses are **gradient-norm matched**
- auxiliary terms are not simply summed with fixed raw scalar weights

The current loss branch also supports an **early curriculum**:
- alpha is blurred early
- alpha is downsampled early
- negative BCE term is weakened early
- depth is weakened early and can be one-sided

The point is not to make early render loss stronger.
The point is to make it **less destructive to emerging thin structures**.

## 2.4 Control-space coupling

The current main coupling route is not direct final-position overwrite.

Instead:
1. collect render-space observation gradients from the surface shell
2. smooth / diffuse them
3. convert them into `dL/dx` and `dL/dF` penalties
4. inject them into the differentiable physics pass
5. let the optimizer update `dFc`

So the render branch is currently a **control-space bias**, not a direct position solver.

## 2.5 Multi-view aggregation

The bunny ear failure is view-dependent, so current observation is not plain mean-only supervision.

The pipeline supports:
- mean
- hard-max penalty
- top-k penalty

This exists because average multi-view loss can improve while one critical ear view remains bad.

---

# 3. What We Now Believe About the Bunny-Ear Failure

The strongest current hypothesis is:

> **the main failure is not that physics cannot raise ear particles, but that early render coupling suppresses sparse upward ear seeds before they become visually dominant**

More concretely:
- physics-only can send particles upward toward ear-like protrusions
- these early seeds are sparse and weak in image space
- naive render coupling treats them as small protrusion error or noisy false positive mass
- smoothing / diffusion / coupling then collapses them back toward a one-lobe solution

This is why:
- late render coupling helped
- very early sharp render guidance hurt
- early curriculum partially helped but was still too aggressively coupled to physics

---

# 4. What Worked

## 4.1 Physics-only can generate ear seeds

The PPC=`3` physics-only run showed:
- upward thin-structure growth is physically reachable
- ear-like mass can emerge without render guidance

This means the failure is not simply "physics cannot make ears."

It is closer to:
- physics can create the seed
- later optimization often destroys or smooths it

## 4.2 Turning render on after the seed helps

The most convincing result so far is:
- run physics-only until a meaningful seed appears
- then continue from a checkpoint with render guidance enabled

This improved:
- observation loss
- depth consistency
- and even the physics objective

In other words:
- render guidance is not inherently in conflict with physics
- it becomes useful once the rollout is already in the right basin

## 4.3 Slower render-to-physics coupling is consistently better

In the PPC=`4`, `25ep x 20` schedule sweep, the dominant pattern was:
- **longer warmup was consistently better**
- `render_gain_start` mattered less than `warmup length`

Best family:
- very small initial render gain
- long warmup, especially `24` episodes

This is important because it narrows the remaining search space:
- not "find a magical new loss"
- but "slow down early coupling"

---

# 5. What Failed, and Why

This section is the most important one.

## 5.1 Full-particle render supervision was too diffuse

Earlier branches supervised too much of the full particle set.

Observed problem:
- render gradient spread too broadly
- thin structures were diluted
- the signal was not focused on visible geometry

This motivated the surface-aware pivot.

## 5.2 Direct field overwrite was too detached from physics

Earlier decoupled correction branches directly corrected positions in field space.

That helped alignment, but it did not solve the core coupling problem:
- the renderer corrected geometry
- physics evolved its own internal state
- the interaction was not sufficiently coherent

This motivated the `dFc` control-space route.

## 5.3 Multi-scale silhouette alone did almost nothing

We tried adding multi-scale silhouette terms.

Observed result:
- metrics were nearly identical to baseline
- the optimizer trajectory barely changed

Interpretation:
- the term existed numerically
- but its actual gradient influence on the relevant modes was negligible

## 5.4 Screen-space curvature proxy was too weak

We first used curvature derived from rendered depth maps.

Observed result:
- raw curvature loss and raw gradient norms were tiny
- gradient matching immediately hit the scale cap
- the term still failed to materially change optimization

Interpretation:
- the proxy was too weak and too indirect

## 5.5 Mesh-curvature concavity loss was strong but conflicted

We then switched to mesh-derived curvature and explicit concavity supervision.

That confirmed the target signal existed:
- negative curvature in the ear valley was real
- projected maps were not empty

But the resulting loss still failed as a main optimization term.

Observed pattern:
- total objective increased a lot
- worst-view BCE did not improve
- `dFc_mean` barely moved

Interpretation:
- the concavity term was not dead
- it was **structurally misaligned with the current control manifold**
- it increased optimization pressure without creating the right bifurcation mode

Conclusion:
- direct concavity loss was removed from the active path
- concavity was retained only as a weak region prior for BCE weighting

## 5.6 Global gradient matching was not enough for localized losses

This became very clear with concavity.

Even if a loss is only `3%` or `5%` of the global anchor gradient norm, it can still be very aggressive if:
- it is spatially localized
- it acts on a narrow structured band

This taught us:
- global norm matching is useful
- but it does not guarantee local compatibility

## 5.7 Early render curriculum helped, but not enough

We introduced:
- blurred early alpha
- downsampled early alpha
- weaker negative BCE
- weaker early depth

This was directionally correct.

But the `PPC=3` `40ep` run still underperformed the best physics-only + late-render behavior.

Observed issue:
- loss was more forgiving
- but control coupling was still effectively too strong too early

The key problem was not only the loss form.
It was also the **rate at which render guidance was translated into physics force**.

## 5.8 Short warmup destroys the seed

This was strongly confirmed in the PPC=`4` sweep.

Short warmup settings were consistently worse than long warmup settings.

Interpretation:
- even with better early loss, if the control signal reaches full strength too soon,
  the same seed-suppression failure comes back

## 5.9 Surface can get visually rougher under render guidance

This is not a bug, but it is a real observed side effect.

Why:
- alpha/depth supervision constrains coarse visible geometry
- it does not directly penalize fine surface roughness
- Gaussian splatting can preserve or even emphasize clumpy shell structure

So render guidance can improve:
- silhouette
- depth alignment

while not necessarily improving:
- local surface smoothness

For now, roughness is not the main bottleneck.
Ear preservation is.

---

# 6. The Main Lessons So Far

## Lesson 1

The problem is **not** "how do we add more image losses?"

It is:

> **how do we inject visual supervision into differentiable physics without destroying sparse but useful structural seeds?**

## Lesson 2

The important axis is not raw loss weight.

It is:
- coupling schedule
- locality of the signal
- whether the control manifold can realize the imposed visual correction

## Lesson 3

The best behavior so far comes from:
- letting physics enter a useful basin first
- or making early render guidance extremely weak and forgiving

## Lesson 4

The bunny ear case is not just a hard example.

It is a diagnostic case that reveals:
- asymmetric local minima
- view averaging failure
- seed suppression under early coupling

---

# 7. Current Best-Supported Story

The cleanest current story is:

1. Full volumetric physics alone is physically plausible but visually underconstrained
2. Thin-structure failures arise because early sparse structural seeds are weak in image space
3. Naive render coupling suppresses those seeds
4. Surface-aware observation and control-space guidance fix the coupling structure
5. A slow curriculum on render-to-physics coupling is more important than simply adding stronger losses

This is also the most paper-compatible story.

---

# 8. Current Operating Points

## 8.1 Reliable qualitative operating point

The strongest current qualitative strategy is still:
- physics-only rollout until ear seeds appear
- then render-guided continuation from a checkpoint

This is the most convincing evidence that render guidance can help when introduced in the right basin.

## 8.2 Best current from-start schedule direction

For from-scratch runs, the strongest current direction is:
- small initial render gain
- long warmup
- forgiving early alpha/depth curriculum

In the PPC=`4` sweep, the best family was:
- long warmup `24`
- very small gain start, especially `0.005` or nearby

## 8.3 What is still open

Even the best current from-scratch schedules are still not yet strong enough to claim:
- robust two-ear recovery from the beginning
- complete resolution of the thin-structure collapse

So this is still a tuning and validation phase, not a closed method story.

---

# 9. Remaining Tuning Space

The remaining search space is now relatively narrow.

The most important axes are:
- `control_guidance.render_gain_start`
- `control_guidance.render_gain_warmup_eps`
- `control_guidance.dLdF_weight`
- `control_guidance.dLdx_weight`
- early alpha blur/downsample duration
- early depth weakening duration

The least promising axes right now are:
- direct concavity loss
- more multi-scale silhouette terms
- stronger raw curvature penalties

Those have already been explored enough to say they are not the main driver.

---

# 10. Immediate Next Steps

The current practical next steps are:

1. finish the `+25ep` checkpoint-continuation sweep on the PPC=`4` schedule family
2. check whether the best `25ep` schedules stay good or collapse when run longer
3. compare:
   - early weak coupling from scratch
   - checkpoint-based late coupling
4. keep ear preservation as the primary criterion
5. ignore surface roughness until the ear failure is resolved

---

# 11. One-Sentence Summary

> The current evidence suggests that bunny-ear failure is primarily a **coupling-schedule problem**, not a simple loss-design problem: physics can create the right thin-structure seeds, but render guidance must be injected slowly enough that those seeds are refined rather than erased.
