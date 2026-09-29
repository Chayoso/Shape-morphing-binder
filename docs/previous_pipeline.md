# Problems of the previous pipelines

Two earlier pipelines preceded settled transport: the public release's windowed method, and the `v3-grid-gs` line
that extended it from 2026-09-03 to 09-29. Their code and full records stay on git (tags `v3-grid-gs-final` and the
release commit `680622e`); this file keeps what was learned.

## How the windowed method worked

Each window simulated `T` driven steps and scored the body at that moment against a paced target: an intermediate
target that moved every particle one loss cell further along an optimal-transport plan toward the mesh. The main
physics loss compared cell sums of mass on the loss grid. Around this core the v3 line added pins, per-particle step
control, pace rules and commit-time position corrections.

## What went wrong

**1. The surface never came to rest (the "breathing").** Scoring the body while the control was still pushing let a
window accept states that sprang back once the next window started. The outer layer then reversed direction from one
window to the next. In the release's method at 300k, 70 % of windows reverse and every window of the second half does
(a streak of 99), which leaves a lumpy surface. The v3 line stopped this with per-particle step control and pins that
freeze settled particles, not by removing the cause.

**2. Thin features grew as a haze.** At 300k the ears grew as a thin, sparse haze before filling. Measured end to end:
- The cell-sum loss cannot see a filled region translating; its gradient lives on the density jump at the surface.
- The paced target led the body by a whole cell while a window delivered about a tenth of that.
- The loss filled the lead with the outermost one or two particle layers, which move 2–6× faster than the bulk under a
  stress control, so the skin was stripped ahead of the body.

**3. Levers inside the old formulation were exhausted.** Each of these was implemented, measured and refuted:
- a finer loss grid (it observes less: the sampling noise grows faster than the signal);
- a fixed small lead (fills the haze on the bunny but stalls long transports and hurts 14 of 18 gallery meshes);
- a lead governor that tries candidates (dithers during the growth and stalls the dragon);
- a surface-tapered control (the bulk loses speed before the skin separates);
- neighbourhood correspondence terms (the same null space as the cell sum).

**4. The fixes broke momentum.** Pins and the commit-time position edits (a null-space projection removing 13–85 % of
a window's displacement, stress assimilation) moved the body outside the physics. The centre of mass drifted 0.6–0.8
particle spacing over a morph, with render off as well as on.

**5. Rules accumulated faster than causes were removed.** By 09-29 the v3 line had about 27 000 added lines and
hundreds of options, most of them opt-in candidates that were never adopted. Each new rule interacted with the
others, and every change needed a full gallery to trust.

**6. Rendering looked hazy at 4K.** Mesh reconstructions (Poisson, marching cubes) shimmered frame to frame, so the
deliverable moved to Gaussian splats. The remaining 4K haze comes from sparse material and the splat radius growing up
to 4× to cover it; it is not caused by the deformation gradient, which the renderer does not use.

## What settled transport changes

| Problem | Settled transport |
|---|---|
| Scored while pushing | Scored after a released phase of equal length: rebound costs the control directly |
| Cell sum blind to translation | Sinkhorn divergence to the fixed target: its gradient is a transport field over the whole body |
| Paced target a cell ahead | No paced target: the target never moves |
| No term against thinning | A local support term bounded by the transport energy |
| Pins and position edits | None needed: late surface motion 0.007 spacing per frame without pins; centre of mass within 0.02 spacing |
| Render weight rebalanced every window | Calibrated once and held |

Same seed, 300k bunny: silhouette IoU 0.9851 against 0.9769 (v3 line) and 0.9646 (release legacy), 7 minutes against
12 and 28.

## What the earlier line still does better

The v3 line fits the head and body relief more closely at the end: 7.8 % of the target surface lies farther than 1.5
particle spacings from the body, against 10.4–11.7 % for settled transport. Its arrival snapping placed particles
exactly on target points, and it ran more windows. Closing this gap without reintroducing pins or position edits is
an open item in docs/experiments.md.
