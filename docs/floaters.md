# Floating Gaussian dossier

Status: **representation fix implemented; final 20k visual validation in progress
(2026-09-02)**.

## What counts as a floater

Two different errors must not share one metric:

- A **physical tail** is simulated material far from the target. It remains in raw-state
  Chamfer/target-NN/self-NN metrics and is never deleted.
- A **floating Gaussian** is a render primitive that no longer has enough continuum
  support to represent a material surface patch.

Target-NN distance in multiples of sample spacing is useful residual telemetry, but it is
not a floater definition: increasing `N` shrinks that spacing and makes the same geometric
error look worse. A target-mismatched but materially connected surface point must not be
hidden.

## Current representation

1. MPM remains volumetric and keeps every particle and its mass.
2. A persistent source/material-coordinate surface estimator selects the render Gaussian
   subset once. The target uses its independently estimated surface subset.
3. The differentiable Gaussian objective renders exactly this subset. It receives both
   center gradients and covariance gradients through
   `Sigma = sigma0^2 F F^T`; the terminal `(x,F)` covector is pulled through the complete
   MPM adjoint to `dFc`.
4. Export/viewer opacity additionally uses a **target-free frozen material-neighbour
   support check**. A primitive fades only when it is an extreme current-body singleton
   and fewer than two of its source-material bonds survive. This NumPy-only decision is
   outside optimization, so tearing cannot be rewarded by a disappearing render loss.
5. Unchecking `surface Gaussians` in the viewer exposes all physical particles for audit.

The opacity check never mutates `x/F/v/C`, particle count, mass, or momentum; all reported
simulation metrics continue to consume raw state.

## Resolution and Gaussian size

The 5k diagnostic used surface fraction 50% and sigma scale 1.4, giving absolute
`sigma0=0.09269` (`3 sigma=0.2781`). A 20k target-only 512px calibration used 10,140
surface samples:

| NN multiplier | absolute sigma | 3-sigma radius | visual result |
|---:|---:|---:|---|
| 0.70 | 0.02823 | 0.08468 | porous |
| 0.85 | 0.03427 | 0.10282 | thin-region pinholes |
| **1.00** | **0.04032** | **0.12097** | smallest acceptably continuous surface |
| 1.15 | 0.04637 | 0.13911 | beginning to round/merge detail |

Production therefore uses 20k particles, about 10k surface Gaussians, sigma scale 1.0,
128px differentiable Gaussian feedback, device-pixel-ratio viewer rendering, and at least
512px CUDA-3DGS QA. Absolute sigma is 56.5% smaller than the 5k/1.4 diagnostic setting.

## Ablation evidence

All runs: `dt=1/240`, `dx=0.5`, smoothing `0.955`.

- 5k/35% had one target-far primitive whose body-NN distance was `5.00x` the current
  median, zero material-neighbour overlap, and bond stretch `1.93..3.91x`: a genuine
  unsupported Gaussian. Material-support opacity reduces it to about 0.03.
- 5k/50% had one target-far primitive, but body-NN was only `2.13x`, three material
  neighbours survived, and bond stretch was `1.11..1.88x`; opacity correctly remains 1.
- 20k short runs with sigma scales 0.85/1.0 reached holes 0.55%/0.26% respectively.
  Thus a large 1.4 multiplier is unnecessary at high sampling density.

Final acceptance requires every saved production commit to be rendered and inspected for
a closed solid, no crossfade ghost, continuous silhouette, no unsupported visible splat,
and material-space colour that rides the deformation.

## Addendum — observability probe and the corrected floater mechanism (2026-09-02)

docs/probes/observability.md on converged flagship states: 97–98% of viewer-visible
floaters carry a NONZERO silhouette gradient (they hold 24–46% of its energy), so
"floater = loss null space" is false for the silhouette. Two things are true instead:
(1) the surface-parent gauss loss is blind to the interior half of the floaters by
construction (the viewer renders every particle); (2) on floaters the image-space
gradient points toward the surface with median cos ≈ 0.3 only — a projected-coverage
loss pushes a floater wherever the ray's alpha mismatch shrinks, not to its nearest
target point. The per-particle assignment term (nn-band) is the one mechanism with
cos ≈ 1 by construction, which is why it alone ever killed a floater class. The
unowned residual class is therefore: floater CLUMPS (the DT term's kNN isolation gate
leaves clumps unowned) beyond the nn-band far cap (4.5× spacing) — owned by nothing
but a weakly-directed silhouette. E4 (nn_far_k → ∞) tests exactly this.

E4 verdict (2026-09-02): nn_far_k 4.5 → ∞ (own every particle beyond the berth) at 300
paced commits — chamfer tie (0.1067 vs 0.1078), out_nn 22.3 → 20.5%, far floaters
1743 → 1511 (−13%), max_dt 26.4 → 23.6 sp, all guards zero. ADOPTED as the flagship
default; single-seed, so the hero-scale replicate decides the size of the effect.

## Census on the best 20k state (r3b delivered a180, 2026-09-03)

- floaters >2sp: 1615 (8.1%), far >3sp: 419; NOT clumped (3% in ≥5-particle clusters);
  median distance to the nearest body particle 1.85 sp, p90 4.0 sp → a loose fringe halo
  just outside the surface, not ejecta.
- their assigned target points are EMPTY (median 0 particles within 1 sp) — the
  "saturated target" hypothesis is false; the pull is there, the particle does not arrive.
- the target itself is UNDER-COVERED: 65% of target points have no particle within 1 sp,
  33% within 1.5 sp, 14% within 2 sp (grid-scale hole_frac 0.12% cannot see this).
- silhouette gradient on these floaters: present (99%) but cos to target 0.15 — noise.

Reading: the residual defect lives at 2–4 particle spacings, BELOW the resolution of
both data terms (D_vol cell ≈ 3.4 sp at loss_res 64/20k; silhouette pixel ≈ 2.3 sp at
96 px), so the only particle-scale force is the near-band's constant pull, which
equilibrates against elastic tension off the surface. Next A/Bs (40k, where N supports
finer grids): x1 loss_res 64→96 (cell ≈ 2.3 sp); x2 nn_berth_k 1.5→1.0 (no dead band).
Kill: guards ≠ 0, jitter ↑, or chamfer regression > 2% at no floater gain.

### Mechanism from the census: particle-scale density matching (x3, 2026-09-03)

The J census killed the volume-prior hypothesis (J ≈ 1 everywhere, ears 0.999) and the
local-density census named the defect: particle/target density ratio at particle
positions is 2.0–2.7 in the body (sub-cell CLUSTERING) and 1.3 in the ears, with the ears
holding 9–24% fewer particles than the target. Gaps between clumps read as fuzz, the
fringe halo is their outer edge, and the frayed ears are under-filled. Neither the CIC
D_vol (cell ≈ 3.4 sp) nor the silhouette (pixel ≈ 2.3 sp) can see this scale, and N
cannot support finer grids (20k over a 128³ grid is 0.1 particle/cell).

`d_kde` (physmorph/losses/volumetric.py): at every particle, the kernel density of the
particles (frozen per-window kNN, self counted) versus the kernel density of the target
points (frozen kNN), W = exp(−(r/h)²), h = 2 sp — the SPH form of D_vol. Its gradient
runs from crowded to deficient regions and from outside the target inward, so it owns
both the clumps and the fringe with the right direction (the silhouette's is cos 0.15).
Weight: w_kde × a one-shot calibration equating its x-gradient norm with D_vol's at the
first window (w_kde = 1 ⇒ parity) — a rule, not a tuned scalar. Logged as `d_kde`
(archived state, fresh neighbours, ungated), a freeze track, a gate-merit and a
delivery-merit component. Falsifiers (x3 at 20k vs r3b): cluster ratio 2.0 → ≤ 1.3,
unfilled@1.5sp 33% → < 20%, out_nn 8.1% → < 5%, at chamfer within 2% and guards 0;
any guard, jitter > 3e-3 or a chamfer regression > 2% kills it.

Visual confirmation (r5 photoreal at display sigma 1.0/1.25 sp, 2026-09-03): the body is
SPECKLED with pinholes between clumps and the ears are sparse filaments — the clustering
census made visible. Display sigma 1.6 hides the pinholes but blurs everything (the user's
"작으면 gradient 죽고 키우면 bleb" dilemma is, on the display side, exactly this
trade-off); the fix must be in the particle distribution (x3/x4), not the splat size.

x3 verdict (one-sided KDE, 20k): cluster ratio 2.0 → 1.25 (the clumps DID relax) but
out_nn 8.1 → 21.9%, far 419 → 1812, chamfer +16%, and the archived d_kde rose over the
run. Mechanism: the particle-side residual at an exterior particle (rho_t ≈ 0) can only
be lowered by reducing rho_p — repulsion — while the inward pull through rho_t vanishes
beyond the kernel; clumps dissolved OUTWARD. Revision (v2, x5): the true L2 kernel-
density distance evaluated on BOTH point sets — the target-side residual at deficient
target points attracts particles inward (tests: a particle 4 sp outside now receives an
inward gradient).

Adaptive display sigma (render_photoreal --adaptive_k 8, sigma_i ∝ local spacing,
2026-09-03): the body's speckle disappears, but the ears turn into a blurred cloud —
their local spacing is large because they hold 9–24% fewer particles than the target,
so a larger sigma can only smear them. Verdict: the ear defect is a particle-COUNT
(transport) shortfall, not a display issue; adaptive sigma is a valid body-side aid only.
Next falsifier: is the freeze premature during the slow ear-filling phase? (x7: patience
20, tol 1e-3, 20k; measure the ear-region particle fraction vs the target's 10%.)
