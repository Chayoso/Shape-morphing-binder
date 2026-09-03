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
