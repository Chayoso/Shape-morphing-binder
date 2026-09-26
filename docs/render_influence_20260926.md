# Render-channel influence audit, 2026-09-26

In the matched full bunny pair, rendering improves the final silhouette and ear-tip
mass, but does not remove sparse regions during transport or stop all residual
motion. The current render loss changes particle trajectories. It does not optimize the
exported Gaussian covariance, opacity or studio material as free appearance parameters.
The CUDA mixed60 run has `use_gauss_loss=False`: its image objective is a 64 by 64
soft silhouette plus density-normal shading, over six azimuths and three elevations.
Gradients reach stress, body-force and outer-layer displacement controls through the
same differentiable rollout. The 4K video uses a separate anisotropic splat renderer.

The implementation is in `pipeline/optimizer.py::losses_of` and the gradient assembly
below it; `pipeline/render_loss.py` defines the image terms. Mixed60 uses a paced
intermediate render target, so the channel also influences transport before the final
shape is reached. Turning `lambda_auto` off removes the render-dependent acceptance
and stopping track as well as the render gradient. This ablation measures the whole
render channel, not a gradient-only intervention with stopping held fixed.

## Discretisation and treatment

Both arms use the immutable c291 bunny source and target, each with 300000 particles,
MPM dx=0.3062907544 wu, dt=1/240, T=20, loss grid 36 cubed, 8 optimizer iterations per
window, and a 60-window cap on the same 300-animation recipe. Native source nearest
neighbor spacing is 0.0349885366 wu. Stress + body-force + surface-u controls, pinning,
PIC, shifting and all other settings are held fixed. The sole configuration change
is `lambda_auto: 0.5 -> 0.0`; material optimization is already off in both arms.

The on arm completed with 39 accepted windows, 42 attempts, 782 archived frames and
zero guard events. The off arm completed with 33 accepted windows and attempts,
662 archived frames and zero guard events. Off ran on hyde06 GPU2 from
22:10:17 UTC and finished writing its archive at 22:15:59 UTC. The comparison
audit ran on GPU2 from 22:17:28 to 22:18:02 UTC, using the validated CUDA radius-count
replacement in the `radius_fix` snapshot; it did not alter either simulation.

The comparison probe, `scripts/probes/render_influence.py`, rejects changed input
clouds, changed MPM parameters, or any configuration difference beyond lambda_auto.
Unequal simulation source hashes also fail unless an explicit source-equivalence
review matches both hashes. The on/off snapshots here are physics4/physics6:
their active module ASTs are identical after unused `physmorph.compute` imports
are removed, and the simulation probe is identical. `metrics.py` changed only for
independent reporting and is not called by the simulation probe. The recorded
review is `output/c291/gpuwork/physics4_vs6_review.json`; the audit includes it in
its output. This source review does not imply bitwise CUDA repeatability.

It compares accepted commit ordinals and first crossings of fixed raw-Chamfer
thresholds. The actual Chamfer mismatch at a threshold is retained: these are
threshold-crossing progress matches, not identical states. It never aligns runs by normalized
movie time. Its silhouette uses an independent binary point footprint, not the loss's
soft CIC image; density uses raw neighborhood counts. No reported metric reads a
rendered image. Density and a zero coarse hole fraction do not prove watertightness.

The individual raw audits select their own endpoint sparse-surface/unpinned
cohorts; their tail medians alone are not a same-particle causal comparison. The
comparison also fixes one common material cohort: the union of those two cohorts,
sampled at identical IDs in both arms over their last ten common accepted-commit
intervals. An ID may be pinned in one arm. The common cohort remains selected
from outcomes, so it is a descriptive symmetric comparison, not an unbiased
pre-treatment sample.

## Gradient influence before the ablation result

At the first optimizer iteration of each accepted window in the full CUDA on arm:

| Accepted windows | Median nominal render share | Median raw physics/render cosine |
| --- | ---: | ---: |
| 1-6 | 44.27% | 0.303 |
| 7-20 | 44.96% | -0.067 |
| 21-39 | 35.49% | -0.031 |
| All 39 | 39.27% | -0.025 |

The share is `lambda * ||projected render|| / (||physics core|| + lambda * ||projected render||)`.
This is a norm ratio before the subsequent optimizer, clipping, line search, and W1
term. It is not the fraction of physical displacement or evidence of better geometry.
The older pre-migration mixed60 run measured 40.51% under the same definition.

Per-control median shares in the CUDA on arm are stress 42.57%, body 24.69% and
surface-u 55.40%. In the first window alone they are 38.82%, 30.23%, and 86.16%; only
4.1% of the outer layer initially passes the u transport gate. Global PCGrad removes
the conflicting component of the concatenated render direction. It does not guarantee
nonnegative cosines within every separate control channel.

The existing six-window CUDA on/off pair differs by median 1.535 native spacings
(p95 3.617) at commit 6. Two legacy repeats differ by median 0.478 spacings there.
This supports a material trajectory effect, but is not a full CUDA repeat-noise
estimate or a claim that rendering removes holes or oscillations.

## Full matched results

| Quantity | Render on | Render off |
| --- | ---: | ---: |
| Final independent silhouette IoU | 0.971810 | 0.967927 |
| Final symmetric mean nearest-neighbor distance, wu | 0.055912 | 0.055957 |
| Final raw particles within 0.25 wu of target ear tip | 54 | 36 |
| Target particles in that same tip ball | 89 | 89 |
| Final top-region mean neighbor density / target r8 reference | 0.9321 | 0.9236 |
| Final top-region fraction below half reference density | 7.84% | 8.16% |
| Minimum accepted-rollout det F | 0.7734 | 0.8252 |
| Accepted commits / attempted windows | 39 / 42 | 33 / 33 |
| Measured probe runtime, seconds | 589.05 | 333.91 |
| Admitted pin fraction | 84.637% | 90.164% |
| Maximum admitted-pin position drift over all delivered raw frames | exactly 0 | exactly 0 |

The final silhouette gain is 0.003884, or 0.388 percentage points. Tip-ball mass is
50% higher with rendering, while both endpoints still fall below the target's 89
particles. The endpoints have almost equal raw Chamfer distance (0.081% difference).
This is one bunny pair, not a gallery-wide estimate or a CUDA repeat distribution.
The on arm also takes more windows, more time, and reaches a lower det F. The runtime
pair includes different stopping decisions and is not a controlled kernel benchmark.

At the same accepted commit 33, silhouette IoU is 0.971641 on versus 0.967927 off,
and tip counts are already 54 versus 36. Thus the tip/silhouette difference is not
solely the on arm's six additional commits. The independent whole-shape silhouette
is not uniformly better during growth: at commit 10 it is 0.941184 on versus 0.952452 off.

The fixed spatial top region is y>2.3 wu. Neighbor counts exclude the particle itself,
use the target's median eighth-neighbor radius, and are divided by 8. At equal
accepted commits its mean densities are:

| Commit | Render on | Render off |
| --- | ---: | ---: |
| 3 | 0.2475 | 0.2471 |
| 6 | 0.5943 | 0.6241 |
| 10 | 0.6994 | 0.7347 |
| 20 | 0.8329 | 0.8857 |
| 30 | 0.9134 | 0.9194 |
| 33 | 0.9206 | 0.9236 |

Rendering helps this pair's final outline and tip occupancy but is not the remedy
for early sparse supply: the on density is lower at several matched commits.
These counts are diagnostics of existing material, not tests of watertightness.

For the common cohort of 3074 material points selected from endpoints, evaluated at commits 23 through 33,
median displacement is 0.1042 sp per accepted commit on versus 0.0506 off;
p95 is 0.3963 versus 0.3790. Direction reversal occurs in 3.66% versus 5.70% of
eligible consecutive moving pairs (22746 versus 17010 pairs, each move > 1e-4 sp).
Median net-displacement/path-length ratio among moving particles is 0.923 versus 0.905.
The on arm continues moving more, largely in a consistent direction. These conditional
cohort statistics do not prove that rendering suppresses oscillations; neither arm
achieves all-particle rest. Endpoint-only raw audits also show nonzero unpinned
normal and tangential motion in both arms.

The new full on/off pair differs at commit 1 by median 0.2853 sp, versus 0.0079 sp
between the old CUDA-on prefix and the new full CUDA-on prefix. At commit 6 these
figures are 1.5211 and 1.0280 sp respectively. The latter comparison crosses snapshots
and lacks the old input/code hashes; stream/cache ownership changed. It is only
repeat context. Later trajectory differences must not be interpreted as an exact
render displacement fraction or formal effect-size estimate.

Raw output is `output/c291/gpuwork/render_influence_full.json`; it contains every
accepted-commit measurement, fixed-Chamfer threshold crossings with their actual
progress mismatch, common-cohort statistics, exact pin checks and source review.
The comparison has passed a read-only adversarial review of indexing, cohorts and
provenance; the identified missing code-hash gate was added before its measured run.

The immediate decision is to keep the render channel while repairing thin-region
material supply and settlement. Its current image objective helps this endpoint,
but it is not a direct 4K splat-boundary penalty and its presence does not establish
hole-free morphing or vibration-free arrival.
