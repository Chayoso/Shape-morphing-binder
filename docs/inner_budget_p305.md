# P305: one prepared solve, additional inner iterations

Preregistered diagnostic, not a new rest policy. P304 W20 still rewarded
density fitting under both previous and current target references. Its eight
inner updates were all accepted; the last endpoint change was .0306 source
spacings and the joint body bound was not saturated. This motivates testing
whether the current inner budget ends before the useful descent is finished.

The original N300000 source and raw/no-PIC/no-shift recipe run unchanged through
W19. W20 alone permits 32 iterations **inside the same optimize_window call**.
Prepared plans, density/render targets, starting state, pins, NN/layer operators,
lambda, Adam time and moments carry through iterations 8, 16 and 32. Existing
line-search/pace/convergence exits stay active. A missing checkpoint or outer
rejection is recorded; no replacement solve is substituted. The run ends after
W20. T20, dt1/240, dx .3062907543956724 wu, loss36^3 remain fixed.

At each reached checkpoint, the observer owns the accepted raw path, velocities,
endpoint F and control leaves. A separate rollout at those same controls
measures the true scalar gradient, preserving the accepted evaluation buffers
and Adam state. Its merit/position discrepancy is reported as replay noise.
This raw control-gradient norm is not the previous PCGrad update norm, and is
not a constrained stationarity certificate.

Compare the same start-free and start-arrived-free IDs at every checkpoint.
Report net movement, path length, per-step RMS, terminal physical/geometric
speed, fixed-target binary silhouette/Chamfer/coverage, tip supply and density
of the same source-defined upper cohort. The source cohort uses the original
self-inclusive radius-count selection; self is excluded only in its density
measurement. Only the final selected candidate can be outer-committed; early
inner checkpoints are not presented as completed morphs.

Extra iterations support an under-solving explanation only if they reduce
merit and residual movement without losing shape/supply. Lower merit with
persistent or increasing movement does not support budget extension as a rest
remedy. Rendering influence remains in every accepted-step report, separate
from independent raw-state geometry. No gallery or rendered-quality claim is
made by this single prepared-window diagnostic.

The real CPU regression covers stress/body and stress/body/surface-u with
layer relaxation. It requires nonzero u gradients and updates, exact observer
on/off frames/F/history, and nonempty exact render-influence step records after
mutating the observer's owned arrays and nested telemetry. All numerical
production runs remain on hyde06.

## Completed result: extra iterations are not a rest remedy in this window

Frozen execution `work/p303/code_inner_budget2`, result `inner_budget2`: all
32 steps accepted, zero rejected and all outer state guards zero. W20 commits
the iteration32 accepted buffer. At the discretization above, the same53,561
start-arrived-free IDs give:

| Inner iteration | 8 | 16 | 32 |
| --- | ---: | ---: | ---: |
| Merit | .002031992 | .002006703 | .001977402 |
| Net RMS (native sp) | .41420 | .56588 | .70161 |
| Saved-step RMS (sp) | .02219 | .02949 | .03611 |
| Stored terminal RMS (wu/s) | .18957 | .21315 | .23785 |
| Geometric terminal RMS (wu/s) | .18973 | .21445 | .24299 |
| Fixed-target binary IoU | .961526 | .961507 | .962018 |
| Upper target coverage | .930904 | .931100 | .933777 |
| Tip count (target89) | 75 | 71 | 70 |
| Min endpoint detF | .84715 | .84513 | .84033 |

Merit drops2.69% while net RMS increases69.39%, stored terminal RMS25.47%
and geometric terminal RMS28.07%. Additional optimization rewards shape fitting
with more motion here. This rejects budget extension as this window's rest
remedy; it does not establish a final optimum or whole-morph convergence.
Raw scalar gradients remain nonzero and surface-u maximum gradient increases.
At32,7.02% of body nodes are at the joint coefficient bound (max1.000000119);
the earlier non-saturation observation does not carry to32.

## Rendering influence and validation

Lambda remains .02516812048. From8 to32 the density term falls4.71546e-5 and
weighted rendering falls3.77781e-6. Terminal kinetic and velocity-variance
weighted costs rise about5.585e-7 and6.239e-7 respectively. Nominal render
gradient share rises .4927 to .5608; that is not a causal displacement share.
Standard per-step influence reports are retained with the server run.

Checkpoint gradient replays preserve controls, Adam moments and all accepted
positions/velocities/F exactly. Replay merit differences are below9e-11 and
endpoint RMS below2e-8wu. The independent post-run CUDA verifier binds source,
inputs, protocol, result and sidecars before/after inspection. Each checkpoint
has95,418,816 finite array elements and246,358 exactly stationary start pins.
This verifies the original run; a later serialization finite guard was not
retroactively executed. Report/protocol/receipt are preserved in
`docs/evidence/p305`; full trajectories remain on hyde06. Independent refutation
agrees with the bounded rejection. No default, gallery or visual promotion.
