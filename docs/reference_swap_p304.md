# P304: does changing the target reference explain late movement?

Status: diagnostic, no physical policy promotion. Fresh raw/no-PIC/no-shift
realization, original N300000, T20, dt1/240, dx .3062907543956724 wu, loss36^3,
eight inner iterations, cap24. Preregistered consecutive optimizer attempts
19/20 both pass outer acceptance; their shared endpoint is exactly identical.
The run has zero state guards. It is not a bitwise replay of P303 raw24a.

## Controlled comparison

On the actual W20 positions, retain W19 and W20 prepared density grids,
silhouettes, shading targets and full world-space plans. Evaluate x0, the
straight-segment midpoint and xT with the **same current lambda .02306887**.
Current physical/cleanup terms retain F, V, controls, NN assignments and gates.
These are position-space sensitivity tests, not feasible alternate trajectories
or held-rest states. Gradients dotted with observed displacement are not energy
or causal channel shares. Exact endpoint changes are reported separately.

Cohorts are frozen using current start pins and previous-end/current-start
arrival; current endpoint membership cannot discard escapees. The latter has
49,925 free IDs, of 50,068 free IDs in total. Normal/tangent decomposition uses
only valid current start layer normals (6,251 IDs in the arrived-free cohort).
Arrival radius is .30629075 wu, about 8.75 native source spacings. This is a
transport arrival condition, not exact per-particle convergence or rest.

| W20 endpoint loss change | W19 reference | W20 reference |
|---|---:|---:|
| Density | -3.68616e-5 | -4.01046e-5 |
| Weighted CIC/PBR rendering | -8.84029e-9 | -1.42427e-7 |
| Density plus weighted rendering | -3.68704e-5 | -4.02469e-5 |

Common physical/cleanup terms decrease 4.57736e-6. Both references reward this
observed endpoint movement, and arrived-free data sensitivities are negative
at all three segment samples. Retargeting is therefore not the sole driver
of W20 movement. Continued density fitting remains rewarded after coarse
arrival. This does not show that every free particle is usefully moving or
that late motion in other windows has the same cause.

The current-reference weighted rendering decrease is about 0.35% of its density
decrease along this one path. That is an observed scalar-loss ratio, **not** a
share of physical movement or evidence that rendering is unimportant to the
whole morph. P303's matched full-policy on/off comparison remains the causal
ablation; its nominal gradient share is a third, distinct quantity.

For the arrived-free IDs, net window RMS is .0139064 wu (.3975 native spacings).
MPM-advection sum RMS is .0138119 wu, surface-u .0026615 wu and layer residual
.0022878 wu; component RMS values are not additive. Physical/geometric running
mean-square speeds are .0320122/.0324161 (wu/s)^2, and terminal values are
.0369810/.0370292. This window's motion is chiefly present in the actual
advective path; it is not explained mainly by invisible direct-position motion.
Those are group summaries, not a per-ID proof of normal/tangent convergence.
Advection can contain feedback from earlier u/relaxation updates; this does not
causally remove those operators. Their direct summed contribution is not
identified by separate RMS values, and the surface-only cohort may differ.

## Validation and limitations

Snapshot endpoint density differs from the live value by 1.16e-10; rendering
matches. Midpoint AD/FD relative differences at segment steps .1/.03/.01 are
old data .471%/.303%/.0933%, new data .458%/.307%/.115%, and common terms
5.11%/1.68%/1.27%. The latter is not a tight smooth-gradient certification.
Signs and exact finite changes support the bounded conclusion above.

The original frozen execution preceded the final closure/finite/PBR fail-closed
guards. A separately preserved CUDA post-run verifier binds the original
protocol, numerical source, report, inputs and all sidecars; checks scalar/array
finiteness, exact consecutive raw endpoints and PBR origin/calibration scope;
and reconstructs the accepted-buffer merit .00211114450894 with absolute error
3.34e-10 (allowed 2.42e-7). It does not claim later source checks ran in that job.

Latest source validates those conditions in the observer. The real CPU MPM
two-window observer on/off test gives exactly equal frames, F and key history
values, even after mutating owned diagnostic arrays. Evaluators expire after
the callback. Other related CPU tests pass; no local GPU simulation is used.

No gallery, all-frame visual closure, zero-drift, persistent-rest or 4K artifact
gate has passed. This experiment narrows the mechanism before another control
or stopping-policy change. Evidence is in `docs/evidence/p304`; full owned path
and reference sidecars remain under hyde06 `work/p303/reference_swap2`.
