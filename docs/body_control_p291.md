# P291: bulk actuation and settlement correctness

Status: opt-in implementation; quality gates pending. User authorised implementation
and hyde06 experiments on 2026-09-26. Baseline at implementation: 663a43e.

The goals are continuous material coverage during transport, no material sliding after
individual convergence, and no arrival oscillation. Adding an actuator alone proves
none of these goals. P280/P287/P289/P290 failures remain on record; loss resolution,
MPM cell size, arrival radius and the default lead are unchanged in this experiment.

## External force control

`--body_ctrl` adds a vector coefficient per occupied node of the MPM lattice. Frozen
trilinear interpolation at each window's initial particle positions gives
`b_p = dx sum_i w_pi c_i`, where `||c_i|| <= 1`. Coefficients start at zero each
window. This field is a **nominal free displacement**, not a position update.
It acts as an external body force through the same cubic P2G stencil:

`momentum_i += dt sum_p w_ip m_p a_t b_p`.

For T steps, `q_t = T - 1 - 2t` and
`a_t = q_t / [dt^2 sum_s (T-s) q_s]`. Thus `sum_t a_t = 0` and
`dt^2 sum_t (T-t) a_t = 1`. A free uniform cloud has displacement b and zero
terminal velocity. Elasticity, damping, contact, clamps, nonuniform fields and
nonzero initial velocity invalidate that endpoint guarantee; their actual response
is evaluated by the rollout and existing kinetic penalties.

This changes the actuator class: it is not merely a reparameterisation of dFc,
nor evidence that stress control can never drive the interior. The matched
physics-only comparison must include the same body channel with `lambda_auto=0`.
Active pins and particles with exactly zero control scale receive no body actuation.
Passive released particles may still move under the physical grid coupling.

The same body buffer participates in the fresh and persistent adjoints, no-grad
line search and final accepted rollout. The dimensionless particle field receives
the existing `w_ctrl` regularizer. Legacy replay save/load and gradient dumps are
rejected. `mom_carry>0` is rejected because occupied-node identities change between
windows; carrying by tensor shape would corrupt the optimizer state.

## Settlement corrections

Arrival admission is checked at the accepted endpoint against that window's frozen
full plan image and existing arrival radius. A window-start arrival that departed
is not admitted. Commit PIC and subgrid translation preserve already active pins.
Reported `pinned_frac` and exported `pinned` mean the active constraints; the broader
admitted set is reported separately as `settled_frac`. Reattachment and settle-commit
cannot currently be combined with pins and are explicitly rejected.

Admission still uses the existing reversal/history/ray conditions. These fixes do
not establish immediate convergence detection, and active-pin invariance must not
be substituted for the motion of all unpinned surface particles.

## Validation and acceptance

CPU tests cover impulse/displacement units, mass independence, finite-difference
adjoints, fresh/persistent parity, active-pin invariance, endpoint arrival and a
pipeline optimization. Non-OT render/telemetry defaults were repaired after this
suite exposed an existing branch-local initialization bug.

Run isolated snapshots under `/data/relcfd/chayo/physmorph_v2/work/c291/`, with caches
and temporary files under the same project. First compare 300k bunny baseline and
body at identical dt, T, cell size, loss grid, seeds and render settings; inspect
early material supply before spending on full bunny, dragon and the 40k gallery.
Retain the P284/P285/P286 gates, including the revised bunny ear-tip >=13, windows
<=53, no later ear onset, and no density gain bought by blunting. Full acceptance
also requires no added inversions, raw frame coverage checks, late normal/tangential
motion and reversal checks, and per-frame visual QA of Gaussian-splat output.

Project storage is checked every 30 seconds. Above 100 decimal GB, only the explicit
completed failed bp308/bp308d raw archives are eligible: gzip, verify all bytes by
SHA-256, write a manifest, then remove the original single file. Logs, JSON, accepted
baselines and other users' data remain intact. Exhausting candidates is reported.
