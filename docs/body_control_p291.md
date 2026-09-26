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

## Initial results and the next ablation

Same-schedule 300k bunny prefixes (`ddc6930`, T=20, dt=1/240, dx=0.3062907 wu,
36^3 loss grid, seed 1, animations=300, stop_after_windows=8):

| raw target-radius neighbor density above y=2.3 | baseline | body + dFc |
|---|---:|---:|
| window 4, frame 80 | 0.0580 | 0.0335 |
| window 6, frame 120 | 0.0483 | 0.0858 |
| window 8, frame 160 | 0.2620 | 0.3827 |

The mixed actuator improves later filling but worsens the earliest sparse front.
It **does not pass the no-hole goal** and is not advanced to the full gallery.
The existing global hole_frac metric reads zero on both prefixes and therefore
does not detect this specific defect. These are neighborhood counts, not a
topological-hole certificate. The new raw audit also checked every delivered
frame after admission: max active-pin displacement is exactly 0 in both prefixes
(169 baseline / 2039 mixed particles with at least one later frame checked).
The last 10% of an 8-window prefix is not a converged tail.

`--body_no_dfc` is the next diagnostic: retain elasticity and u, but make learned
dFc identically zero, leaving the bulk-force field as the actuation channel for
transport. It distinguishes mixed stress/body forcing from insufficient bulk
delivery. It requires body_ctrl and is not a proposed default or proof of success.

The earlier `c291_bunny_body8` run used animations=8 and switched c2f at window 5;
it is plumbing evidence only. `base8s` and `body8s` preserve the 300-window schedule.

## Rendering settled points

`render_splat_gpu --settled_freeze` checks every raw delivered frame from actual
pin admission, rejects moving pins and release modes without active-mask history,
then fixes each active pin's normal/radius at its first rendered settled frame.
Opacity support remains based on current neighbors; `--material_support` is
incompatible. This prevents attribute refitting from rotating a settled splat
without masking density loss. It does not freeze unpinned points or establish
that all particles have converged. Visual validation remains pending.

## Independent terminal-force diagnostic

Reference-bound step scaling (`b9c438f`, same N300k/T20/dt1/240/dx0.3062907/loss36^3
and 300-window schedule) still fails: `c291_bunny_norm8s` delivers best window 5,
silIoU 0.7909; matched render-off `normphys8s` delivers window 6, silIoU 0.7920.
Both fail terminal rest (drift_rel 0.01589/0.01625). These are unequal best-window
endpoints, not evidence of a render benefit or detriment. The no-dFc normalized
actuator is not promoted to full runs.

The original single time mode couples actual displacement and terminal velocity;
zero *applied* impulse does not remove initial momentum or elastic recoil.
`--body_terminal_ctrl` adds a second physical-force mode `h_t=1/H^2-(T+1)/(2T)a_t`,
`H=T dt`, acting on another spatial field. For a free particle this mode changes
velocity by `c/H` with zero endpoint displacement. It may supply net impulse.
The joint six-component node coefficient is bounded by one; the regularizer sums
both mode norms per particle. It is not a velocity overwrite, and does not claim
rest in the nonlinear elastic case. The same force must enter all rollout paths.

Tests cover braking a moving free body without changing its prescribed endpoint,
mass independence, adjoint finite differences, persistent parity, pin invariance,
shape mismatch rejection and end-to-end two-mode optimization. Experiments also
record coefficient saturation and each accepted line-search alpha; nominal RMS
alone cannot identify the limitation. First gate: bounded eight-window bunny,
followed by longer runs only if transport, density and residual motion warrant it.
Telemetry units: `body_terminal_rms_wu` is the RMS velocity-equivalent displacement
field c; divide by H for its nominal free velocity change. Saturation is over all
occupied nodes, including inactive support. Accepted alphas are shared line-search
steps, to be read with `body_step_scale` and the coefficient projection; they are
not measured displacement. The coefficient regularizer is not integrated energy.

## Conditioning diagnostic after the stress ablation

`c291_bunny_force60s` (same discretisation, 300-window schedule, capped at 60)
stopped after three outer rejections at window 19; the best delivered commit is
window 8, silIoU 0.7112. This is a failure to reach the shape, not a hole-free
solution. All 15 rendered frames of each 8-window prefix were inspected: baseline
and mixed control have a translucent developing head; the stress-free-control
ablation remains largely a rounded body without an ear. These are diagnostic
Gaussian-splat clips, not approved morph deliverables.

The body field's nominal RMS stays 0.0733–0.0739 wu over windows 2–15 while actual
motion falls from 0.0449 to 0.0089 wu. Its dimensionless coefficient bound is 1;
the dFc bound is 0.02, yet both previously used Adam step size 0.02. Eight updates
therefore explore different fractions of the allowed control ranges.
`--body_step_normalized` tests body step scale `1/dfc_clip` (50 in this recipe),
leaving dFc/u steps, the body norm bound and all objective terms unchanged. Armijo
still uses the actual projected candidate. This is an optimization-conditioning
test, not an assertion that the larger step solves the physical defect.
The scale is a heuristic based on the reference bound: the 3-component body
vectors and 9-component dFc matrices have different interpolation and Rprop
operators, so it does not make their effective optimizer steps equal. Constant
nominal RMS alone does not prove the cause of declining physical delivery.
Its matched render-off mode is `force_normalized_phys`, which retains all three
body flags including `body_no_dfc`.

CPU suite at eb9fb59: 272 passed, 8 skipped (CUDA/reference requirements). Follow-up
CPU tests exercise both body step scalings and the raw audit's delivered-time mask.
The raw motion audit excludes appended hold frames and reports terminal velocity.
