# P291: bulk actuation and settlement correctness

Status: implemented and tested; experimental controllers not adopted. The combined
no-hole, shape-preservation and all-arrived-particle rest goals remain unmet.
User authorised implementation and hyde06 experiments on 2026-09-26.
Baseline at implementation: 663a43e.

Controlled full runs: bunny N=300k, T=20, dt=1/240, dx=0.3062907544 wu,
loss grid 36^3, seed 1, animations=300, diagnostic cap=60.

| run / snapshot | silIoU | highest-tip reference count | attempted windows |
|---|---:|---:|---:|
| fixedbase60 / 3555293 | 0.973296 | 5.867 | 43 |
| fixedbody60 / 3555293, two-mode body only | 0.972067 | 6.000 | 54 |
| mixed60 / 788b16e, two-mode body + dFc | 0.972392 | 6.933 | 39 |
| confirm60 / 788b16e, baseline + confirmed arrival | 0.975239 | 7.200 | 45 |

All fail the declared historical-shape tip gate >=13. The saved target itself has
89 native particles in that ball, or 11.867 reference particles; the gate is not
an exact target-density identity. All candidates are also below that target count.
Active pins checked over every delivered
frame remain exactly fixed (144492 / 288436 / 266360 / 178280 checked particles),
but unpinned surfaces still move and reverse. Mixed control improves early head
density but does not eliminate the sparse growth front. Its tip count drops from
8.933 at raw frame 480 to 6.933 at delivery. All rendered frames inspected:
65 baseline, 87 body-only, 67 mixed and 70 confirmation. Existing rest/hole gates
passing does not establish the user's goals. No dragon/gallery expansion or
production/page promotion is justified by these failed gates.

Latest full CPU suite (5052922): **285 passed, 8 skipped**. The CUDA probe also
checks reference-threshold sparse bonds across fresh/persistent adjoints and the
captured no-grad candidate. A separate C40k prefix checks non-paced compatibility;
it cannot certify complete C convergence or the full gallery.
That prefix completed eight windows on5052922 (T20, dt1/240, dx0.2985634406wu,
loss33^3): legacy admission remains active (30 pins, 27 checked with zero motion),
but its in-transit rest/ejection gates fail. It is runtime evidence only.

The old bm300 tip13.333 is not a matched controller-only comparison: both the
adjoint bond threshold and real pin preservation changed. To isolate admission
timing while retaining those repairs, `--settle_pin_confirm` requires the existing
window-start arrival mask AND arrival at the accepted endpoint against the frozen
full plan. The same mask defines particles needing transit-ray protection. It
adds no distance threshold, preserves all active pins, and does not change the
optimizer's arrived/pace criteria. Stuck-point and viscosity admission rules are
incompatible because they otherwise write the same settled set independently.
The `baseline_confirm` comparison above improves fit and the tip modestly but
fails the original shape/no-hole/rest requirements. It remains a diagnostic,
not a new default; the root cause of missing tip mass is not isolated by this test.

Historical pin audit (bm300_bunny, same N300k/T20/dt1/240/dx0.3062907544/loss36^3):
all 144992 admitted particles having a later delivered frame moved after admission.
Per-particle maximum displacement relative to its admission point: median0.6183sp,
p95=2.4695sp, max8.4114sp (0.2943015wu). Release modes are off. Thus the historical
"pinned" label did not certify immobility through the commit operators. The repaired
baseline and body runs have max0.0wu over the corresponding whole-frame audit.
The new distribution diagnostic reports exact nonzero movement, not a tolerance-
based convergence decision. Raw pin invariance is distinct from all-surface rest.
Final admissions with only a subsequent held frame, not another physical rollout,
number 461 in fixedbody60, 1673 in mixed60 and 2848 in confirm60. This audit alone does not prove that historical
post-pin drift supplied the missing tip mass.

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

Non-paced losses (including the C target's plain OT regime) export no per-particle
arrival contract. Their existing reversal-only settlement is retained and explicitly
logged as `pin_arrival_evidence=legacy_no_arrival_contract`; `arrived_end_frac` is null,
not a fabricated arrival rate. `settle_pin_confirm` rejects those regimes before
optimization. Their historical gallery results do not certify accepted-end arrival,
and the exact-pin changes still need matched quality checks on those shapes.

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
completed failed bp308/bp308d and explicitly listed failed P291 raw archives are eligible: gzip, verify all bytes by
SHA-256, write a manifest, then remove the original single file. Logs, JSON, accepted
baselines and other users' data remain intact. Exhausting candidates is reported.

2026-09-26 cleanup: lossless compression of completed failures alone saved little.
Three verified gzip archives (bp308_bunny, bp308d_dragon, c291_bunny_terminal60s)
were copied to `C:/dev/physmorph_archives/c291/`; compressed SHA-256 was compared
on both machines and the local files fsynced before removing the server duplicates.
Server `.gz.json` and `.gz.offload.json` receipts retain raw/compressed hashes,
sizes and the recovery path. Project usage fell to88.386GB before the next results.
The current accepted baselines, current tests, logs and JSON remain on hyde06.

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

## Replay and adjoint consistency audit

The longer two-mode diagnostic exposed rejected final replays with positive det F.
The adversarial review found two preexisting discrepancies:

* The no-grad trajectory received the disc_ref bond threshold `N/mass_ref_n`, but
  fresh/persistent adjoints silently defaulted to 1. `RolloutSpec.bond_threshold`
  and a shared bonds tuple now preserve the same threshold in all three bridges
  and the candidate/commit path. A three-particle separated cluster distinguishes
  thresholds 1 and 7.5 and tests all paths against a direct trajectory.
* Replay calibration divided its absolute discrepancy by `max(|E|,1)`, while the
  commit check used `max(|E|,1/unit_ratio)`. The calibration now uses the same
  transformed floor. A unit-rescaling test checks that physical tolerances agree.
  The 10x measured-noise multiplier is unchanged. This is not a guarantee that
  start-control noise bounds noise at the accepted, strongly forced state.

The final replay now also requires finite loss and the same finite-state check
as candidates. Null commits retain accepted/final energies, lambda values,
tolerance and (for body runs) endpoint position/velocity differences. Diagnostic
results before this repair cannot establish a correctly differentiated 300k
disc_ref controller; compare repaired candidate and baseline anew.

The trace (`c291_bunny_replay20s`, N300k/T20/dt1/240/dx0.3062907) measured one
rejection at E=0.019885529165 vs accepted 0.019885527130, tolerance 1.9886e-9,
with identical lambda, max endpoint delta 8.34e-7 wu and velocity delta 4.29e-6.
No scalar-tolerance multiplier was increased. The final commit now reuses the
last accepted trajectory **only while that exact evaluation remains in the
buffer**. Every subsequent candidate evaluation invalidates reuse; restoring
control leaves after rejection does not restore the overwritten trajectory.
That path still replays and validates. Both paths keep finite-loss/state checks.
`commit_from_accepted` distinguishes cached accepted merit from an independent
replay measurement. CPU integration tests cover direct reuse and forced candidate
rejection after an accepted step, checking the committed endpoint against the
accepted callback state.

The pre-repair two-mode 60-window run converged at attempted window 55 with
silIoU 0.970760 and no guard events. Its highest-target-point tip count
(end_probe.py convention: radius 0.25 wu, count*40000/N) is only **4.8** reference
particles, versus **13.333** in the archived bm300 delivered endpoint. It fails
the ear gate despite the improved developing-head density. This motivates a
matched mixed-actuator test (`body_terminal`), retaining dFc alongside the two
body modes. The earlier mixed test used only one temporal mode without the
reference-bound step scaling; it does not establish the mixed two-mode result.
No claim that dFc is uniquely necessary follows from this observation.

Baseline provenance clarification: the available bm300 JSON and recomputed raw
delivered endpoint both give silIoU 0.9741246 at frame 724 (deliver_n=725), while
the last archived state gives 0.9735716. These differ from the quoted 0.9733/767
summary. Comparisons here use the declared delivery cutoff and matching raw
frames; stored metadata is not silently replaced by a narrative value.

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
