# P332: whole-window selection and actual handoff

Status: implementation, CPU integration, four small CUDA gates and the registered
N300k identity/actual-successor capability passed; no quality candidate adopted.
Production attempt1 stopped before MPM: validation rejected `phys_loss=auto`
before the existing resolver ran. The follow-up preserves that resolver and
validates its resolved mode before any solve; overlapping and non-overlapping
CPU regressions pass. The corrected CUDA test and production rerun also pass.
This adds an
opt-in capability, not a rest, hole or visual-quality remedy. P331 provided no
admissible candidate and none of its rejected controls may be continued here.

`run_pipeline(..., select_window=callback)` requests a final-result context.
The optimizer creates it only for a non-null, non-pace-bound accepted buffer;
replayed donors preserve the ordinary result with an explicit skip reason.
The merit weight must equal the last accepted history weight. The current
optimizer already updates lambda only at inner iteration zero; the equality
check is defensive, not a fix for a measured lambda-update bug.

The callback runs inside the active numerical context. `original()` identifies
the exact owned donor result without replay. `evaluate(Mx6 coefficients, label)`
runs the frozen two-mode body model and owns the entire X/V/F/C trajectory and
terminal state. This first scope keeps stress, surface u, materials, frozen
plan, pins, basis and gates unchanged. It rejects endpoint PIC/shift, geometric
objectives, GS loss, continuity, material learning, pin KKT and carried moments.

Private candidates require finite and bounded full state, exact pin behavior,
head physical/effective determinant above 1e-4, the original joint coefficient
ball, a fresh complete rounded scalar merit no greater than donor merit plus
32 FP32 epsilon times max(abs(donor merit),1e-12), and the exact active pace
floor. Coast health is additional evidence at frozen pre-handoff policies;
it does not substitute for the actual successor. `certify(choice, raw_report)`
binds the trusted caller's raw decision; it does not independently validate that
decision. Production use still needs the registered independent raw gates.

`resolve(choice)` returns a complete owned result. Missing/failed raw evidence
or invalid proposals keep the exact donor. Foreign or expired tokens fail.
The runner closes the context in `finally` before ordinary promotion,
assimilation, outer acceptance/rollback, arrival, pin admission and v/C zeroing.
If that ordinary outer gate rejects a selected result, the existing rollback
and cold restart apply; there is no retry of the original donor in that window.

Donor Adam history/counts/render update telemetry remain donor observations.
A fresh `selected_observation` supplies selected state losses and kinetic terms;
gradient/work/step claims are cleared. No extra Adam iteration is invented.
The rendering report separates selected forwards, actual outer commits and
donor accepted-update influence. Prior inner/checkpoint/rollout observers still
observe the donor; they cannot relabel those trajectories as selected evidence.

CPU integration covers identity/None through real assimilation, new pins, layer
and bond preparation; a tiny changed private forward; missing certification;
callback failure/lease expiration; no-accept and final-invalid donors; actual
trailing gradient stop with its unchanged lambda; replay skip; and normal
outer rollback/cold restart. Synthetic certificates in these API tests are
explicitly not production quality certificates. The local tests use N160,
T3, dt1/240, dx1wu, loss12 cubed, four 24px views and two inner iterations.

## Registered CUDA capability

First run the explicit opt-in small CUDA integration tests on hyde06. Then
`scripts/probes/window_selection.py` uses the frozen raw mixed-body recipe:
N300000, T20, dt1/240, dx0.3062907543956724wu, loss36 cubed, 18 render views at
64px and eight inner iterations. The only horizon change is stop_after_windows
21. Only W20 requests a context; its original choice must be exactly identical
to the actual donor six-tuple after mutation of an owned inspection. No replay
or modified control is used for that identity choice.

The ordinary W21 solve supplies its real prepared step-zero state, after layer
and bond preparation. Save complete prepared state for W20/W21 and W20's
selected X/F path plus terminal v/C. Require unchanged positions/physical F,
unchanged surviving-free v/C, zero next-pinned v/C, retained old pins and a
positive new-pin witness. Record actual Fp changes and whether the controlled
successor passes the ordinary outer gate. Verify the selected path equals the
final raw archive segment, separately from delivery/truncation policy.

Reserve 3GB output; use immutable source/code bindings and the shared GPU launch
lock/interval. No render video or gallery promotion follows this capability.
An exact same-prefix two-arm fork still requires a complete resumable runner
state. Optimization through actual handoff still needs the F-to-Fp VJP and a
declared treatment of discrete admission/preparation branches. Those are not
provided by this callback API.

## Recorded capability result

Frozen `40d4bc95067629ecf82c53e6c4da5e7bbf77cd2f`, server
`work/p303/code_window_selection2`, run `work/p303/p332_identity2`.
All numbers below use the registered N300000/T20/dt1/240/dx0.3062907543956724wu,
loss36-cubed, eight-iteration, 18-view/64px CIC/PBR setting. Shared GS loss is off.

The run completes 21 ordinary accepted windows/168 accepted updates in
354.242s including evidence I/O. W20 original selection is exactly identical
to its donor six-tuple; inspection mutation is isolated and no replay occurs.
Its X/F path equals the final raw archive segment exactly. The ordinary W21
controlled solve also passes the outer gate. All six guard counters are zero.

The actual W20-to-W21 handoff retains 241225 old pins and admits 4380 new pins,
leaving 54395 free particles. X and physical F are unchanged; surviving-free
v/C are unchanged; next-pinned v/C are zero. Actual Fp changes for all 4380 new
pins and 54395 surviving-free particles, and for none of the old pins. These
are imposed handoff/pin policies, not a natural-rest or no-holes certificate.

Independent archive audit passes 2335 checks in 6.002s: source/evidence bindings,
CUDA raw-array identities/health and handoff counts, plus separately recomputed
render-report scalar bookkeeping. It does not rerun the complete prefix/Adam,
reconstruct the full assimilation formula, or independently reproduce the
unexported original six-tuple/final-prefix comparison; those are producer
receipts supported by the real CPU/CUDA integration tests. See
`evidence/p332/independent_audit2.json` and `evidence/p332/summary.json`.

Render influence: median lambda over 21 windows is 0.0644935. Median nominal
render direction share over 168 actual donor Adam updates is 0.503282; by channel,
body 0.361819, stress 0.563319, surface-u 0.755468. Median observed prepared image
loss change per accepted update is -8.69115e-5. These norm/direction observations
are not causal displacement percentages. There is no added optimizer step or
selected changed forward; 64px feedback does not supervise the exact exported
4K covariance/footprint.

The seven output files total 663391784 bytes (directory `du -sb`: 663391793).
The external monitor sampled a 20428 MiB process peak
(20452 MiB device); these are not exact peaks. Project usage after the run/audit
was 77062183753 bytes, below 100 GB. Before/corrected videos remain unchanged.

The explicit CUDA suite passes 4/4 in 5.63s. Independent focused CPU review covers
116 initial distinct cases plus 2 automatic-resolution regressions; root's
broader 84-case regression scope passes and overlaps some of those cases.
Do not sum these as disjoint totals. Initial pre-MPM validation failure is
retained in `evidence/p332/preflight_failure1.json`.

The next derivative boundary is recorded in `post_assimilation_adjoint_plan.md`.
It remains a design: candidate quality, complete preparation derivatives,
natural rest, full gallery and 4K artifact removal are still open.
