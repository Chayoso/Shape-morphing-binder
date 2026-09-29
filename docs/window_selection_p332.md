# P332: whole-window selection and actual handoff

Status: implementation, CPU integration and three small CUDA gates passed.
Production attempt1 stopped before MPM: validation rejected `phys_loss=auto`
before the existing resolver ran. The follow-up preserves that resolver and
validates its resolved mode before any solve; overlapping and non-overlapping
CPU regressions pass. A fourth CUDA auto-resolution case and production rerun
are pending. This adds an
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
