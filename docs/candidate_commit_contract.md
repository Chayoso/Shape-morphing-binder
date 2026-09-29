# Candidate merit and coupled continuation: implementation prerequisites

This began as a design audit. P332 implements the opt-in whole-result selection
boundary described in `window_selection_p332.md`; no physical-quality candidate
is promoted by that capability. P309 remains a
noncommitting fresh-window diagnostic. Its prepared-data/raw-quality gates
alone cannot authorize a production commit or persistent-rest claim.

## Evaluate the original objective at the candidate

Reuse the current window's `losses_of` and `phys_total` through a callback-lived,
read-only evaluator. Pass the candidate's actual endpoint x/F/v, complete V
history and expanded body energy. Do not install candidate coefficients into
production leaves temporarily. `phys_core` currently reads `body_field()` from
the live original coefficient leaf; an explicit candidate-energy override is
required. The private model's energy uses the same basis, gate and dx scaling.

For the recorded raw300k recipe, active common terms include stored terminal
kinetic energy (weight5), stored velocity variance (200), stress/body control
energy (.001), stress-control smoothness (100), J-volume (50), box (10), gated
DT (.2) and frozen nearest-neighbor band (.2). Their effective unit scaling is
the live window's `wu`, not necessarily1. Freeze references, gates/neighbors,
lambda and stress/u/material coefficients. Candidate body coefficients differ.
Additional configuration terms must either be
evaluated faithfully or rejected explicitly. Geometric rest/variance and
geometric F are off in this recipe; do not claim a general evaluator from it.

Prepared volume plus lambda times prepared render omits these terms.
`common_positions` also fixes original F/v/V and original body energy;
`evaluate_variance` changes only one scalar. Neither is a trajectory-candidate
evaluator. Restore mutable render/loss caches in `finally` and expire the new
evaluator after its callback. Baseline closure must recover the original
accepted merit and prepared components, with unchanged replay bounds. Verify
candidate body-energy sensitivity independently and production/Adam isolation.
Verify population velocity variance and candidate-F dependence independently.
Keep P309's geometric running measure separate: the original recipe has no
geometric running penalty. Lower scalar merit alone does not prove the original
Armijo condition or outer acceptance, and a replacement cannot inherit the
unchanged accepted-buffer `E_final=E_accept` shortcut.

## Preserve the actual accepted trajectory

Keep `on_checkpoint` read-only. A future commit hook needs a separate explicit
contract and complete owned selected state. The default private model returns
full X/V but terminal F/C only; frame export also requires the candidate's
complete F sequence (and geometric F if enabled). Retain the actual selected
terminal C; an extra rollout cannot be represented as exact original C after
the documented failed C-repeat gate. Do not replace only x or add a final
position correction absent from the evaluated trajectory.
All X/V/F/C and component losses must describe the same selected forward.

The separate preparatory `FrozenBodyWindow.evaluate(..., retain_full_state=True)`
option now captures owned, detached `F_sequence` (post-step1..T) and `F_initial`
directly from that same live trajectory before another rollout. It checks finite
values and exact F_sequence[-1]/terminal-F identity. Default behavior/returns are
unchanged. This is evidence/export ownership only; the snapshots are not an
added differentiable loss channel. The existing detached terminal C and owned
X/V must accompany them. Geometric-F sequence capture and a production adoption
hook remain absent; the currently supported raw recipe has geometric F off.
P314's already frozen `db241d0` run does not contain this later capture option.

The outer runner must receive consistent frames, end-state x/v/C/F, control
metadata and merit history. Its ordinary containment/orientation guards must
stay zero, full-state promotion and elastic assimilation must run normally,
and existing outer rejection/rollback must remain authoritative. A rejected
candidate must not donate state, optimizer moments, accepted metrics/history or
frames; its labeled rejection diagnostics remain part of the record.

## Observe actual subsequent windows

Continue in the live pipeline with normal assimilation, plan refresh, control
initialization and pin admission; do not replay an isolated zero-force window
as a substitute. Preserve window-start material IDs through the continuation.
Report their motion before pinning and the new-pin set separately: a new pin
zeros motion by construction and is not evidence of natural rest. Preserve
both free and coarse-arrived-free cohorts, since arrival is not convergence.
The current `on_commit` callback precedes pin admission; collect the later
pre/post-pin state explicitly rather than assuming it is already represented.

A same-window decrease does not establish fewer holes or rest across the morph.
Coupled continuation, full-horizon/gallery gates and all-frame raw/visual QA,
including4K appearance, remain necessary. Report actual render components,
lambda/direction norms and matched causal evidence as separate quantities.

## Read-only merit API implementation

`optimize_window(..., checkpoint_merit=True)` now provides
`packet['evaluate_merit']` only with the private raw terminal-body checkpoint
rollout. Default is off. Geometric F, Gaussian/GS, continuity and endpoint-KKT
modes are rejected, in addition to the checkpoint's existing fixed-material,
raw/no-shift/no-geometric-rest restrictions. There is no adoption API.

The evaluator validates field layout/dtype/device/finiteness, nonnegative scalar
body energy and x/X[-1], v/V[-1] equality, while requiring the caller's valid
trajectory/exact-pin evidence. It uses stored-V population variance, shared
`losses_of`/`phys_total`, candidate body energy and fixed stress/lambda. Like
production `scalars`, it combines separately converted physical/render Python
floats; independent review caught and removed a different FP32 rounding path.
Each checkpoint owns a distinct expiring lease, so an older callback cannot
become active during a later checkpoint. All mutable caches restore in finally,
and archive evidence excludes the callable.

The two real CPU MPM observer cases (layer control off/on) verify accepted-merit
closure, exact scalar recombination, body-energy sensitivity with density wu!=1,
independent population variance and F/J-volume sensitivity, invalid inputs,
render-error recovery, lease expiration even inside a later checkpoint, and
exact production/Adam isolation. Ten body-control and two prepared-reference
cases also pass. These14 distinct cases are not a full-suite or CUDA-quality
claim. P310 uses frozen b5809b5 from before this addition. P311 subsequently
closes all three original baseline merits on CUDA under the recorded raw300k
recipe (maximum allowance ratio.0292933), verifies exact physical+lambda*render
arithmetic and production-state isolation. Both terminal-strength arms still
fail raw IoU; no candidate is adopted. Coupled continuation remains outstanding.

## Actual-forward seam after P331 (implemented in P332; capability gates separate)

P331 finds no admissible candidate; do not select its least-bad rejected arm.
First validate an original-result identity control through an opt-in,
runner-owned whole-window selection seam immediately after optimize_window
returns and before warm starts or state promotion. A proposed WindowCandidate
must own the same-forward X/V/F/Fg sequences, terminal C, controls and fresh
component metrics. Capture those while the merit lease is live and verify the
donor checkpoint still matches the original final accepted optimizer state.
No endpoint-only overwrite, borrowed original metrics or invented Adam history.

The selected whole result must pass the existing runner guards, assimilation,
outer acceptance/rollback, arrival/pin admission and v/C zeroing. Continue one
ordinary successor with its actual bond rebase, layer geometry and refreshed
plan. Reuse P324's canonical step-zero observation after preparation; on_commit
precedes pin admission and cannot stand in for the successor's prepared state.
An optional zero-control replay of that actual state is additional withdrawal
evidence, not a replacement for the normal controlled successor.

Use a disposable diagnostic run for the first forward capability. An exact
same-prefix two-arm fork would additionally require an owned resumable runner
state including reversal/still/pin history, sticky plan, balancer and outer/
stopping history; the current rollback dictionary is not that checkpoint.
Do not duplicate the large post-solve commit block inside a probe.

Forward diagnosis needs no new assimilation derivative. Optimizing through the
actual handoff additionally needs a connected F-to-Fp VJP through elastic
stretch/power, volume normalization/clamp and old/new-pin exceptions, gradient-
bearing coast Fp, and a declared treatment of discrete pin/neighbor/fragment
branches and continuous layer/bond dependencies. Freezing the branches yields
a conditional derivative, not a derivative of the entire preparation policy.
The existing withdrawal adjoint does not supply these capabilities.

Before any candidate continuation, verify original-result state/policy identity,
rejection isolation and archive clocks. Track new pins separately from common
surviving free material IDs. Report successor render influence and fixed-target
outer-render acceptance separately from the donor's prepared render reference.
This seam alone does not fix holes, natural rest or the exported4K covariance.

## Candidate integration after P335

**The prior-prefix estimate route described below was rejected by P336 native1.**
Its rest volumes differed and its proposed pin set would release9046 current
old pins, before any candidate search could run. Do not repair those borrowed
arrays. P337 instead shares CURRENT runner admission/history and geometry,
and first checks an unchanged original head against its actual successor.
Its native gate is pending. A later conditional search may freeze that current
original-head preview; actual candidate admission/preparation and continuation
remain separate required gates. The remainder records the earlier design and
its still-applicable lifetime, ownership and actual-policy constraints.

P335's first FP32 native full-coast velocity/C gate failed and remains retained.
The separately labelled FP64-arithmetic/FP32-state recipe on frozen504ea6a now
passes fresh21-window native closure and an independent3577-check actual-coast
audit at N300k/T20/dt1/240/dx.3062907544/grid36^3. That scoped capability permits
an experimental conditional-model search; it is not a quality pass or adoption.
P335 owns a real prepared successor but measures only after the
ordinary run returns. Its saved owner is not a saved complete-merit evaluator.
The original live selection occurs before assimilation/admission, and the next
layer geometry and OT-u gate are prepared inside the next optimizer. Extending
the old merit lease across those mutations would not freeze its captured
configuration/target/loss dependencies.

The smallest bounded candidate experiment can use a prior identity branch's
actual successor as an explicitly labelled, owned fixed-policy estimate. In a
fresh final selection context, use its CURRENT head owner, full scalar merit,
merit covector, references and lambda. Reject incompatible materials, grid,
horizon, pin release or unsupported policies; the borrowed policy cannot be
presented as the new run's actual successor. A different prefix is not a
matched causal comparison, even with the same seed/configuration.

Supply the post-assimilation model to the existing joint search internally.
Retain its three original repeats, nonlinear complete-merit and raw-phase
gates, confirmation repeats, owned same-forward X/V/F/C and body coefficients.
The current public selection evaluation is no-grad and inspection detaches
state, so it cannot be repurposed as the gradient path. Register the confirmed
same-forward result inside the live context; do not accept arbitrary supplied
endpoint arrays or run another forward and label it the confirmed result.

The selection health contract must use the chosen model's actual coast boundary:
next pins zero v/C and anchor at head xT, while old head pins retain their old
anchors. Only the RAW HEAD is selected for ordinary promotion. Never install
predicted coast Fp/pin projections into the runner or apply assimilation twice.
No estimated-policy candidate passing all gates means the original result is
returned unchanged.

After ordinary commit, capture the candidate's freshly prepared successor and
its actual passive coast. Compare actual versus predicted policy membership,
full raw phase coverage/shape and the SAME surviving-free material IDs; report
newly imposed pins separately. Continue ordinary controlled windows as a
separate coupled-quality gate. If this later actual-policy check fails, reject
the entire experimental branch and keep the identity deliverable. It is not
an in-process fallback to the original window.

Such a fallback requires a larger transactional runner refactor: one shared
commit/preparation implementation operating on owned physical, admission,
target, balancer, plan, reversal/still/history, scale and stopping state.
The current rollback dictionary is not that checkpoint. Do not re-enter a
completed loop iteration, duplicate its commit block, or revive an expired
merit closure to simulate a transaction. No candidate is adopted by this plan.
