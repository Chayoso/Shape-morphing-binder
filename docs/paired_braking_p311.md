# P311: paired terminal-strength comparison with original-merit reporting

Status: fresh paired CUDA comparison completed; both arms reject all candidates.
Original-merit CUDA closure passes. No candidate has been adopted.
P310 exhausts its same-origin displacement schedule without a joint feasible
candidate. Its fixed terminal05 already introduces a prepared-volume deficit.
The existing preliminary terminal025 has a smaller deficit but is not uniformly
better in raw quality. Compare these two strengths within one fresh callback;
do not compare separate stochastic W20 realizations as a causal ablation.

Keep N300000,T20,dt1/240,dx.3062907543956724wu,loss36^3,budget8,
raw/no-PIC/no-shift. Enable the read-only original-merit evaluator from85e4b5d.
Use the actual newly generated terminal05 and terminal025 coefficients from
the same preliminary schedule. No old state/archive admission or production
candidate commit. The failed C-repeat gate remains failed.

Prepare ONE original three-replay baseline before the arms. Both arms share
the same original controls, baseline records/ceilings, initial state, reference,
lambda, source-upper and window-start cohorts, raw metrics and original-merit
reference. Neither arm may independently recalibrate or expand those ceilings.
Explicitly verify these identities before/after each arm. Observe each arm's
own three terminal-running repeats and report its noise/floor separately.
The shared baseline is an owned immutable package, not one mutable rows list.
Deep-copy only arm-local result lists; verify shared package hashes/values and
the actual original displacement/selected terminal roles before/after each arm.

Run terminal05 then terminal025, both beginning at the original displacement.
Each arm uses the unchanged P310 search: one origin,11 radii, up to two replaced
model-remainder corrections per radius, at most33 search forwards, and all
original data/resolved-running/P306 gates before acceptance. Data-pass/raw-fail
continues to a smaller radius without changing origin. The first all-gate
candidate gets three fixed-coefficient repeats; stop that arm even if a repeat
fails. Always run and preserve the other arm, even if the first succeeds.
Both arms therefore use at most66 search plus6 fixed-candidate repeat forwards;
preliminary scheduling/baselines/running-noise observations are separate.

All same-forward valid candidates and original repeats get original-merit
reports from their actual x/F/v/V/body energy. Do not include P309's running
objective in that original merit. Require the shared original baseline merit
to recover the accepted history scalar within the existing32eps relative
closure convention; fail if the evaluator cannot close it. Verify exact
reported recombination of physical plus lambda times render. Preserve full
merit, prepared channels, stored kinetic/variance, body energy, actual wu and
lambda. Report original-merit nonincrease against the same original baseline
separately; do not relabel it as part of the unchanged P306 gate or as an
Armijo/outer-acceptance certificate. No new production policy follows.
Check all three baseline closures individually, not their mean. The report-only
nonincrease label uses the fixed maximum of those three original replay merits,
with accepted-history delta also reported. Do not add the32eps closure allowance
to this ceiling. Report-only merit failure does not alter the P306 search-stop
rule or prevent running the other arm.

Retain original-only and each arm's selected X/V/F/C/control evidence, input
and source hashes, actual render components and exact post-callback production
isolation. A positive result still needs actual full-state adoption/normal
assimilation/next-window validation under `candidate_commit_contract.md`.
No post-arrival rest, full-morph no-hole or4K-quality claim follows here.

Do not add separate silhouette/PBR constraints in this experiment. Their
opposing changes are informative, but P309 also shows increased prepared
silhouette with improved raw IoU. The existing evidence does not isolate
component compensation as the cause of raw quality loss.

Implementation: `scripts/probes/paired_braking_repair.py` builds one
`SharedRepairBaseline` whose records and original coefficients/C are owned
bytes. Each arm decodes independent records and tensors. Content hashes bind
the numerical spec, fixed controls/basis/gate, actual accepted history/state,
cohort masks, references, source/target and the original objective's prepared
inputs, lambda and unit weight before/after both arms. Finished OT solver
scratch and mutable adjoints are excluded; the actual frozen plan/pace target
is included. Hashing copies data to the host for evidence I/O only.

An invalid selected-terminal forward is retained as a failed arm; the other
arm still runs. Baseline failure or context mutation aborts the comparison.
Both successful and failed arms preserve their own artifacts and report-only
merit labels. Preliminary terminal candidates also report original merit.

Local validation: four paired diagnostic cases plus four existing remainder
cases pass, including immutable package ownership, actual context changes,
both-arm execution after first-arm success/invalidity and deliberate original
merit worsening that cannot alter the P306 selection. Two actual Warp-CPU
observer cases (layer off/on) pass merit closure, callback isolation, expired
lease rejection, injected evaluation failure and stable numerical bindings.
The related ten body-control, two prepared-reference and fourteen affine
operator cases also pass (36 focused cases total). Compile and whitespace
checks pass. These checks do not substitute for CUDA baseline closure or
physical quality.

Independent review reran the eight paired/remainder cases and found no blocker
for the recorded raw recipe. Its completeness note is addressed by binding the
actual render-balancer active flag and dynamic target OT neighbors in addition
to the config/prepared inputs. Review approval covers diagnostic execution only.

## Completed paired_repair1: both strengths fail only IoU after data restoration

Frozen bdd5d46 ran on hyde06 GPU1. N300000,T20,dt1/240,
dx.3062907543956724wu,loss36^3,budget8,raw/no-PIC/no-shift are unchanged.
There are20 ordinary commits/160 accepted inner updates, all guards0 and exact
post-callback production isolation. Both arms share39,852 original free IDs,
including39,649 coarse-arrived-free IDs. All shared package, numerical context
and selected coefficient checks pass before/after both arms.

All three original-merit repeats recover accepted history; maximum closure
ratio.0292933 against the existing32eps allowance (pass <=1). Their merits are
.0020523923648060128,.0020523923648060128,.002052392131975369; the fixed report-only
ceiling is the maximum, with no added tolerance. Accepted history is
.0020523921354619733. Exact separately rounded physical+lambda*render
recombination holds throughout. This closes the new API on this CUDA recipe;
it does not repair the older failed C-repeat/archive-admission gate.

| Arm | Search records / forwards | Data-restored candidates | Accepted / repeated |
|---|---:|---:|---:|
| terminal05 |30 /29|4|0 /0|
| terminal025 |32 /31|2|0 /0|

Each arm has one half0/correction2 no-feasible-active-set result. All60 evaluated
trials fail unchanged P306 gates, including retrospective application. All six
data-restored candidates also have resolved running decrease and lower original
merit, but fail raw binary silIoU alone. The smaller terminal strength therefore
does not resolve joint feasibility in this bounded search. Original-merit
nonincrease holds for27/29 and29/31 trials respectively; this report did not
affect either arm's search or stopping.

Representative data-restored candidates, relative to shared baseline0:

| Quantity |05 half6/correction1|025 half5/correction2|
|---|---:|---:|
| Arrived-free running mean square | -3.3761% | -3.4997% |
| Net / saved-step RMS | -1.3223% / -1.7025% | -1.2905% / -1.7655% |
| Stored / geometric terminal RMS | -26.2408% / -26.2425% | -13.9596% / -13.9610% |
| Original merit delta | -9.64847e-7 | -8.14907e-7 |
| Raw silIoU delta | -2.18477e-6 | -3.21587e-5 |
| Upper / overall reference coverage | +2/15312 / +5/300000 | +2/15312 / +4/300000 |
| Fixed-source upper density | +7.44934e-5 | +1.11740e-4 |

Both retain tip count and improve Chamfer. These small binary-silhouette failures
are not alone proof of a visible4K hole; they still reject the candidates under
the preregistered gate. No fixed-candidate replay branch or production adoption
ran, and no coupled continuation or persistent-rest inference is available.

Rendering: at common lambda.01497485162913 the representative05 candidate's
prepared silhouette increases2.01049e-7 while PBR decreases2.00933e-7. The025
candidate changes+1.50991e-7/-1.50932e-7. Each combined render is2.32831e-10 above
baseline0, within the unchanged shared repeat ceiling; weighted change is
+3.48660e-12. The original merit decreases despite that prepared silhouette
tradeoff. Ordinary optimizer direction share has median.4903952 and lambda
median.0668099; these are not causal motion shares or4K visual measurements.

Evidence: `docs/evidence/p311` contains bound result/protocol and the scalar-only
audit. Full raw X/V/F/C/control and immutable-baseline artifacts remain under
hyde06 `work/p303/paired_repair1`. The audit recomputes finite evidence, common
ceilings, individual merit closures, P306 and resolved-running conditions,
both-arm results and report-only merit labels. Independent result review matches
all70 numerical source files and7 helpers and reconstructs all halving/correction
transitions, replaced remainders, trust checks and frozen running thresholds.
It confirms the scalar results, without independently remeasuring raw-array
sidecars or rehashing the large input archives. Prepared data and the separate
original-merit reevaluation differ by up to3.49e-10 in volume/2.33e-10 in render,
within the existing32eps parity checks; their channel values are not bitwise
identical. Exact recombination means each merit's own physical+lambda*render.
Full-morph holes/rest and high-resolution Gaussian artifacts remain open.
