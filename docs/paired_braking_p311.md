# P311: paired terminal-strength comparison with original-merit reporting

Status: driver implemented; CPU checks and independent implementation review
passed; fresh CUDA run pending. No candidate has been adopted.
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
