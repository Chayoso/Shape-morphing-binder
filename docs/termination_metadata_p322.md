# P322: explicit optimization termination metadata

The runner's historical `converged` value is the global `frozen` flag. It can be
set by gradient stopping, plateau, rejected candidates or a displacement-cycle
rule. It does not establish that individual particles are physically at rest.
P322 preserves that value and every stopping condition. It adds the result's
`termination` dictionary and replaces misleading runner log descriptions.

| Reason | Existing trigger in `physmorph/pipeline/runner.py` |
|---|---|
| `window_start_gradient_stop` | No accepted inner history and `stats.grad_converged` |
| `null_commit_patience` | Null commit raises the existing stale count to patience |
| `outer_rejection_patience` | The rejection branch reaches stale patience |
| `outer_rejection_streak` | Consecutive rejected candidates reach `reject_stop` |
| `accepted_track_plateau` | Accepted raw loss tracks fail the existing progress test for patience |
| `displacement_cycle` | The enabled net/summed displacement rule reaches its existing count |
| `manual_window_cap` | No freeze; a positive `stop_after_windows` smaller than `animations` ends the loop |
| `configured_window_budget` | No freeze; the configured loop budget is exhausted, including zero attempts |

The first triggered rule supplies `reason`; `triggers` retains both rejection
rules when they fire on the same attempt. A freeze on the final budgeted attempt
keeps its rule reason. `attempt_index` is zero-based and `attempt_number` is
one-based. For budgets they identify the last optimizer call, or are null when
there were no calls. Copied holds, dressing suffixes and C2F history events do not
increment `optimizer_attempts` or overwrite the triggering attempt.

`optimization_stopped=True` describes a normal returned call. It does not say a
stationary point was found. `stopping_rule_triggered` equals the preserved legacy
`converged` flag; it is false for a budget-only return. Every result explicitly
states `individual_rest='not_evaluated'`. Existing pin admission, held frames and
best-commit delivery truncation retain their original semantics. Exceptions still
raise and do not acquire a success/termination record.

No config, objective, state, gate, loop bound, history row, callback or saved
frame is changed. The standard `pipeline_run.py` JSON exporter and GPU/full-horizon
diagnostic exporters preserve the new dictionary beside legacy stop fields.
Historical JSONs and other serializers are not
automatically rewritten. Frozen P320 jobs continue using their original source
and are unaffected.

Validation uses scripted optimizer outcomes through the real runner for every
trigger, overlapping rejection triggers, budget/manual cap, zero attempts,
last-attempt freeze and held/dressing suffixes. The existing hold-bookkeeping
regressions remain applicable. A separate before/after comparison runs the real
CPU optimizer with N160,T3,two inner iterations,two windows, mixed body/stress,
layer control/relaxation, render guidance and C2F, for raw and shared-PIC modes.
Every previous result field is recursively exact (450/456 comparisons), including
byte-exact arrays; only the new dictionary and intended logs differ. This is a
reporting parity check, not new physical-quality evidence.

The real optimizer parity receipt predates one final metadata-only refinement:
cycle evidence now uses Python `k_cyc ** -0.5` instead of backend
`1.0 / np.sqrt(k_cyc)`, making it serializable under the CUDA array backend without
an additional device scalar read. The physical cycle predicate is unchanged.
Reversing precisely that single text replacement recovers the tested source SHA;
both hashes and the historical receipt are in `docs/evidence/p322`. The current
source separately passes16 dedicated plus6 bookkeeping cases, independently
reviewed, including strict JSON/native scalar checks. Root also passes20
termination/real-trace cases after diagnostic export propagation. Do not present
the older receipt as execution of the final source bytes.
