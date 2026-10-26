---
name: thin-root-cause-2026-09-30
description: "2026-09-30 state of the thin-feature problem after the rollback to C_R: what is measured, what was refuted, the agreed order (measure Adam/noise/dead-state first, then continuous floor A'', beast acceptance, noise floor, c2f trigger), and the prediction that thin barely moves until the objective sees sub-cell."
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-09-30T16:35:59.039Z
---

Code is back at C_R (settled base + ratio support + target floor + loss grid following N; commit 7a0cf2f,
2026-09-30). Everything stacked after it (two-sided support, coverage term, relaxation off, u preconditioner,
per-block steps, uniform Adam) was removed as patches. Measurements stay (thin set, surface roughness, null reasons,
line-search probe). Details and numbers: docs/experiments.md ("Rollback to C_R", D5, R5, R6 partial).

**Settled by measurement (2026-09-30):**
- u has sub-cell mobility: given its own step (R6, 7 runs) it moved at spacing scale, and thin got WORSE on 7/7
  (+1.3 to +10 pt), roughness +10-20 %, silhouette slightly up. So the limit is not the controls' reach or the shared
  step; the objective cannot tell a good sub-cell arrangement from a bad one (T2: 64/96 px see no thin gap, transport
  blurred by one loss cell = 4.4 spacings at 40k; gaps are 1.6-2 spacings).
- T1's "80-95 % of end-uncovered thin points were covered once" is churn, not capability. Do not cite it as proof.
- The 300k 80-minute run = 26 s/window (finer loss grid, 2x) x 170 windows (a render-driven tail of ~1 %/window);
  MJ's 7 minutes was the log-support trap stopping the run at window 30, not speed.
- Defects of existing definitions still open: the target floor is read at the nearest target point (piecewise
  constant, step-independent jumps); the line-search noise floor 1e-7 is below the changes seen at tiny steps
  (1-8e-6, probably deterministic non-smoothness, NOT to be covered by a tolerance); c2f is tied to window 150 and
  never fires under C_R; beast collapses under plain C_R too (a committed state that is invalid under any nearby
  control; acceptance checks only the window's own 2T steps); Adam restarts every window (fact) and may act as sign
  descent on 7.2M dFc coordinates (unmeasured).

**Agreed order (user + MJ plan, 2026-09-30):** measure first (per-iteration Adam telemetry; repeated-evaluation noise
vs deterministic jumps at failed trials; dead-state probe around beast's freeze) -> fix 1 continuous same-location
floor A'' = 1/2 (t(x) - K(0)) with the gradient kept, pre-checking its negativity rate near thin tips -> beast
acceptance definition from the dead-state probe -> noise floor only after the probe separates noise from jumps -> c2f
by the existing tol/patience event, 64 -> 96 first. One change per experiment, 40k gallery + 300k dragon each.
Registered prediction: these change cost/stability, not thin (they add no sub-cell signal). If thin stays, the next
method question is the objective at particle scale (finer transport observation; a smooth surface coverage field),
not another control or optimizer.

**Why:** the u chain (u -> gate -> relaxation -> preconditioner -> block steps) was symptoms stacked on a blind
objective. **How to apply:** see [[no-parameter-only-fixes]]; do not reopen block steps or the preconditioner as
fixes; the surface roughness metric measures the outer band's depth scatter (C_R baseline 1.2 spacings RMS) and
must be redefined on the outermost layer before it gates anything.
