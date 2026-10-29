---
name: near-band-role-and-next-steps
description: "User/advisor guidance 2026-10-01 ~10:00 CDT on the cleanup reformulation: the near band is kept only between the sampling berth and one loss cell (responsibility split by resolution); how to judge R12d (C's early stop does not count against it); the selection merit must be aligned with the truncated definition before final adoption; the runtime profile list for the 23 s vs 15.5 s window."
metadata:
  type: project
---

Guidance from the user (2026-10-01, after R12c), to apply when judging R12d and planning what follows:

- **Role split by resolution (the paper's framing):** 0…berth = close enough, nothing acts; berth…one loss cell
  (h_loss) = the Sinkhorn blur cannot tell positions apart → the local near band; beyond h_loss = the transport owns
  the particle. Sinkhorn = global transport, surface proximity = target-side holes, near band = body-side local
  adherence, W1 = far isolated spray. The far part of the old near band acted at 300k as a stale correspondence
  constraint, not a cleanup.
- **Judging R12d:** if heart/spot keep their thin and the 300k run speeds up, the truncation is adopted. C stopping
  near 20 windows with R11f's geometry is NOT a reason to reject it: separate "why could it not run on" (the parked
  late merit alternation / premature rejection, [[parked-defects-2026-09-30]]) from "is the geometry wrong".
- **Must be done before FINAL adoption (the user's one strong concern):** R12d's objective truncates the near band
  at one loss cell but the selection merit's common geometry still carries the unlimited near band (kept for
  attribution). After R12d succeeds, run a small probe with the selection merit using the SAME truncated definition
  (objective and accept/reject ruler must agree; a correctness fix, not a new loss; may matter for C's alternation).
  Note to raise: in the common form the eligibility is evaluated at the current state, so the merit jumps when a
  particle crosses the band's edge.
- **Runtime, after the formulation is fixed (do not chase now):** every arm with w_nn > 0 costs 23 s per 300k window
  against 15.5 s at w_nn = 0 although the term is computed in both. Profile: (1) MPM rollout time per candidate,
  (2) active grid blocks / active particles, (3) mean candidates per line search, (4) invalid / rejected trials,
  (5) fragment / contact branch counts, (6) deformation magnitude with the near band on and off.
- **R12f judging rules (user, 2026-10-01 ~12:50 CDT):** what matters is whether the W1 definition changes DECISIONS,
  not the size of the gap; the delivered BEST WINDOW is the most important of the four flips (accept/reject, improved,
  stop window, best window). (a) gap large, flips ≈ 0 → align the merit to the isolated W1 (consistency fix); (b) only
  `improved` flips, accept/reject and best window same → mostly fine to align, but check that the stop window does not
  move (plateau/patience read `improved`); (c) accept/reject or best window flip fairly often → do NOT unify; first
  analyse what invariance / common-ruler role the all-particle W1 plays in the selection.
- **Scale step diagnostic (user):** compare |∇L_W1| / |∇L_OT| and |∇L_near| / |∇L_OT| at 40k / 100k / 300k; only once
  "N grows → × how much" is measured, define the normalisation (goal = same physical meaning and gradient balance at
  40k and 300k, not sum → mean as such).
- The flaky `test_line_search_probe_is_diagnostic_only` stays a nondeterminism defect on record; do not fix it inside
  another experiment (it would blur the attribution).
- Open point I raised: the user's formula writes the near band as a mean; the code is a SUM with a constant pull per
  particle (as the W1), so its size against the transport still grows with N — the scale step for w_dt / w_nn.

**Why:** the user wants the loss fixed by measured roles, not by weights; stability is frozen (R11f).
**How to apply:** judge R12d by these rules, then the merit-alignment probe, then the scale step, R13, runtime.

See [[resume-2026-09-30-r11d]], [[loss-reformulation-target]], [[no-parameter-only-fixes]].
