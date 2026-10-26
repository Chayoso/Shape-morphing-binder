---
name: parked-defects-2026-09-30
description: "2026-09-30 15:05 CDT, user directive: from now on ONLY the loss reformulation (four terms: transport, rendering, stability, cleanup; 300k in 15 minutes). Everything else measured today is parked here with its evidence, not to be worked on until the loss is settled."
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-09-30T20:07:50.326Z
---

User directive (2026-09-30, ~15:05 CDT): "일단 다른 문제들은 memo 해 두고 지금부터는 무조건 loss reformulate만 하자."
Target: exactly four terms (Transport = where to go; Rendering = how it should look; Stability = settled after the
release; Cleanup = pathological strays) and a 300k run in about 15 minutes. The current-particle support is to be
replaced (not supplemented) by a target-surface coverage term. Details and numbers: docs/experiments.md.

**Parked, with the measured state (do not work on these until asked):**
- Evaluation nondeterminism: two rollouts of one control differ in the positions in every window (max 5e-6 spacings,
  never zero; loss 1e-7–5e-6 relative), from the transfers' atomics (`k_p2g` wp.atomic_add). The line search's
  noise floor (1e-7) and the commit tolerance assume finer resolution; one `commit_replay` null per run. Not a
  tolerance fix.
- Beast freeze: the head of an ejected filament (~17 particles) coasts past the target's tip into the two-cell
  domain margin around window 10–12; the global trajectory check then vetoes every candidate. Both fragment
  detectors are right (the stream is grid-coupled). Root = ejection + mass-averaged terms blind to a flung handful
  (same family as thin). Domain box and veto are downstream; do not patch them. Rate at 40k, seed 97, six runs per
  code (R11f-b, 2026-09-30 23:45): PX85 2/6, R11d (no end drift) 0/6, R11f (drift) 4/6 — codes carrying the end drift
  6/12 vs 0/6 without (one-sided exact p = 0.05); every freeze is `domain` at window 11–12 with |v|max 2.8–3.8, the
  sound runs of all three codes agree. The user parked it as a forward/viability defect, NOT a stability-loss problem
  (the drift stays: without it the end is faster and small meshes lose thin).
- Watch list (R11f adoption): A's thin is +2.7 points against PX85 in both runs (9.0, 8.6 vs 5.8, 6.4), within the
  limit and reproducible; look again after the cleanup pruning.
- Late merit alternation at 300k (±1–2 % window to window) and `reject_stop = 3` stopping runs early (R7's 300k at
  window 139). Mechanism unmeasured.
- The u channel + forward relaxation (Taubin λ-step without μ): undoes sub-cell shape, costs silhouette, keeps the
  line search smooth; R6 showed u freed without a seeing objective makes thin worse. Revisit only after the loss
  sees sub-cell.
- Adam restarts every window; not sign descent (D6 a refuted); direction weakly aligned with the gradient after
  the first iteration (cos 0.3–0.4). Not established as a defect.
- R8 (event c2f) never judged (killed at 6/19; JSONs in /data/relcfd/chayo/results_before_clean_2026-09-30.tgz).
  The trigger is in the code (repo_r15) and worked on the bunny (switch at the rejection streak, 6 windows at 96 px).
- Loss redundancy noted (three end-velocity terms, four pull-to-target terms, w_ctrl dead): superseded by the
  four-term reformulation rather than an ablation ladder.
- Surface roughness metric measures the outer band's depth scatter (baseline 1.2 spacings); redefine on the
  outermost layer before using it as a gate.

See [[thin-root-cause-2026-09-30]], [[no-parameter-only-fixes]].
