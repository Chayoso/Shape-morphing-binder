---
name: no-parameter-only-fixes
description: "User rule (2026-09-16, reinforced 2026-09-29 and 2026-09-30) — fix the algorithm's essence; no patches (땜빵), no parameter fiddling, no new procedure stacked beside the old one to rescue an earlier add-on."
metadata:
  node_type: memory
  type: feedback
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-09-30T16:35:39.109Z
---

2026-09-29, standing from now on: "이제부터 알고리즘 본질 수정이지 땜빵 하거나 or parameter 장난질 하지 말라고".
2026-09-16 origin: "파라미터만 바꿔서 되게끔은 하지 마". It came up while I was tuning v_max, eject_iso_k and w_esc
for mass ejection. A fix that consists of a cap, threshold, weight, budget or special-case branch is not
acceptable, even if the numbers come out right. The mechanism itself must remove the defect by construction.
Any constant that remains must come from the discretisation or the physics (a spacing, a time step, a mass, the
reference particle count), never tuned per shape or per run.

**Why:** tuned constants and local patches transfer badly across the gallery and to other N. The user wants the
pipeline to be correct by design under "rendering controls physics", and a paper-grade method, not a stack of fixes.

**How to apply:**
- Before proposing a fix, name the mechanism that causes the defect, measured and not guessed.
- Say why the fix removes that mechanism, and where every scale in it comes from.
- Prefer constraints enforced inside the algorithm (line-search feasibility, projections, conservation laws,
  gating by support, resolution that follows the sampling) over penalties, caps or longer budgets.
- A budget or tolerance change is a diagnostic control only (e.g. the long-budget run in R2), never the fix.
- 2026-09-29 again ("언제나 말했지만, 알고리즘 쪽만 수정해야해. 이상하게 위에 덧대지 마"): do not bolt a new loss term,
  weight or channel on top of the objective to cover a gap (e.g. a separate coverage term in the render channel with a
  borrowed weight). Fix the definition of the existing mechanism that is wrong (e.g. make the one-sided support
  two-sided with the same estimator). Measurements (metrics, probes) are fine; they are not the algorithm.
- Record failed parameter attempts as such in the docs.
- 2026-09-30 ("기존 알고리즘 위에 새 알고리즘 더하고 있는 거 아냐?"): a new procedure beside the existing one also counts
  as a patch, even when it is literature-backed and has no tuned constant. Example: `--block_steps`, a second
  per-channel line search added to rescue the u channel. Before any fix, ask whether it exists only to make an earlier
  add-on work (u → relaxation → preconditioner → block steps → uniform Adam is such a chain). Prefer replacing a
  defective definition (a discontinuous floor, a noise floor below the measured noise, a schedule tied to the budget)
  or removing a layer over adding one.
- 2026-09-30, the user's final form of the rule (verbatim): "'문제 A를 막기 위한 추가 알고리즘 B 설계'는 절대로 하지 말고.
  현재 코드의 어떤 부분이 문제인지 확인 후, root problem 탐색 > 그리고 나서 필요 하면 알고리즘 수정. 필요 없는
  overengineering/추가 알고리즘 땜빵은 원천 금지야." Working order, every time: (1) measure which part of the CURRENT
  code misbehaves (telemetry, probes; no algorithm change); (2) search for the root problem behind it; (3) only then
  change the algorithm, and only the definition found at fault. Never design an extra algorithm B to block a problem
  A. If a fix needs a new procedure, switch, weight or schedule beside the existing ones, it is the wrong fix.

See [[thin-root-cause-2026-09-30]], [[thin-feature-transport]], [[generalise-beyond-bunny]], [[michael-settled-transport-branch]].

**Addition, user 2026-10-04 (while approving "A", the redistribution of near-surface particles):** "항상 그냥 원인을 감추지만 말아 줘. overengineering도 금지. 코드 20줄이면 짤 껄 200줄로 늘리지 마. 항상 '알고리즘' core를 수정하는 쪽으로 가야 해." Applied: a fix goes into the algorithm's core (the rollout / objective definition where the cause is), never a cover on the display or a side mechanism; the smallest code that changes that definition; a display-side remedy that only hides a measured particle-side cause is not offered as the fix.
