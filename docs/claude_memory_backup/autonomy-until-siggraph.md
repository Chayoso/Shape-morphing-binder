---
name: autonomy-until-siggraph
description: "User directive 2026-10-14 — run all experiments to SIGGRAPH level autonomously; decide routine choices myself (\"bypass\"); bring the user only truly serious decisions (changing the frozen algorithm), and only after the experiments are done"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-14T06:26:07.386Z
---

The user, 2026-10-14: "모든 실험이 완료 되고, SIGGRAPH급이 나올 때 까지 진행 해 줘. 나한테 크게 decision 할 건 일단 실험 모두 끝내고 나서 진짜 심각한 것들 (아예 알고리즘 fix된 걸 고쳐야 하는 경우가 아니면 bypass로 계속 진행 해." The user then checks back after about 6 hours.

**Why:** the paper deadline is the end of November ([[siggraph-target]]), and stopping to ask for routine choices costs days.

**Refined the same day:** "바꿔야 하는 원인이면, 큰 틀안에서 바꾸지 않는 이상 수정해서 돌려 줘".
- A definition fix that stays inside the big framework may be implemented and run without asking. It is still pre-registered and gated: the D135 rise, the end quality, the render lead and at rest must hold.
- Only a change to the big framework waits for the user: the MPM + transport + render-terms structure, the control parameterisation, or the objective's composition.

**How to apply:**
- Decide routine choices myself and record each one in docs/experiments.md: protocols, seeds, baselines, gate readings, cleanup, run order, renders, a new feature's design that leaves the frozen single-body path bit-for-bit (e.g. [[showcase-throw-scene]]).
- Queue for the user, with the evidence attached, and do NOT implement: anything that changes the frozen algorithm (freeze-2026-10-09b defaults, the D135 recipe), e.g. a D146 definition change. Keep running the evidence on the frozen code meanwhile.
- Pre-registration, no-patches and the measure-first rules still apply ([[no-parameter-only-fixes]]).
- Keep a running list of the queued serious decisions to present when the user returns.
