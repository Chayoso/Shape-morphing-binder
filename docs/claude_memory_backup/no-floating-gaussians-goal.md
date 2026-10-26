---
name: no-floating-gaussians-goal
description: "User goal 2026-10-07: eliminate the floating Gaussians; ROOT CAUSE found by D128 (the stream dilates J≈4 unresisted because the stress reads the γ=0.955-smoothed F that keeps only 4.5 % of each step's deformation); user approved the root fix (exact volume) and asked to keep tracking until solved, then FREEZE the algorithm"
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-08T00:09:45.796Z
---

User, 2026-10-07: "이제 저 float 한 gaussian을 아예 없애는 걸 목표로" → after D128: "OK. 근본 수정해야지. 이거 해결 될 때 까지 계속 확인 및 추적 해 줘. 해결 되면 이제 알고리즘 픽스 해 버리게".

**Chain of evidence:**
- D125: no particle leaves the grid support; two populations: (ii-a) transit stream to the thin parts, identical in render arm and twin (dominant on screen); (ii-b) render flakes near the target (D120 halved them).
- D126 (spray gate = grid decoupling): refuted, the opposite way (the grid gate is empty at 300k → spray off → more apart).
- D127 (render on the body's connected disc set): weak (bunny <2-pitch excess −17 %, display median −60 %, max unchanged; windows +25 %).
- D128 (fb0ff2d / 0894f7d): NOT sub-cell rearrangement (s8/sF8 ≈ body's); the stream is a CONTINUUM DILATION, J = 3.8–4.8 on the motion's own deformation gradient Fg; but the stress reads F updated as F ← 0.045 F_new + 0.955 F_old (`kernels.py` k_update, `MPMParams.smoothing` 0.955, the TVCG F-smoothing), so the stress sees J ≈ 1.00–1.01: no volumetric resistance.

**Fix chosen (D129, the user approved A over the display kernel B):** track the volume exactly (J from the grid velocity divergence) and let the stress read it; the smoothing keeps acting on the shape (isochoric) part. A/B flag first; staged 40k → 300k pairs → 40k gallery; criteria = stream J and apart counts down, end quality as D120, render lead and momentum held. When solved: adopt as default and FREEZE the algorithm (with D120; D127 if it helps on top).

Related: [[pipeline-freeze-2026-10-07]], [[frame-by-frame-particle-forensics]], [[no-parameter-only-fixes]].

**Closed 2026-10-15 ~22:00 CDT (user):** "이제 더 이상 particle 떠 다니는 거 뭐 어쩔 수 없지 뭐...막 심각한 것만 보자고".
- The small transient floaters (dragon 1–3-particle crevice beads at 4–6 s) are ACCEPTED as a known limitation of freeze-2026-10-15. D151, the last attempt, was not adopted.
- **How to apply:** no more fixes or hunts for small floaters. Investigate only serious cases:
  - pieces that persist to the end;
  - large pieces (≥ ~10 particles) or far ones (≥ ~5 spacings);
  - pieces that break a run's judge or end quality (e.g. D147 s99: 103 particles at 8.69 sp after control loss);
  - anything clearly visible in a delivered video.
- Report small ones as numbers only. The display/post-process question is deferred ([[resume-2026-10-15-night]]).
