---
name: always-report-render-influence
description: "User rule 2026-09-22: every report/page must carry a 'rendering influence' summary (render -> control -> outcome) alongside whatever else is reported"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-09-22T17:40:07.833Z
---

**Instruction (user, 2026-09-22, verbatim):** "랜더링 영향력도 같이 한 번에 정리 해 주고 항상" — always summarise the
rendering influence together with the results, in one place.

**Why:** the project's premise is "rendering controls physics"; the user wants every deliverable to show, not
assume, that the render channel changes the control and the outcome under the recipe being reported.

**How to apply:** in every report/page/summary add a short "렌더링 영향력" block with the standing measurements:
(1) the per-window render share of the accepted control update (g_share, cosine to the physics gradient —
deterministic), (2) the outcome against the run-to-run spread: render-on vs λ = 0 twin (silIoU, chamfer, det F,
surface columns), (3) what the channel does NOT change (the physics term's level, the trajectory beyond the chaos
floor), (4) through which path (stress control at the cell scale; the kinematic u below the cell). Numbers, not
adjectives; cite docs/surface_gradient.md §10. Related: [[render-controls-physics-contract]], [[highres-150k-phase]].
