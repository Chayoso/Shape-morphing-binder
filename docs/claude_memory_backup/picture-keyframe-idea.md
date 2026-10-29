---
name: picture-keyframe-idea
description: "User-liked future direction (2026-10-24): intermediate picture keyframes (\"pass through this silhouette at t=1 s\") as render-only control; set up later, after D166/D167"
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-24T06:49:37.289Z
---

**The idea.** Target pictures at intermediate times, like a storyboard: "pass through this silhouette or pose at t = 1 s". The physics objective has only the final 3D target, so the intermediate frames are owned by the render alone. That makes the render's influence large by construction.

The user, 2026-10-24: "중간 그림 keyframe 좋은데? 이거 나중에 설정 해 보자".

**Why:** the render is marginal today because the target pictures carry the same information as the 3D target, and PCGrad and λ make it subordinate to the physics. The render wins only in the physics objective's null space: colour layout, sub-cell relief that the 3D target lacks, and the cell-scale reading. Keyframes are another such null space.

**How to apply:**
- Start this only after D166 (what erases relief at release) and D167 (λ definition) are read. See [[resume-2026-10-23-rest-weight]].
- The design issue to solve first: at a keyframe window the render target differs from the final target. PCGrad then removes the render's conflicting component. Possible fixes: a time-dependent physics target, or a different combination rule at keyframe windows.
- Read 4–5 papers first, per [[papers-before-ambiguous-design]]: keyframe control (Treuille 2003, adjoint fluid control), multi-objective weighting.
- Related: the overall plan agreed on 2026-10-24 is "division of labour": physics owns cell scale and rest; the render owns sub-cell relief and appearance ([[render-own-contribution]]).
