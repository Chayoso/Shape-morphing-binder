---
name: intuitive-visualization
description: "User rule 2026-10-14: every numeric visualization must read at one glance — one question per panel, plain-word legends, no per-window rescaling or flicker"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-14T21:13:56.662Z
---

User, 2026-10-14, on the render-gradient videos: "수치들 시각화 할 떄 '직관적'으로 바로 이해할 수 있게 해 줘. 딱 보면, 아 저 색은 dFc에 render gradient가 영향을 주는 곳이구나 … 크기가 저렇구나 … 충돌하는 부분이구나 … render에 의해 dFc가 이렇게 바뀐 거구나".

Earlier complaints on the same videos:
- "10의 0승이면 1 아냐? 저거 색을 어떻게 읽어야 해?" The bar was a log ratio to a per-window p99, and nobody could read it.
- "너무 깜빡깜빡거려". Values were held per window and the reference was rescaled per window, so the colours jumped every 3 frames.

**Why:** the videos are proof for the paper and for the user. A colour that needs the formula to be read proves nothing to a viewer.

**How to apply:**
- Each panel answers ONE question, and its title is that question.
- Legends say what a colour MEANS ("orange: render dominates", "red: conflict"); the formula goes in small print.
- Prefer linear % or shares over log ratios. Use one fixed scale per run; never rescale per window.
- Interpolate between window values in time, and label it.
- Draw "not active here" separately from "zero".
- Send the legend text to the coordinator before rendering.

Applies to every figure and video, not just RGV. See [[render-look-default]] and [[frame-by-frame-particle-forensics]].
