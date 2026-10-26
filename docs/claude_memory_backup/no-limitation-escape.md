---
name: no-limitation-escape
description: "Standing user rule (2026-10-20) — never propose \"just write it as a limitation\"; if another method exists, try it first; SIGGRAPH reviewers don't believe limitations"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-21T03:50:11.604Z
---

User, 2026-10-20: "언제나 제안할 때, 그저 한계라고 적지 마. 다른 방법이 있다면 먼저 진행한 다음에 가자. 짜피 SIGGRAPH에 한계라고 적어도 아무도 안 믿어 줘."

**Why:** reviewers read an admitted limitation as an unsolved flaw. Only a measured attempt (what was tried, why it failed) carries weight. A limitation offered as the recommended option looks like giving up.

**How to apply:**
- When something fails (gravity bunny, the late tail, the small 40k lead, runtime, D155's end quality), propose and run the next concrete method first: measure → root cause → definition change, per [[no-parameter-only-fixes]].
- Never make "record as limitation" the recommended option.
- Offer it only after the reasonable methods have been tried and measured, and then say what was tried.
- The user's earlier closure of small floaters ([[no-floating-gaussians-goal]], 2026-10-15) still holds unless the user reopens it. Serious cases are still pursued.
