---
name: papers-before-ambiguous-design
description: "Standing rule — for any ambiguous design decision, find and read 4-5 related papers (2023-2026) BEFORE implementing; especially anything touching gradient/constraint propagation"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: f83ab3dd-27cd-4a63-8354-f08614b6a754
  modified: 2026-09-02T00:24:48.239Z
---

The user (2026-09-01): "항상 애매한 건, 관련 논문 4-5개 찾아봐서 읽어보고 진행 해. 특히 propagation 부분."

**Why:** The project (PhysMorph improvement, TOG ambition) pre-registers mechanisms against literature; ad-hoc design choices without anchoring have repeatedly been caught by adversarial rounds. Propagation (how render gradients reach/move particles — surface-band local-global, sparse-fringe pull) is the area the user flags as most ambiguous.

**How to apply:** Before implementing any non-obvious mechanism (loss form change, solver variant, propagation scheme), run a literature search for 4-5 recent papers, read the relevant parts, and cite them in docs/rationale.md pre-registration. Related: [[structural-fixes-only]] ("parameter만 바꾸는 짓은 하지마"), adversarial verification via Codex gpt-5.6-sol xhigh + Opus at every tranche.
