---
name: siggraph-target
description: User goal 2026-10-09 — target SIGGRAPH (technical papers) with the frozen pipeline; the evidence roadmap and the reviewer attacks to close before submission
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-14T05:48:26.438Z
---

On 2026-10-09, after the D137 freeze decision, the user set the goal: "SIGGRAPH를 목표로 하자 우리". The next main deadline is presumably SIGGRAPH 2027 technical papers, usually late January; check the official date before planning backwards.

**2026-10-14 user deadline:** "최대한 10 - 11월 내로 paper 끝낼 수 있게 하자", i.e. finish the paper by the end of November 2026, well before the January deadline. The official SIGGRAPH 2027 date was not yet announced on 10-14.
- Plan: evidence runs by ~11-07, figures and videos by ~11-15, writing through 11-30.
- The evidence plan is D145 (agent, plan first).
- The late wobble forensic is D146: a per-window pulse at 1/6 s in the deliverable after ~7 s. The dragon's late frame-to-frame change stays ~0.085, about 3× the bunny's.
- Tag deliverable-2026-10-13 (592e97a) = 09b defaults + exact runtime batches + scripts/render_deliverable.py (`+field=zbs` display).

**Why:** the user is anxious about acceptance and wants the work to reach a top venue. This project is the follow-up to their own TVCG 2025 F-control paper and PhysMorph-GS ([[competitor-landscape]]).

**How to apply:** after freeze-2026-10-09, order the work so that every piece serves the paper.
1. Runtime to ~10 min per 300k run ([[runtime-10min-goal]]).
2. An ablation ladder: u, relaxation, volume-conserving body, cell-from-N, render on/off.
3. At least one runnable external baseline: ICLR 2025 implicit velocity morph or 4Deform. The Xu objective in our simulator (D112/D118) is not enough on its own.
4. Seeds and spread on the gallery.
5. Final videos and figures in the studio look ([[render-look-default]]). The D115 display layer must be disclosed as display-only, with metrics on the base discs and with/without shown side by side.
6. A crisp 3-line contribution statement against the user's own prior papers.
7. A limitations section: the single-field rupture at topology changes (the dragon crevice remnant at 1.9–2.5 s), bob's sub-pitch rim shift, beast's small twin lead.

Reviewer attacks to close: u = a hand edit, videos ≠ sim, the small 40k IoU gains, speed, the display layer. See [[pipeline-freeze-2026-10-07]] and [[d137-cell-from-n]].

**User, 2026-10-21: "SIGGRAPH에 내야 하는거야. 항상 명심해."** Every decision is judged by whether it makes the SIGGRAPH submission stronger: visible results, credible metrics, no "limitation" escapes ([[no-limitation-escape]]).
- Also requested: many diverse 300k examples (a showcase gallery for the paper and video), D156 (image-only target), and the Chapter 4 TODO fill.
