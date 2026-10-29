---
name: all-means-until-solved
description: User directive 2026-09-25 — keep going by every means until ALL the problems (oscillation, thin-feature droplets, surface smoothness) are solved; no global dx reduction as the answer (cost + ppc), resolution via N / adaptive thin-feature carriers instead
metadata:
  type: feedback
---

The user (2026-09-25, after the pin's stream findings): "일단 뭐...계속 진행 해 줘. 모든 문제 해결 될 때 까지 수단과 방법을
가리지 마." and asked whether the paper side is exhausted.

**Why:** the three defects are the blocker for the deliverable; the user wants closure, not a report of limits.
The dx question was settled the same day: a global dx reduction is not the answer (adjoint grid memory ×16 for
dx/2, ppc collapses without N ∝ dx⁻³; the 300k native-dx run c300 did not fill the ear); resolution comes with N
(native dx at 300k, `--disc_ref` off if dp300 clears it) and with thin-feature-only carriers (Ando split, R-2).

**How to apply:** keep iterating mechanisms with pre-registered predictions (REFUTE rule stays), on the whole
gallery ([[generalise-beyond-bunny]]), until the acceptance state ([[until-zero-oscillation]],
[[after-oscillation-droplets-smooth]]). Papers: the surveys in docs/related_work.md cover render→physics,
thin-feature transport, particle→surface, shifting/oscillation, geometry, MPM↔grid↔surface, droplets/filaments,
temporal coherence, persistence, competitors; the remaining gaps to read when a mechanism is ambiguous: rigid /
kinematic boundaries with SLIP in MLS-MPM (CPIC, Hu 2018) for the pinned body as a wall, and adaptive-resolution
MPM for thin features. Never stop at "it can't be done"; state the cost and pick the next lever.
