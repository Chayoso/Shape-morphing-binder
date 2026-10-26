---
name: generalise-beyond-bunny
description: "User rule 2026-09-24: every fix (oscillation, ear/thin features, reconstruction) must generalise beyond the bunny — verify on the whole 40k gallery (and the other N) before calling it adopted"
metadata:
  type: feedback
---

The user (2026-09-24 15:20, while the Rprop twins ran on the bunny): "이거 bunny 뿐 아니라 일반화가
되어야 하는 거 알지?" — the fixes must hold on all targets, not just the bunny.

**Why:** the bunny is the development case (the 300k defects were found there); the deliverable is
the gallery (10 targets at 40k, plus the higher N), and a mechanism tuned by the bunny's readings
is exactly the "per-shape fix" the no-parameter rule forbids.

**How to apply:** after a mechanism passes its bunny pre-registration, run the same flag on the
full 40k gallery (cheap: ~2 min per target) and on the other 300k targets available, with the same
metrics (silIoU / det F / layer breathing / stride-19 flicker / bump) against the g41 gallery; a
fix that helps the bunny and hurts another target is not adopted — it goes back to the mechanism.
Record per-target tables in docs/experiments.md. Related: [[no-parameter-only-fixes]],
[[until-zero-oscillation]].
