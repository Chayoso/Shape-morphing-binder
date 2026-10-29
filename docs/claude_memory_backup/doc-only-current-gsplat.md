---
name: doc-only-current-gsplat
description: "user rule 2026-09-25 — the report page shows ONLY the current results, rendered as Gaussian splats (no mesh/remesh videos, no history of refuted runs on the page)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-09-25T22:42:00.184Z
---

From 2026-09-25 17:40 CDT the user wants the deliverable page (the claude.ai artifact) to carry only
the CURRENT forms and the CURRENT render: Gaussian-splat videos (the `render_splat_video*.py` v6
settings: adaptive radius, support opacity, normals at 3 spacings, hemispheric lighting). The old
Poisson / marching-cubes videos and the history of refuted forms were wiped from the page (kept
in docs/experiments.md and the server's r300/ folders).

**Why:** the user judges by the splat look ("bm300 with the splat render is the best result"); a
page full of superseded videos and mesh renders hides the state; the history belongs in the log.

**How to apply:** when a new run becomes current, render it with the v6 splat settings and replace
the page's video; keep the page to: the delivered render's definition, the chosen form's numbers,
the measured open defect(s), the running experiments with their pre-registered gates, and the
plan. Republish with the same artifact URL; null old files in `files` when wiping. See
[[resume-2026-09-25-pin-gates]], [[always-report-render-influence]].
