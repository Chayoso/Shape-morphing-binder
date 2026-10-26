---
name: bunny-target-streak
description: "2026-09-16 target forensic — trimesh 'base' axis fill drew 1-voxel streaks on the non-watertight bunny (a dotted line above the ear at 40k); fixed by orthographic fill + streak strip; all bunny numbers before commit 0ccc43a used the streaky target"
metadata: 
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-09-16T11:45:03.199Z
---

User spotted a thin line above the ear in the 40k renders and asked whether the target
discretisation was wrong. It was: `sampling/mesh._fill_centers` used trimesh's axis 'base' fill
first, which on the non-watertight bunny (Euler −3) draws 1-voxel columns between unrelated
surface voxels (485 streak voxels, columns of 28–67, plus 4,870 base-only slab voxels). Sampling
with replacement puts ~0.2 % of particles on them: scattered points at 20k, a dotted line at 40k.
Both n used the same 110³ fill; only visibility scaled with n.

Fix (commit 0ccc43a, deployed to hyde06 2026-09-16): 'orthographic' fill first (enclosed in all
three axis projections; zero line-like voxels, 267,451 centres = +4.3 % volume, fills interior
layers 'base' missed), then strip interior voxels with ≤ 2 of 6 filled neighbours
(`STREAK_REPORT`); `tests/test_sampler_fill.py` guards it. Batch r re-runs the deliverable arms
(20k baseline, 20k `render_ctrl 36`, 40k `--ppc 8` density recipe) on the fixed target — done
2026-09-16: chamfer 0.1568 / 0.1566 / 0.1135 (before 0.1599 / 0.1601 / 0.1142), silIoU 0.9650 / 0.9594 /
0.9541, thin-feature verdicts unchanged (per-particle 0.61→0.36, 36³ 0.52→0.23 full ear mass, 40k
0.43→0.30, ear under-fill 0.74 %). Gallery has a target-fix section + fixed-target GIFs.

**Why:** every bunny number logged before 2026-09-16 was measured against the streaky target
(chamfer/silIoU shifts at the few-percent level expected; thin-feature verdicts do not rest on
the streaks). The user's visual check caught what the metrics did not.
**How to apply:** treat pre-0ccc43a bunny numbers as "streaky-target" values; compare only
within one target version; when a render shows a straight line of particles, suspect the
voxel fill before the physics. See [[thin-feature-transport]], [[render-controls-physics-contract]].
