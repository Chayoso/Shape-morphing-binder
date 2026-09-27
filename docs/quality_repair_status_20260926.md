# Morph quality repair status — 2026-09-26

The active CUDA migration and correctness work are implemented. The requested
combination of hole-free growth, stationary individually converged material and
clean 4K appearance is **not yet achieved**. Experimental quality changes remain
opt-in; the original delivered result and default physical recipe are preserved.

## What is established

| Requirement | Result | Evidence |
| --- | --- | --- |
| Active physics/render numerical work on GPU | CUDA path implemented and executed; immutable input preparation and file I/O remain host work. Unsupported historical modes do not silently fall back to CPU. | [GPU execution](gpu_execution.md) |
| Admitted pins stay fixed | Exactly zero subsequent displacement in full raw audits. This says nothing about the optimum of the admitted positions. | [Raw quality comparison](quality_comparison_p292.md), [full PIC-off test](no_pic_full_p292.md) |
| Free material stops at its own optimum | Unresolved. The current admission rule counts cumulative smoothed direction reversals; it is not a velocity or local-optimality test. | [Motion accounting](motion_accounting_p292.md) |
| No missing material during growth | Unresolved. Removing endpoint XPIC improves thin-tip supply but loses the silhouette gate. Neighborhood density and target coverage do not certify watertightness. | [Full PIC-off test](no_pic_full_p292.md) |
| Stable detailed 4K appearance | Unresolved. Tested support/material-normal changes failed their temporal gates. Export does not optimize Gaussian appearance against a 4K loss. | [Renderer experiments](render_artifacts_p292.md), [motion versus appearance](motion_vs_appearance_20260926.md) |

All P292 physical numbers below use bunny N=300000, T=20, dt=1/240,
dx=0.3062907544 world units, a 36-cubed loss grid, eight inner iterations and
the original mixed60 source/target. A cap of 60 windows retains the 300-window
schedule; these are not 60-window schedule retunings. Motion units use the
source native nearest-neighbor spacing 0.0349885366 world units.

The full PIC-off test changes tip-ball count from 53 to 87 (sampled target 89)
and top-region mean density from 0.90748 to 0.97712. Silhouette IoU changes
0.973039 to 0.966396, below the 0.971 acceptance gate. A common 1037-ID free
surface cohort has less raw-step movement but more accepted-window displacement
and reversal. This is a tradeoff, not the requested fix.

## Implemented correctness changes

`outer_render_committed` measures the fixed-target outer silhouette merit on
the positions actually promoted after endpoint corrections. Only accepted
candidates advance the previous track, and render-resolution changes clear
that track. Direct GPU recomputation matched all eight measured commits.

`render_paced_arrived` is a separate opt-in experiment: an accepted state whose
particles all satisfy the existing frozen-plan arrival test latches the fixed
render reference for subsequent solves. Rejected candidates cannot trigger it;
the latch survives a resolution rebuild. It does not change pins, patience,
arrival radius or termination, and does not make arrival proof of target coverage.
The complete silhouette/shading reference changes, including its normal grid.
The [controlled full pair](render_handoff_p292.md) activated it at accepted26;
solves27–37 used the fixed reference. At common commit37, silhouette IoU was
0.966288 versus0.968164, still below0.971. The same423 free IDs retained drift.
Substantial divergence before activation prevents a causal effect-size claim.

## Local requested artifacts

Original lossless 3840-by-2160 PNGs, using raw simulation frame numbers:
`output/c291/photoreal4k/original_selected_raw_frames/raw_0000.png`,
`raw_0036.png`, `raw_0108.png`, `raw_0420.png`. These are byte-for-byte copies of
the original renderer outputs, with hashes and individual visual inspection
recorded in that folder's `manifest.json`. Existing artifacts remain visible.

The PIC-on/off diagnostic MP4s are under `output/p292/no_pic60_overview/`.
All 362 encoded frames were inspected; see [visual report](no_pic60_visual_p292.md).
These videos do not replace the original result or override the failed raw gates.

## Structural issue still to resolve

The inner optimization evaluates the rollout endpoint, whereas endpoint XPIC
and shifting subsequently change the delivered positions without reconciling
stored velocity, affine momentum or deformation. Stored terminal velocity,
actual last-rollout displacement divided by dt, and the final saved transition
divided by dt are different quantities. Merely lowering stored velocity or
freezing more particles does not resolve that mismatch.

Following the failed handoff gate, the opt-in `commit_pic_objective` prototype
implements a fixed-stencil XPIC operator and its exact transpose, with owned
promoted endpoints shared by gradient, candidate evaluation, replay and runner
promotion. Other external position operators are disabled. Scope guards and CPU
operator/integration tests passed independent review. The CUDA operator and
one-window integration gates passed; the matched eight-window runs each completed
eight accepted commits with zero state guards and exact candidate endpoint
agreement. See [the shared-endpoint report](shared_endpoint_p293.md).
See method §10.40 for supported terms and calibration conventions. This aligns
an explicitly hybrid endpoint map, not physical v/C/F. Actual geometric-rest
supervision remains a subsequent change; no quality improvement is asserted.
