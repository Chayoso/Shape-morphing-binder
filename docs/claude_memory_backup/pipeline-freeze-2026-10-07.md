---
name: pipeline-freeze-2026-10-07
description: "User decision 2026-10-07 13:30 CDT: the pipeline is FROZEN at D105 recipe + D120 (lambda EMA off) + D123 reading (exterior radius 2.5); no more mechanism hunting for sub-pitch relief; from here only paper-evidence experiments (ablations, gallery, seeds, baselines) and no recipe tuning"
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-07T18:42:39.268Z
---

User, 2026-10-07 ("일단 D123을 기준으로 가야 하는 게 맞는 거 같아 … 파이프라인을 고정 시키고"): freeze the pipeline; the "physics makes sub-pitch wrinkles" line (D122 and its successors) is closed as future work; the 300k sampling limit is stated in the paper.

**Frozen state:** settled-base HEAD f2e3401. D105 recipe (`--layer_relief`, min_spacing 0.9, render_exterior) + `--lambda_ema 1` (D120; to become default after the 40k gallery) + display/evaluation at `--exterior_radius 2.5` (D123 reading; the loss's field stays 3). Flags `--surface_density` (D122, refuted), `--relax_in_transit` (D119, reverted), `--u_off` (D124 ablation) are experiment-only.

**Reviewer analysis (my reading, agreed by the user):** two attacks must be closed: (1) "u is a hand-placed position edit" → D124 u = 0 ablation + hold ratio (the tail layer's normal excursion 69–72 % held, d116_pulse.py); (2) "the videos are not the simulation" → main videos/figures with base discs at r = 2.5, the D111/D115 detail layer only as labelled supplementary. Also: never claim "render beats twin on every metric" (bunny end kinetic energy is above the twin's in D105 and D120); claim "no momentum injected" with the COM/rotation numbers.

**Evidence plan after D124:** D120 40k gallery → default; 300k seed spread (2 more seeds, both arms); ablation ladder (released-end scoring off, PCGrad off, relaxation off, min_spacing off, Xu loss + our windows); one external baseline (ICLR 2025 implicit velocity morph or OT interpolation); regenerate videos/figures at r = 2.5 base discs. No recipe tuning to hit numbers ([[no-parameter-only-fixes]]).

Related: [[physical-detail-not-display-trick]], [[render-look-default]], [[competitor-landscape]].
