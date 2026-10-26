# D161 RESUME (updated 2026-10-22 01:40 CDT)

## Done (docs on settled-base, last commit 95ab7ac)
Response curve, then the 40k runs (coin and the smoothed bunny's flank; "PM", em 20 a, δ 1 a; seeds 97/99):
- D161 plain: FAIL by P1. The render carves a tenth (share 0.08–0.19).
- D161b (`--proximity_berth`): FAIL, and less than plain. The rule lowered λ together with the hold.
- Per-window records: after arrival λ falls 110–380× with |g_phys|; the step is not short; runs end by reject_stop.
- D161c (`--lambda_ema 0`, the four full runs): the render carves 47–58 % of the ceiling, at the ceiling's correlation, visible on the display. It fails P3/P4 (v_mean 3–11× the twin's; Chamfer +0.04 / +0.12 a).
- Conclusion: the blocking definition is the render weight's reference (the physics residual).
- Proposals for the user: (i) the render's push measured over its own floor; (ii) the placement terms at the transport's resolution.
- Local copy: `output/results_2026-10-21/d161_relief/`, with README_ko.md (sections 1–7), figures and videos (plain arm).

## Running (detached, CPU)
- D161c videos: `after_c/vid_{coin_97,coin_99,bunny_97}/pair.mp4` (PIDs in `after_c/vid.pids`).
- When done, copy them to the local `d161_relief/videos/d161c_*.mp4`.

## Next
- Report to the coordinator (Korean).
- 300k only after asking. Any next definition is the user's call.
