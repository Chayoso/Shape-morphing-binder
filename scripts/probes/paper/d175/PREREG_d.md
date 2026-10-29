# D175d pre-registration: the count comparison (D175b) with the segment replay measure

Written 2026-10-29, after the D175c read (output/gpu/D175_SUMMARY.md) and before any D175d run. Coordinator's order 1.

## What is run

- Arm W: D175b's X flags, `--keyframe_compare column`, the soundness terms kept (`--keyframe_soundness every`, the
  default). D175c's `final` is dropped (it overshoots).
- Code: wip/d175 at or after 20ad30f8. It carries the segment replay measure: inside keyframe segments only, the commit
  check reads D164's max(start pair, the window's taped pairs, one float32 ulp of E). It also carries records only: the
  replay and line-search splits, and the per-block render share added for this run.
- No new definition: D175b's comparison, storyboard, held λ (both blocks in a segment, Hu after t_k) and bars.
- Seeds 97 / 99, 40k, sphere → bunny, keyframe E at t = 1 s, Hu flags.
- The gate is re-run for the record-only addition: G1-G4 bit-SAME (defaults, Hu, Hu on the coin, the settle with
  relaxation) and the unit tests.
- References reused from job 208: D97 / D99 (no keyframe) and T97 (λ = 0).

## Bars (PREREG.md / PREREG_b.md, unchanged)

- **P1:**
  - G ≥ 0.6 on the run views and ≥ 0.5 on the held-out views;
  - 3D IoU(W) ≥ 3D IoU(T97) + 0.10;
  - KF-IoU(W) ≥ KF-IoU(D, same seed) + 0.05.
- **P2:** final silhouette IoU ≥ min(D) − 0.005; Chamfer ≤ 1.1 × max(D); thin_uncovered ≤ max(D) + 0.02.
- **P3:** end v_mean ≤ 1.5 × max(D); Jmin_traj > 0 at every window; no serious piece.
- **PASS** = P1, P2 and P3 on W97 and W99.

## Reported as well

- The aspect at frame 240.
- The segment windows' holds and replay verdicts.
- The render share per control block, from each window's first gradient:

  λ_block |PCGrad(g_p, g_r)_block| / (|g_p,block| + λ_block |PCGrad(g_p, g_r)_block|)

  for dF_c and u, as means over the segment windows and over the final target's windows.

## Prediction

- With no window lost to a replay null, W97's 3D gain reaches ≥ +0.10. X97 reached +0.095 having lost 3 of 6 segment
  windows, 1 of them to the replay null.
- The two exhausted line searches (the soundness brake measured in auto/200_ls) may still cost W97 windows.

## If W97 still misses

Order 2: pre-register the outer-reject definition, then run it on top of W. Inside a segment, the outer reject compares
the candidate with the zero-control rollout from the same start (the hold the clock would commit), not with the epoch's
best.

## If W passes both seeds

Order 3: the 300k showcase.
- Bunny → dragon, with a pose keyframe: the bunny stretched 1.3× in y about its centroid at t = 1 s, `--keyframe_compare
  column`.
- Full and twin.
- Frames at t_k against the keyframe, and the twin | full video.
