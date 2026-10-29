
## D175d read (2026-10-29 13:46 CDT; auto/200_w/summary.md; PREREG_d.md 13:32; gate f PASS: G1-G4 bit-SAME, 6 unit tests)

**VERDICT: FAIL. W99 passes every bar; W97 passes P1 and P2 and fails P3 on one serious piece.**
- W97, with no segment window lost (0 holds, every segment commit kept):
  - KF-IoU 0.952 (held-out 0.953); G 0.753 / 0.781;
  - 3D IoU 0.929, +0.173 over the twin;
  - aspect 1.368 at t_k.
- W99 (identical to X99 through the segment):
  - KF-IoU 0.929; 3D 0.903, +0.148 over the twin;
  - aspect 1.308.
- So the replay fix alone closed W97's keyframe gap: +0.095 → +0.173.

**The piece:** 10-12 particles, 1.5-1.8 pitches off the body, at the bunny's ear-tip height (y +2.45 from the centre).
- It sits in the final-target epoch, frames 612-756 and 816-840 (t 2.55-3.50 s), long after t_k = 1 s, and re-merges
  by frame 852.
- It is at the 10-particle mark. For comparison, the largest piece: D97 6, T97 7, X97 9.

**Render share per control block** (λ_b |PCGrad(g_p, g_r)_b| / (|g_p,b| + λ_b |·|), first gradient of each window):

| | dF_c | u |
|---|---|---|
| inside the segment (W97 / W99) | 0.561 / 0.581 | 0.999 / 0.999 |
| after t_k (W97 / W99) | 0.245 / 0.243 | 0.962 / 0.948 |

Order 2 (the outer reject against the hold) is not run: W97 had no segment hold or reject, so it does not address this
miss.

**Next measurement (no definition change):** forensics on the ear-tip piece, frame by frame.
- Its particle ids' history from t_k on.
- Which term's position gradient pulls them off (a term dump over windows 15-21).
- Whether D97's own ear tips carry the same set at 6 particles. That would make the stretched start push it over the
  mark, rather than the keyframe comparison.
