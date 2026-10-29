
## D175b read (2026-10-29 12:54 CDT; output/gpu/d175/auto/200_col/summary.md; gate c PASS, G1-G4 bit-SAME)

**VERDICT: FAIL. X99 passes every bar; X97 fails P1.**
- X99:
  - KF-IoU 0.929 (held-out 0.929); G 0.634 / 0.670;
  - 3D IoU 0.903, +0.148 over the twin;
  - aspect at t_k 1.308 (the silhouette comparison gave 1.074).
- X97:
  - KF-IoU 0.877; G 0.367 / 0.399;
  - 3D IoU 0.851, +0.095 over the twin (bar +0.10);
  - aspect 1.173 (the silhouette comparison gave 1.059).
- P2 and P3 pass on both seeds.

**The count picture moves the bulk:**
- the aspect is 1.17 / 1.31 against 1.06 / 1.07;
- the 3D gain over the twin is +0.095 / +0.148 against +0.058 / +0.060.

**Why X97 falls short: it lost 3 of 6 segment windows, and 2 more took one step each.**
- Window 1: a commit_replay null. ΔE is 2.8 × the tolerance, 100 % of it the render: the (b) mechanism, the render's
  replay at the 1e-7·|E| floor.
- Windows 3 and 5: ls_exhausted, 10 trials each failing the merit.
- Windows 2 and 4: one accepted step each.
- The count gradient opposes the soundness gradient (g_raw_cos −0.82 … −0.85; X99 −0.72 … −0.89 from window 2 on). After
  one step the line search finds no decrease of soundness + λ·count.

**Next measurement (no weight):** the merit change of X97's rejected trials at windows 3 / 5, split by term (λ Δcount
against Δstab and Δ end drift). It decides whether the soundness terms (the released motion of a body pushed in bulk)
block the descent, or the count term's own curvature does.
