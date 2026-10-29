
## D175c read (2026-10-29 13:30 CDT; auto/200_snd/summary.md, gate e PASS: G1-G4 bit-SAME, 6 unit tests)

**VERDICT: FAIL on both seeds (P1).**
- KF-IoU 0.784 / 0.790; G −0.11 / −0.08; 3D 0.737 / 0.744, below the twin.
- P2 and P3 pass: final shape, rest, no pieces, Jmin > 0.

**What happened: the pose is reached, then overshot.**
- Without the soundness terms the push reaches the keyframe's aspect early: 1.475 / 1.479 at frame 120 (t = 0.5 s).
- It then overshoots, to 2.19 / 2.16 at frame 240 and 2.30 / 2.26 at frame 288.
- Windows 4-6 did brake: 8 accepted steps each, end speed 0.80 → 0.56 pitches per horizon in window 4, λΔcount < 0 on
  every accepted trial (auto/200_snd/ls_z.md).
- But the selection's 5 % brake rejected them (outer gain −3.7 / −14 / −25 against window 3's best), so the storyboard
  committed the zero-control hold instead. The body coasted.

**Measured next blocker: inside a segment the selection judges a window against the epoch's best.**
- The clock leaves a different alternative: the hold, the zero-control rollout from the same start.
- So the brake replaces a braking window with a coast.

**Next definition (not run, to pre-register):** inside a segment, an outer reject compares the candidate's merit with its
own hold's (the zero-control rollout from the window's start), the alternative the clock actually takes.
