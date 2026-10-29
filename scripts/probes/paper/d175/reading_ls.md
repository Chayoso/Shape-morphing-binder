
## Line-search split and the segment replay measure (2026-10-29 13:13 CDT; auto/200_ls/ls.md, gate d PASS)

**(1) What blocks the step: the soundness terms brake.**
- In each run, window 1's first step is the push: λΔcount −6.0e-3, and the body starts moving (end speed 0 → 0.36
  pitches per horizon).
- Every later segment step brakes. The released motion (about 2/3) and the end drift fall while λ·count rises: accepted
  steps median Δstab −3.6e-5 against λΔcount +3.7e-5.
- All 156 failed trials are "the count term rises". They do not leave the body moving (end and released speed ×1.00).
- So the blocker is the soundness terms doing their job, rest at every window end, inside a segment whose keyframe lies
  6 windows ahead. The trade is decided by the scale of λ·count against the soundness terms, which share no unit.

**(2) Window-1 replay null: a measurement defect.**
- The start pair at rest reads 0 (render Δ exactly 0), so the tolerance fell to the 1e-7 floor (≈ 1.3 ulps of E), while the
  candidate's render replays differ by 1.1–3.7 ulps.
- Fix (solve.py, commit(); keyframe segments only): D164's measure, the max of the start pair, the window's taped pairs and
  one ulp of E.
- Gate d: G1-G4 bit-SAME. All 18 segment commits kept (noise 0.65–1.9e-7).

**Pre-registered next:** tmp/d175/PREREG_c.md (13:14 CDT), `--keyframe_soundness final`. A segment's merit is its keyframe
comparison alone; the soundness terms return with the final target. Arm Z = 97 / 99 (job 200_d175_snd).
