
## Reading (agent D175, 2026-10-29 11:58 CDT)

**Verdict: FAIL against the pre-registered bars.**
- P1 fails:
  - the 3D IoU gain over the twin is +0.058 (S97) and +0.060 (S99), against the bar of +0.10;
  - S99's gap closure on the run views is 0.578, against the bar of 0.6.
- Everything else passes on both seeds:
  - G on the held-out views, 0.603 / 0.546;
  - KF-IoU against D, +0.24 / +0.23;
  - all of P2 (final shape) and P3 (rest).
- No showcase was run: it was gated on a pass.

**The predictions.**
- Twin KF-IoU 0.805 (predicted ≈ 0.79).
- S 0.926 / 0.918 (predicted ≥ 0.92).
- N: G_N = −0.15 and g_raw_cos < 0 in 5 of 6 segment windows (predicted < 0.2 and ≥ 4 of 6).
- S ends like D (final sil IoU 0.988 / 0.989 against D's 0.988 / 0.988), 160 / 760 frames later.
- So the conflict analysis behind ownership holds: the naive swap moves away from the keyframe.

**Render influence.**
- λ = 0.3113 (s97) and 0.2219 (s99), the pre-pass value, identical to D's own first calibration.
- g_share is 0.97 / 0.93 inside the segment and 0.93 / 0.92 after t_k (D: 0.92 / 0.91).
- The twin reads the same storyboard with λ = 0 and stays the sphere (KF-IoU 0.805 → 0.805; all 6 segment windows are holds).
- All of S's gain on the keyframe is the render's.
- What the render does not change: the final shape and the rest (P2 / P3 within D's two seeds).

**What misbehaves (measured; forensic files below).**
1. The silhouette is met at the outline, not by the pose.
   - At frame 240, S's second-moment aspect sqrt(var_y / var_xz) is 1.059 / 1.074. The keyframe's is 1.480 and the sphere's 1.00.
   - The median |y| is 0.79, the sphere's 0.80 (keyframe: 1.03).
   - But the |y| q99.9 reaches 2.74 (keyframe: 2.92). About 600 particles (1.5 %) are pulled out to the poles, while the
     equator excess stays (r_xz q99 2.24 against the keyframe's 1.96; w_spray = 1 < w_hole = 2).
   - forensic_pull.md: 86–89 % of the render's position pull |g_rend|² lands on the outermost pitch (11 % of the
     particles). Under 1 % reaches the interior beyond 3 pitches (68 % of the particles). The interior's pull is 15–60× weaker
     rms and uncorrelated with the bulk stretch (cos −0.07 … +0.01).
   - This is the design's named main risk, Treuille's flatness: inside a segment no transport carries the bulk, and the
     silhouette pulls only the outline.
2. S99 lost 2 of its 6 segment windows (windows 4–5) to `commit_replay` nulls. Each was committed as a hold, so the
   clock stayed right but the pose made no progress there.

**The next measurement** (no weight, no schedule; per the standing rule):
- (a) The design's pre-declared next definition, Treuille's blurred comparison: the keyframe pictures and the body's
  compared after a Gaussian blur whose width shrinks as the comparison converges. First measure, on S97's dumps,
  whether a blurred picture's position pull reaches below the outermost pitch and aligns with the stretch.
  - The exterior discs lie within the field's kernel of the surface, so the position pull stays in the outer few pitches.
    The blur changes how coherent that pull is, not how deep it reaches.
  - That measurement decides whether (a) can move the bulk through dF_c's stress, before any pre-registration of D175b.
- (b) The two S99 `commit_replay` nulls: replay the accepted candidate and split E_final − E_accept into the soundness
  terms and the render term, against the segment's tolerance (10 × replay_rel × |E|, where |E| carries no transport inside
  a segment).

Files:
- output/gpu/d175/auto/208_kf40/{summary.md, verdict.json, rows.json, silhouettes.png, forensic_pull.md};
- the gate: output/gpu/d175/auto/208_gate/summary.md;
- the runs: output/gpu/d175/runs/{D,S,T,N}_*.

## Next measurements (a) (b) (c), 2026-10-29 12:20 CDT (criteria stated first in tmp/d175/MEAS_a.md; files auto/208_next/)

**(a) VERDICT: no blurred comparison reaches the bulk, so no arm** (pull.md).
- Position pull on S97's segment states: for every comparison (current, blur 2 / 4 / 8 px, pyramid, sdf), ≤ 1.5 % lies
  deeper than 3 pitches. The exterior discs' kernel bounds it, whatever the picture comparison.
- Through the dynamics (Adam's first step at the source, one rollout):

  | comparison | share of |Δx|² below 3 pitches | cos with the stretch below 3 | aspect gain per step |
  |---|---|---|---|
  | current | 0.22 | 0.49 | +0.033 |
  | blur 2 | 0.18 | 0.47 | +0.034 |
  | blur 8 | 0.14 | 0.34 | +0.019 |
  | pyramid | 0.23 | 0.39 | +0.028 |
  | sdf | 0.12 | 0.24 | +0.006 |
  | 3D target (reference) | 0.20 | 0.62 | +0.055 |

- Blurring does not deepen or align the push.
- The current comparison already moves the bulk at the first step, at 60 % of the 3D target's gain.
- REACH (≥ max(0.25, 2 × current)) fails for every comparison, the 3D reference included.

**(b) The replay nulls are float32-level noise in the render replay against a tolerance at the noise floor** (next.md).
- 2 of 18 segment commits were nulled, both at window 1 (the source).
- E_final − E_accept was 1.68e-9 / 2.38e-9 against tolerances of 1.55e-9 / 2.15e-9.
- 98 % of each difference is λ × Δrender; the soundness terms contribute ~0, and no disc set was found anew.
- In a segment |E| carries no transport, so the tolerance is the 1e-7 relative floor; the render's replay sits right
  there.
- The final-target windows: 2 nulls in 127 commits.

**(c) 1 s is physically enough** (next.md). dF_c alone (λ 0, u off) driven toward the stretched sphere as a 3D target
reaches aspect 1.41 at frame 144 (0.6 s) and 1.46 at t = 1 s (keyframe 1.48). The render run reached 1.06.

**Reading.** Neither time, stiffness nor the comparison's first step is the limit.
- Over the segment the render's bulk push fades: by window 1's end 18 % of the position pull sits on the hull points
  (the pole spikes), and the bulk aspect stalls at 1.06.
- **Next measurement:** the same dynamics reading (Adam-step Δx depth and alignment) at the start of each segment window
  2-6. This needs a rerun that saves the full window-start states (x, F, v, C, Fp, J). It would show when and why the
  bulk alignment collapses, e.g. a hole term already satisfied by the thin pole spikes (the soft alpha saturating).
