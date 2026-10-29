# D175 resume (agent D175: picture keyframes, started 2026-10-29 11:00 CDT)

Read ONE file first: output/gpu/D175_SUMMARY.md (the verdict against the pre-registered bars; written by the reader).
Then tmp/d175/PREREG.md (arms, metrics, bars; written 11:15 CDT before any run).

## What D175 is
- Picture keyframes, a render-only control of the trajectory: "pass through these pictures at t_k".
  Design: docs/design/keyframe_render_control.md (storyboard ownership, option C).
- Code: wip/d175 (local worktree C:/dev/smb_d175, pushed) = wip/d172 e370038a + the storyboard. Server copy: repo_d175
  (VERSION_D175_SHA).
  - physmorph/pipeline/run/storyboard.py: frame_of, entry_of, the Storyboard (the in-place swap of the pack's pictures,
    relief and flags; the λ pre-pass).
  - runner.py: the shared hold() (settle windows + segment holds); no stop inside a segment; an epoch per entry.
  - objective.py: geometry False → the end drift only, no cleanup.
  - solve.py: held λ on both blocks in a segment; no transport repeat in a segment.
  - target.py: sample_pictures / keyframe_pictures, TargetPack.geometry / keyframe.
  - pipeline_run.py: --keyframe / --keyframe_time / --keyframe_geometry.
- λ per block: pre-pass at the source on the FINAL target, then held (--lambda_ema 0). Held on dF_c and u inside a segment;
  after t_k, Hu (held on u, the rule on dF_c).

## Queue (tmp/gpu1_queue, card_runner_n.sh on cards 0 / 1 / 3; never gpu2)
- 208_d175_kf40: three stages.
  - Keyframes: output/gpu/d175/assets/kf_s97 / kf_s99.npz (make_kf.py).
  - V3 gate: G1-G4 bit-SAME, the unit tests, the keyframe smoke K1-K3 + kf_check.py → auto/208_gate/summary.md.
  - The six runs: D_97 / D_99 / S_97 / S_99 / T_97 / N_97 in output/gpu/d175/runs, then the reader → auto/208_kf40 and
    output/gpu/D175_SUMMARY.md.
- Status log: output/gpu/d175/status.txt.

## State 2026-10-29 12:00 CDT: DONE, verdict FAIL (nothing queued, nothing running)
- Gate PASS:
  - G1-G4 bit-SAME, so the defaults, Hu, Hu on the coin, and the settle through the shared hold() are all unchanged;
  - 5 unit tests pass;
  - K1-K3 + kf_check pass: the segment ends at frames 40 / 80 / 120, the segment merit = soundness + held render (residual
    ≤ 5e-10), one held λ, epoch 1 on the final target.
- 208 FAIL on P1 only:
  - the 3D IoU gain over the twin is +0.058 / +0.060 (bar +0.10);
  - S99's G on the run views is 0.578 (bar 0.6);
  - P2 and P3 pass on both seeds;
  - KF-IoU: S 0.926 / 0.918, twin 0.805, D 0.689 / 0.691, N 0.777.
- Cause, measured:
  - the silhouette is met by ~1.5 % of the particles pulled to the poles; the bulk did not stretch (aspect 1.06 / 1.07, against
    the keyframe's 1.48);
  - 86-89 % of the render's position pull lands on the outermost pitch (forensic_pull.md, job 208_dump);
  - S99 also lost 2 of 6 segment windows to commit_replay nulls (committed as holds).
- No showcase: it is gated on a pass. 209_d175_show300.sh, post_show.sh, make_kf_letter.py and d175_video.py are ready but
  not queued.
- Next measurement: see output/gpu/D175_SUMMARY.md, "Reading": (a) the blurred comparison's position pull on S97's dumps;
  (b) the S99 replay split.
- Disk: output/gpu/d175 is 0.7 GB.

## Resume
- A failed run: delete its .rc and re-queue the job (finished runs skip; the gate skips once gates/V3.done exists — delete
  it to re-gate).
- The reader by hand, on a card (the exterior operators are CUDA), from repo_d175 in envr's env:
  `$PY ../tmp/d175/d175_read.py ../output/gpu/d175/auto/208_kf40 ../output/gpu/d175/runs ../output/gpu/d175/assets --summary ../output/gpu/D175_SUMMARY.md`
- Stop: remove our not-yet-claimed queue file. A claimed job ends with its card runner's job. Never pkill / pgrep -f.
- If PASS: the showcase at 300k, bunny → dragon, with a letter or pose silhouette keyframe. Deliverables: frames at t_k vs the
  keyframe pictures, and a twin | full video.
