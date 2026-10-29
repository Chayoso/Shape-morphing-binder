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

## Resume
- A failed run: delete its .rc and re-queue the job (finished runs skip; the gate skips once gates/V3.done exists — delete
  it to re-gate).
- The reader by hand, on a card (the exterior operators are CUDA), from repo_d175 in envr's env:
  `$PY ../tmp/d175/d175_read.py ../output/gpu/d175/auto/208_kf40 ../output/gpu/d175/runs ../output/gpu/d175/assets --summary ../output/gpu/D175_SUMMARY.md`
- Stop: remove our not-yet-claimed queue file. A claimed job ends with its card runner's job. Never pkill / pgrep -f.
- If PASS: the showcase at 300k, bunny → dragon, with a letter or pose silhouette keyframe. Deliverables: frames at t_k vs the
  keyframe pictures, and a twin | full video.
