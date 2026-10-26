# D163 RESUME (2026-10-22 16:55 CDT)

D163 is a showcase. The words "SIGGRAPH" (+z) and "VANDERBILT" (-z) exist only in the render's target pictures; the 3D target
is a smooth plaque (d163plaque.obj); the source is the sphere. The pre-registration and the reads are in docs/experiments.md:
7e43c0f (pre-registration), 472eca1 (the 40k read), 12bf2b5 (amendment 2, arm S), 742e642 (the 300k read), and the note
on the dropped R rerun.
Code: wip/d163 (no pipeline change), scripts/probes/paper/d163/ = tmp/d163/. Deployments: repo_d163 / repo_d163s (byte
copies of repo_d162 / repo_d162s at 539b52a); repo_d163bs is made by startS40c.sh from repo_d162bs after D162b's GO.

## Done
- 40k: T97, T99, R97, H97, H2_97, F97, FC97, TC97. Read: after/summary_p40.md. Only H carves (0.91 of the ceiling).
- 300k: T97 and H97 (runs300/). Read: after/summary_p300early.md. H carves 0.58 of the ceiling at corr 0.50; P3 and P4 fail.
- Video H over T: video/p300_H_T/ (4k, 1080p, the sheet, closeup, closeup2 with the ceiling sample, stills).
  Local copies: output/results_2026-10-22/d163_text_relief/.

## Dropped
- The R97 300k rerun (the coordinator, 16:55). At 40k the rule does not carve (0.26 of the ceiling; the twins read
  0.20-0.26), and the 300k deliverable is H against the twin.
- The first R97 300k run had died of an OOM at 26.7 min, caused by another user's job.
- The rerun chain (pid 1785284) and its reader (pid 1896888) were stopped. Neither held a card.

## Armed (detached; check with ps -p PID)
- startS40c.sh (its pid is in tmp/d163/startS40c.pid) waits only for output/gpu/d162b/gates/GO.
  - D162b's gates stopped twice (16:17, 16:51). The earlier waiter, startS40b.sh, exited on the first STOP.
  - After GO it copies repo_d162bs to repo_d163bs and runs plq40k_S_97 by the strict claim (cards 0 and 3).
  - Then it reads the run (d163_after.sh p40S, scene40_S.json).

## Next
1. When p40S is read: if S carves (share >= 0.5 with P2), run plq300k_S_97 (the same flags at 300k, repo_d163bs). Then
   render the video S over T with d163_video.sh (TAG p300_S_T; CAP_S=8). Report P4 (the rest) for S.
2. Cleanup after the reads. runs300/plq300k_T_97_frames12.npz (1.3 GB) is needed for any further video; delete it only
   after the S video. Log every deletion in output/gpu/cleanup.log.
