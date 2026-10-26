# D162 RESUME (updated 2026-10-22 11:30 CDT)

D162 tests one definition for the render's weight, run as two arms:
- (1) `--render_weight_ref floor`: λ_t = λ_0·max(0, 1 − L_floor/L_r). L_floor is the pictures' own first sample, read the way the body is read.
- (2) `--placement_at_cell`: proximity covered within one loss cell, the near band empty, the spray gate measured in loss cells.

Arms: F = (1); FC = (1)+(2); TC = the λ = 0 twin of FC. Pre-registered in docs: b21049b, amendment 1 03b5720, amendment 2 1fb5ed0, amendment 3 aed964c.

Code: `wip/d162` 539b52a (rt1 + d159 + d160 + d161 + D162, all options default off; the floor records reach the history). Deployed as repo_d162 and repo_d162s (`tmp/d160/patch_s.pl`). The tag as tagged, for the bisection: repo_d162tag (6c9b0e9).

## Gate try 2's V1 failure (diagnosed, amendment 3)
- The test: `test_null_windows_add_no_simulated_time_and_are_not_delivered`, null_reason commit_replay on the first window.
- The cause: the frozen commit check's tolerance, max(1e-7, 10 × replay_rel) from one replay pair at the start control (at rest at the first window, so 0 or 1 ulp), against a few ulps of atomics noise.
- Tallies: the tag 0/60, wip/d162 5/80 (+1 in the gate), d161 2/20 (code-identical to the tag on this path).
- In production: the tag's runs null 4.1 % of windows by commit_replay (d145t_b1: 222/5370).
- Not a merge interaction. Frozen code, so it is the user's call. Logs: output/gpu/d162/gates_try2/diag*_*.log.

## Running (detached; PIDs recorded by the processes themselves)
- `alpha.pid` (run_alpha.sh = d162_alpha.sh): first `"V R X"` with A1 (gates, 24 scene runs, then the arrangement read X), then `R` with A2 (26 gallery runs). It runs the private chain copy run_alpha_chain.sh.
- `beta.pid` (run_beta.sh): waits for this launch's "V gates passed" / "V STOP" status line, then `R` with list B (50 gallery runs). Chain copy: run_beta_chain.sh.
- `after.pid` (run_after.sh): reads the scenes after "R done (d162_runs_A1.txt)", and the gallery after A2 + B. Outputs go to output/gpu/d162/after/.
- X writes after/arrange.{jsonl,md,log}. Reading rule: amendment 3 (e_a ≤ 0.05 with e_b > 0.05 + 2 se means arrangement ate the floor).
- NEVER overwrite a script that a running chain is executing (bash reads by offset). Edit d162_*.sh, then launch from a fresh copy.

## Next
1. Read the outputs. Verdict per arm: P1/C1 + P3–P5/C2–C4 + G, and the X arrangement reading.
2. Compare with the tag, D161c / D160c (λ held), D160d (`tmp/d160/for_d162.md`, `output/gpu/d160/gate_D/`) and the twins.
3. Docs read entry. Local `output/results_2026-10-22/d162_render_weight/README_ko.md`, plus a line in output/README.md.
4. Delete bulky dumps after reading (disk 457/500 GB). Report in Korean, with the render→control→outcome summary.
5. If a 40k arm passes, tell the coordinator; 300k only with their go.

## State at 15:05 CDT (read 1 done, docs c7976be)
- The A1 scenes and X are read: after/summary_*.md, colour.md, c23_colour.md, arrange.md, arrange_stats.txt. Local copy: output/results_2026-10-22/d162_render_weight/.
- Verdict before the gallery: F and FC do not pass.
  - Relief P1 fails everywhere.
  - Colour C1 passes on band (F C3 pass, C2 2 cells), fails on ears.
  - Arrangement ate the floor on the coin.
  - Relief runs switch the weight on and off; colour runs keep it held.
- Next definition named (not run): (1b) the excess read on the shape, e = 1 - b_floor/b.
- Remaining: the gallery. alpha runs A2 (waiting for a card at 14:56), beta runs B on gpu1. after.pid reads it once "R done" appears for both A2 and B (after/gallery.md, windows_gallery.md). Then G per arm, docs read 2, the local README section 9, and thinning the gallery archives after their read.

## D162b (started 15:11 CDT; pre-registered docs be85772)
- Code: wip/d162 56ed3c8, `--render_weight_ref shape` (physmorph/pipeline/shape_read.py). Repos: repo_d162b / repo_d162bs. Output: output/gpu/d162b/.
- `d162b.pid` (run_d162b_chain.sh "V R X"):
  - It takes gpu1 over from beta (HANDOVER_FROM = the beta wrapper's pid) once beta's current run ends. List B was cut by emptying tmp/d162/d162_runs_B.txt; the full list is d162_runs_B.full.txt.
  - The gates run first. R then WAITS for output/gpu/d162b/gates/GO: read t_shape / t_attempt in gates/S_*.json, write the cost amendment, then `touch gates/GO`.
  - X follows on coin S 97 and band S 97.
- `d162b_after.pid` (run_d162b_after.sh) reads after "R done (d162b_runs.txt)": after/summary_*_S.md, colour.md, c23_colour.md, windows_d162b.md (switches, the rest after the excess, the cost).
- After D162b, restore the gallery: `cp d162_runs_B.full.txt d162_runs_B.txt`, then run list B and list A2 (d162_runs_A2.txt) with a fresh chain copy (finished runs are skipped). The old after.pid still waits for both "R done" lines to read the gallery.
- alpha was stopped before A2 (15:06).

## State at 18:45 CDT
- D162b is done and read (docs 3751759; local output/results_2026-10-22/d162_render_weight/README_ko.md, the D162b section). Nothing runs for D162b.
- Gallery: `gal.pid` (run_gal.sh)
  - It runs the rest of list B and list A2 as d162_runs_BA2.txt on a normal claim, then d162_gallery.py into output/gpu/d162/after/gallery.md.
  - The old after.pid was stopped (it waited on list names).
  - Next: the G verdict per arm, then thin the gallery archives after the read.
- Next experiment (not started; it needs the coordinator's go): split the render's per-window push on the exterior into normal / tangential and stroke-scale / sub-pitch (work telemetry records the per-particle render gradient).

## State at 19:50 CDT (D162c read; docs 5c31132)
- D162c (measurement) is done.
  - (A) is in output/gpu/d162c/A*.jsonl; (B) pushes are in output/gpu/d162c/push/*.npz (≈250 MB, delete once no longer needed) and B.jsonl; the tables are in d162c.md.
  - Reading: the push decomposes alike in failing and succeeding runs. The realized motion separates them by size. Next definition named (not run): the floor-referenced weight read on pictures low-passed at one pitch.
- The gallery remainder was stopped after gal_homer40k_F_97; the full list is d162_runs_BA2.full.txt.
- gpu1: tmp/gpu1_queue/run_gpu1_runner.sh (runner.pid) takes the lock over from the gallery chain (pid 2327245) when homer ends, then runs NN_agent_name.sh jobs. Contract in tmp/gpu1_queue/README.md; log in runner.log.

## D162d (20:00 CDT; pre-registered docs c0ddc74; code wip/d162 148706f, deployed 21423f48 as repo_d162d / repo_d162ds)
- `--render_weight_ref floor_pitch`: F's weight with L_r and L_floor read on pictures low-passed at one exterior pitch (target.render_term_lowpass). Arm name Lp.
- It runs as gpu1 queue job tmp/gpu1_queue/02_d162_d162d.sh: gates V2 / V3 / V4, then the 8 runs (d162d_runs.txt). Status in output/gpu/d162d/status.txt.
- d162d_after.pid (run_d162d_after.sh) reads after "R done (d162d_runs.txt)" on the CPU:
  - relief summaries, colour.md, c23_colour.md;
  - windows_d162d.md (switches, rest);
  - trace.md / trace.png (the per-window weight trace with what is still to do; R1).
- Then: the docs read entry, the local README section, the Korean report. If it passes at 40k, tell the coordinator: next is the 300k plaque (SIGGRAPH / VANDERBILT) with D162d.

## State at 22:20 CDT (docs 3219c9c: D162d read; D162e / D162f pre-registered)
- D162d FAILED: relief P1 13-30 %, R1 fails; band C1 passes but rest fails. Local README section + reads_d162d/trace.png.
- gpu1 queue, mine:
  - 22_d162_d162e_uoff.sh: the held arm with u off, coin / bunny 40k x 97/99, into output/gpu/d162e/runs.
  - 23_d162_d162f_ripple300.sh: the 300k ripple bunny, T then H, into output/gpu/d162f/runs.
  - 24_d162_d162f_band300.sh: the 300k band, T then D160d, repo_d160s.
  - 25_d162_d162f_band_video.sh: d160_studio.py side by side, with a 300k independent sample.
  - Before them: figs 11 (running) and D164's 21.
- d162f_after.pid (run_d162f_after.sh, CPU) reads each part once its "R done" line appears:
  - output/gpu/d162e/after/summary_*_Hu.md (the u answer: Hu vs D161c 0.47-0.58 vs twins);
  - output/gpu/d162f/after/read_ripple.json and vid_ripple_{whole,close}/pair.mp4;
  - output/gpu/d162f/after/read_colour.jsonl. The band video comes from job 25.
- (a) the plaque: reuse D163's H vs T (output/gpu/d163/video/p300_H_T).
- To report (Korean): the u answer; then the 300k ripples and band with the rest cost (H / D end v_mean against T).
- D162g (docs 9f71648): job 26 = H+C (--lambda_ema 0 --placement_at_cell), coin / bunny 40k x 97/99, into output/gpu/d162g/runs. d162g_after.pid reads it against TC (d162/runs40) and D161's twins: output/gpu/d162g/after/summary_*_HC_{TC,D161}.md. If H+C holds rest (P4 at both seeds) with P1, rerun the 300k ripples and the plaque with H+C.

## STATE 2026-10-22 23:10 CDT: everything queued on the gpu1 queue (tmp/gpu1_queue, runner pid in runner.pid)
Job -> output -> auto summary (written by d162_auto.pid = tmp/d162/run_d162_auto.sh):
- 22 u off under the held weight (D162e), coin/bunny 40k x 97/99 -> output/gpu/d162e/ -> output/gpu/d162/auto/22_uoff/summary.md (Hu share vs D161c 0.47-0.58: what dF_c writes alone).
- 23 ripples 300k, H and T (D162f b) -> output/gpu/d162f/ (videos after/vid_ripple_{whole,close}/pair.mp4) -> auto/23_ripple300/summary.md.
- 24 + 25 band 300k, D160d and T, plus the studio video (D162f c) -> output/gpu/d162f/ (after/vid_band300k/side_by_side.mp4) -> auto/24_band300/summary.md.
- 26 H+C (D162g), coin/bunny 40k -> output/gpu/d162g/ -> auto/26_heldcell/summary.md. Rule: if P4 passes at both seeds with relief near D161c, rerun the 300k ripples and the plaque with H+C (new jobs 29+).
- 27 the settle test (D162h): held + --settle_windows 30, coin/bunny/band 40k -> output/gpu/d162h/ -> auto/27_settle/summary.md with "VERDICT: A|B|mixed".
  - A (rests, keeps >= 0.8): the auto-reader queues 28 (D162i: held + settle on ears and a 3-mesh gallery spot check) -> auto/28_d162i/summary.md.
  - B (springs back): D162g plus a plastic-baking definition, to be designed.
- (a) the plaque: D163's H vs T video, output/gpu/d163/video/p300_H_T (reused).
Code: wip/d162 d9f07b1 (repo_d162h / repo_d162hs carry --settle_windows); docs 57ece0e (D162h / D162i), 9f71648 (D162g), 3219c9c (D162d read, D162e / D162f).
- job 29 (stiffer E x3 / x7, held, settled) -> output/gpu/d162s/ -> output/gpu/d162/auto/29/summary.md (d162_auto29.pid)
- job 29b (29_d162z_minimal.sh: held weight, cleanup off, placement at the cell; settled if D162h read A) -> output/gpu/d162z/ -> output/gpu/d162/auto/29b/summary.md (d162_auto29b.pid)
- job 82 (500k deliverables, or 300k when < 36 GB free; definition from the auto summaries) -> output/gpu/d162k/ -> output/gpu/d162/auto/82/summary.md (run outcomes; readers to add)

## 2026-10-23 10:45 reruns: jobs 50 (settle, verdict -> auto/27_settle; D162i = job 56 on A), 51 (stiffness -> auto/29), 52-54 (500k plaque/ripples H+C, band D160d -> auto/82, plaque video d163/video/plq500k_HC_T), 55 (300k ripples H+C / band -> auto/23_ripple300, 24_band300); runners card0/card3 (card_runner.sh) + gpu1; auto driver d162_auto2.pid
