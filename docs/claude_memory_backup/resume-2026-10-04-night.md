---
name: resume-2026-10-04-night
description: "Full handoff of 2026-10-04 (written 23:00 CDT at the user's request when the model was switched to Opus): code state and defaults, every experiment D70–D88 with numbers, what is running on hyde06, scripts and repo copies, open user decisions, lessons"
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-05T04:00:49.422Z
---

Written 2026-10-04 ~23:00 CDT because the user switched the model ("지금 model 바꿨는데, 그 전 기록 자세하게 정리 해 둬. 혹시 모르니까").
The authoritative record is `docs/experiments.md` (entries D59–D88, newest near the top, around lines 460–1100). This file is the map.
Related: [[render-must-beat-physics-300k]], [[min-spacing-d70-d72]], [[exterior-layer-stages-2-3-directive]], [[always-report-render-influence]], [[algorithm-before-n]], [[no-parameter-only-fixes]], [[generalise-beyond-bunny]], [[gallery-single-run-spread]].

## 1. The user's directives of the day (verbatim, in order)
- "현 코드 push 하고 그리고 나서 A를 진행 해 보자" (A = redistribute near-surface particles → became D70 min spacing).
- "항상 그냥 원인을 감추지만 말아 줘. overengineering도 금지. 코드 20줄이면 짤 껄 200줄로 늘리지 마. 항상 '알고리즘' core를 수정하는 쪽으로 가야 해".
- "언제나 말했듯이, '항상' 랜더 Gradient가 영향을 끼치는지 확인 해야 해".
- "또 지금 랜더링 gradient를 먹인 거 보다 physics가 좋아?" → then: "ㄴㄴ 300K + 에서 무조건 'render gradient'를 먹인 게 'physics only'보다 좋아야 해. 모든 metric과 momentum 보존까지 전부. A, B 둘다 봐야 할 거 같네. 근데 이전 결과에서는 13% 정도 render 넣은 게 더 좋지 않았어?"
- "agent 하나 더 띄워서 이제 N 과 px 간의 관계 조사하고, 현재 surface가 어디까지 detail 하게 갔는지도 조사해 줘" (done: D76).
- "OK. 그럼 실험 및 진행 해서 Render gradient가 유의미한 결과를 낼 때까지 계속 진행 해 줘".
- "기존 결과들도 잘 나와야 하고 (v_dragon_300K_PA_whole.mp4 결과가 꽤 좋았어. 이 결과 base로 detail 살리면 될 듯 해, N을 늘리던 px를 늘리던 뭐 다른 알고리즘을 쓰던)" — base = D70 dragon 300k PA; N is an allowed lever again (see [[algorithm-before-n]] update).
- "OK. 계속 진행 해 줘 지금 10 ~ 20% 이상 나와야 의미 있는 결과인 거니까" — bar: render ahead of the physics-only twin by ≥10–20 % on each measure.
- "momentum 쪽도 잘 기록 해 줘" — momentum in every record/report (D86).
- Earlier the same day: VS Code / local memory cleanup — nothing killed; three other claude.exe sessions (PIDs 8704, 28932, 29328) left alone pending the user's word (never answered).

## 2. Code state (branch `settled-base`, HEAD 1d77570, all pushed to GitHub Chayoso/Shape-morphing-binder)
Committed today and what is DEFAULT:
- `--render_exterior` (D62, earlier): render terms on tracked exterior discs. Opt-in.
- `--min_spacing 0.9` (D70, 6a92201; `kernels.k_update`): opt-in, default 0. D71's surface variant was refuted and removed (4ba2cf7).
- D74 (2b426de, `scripts/pipeline_run.py`): under `--loss_follows_n` the render resolutions follow (N/40000)^(1/3) (300k: 64→126, 96→188 px; an explicit `--render_res_hi` is kept). Default whenever `--loss_follows_n` is given.
- D77 bound on the step (0bb4696) — WITHDRAWN by d39b37b.
- **D81 (d39b37b) — CHANGES THE DEFAULT for every run**: the render weight λ is calibrated at every window's first gradient (LambdaBalancer rule λ|g_r| = 0.5|g_p| with its EMA 0.3), held for the window; `TargetPack.settled_scale` removed; `Selection.check_lambda(rec, lam, render)` rescales the last-accepted and best references linearly (merit is linear in λ) instead of opening a new epoch; `best_window(..., w_pbr)` scores all windows at the last λ. Tests 276 passed.
- `--layer_relief` (D88, a1600f3, opt-in): `k_layer_project` relaxes d − d̄ towards `ref[p]` (target mesh's own rough residual at the particle's nearest surface point, zero beyond one spacing). Path: `prepare(surface=n//4)` → `Prepared.tgt_surface` (via `sampling.mesh.surface_in_frame`, `load_normalized(frame=...)`) → `run_pipeline(surface=...)` → `target.target_relief` → `TargetPack.relief` (`window/layer.TargetRelief`) → `window/setup.py` layer tuple index 8 → `traj.layer_ref`. Tests 277 passed, 2 skipped.
- Probes committed: `scripts/probes/settled/{spacing_flux_probe, particle_depth_probe, momentum_probe}.py` (momentum_probe now writes one JSON row per kept-frame pair).
NOT committed (server copies only; diagnostics): D82 relaxation off (`repo_r72`, setup.py frac 0.0), **D83 fine picture from the first window (`repo_r73`: pipeline_run.py `render_res = render_res_hi`) — the best 300k result so far, candidate for the code**, D84 weight by value (`repo_r74`: balancer fed obj.scalar(e,0) and e.lr).

Recommended 300k recipe now: `--n 300000 --seed 97 --support_form proximity --loss_follows_n --render_exterior --render_res_hi 192 --min_spacing 0.9` on HEAD (+ D83 if adopted). Physics-only twin: same with `--render_weight_scale 0` and without `--render_exterior/--render_res_hi`.

## 3. Results, all frames, last 20 kept frames (render arm vs physics-only twin PA = D70 `--render_weight_scale 0 --min_spacing 0.9`)
PA (the user's "base"): bunny front 0.9900/0.0075, crop 0.9784/0.0086, far 0.9907/0.0068; dragon front 0.9881/0.0102, horn crop 0.9784/0.0139, far 0.9890/0.0108. Floor (D65 second sample): bunny 0.9906/0.0063, 0.9795/0.0071, 0.9904/0.0059; dragon 0.9912/0.0074, 0.9793/0.0112, 0.9918/0.0072.
- Weight held from window 1 (old default, D70 LA): render behind on the dragon (crop 0.9724/0.0162), share 0.94, transport 8× and kinetic 50–100× the twin's (D73 cause: |g_phys| decays 30×, |g_render| 1.4×).
- D74 alone (126 px coarse, LC): coarse-stage misfit gone, starvation stays; dragon crop still behind.
- D75 hand probe (`--render_weight_scale 0.1`, dragon LW): crop 0.9863/0.0106 — proved the share is the cause.
- D77/D78 (step bound, merit with first λ): ahead on display but stops ~50 commits; fails at 40k (−0.004…−0.008 sil IoU) → withdrawn.
- **D81 (LF, HEAD)**: bunny front 0.9935/0.0066, crop 0.9873/0.0058, far 0.9938/0.0056 (1−IoU −35/−41/−33 %, diff −12/−33/−18 %), yardstick sil −52 %, shading −21 %, thin 4.0 vs 4.8 %, transport end 2.6e-5 vs 2.1e-5 (+10 % at equal windows; LF stops at 71 commits vs 87). Dragon front 0.9924/0.0084, crop 0.9852/0.0110, far 0.9926/0.0089 (−36/−31/−33 %, −18/−21/−18 %), yardstick −57 %/−31 %, thin 4.3 vs 5.3 %, transport −16 %, behind on nothing. Weak spot: fine stage only ~3 commits (render 5–7 % of merit late).
- **D83 (LG, repo_r73, 192 px from window 1, D81 weight)**: dragon front 0.9932/0.0080, crop 0.9870/0.0104, far 0.9929/0.0089 (−43/−40/−35 %, −22/−25/−18 %), yardstick 0.000338/0.000439, thin 3.0 %, transport 5.4e-5 (−21 %); rest slightly worse (kinetic last-ten 4.3e-5 vs 3.4e-5, holes 0.042 vs 0.037 %). Bunny front 0.9940/0.0065, crop 0.9867/0.0059, far 0.9938/0.0055 (−40/−38/−33 %, −13/−31/−19 %), yardstick 0.000289/0.000432, thin 3.5 %, transport = twin's (2.15e-5), 86 commits; behind only kinetic last-ten +16 %, rotation 0.013° vs 0.011°.
- D84 (LH, weight by value): not better at 300k (differences 5–26 % worse than LF, transport 3× twin).
- 40k gallery (D72/D80/D85, `output/gpu/d72`, 19 meshes): held weight S beats D81's SF in run silhouette IoU on 18/18 (by 0.002–0.009) but with transport 2.4–12× and kinetic 3.6–40× the twin's; SF beats its twin SP on 18/18 (+0.002…+0.010 sil IoU) with transport 0.1–0.7× S's. SH (value weight) loses 0.001–0.005; SG (D83 at 40k) = SF. **Open user decision: which weight is the 40k/default** (my recommendation: D81 for all N).
- Momentum (D86, `output/gpu/d86`, plots in `output/video_2026-10-04/d81_weight_every_window/momentum_*.png`): every run's centre of mass moves < 0.03 pitches, rotation ≤ 0.11°; D81's kinetic energy follows the twin's curve; old held weight 10–100× from window 30.
- Detail (D76 agent + D82/D87): carried share of the mesh's relief by octave (2.7/5.4/10.8/21.6 pitches): 300k sample 0.36/0.66/0.86/0.95; PA 0.03/0.18/0.73/0.92; LF 0.05/0.20/0.72/0.93; relaxation off (PR) 0.24/0.53/0.97/0.95 but roughness 9.2° and display much worse; one window relaxing towards the mesh's relief (D87 offline) PA 0.19/0.39/0.73/0.91. Render does not write sub-cell relief; the relaxation (towards zero) erases it. N: each finer octave costs 8× particles; 2.4M ≈ next octave at today's run-to-sample gap.
- D88 (`--layer_relief`) runs finished 22:13–22:53 (PT twin, LR render, bunny+dragon, `output/gpu/d88`, repo_r75); first numbers: bunny LR sil 0.9880 (LF 0.9863), thin 3.4 %; dragon PT yardstick sil 0.00149 (PA 0.00105, WORSE), transport 1.6× PA; dragon LR transport 9.7e-5 (LF 5.7e-5). Display/bands not yet read at 23:00. Undecided whether the relief reference helps; D88 criteria in experiments.md.

## 4. Running on hyde06 at 23:00 CDT (`ssh hyde06j`, B=/data/relcfd/chayo/physmorph_v2, G=$B/output/gpu)
- D88 evaluations: `runeval2.sh` chains for d88 dragon PT/LR and bunny PT (then bunny LR, then `bands_new.sh` per run → `$G/d76/bands_*_new_HHMM.log`, read with `tmp/band_table.py`). Status in `$G/d88/status.txt`, `$G/d76/status.txt`.
- Gallery: four workers `tmp/d85.sh 0..3` on `tmp/d85.queue` (SF arm on V, then D72's paused baseline runs B2/SP/BP etc.); outputs `$G/d72`, done count `grep -c "probes done" $G/d72/status.txt` (127 at 23:00). Tables: `tmp/d72_table.py`, `tmp/d80_table.py ARM`.
- **Storage: 47 GB of the 50 GB limit** ([[cleanup-over-100gb]]): next step must be a cleanup (archive logs first; candidates: `*_frames12.npz` of withdrawn/diagnostic runs d67, d71, d74, d77, d78, d79, d84, d82 once their probes are read; d72 deletes its own frames).

## 5. Tools (server `tmp/`, all scp'd from the local scratchpad; never run python locally)
Runs: `run12.sh GPU DIR TAG MESH N REPO [args]` (keeps every 12th frame → `TAG_frames12.npz`), `runeval2.sh GPU DIR MESH TAG BOX` (yardstick t_, arrangement a_, momentum m_, offsets o_, display e_), `stop_runs.sh DIR`. Crop boxes: bunny `1900,100,2800,700`, dragon `1900,60,2500,560`.
Readers: `gs_probe.py JSON…` (run summary), `shown20.py` (display last 20), `last20.py` (yardstick), `off_rows.py`, `yw.py KEY LABEL=LOG…` (by window; KEY sil|pbr|front.iou|crop.difference…), `why_rows.py` (gradients/energies by window), `fine_rows.py`, `rescore.py` (merit rescored with other λ), `nulls.py` (commit-check discards), `momentum_rows.py`, `momentum_plot.py`, `band_table.py`, `d87_rows.py`, `render_twin_plot.py` (repo probe; 16 panels).
Repo copies: r65 stage-2 probes (terms.sh/off.sh/d63v.sh use it), r66 D70, r68 D74, r71 = D81+D74 (HEAD before layer_relief), r72 D82 diag, r73 D83 diag, r74 D84 diag, r75 = HEAD with layer_relief.
Local outputs: `output/video_2026-10-04/{d70_min_spacing (v_dragon300k_PA_whole/crop.mp4 = the user's base), d71_layer_kept, d81_weight_every_window (twin_*.png, momentum_*.png, v_dragon300k_LF_*.mp4)}`.

## 6. Next steps that were planned
1. Read D88 fully (display, bands, momentum) against PA/LF; decide whether the relief reference stays.
2. Put D83 (one fine resolution from the first window on the exterior path) into the code as a definition change, register, test, gallery-check.
3. Remaining gap to "every metric": kinetic energy at the end (+16–27 %) and the bunny's net rotation.
4. Server cleanup below 50 GB.
5. Ask/answer the 40k default-weight decision; the exterior display in the final renderer (`render_splat_photoreal.py`) and dropping the particle-cloud render path are still open from 10-04 morning.

## 7. Lessons of the day (do not repeat)
- A measured mechanism is not a measured outcome: D71 inferred swelling from the display offset; the particles were further in.
- Check records before editing: the commit-check discards were the base rate, not the new weight (edit reverted before commit).
- One 300k run or evaluation per GPU; a second job OOM'd when an evaluation grew to 34 GB.
- Point-wise normal flipping by a sample test mis-turned 12–15 % of mesh normals and produced tens-of-pitch references (D88 first launch withdrawn after 3 min).
- g_share near 1 is not "strong render influence"; it was the physics objective being starved.
- ssh: never `pkill -f` a pattern present in the ssh command line; write Python files locally and scp (heredoc quoting breaks); the env script `cd`s to the default checkout, so `cd` into the intended repo after sourcing it.

## 8. Continued with Opus, 2026-10-05 00:00–00:45 CDT (HEAD be618cd)
- User 23:00: "10.04 23:00 기준까지의 작업 및 코드 한 번 push 하고, 그 다음은 Opus로 진행하는데, 계속 goal 까지 작업 지속 해 줘" — everything up to 23:00 was already pushed (1d77570); continuing toward the goal.
- Server cleanup: kept-frame npz of d71, d74, d75, d77, d78, d84 deleted (records/logs kept) → 38 GB.
- **D89 committed (9c301bb)**: one render resolution (the fine one, following N) from the first window in `scripts/pipeline_run.py` (c2f inactive from the CLI). Matched physics-only twins PN (d89) and a repeat render arm LN: bunny LN replicates LG (86 commits, sil 0.9869, thin 3.2 %); twins' own spread is large (bunny kin last-ten PA 4.5e-6 vs PN 7.9e-6; dragon PA 3.4e-5 vs PN 5.0e-5) and the render arms' end rest lies inside it — the earlier "kinetic gap" was the old twin's c2f extension.
- **D90 (922b4de) — important correction**: every display number since D62 read runs against their own target sample, which is also the render term's target. Against an INDEPENDENT 300k sample (seed 99, exact frame; `output/gpu/d90/{bunny,dragon}_ind300k.npz`, built by `scripts/probes/settled/independent_sample.py`) the render arm's lead is about half: dragon LG vs PN −11…−23 % (all display measures ≥10 %), bunny LG vs PA −4…−26 % (front/far difference only −6/−8 %). A 2.4M reference at run pitch is biased (larger shell) — not usable. D88 relief reference is behind on every display measure against the independent sample (but carries 3× the mesh relief at 5.4 pitches: bunny 0.53, dragon 0.49) → not adopted as is.
- **D91 committed (be618cd), running**: `--render_target_draws K` (default 1): render target pictures = mean over K independent samples (seeds 198, 298, …; 99 reserved for evaluation). Runs `output/gpu/d91/{bunny,dragon}300k_LK` (K=8) started 00:38 on GPUs 2, 3. Schedule `tmp/d91.sh` then reads LK, LN, PN, PA, LF, LG with `tmp/runeval3.sh` (yardstick `t3_`, display `e3_` against the independent sample, all frames) and bands. Readers: `tmp/d90_rows.py` (end frames), shown20/last20 on e3_/t3_ logs.
- Probes now take `ref=`: `render_terms_probe.py FRAMES [RES] ref=NPZ`, `surface_layer_probe.py ... ref=NPZ [refpitch=run]`.
- Error made: a stray local `python3 -` in a command (hung on stdin, stopped) — never put python in local commands.
