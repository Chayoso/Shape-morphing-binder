# D174: runtime (~10 min per 300k Hu run) and memory (a ~2M Hu run on one 48 GB card)

Agent D174, 2026-10-29. Code: wip/d174 (local worktree C:/dev/smb_d174, pushed), server repo_d174 (= wip/d174 3b5dfd77: exact batches 1-2,
A/B switches off, memory m1 + m2; checksum-verified). Nothing is adopted; the A/B and the 2M plan are the user's call.
Hu = `--lambda_ema 0 --settle_windows 30 --no-settle_relax --render_split u` (D172).

## 1. Where a 300k Hu run's time goes (solo profile runs, seed 97, `--profile`, repo_d172)

| | bunny 300k | dragon 300k |
|---|---|---|
| wall | 489 s (8.2 min) | 1361 s (22.7 min) |
| attempts (kept / null / outer-rejected) | 61 (59 / 1 / 1) | 138 (130 / 5 / 3) |
| seconds an attempt | 6.48 | 8.97 |
| start-up: prepare / thin set / target build | 30.9 / 20.1 / 10.3 s (first run of the target: samples computed) | 57.2 / 20.5 / 8.2 s |
| settle (30 windows) | 27 s (0.90 s each) | 32 s (1.08 s each) |
| end (metrics, save) | 2 s | 3 s |
| GPU busy in the window loop (1 Hz) | 95 % | 97 % |
| evaluations a window (warm 2, replay 1, line search 9.1-9.2, commit 1) | 13.1 | 13.2 |
| line-search trials an iteration | 1.15 | 1.15 |

Per window (s, exclusive of nested sections; dragon / bunny), top 5 marked:

| section | dragon | bunny | note |
|---|---|---|---|
| **1. Sinkhorn transport solves** (2 a evaluation) | **2.53** (28.8 %) | 0.70 (11.1 %) | 192 / 53 ms an evaluation; 87^3 / 71^3 loss grid; the axis log-sum-exp is ~95 % of a solve |
| **2. MPM adjoint sweeps** (3 an iteration x 8) | **2.22** (25.3 %) | 2.13 (33.8 %) | render's 0.94 (incl. the render backward), physics' 0.65, cleanup's 0.63 |
| **3. MPM forward rollouts** | **0.98** (11.1 %) | 0.93 | candidates and checks 0.61, tape 0.37 (45-53 ms a rollout) |
| **4. surface proximity** (body KD-tree + 32-NN at the outer target points, every evaluation) | **0.74** (8.5 %) | 0.43 | 57 / 32 ms an evaluation |
| **5. render forward** (18 views, exterior discs) + exterior searches | **0.71** (8.1 %) | 0.73 | 1.6 / 2.2 searches a window, 80 ms each |
| u gate's transport map (window start) | 0.32 | 0.19 | |
| commit records (momentum record, near-band far part) | 0.22 | 0.17 | |
| window set-up kNNs (spacing, layer, near band) | 0.26 | 0.21 | |
| eval checks (min det, domain), assimilation, line-search bookkeeping | 0.39 | 0.39 | |

Host syncs and idle gaps: the GPU is busy 95-97 % of the window loop even with the profile's syncs (no profile: 95 %
bunny); the CPU side is not where the time is. The start-up is CPU-bound (prepare: GPU 7-11 % busy).

## 2. Exact speed-ups (bit-SAME; V3 gates in deterministic mode, bunny 40k, 4 windows: G1 defaults vs the tag d153/rt1/S1,
G2 Hu settle 30 vs repo_d172, G3 Hu coin with its render target vs D172's X2, G4 the settle with its relaxation)

| change | saving (measured) | gate |
|---|---|---|
| E1 settle windows: the rollout alone (the window objective and its construction were evaluated and never read there); one capture without the settle relaxation | settle 25 -> 12.4 s (bunny), 32 -> ~14 s (dragon) | e1 PASS (G1-G4 SAME) |
| E2 the target's draws cached (the fill's bisection per run) | 4.6 s (bunny), ~14 s (dragon) of the prepare | e1 PASS |
| E3 the thin set (a measurement) cached by the target sample's bytes | 20 s each | e1 PASS |
| E4 the Sinkhorn axis log-sum-exp from a shared-memory CUDA kernel: the same IEEE operations in the same order, the Toeplitz axis cost (checked bit for bit) in shared memory, two outputs a thread (losses/lse_cuda.py) | 0.38 -> 0.21 ms a transform at 87^3 (1.84x), 1.2x at 71^3; bit for bit on random fields with empty nodes, lines and NaNs | e2 PASS |
| E5 the u gate's transport map on the layer's rows only (the gate is zero elsewhere) | (in the per-attempt figure below) | e2 PASS |
| M1 memory batch (section 4) | - | m1 PASS; also SAME at 300k (deterministic, scales scene, 2 windows + settle) |

Measured on solo timing runs without `--profile` (caches warm, as a repeated run finds them):
- bunny 300k Hu: repo_d172 511 s (69 attempts, 6.30 s each; start-up 48.6 s, settle 25.0 s) -> repo_d174 371 s
  (55 attempts, 6.19 s each; start-up 16.6 s, settle 12.4 s). At equal attempts: -32 s start-up, -12.6 s settle,
  -0.11 s an attempt: about -52 s (-10 %); the rest of the measured -140 s is the run's own length (55 vs 69 attempts,
  the atomics' draw). End metrics: sil_iou 0.9891 / 0.9891, chamfer 0.0551 / 0.0551, thin 3.9 % / 3.3 %.
- dragon 300k Hu: repo_d172 1048 s (100 attempts, 9.59 s each; start-up 54.2 s, settle 32.7 s) -> repo_d174 963 s
  (108 attempts, 8.55 s each, -1.04 s = -10.8 % an attempt, mostly E4; start-up 23.7 s, settle 13.4 s). At equal
  attempts: -153 s (-15 %): 100 attempts 1048 -> ~895 s, 138 attempts (the profile run's length) ~1413 -> ~1220 s.
  End metrics: sil_iou 0.9874 / 0.9870, chamfer 0.0547 / 0.0547, thin 5.0 % / 4.7 %.
- So, exactly: the bunny 300k Hu run takes 6-8 min, the dragon 15-20 min (its length, 100-140 attempts, is the run's
  own draw).

Already done before (not redone): records off (D17), 96 px from the start (D18), the exterior search 1.77 -> 1.16 s
(D96), D138's batches 1-7, rt1 (D153). Measured and not worth it here: the cross / self solves on two streams (D138:
1.2 %), the commit-time kNN dedupe (< 0.5 %).

## 3. Result-changing speed-ups (gated A/B against Hu; nothing adopted)

| arm | relief min (coin, bunny) | colour min (band) | kept min | rest | gallery 40k s97 holds | 300k scales | every verdict Hu's? |
|---|---|---|---|---|---|---|---|
| Hu (D172) | 0.45 | 0.07 | 0.91 | ok | 6/6 | PASS (share 0.94, kept 1.00, 86 windows) | - |
| Hu2 = Hu --render_every 2 | 0.45 | 0.07 | 0.95 | ok | 5/6 (bob thin +0.035) | FAIL (share 0.20, 35 windows) | no |
| Hu4 = Hu --render_every 4 | 0.41 | 0.08 | 0.94 | ok | 6/6 | PASS (share 0.53, kept 0.99, 56 windows) | yes, by the letter |

`--render_every K`: the render gradient's adjoint sweep at every K-th iteration, the window's last one reused between.
Saving a window: 4 / 6 of the 8 render sweeps, dragon 0.47 / 0.71 s (5 / 8 %), bunny 0.44 / 0.66 s (7 / 10 %).
Hu4 keeps every verdict by the letter but loses 44 % of the 300k scales' content (share 0.94 -> 0.53; the run
stopped 30 windows earlier); not recommended. Hu2 fails. Readers: D172's (output/gpu/d174/auto/ab_*, AB_SUMMARY.md).

Not queued, with the measured reason:
- fewer line-search candidates (`--ls_max`, coded, off): 1.15 trials an iteration; a cap changes only exhausted searches
  (5 of 138 dragon windows): <= 1 %.
- lower picture resolution early: the render's work is the discs' splat (discs x 18 views x 4 corners), not the
  pixels (picture-sized operations ~5 % of the render, the render ~8 % of a window): <= 1 %; and it would change
  the held weight Hu calibrates on the first window.
- mixed precision in the render: scatter-bound, <= 3 % of a window at best, and not exact.
- fewer settle windows once at rest: after E1 the 30 settle windows cost ~12 s; the bunny reaches its rest floor at
  settle window ~20, so <= 1 %.

What 10 minutes would cost: the bunny is there (6.2-7.7 min solo with the exact changes). The dragon is not: with the
exact changes 15-20 min (measured 16.1 min at 108 attempts), Hu4 ~8 % less. Ten minutes means stopping near kept window
55-60 instead of 100-130 (from the
solo profile run's archive, output/gpu/d174/auto/cut_dragon.md): sil_iou 0.9874 vs 0.9880 delivered (-0.0006),
chamfer 0.0552 vs 0.0545 (+0.0007), thin uncovered 6.0 % vs 4.2 % (+1.8 points), merit 4x higher, end v_mean 3x
higher before the settle. Those sit inside the gallery tolerances (0.003 / 0.002 / 0.02) but thin uses nine tenths of
its budget; it would need a stop-rule A/B (not in the candidate list; the user's call).

## 4. Memory (scales scene, Hu, 2 windows + 1 settle window; tmp/d174/d174_mem.py: phases + every device buffer by owner)

| code | 300k device peak (window loop) | 500k | fit | 1M | 1.5M | 2M | 3M |
|---|---|---|---|---|---|---|---|
| repo_d174 | 18.5 GiB | 30.2 GiB | 0.9 + 58.5 GiB / M | 59 | 89 | 118 | 176 |
| repo_d174m (+ m1) | 16.0 GiB | 24.9 GiB | 2.6 + 44.6 GiB / M | 47 | 69 | 92 | 136 |
| repo_d174m2 (+ m1 + m2) | 14.5 GiB | 23.0 GiB | 1.7 + 42.5 GiB / M | 44 | 66 | 87 | 129 |

Where it goes (bytes a particle, 300k = 500k within 1 %, repo_d174 -> m1):
- MPM tape (PersistentAdjoint: 41 steps of x, v, C, F, Fraw, Fs, P, xu, J, ld, ls and their gradients; the control and
  its gradient): 23.4 -> 22.0 KB (Warp pool). Values ~9.3 KB, gradients ~9.3 KB.
- eval trajectory (the candidates' and the commit's rollout; per step x, v, C, F, Fraw, J; shared scratch): 7.7 -> 7.0 KB.
- optimiser (dF_c, u, Adam moments): 2.2 KB; window (spec, layer, spacing lists, dc_buf): 0.64 -> 0.68 KB.
- grid (MPM, per step on the tape): 0.18 GiB fixed (the shape's cell, 44^3); Sinkhorn graphs and potentials 0.03 GiB;
  target pack 0.1 GiB; render: exterior discs' lists 0.15 -> 0.21 GiB (300k -> 500k), pictures negligible; proximity
  lists inside the evaluations' transients; CuPy pools 0.3-1.0 GiB.
- torch transients in a line search: peak 8.5 -> 6.2 -> 4.7 GiB (300k), 14.1 -> 10.3 -> 7.4 GiB (500k) with m1 / m2. The
  spike m2 removed (+3.1 GiB at 300k, +5.6 at 500k, only in the line searches that look for the exterior's discs again)
  was the lattice's ring of nodes around every occupied cell (cells x 216 offsets as int64 and their sort).
- after m2 the 500k peak is Warp's pool 12.5 GiB (tape 10.2, eval trajectory 3.3) + torch's reserved 8.8 GiB (allocated
  peak 7.0) + CuPy 0.5 + the context 0.5; at 2M: Warp ~53 GiB, torch ~30 GiB.

M2 (implemented, exact, gate m2 PASS; the chunked ring against one chunk bit for bit on 300k bodies at three pitches):
the exterior lattice's ring of nodes a chunk of 65536 cells at a time, each chunk's unique keys merged by one more
unique (the same sorted set). -1.5 GiB at 300k, -1.9 GiB at 500k; 2M 92 -> 87 GiB; no time cost.

M1 (implemented, exact, gate m1 PASS and bit-SAME at 300k deterministic): the driven steps' control buffers only (the
released steps read the trajectory's release_dFc, never their entries: tape control, its gradients and the eval
trajectory's dc_buf halve), evaluations keep no expanded control and no velocity stack, the domain check frame by
frame. -2.5 GiB at 300k, -5.3 GiB at 500k; slope 58.5 -> 44.6 GiB per million particles; 2M 118 -> 92 GiB; no time
cost.

The 2M plan (each saving at 2M from the per-particle measurements above):

| step | kind | saving at 2M | time cost | state |
|---|---|---|---|---|
| m1 driven-only control, no kept stacks | exact | -26 GiB (118 -> 92) | none | done, gated |
| m2 the exterior lattice's node ring in chunks | exact | -5 GiB (92 -> 87; the spike scaled less than feared) | none | done, gated |
| tape gradients as a two-step ring (step t's adjoint reads step t+1's cotangents only; the V seeds injected per step) | exact (each element receives the same additions in the same order) | ~-17 GiB | ~0 | proposed |
| eval trajectory on the tape's state buffers (the line search runs after the sweeps and before the next forward) | exact | ~-12 GiB | ~0 | proposed |
| or: tape checkpoints every 5 steps, one segment recomputed per sweep | exact in deterministic mode; the atomics' order otherwise | ~-11 GiB | +4 % (fused three-cotangent sweep) to +12 % | proposed |
| render views / candidate buffers in place (Adam step, gradient sums with the same rounding) | exact | a few GiB | ~0 | proposed |
| lower-precision storage | not exact (every stored field feeds the result) | - | - | not proposed |

After m1 + m2 the 2M estimate is 87 GiB (Warp ~53, torch ~30, CuPy ~2). With the gradient ring (-17), the eval
trajectory on the tape's buffers (-12) and the in-place line-search candidates (~-5): about 53 GiB; the tape checkpoints
(-11, in place of or beside the shared eval buffers) bring it to about 42-45 GiB: 2M fits 48 GB only with all of them and
a thin margin; 1.5M (~33-35 GiB with the same steps) fits comfortably, and 1M already fits now (44 GiB). Time per 2M run: per window ~60 s (MPM / adjoint / evaluations ~6.7x the
300k's, Sinkhorn ~12x: the loss grid follows N to ~164^3 and a transform costs n^4), ~130 windows for a dragon-like
target: ~2.2-2.5 h on one card. No run above 500k was launched.

## Files
- output/gpu/d174/runs: P_* (profile runs), T_* (timing), M_* (memory probes), the A/B runs (D172's names, arms Hu2 / Hu4)
- output/gpu/d174/auto: cut_*.md, mem_*.json/log, ab_*; output/gpu/d174/gates: e1, e2, ab, m1 (+ d300_cmp.txt)
- tmp/d174: the tools and RESUME.md

## 5. The 10-minute dragon through the existing stop rule (user decision 2026-10-29: +1.8 pt thin and a small shape loss accepted, no render-content loss)

The only existing stop parameters are PipelineConfig.tol (a window improves when its merit is below best - tol |best|,
run/selection.py:65 and :115) and patience (5 stale windows, :118 / :121; the plateau :41). Arms on repo_d174 copies
with only tol changed (repo_d174t10 / t07 / t06 / t05; flag-free), Hu's flags, seed 97; read with the fast readers
(tmp/d174/d174_read2.py; output/gpu/d174/STOP_SUMMARY.md, auto/st_*).

| tol | dragon 300k solo: wall, attempts | dragon sil_iou / chamfer / thin | bunny 300k: wall, thin | scales share (>= 0.40; Hu 0.94) | plaque share / kept | relief min (coin, bunny) | gallery 40k s97 |
|---|---|---|---|---|---|---|---|
| 0.003 (Hu) | 16.1 min, 108 | 0.9870 / 0.0547 / 4.7 % | 6.2 min, 3.3 % | 0.94 | 0.63 / 1.00 | 0.45 | 6/6 |
| 0.05 | 11.2 min, 68 | 0.9869 / 0.0552 / 5.7 % (+1.0 pt) | 5.0 min, 4.2 % | 0.75 | 0.63 / 1.03 | 0.50 | 6/6 |
| 0.06 | 8.7 min, 46 | 0.9850 / 0.0560 / 7.0 % (+2.3 pt) | - | - | - | - | - |
| 0.07 | 7.7 min, 39 | 0.9854 / 0.0563 / 7.1 % (+2.4 pt) | - | 0.68 | 0.64 / 1.01 | - | - |
| 0.10 | 7.5 min, 37 | 0.9848 / 0.0563 / 6.9 % (+2.2 pt) | 4.6 min, 4.6 % | 0.60 | 0.64 / 1.03 | 0.50 | 5/6 (dragon chamfer +0.0025) |

Every arm is at rest (G3, end v_mean as Hu's); the band's colour stays at Hu's 0.07 (Hu does not carry colour).

Why no value lands at ~window 55: tol is not a stop parameter alone. The same `improved` flag sets the step anneal
(selection.py:119: x1.15 after an improving window, x anneal_stale = 0.7 after any other) and the latch (:80-82). With
tol >= 0.06 the dragon alternates improving and non-improving windows from window 18-23, the step scale falls to its
floor 0.05 by window 34-42 and the run crawls to the stop (37-46 windows) with a worse shape than Hu had at the same
window (thin 6.9-7.1 % against 6.3 % at window 41 of Hu's run). With tol 0.05 the floor comes at window 50 (68 windows).
The windows jump 68 -> 46 -> 39 between tol 0.05, 0.06 and 0.07: a cliff, not a dial.

Result: within the existing parameters, <= 10 min costs more shape than accepted (thin +2.2-2.4 pt, sil_iou -0.0016 to
-0.0022, chamfer +0.0013-0.0016; gallery 5/6 at 0.10) and lowers the scales' content to 0.60-0.68 (above the 0.40 bar,
far from 0.94); the plaque and the relief scenes keep theirs. The cheapest existing setting that keeps every bar and
the accepted shape cost is tol 0.05: 11.2 min (scales 0.75). Nothing committed as a default.

The smallest definition change that would make it a pure stop: count the stop's stale windows (selection.py:115-118,
the plateau's `last` :116 / :41, the stop :121) against a tolerance of their own, while the anneal (:119) and the latch
(:65 / :80-82) keep tol = 0.003; one new value (default = tol: bit-SAME), behind an option. Replayed on the recorded
runs without the anneal's feedback (tmp/d174/d174_stop.py), a stop tolerance of 0.10-0.12 stops the dragon at 60-64
attempts in all three recorded runs (~10-10.5 min with the exact speed-ups), the bunny at 33-36, the scales at 55-57,
the plaque at 31-40 and the 40k gallery at 21-35 windows. Not implemented (the user's call).
