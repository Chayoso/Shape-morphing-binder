---
name: michael-settled-transport-branch
description: "2026-09-29 — settled transport is the base (settled-base); GPU-only refactor uncommitted (S3 failed, likely trap timing); support ratio form fixes the 300k line-search trap; C_R adoption candidate; open: late reversals, cost"
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-09-29T14:44:11.411Z
---

`origin/michael/settled-transport` has NO common ancestor with `v3-grid-gs`: it is the public release
(`origin/main` at 680622e, the 2026-09-23 state) plus ONE commit 835af64 (Michael Jin, 2026-09-29, 27 files,
+2526 lines, 9 test files). It has none of our post-09-23 features (settle_pin*, ctrl_rprop*, disc_ref,
pace rules); ours has none of his. No automatic merge; porting is by hand.

His method (`--solver_mode settled_transport --ot_iters 1600 --support_weight 8 --nn_sampling_berth`):
- objective = fixed-target grid Sinkhorn divergence (debiased, separable log-domain, eps scaling; envelope
  gradient, Sinkhorn solved under no_grad), NO paced target;
- T driven + T released steps per window (dFc and u withdrawn), loss at the released end, released-phase
  motion penalty, drift term (T dt v)^2 in the transport state energy;
- local support = KDE log-density floor (half the target's median density) bounded by the transport
  energy E + E B / (E + B);
- matched shading target, render weight calibrated once per resolution, one-sided PCGrad kept;
- adjoint fixes: polar-factor VJP, per-step fracture masks (bond_history), bond velocity replay fix.
Caveats: validated at 100k with unit masses (`--mass_ref_n 0`), uniform masses required, 2T rollout
(GPU memory 24 GB at 300k vs 12 GB), support_weight 8 and ot_iters 1600 are validation values.

**Why:** my measured causes of the 300k transit vapour (cell-sum null space, pace one cell ahead, no
thinning term) are each removed at the objective level; the lead, taper and governor levers within our
formulation were all refuted on 2026-09-26 ([[resume-2026-09-25-pin-gates]]). I told the user I would
choose his formulation as the base CONDITIONAL on a paired 300k test, and port his adjoint fixes regardless.

**How to apply:** B1 (experiments.md 2026-09-29 09:50 CDT): separate checkout
`/data/relcfd/chayo/physmorph_v2/repo_mj` (git archive), outputs `$OUT/mj/` — `mj300` (settled, GPU 0),
`mjb300` (his legacy baseline, GPU 1), `ours300` (current code adb7492, bm300's flags, GPU 3), all
`--n 300000 --seed 97`. Probes re-uploaded to `$OUT/mj/probes/` (the old `$OUT/scratch` and every
reference archive were removed in a cleanup between 09-26 and 09-29; local copies of all probes live in
the session scratchpad). Gates include the eye-corrected ones (ear tip >= 13, ear onset, window count).
Another agent/session has been committing to v3-grid-gs (125 commits 09-26..09-29, P-numbers up to P338):
label my entries B1.. to avoid colliding with their numbering, and commit only my own files.

**B1 result 2026-09-29 10:20 CDT (paired, seed 97, 300k bunny):** mj300 = 7 min / 31 windows, silIoU 0.9851,
det F 0.934, 0 strays, late surface motion 0.007/0.009 with NO pins, uniform progress by depth, ear onset much
earlier, splat tail 0.0003; ours300 (current code, bm300 flags) = 12 min / 47 windows, 0.9769, det F 0.608, 30
strays, unpinned surface 0.128/0.113, tail 0.0008. Both miss the ear-tip gate (11.1 vs 8.8, gate 13). mj300
weaker at frames 152-190 transit density, under-fill 0.166 vs 0.140, less facial relief, ring banding on the
growing ears. 4K PBR renders via `$OUT/mj/render4k.sh` (ours) and `render4k_nopin.py` (mj: supplies empty pin
arrays the renderer requires). Page https://claude.ai/artifact/TZEZd3xnsfGQr5y4T4fRMc (source
output/r300_page/mj_compare.html). mjb300 (his legacy baseline) was still running.

**B2 2026-09-29 11:15 CDT (fd8c865):** gradients verified on the settled path (157 tests pass; FD vs autograd
0.97-1.03 at W1 and W20; graph path == line-search path). Render share 0.33 (W1) -> 0.90 (W20); u channel is
~100 % render-driven (physics grad on u ~0). Render-off twin (MJ_LAMBDA_SCALE=0 via `$OUT/mj/mjdbg*_chain.sh`):
settled 0.9851 -> 0.9672 (seed spread 0.002), surface error 11.7 % -> 23.5 %; current code 0.9769 -> 0.9716.
Momentum: settled keeps the COM within 0.02 sp over the whole morph (stress cannot change momentum; only the
outer layer's relaxation/u are position edits, 11-22 % of that layer's motion); the current code drifts the COM
0.6-0.8 sp (commit PIC null-space projection, pins, assimilation), render off too. mjb300 (release legacy) =
D1 breathing: 70 % negative-corr windows, 100 % in the second half, streak 99.

**2026-09-29 11:50 CDT — settled-base created (user: "이쪽 코드로 local을 잡자").** New branch `settled-base`
(pushed, origin/settled-base) = 835af64 + renderers/modules/tests/probes/hyde06_env from v3-grid-gs. It lives in a
separate WORKTREE `C:\dev\Shape-morphing-binder-settled` because another session works in
`C:\dev\Shape-morphing-binder` on v3-grid-gs (do not switch that checkout). Server: checkout `repo_settled`
(git archive; env script sets REPO=repo_settled, OUT=output/settled, $SETTLED), reference snapshot `repo_v3snap`
(v3-grid-gs d2f8915). Added: `--render_weight_scale` (render-off twin), `--reject_stop`, pin-less archive support
in both renderers. Tests 207 pass / 2 skip. Log: settled-base docs/experiments.md (S1, S2). S2 gates running
(`$OUT/s2/`: st40000_* settled GPU 0 then st300000_dragon; v340000_* g41pw form GPUs 1/3; v3300000_dragon bo300
form GPU 2); table via `scripts/probes/settled/gate_table.py $OUT/s2 40000 <targets>`.

**2026-09-29 11:45 CDT (new session) — the user's standing list, all on settled-base (mj base):** (1) docs: repo keeps
3 (README, docs/method.md, docs/experiments.md; CLAUDE.md is a 1-line pointer) — done in 1c4135e; the 3 untracked v3
docs archived to C:\dev\physmorph_archives\untracked_docs_20260929 and deleted; tracked v3 docs leave the working tree
when the local folder switches to the new main. (2) experiments on many meshes: S2 gallery on hyde06 `$OUT/s2`.
(3) GPU-only with CuPy like the v3 line: CuPy 14.0.0 (device KDTree + ndimage EDT) exists isolated at
`/data/relcfd/chayo/physmorph_v2/deps/gpu_pipeline` (also ships numpy 2.2.6; v3's `scripts/ops/gpu_env.sh` sets
PYTHONPATH/CUDA_HOME/CUPY_CACHE_DIR); v3's pattern = CPU "prepare" stage (mesh -> points, cached) then CUDA
execution with no CPU fallback (`physmorph/compute.py` on v3-grid-gs). Profile (300k settled, 4 windows): loop is
GPU-bound (.item() sync 20 s, backward 12 s, Sinkhorn 21 s); CPU = trimesh voxelize in sampling (~10 s setup), numpy
stats, knn_self_torch CPU bounds/fallback, batched_det .cpu, assimilation .cpu, target_dt_grid scipy EDT (c2f).
(4) refactor every file to <= 500 lines (10 files over: optimizer 1841, runner 1226, render_photoreal 1033,
test_render_controls_physics 950, pipeline_run 940, ot 677, volumetric 643, surface_recon 601, config 569,
viewer/server 539); plan = prune options never executed by the settled (and legacy) recipe using coverage
(coverage 7.16.2 isolated at deps/tools; runs in `$OUT/cov`, chain cov_chain.sh -> cov.json), then split, then CuPy.
Warp kernel bodies and autograd backward methods show as unexecuted under coverage — never prune them on coverage alone.
(5) then switch origin/main to settled-base (fast-forward from 680622e) and the local folder to main; remove worktree.
Also pending: the red->green loss/render-gradient visualization (viz300 run with --grad_dump at `$OUT/viz`).
Every experiment on hyde06 only (user repeated 2026-09-29).

**2026-09-29 12:50 CDT — GPU-only rewrite done in the working tree (uncommitted, branch settled-base; local folder is
now on settled-base, worktree removed).** New layout: physmorph/gpu.py (CuPy KDTree/ndimage via DLPack, float64 like
cKDTree), prepare.py (CPU sampling, cached via PHYSMORPH_CACHE), pipeline/{config,target}.py, pipeline/window/
{setup,objective,rollout,solve,telemetry,layer}.py, pipeline/run/{state,selection,runner}.py, losses/grid_ot.py
(grid part of ot.py verbatim); optimizer/runner/control_basis/grad_combine/grid_smooth/surface_local/dressing/
gauss_loss/ot.py and v3-era probes deleted; CLI 16 flags (recipe = defaults, no $RECIPE/$SETTLED). Gotcha: torch
preloads a CUDA 11 NVRTC; import cupy before torch (physmorph/__init__.py does it; conftest imports physmorph).
Server deploy: scratchpad rf/deploy.sh -> repo_gpu, outputs output/gpu. Verified: first window equal to the
pre-refactor code within 1e-7..8e-7 relative (40k and 300k), tests 256 pass / 2 skip (exit 0), 2x faster.
S2 result logged in experiments.md: all 19 x 40k + 300k dragon pass, but the 300k dragon's surface gap is 34.5 %
vs v3 11.7 % (relief gap grows with N). S3 (new code on the whole gallery vs S2) running: output/gpu/g/status.txt.
A subagent is splitting scripts/render_photoreal.py (1033 lines) with pixel-compare verification in repo_photo.

**2026-09-29 evening — status (all in docs/experiments.md on settled-base, refactor STILL UNCOMMITTED).**
- S3/S3b (refactor equivalence) formally FAILED at 300k dragon (new median silIoU 0.9761 vs old 0.9775, p=0.026);
  a subagent found no logic difference. Later finding: 300k dragon runs of BOTH codes stop in a line-search trap
  whose timing is chaotic, so S3b likely measured trap timing; re-judge S3 after adopting the fix. Commit/push of the
  138-file refactor and the main switch are waiting on that.
- "300k worse than 40k" was mostly the 1.5-own-spacing threshold (R1d, `uncovered_40k` column in thin_regions.py).
- Trap mechanism (D1/D1b, `--ls_probe`): the log-form support penalty is quadratic in log-density deficit, so one
  isolated particle holds 40-60 % of B; its end position jumps with any dFc change -> step-independent +1 % objective
  jump -> line-search collapse -> run stops ~window 30. Fix = `--support_form ratio` (relu(1 - s/f)^2, bounded by 1).
- Adoption candidate C_R = `--support_form ratio --support_target_ref --loss_follows_n` (R3): 300k dragon silIoU
  0.9833-0.9838, v3-level coverage at fixed world distance with aniso ~1.15; 40k gallery follow-up (R3d) passes;
  render twin confirms render effect (+0.010). Open before adoption: late second-half reversals (render-driven tail,
  g_share 0.9+) fail the S2 gate on more meshes than A; cost 80 min per 300k dragon run; C_R not yet at 300k
  nefertiti/V. Server repos: repo_r5 = current tree; outputs output/gpu/r3, census jsons output/gpu/thin.
