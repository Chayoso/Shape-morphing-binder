# AGENTS.md — PhysMorph collaboration guide

Guide for AI coding agents (Claude Code, Codex, …) working in this repo.

## ⚠️ Read first

**This project is now its own git repo** (`git init`, 2026-07-27, branch `main`). Remote
(added 2026-09-01): `origin = https://github.com/Chayoso/Shape-morphing-binder.git` — the
original PhysMorph-GS repo with its OWN unrelated `main` and paper-era branches. **Never
push to origin/main from here** (histories are unrelated); push work branches only
(`v3-grid-gs` = live work, `v3-vbd-experiments` = snapshot). HTTPS credentials are cached;
`gh` is not installed. Commit freely; deletes are recoverable *for tracked files only*.

Still permanent, because they are deliberately untracked: `output/`, `gaussian-splatting/` (an
unmodified graphdeco-inria clone, 134 MB), `legacy/build/`. Archive those before removing.

Prior work (all MatCast scripts/docs, the earlier heroes, every result) was wiped on 2026-07-27 for
a clean restart — *before* the repo existed, so none of it is in history. It survives only in
**`C:/dev/physmorph_archive_20260727.zip`** (81 files, 1.0 MB) and in published artifacts on
claude.ai. Server results were wiped the same day (1.9 GB → 59 MB).

## What this project is

**PhysMorph**: a source shape is carried to a target by *real* elastodynamics — the optimisation
variable is the MPM **deformation-gradient control field `dFc`** (`F_e = (F + dFc) Fp⁻¹`) — PLUS,
in the current recipe, a direct position channel on the outer layer (`--layer_ctrl`: the per-particle
normal displacement `u`, one spacing a window) and the layer relaxation projection
(`--layer_relax`), both applied to positions outside the stress path (docs/method.md 10.15–10.16,
surface_gradient.md §6–§7; the hybrid was named by the 2026-09-23 audit, docs/diagnosis_300k_20260923.md). Not a
servo pulling particles to a goal. The thesis being pursued: **render guidance makes the morph
qualitatively better than 3-D supervision alone.** Baseline = the same dFc optimisation driven by
volumetric mass matching only (Xu et al.).

## Layout — only two code trees remain

### `physmorph/` — the Warp rewrite (this is the working codebase)
- **`pipeline/` — the blessed paths (docs: `overview.md` / `method.md` / `experiments.md`,
  rewritten 2026-09-01).** `config.py`, `render_loss.py` (multi-elevation asymmetric
  D_render + EMA λ balancer), `grid_smooth.py` (Sobolev/grid-GS render direction, §6;
  Chebyshev-accelerated, multi-channel since 2026-09-14), `control_basis.py` (coarse
  node × time-knot control basis), `grad_combine.py` (PCGrad both sides / CAGrad /
  physics-anchored blend), `optimizer.py` (multi-leaf line-searched Adam over the control
  leaf + material field, warm-startable), `runner.py` (dynamic family). Physics-only
  baseline = same path, `lambda_auto=0`. **2026-09-14 contract** (`docs/render_controls_physics.md`,
  all opt-in): `control_grid/control_tknots`, `render_F_geom` (geometric F_g from
  `mpm/kernels.k_geom_update`, exposed by `mpm/function.warp_mpm_ext`), `w_kin_running`,
  `grad_project_mode`, `render_gs_cheb`, `gauss_robust_eps`, `loss_units=density`,
  `mpm/discretisation.py` + `pipeline_run --ppc`, arms `render_ctrl*`.
- `metrics.py` — gate metrics (chamfer, sil_iou, hole_frac, jitter); raw sim state only,
  no operator shared with any loss.
- `mpm/` — MLS-MPM engine ported from the C++ oracle. `kernels.py` (cubic B-spline 4³, APIC,
  `eta_sym` objective viscosity, `eta_mode` exponential damping, `v_max` clamp, opt-in
  support-gated APIC `k_cell_count`/`k_support_gate` — `MPMParams.gate_r_lo/r_hi/n0`,
  `pipeline_run --gate_lo/--gate_hi`, method.md eq (22)), `state.py`
  (`MPMParams`), `traj.py` (per-step arrays on `wp.Tape`), `function.py` (torch autograd bridge:
  `dFc` + optional per-particle `λ,μ` leaves → rollout → `x_T, F_T, v_T`), `step.py`,
  `conditioning.py` (`condition_F`: reflection repair, counted; no silent SV projection).
- `losses/` — `volumetric.py` **`d_vol`: mass matching, the Xu et al. objective**;
  `silhouette.py` CIC splat primitives (azimuth + elevation).
- `plasticity/` — `assimilate_elastic` only (exact elastic-stretch commit assimilation).
- `render/` (3DGS raster, covariance — G6 heroes), `sampling/` (voxel fill: 'orthographic'
  first + streak strip since 2026-09-16 — the 'base' fill drew 1-voxel columns on the
  non-watertight bunny, visible as a line above the ear at 40k; `tests/test_sampler_fill.py`),
  `viewer/` (in-process
  `LiveServer(port)` AND the file-backed `filehub.FileHub` / `LiveServer.to_dir` sink read
  by the standalone `scripts/viewer_serve.py`; local `scripts/viewer_tunnel.py` keeps the
  ssh tunnel — `docs/viewer.md`). Probes: `scripts/probes/oscillation_triage.py`
  (`docs/oscillation_triage.md`), `scripts/probes/gate_probe.py` and the hyde06-side
  `scatter_probe2.py` (`docs/thin_feature_transport.md` — the 2026-09-15 "scatter then
  return" dossier: sub-cell density deficit of the ear stream, not volume, not fracture;
  `w_coh`/`w_bond`/`vol_frontier` falsified; control basis + `--ppc 8` were the levers). 2026-09-17: the MASS-EJECTION cause is the cell size relative to the shape (numerical fracture needs a one-cell gap); the recipe is now `--cell_diag 26` (dx = source bbox diagonal / 26, ppc = N dx^3 / V) — docs/method.md §10.9, docs/experiments.md 2026-09-17.
- `tests/` — 39 CPU/warp-CPU tests incl. an end-to-end pipeline smoke; run `python -m pytest`.
- **Deleted 2026-09-01** (git history ≤ `2607972`): v1 loops (`morph.py`,
  `morph_physical.py`, `style_transfer.py`), `losses/render_guidance.py`, v1 plasticity
  (Sinkhorn/sliced-OT/auction/`update_fp`), `surface/`, `trajectory_opt.py` + old scripts.
  Parity gates live on as G1a/G1b (pipeline_run + tests); the C++ oracle is `legacy/`.
- `docs/method.md` is the **equation contract** these files cite as `docs/SPEC.md` (renamed; the
  docstring paths were never updated). Equation numbers in `mpm/*.py` refer to it.
  `docs/experiments.md` carries the gate contract + result log.

### `legacy/` — the C++ original (Xu et al. DiffMPMLib3D) + Python bindings
- `DiffMPMLib3D/` — `CompGraph.{h,cpp}` (`OptimizeDefGradControlSequence`, `EndLayerMassLoss`),
  `ForwardSimulation.cpp` (the oracle our Warp kernels were ported from), `BackPropagation.cpp`,
  `Elasticity.cpp`, `Grid`, `PointCloud`.
- `diffmpm_bindings.cpython-310-x86_64-linux-gnu.so` — **prebuilt, and it RUNS on hyde06**
  (server python is 3.10.20, matching). It links libtorch, so **`import torch` BEFORE
  `import diffmpm_bindings`** or the import dies on `libc10.so`.
  Exports: `CompGraph`, `OptInput`, `Grid`, `PointCloud`, `E2ESession`,
  `load_point_cloud_from_obj`, `load_shell_biased_point_cloud_from_obj`, `p2g`,
  `calculate_point_cloud_volumes`, `calculate_lame_parameters`, `get_positions_from_pc`.
- `run.py`, `utils/`, `configs/` — a Python driver layer with **extensions** (control_guidance,
  covariance_opt, chamfer_plasticity, rendering_utils). NOT part of the clean baseline: for a
  pure Xu et al. comparison, drive `CompGraph` through the bindings directly.
- `configs/ablation_bunny_ppc6_*.yaml` — isosphere→bunny at grid_dx 1.0, dt 1/240,
  smoothing 0.955, ppc 6, shell sampling. Same target we use.

## Things the C++ has that the Warp rewrite does NOT (measured 2026-07-27)

These are the leading candidates for the unresolved problems, and they are ports, not research:
1. **Step acceptance** — `OptInput.max_ls_iters`, `optimize_single_timestep(..., max_line_search_iters=10)`.
   A line search rejects a step whose forward rollout blows up; the Warp loops use fixed-step Adam
   with clipping only.
2. **Adaptive alpha** — `adaptive_alpha_enabled / _target_norm / _min_scale`.
3. **Gradient-norm λ balancing** — `CompGraph.get_control_layer_grad_norm()` docstring:
   *"lambda = alpha * phys_norm / render_norm"*. The Warp loops use fixed gains with RMS
   normalisation instead.
4. **Render-gradient injection hooks** — `accumulate_render_grads(dLdF, dLdx)`,
   `clear_render_gradients()`. Render guidance is designed into the C++ backprop.

## Hard rules

1. **ALL simulation runs go to hyde06 — no pipeline runs on this machine** (user directive
   2026-09-01, reaffirmed after a brief local-run experiment). The machine does have an
   RTX 4090 Laptop GPU (16 GB, torch cu128 + warp both see it) — do not rediscover this —
   but it is only for what the user explicitly approves. Local work = writing code,
   `python -m py_compile`, and the `tests/` suite (which is CPU/warp-CPU only).
   Server access via the jump host:
   `ssh -J chayo@hyde01.dabh.io chayo@hyde06.dabh.io`,
   **everything under `/data` (user rule 2026-09-16 — nothing is run from or written to `$HOME`)**:
   repo copy `/data/relcfd/chayo/physmorph_v2/repo` (deployed by `scripts/ops/deploy.sh`),
   outputs `/data/relcfd/chayo/physmorph_v2/output`, python
   `/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python` (3.10.20). Server-side scripts
   `source scripts/ops/hyde06_env.sh` (REPO / OUT / PY / thread caps / RECIPE); the pipeline,
   its flags and the ops scripts are summarised in `docs/pipeline.md`.
   Long jobs: `setsid nohup env CUDA_VISIBLE_DEVICES=<n> $PY script.py … > log 2>&1 < /dev/null &`.
   Each ssh command needs its own `cd $REPO;` — chaining `cd && … &` backgrounds the whole
   list and later commands run in `$HOME` with unset vars. Never put an unbracketed run name
   in a `pgrep -f` pattern inside the same ssh command (it matches the ssh shell itself).
   Check `nvidia-smi` first; never kill other users' jobs; do not touch `~/Shape-morphing-binder`
   on hyde06. C++ bindings live at `~/xu_baseline/`. **2026-09-14: the jump host hyde01
   rejected the local ed25519 key all day** (JumpCloud re-syncs `authorized_keys`); when
   `Permission denied (publickey,password)` appears at hyde01 itself, only the user can
   re-register the key — do not burn the session retrying.
2. **Rendered deliverables get per-frame visual QA before shipping** — extract every frame, inspect
   against the rubric (closed solid / no crossfade ghost / silhouette continuity / texture rides the
   surface), fix, re-run.
3. **Metrics never consume the renderer.** Reported numbers come from raw simulation state.
4. **State the discretisation with every recovered/fitted number.** The rollout is first-order in
   Δt and the error at coarse `sub` is a large fraction of the signal; fitting across a
   discretisation mismatch biases results by tens of percent.
5. Adversarial verification before anything ships: a subagent gate (Workflow `agent(model=…, effort="high")`)
   that is told to *refute*. Findings cite `file:line`; the implementer answers every one.

## Conventions

- Python: numpy / torch / warp; scripts are argparse CLIs; terse comments that state constraints.
- Compile-check before pushing: `python -m py_compile <file>`.
- Korean is the user's working language; code and docs are English.
