# AGENTS.md — PhysMorph collaboration guide

Guide for AI coding agents (Claude Code, Codex, …) working in this repo.

## ⚠️ Read first

**Nothing here is under version control.** The enclosing git repo is `C:/dev` — that is *vcpkg*,
not this project. `git ls-files physmorph docs legacy` returns zero. **Every delete is permanent.**
Archive before removing anything you did not create.

Prior work (all MatCast scripts/docs, the earlier heroes, every result) was deliberately wiped on
2026-07-27 for a clean restart. It survives only in
**`C:/dev/physmorph_archive_20260727.zip`** (81 files, 1.0 MB) and in published artifacts on
claude.ai. Server results were wiped the same day (1.9 GB → 59 MB).

## What this project is

**PhysMorph**: a source shape is carried to a target by *real* elastodynamics — the optimisation
variable is the MPM **deformation-gradient control field `dFc`** (`F_e = (F + dFc) Fp⁻¹`), not a
servo pulling particles to a goal. The thesis being pursued: **render guidance makes the morph
qualitatively better than 3-D supervision alone.** Baseline = the same dFc optimisation driven by
volumetric mass matching only (Xu et al.).

## Layout — only two code trees remain

### `physmorph/` — the Warp rewrite (this is the working codebase)
- `mpm/` — MLS-MPM engine ported from the C++ oracle. `kernels.py` (cubic B-spline 4³, APIC,
  `eta_sym` objective viscosity, `eta_mode` exponential damping, `v_max` clamp), `state.py`
  (`MPMParams`), `traj.py` (per-step arrays on `wp.Tape`), `function.py` (torch autograd bridge:
  `dFc` leaf → rollout → `x_T, F_T`), `step.py`.
- `losses/` — `volumetric.py` **`d_vol`: mass matching, the Xu et al. objective**;
  `silhouette.py` `d_img` multi-view soft silhouette; `render_guidance.py` per-particle
  displacement from silhouette **and colour/gram** terms.
- `morph.py` — **`morph_mass()`: single-graph per-frame Adam on `dFc`, `L = D_vol + λ·D_img`.**
  This is "Xu et al. + render loss" and is the natural starting point.
- `morph_physical.py` — plasticity-driven rest-state migration; blends transport and render
  *displacements* (`render_gain`).
- `style_transfer.py` — displacement-space loop with colour-gram style terms.
- `plasticity/` (Sinkhorn / sliced-OT / assignment), `render/` (3DGS raster, covariance),
  `surface/`, `sampling/`, `viewer/`.
- `docs/method.md` is the **equation contract** these files cite as `docs/SPEC.md` (renamed; the
  docstring paths were never updated). Equation numbers in `mpm/*.py` refer to it.

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

1. **No GPU work on this machine.** Local = writing code + `python -m py_compile`. All sims/renders
   run on hyde06 via the jump host: `ssh -J chayo@hyde01.dabh.io chayo@hyde06.dabh.io`,
   repo copy `~/physmorph_v2`, python
   `/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python` (3.10.20).
   Long jobs: `setsid nohup env CUDA_VISIBLE_DEVICES=<n> $PY script.py … > log 2>&1 < /dev/null &`.
   Each ssh command needs its own `cd ~/physmorph_v2;` — chaining `cd && … &` backgrounds the whole
   list and later commands run in `$HOME` with unset vars.
   Check `nvidia-smi` first; never kill other users' jobs; do not touch `~/Shape-morphing-binder`
   on hyde06. C++ bindings live at `~/xu_baseline/`.
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
