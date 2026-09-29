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
  `mpm/discretisation.py` + `pipeline_run --ppc`, arms `render_ctrl*`. **2026-09-24 (all opt-in, method.md §10.17a–10.23):** `--disc_ref`, `--plan_native`, `--shift_sub`, `--commit_pic`, `--pace_project`, `--w_kde`, `--ot_handoff`, `--outer_latch_reversal`, `--rest_commit[_reversal]`, `--dvol_form`, `--render_until`, `--rebound_probe`.
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
- **Render influence reporting (user 2026-09-28):** report rendering-loss influence
  with future runs. Keep adaptive lambda/direction norms, actual accepted updates,
  image-loss changes and independent raw-state evidence distinct; norm share is
  not causal displacement share. Standard drivers write `*.render_influence.json`
  and `.md`. P302 shared surface loss is experimental and OFF by default; its
  broad finite-difference and cap1 quality gates remain open. See
  `docs/surface_render_p302.md` before using or promoting it.
- **P303 continuation:** `docs/continuous_raster_p303.md` records the opt-in
  continuous CUDA raster and its isolated build/stream contract. Its cutoff and
  recorded live-cloud directional tests pass; legacy remains default and no GS
  quality/full-morph/rest/artifact promotion follows. The corrected-core,
  shift-off raw-endpoint comparison improves motion but loses target coverage;
  do not promote PIC removal from those motion numbers alone.
- **P304 continuation:** `docs/reference_swap_p304.md` distinguishes coarse
  transport arrival (8.75sp in the audited300k run) from convergence/rest.
  Both old/current references reward W20's actual density fitting; target
  refresh is not its sole cause. The observer is read-only and opt-in. Do not
  infer all-window causality or introduce another pin threshold from this test.
- **P305/P306 continuation:** `docs/inner_budget_p305.md` rejects extra inner
  iterations as W20's rest remedy: lower merit accompanied greater motion.
  This is not final convergence. `docs/terminal_braking_p306.md` preregisters a
  private, noncommitting terminal-body feasibility probe. Its displacement
  coefficients must remain fixed; only the remaining joint radius is available.
  No additional kinetic weight or stopping policy is promoted by these probes.
- **P306/P307 status:** terminal-only braking reduces speeds but fails strict
  density/coverage gates (`docs/terminal_braking_p306.md`). P307's original
  capture and C-repeat gate remain failed; exact-input/ownership checks isolate
  C-only variation also within the same instance. A fresh live-callback
  compensation ran without archive admission:4 endpoint updates reduce its
  discrepancy78.84% and terminal speeds about26%, but increase saved-step RMS
  and fail volume/local-density gates (`docs/braking_compensation_p307.md`).
  No compensated state was committed. Do not lower gates, add endpoint-only
  iterations or promote a stopping policy from these records.
- **P308 status:** constrained running repair is diagnostic-only. Its fresh
  run accepted0 repairs:8 linear-feasible forward trials missed the actual
  volume/render ceilings; in3 smaller trust balls the active-set solver found
  no feasible linear step.
  H5 passes all raw shape/supply and motion gates but still fails both data
  gates, beyond the observed scalar repeat ranges. Do not promote it or widen
  ceilings. See `docs/running_repair_p308.md` for observed-model remainders and
  the bounded correction direction; persistence and4K quality remain untested.
- **P309 status:** model-remainder correction restores prepared data and reduces
  running motion in4 accepted repairs, but all fail raw coverage/shape gates.
  All68 evaluated trials also fail the final gate; no held-out replay or commit.
  `docs/remainder_repair_p309.md` records the result. P310 filters raw quality
  before moving the origin, keeping thresholds fixed. These metrics then serve
  candidate selection, not independent post-selection validation. Total-merit
  and actual coupled continuation requirements are in
  `docs/candidate_commit_contract.md`; `on_checkpoint` remains read-only.
- **P310 / next diagnostic:** same-origin quality filtering accepts0 repairs
  (24 forwards,3 no-step results); only h6/c2 restores data and still fails raw
  IoU/upper coverage/source density. `docs/paired_braking_p311.md` preregisters
  .5/.25 terminal strengths within one fresh callback, sharing ONE baseline
  triplet/ceilings/cohorts/lambda. Both arms must be retained. The read-only
  original-merit API is implemented/CPU-tested but has no GPU closure yet.
  No candidate adoption, persistent-rest or4K promotion has occurred.
- **P311 status:** both strengths now compared in one fresh callback with one
  immutable baseline. All60 forward trials fail P306; the six data-restored
  candidates lower original merit but fail raw IoU alone. Original-merit CUDA
  closure passes all3original repeats, exact scalar recombination and callback
  isolation pass. No fixed-candidate repeat or adoption. See
  `docs/paired_braking_p311.md`. No quality promotion follows.
- **P312 status:** matched aggregate/silhouette repair is implemented and tested
  (34 CPU cases, independent code/result review). It shares one immutable
  original baseline and terminal05 origin/linearization/noise threshold.
  All56 forwards fail original raw gates;4 aggregate and5 treatment candidates
  restore their full prepared constraints. No accepted repair or candidate
  replay. Treatment h9/c2 fails overall coverage by net2/300000 despite improved
  silhouette/PBR/merit. Rejected candidate arrays are not saved; do not claim
  their exact lost target IDs from another forward. `docs/coverage_paths_p313.md`
  preregisters archive-only CUDA localization of the SAVED baseline triplet and
  actual common terminal05 origin. Net deficits are not lost-ID counts. W20
  rendering used18 views at64 pixels;96 is only the later configured C2F size.
- **P313 status:** saved original3 versus common terminal05 support localization
  closes on CUDA and independent direct FP64 distances. Four endpoint target
  losses and two gains give net2; original coverage bits agree across repeats.
  These are relative phase-dependent changes, not persistent visible holes:
  287243 differs at8 and20 only, and24591 misses a final entry. All six nearest
  suppliers are coarsely arrived-free, not certified at rest. See
  `docs/coverage_paths_p313.md` and its complete bounded audit. P314's separate
  opt-in support repair is implemented/reviewed (62 CPU cases); it keeps fixed material
  witnesses and records all-target losses/gains without changing coverage radii.
  No physical state or renderer is promoted.
- **P314 status:** the fixed-support CUDA comparison accepts0 repairs; all48
  valid forwards fail original raw IoU. Treatment h8/c1 has no lost target IDs
  and decreases terminal/step motion, but its protected witness is only8.48e-8wu
  inside the radius and no candidate repeats ran. Do not call it robust support
  or rest. Prepared silhouette improves while PBR worsens. Independent source,
  scalar, saved-control/remainder and endpoint audit passes. See
  `docs/support_repair_p314.md`. P315 localizes archived raw mask changes with
  unchanged metrics; W20 is still not a convergence or full-morph rest test.
- **P315 status:** archived raw silhouettes differ at9 view-pixels/8views,
  supplied by9 arrived-free IDs. Origin/control/support masks are identical.
  No new holes does NOT mean no holes:3 existing internal-hole pixels persist
  in view14 while the two-view160 metric is0. Fixed-size histogram counts/masks
  pass168 CUDA cloud/view comparisons and helper graph capture. Its1.896x warm
  helper speedup is not an end-to-end or host-free runtime claim. See
  `docs/silhouette_pixels_p315.md`; no physical candidate is promoted.
- **P316 preparation:** `docs/full_horizon_p316.md` defines a serial full-horizon
  baseline/raw stopping comparison with unchanged ordinary policies and30GB
  reserved per arm. The read-only trace owns pre-solve pins, full plans/arrival
  masks, raw endpoints and per-ID stored terminal speed-squared, adds no solve,
  and maps actual/promoted/delivery/held scopes.4 independent CPU cases pass,
  including real C2F/PIC/raw parity. A global stop or pin-imposed zero motion is
  not natural rest; no P314 candidate is adopted by this driver.
- **P316 baseline / P317 analysis:** frozen9cab095 baseline stops at43 attempts
  after3 outer rejections, with40 actual commits andone held row. Full-hash
  independent archive/pin/delivery audit passes; the global frozen flag is not
  natural rest. Raw comparison uses the same frozen code serially.
  `docs/horizon_motion_p317.md` defines bounded archive-only CUDA motion analysis:
  same-plan arrivals, actual/PIC displacement separation, fixed-ID follow-up
  denominators, pin timing, accepted-reference relabels and delivery scope.
  No physical policy or rest/hole/4K quality promotion follows from analysis.
- **P316/P317 completed / P318/P319:** full-horizon ordinary baseline/raw stop
  after40/28 accepted windows, by rejection/plateau rather than verified rest.
  Independent input/history and bounded last-path CUDA audits pass. Baseline's
  SAME final-free IDs have a raw last-step RMS.000375wu plus a PIC correction
  RMS.00856wu yielding a saved step RMS.00823wu; RMS magnitudes are not additive.
  Discretization:N300k,T20,dt1/240,dx.3062907544wu,loss36^3,iters8.
  The raw arm's distinct own-end cohort has no endpoint correction but still
  moves. See `docs/horizon_motion_p317.md`; no physical adoption.
  P318 batches render-report observations intoone host copy (53 CPU/4 actual
  CUDA cases pass); remaining raster/optimizer host decisions still exist.
  P319 (`docs/horizon_shape_p319.md`) observes all accepted archived phases and
  separate same-time raw xT,24-view128/256 masks and target coverage. It is not
  a3D watertightness/4K certificate or a renderer/physics policy change.
  Both full observers and bounded independent CUDA audits now pass (841/589
  labelled samples,13026/9246 checks,7 source states reprojected per arm).
  Intermediate projected holes remain; PIC improves same-endpoint coverage but
  can either add or remove projected holes. See retained P319 evidence and limits.
