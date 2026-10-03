# PhysMorph — render-guided differentiable MPM shape morphing

A particle body (a sphere by default) morphs into a target mesh inside a differentiable MLS-MPM simulation (NVIDIA
Warp). The optimiser controls the body window by window through a per-particle stress increment `dFc` and a normal
offset `u` of the outer particle layer. The losses are a transport term on the mass grid and a differentiable
rendering term (silhouette and shading), so rendering steers the physics through the controls. The deliverable is the
particle trajectory and a 4K Gaussian-splat video of it.

This file is the pipeline summary. The other two documents are [docs/experiments.md](docs/experiments.md) (what to
run and what has been measured) and [docs/previous_pipeline.md](docs/previous_pipeline.md) (what went wrong in the
earlier pipelines and why this one replaced them).

## The pipeline

**Setup.** The source and target meshes are sampled into `N` particles each (stratified volume sampling). The MPM cell
is set from the shape: `dx` = source bounding-box diagonal / 26. Every particle has the same dynamics mass
`40000 / N`, so the body's mass does not depend on `N`.

**One window (settled transport).** The body is simulated for `2T` steps (`T = 20`, `dt = 1/240`): `T` driven steps
with the controls, then `T` released steps with the controls set to zero while the physics keeps running. Every loss
is evaluated on the released end state, so a control is scored by where the body comes to rest once it lets go.

**Controls.**
- `dFc`: an increment of each particle's deformation gradient. The stress is the fixed-corotated first
  Piola–Kirchhoff stress of `Fe = (F + dFc) Fp⁻¹`.
- `u`: a normal offset of each outer-layer particle, spread over the driven steps, and applied only where the
  remaining transport is within one cell.

The outer layer is also relaxed toward the plane of its neighbours every step, and bonds keep detached fragments
moving with their source neighbours. The layer is the body's: a set of particles that is not connected to the body
within one layer spacing is not relaxed (it has no surface of its own), and `u` moves its members along the
direction away from the material around them. Relaxed against itself such a set contracted into a floating clump
(`docs/experiments.md`, D20–D27).

**Objective.** Eight terms, all evaluated on the released end state. Which terms are needed was measured by
switching terms off on the 19-mesh gallery (`docs/experiments.md`, R9–R14b and FV). The geometry and settling terms
carry no weight of their own; the two local terms keep their original weight (0.2 in the legacy unit) and the
render term its calibration and internal constants.
- Geometry, on one scale measured once at the source:
  - Transport: a debiased Sinkhorn divergence between the body's mass on the loss grid and the fixed target's, blur
    one loss cell. The loss grid follows the particle count above 40k (`--loss_follows_n`). The Sinkhorn cost is
    separable by axis, so each sweep is three one-dimensional log-sum-exp passes on the GPU. The cross problem
    (body against target) is solved with alternating sweeps down an ε ladder, the self problem with the symmetric
    averaged sweep at the blur itself; every solve starts from zero duals (D29).
  - Surface proximity (`--support_form proximity`): at every outer target point, the kernel of the body's nearest
    particle against half the kernel at one sampling pitch. It charges a target point that has no particle within
    1.53 spacings; no weight, no bound.
  - Residual drift of the released end, `(T·dt)² mean |v_T|²`: the end is at rest.
- Settling: the released motion, `(T·dt)²` times the mean of `|v|²` over the released steps and particles. Without
  it runs stop while the body is still moving.
- Local:
  - Near band: a pull to the nearest target point for particles between the sampling berth (about two target
    spacings) and one loss cell from the target, where the transport's blur cannot tell positions apart. Early in a
    run it opposes the transport on those particles, more at larger `N`; the converged geometry is not degraded.
  - Spray cleanup: a pull down the target's distance field on the particles an isolation gate marks.
- Rendering: a multi-view silhouette loss (particles splatted with opacity `1 − e^{−kw}`) and a shading loss against a
  target rendered by the same operator.

There is no regulariser of the control or of the volume. The earlier code's control magnitude, control smoothness
and volume prior were removed: the first two were 1e-10 and 1e-8 of the merit, and removing all three changed
nothing that was measured (det F, anisotropy, control roughness, surface roughness; R14, R14b). The domain box is a
validity check of the rollout, not a term.

**Gradients.** The physics and render losses are differentiated separately through the same `2T`-step adjoint (Warp
tape, captured as CUDA graphs) back to `dFc` and `u`. The Sinkhorn term uses the envelope theorem: its gradient is
the difference of the converged potentials, passed through the rasterisation weights. The render gradient drops any
component that opposes the physics gradient and is weighted by `λ`, set once so that `λ·|g_render| = 0.5·|g_physics|`
and then held. A backtracking line search accepts a step only when the full objective decreases and the state stays
valid.

**Acceptance and delivery.** Each window's result is scored by one merit, the objective read at the committed state.
A result that raises the merit by more than 5 % is rejected and the state is kept. The run ends at the best state after 3 consecutive rejections, or when the
merit stops improving for 5 windows.

**What it conserves.** Stress cannot change total momentum: it enters the grid transfer as `G·(x_i − x_p)`, and the
B-spline weights satisfy `Σ w_ip (x_i − x_p) = 0`. The Kirchhoff stress is symmetric, so angular momentum is kept too.
Drag only decays momentum. The layer relaxation and `u` move outer-layer positions without a velocity: 11–22 % of that
layer's motion, 4 % of the particles. Measured over whole 300k morphs, the centre of mass moves at most 0.02 particle
spacing.

**Verification.** The test suite passes on the GPU server. The gradient path and the line-search path agree to
seven digits, and finite differences agree with autograd within 3 % for every loss term at windows 1 and 20 of a 300k
run. `scripts/probes/settled/gradcheck.py` repeats the check per channel (physics, render, cleanup) and control leaf
(`dFc`, `u`); the `u` channel is strongly nonlinear, so its check needs steps that change the loss by 1e-5 of itself.

**Rendering.** `scripts/render_splat_photoreal.py` renders the 4K deliverable: each particle is a disc-shaped
Gaussian whose normal comes from the smoothed density gradient and whose radius is the target spacing scaled by the
local 8th-neighbour distance (1–4×), shaded per pixel as a uniform ceramic material. `scripts/render_splat_gpu.py`
renders a quick two-view video with the same splats.

## Run (on the GPU server)

All experiments run on hyde06. The run stage is GPU-only: particle states stay on the device between windows, and
neighbour queries and grid morphology use CuPy (device KD-tree, `cupyx.scipy.ndimage`) through DLPack, with no CPU
fallback. The only CPU work is the prepare stage (mesh loading and volume sampling with trimesh), which is cached.
`scripts/ops/hyde06_env.sh` puts the isolated CuPy folder on `PYTHONPATH`.

```bash
ssh hyde06j
source /data/relcfd/chayo/physmorph_v2/repo_settled/scripts/ops/hyde06_env.sh   # REPO, OUT, PY, CuPy

# sphere -> bunny, 300k particles (the validated recipe; the two flags are not yet the defaults)
$PY scripts/pipeline_run.py --tgt assets/bunny.obj --n 300000 --seed 97 --support_form proximity --loss_follows_n \
    --out $OUT/bunny

# sphere -> dragon, 300k particles, 40 window attempts: about 11 minutes on one GPU (docs/experiments.md, D30)
$PY scripts/pipeline_run.py --tgt assets/dragon.obj --n 300000 --seed 97 --support_form proximity --loss_follows_n \
    --animations 40 --out $OUT/dragon

# the 4K video
PYTHONPATH=$REPO $PY scripts/render_splat_photoreal.py $OUT/bunny_render_full_dt_iso_nn.npz $OUT/bunny_4k.mp4 \
    --width 3840 --height 2160 --stride 12 --fps 20 --azimuth 35 --elevation 18 --frames-dir $OUT/bunny_4k_frames
```

`--render_weight_scale 0` is the render-off twin. `--reject_stop` and `--patience` set how long a converged run keeps
trying; `--ot_iters` is the Sinkhorn sweep budget. Deploy code with `git archive` into
`REPO`. All four GPUs may be used when free; never stop other users' jobs. Import `physmorph` before `torch` in new
scripts: the server's torch loads a CUDA 11 NVRTC that CuPy must not bind to (`physmorph/__init__.py`).

## Working rules

- Write each experiment's prediction and pass criteria in `docs/experiments.md` before reading its result, then the
  result. Use the server clock.
- Fix mechanisms, not numbers: constants come from the discretisation or the physics, never tuned per shape. A change
  is adopted only when it holds on the 19-mesh gallery and at 300k.
- Every result states the render's influence: its share of the control update and the render-off twin against the
  seed-to-seed spread.

## Layout

```
physmorph/gpu.py          device helpers of the run stage (CuPy KD-tree and ndimage through DLPack)
physmorph/prepare.py      the prepare stage: sampling (cached), the discretisation, the near-band berth
physmorph/pipeline/       config, target (grids, images, calibrations), render losses,
    window/               one window: setup, objective, rollouts, solve (Adam + line search), telemetry, layer
    run/                  the window loop: device state and archive, acceptance and stopping, runner
physmorph/losses/         grid Sinkhorn transport, local support, rasterisation and cleanup terms, silhouette
physmorph/mpm/            Warp MLS-MPM kernels, trajectory rollout and adjoint, F repair
physmorph/render/         splat renderers and their support, surface reconstruction, covariance
physmorph/sampling/       mesh loading and volume sampling
physmorph/viewer/         live and file-backed viewer
scripts/                  pipeline_run.py, renderers, ops/ (server environment), probes/settled/ (measurements)
tests/                    pytest suite (run on the GPU server)
assets/                   source and target meshes
```

Every source file is under 500 lines. The earlier pipelines' code, options and probes are at the tags
`v3-grid-gs-final` and `settled-base-prerefactor`.

Credits: settled transport by Michael Jin (2026-09-29), on the PhysMorph code release.
