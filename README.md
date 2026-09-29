# PhysMorph — render-guided differentiable MPM shape morphing

A particle body (a sphere by default) morphs into a target mesh under a differentiable
MLS-MPM simulation (NVIDIA Warp). The optimiser controls a per-particle stress field
window by window; the loss is a volume-transport term on the density grid plus a
differentiable rendering term (silhouette and shading of the outer particle layer) —
rendering controls the physics. The deliverable is a particle trajectory and a
photoreal video of its reconstructed surface.

## Install

```bash
conda env create -f environment/environments.yml   # torch, warp-lang, scipy, trimesh, ...
conda activate diffmpm_v2.3.0
pip install -e .
```

The render loss needs a CUDA GPU and the differentiable Gaussian rasteriser
(`diff_gaussian_rasterization` or `diff_gauss`), installed as a package.
The photoreal renderer uses EGL (`EGL_PLATFORM=surfaceless` on a headless host).

## Run

```bash
# sphere -> bunny, 40k particles, the current recipe
python scripts/pipeline_run.py --arms render_full_dt_iso_nn --tgt assets/bunny.obj --n 40000 \
  --cell_diag 26 --phys_loss auto --loss_units density --warm_start --w_kin 5 --w_kin_var 200 \
  --animations 300 --loss_res 64 --pace 0 --anneal 0.7 --mom_carry 0 --nn_far_k 1000 --bonds \
  --domain auto --layer_relax --pbr_denoised --layer_ctrl --sampler stratified --layer_gate_ot --out output/bunny

# the video of the reconstructed surface (tracked with the material)
python scripts/render_photoreal.py --npz output/bunny_render_full_dt_iso_nn.npz \
  --out output/bunny.mp4 --res 720 --stride 3 --surface poisson --track --track_keep --track_stretch 2 --track_every 0

# a single frame
python scripts/render_photoreal.py --npz output/bunny_render_full_dt_iso_nn.npz \
  --out output/bunny_f200.png --still 200 --surface poisson
```

`python scripts/pipeline_run.py --help` lists every flag; `--control_grid G` swaps the
per-particle control field for a coarse trilinear basis, `--lambda_auto 0` switches the
render channel off (the physics-only twin).

### Settled transport with local particle support

The opt-in method evaluates transport and rendering after a controlled phase of
`T` steps followed by `T` steps with `dFc` and `u` withdrawn. The physical simulation,
passive layer relaxation and material terms remain active during release. It uses
fixed-target grid transport, a matched shading target, rendering weights calibrated
once per loss resolution, and a released-phase motion penalty. A bounded local
density penalty discourages particle separation without overwhelming the remaining
transport objective. Delivery selects a valid state using the complete calibrated
merit; the local support penalty is not a topology or whole-trajectory guarantee.

Use the Run command above with a separate output directory and append:

```bash
--solver_mode settled_transport --ot_iters 1600 --support_weight 8 --nn_sampling_berth
```

`--nn_sampling_berth` calibrates the near-target cleanup radius from the target
particle spacing. Solver mode defaults to `legacy`; the local support penalty defaults
to off. The original PCGrad path remains available.

For paired 100k runs, use the same target, seed and base command on both sides
(`--n 100000 --seed 97` for the Bunny pair), then add the flags above only to the
new-method run. The reported pairs use Bunny / 97, Teapot / 101, Fandisk / 103,
Spot / 107 and Nefertiti / 109, with their corresponding meshes in `assets/`.

**Upstream compatibility:** this branch is based on `origin/main` at `680622e`.
That update changes dynamics mass to `40000/N`; the historical 100k comparisons
used unit masses. Pass `--mass_ref_n 0` to **both** arms when reproducing those
settings. The default `--mass_ref_n 40000` preserves the latest upstream behavior.
Previously measured quality gains are not validation of the new mass setting.
GPU neighbor queries and floating-point reductions can also affect exact replay.

## Layout

```
physmorph/mpm/        Warp MLS-MPM kernels, trajectory rollout and adjoint, outer-layer relaxation
physmorph/pipeline/   optimiser (windows, control basis, losses), runner, config
physmorph/render/     surface reconstruction of the outer layer (Poisson), render losses, photoreal renderer
physmorph/sampling/   mesh loading and volume sampling (stratified, reliable-axis fill)
physmorph/viewer/     live viewer and .ply export
scripts/              pipeline_run.py, render_photoreal.py, drivers; probes/ = measurement scripts
tests/                pytest suite (kernels, adjoints, layer relaxation, sampling, reconstruction)
assets/               source and target meshes
environment/          conda environment
```

## Tests

```bash
pytest -q
```
