# PhysMorph — render-guided differentiable MPM shape morphing

A particle body (a sphere by default) morphs into a target mesh inside a differentiable MLS-MPM simulation (NVIDIA
Warp). The optimiser controls a per-particle stress increment `dFc` and a normal offset `u` of the outer particle
layer, window by window. The losses are a transport term on the mass grid and a differentiable rendering term
(silhouette and shading of the particle body), so rendering steers the physics through the controls. The
deliverable is the particle trajectory and a 4K Gaussian-splat video of it.

The production method is **settled transport** (Michael Jin, 2026-09-29): each window drives the body for `T` steps,
then releases the controls for another `T` steps and scores the released state against the fixed target. The method
and its properties are in [docs/method.md](docs/method.md); results and their history in
[docs/experiments.md](docs/experiments.md).

## Install

```bash
conda env create -f environment/environments.yml   # torch, warp-lang, scipy, trimesh, ...
conda activate diffmpm_v2.3.0
pip install -e .
```

A CUDA GPU is required. The render loss and the video renderer use the differentiable Gaussian rasteriser
(`diff_gaussian_rasterization` or `diff_gauss`), installed as a package.

## Run

```bash
# sphere -> bunny, 300k particles, settled transport
python scripts/pipeline_run.py --arms render_full_dt_iso_nn --tgt assets/bunny.obj --n 300000 --seed 97 \
  --cell_diag 26 --phys_loss auto --loss_units density --warm_start --w_kin 5 --w_kin_var 200 \
  --animations 300 --loss_res 64 --pace 0 --anneal 0.7 --mom_carry 0 --nn_far_k 1000 --bonds \
  --domain auto --layer_relax --pbr_denoised --layer_ctrl --sampler stratified --layer_gate_ot \
  --solver_mode settled_transport --ot_iters 1600 --support_weight 8 --nn_sampling_berth \
  --out output/bunny

# the 4K video (Gaussian splats, ceramic material)
PYTHONPATH=. python scripts/render_splat_photoreal.py output/bunny_render_full_dt_iso_nn.npz output/bunny_4k.mp4 \
  --width 3840 --height 2160 --stride 12 --fps 20 --azimuth 35 --elevation 18 --frames-dir output/bunny_4k_frames
```

Useful switches: `--render_weight_scale 0` is the render-off twin (every other term unchanged); `--reject_stop` and
`--patience` set how long a converged run keeps trying; `--solver_mode legacy` runs the earlier windowed method;
`--mass_ref_n 0` reproduces unit particle masses. `scripts/render_splat_gpu.py` renders a quick two-view splat video.
`python scripts/pipeline_run.py --help` lists every flag.

## Results (sphere → bunny, 300k particles, seed 97)

| | settled transport | earlier line (v3-grid-gs) | release, legacy mode |
|---|---|---|---|
| silhouette IoU | **0.9851** | 0.9769 | 0.9646 |
| wall time / windows | **7 min** / 31 | 12 min / 47 | 28 min / 183 |
| minimum det F / stray particles | **0.934 / 0** | 0.608 / 30 | 0.581 / 40 |
| late surface motion (spacings per frame) | **0.007 / 0.009** | 0.128 / 0.113 | 0.049 / 0.080 |
| window-to-window surface reversals | 12 % | 0 % (pinned) | 70 % |
| centre-of-mass drift over the morph | **≤ 0.02 spacing** | 0.6–0.8 spacing | 0.03 spacing |
| target surface farther than 1.5 spacings | 11.7 % | **7.8 %** | 23.5 % |

Settled transport grows the ears from their base, comes to rest without pins and keeps momentum; the earlier line
still fits the head and body relief more closely. The 19-mesh gallery at 40k particles and the 300k dragon are the
adoption tests in progress (docs/experiments.md, S2).

## Working on this repository

- Heavy runs go to the GPU server (hyde06). `source scripts/ops/hyde06_env.sh` sets `REPO`, `OUT`, `PY`, `RECIPE` and
  `SETTLED`; deploy with `git archive` into `REPO`. All four GPUs may be used when free; never stop other users' jobs.
- Record every experiment in `docs/experiments.md` before reading its result (prediction and gates), then its result,
  including refutations. The log is append-only and uses the server clock.
- Fix mechanisms, not numbers: constants come from the discretisation or the physics, never tuned per shape. A change
  is adopted only when it holds on the 19-mesh gallery and at 300k.
- Every result states the render's influence: its share of the control update and the render-off twin against the
  seed-to-seed spread.
- Measurement probes live in `scripts/probes/settled/` (listed in docs/method.md, section 8).

## Layout

```
physmorph/mpm/        Warp MLS-MPM kernels, trajectory rollout and adjoint, outer-layer relaxation
physmorph/pipeline/   optimiser (windows, control basis, losses), runner, config
physmorph/losses/     transport (grid Sinkhorn), local support, volumetric terms
physmorph/render/     render losses, splat renderers and their support
physmorph/sampling/   mesh loading and volume sampling
scripts/              pipeline_run.py, renderers, ops/ (server environment), probes/
tests/                pytest suite
assets/               source and target meshes
```

## Tests

```bash
pytest -q        # on a CUDA machine
```
