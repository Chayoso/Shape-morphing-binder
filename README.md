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
  --out output/bunny.mp4 --res 720 --stride 3 --surface poisson --track

# a single frame
python scripts/render_photoreal.py --npz output/bunny_render_full_dt_iso_nn.npz \
  --out output/bunny_f200.png --still 200 --surface poisson
```

`python scripts/pipeline_run.py --help` lists every flag; `--control_grid G` swaps the
per-particle control field for a coarse trilinear basis, `--lambda_auto 0` switches the
render channel off (the physics-only twin).

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
