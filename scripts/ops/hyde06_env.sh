#!/bin/bash
# Common environment for every hyde06-side ops script. Everything lives under /data (user rule
# 2026-09-16): the deployed repo, the outputs, the viewer packets. Nothing in $HOME.
export REPO=/data/relcfd/chayo/physmorph_v2/repo
export OUT=/data/relcfd/chayo/physmorph_v2/output
export PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
export STATUS=$OUT/rcp_ladder_status.log
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
# the production recipe (render_full_dt_iso_nn arm); N is passed by the caller
# 2026-09-19 (docs/surface_gradient.md §6-7, accepted by the user): the outer-layer relaxation projection, the
# denoised shading reference and the position-mode control channel; the deliverable surface is Poisson
# (photoreal_batch.sh SURFACE). RECIPE_V8 is the v8 gallery's recipe for comparison runs.
export RECIPE_V8="--cell_diag 26 --phys_loss auto --loss_units density --warm_start --w_kin 5 --w_kin_var 200 --animations 300 --loss_res 64 --pace 0 --anneal 0.7 --mom_carry 0 --nn_far_k 1000 --bonds --domain auto"
# 2026-09-21 (§8): + stratified sampling of source and target (G5); the quad splat and the u-step projection falsified
# 2026-09-22 FROZEN (docs/final_plan.md §6, the decision page): the u channel is KEPT (silhouette +0.4..+1.9 on every target,
# the better bunny surface; bob/dragon lose 1.1-1.4 deg of normal error to it); the fill and the reconstruction fixes of
# method.md §10.13 are in the code, not in the flags. The 40k gallery g40 runs this recipe.
# 2026-09-22 20:30 (docs/surface_gradient.md §15c, method.md §10.16): + --layer_gate_ot — the u channel acts only where the
# remaining transport to the OT image is within one MPM cell; the mid-morph lumps −16..−21 %, the end within the spread. Gallery g41.
export RECIPE="$RECIPE_V8 --layer_relax --pbr_denoised --layer_ctrl --sampler stratified --layer_gate_ot"
cd $REPO
