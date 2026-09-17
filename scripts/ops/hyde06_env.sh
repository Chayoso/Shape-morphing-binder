#!/bin/bash
# Common environment for every hyde06-side ops script. Everything lives under /data (user rule
# 2026-09-16): the deployed repo, the outputs, the viewer packets. Nothing in $HOME.
export REPO=/data/relcfd/chayo/physmorph_v2/repo
export OUT=/data/relcfd/chayo/physmorph_v2/output
export PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
export STATUS=$OUT/rcp_ladder_status.log
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
# the production recipe (render_full_dt_iso_nn arm); N is passed by the caller
export RECIPE="--ppc 27 --loss_units density --warm_start --w_kin 5 --w_kin_var 200 --animations 300 --loss_res 64 --pace 0 --anneal 0.7 --mom_carry 0 --nn_far_k 1000 --bonds --domain auto"
cd $REPO
