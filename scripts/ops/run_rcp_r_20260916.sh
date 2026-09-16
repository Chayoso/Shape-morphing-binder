#!/bin/bash
# batch r (2026-09-16, code 0ccc43a): the deliverable arms re-run on the FIXED target
# (orthographic voxel fill, no axis-fill streaks). Same recipes as batches a/n/j.
cd ~/physmorph_v2
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
OUT=/data/relcfd/chayo/physmorph_v2/output
COMMON="--animations 300 --pace 0 --anneal 0.7 --mom_carry 0 --nn_far_k 1000 --live_dir $OUT/live"
GPU=$1; B=$2
if [ "$B" = "r1" ]; then
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/pipeline_run.py --arms render_full_dt_iso_nn --n 20000 --loss_res 64 $COMMON --out $OUT/rcpr_20k_a > $OUT/rcpr_20k_a.log 2>&1
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/pipeline_run.py --arms render_ctrl --control_grid 36 --n 20000 --loss_res 64 $COMMON --out $OUT/rcpr_20k_ctrl36 > $OUT/rcpr_20k_ctrl36.log 2>&1
elif [ "$B" = "r2" ]; then
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/pipeline_run.py --arms render_full_dt_iso_nn --n 40000 --ppc 8 --loss_units density --warm_start --w_kin 5 --w_kin_var 200 --loss_res 64 $COMMON --out $OUT/rcpr_40k_ppc8_recipe > $OUT/rcpr_40k_ppc8_recipe.log 2>&1
fi
echo "BATCH $B DONE $(date)" >> $OUT/rcp_ladder_status.log
