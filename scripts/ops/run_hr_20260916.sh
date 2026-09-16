#!/bin/bash
# high-resolution gallery (2026-09-16, code 0ccc43a): sphere -> 10 targets at 40k --ppc 8 in
# density units with the kinetic recipe; each with its physics-only twin (--lambda_auto 0,
# same code path). Every run writes live packets for the 3D viewer (--live_dir).
cd ~/physmorph_v2
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
OUT=/data/relcfd/chayo/physmorph_v2/output
S=$OUT/rcp_ladder_status.log
COMMON="--n 40000 --ppc 8 --loss_units density --warm_start --w_kin 5 --w_kin_var 200 --animations 300 --loss_res 64 --pace 0 --anneal 0.7 --mom_carry 0 --nn_far_k 1000 --live_dir $OUT/live"
GPU=$1; B=$2
if [ "$B" = "h1" ]; then TARGETS="bunny armadilo dragon spot bob"; else TARGETS="teapot heart A C V"; fi
for T in $TARGETS; do
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/pipeline_run.py --arms render_full_dt_iso_nn --tgt assets/$T.obj $COMMON --out $OUT/hr_$T > $OUT/hr_$T.log 2>&1
  echo "HR $T render DONE $(date)" >> $S
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/pipeline_run.py --arms render_full_dt_iso_nn --lambda_auto 0 --tgt assets/$T.obj $COMMON --out $OUT/hr_${T}_phys > $OUT/hr_${T}_phys.log 2>&1
  echo "HR $T phys DONE $(date)" >> $S
done
echo "BATCH $B DONE $(date)" >> $S
