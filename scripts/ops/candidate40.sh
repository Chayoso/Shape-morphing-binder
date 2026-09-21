#!/bin/bash
# One candidate of the last implementation (docs/final_plan.md §3) on the factorial's three targets:
# RECIPE + the candidate's flags on bunny / dragon / bob (same seed and particles as fx_*), three runs at
# a time on one GPU, then the factorial's readings for the new cell so it reads against fx_11 / fx_11c
# (the spread) and fx_10 (u off): end metrics, json telemetry, layer RMS, QA, end frame vs the true mesh.
# Usage: candidate40.sh <gpu> <cell> [flags...]   e.g. candidate40.sh 0 P3 --layer_F
#   -> runs fx_<cell>_<T>, markers "FX fx_<cell>_<T> DONE" in $STATUS, CAND_<cell>_DONE in the log
source /data/relcfd/chayo/physmorph_v2/repo/scripts/ops/hyde06_env.sh
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 EGL_PLATFORM=surfaceless MPLBACKEND=Agg
GPU=$1; CELL=$2; shift 2
ARM=render_full_dt_iso_nn
E=$OUT/surf_test/endframes; F=$OUT/fx_figs; mkdir -p $E $F
run() { local T=$1; shift
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/pipeline_run.py --arms $ARM --tgt assets/$T.obj --n 40000 $RECIPE "$@" --out $OUT/fx_${CELL}_$T > $OUT/fx_${CELL}_$T.log 2>&1
  echo "FX fx_${CELL}_$T DONE $(date) exit $?" >> $STATUS; }
run bunny "$@" & sleep 30; run dragon "$@" & sleep 30; run bob "$@" &
wait
echo "CAND_${CELL}_RUNS_DONE"
dn_of() { $PY -c "import numpy as np; z=np.load(\"$OUT/$1_$ARM.npz\"); print(int(z[\"deliver_n\"]))" 2>/dev/null | tail -1; }
endframe() { local R=$1; local dn=$(dn_of $R); $PY scripts/render_photoreal.py --npz $OUT/${R}_$ARM.npz --out $E/${R}_end.png --save_mesh $E/${R}_end.ply --still $((dn-1)) --res 720 --surface poisson --label "$R end" 2>&1 | grep -E "Traceback" | head -2; }
export CUDA_VISIBLE_DEVICES=$GPU
echo "## end metrics (log ARM line)"
for T in bunny dragon bob; do R=fx_${CELL}_$T; echo "$R: $(grep -E "ARM $ARM:" $OUT/$R.log | tail -1 | cut -c28-200)"; done
echo "## telemetry (json)"
$PY scripts/probes/fx_summary.py $OUT --cells=11,11c,10,$CELL bunny dragon bob 2>&1 | grep -v Warp
echo "## layer RMS (morph mean / end / target floor)"
for T in bunny dragon bob; do R=fx_${CELL}_$T; echo "$R: $($PY scripts/probes/layer_rms.py $OUT/${R}_$ARM.npz 40 --target 2>&1 | grep -E "mean over|target cloud" | tr '\n' ' ')"; done
echo "## QA (grid fragments / end fragments / re-attachments / stray census)"
for T in bunny dragon bob; do R=fx_${CELL}_$T
  frag=$(grep -o "anim [0-9]*: fragments [0-9]*" $OUT/$R.log | tail -1 | grep -o "[0-9]*$")
  reatt=$(grep -o "re-attached [0-9]*" $OUT/$R.log | awk '{s+=$2; n++} END{print s+0, n+0}')
  CUDA_VISIBLE_DEVICES="" $PY scripts/probes/grid_fragments.py $OUT $R 3 $F/${R}_gridfrag.txt > /dev/null 2>&1
  $PY scripts/probes/stray_census.py $OUT ${R}_$ARM 2>&1 | grep -v Warp > $F/${R}_census.txt
  echo "$R: gridfrag $(tail -1 $F/${R}_gridfrag.txt | cut -c1-90) | frag ${frag:-?} reatt $reatt | $(head -3 $F/${R}_census.txt | tail -1 | cut -c40-130)"; done
echo "## end frames vs the true mesh (surface_gt --gt_all)"
for T in bunny dragon bob; do
  case $T in bob) TP=$E/fx_11_bob_target.ply ;; *) TP=$OUT/surf_test/g5_$T/poisson_target.ply ;; esac
  R=fx_${CELL}_$T; endframe $R
  echo "$R: $($PY scripts/probes/surface_gt.py $OUT/$R.json $TP $E/${R}_end.ply --gt_all 2>&1 | grep -E "^[a-z]+/" | cut -c1-175 | tr '\n' ' ')"; done
echo "CAND_${CELL}_DONE"
