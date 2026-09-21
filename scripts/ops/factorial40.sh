#!/bin/bash
# The render x u factorial under the current recipe at 40k (docs/surface_gradient.md §10, pre-registered).
# Per target six runs, same seed and particles (stratified, seed 1):
#   fx_11  RECIPE (render on, u on)          fx_11c RECIPE again (identical configuration: the run-to-run spread)
#   fx_01  RECIPE --lambda_auto 0 (render off, u on)
#   fx_10  RECIPE without --layer_ctrl (render on, u off)
#   fx_00  RECIPE without --layer_ctrl, --lambda_auto 0 (render off, u off)
#   fx_cut RECIPE --render_until K (render switched off from window K = a third of the run: 20 / 20 / 8)
# then the readings: end metrics, json telemetry (g_share, lambda, cosine, D_vol), layer RMS, QA columns,
# end frames vs the true mesh (Poisson, surface_gt --gt_all), divergence from the render-on twin with the control.
# Markers: "FX <run> DONE" in $STATUS; FX_RUNS_DONE / FX_DONE in the log.
# Usage: factorial40.sh <gpuA> <gpuB>   (bunny then bob on A, dragon on B; three runs at a time per GPU)
source /data/relcfd/chayo/physmorph_v2/repo/scripts/ops/hyde06_env.sh
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 EGL_PLATFORM=surfaceless MPLBACKEND=Agg
GA=${1:-0}; GB=${2:-2}
ARM=render_full_dt_iso_nn
RECIPE_U0="$RECIPE_V8 --layer_relax --pbr_denoised --sampler stratified"
declare -A K=( [bunny]=20 [dragon]=20 [bob]=8 )
E=$OUT/surf_test/endframes; F=$OUT/fx_figs; mkdir -p $E $F
CELLS="11 11c 01 10 00 cut"

run() { local G=$1 NAME=$2 T=$3; shift 3
  CUDA_VISIBLE_DEVICES=$G $PY scripts/pipeline_run.py --arms $ARM --tgt assets/$T.obj --n 40000 "$@" --out $OUT/${NAME}_$T > $OUT/${NAME}_$T.log 2>&1
  echo "FX ${NAME}_$T DONE $(date)" >> $STATUS; }
cells() { local G=$1 T=$2
  run $G fx_11 $T $RECIPE & sleep 30
  run $G fx_11c $T $RECIPE & sleep 30
  run $G fx_01 $T $RECIPE --lambda_auto 0 &
  wait
  run $G fx_10 $T $RECIPE_U0 & sleep 30
  run $G fx_00 $T $RECIPE_U0 --lambda_auto 0 & sleep 30
  run $G fx_cut $T $RECIPE --render_until ${K[$T]} &
  wait; }

( cells $GA bunny; cells $GA bob ) &
( cells $GB dragon ) &
wait
echo FX_RUNS_DONE

dn_of() { $PY -c "import numpy as np; z=np.load(\"$OUT/$1_$ARM.npz\"); print(int(z[\"deliver_n\"]))" 2>/dev/null | tail -1; }
endframe() { local R=$1; local dn=$(dn_of $R); $PY scripts/render_photoreal.py --npz $OUT/${R}_$ARM.npz --out $E/${R}_end.png --save_mesh $E/${R}_end.ply --still $((dn-1)) --res 720 --surface poisson --label "$R end" 2>&1 | grep -E "Traceback" | head -2; }
target_ply() { local R=$1; $PY scripts/render_photoreal.py --npz $OUT/${R}_$ARM.npz --out $E/${R}_target.png --save_mesh $E/${R}_target.ply --still -2 --res 720 --surface poisson --label "$R target" 2>&1 | grep -E "Traceback" | head -2; }
export CUDA_VISIBLE_DEVICES=$GA

echo "## end metrics (log ARM line)"
for T in bunny dragon bob; do for C in $CELLS; do R=fx_${C}_$T; echo "$R: $(grep -E "ARM $ARM:" $OUT/$R.log | tail -1 | cut -c28-200)"; done; done
echo "## telemetry (json): windows / accepted / frames / minutes / g_share 1-20 and all / lambda / cosine / D_vol first-end / end metrics"
$PY scripts/probes/fx_summary.py $OUT bunny dragon bob 2>&1 | grep -v Warp
echo "## layer RMS (morph mean / end / target floor)"
for T in bunny dragon bob; do for C in $CELLS; do R=fx_${C}_$T; echo "$R: $($PY scripts/probes/layer_rms.py $OUT/${R}_$ARM.npz 40 --target 2>&1 | grep -E "mean over|target cloud" | tr '\n' ' ')"; done; done
echo "## QA (grid fragments / end fragments / re-attachments / stray census)"
for T in bunny dragon bob; do for C in $CELLS; do R=fx_${C}_$T
  frag=$(grep -o "anim [0-9]*: fragments [0-9]*" $OUT/$R.log | tail -1 | grep -o "[0-9]*$")
  reatt=$(grep -o "re-attached [0-9]*" $OUT/$R.log | awk '{s+=$2; n++} END{print s+0, n+0}')
  CUDA_VISIBLE_DEVICES="" $PY scripts/probes/grid_fragments.py $OUT $R 3 $F/${R}_gridfrag.txt > /dev/null 2>&1
  $PY scripts/probes/stray_census.py $OUT ${R}_$ARM 2>&1 | grep -v Warp > $F/${R}_census.txt
  echo "$R: gridfrag $(tail -1 $F/${R}_gridfrag.txt | cut -c1-90) | frag ${frag:-?} reatt $reatt | $(head -3 $F/${R}_census.txt | tail -1 | cut -c40-130)"; done; done
echo "## end frames vs the true mesh (surface_gt --gt_all; the target's Poisson surface first)"
target_ply fx_11_bob
for T in bunny dragon bob; do
  case $T in bob) TP=$E/fx_11_bob_target.ply ;; *) TP=$OUT/surf_test/g5_$T/poisson_target.ply ;; esac
  for C in $CELLS; do R=fx_${C}_$T; endframe $R
    echo "$R: $($PY scripts/probes/surface_gt.py $OUT/$R.json $TP $E/${R}_end.ply --gt_all 2>&1 | grep -E "^[a-z]+/" | cut -c1-175 | tr '\n' ' ')"; done; done
echo "## divergence from the render-on twin (render_effect.py; control = the identical re-run)"
for T in bunny dragon bob; do
  $PY scripts/probes/render_effect.py $OUT $T fx_11 fx_01 fx_cut ${K[$T]} --ctrl fx_11c --png $F/render_effect_$T.png 2>&1 | grep -v Warp
  echo "-- u off (fx_10 vs fx_00):"; $PY scripts/probes/render_effect.py $OUT $T fx_10 fx_00 fx_cut ${K[$T]} 2>&1 | grep -v Warp | grep -E "phys twin|fx_10|fx_00"
done
echo FX_DONE
