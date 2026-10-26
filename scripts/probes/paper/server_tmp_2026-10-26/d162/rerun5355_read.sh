#!/bin/bash
# jobs 53/54/55 reads (CPU): D161 reader on rip500k (H+C vs T); D160 reader on band 300k and band 500k (D vs T).
# frames symlinked under bunny<N>k_* names so d160_read derives mesh=bunny (its ind_<mesh>.npz).
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; T1=$B/tmp/d161; A1=$G/d161/assets; F=$G/d162f; K=$G/d162k
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=12 D157_THREADS=12 MPLCONFIGDIR=$B/cache/matplotlib PYTHONPATH=$B/repo_d162d:$B/deps/d157_lpips
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
cd $B/repo_d162d
mkdir -p $K/after $F/after/lnk $K/after/lnk
echo "{\"scene\": \"rip500k\", \"N\": 500000, \"smooth\": \"$A1/d161smooth_bunny.obj\", \"relief\": \"$F/assets/d162fR300ripple_bunny.obj\", \"runs\": {\"full97\": \"$K/runs/rip500k_HC_97\", \"twin97\": \"$K/runs/rip500k_T_97\"}}" > $K/after/scene_rip500k.json
( nice -n 10 $PY $T1/d161_read.py $K/after/read_rip500k.json $K/after/scene_rip500k.json > $K/after/read_rip500k.log 2>&1; echo rc=$? >> $K/after/read_rip500k.log ) &
for N in 300 500; do
  if [ $N = 300 ]; then D=$F; else D=$K; fi
  for a in T D; do ln -sf $D/runs/band${N}k_${a}_97_frames12.npz $D/after/lnk/bunny${N}k_band_${a}_97_frames12.npz; done
  printf "band${N}k_T_97 band $D/after/lnk/bunny${N}k_band_T_97_frames12.npz\nband${N}k_D_97 band $D/after/lnk/bunny${N}k_band_D_97_frames12.npz twin=band${N}k_T_97 timeline\n" > $D/after/jobs_colour.txt
  ( nice -n 10 $PY $B/tmp/d160/d160_read.py $D/after/read_colour.jsonl $D/after/jobs_colour.txt > $D/after/read_colour.log 2>&1; echo rc=$? >> $D/after/read_colour.log ) &
done
wait
