#!/bin/bash
# d162_auto29b.sh -- job 29b's auto-reader (CPU, detached): D161's reader on the coin lettering, D160's on the band, then
# output/gpu/d162/auto/29b/summary.md (d162_auto.py minimal: the gallery against the tag's V / W).
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; W=$B/tmp/d162; T1=$B/tmp/d161; A1=$G/d161/assets; O=$G/d162z; R=$O/runs
mkdir -p $O/after $G/d162/auto/29b
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=6 D157_THREADS=6 PYTHONPATH=$B/repo_d162d:$B/deps/d157_lpips
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
until grep -q "R done (d162z)" $O/status.txt 2>/dev/null; do sleep 120; done
cd $B/repo_d162d
echo "{\"scene\": \"coin40k_M\", \"N\": 40000, \"smooth\": \"$A1/d161coin.obj\", \"relief\": \"$A1/d161L40coin.obj\", \"runs\": {\"full97\": \"$R/coin40k_M_97\", \"twin97\": \"$G/d161/runs40/coin40k_twin_97\"}}" > $O/after/scene_coin.json
nice -n 10 $PY $T1/d161_read.py $O/after/read_coin.json $O/after/scene_coin.json > $O/after/read_coin.log 2>&1 &
printf "bunny40k_band_P97 band $G/d160/runs40/bunny40k_band_P97_frames12.npz\nbunny40k_band_M_97 band $R/bunny40k_band_M_97_frames12.npz twin=bunny40k_band_P97\n" > $O/after/jobs_colour.txt
nice -n 10 $PY $B/tmp/d160/d160_read.py $O/after/read_colour.jsonl $O/after/jobs_colour.txt > $O/after/read_colour.log 2>&1 &
wait
$PY $W/d162_auto.py minimal $G/d162/auto/29b/summary.md > $O/after/auto.log 2>&1
