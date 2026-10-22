#!/bin/bash
# d159_g40read.sh -- D159: the 40k gallery's M97 end states read against the exact mesh as D154 (3b) read the tag's V / W
# (d154_mesh_read.py with D154_REF=own, the meshes in the seed-97 frames d154/meshes40; CPU, two readers, nice 10).
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d159; W=$B/tmp/d154; R=$O/read; PYB=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
mkdir -p $R $O/g40/jobs
source $B/tmp/d146/cache_env.sh
rm -f $O/g40/jobs/j_*.txt
k=0
for m in A armadilo beast bimba bob bunny C cheburashka cow dragon fandisk heart homer maxplanck nefertiti ogre spot teapot V; do
  F=$O/g40/${m}40k_M97_frames12.npz
  [ -e $F ] || { echo "$(date '+%m-%d %H:%M:%S') g40 read: no $F" >> $O/status.txt; continue; }
  echo "$m 97 M $F end" >> $O/g40/jobs/j_$((k % 2)).txt
  k=$((k + 1))
done
cd $W
for j in 0 1; do
  CUDA_VISIBLE_DEVICES="" D154_REF=own D154_MESHES=$G/d154/meshes40 D154_THREADS=6 OMP_NUM_THREADS=6 TMPDIR=$B/tmp/d159/tmpdir \
    nice -n 10 $PYB d154_mesh_read.py $R/g40_m97_$j.jsonl $O/g40/jobs/j_$j.txt > $R/g40_m97_$j.log 2>&1 < /dev/null &
done
wait
cat $R/g40_m97_0.jsonl $R/g40_m97_1.jsonl 2>/dev/null | grep '^{' > $R/g40_m97.jsonl
echo "$(date '+%m-%d %H:%M:%S') g40 read done: $(wc -l < $R/g40_m97.jsonl) states" >> $O/status.txt
