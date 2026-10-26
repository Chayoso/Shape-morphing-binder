#!/bin/bash
# d164_after.sh [N300] -- D164's CPU readings (no card; nice 10): the 40k exact-mesh readings of the D164 runs' last
# kept frames (tmp/d154/d154_mesh_read.py, D154_REF=own, the meshes in their runs' frames d154/meshes40, two readers,
# as D154 (3b) read the tag's V / W), then d164_read.py (the tables, G). N300=1: also the 300k part.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d164; W=$B/tmp/d164
PYB=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
mkdir -p $O/read $W/jobs40
S=$O/status.txt
log() { echo "$(date '+%m-%d %H:%M:%S') [after] $*" >> $S; }
rm -f $W/jobs40/j_*.txt $O/read/g40_*.jsonl
k=0
for f in $O/runs40/*40k_[LP]*_frames12.npz; do
  b=$(basename $f _frames12.npz); mesh=${b%%40k_*}; rest=${b#*40k_}; arm=${rest:0:1}; seed=${rest:1}
  echo "$mesh $seed $arm $f end" >> $W/jobs40/j_$((k % 2)).txt
  k=$((k + 1))
done
log "mesh-read jobs: $k states"
cd $B/tmp/d154
for j in 0 1; do
  [ -s $W/jobs40/j_$j.txt ] || continue
  CUDA_VISIBLE_DEVICES="" D154_REF=own D154_MESHES=$G/d154/meshes40 D154_THREADS=6 OMP_NUM_THREADS=6 TMPDIR=$W/tmpdir \
    nice -n 10 $PYB d154_mesh_read.py $O/read/g40_$j.jsonl $W/jobs40/j_$j.txt > $O/read/g40_$j.log 2>&1 < /dev/null &
done
wait
log "mesh reads done: $(cat $O/read/g40_*.jsonl 2>/dev/null | grep -c '^{') rows"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 nice -n 10 $PYB $W/d164_read.py $O/d164_tables.md $([ "${1:-0}" = 1 ] && echo --n300) \
  > $O/d164_read.log 2>&1 < /dev/null
log "d164_read exit $?: $(grep -c MISS $O/d164_tables.md) MISS cells; $(grep '^- G' $O/d164_tables.md | tr '\n' ' ' | cut -c1-300)"
