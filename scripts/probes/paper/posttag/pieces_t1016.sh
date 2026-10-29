#!/bin/bash
# pieces_t1016.sh -- D145 post-tag: pieces145.py (CPU, nice 10, 8 threads) on the end frames of the tag's B1, amendment
# 6's 300k arms and the pre-tag B1 (pieces_DIR.jsonl), then on every kept-frames archive still on disk in those folders
# (every 4th kept frame and the last; pieces_full_DIR.jsonl); with the argument "full" only the second part. Started
# detached; "done" in d145t_evidence/pieces.status. DIRS="..." limits the folders.
source /data/relcfd/chayo/physmorph_v2/tmp/d145/env_d145.sh
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; E=$G/d145t_evidence; mkdir -p $E
source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8
P="nice -n 10 $PY $B/tmp/d145r/pieces145.py"
[ "$1" = full ] || {
for d in ${DIRS:-d145t_b1 d145t_a6 d145_b1}; do
  specs=""; for f in $G/$d/*300k_*_end1.npz; do specs="$specs $(basename $f _end1.npz)=$f"; done
  rm -f $E/pieces_$d.jsonl; $P $E/pieces_$d.jsonl $specs --every 1 > $E/pieces_$d.log 2>&1
  echo "end $d $(date '+%m-%d %H:%M')" >> $E/pieces.status
done
}
for d in ${DIRS:-d145t_b1 d145t_a6 d145_b1}; do
  rm -f $E/pieces_full_$d.jsonl
  for f in $G/$d/*300k_*_frames12.npz; do
    [ -e $f ] || continue
    $P $E/pieces_full_$d.jsonl $(basename $f _frames12.npz)=$f --every 4 >> $E/pieces_full_$d.log 2>&1
  done
  echo "full $d $(date '+%m-%d %H:%M')" >> $E/pieces.status
done
echo "done $(date '+%m-%d %H:%M')" >> $E/pieces.status
