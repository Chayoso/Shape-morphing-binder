#!/bin/bash
# d160_after.sh -- D160's CPU readings after the chain's runs (agent D160; no card; nice 10, at most three processes):
# pieces (tmp/d145r/pieces145.py) on every run and on the tag's bunny / dragon V / W 97 / 99 / 101 ends; D155's gate
# reader on every run; D157's reader (tmp/d157/d157_read.py) on the L runs' ends with each mesh-seed's own sample as its
# context; then d160_gate.py. Waits for the chain's "R done" (and for its "C colour readings" before the gate).
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d160; W=$B/tmp/d160; T=$B/tmp
source $T/d146/cache_env.sh
export TMPDIR=$W/tmpdir CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
S=$O/status.txt
log() { echo "$(date '+%m-%d %H:%M:%S') [after] $*" >> $S; }
until grep -q "R done" $S 2>/dev/null; do sleep 120; done
log "start"
cd $W
R=$(ls $O/runs40/*_frames12.npz | sed 's#.*/##; s#_frames12.npz##')
TAGS=""; for m in bunny dragon; do for a in V W; do for s in 97 99 101; do
  TAGS="$TAGS ${m}40k_$a$s=$G/d155/gate/${m}40k_$a${s}_frames12.npz"; done; done; done
( export PYTHONPATH=$B/repo_d160
  nice -n 10 $PY $T/d145r/pieces145.py $O/pieces.jsonl $(for n in $R; do echo "$n=$O/runs40/${n}_frames12.npz"; done) \
      --every 4 --threads 6 > $O/pieces.log 2>&1
  nice -n 10 $PY $T/d145r/pieces145.py $O/pieces.jsonl $TAGS --every 100000 --threads 6 >> $O/pieces.log 2>&1 ) &
( nice -n 10 $PY $T/d155/d155_gate_read.py $O/gate_d160.jsonl $(for n in $R; do echo "$n=$O/runs40"; done) \
      > $O/gate_d160.log 2>&1 ) &
J=$O/jobs_d157.txt; : > $J
for s in 97 99; do
  for m in bunny dragon; do
    first=1
    for sc in half band ears turn; do
      F=$O/runs40/${m}40k_${sc}_L${s}_frames12.npz
      [ -e $F ] || continue
      [ $first = 1 ] && { echo "$m $s 40k own $F tgt" >> $J; first=0; }
      echo "$m $s 40k ${sc}_L $F end" >> $J
    done
  done
done
( source $T/d157/env.sh; export TMPDIR=$W/tmpdir; cd $T/d157
  nice -n 10 $PY $T/d157/d157_read.py $O/d157_L.jsonl $J > $O/d157_L.log 2>&1 ) &
wait
log "pieces, gate reader, D157 reader done ($(wc -l < $O/pieces.jsonl) / $(wc -l < $O/gate_d160.jsonl) / $(wc -l < $O/d157_L.jsonl) rows)"
until grep -q "C colour readings" $S 2>/dev/null; do sleep 120; done
( export PYTHONPATH=$B/repo_d160; $PY $W/d160_gate.py $O/gate > $O/gate.log 2>&1 )
log "gate read: $(tail -4 $O/gate.log | head -3 | tr '\n' ' ')"
