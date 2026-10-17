#!/bin/bash
# b13_cpu1017.sh -- B13's 300k reading, 2026-10-17 (agent B13r, CPU only, nice 10, no card). The readers the chain did
# not run, then the tables:
#   (1) pieces145.py (D145 post-tag) on the 36 end frames (ev1017/pieces_b13.jsonl) and on the 12 seed-97 kept-frames
#       archives, every 4th kept frame and the last (ev1017/pieces_full_b13.jsonl), as pieces_t1016.sh read B1;
#   (2) visible_apart.py (D148) on the 12 seed-97 archives, windows 3-10 and the whole run every 4th kept frame, into
#       b13/vis_w3_10_NAME.jsonl and b13/vis_full_NAME.jsonl, as read300_cpu.sh does for d145t_* (its visible-pieces
#       step reads d145t_* only, so the B13 runs have none; the 99 / 101 archives are gone as pre-registered);
#   (3) b13_cpu_reads.sh (the volume routine on seed 97: L97 / F97 / FM97; rot145 / angL145 on every run);
#   (4) tab145.py rows of the 36 runs, tab145t (the b13 copy: E = b13/ev1017), B1's L rows at seeds 97 / 99 / 101 from
#       d145t_evidence/tab_tx.jsonl beside them (B1's own reading, unchanged), band145t F vs L, FM vs L, FM vs F at
#       seeds 97 / 99 / 101, b13_read300.py (the pre-registered items and the force records), b13_md1017.py (tables).
# Status lines in b13/ev1017/status.txt. Re-runnable: STEP=tables runs (4) only.
source /data/relcfd/chayo/physmorph_v2/tmp/d146/cache_env.sh
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; D=$G/b13; E=$D/ev1017; T=$B/tmp; mkdir -p $E
export PHYSMORPH_OUT=$D
source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8
S=$E/status.txt
M="bunny dragon teapot bob armadilo homer"
N="nice -n 10 $PY"
echo "start ${STEP:-all} pid $$ $(date '+%m-%d %H:%M:%S')" >> $S
if [ "${STEP:-all}" = all ]; then
  # (1a) end frames, one stream
  ( specs=""; for m in $M; do for s in 97 99 101; do for a in F FM; do n=${m}300k_$a$s
      specs="$specs $n=$D/${n}_end1.npz"; done; done; done
    rm -f $E/pieces_b13.jsonl; $N $T/d145r/pieces145.py $E/pieces_b13.jsonl $specs --every 1 > $E/pieces_b13.log 2>&1
    echo "pieces end rc $? $(date '+%m-%d %H:%M:%S')" >> $S ) &
  # (1b) the seed-97 archives, three streams of four
  i=0
  for grp in "bunny dragon" "teapot bob" "armadilo homer"; do
    i=$((i + 1))
    ( rm -f $E/pieces_full_b13_$i.jsonl
      for m in $grp; do for a in F FM; do n=${m}300k_${a}97
        $N $T/d145r/pieces145.py $E/pieces_full_b13_$i.jsonl $n=$D/${n}_frames12.npz --every 4 >> $E/pieces_full_b13_$i.log 2>&1
        echo "pieces full $n rc $? $(date '+%m-%d %H:%M:%S')" >> $S; done; done ) &
  done
  # (2) visible pieces, one stream (repo_t, as read300_cpu.sh)
  ( export PHYSMORPH_RUN_REPO=$B/repo_t; source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1; cd $B/repo_t
    for m in $M; do for a in F FM; do n=${m}300k_${a}97; F=$D/${n}_frames12.npz
      rm -f $D/vis_w3_10_$n.jsonl $D/vis_full_$n.jsonl
      $N scripts/probes/paper/visible_apart.py $F --label $n --jsonl $D/vis_w3_10_$n.jsonl --threads 8 > $D/vis_$n.log 2>&1
      $N scripts/probes/paper/visible_apart.py $F --label $n --w0 0 --w1 100000 --every 4 --jsonl $D/vis_full_$n.jsonl \
        --threads 8 >> $D/vis_$n.log 2>&1
      echo "vis $n rc $? $(date '+%m-%d %H:%M:%S')" >> $S; done; done ) < /dev/null &
  # (3) the volume routine and the rotation records
  ( bash $T/b13/b13_cpu_reads.sh > $E/cpu_reads.log 2>&1; echo "cpu_reads rc $? $(date '+%m-%d %H:%M:%S')" >> $S ) &
  wait
  cat $E/pieces_full_b13_1.jsonl $E/pieces_full_b13_2.jsonl $E/pieces_full_b13_3.jsonl > $E/pieces_full_b13.jsonl
  echo "readers done $(date '+%m-%d %H:%M:%S')" >> $S
fi
# (4) the tables
cd $B
cp $D/endgen_b13.jsonl $E/endgen_b13.jsonl
specs=""; for m in $M; do for s in 97 99 101; do for a in F FM; do specs="$specs b13/${m}300k_$a$s"; done; done; done
rm -f $E/tab_b13.jsonl
$N $T/d145/tab145.py $E/tab_b13.jsonl $specs > $E/tab_b13.log 2>&1
$N $T/b13/tab145t_b13.py $E/tab_b13.jsonl $E/tab_b13x.jsonl > $E/tab_b13x.log 2>&1
grep -E '"name": "[a-z]+300k_L(97|99|101)"' $G/d145t_evidence/tab_tx.jsonl > $E/tab_Lx.jsonl
cat $E/tab_b13x.jsonl $E/tab_Lx.jsonl > $E/tab_all.jsonl
for p in "F L" "FM L" "FM F"; do
  set -- $p
  { echo "=== B13 on the tag: $1 vs $2, seeds 97 / 99 / 101"; $N $T/d145r/band145t.py $E/tab_all.jsonl $1 $2 $M --seeds 97,99,101; } \
    > $E/band_$1_vs_$2.txt 2>&1
done
$N $T/b13/b13_read300.py $E/b13_rows.jsonl > $E/b13_read300.txt 2>&1
echo "b13_read300 rc $? $(date '+%m-%d %H:%M:%S')" >> $S
$N $T/b13/b13_md1017.py $E > $E/tables_b13.md 2> $E/tables_b13.err
echo "tables rc $? $(date '+%m-%d %H:%M:%S')" >> $S
echo "done $(date '+%m-%d %H:%M:%S')" >> $S
