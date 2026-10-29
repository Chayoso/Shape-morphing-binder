#!/bin/bash
# d159c_read.sh MESH -- D159c's CPU readings (no card): B1's apart / pulse / visible pieces / end pieces; R1 (end and 4 s), R2
# (against the twin P97; the floor and D159 / tag already read), R5 (term dumps, then deleted and logged), R6 (window).
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d159; T=$B/tmp; W=$T/d159; R=$O/read
# RUN, not C: tmp/d146/cache_env.sh sets C (the cache dir) and clobbered it on the first launch (2026-10-22)
mesh=$1; RUN=${mesh}300k_C97; F=$O/${RUN}_frames12.npz
S=$O/status.txt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $S; }
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
source $T/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
( source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null 2>&1; source $T/d146/cache_env.sh; export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
  nice -n 10 $PY $T/d120_apart.py $F 0.35 > $O/apart_$RUN.log 2>&1 < /dev/null
  nice -n 10 $PY $T/d116_pulse.py $F $O/$RUN.json 0.3 > $O/pulse_$RUN.log 2>&1 < /dev/null )
( export PHYSMORPH_RUN_REPO=$B/repo_t; source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1; cd $B/repo_t
  source $T/d146/cache_env.sh; export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
  nice -n 10 $PY scripts/probes/paper/visible_apart.py $F --label $RUN --w0 0 --w1 100000 --every 4 --jsonl $O/vis_full_$RUN.jsonl --threads 8 > $O/vis_$RUN.log 2>&1
  nice -n 10 $PY $T/d145r/pieces145.py $O/pieces.jsonl $RUN=$O/${RUN}_end1.npz --every 1 --threads 8 > $O/pieces_$RUN.log 2>&1 ) < /dev/null
export PYTHONPATH=$B/repo_d159 OMP_NUM_THREADS=12 D159_REPO=$B/repo_d159
nice -n 10 $PY $W/d159_relief.py $R/relief_c_$mesh.jsonl $mesh C97=$F C97_4s=$F@80 > $R/relief_c_$mesh.log 2>&1 < /dev/null
nice -n 10 $PY $W/d159_course.py $R/course_c_$mesh.jsonl $mesh $G/d145t_b1/${mesh}300k_P97_frames12.npz C97=$F > $R/course_c_$mesh.log 2>&1 < /dev/null
TD=$O/terms/$RUN
nice -n 10 $PY $W/d159_terms.py $R/terms_$mesh.jsonl $mesh C97 $TD > $R/terms_c_$mesh.log 2>&1 < /dev/null
if [ $? = 0 ] && [ -d $TD ]; then
  sz=$(du -sm $TD | cut -f1); n=$(ls $TD | wc -l); rm -rf $TD
  echo "$(date '+%Y-%m-%d %H:%M') D159c: deleted the term dumps of $RUN ($n files, ${sz} MB, output/gpu/d159/terms), read into read/terms_$mesh.jsonl" >> $G/cleanup.log
fi
$PY $W/d159_window.py $R/window_c.jsonl ${mesh}_D159c_C97=$O/$RUN.json > $R/window_c_$mesh.log 2>&1
log "D159c readcpu $mesh done"
