#!/bin/bash
# d159_readcpu.sh MESH -- D159's CPU readings of one mesh once its runs exist (no card; nice 10):
#   B1's: apart particles over the first 35 % (d120_apart.py), the tail pulse (d116_pulse.py), the visible pieces over the
#   run every 4th kept frame (visible_apart.py, from repo_t as SHOWCASE runs it), the pieces at the end (pieces145.py) for
#   D159 M97 and for the tag's L97 (B1 has no pieces145 line for it);
#   R1 d159_relief.py: M97 (end, and t = 4 s), the tag's L97 (end, 4 s), L99-L105 (end; their seeds' frames), P97 (end),
#      P99 re-run (end), the own sample (tgt);
#   R2 d159_course.py: the twin P97's layer, A = M97 and the P99 re-run (the floor); the tag's L97 was read before
#      (read/course_tag_MESH.jsonl, same code);
#   R3 D157's reader on M97's end (jobs in read/d157_jobs_MESH.txt).
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d159; T=$B/tmp; W=$T/d159; R=$O/read
mesh=$1; M=${mesh}300k_M97; P99=${mesh}300k_P99
mkdir -p $R
S=$O/status.txt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $S; }
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
source $T/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
F=$O/${M}_frames12.npz
( source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null 2>&1; source $T/d146/cache_env.sh; export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
  nice -n 10 $PY $T/d120_apart.py $F 0.35 > $O/apart_$M.log 2>&1 < /dev/null
  nice -n 10 $PY $T/d116_pulse.py $F $O/$M.json 0.3 > $O/pulse_$M.log 2>&1 < /dev/null )
( export PHYSMORPH_RUN_REPO=$B/repo_t; source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1; cd $B/repo_t
  source $T/d146/cache_env.sh; export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
  nice -n 10 $PY scripts/probes/paper/visible_apart.py $F --label $M --w0 0 --w1 100000 --every 4 --jsonl $O/vis_full_$M.jsonl --threads 8 > $O/vis_$M.log 2>&1
  nice -n 10 $PY $T/d145r/pieces145.py $O/pieces.jsonl $M=$O/${M}_end1.npz ${mesh}300k_L97=$G/d145t_b1/${mesh}300k_L97_end1.npz \
    --every 1 --threads 8 > $O/pieces_$M.log 2>&1 ) < /dev/null
log "readcpu $mesh: B1's CPU readings done"
export PYTHONPATH=$B/repo_d159 OMP_NUM_THREADS=12
k4=80
nice -n 10 $PY $W/d159_relief.py $R/relief_$mesh.jsonl $mesh M97=$F M97_4s=$F@$k4 L97=$G/d145t_b1/${mesh}300k_L97_frames12.npz \
  L97_4s=$G/d145t_b1/${mesh}300k_L97_frames12.npz@$k4 P97=$G/d145t_b1/${mesh}300k_P97_end1.npz \
  L99=$G/d145t_b1/${mesh}300k_L99_end1.npz:99 L101=$G/d145t_b1/${mesh}300k_L101_end1.npz:101 \
  L103=$G/d145t_b1/${mesh}300k_L103_end1.npz:103 L105=$G/d145t_b1/${mesh}300k_L105_end1.npz:105 \
  P99re=$O/${P99}_frames12.npz:99 own=$G/d145t_b1/${mesh}300k_L97_end1.npz@tgt > $R/relief_$mesh.log 2>&1 < /dev/null
log "readcpu $mesh: R1 rc $?"
nice -n 10 $PY $W/d159_course.py $R/course_$mesh.jsonl $mesh $G/d145t_b1/${mesh}300k_P97_frames12.npz M97=$F \
  P99=$O/${P99}_frames12.npz:99 > $R/course_$mesh.log 2>&1 < /dev/null
log "readcpu $mesh: R2 rc $?"
( source $T/d157/env.sh; cd $T/d157
  echo "$mesh 97 300k M $O/${M}_end1.npz end" > $R/d157_jobs_$mesh.txt
  nice -n 10 $PY d157_read.py $R/d157_$mesh.jsonl $R/d157_jobs_$mesh.txt > $R/d157_$mesh.log 2>&1 < /dev/null )
log "readcpu $mesh: R3 (D157) rc $?"
