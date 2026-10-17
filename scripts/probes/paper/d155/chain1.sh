#!/bin/bash
# D155 chain 1 (CPU only): the B7 window decomposition and the kept-frame readers
source /data/relcfd/chayo/physmorph_v2/tmp/d155/env.sh
cd $D
G=/data/relcfd/chayo/physmorph_v2/output/gpu
S=$O/status.txt
for m in bunny dragon; do
  ( nice -n 10 $PY d155_b7.py $m $O/b7_$m.json 8 > $O/b7_$m.log 2>&1 < /dev/null; echo "b7 $m rc $? $(date '+%m-%d %H:%M:%S')" >> $S ) &
done
wait
run_fr() { nice -n 10 $PY d155_frames.py $1 $O/fr_$2.json 8 > $O/fr_$2.log 2>&1 < /dev/null; echo "fr $2 rc $? $(date '+%m-%d %H:%M:%S')" >> $S; }
run_fr $G/d145t_b1/bunny300k_L97_frames12.npz bunny300k_L97 &
run_fr $G/d145t_b1/bunny300k_P97_frames12.npz bunny300k_P97 &
run_fr $G/d145t_b7/bunny300k_RG97_frames12.npz bunny300k_RG97 &
wait
run_fr $G/d145t_b1/dragon300k_L97_frames12.npz dragon300k_L97 &
run_fr $G/d145t_b1/dragon300k_P97_frames12.npz dragon300k_P97 &
run_fr $G/d145t_b7/dragon300k_RG97_frames12.npz dragon300k_RG97 &
wait
for m in teapot bob armadilo homer; do run_fr $G/d145t_b1/${m}300k_L97_frames12.npz ${m}300k_L97 & done
wait
echo "chain1 done $(date '+%m-%d %H:%M:%S')" >> $S
