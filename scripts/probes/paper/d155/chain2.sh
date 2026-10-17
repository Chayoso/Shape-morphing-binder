#!/bin/bash
# D155 chain 2 (CPU only): the kept-frame readers again with the normal part still to come (lag_n)
source /data/relcfd/chayo/physmorph_v2/tmp/d155/env.sh
cd $D
G=/data/relcfd/chayo/physmorph_v2/output/gpu
S=$O/status.txt
run_fr() { nice -n 10 $PY d155_frames.py $1 $O/fr_$2.json 8 > $O/fr_$2.log 2>&1 < /dev/null; echo "fr2 $2 rc $? $(date '+%m-%d %H:%M:%S')" >> $S; }
until grep -q "chain1 done" $S 2>/dev/null; do sleep 20; done
run_fr $G/d145t_b1/bunny300k_L97_frames12.npz bunny300k_L97 &
run_fr $G/d145t_b1/bunny300k_P97_frames12.npz bunny300k_P97 &
run_fr $G/d145t_b7/bunny300k_RG97_frames12.npz bunny300k_RG97 &
run_fr $G/d145t_b1/dragon300k_L97_frames12.npz dragon300k_L97 &
wait
run_fr $G/d145t_b1/dragon300k_P97_frames12.npz dragon300k_P97 &
run_fr $G/d145t_b7/dragon300k_RG97_frames12.npz dragon300k_RG97 &
for m in teapot bob; do run_fr $G/d145t_b1/${m}300k_L97_frames12.npz ${m}300k_L97 & done
wait
for m in armadilo homer; do run_fr $G/d145t_b1/${m}300k_L97_frames12.npz ${m}300k_L97 & done
wait
echo "chain2 done $(date '+%m-%d %H:%M:%S')" >> $S
