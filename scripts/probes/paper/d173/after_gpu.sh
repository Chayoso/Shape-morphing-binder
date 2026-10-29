#!/bin/bash
# after_gpu.sh WHAT QUEUE_NAME -- D173's detached readers and deliverables that need a card (card 3, beside the queue's job):
#   tex40 / tex300   d173_texread.py, then the painted (kernel / nn) and grey end frames and a nn video (deliver_colour2.sh)
#   cmb300           the relief share (d172_detail.py, the pattern region), the colour (D160's reader), the edge widths
#                    (d173_edge.py), the grey / painted deliverables and the relief close-ups (d170_render.py close)
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d173; O=$B/output/gpu/d173; G=$B/output/gpu; Q=$B/tmp/gpu1_queue
WHAT=$1; QN=$2; D=$O/auto/$WHAT; mkdir -p $D
log() { echo "$(date '+%m-%d %H:%M:%S') [after $WHAT] $*" >> $O/status.txt; }
for i in $(seq 1 2160); do [ -e $Q/$QN.done ] && break; sleep 30; done
export PHYSMORPH_RUN_REPO=$B/repo_d173 PHYSMORPH_OUT=$G; source $B/repo_d173/scripts/ops/hyde06_env.sh > /dev/null 2>&1
source $B/tmp/d146/cache_env.sh; export TMPDIR=$W/tmpdir CUDA_VISIBLE_DEVICES=3 D157_THREADS=16 OMP_NUM_THREADS=16; mkdir -p $TMPDIR; cd $B/repo_d173
log "start"
case $WHAT in
  tex40|tex300)
    n=40000; seeds=97,99; [ $WHAT = tex300 ] && { n=300000; seeds=97; }
    $PY $W/d173_texread.py $D $n $seeds > $D/read.log 2>&1
    log "read exit $?: $(grep VERDICT $D/summary.md | cut -c1-300)"
    k=$((n / 1000))
    bash $W/deliver_colour2.sh tex${k} $O/runs/tex${k}k_Rx_97 $O/runs/tex${k}k_T_97 3 texture >> $D/deliver.log 2>&1 ;;
  cmb300)
    SM=$G/d161/assets/d161smooth_bunny.obj; RL=$G/d170/assets/d170L300_bunny.obj
    echo "{\"scene\": \"cmb300\", \"N\": 300000, \"tgt\": \"$SM\", \"smooth\": \"$SM\", \"detail\": \"$RL\", \"s_bg\": 10.0, \"runs\": {\"Rx_97\": \"$O/runs/cmb300k_Rx_97\", \"T_97\": \"$O/runs/cmb300k_T_97\"}}" > $D/scene_relief.json
    CUDA_VISIBLE_DEVICES= $PY $B/tmp/d172/d172_detail.py $D/read_relief.json $D/scene_relief.json > $D/read_relief.log 2>&1
    printf '%s\n' "cmb300k_T_97 band $O/runs/cmb300k_T_97_frames12.npz mesh=bunny" "cmb300k_Rx_97 band $O/runs/cmb300k_Rx_97_frames12.npz mesh=bunny twin=cmb300k_T_97 timeline" > $D/jobs_colour.txt
    $PY $B/tmp/d160/d160_read.py $D/read_colour.jsonl $D/jobs_colour.txt > $D/read_colour.log 2>&1
    printf '%s\n' "cmb300_Rx_97@tgt $O/runs/cmb300k_Rx_97_frames12.npz" "cmb300_Rx_97 $O/runs/cmb300k_Rx_97_frames12.npz" "cmb300_T_97 $O/runs/cmb300k_T_97_frames12.npz" > $D/edge_jobs.txt
    $PY $W/d173_edge.py $D/edge.jsonl $D/edge_jobs.txt > $D/edge.log 2>&1
    log "reads done"
    bash $W/deliver_colour2.sh cmb300 $O/runs/cmb300k_Rx_97 $O/runs/cmb300k_T_97 3 band >> $D/deliver.log 2>&1
    stop=$($PY $B/tmp/d172/pick.py stop $O/runs/cmb300k_Rx_97.json)
    CUDA_VISIBLE_DEVICES= $PY $B/tmp/d170/d170_render.py close $O/deliver/cmb300/close $SM 300000 $RL $O/runs/cmb300k_T_97_frames12.npz \
      $O/runs/cmb300k_Rx_97_frames12.npz $stop auto auto 1.0 84 > $D/close.log 2>&1 ;;
esac
log "done"
