#!/bin/bash
# 209_d175_show300.sh -- queue job (agent D175, after 208 passed): the SIGGRAPH-style showcase. Bunny -> dragon at 300k
# with a picture keyframe: the extruded letter S (make_kf_letter.py) at t_k = $TK s, Hu. Arms (seed 97): dF_c + layer u +
# render loss (full) and dF_c + layer u (twin), both reading the same storyboard. Then the reader (KF-IoU at t_k, the final
# shape, rest) and the deliverables in output/gpu/d175/deliver/show300: the sheet (the keyframe | twin | full at t_k)
# and the twin | full video (4k + 1080p).
J=209_show300; TITLE="D175 showcase: bunny -> letter S -> dragon, 300k"
source /data/relcfd/chayo/physmorph_v2/tmp/d175/job_common.sh
TK=${TK:-1.0}; FK=$(awk "BEGIN{print int($TK * 240 + 0.5)}")
SC="--src assets/bunny.obj --tgt assets/dragon.obj --n 300000 --seed 97"
DL=$O/deliver/show300; mkdir -p $DL
log "start (card $CUDA_VISIBLE_DEVICES), t_k $TK s = frame $FK"
( envr $REPO; [ -s $A/kf_S300.npz ] || $PY $W/make_kf_letter.py $A/kf_S300.npz $SC --letter S > $A/kf_S300.log 2>&1 < /dev/null )
[ -s $A/kf_S300.npz ] || { log "letter keyframe not built ($A/kf_S300.log): stop"; exit 1; }
log "keyframe: $(grep '^\[kf\]' $A/kf_S300.log | cut -c1-300)"
PAR=2
runp show_S_97 $SC $HU --keyframe $A/kf_S300.npz --keyframe_time $TK
runp show_T_97 $SC $HU --keyframe $A/kf_S300.npz --keyframe_time $TK $TW
wait
log "runs done; reader"
( envr $REPO; $PY $W/d175_read.py $O/auto/$J $O/runs $A --src assets/bunny.obj --tgt assets/dragon.obj --n 300000 \
    --runs S97,T97 --kf kf_S300.npz --prefix show_ --t_k $FK \
    --title "D175 showcase: bunny -> letter S at t = $TK s -> dragon, 300k" > $O/auto/$J/read.log 2>&1 < /dev/null )
log "reader exit $?"
log "done (the sheet and the video: tmp/d175/post_show.sh, CPU, detached by the agent)"
