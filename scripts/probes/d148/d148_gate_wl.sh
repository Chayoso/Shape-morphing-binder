#!/bin/bash
# d148_gate_wl.sh -- D148c (walls) 300k gate, pre-registered 14:30 CDT: two lanes of d148_run.sh on repo_d148wf, at most
# 2 cards held by D148: WG (dragon arm, side+front) -> WB (bunny arm, side+front) -> WG99 (dragon arm seed 99, side);
# WT (dragon twin, side) -> WBT (bunny twin).
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp/d148; D=$B/output/gpu/d148
export D148_MAX=2 D148_REPO=repo_d148wf
cd $T
setsid nohup bash -c "./d148_run.sh dragon 300000 WG 1 --gap_split walls; ./d148_run.sh bunny 300000 WB 1 --gap_split walls; ./d148_run.sh dragon 300000 WG99 2 --gap_split walls --seed 99" > $T/q_wl1.log 2>&1 < /dev/null &
sleep 40
setsid nohup bash -c "./d148_run.sh dragon 300000 WT 2 --gap_split walls --render_weight_scale 0; ./d148_run.sh bunny 300000 WBT 0 --gap_split walls --render_weight_scale 0" > $T/q_wl2.log 2>&1 < /dev/null &
echo "gate WL queued $(date +%m-%d\ %H:%M)" >> $D/status.txt
