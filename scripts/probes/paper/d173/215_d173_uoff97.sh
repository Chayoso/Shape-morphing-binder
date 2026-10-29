#!/bin/bash
# 215_d173_uoff97.sh -- queue job (agent D173): the u-off arm of the attribution (stylized scales E, 300k, seed 97): Hu with u
# disabled (--u_off, D124: u's gate zero on every particle), the render reaching dF_c alone at the rule's weight: how much of
# the relief the stress increment writes on its own.
J=215_uoff97; TITLE="D173 attribution, u off, seed 97"
source /data/relcfd/chayo/physmorph_v2/tmp/d173/job_common.sh
gate_t; log "start"
run1 atr300k_Hu0_97 $SM300 --seed 97 --render_tgt $G/d172/assets/d172SE300_dragon.obj $HU --u_off
log "done"
