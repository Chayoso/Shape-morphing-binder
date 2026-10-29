#!/bin/bash
# 206_d174_cut.sh -- queue job (agent D174): what a shorter run delivers (d174_cut.py on the two solo profile runs: the metrics
# at every 10th kept window against the wall a run stopped there would take), ~2 min.
J=206_cut
source /data/relcfd/chayo/physmorph_v2/tmp/d174/d174_common.sh
( envr repo_d174; $PY $W/d174_cut.py $O/runs/P_dragon300k_Hu_97 86 32 > $O/auto/cut_dragon.md 2> $O/auto/cut_dragon.err
  $PY $W/d174_cut.py $O/runs/P_bunny300k_Hu_97 61 27 > $O/auto/cut_bunny.md 2> $O/auto/cut_bunny.err )
log "cut tables written (auto/cut_dragon.md, auto/cut_bunny.md)"
