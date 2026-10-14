#!/bin/bash
# b6.sh IN_FRAMES OUT_NPZ -- D145 B6 (ot_disp.py) in the deliverable code's environment, on the card the pool gave it
B=/data/relcfd/chayo/physmorph_v2
source $B/repo_d145/scripts/ops/hyde06_env.sh > /dev/null
mkdir -p $(dirname $2)
cd $B/repo_d145
$PY $B/tmp/d145/ot_disp.py "$1" "$2"
