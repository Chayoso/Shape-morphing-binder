#!/bin/bash
# 206_d174_lse3.sh -- agent D174: the shared-memory log-sum-exp's block height (8 / 16 / 32 outputs a column), ~1 min
B=/data/relcfd/chayo/physmorph_v2
export PHYSMORPH_RUN_REPO=$B/repo_d174; source $B/repo_d174/scripts/ops/hyde06_env.sh > /dev/null 2>&1
source $B/tmp/d146/cache_env.sh; export TMPDIR=$B/tmp/d174/tmpdir; cd $B/repo_d174
for n in 87 71 106; do $PY $B/tmp/d174/bench_lse3.py $n 200; done > $B/tmp/d174/lse3.out 2>&1
