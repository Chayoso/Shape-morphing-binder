#!/bin/bash
# the rest of bob after the stopped ISDn1 (D145 methods table, 2026-10-21): pieces145.py with the collapsed-frame skip
source /data/relcfd/chayo/physmorph_v2/tmp/d145/env_d145.sh; source /data/relcfd/chayo/physmorph_v2/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8
m=bob; s=""
for t in ISDn1 FDn1; do s="$s $m:$t=/data/relcfd/chayo/physmorph_v2/output/gpu/d145_b14/${m}300k_${t}_frames12.npz"; done
for a in LSa97 LSb97 FCa97 FCb97; do s="$s $m:$a=/data/relcfd/chayo/physmorph_v2/output/gpu/d145_b12/${m}300k_${a}_frames12.npz"; done
rm -f /data/relcfd/chayo/physmorph_v2/output/gpu/d145_ext/pieces_ext_bobrest.jsonl
nice -n 10 $PY /data/relcfd/chayo/physmorph_v2/tmp/d145r/pieces145.py /data/relcfd/chayo/physmorph_v2/output/gpu/d145_ext/pieces_ext_bobrest.jsonl $s --every 1 >> /data/relcfd/chayo/physmorph_v2/output/gpu/d145_ext/pieces_ext.log 2>&1
echo "done pieces bob rest $(date "+%m-%d %H:%M")" >> /data/relcfd/chayo/physmorph_v2/output/gpu/d145_ext/status.txt
