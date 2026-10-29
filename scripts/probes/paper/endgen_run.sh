#!/bin/bash
# D145: B5's end check on Xu et al.'s six seed-97 archives and the OT maps (bunny, dragon); a pool job (GPU metrics).
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; E=$G/d145_evidence
export PHYSMORPH_RUN_REPO=$B/repo_d145; source $B/repo_d145/scripts/ops/hyde06_env.sh > /dev/null 2>&1
cd $B/repo_d145; rm -f $E/endgen_1015.jsonl
specs=""; for m in bunny dragon teapot bob armadilo homer; do specs="$specs $m=$G/d145_b4/${m}300k_XU97_frames12.npz"; done
$PY $B/tmp/d145/endgen145.py $E/endgen_1015.jsonl $specs bunny=$G/d145_b6/bunny300k_OT_frames12.npz dragon=$G/d145_b6/dragon300k_OT_frames12.npz
