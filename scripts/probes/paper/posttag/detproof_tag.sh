#!/bin/bash
# detproof_tag.sh -- D145 post-tag gate (a pool job, one card): 40k bunny seed 97, four windows, Warp run-to-run
# deterministic (tmp/rt/det_run.py). A = repo_t (the tag, full archive, cut afterwards by tmp/frames12.py); S = repo_ts
# (kept frames written in place); G = repo_tgf with the hooks on at every window (gradient dump, lambda = 0 counterfactual).
# S and G must equal A bit for bit (det_cmp12.py: kept frames, every metric, every history field but clocks).
B=/data/relcfd/chayo/physmorph_v2; D=$B/output/gpu/d145t_detproof; mkdir -p $D
source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1
run() { local repo=$1 out=$2; shift 2
  ( export PHYSMORPH_OUT=$B/output/gpu PHYSMORPH_RUN_REPO=$B/$repo; source $B/$repo/scripts/ops/hyde06_env.sh > /dev/null 2>&1
    cd $B/$repo && $PY $B/tmp/rt/det_run.py scripts/pipeline_run.py --tgt assets/bunny.obj --n 40000 --seed 97 --animations 4 \
      --out $D/$out "$@" > $D/$out.log 2>&1 ) ; }
run repo_t A && $PY $B/tmp/frames12.py $D/A_render_full_dt_iso_nn.npz $D/A >> $D/A.log 2>&1 && rm -f $D/A_render_full_dt_iso_nn.npz
run repo_ts S
run repo_tgf G --grad_dump $D/G_dump --dump_windows 0,+1 --cf_dir $D/G_cf --cf_windows 0,1,2,3 \
  && $PY $B/tmp/frames12.py $D/G_render_full_dt_iso_nn.npz $D/G >> $D/G.log 2>&1 && rm -f $D/G_render_full_dt_iso_nn.npz
$PY $B/tmp/d145/det_cmp12.py $D/A $D/S > $D/ts.txt 2>&1
$PY $B/tmp/d145/det_cmp12.py $D/A $D/G > $D/tgf.txt 2>&1
rm -rf $D/G_dump $D/G_cf
echo "repo_ts: $(tail -1 $D/ts.txt) | repo_tgf: $(tail -1 $D/tgf.txt)" > $D/verdict.txt
cat $D/verdict.txt
