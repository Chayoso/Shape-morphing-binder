#!/bin/bash
# detproof_jm.sh -- D145: the Jx_mean record (d145-jmean 2b0cafc) against refreeze-prep d2f779d, 40k bunny seed 97, four
# windows, Warp run-to-run deterministic mode: the kept frames, every metric and every shared history field bit for bit.
source /data/relcfd/chayo/physmorph_v2/tmp/d145/env_d145.sh
B=/data/relcfd/chayo/physmorph_v2; D=$B/output/gpu/d145_detproof_jm; mkdir -p $D; export PHYSMORPH_OUT=$B/output/gpu
for r in A:repo_rp_base B:repo_rp_jmean; do t=${r%%:*}; repo=${r#*:}
  export PHYSMORPH_RUN_REPO=$B/$repo; source $B/$repo/scripts/ops/hyde06_env.sh > /dev/null; cd $B/$repo
  $PY $B/tmp/rt/det_run.py scripts/pipeline_run.py --tgt assets/bunny.obj --n 40000 --seed 97 --animations 4 --out $D/det$t > $D/det$t.log 2>&1
  $PY $B/tmp/frames12.py $D/det${t}_render_full_dt_iso_nn.npz $D/det$t >> $D/det$t.log 2>&1 && rm -f $D/det${t}_render_full_dt_iso_nn.npz
done
$PY $B/tmp/d145/det_cmp12.py $D/detA $D/detB > $D/detproof.txt 2>&1
$PY -c "import json; h=list(json.load(open(\"$D/detB.json\"))[\"arms\"].values())[0][\"history\"]; print(\"Jx_mean in B:\", [round(r.get(\"Jx_mean\", float(\"nan\")), 6) for r in h])" >> $D/detproof.txt 2>&1
tail -3 $D/detproof.txt
