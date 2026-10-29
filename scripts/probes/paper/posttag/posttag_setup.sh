#!/bin/bash
# [POSTTAG_FX_FLAG=--no-<fix>] posttag_setup.sh TAG -- D145 post-tag, server half (after posttag_local.sh): the tag's code
# copies for the evidence re-run (POSTTAG_FX_FLAG: the floating-Gaussian fix's off-flag when the tag carries the fix;
# amendment 6's NFX arm is generated only then)
#   repo_t   the tag as is (full archive): B3 / B9 40k (g40, g40f) and the determinism reference
#   repo_ts  + patch_s.py (kept frames written in place of the full archive; output only): every 300k end300 run
#   repo_tgf + patch_d145g.py (D121 hooks) + rgv_patch.py + rgv_patch2.py (RGV's records), full archive: B7 / RGV (jv)
# Every patch asserts its exact match (stops here if the tag moved the code: the patch is ported first). The evidence
# queue is generated and every flag in it checked against the tag's runner; then the determinism gate goes to the head
# of the pool's queue and posttag_watch.sh appends the evidence queue once the gate passes.
set -e
TAG=$1; B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp/d145; P=$T/posttag; L=$B/output/gpu/d145_pool.log; Q=$T/main.queue
[ -s $P/tag_$TAG.tgz ] || { echo "no $P/tag_$TAG.tgz (run posttag_local.sh first)"; exit 1; }
for r in repo_t repo_ts repo_tgf; do [ -e $B/$r ] && { echo "$B/$r exists"; exit 1; }; done
mkdir -p $B/repo_t && tar -xzf $P/tag_$TAG.tgz -C $B/repo_t
echo "$TAG" > $B/repo_t/TAG
[ -f $B/repo_t/scripts/probes/paper/visible_apart.py ] || { echo "the tag has no visible_apart.py"; exit 1; }
source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1
cp -a $B/repo_t $B/repo_ts && $PY $B/repo_t/scripts/probes/paper/patch_s.py $B/repo_ts
cp -a $B/repo_t $B/repo_tgf && $PY $T/b7/patch_d145g.py $B/repo_tgf && $PY $B/repo_t/scripts/probes/paper/rgv_patch.py $B/repo_tgf \
  && $PY $B/repo_t/scripts/probes/paper/rgv_patch2.py $B/repo_tgf
echo "$(date '+%m-%d %H:%M') post-tag: repo_t / repo_ts / repo_tgf from $TAG (patch_s; hooks + rgv_patch + rgv_patch2)" >> $L
$PY $P/gen_posttag.py $Q $P/main_posttag.queue > $P/gen.log
# every flag of the evidence queue must exist in the tag's runner (repo_tgf: the tag + the hooks, whose flags only B7 uses)
miss=$(grep -v "^#" $P/main_posttag.queue | grep -o -- " --[a-z_-]*" | sed -e 's/^ //' -e 's/^--no-/--/' | sort -u \
  | while read -r fl; do grep -q -- "\"$fl\"" $B/repo_tgf/scripts/pipeline_run.py || echo "$fl"; done)
[ -z "$miss" ] || { echo "flags missing in the tag: $miss"; echo "$(date '+%m-%d %H:%M') post-tag: flags missing in $TAG: $miss (queue not armed)" >> $L; exit 1; }
# the gate: the determinism proof first, the evidence queue after it
echo "detproof_tag job 0 - d145t_detproof - 0 bash $P/detproof_tag.sh" > $P/gate.lines
first=$(grep -v "^#" $Q | while read -r n m x r d rest; do [ -z "$n" ] && continue; [ -e $B/output/gpu/$d/$n.claim ] || { echo "$n"; break; }; done)
if [ -n "$first" ]; then awk -v f=$P/gate.lines -v n="$first" '$1==n && !done {while ((getline l < f) > 0) print l; done=1} {print}' $Q > $Q.new && mv $Q.new $Q
else cat $P/gate.lines >> $Q; fi
setsid nohup bash $P/posttag_watch.sh > $P/watch.out 2>&1 < /dev/null &
echo "$(date '+%m-%d %H:%M') post-tag: determinism gate queued (detproof_tag); posttag_watch.sh appends main_posttag.queue ($(grep -vc '^#' $P/main_posttag.queue) items) when it passes" >> $L
