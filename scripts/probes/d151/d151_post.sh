#!/bin/bash
# d151_post.sh -- D151's CPU readings once each run's side render and detector are written: visible_apart over the whole
# run (every 4th kept frame) for the new runs (the base's exist), and d148_why_floating.py's confirmed pieces for the new
# runs and the base (D149's SL / SP rendered by d151_base_render.sh). Logs in tmp/d151, records in output/gpu/d151.
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp/d151; G=$B/output/gpu; D=$G/d151; R=$D/render
export PHYSMORPH_RUN_REPO=$B/repo_d148x; source $B/repo_d148x/scripts/ops/hyde06_env.sh > /dev/null; cd $B/repo_d148x
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
F="Warp\|CUDA\|Devices\|\"cpu\"\|Kernel\|cache/warp\|Toolkit\|initialized"
one() {   # one NAME FRAMES JSON NEW(1/0)
  name=$1; fr=$2; js=$3; new=$4
  until [ -f $D/img_${name}_side_tracks.json ] || grep -q "chain stopped at $name" $D/status.txt 2>/dev/null; do sleep 120; done
  [ -f $D/img_${name}_side_tracks.json ] || return
  if [ "$new" = 1 ]; then
    mesh=${name%%300k_*}; tag=${name##*_}
    nice -n 5 $PY scripts/probes/paper/visible_apart.py $fr --label $tag --w0 0 --w1 999 --every 4 \
      --jsonl $D/vfull_${mesh}_$tag.jsonl --threads 12 2>&1 | grep -v "$F" > $T/vfull_$name.log
  fi
  nice -n 5 $PY $B/tmp/d148/d148_why_floating.py $name $fr $js $D/why_$name \
    side:$D/img_${name}_side_tracks.json:$R/${name}_side/frames:90 2>&1 | grep -v "$F" > $T/why_$name.log
  echo "post $name $(date '+%m-%d %H:%M')" >> $D/status.txt
}
for n in dragon300k_NL dragon300k_NP bunny300k_NL bunny300k_NP; do
  one $n $D/${n}_frames12.npz $D/$n.json 1 &
done
for n in dragon300k_SL dragon300k_SP bunny300k_SL bunny300k_SP; do
  one $n $G/d149/${n}_frames12.npz $G/d149/$n.json 0 &
done
wait
echo "post done $(date '+%m-%d %H:%M')" >> $D/status.txt
