#!/bin/bash
# d151_base_render.sh -- D151's base side renders for the side-by-side videos: D149's SL (dragon, bunny) and SP (dragon,
# bunny), render_deliverable +field=zbs +az=90 with D148's dense discs and the detector, as d151_chain.sh renders its own;
# on the first free card (gpu_locks, agent D148, at most 2), released by a trap. Output in output/gpu/d151.
source /data/relcfd/chayo/physmorph_v2/tmp/d146/cache_env.sh
B=/data/relcfd/chayo/physmorph_v2; L=$B/gpu_locks; RS=$B/repo_d151; G=$B/output/gpu; D=$G/d151; T=$B/tmp
mkdir -p $D $L; S=$D/status.txt
# after the gate's chains (they hold D148's cards first): both meshes' chains done or stopped
until grep -q "chain dragon done\|chain stopped at dragon" $S 2>/dev/null && grep -q "chain bunny done\|chain stopped at bunny" $S 2>/dev/null; do
  sleep 60
done
gpu=""
while [ -z "$gpu" ]; do
  held=$(grep -l "^agent D148" $L/gpu*.lock/info 2>/dev/null | wc -l)
  if [ "$held" -lt 2 ]; then
    for g in 3 2 1 0; do
      if [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)" -lt 2000 ] && mkdir $L/gpu$g.lock 2>/dev/null; then
        gpu=$g; break
      fi
    done
  fi
  [ -z "$gpu" ] && sleep 30
done
trap 'rm -rf $L/gpu$gpu.lock' EXIT
printf "agent D148\nrun d151/base side renders (D149 SL, SP)\npid %s\nstart %s\n" "$$" "$(date '+%Y-%m-%d %H:%M %Z')" > $L/gpu$gpu.lock/info
export PHYSMORPH_RUN_REPO=$RS
source $RS/scripts/ops/hyde06_env.sh > /dev/null; cd $RS
W=$D/render; mkdir -p $W
for mesh in dragon bunny; do
  for arm in SL SP; do
    name=${mesh}300k_$arm; F=$G/d149/${name}_frames12.npz
    DN="$G/d111/${mesh}_dense15.npz,$G/d111/${mesh}_dense15_s9.npz,$G/d111/${mesh}_dense15_s10.npz,$G/d111/${mesh}_dense15_s11.npz"
    r=${name}_side
    t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/render_deliverable.py $F $DN $W/$r +stride=1 +video=1 +fit=0 +field=zbs +az=90 > $W/$r.log 2>&1 < /dev/null
    ffmpeg -loglevel error -y -framerate 20 -i $W/$r/frames/%04d.jpg -c:v libx264 -crf 18 -pix_fmt yuv420p $D/$r.mp4
    echo "render base $r rc=$? $(( $(date +%s) - t0 )) s gpu $gpu $(date '+%m-%d %H:%M')" >> $S
    $PY $T/d137/d140_image.py $W/$r/frames $F 90 $D/img_$r > $D/img_$r.log 2>&1 < /dev/null
  done
done
echo "base renders done, released gpu $gpu $(date '+%m-%d %H:%M')" >> $S
