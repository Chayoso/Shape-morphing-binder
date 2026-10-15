#!/bin/bash
# d151_chain.sh MESH [MESH ...] -- D151's 300k gate (pre-registered 2026-10-15 14:02 CDT): for each MESH, MESH300k_NL
# (render arm) then MESH300k_NP (twin, --render_weight_scale 0) on repo_d151 (refreeze-cand 169f251 + --layer_link), seed
# 97, the refreeze-cand defaults plus --layer_link; on the first free card (gpu_locks, mkdir; < 2 GB in use; at most 2
# cards held by agent D148; output/ under 480 GB), released by a trap. After each run, as D149's chain: the end frame's
# yardstick (t3end_) and display (e3end_) against d90's independent sample, momentum on every kept frame (m_), the apart
# particles on the CPU (apart_, the first 35 % of the kept frames); then the side render (render_deliverable +field=zbs
# +az=90, D148's dense discs) and the detector (d140_image). Status in output/gpu/d151/status.txt.
source /data/relcfd/chayo/physmorph_v2/tmp/d146/cache_env.sh
B=/data/relcfd/chayo/physmorph_v2; L=$B/gpu_locks; RS=$B/repo_d151; G=$B/output/gpu; D=$G/d151; T=$B/tmp
mkdir -p $D $L; S=$D/status.txt
until [ "$(du -s --block-size=1G $B/output | cut -f1)" -lt 480 ]; do sleep 60; done
# the full suite on repo_d151 first (tmp/d151_suite.sh): only on its exit 0
until grep -q "^exit" $T/d151_suite.log 2>/dev/null; do sleep 60; done
grep -q "^exit 0 " $T/d151_suite.log || { echo "suite failed: chain $* not started $(date "+%m-%d %H:%M")" >> $S; exit 1; }
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
printf "agent D148\nrun d151/%s 300k NL then NP (+ readings, side renders)\npid %s\nstart %s\n" "$*" "$$" "$(date '+%Y-%m-%d %H:%M %Z')" > $L/gpu$gpu.lock/info
echo "holding gpu $gpu for $* $(date '+%m-%d %H:%M')" >> $S
for mesh in "$@"; do
  IND=$G/d90/${mesh}_ind300k.npz
  box=1900,100,2800,700; [ $mesh = dragon ] && box=1900,60,2500,560
  for arm in NL NP; do
    name=${mesh}300k_$arm
    extra=""; [ $arm = NP ] && extra="--render_weight_scale 0"
    (
      export PHYSMORPH_RUN_REPO=$RS PHYSMORPH_OUT=$D
      source $RS/scripts/ops/hyde06_env.sh > /dev/null; cd $RS
      ( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu)"; sleep 10; done ) > $D/$name.mem &
      mem=$!
      t0=$(date +%s)
      CUDA_VISIBLE_DEVICES=$gpu $PY scripts/pipeline_run.py --tgt assets/$mesh.obj --n 300000 --seed 97 --layer_link \
        $extra --out $D/$name > $D/$name.log 2>&1 < /dev/null
      rc=$?; kill $mem; echo $rc > $D/rc_$name
      echo "run $name rc $rc wall $(( $(date +%s) - t0 )) s peak_mib $(sort -k2 -n $D/$name.mem | tail -1 | cut -d' ' -f2) $(date '+%m-%d %H:%M')" >> $S
    )
    [ "$(cat $D/rc_$name 2>/dev/null)" = 0 ] || { echo "chain stopped at $name $(date '+%m-%d %H:%M')" >> $S; continue; }
    F=$D/${name}_frames12.npz
    (
      source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null; cd $B/repo_r77
      last=$($PY $T/lastraw.py $F)
      $PY $T/d145/endframe.py $F $D/${name}_end1.npz
      CUDA_VISIBLE_DEVICES=$gpu $PY scripts/probes/settled/render_terms_probe.py $D/${name}_end1.npz ref=$IND > $D/t3end_$name.log 2>&1 < /dev/null
      rm -f $D/${name}_end1.npz
      CUDA_VISIBLE_DEVICES=$gpu $PY scripts/probes/settled/surface_layer_probe.py $F $D/e3end_$name $box measure ref=$IND target own $last > $D/e3end_$name.log 2>&1 < /dev/null
      echo "endeval $name state $last $(date '+%m-%d %H:%M')" >> $S
    )
    $T/mom.sh $gpu $D/m_$name.log "$mesh $arm=$F"
    echo "momentum $name $(date '+%m-%d %H:%M')" >> $S
    (
      source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null
      OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" nice $PY $T/d120_apart.py $F 0.35 > $D/apart_$name.log 2>&1 < /dev/null
      echo "apart $name rc $? $(date '+%m-%d %H:%M')" >> $S
    ) &
    (
      export PHYSMORPH_RUN_REPO=$RS
      source $RS/scripts/ops/hyde06_env.sh > /dev/null; cd $RS
      W=$D/render; mkdir -p $W
      DN="$G/d111/${mesh}_dense15.npz,$G/d111/${mesh}_dense15_s9.npz,$G/d111/${mesh}_dense15_s10.npz,$G/d111/${mesh}_dense15_s11.npz"
      r=${name}_side
      t0=$(date +%s)
      CUDA_VISIBLE_DEVICES=$gpu $PY scripts/render_deliverable.py $F $DN $W/$r +stride=1 +video=1 +fit=0 +field=zbs +az=90 > $W/$r.log 2>&1 < /dev/null
      ffmpeg -loglevel error -y -framerate 20 -i $W/$r/frames/%04d.jpg -c:v libx264 -crf 18 -pix_fmt yuv420p $D/$r.mp4
      echo "render $r rc=$? $(( $(date +%s) - t0 )) s gpu $gpu $(date '+%m-%d %H:%M')" >> $S
      $PY $T/d137/d140_image.py $W/$r/frames $F 90 $D/img_$r > $D/img_$r.log 2>&1 < /dev/null
    )
  done
  echo "chain $mesh done $(date '+%m-%d %H:%M')" >> $S
done
wait
echo "released gpu $gpu $(date '+%m-%d %H:%M')" >> $S
