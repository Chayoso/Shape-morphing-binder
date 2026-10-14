#!/bin/bash
# d148_run.sh MESH N TAG ARM [pipeline_run args] -- one D148 run on the hook copy (D148_REPO, default repo_d148f: the
# --gap_split code + D140's read-only dump of windows 1-9, frames12 in place of the archive), seed 97 unless the args
# say otherwise, on the first free card (gpu_locks, mkdir; < 2 GB in use; at most D148_MAX cards held by D148, default 2;
# output/ under 480 GB). Then: D139's shell-against-bulk reading (the dump removed after, logged), d120_apart (the
# particles apart from the body), the split record ([gap] lines of the log); at 300k the end evaluation (t3end_, e3end_
# as D144/D146's chain), d118_rows and the relief's discs; ARM=1: side and front renders (render_deliverable +fit=0
# +field=zbs) and the detector on both; ARM=2: the side render and the detector; ARM=0: none.
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp; G=$B/output/gpu; D=$G/d148; L=$B/gpu_locks; mkdir -p $D $L
source $T/d146/cache_env.sh                                       # caches and temp on /data
mesh=$1; n=$2; tag=$3; arm=$4; shift 4
nk=$(( n / 1000 ))k; name=${mesh}${nk}_$tag; S=$D/status.txt
RP=${D148_REPO:-repo_d148f}; MAX=${D148_MAX:-2}
until [ "$(du -s --block-size=1G $B/output | cut -f1)" -lt 480 ]; do sleep 60; done
gpu=${D148_GPU:-}                                              # set: the caller holds that card's lock
while [ -z "$gpu" ]; do
  held=$(grep -l "^agent D148" $L/gpu*.lock/info 2>/dev/null | wc -l)
  if [ "$held" -lt "$MAX" ]; then
    for g in ${D148_CARDS:-3 2 1 0}; do
      if [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)" -lt 2000 ] && mkdir $L/gpu$g.lock 2>/dev/null; then
        gpu=$g; break
      fi
    done
  fi
  [ -z "$gpu" ] && sleep 30
done
if [ -z "${D148_GPU:-}" ]; then
  trap 'rm -rf $L/gpu$gpu.lock' EXIT
  printf "agent D148\nrun d148/%s\npid %s\nstart %s\n" "$name" "$$" "$(date '+%Y-%m-%d %H:%M %Z')" > $L/gpu$gpu.lock/info
fi
echo "start $name gpu $gpu $* $(date '+%m-%d %H:%M')" >> $S
(
  export PHYSMORPH_RUN_REPO=$B/$RP PHYSMORPH_OUT=$G D140_DUMP=$D/d140_$name D140_W1=9
  source $B/$RP/scripts/ops/hyde06_env.sh > /dev/null; cd $B/$RP
  ( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu)"; sleep 10; done ) > $D/$name.mem &
  mem=$!
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/pipeline_run.py --tgt assets/$mesh.obj --n $n --seed 97 --out $D/$name "$@" \
    > $D/$name.log 2>&1 < /dev/null
  rc=$?; kill $mem; echo $rc > $D/rc_$name
  echo "$name sim $rc wall $(( $(date +%s) - t0 )) peak_mib $(sort -k2 -n $D/$name.mem | tail -1 | cut -d' ' -f2) $(date '+%m-%d %H:%M')" >> $S
  [ $rc = 0 ] || exit 1
  $PY $T/d137/d139_shear.py $name=$D/d140_$name:$D/${name}_frames12.npz > $D/shear_$name.log 2>&1 < /dev/null
  sz=$(du -sm $D/d140_$name | cut -f1); rm -rf $D/d140_$name
  echo "$(date '+%Y-%m-%d %H:%M') d148: removed $D/d140_$name ($sz MB; D140's read-only dump of windows 1-9, read into shear_$name.log)" >> $G/cleanup.log
  $PY $T/d120_apart.py $D/${name}_frames12.npz > $D/apart_$name.log 2>&1 < /dev/null
  grep "^\[gap\]" $D/$name.log > $D/gap_$name.log
)
[ "$(cat $D/rc_$name 2>/dev/null)" = 0 ] || { echo "chain stopped at $name $(date '+%m-%d %H:%M')" >> $S; exit 1; }
F=$D/${name}_frames12.npz
if [ "$n" = 300000 ]; then
  (
    source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null; cd $B/repo_r77
    IND=$G/d90/${mesh}_ind300k.npz
    box=1900,100,2800,700; [ $mesh = dragon ] && box=1900,60,2500,560
    last=$($PY $T/lastraw.py $F)
    one=$D/${name}_end1.npz
    $PY $T/d146/end1.py $F $one
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/probes/settled/render_terms_probe.py $one ref=$IND > $D/t3end_$name.log 2>&1 < /dev/null
    rm -f $one
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/probes/settled/surface_layer_probe.py $F $D/e3end_$name $box measure \
        ref=$IND target own $last > $D/e3end_$name.log 2>&1 < /dev/null
    echo "endeval $name state $last $(date '+%m-%d %H:%M')" >> $S
  )
  (
    source $B/repo_r111/scripts/ops/hyde06_env.sh > /dev/null; cd $B/repo_r111
    CUDA_VISIBLE_DEVICES=$gpu $PY $T/d118_rows.py $mesh $tag=d148/$tag > $D/rows_$name.log 2>&1 < /dev/null
  )
  (
    export PHYSMORPH_RUN_REPO=$B/$RP
    source $B/$RP/scripts/ops/hyde06_env.sh > /dev/null; cd $B/$RP
    last=$($PY $T/lastraw.py $F); mkdir -p $D/discs
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/probes/settled/exterior_offset_probe.py $F assets/$mesh.obj $last save=$D/discs > $D/pend_$name.log 2>&1 < /dev/null
  )
fi
if [ "$arm" != 0 ]; then
  (
    export PHYSMORPH_RUN_REPO=$B/$RP
    source $B/$RP/scripts/ops/hyde06_env.sh > /dev/null; cd $B/$RP
    W=$D/render; mkdir -p $W
    DN="$G/d111/${mesh}_dense15.npz,$G/d111/${mesh}_dense15_s9.npz,$G/d111/${mesh}_dense15_s10.npz,$G/d111/${mesh}_dense15_s11.npz"
    views="side +az=90"; [ "$arm" = 1 ] && views="side +az=90 front +az=35"
    set -- $views
    while [ $# -gt 0 ]; do
      v=$1; az=$2; shift 2; r=${name}_$v
      t0=$(date +%s)
      CUDA_VISIBLE_DEVICES=$gpu $PY scripts/render_deliverable.py $F $DN $W/$r +stride=1 +video=1 +fit=0 +field=zbs $az > $W/$r.log 2>&1 < /dev/null
      ffmpeg -loglevel error -y -framerate 20 -i $W/$r/frames/%04d.jpg -c:v libx264 -crf 18 -pix_fmt yuv420p $D/$r.mp4
      echo "render $r rc=$? $(( $(date +%s) - t0 )) s gpu $gpu $(date '+%m-%d %H:%M')" >> $S
      $PY $T/d137/d140_image.py $W/$r/frames $F ${az#+az=} $D/img_$r > $D/img_$r.log 2>&1 < /dev/null
    done
  )
fi
echo "done $name $(date '+%m-%d %H:%M')" >> $S
