#!/bin/bash
# d151_fix.sh -- D151's chains ran all four runs (rc 0) but every step after them found no file: repo_d151's
# pipeline_run.py writes the full archive (NAME_render_full_dt_iso_nn.npz), while D149's kept-frames file
# (NAME_frames12.npz) is a server-only patch of repo_d149s. This builds NAME_frames12.npz from each full archive with
# D149's rule (frames12_from_archive.py, the same arrays) and then runs d151_chain.sh's post-run block unchanged (end
# yardstick + display, momentum, apart, the side render, the detector). The runs are not repeated. d151_post.sh (running
# since 14:12) picks up visible_apart / why_floating when each detector's tracks appear. One card per mesh (lock protocol,
# at most 2 held), released by a trap.
source /data/relcfd/chayo/physmorph_v2/tmp/d146/cache_env.sh
B=/data/relcfd/chayo/physmorph_v2; L=$B/gpu_locks; RS=$B/repo_d151; G=$B/output/gpu; D=$G/d151; T=$B/tmp
S=$D/status.txt
echo "fix: frames12 from the full archives, then the post-run block (d151_fix.sh) $(date '+%m-%d %H:%M')" >> $S
(
  source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null
  for name in dragon300k_NL dragon300k_NP bunny300k_NL bunny300k_NP; do
    [ -f $D/${name}_frames12.npz ] || CUDA_VISIBLE_DEVICES="" $PY $T/d151/frames12_from_archive.py \
      $D/${name}_render_full_dt_iso_nn.npz $D/${name}_frames12.npz >> $T/d151/fix.log 2>&1
    echo "frames12 $name rc $? $(date '+%m-%d %H:%M')" >> $S
  done
)
post_mesh() {  # post_mesh MESH: claim a card, then the chain's post-run block for MESH300k_NL and MESH300k_NP
  local mesh=$1 gpu="" name F IND box last
  while [ -z "$gpu" ]; do
    if [ "$(grep -l "^agent D151" $L/gpu*.lock/info 2>/dev/null | wc -l)" -lt 2 ]; then
      for g in 3 2 1 0; do
        if [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)" -lt 2000 ] && mkdir $L/gpu$g.lock 2>/dev/null; then
          gpu=$g; break
        fi
      done
    fi
    [ -z "$gpu" ] && sleep 30
  done
  trap "rm -rf $L/gpu$gpu.lock" EXIT
  printf "agent D151\nrun d151/%s post-run block (fix: frames12 from the archives)\npid %s\nstart %s\n" "$mesh" "$BASHPID" \
    "$(date '+%Y-%m-%d %H:%M %Z')" > $L/gpu$gpu.lock/info
  echo "fix: holding gpu $gpu for $mesh $(date '+%m-%d %H:%M')" >> $S
  IND=$G/d90/${mesh}_ind300k.npz
  box=1900,100,2800,700; [ $mesh = dragon ] && box=1900,60,2500,560
  for arm in NL NP; do
    name=${mesh}300k_$arm; F=$D/${name}_frames12.npz
    [ -f $F ] || { echo "fix: no $F, $name skipped $(date '+%m-%d %H:%M')" >> $S; continue; }
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
      rm -rf $W/$r
      t0=$(date +%s)
      CUDA_VISIBLE_DEVICES=$gpu $PY scripts/render_deliverable.py $F $DN $W/$r +stride=1 +video=1 +fit=0 +field=zbs +az=90 > $W/$r.log 2>&1 < /dev/null
      ffmpeg -loglevel error -y -framerate 20 -i $W/$r/frames/%04d.jpg -c:v libx264 -crf 18 -pix_fmt yuv420p $D/$r.mp4
      echo "render $r rc=$? $(( $(date +%s) - t0 )) s gpu $gpu $(date '+%m-%d %H:%M')" >> $S
      $PY $T/d137/d140_image.py $W/$r/frames $F 90 $D/img_$r > $D/img_$r.log 2>&1 < /dev/null
    )
  done
  wait
  echo "fix: $mesh done, released gpu $gpu $(date '+%m-%d %H:%M')" >> $S
}
post_mesh dragon &
post_mesh bunny &
wait
echo "fix done $(date '+%m-%d %H:%M')" >> $S
