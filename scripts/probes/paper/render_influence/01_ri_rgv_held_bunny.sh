#!/bin/bash
# 01_ri_rgv_held_bunny.sh -- gpu1 queue job (agent RI, render influence; the coordinator's order 2026-10-22 19:30):
# RGV's 2 x 2 (A, B2, C, D) for the render arm with its weight held, beside the tag's.
#  1. the run: bunny 300k seed 97 on repo_tgf (= the tag freeze-2026-10-15 + D121's hooks + rgv_patch.py +
#     rgv_patch2.py, the code of the tag's RGV run RG97; rt1 = the tag bit for bit under Warp's deterministic mode),
#     with --lambda_ema 0 (lambda kept at the first window's calibration: the arm that carved the letters and moved
#     colour), the gradient dump and the lambda = 0 counterfactual at every window, exactly RG97's flags otherwise;
#  2. its kept frames (frames12.py), RGV's values (rgv_values.py, CPU), the dumps deleted after (cleanup.log);
#  3. the panels A, B2, C, D at 1080p, front and side, with the 2 x 2 (full and half size), drawn by RGV's renderer in a
#     scratch copy that flips the body left-right before the text (the studio camera mirrors);
#  4. the tag's RG97 drawn again the same way (front), so that "tag | held" match in orientation;
#  5. mp4s at 20 fps (kept frames, real time), the half-size 2 x 2 also at 10 fps. Frames kept for the stills.
B=/data/relcfd/chayo/physmorph_v2
D=$B/output/gpu/rgv_held
N=bunny300k_RH97
T=$B/tmp/render_influence/rgv_held
mkdir -p $D
S=$D/status.txt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $S; }
export PHYSMORPH_RUN_REPO=$B/repo_tgf PHYSMORPH_OUT=$D OMP_NUM_THREADS=8
source $B/repo_tgf/scripts/ops/hyde06_env.sh > /dev/null
source $B/tmp/d146/cache_env.sh
cd $B/repo_tgf
CFW=$(seq -s, 0 399)
if [ ! -e $D/${N}_frames12.npz ]; then
  log "run $N start (gpu1 queue, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
  ( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)"; sleep 10; done ) > $D/$N.mem &
  mem=$!
  t0=$(date +%s)
  $PY scripts/pipeline_run.py --tgt assets/bunny.obj --n 300000 --out $D/$N --seed 97 --lambda_ema 0 \
    --grad_dump $D/${N}_dump --dump_windows 0,+1 --cf_dir $D/${N}_cf --cf_windows $CFW > $D/$N.log 2>&1 < /dev/null
  rc=$?
  kill $mem
  log "run $N rc $rc wall $(( $(date +%s) - t0 )) s peak_mib(card total) $(sort -k2 -n $D/$N.mem | tail -1 | cut -d' ' -f2)"
  [ $rc = 0 ] || exit $rc
  A=$D/${N}_render_full_dt_iso_nn.npz
  $PY $B/tmp/frames12.py $A $D/$N > $D/$N.frames12.log 2>&1
  rc=$?
  log "frames12 rc $rc ($(tail -1 $D/$N.frames12.log))"
  [ $rc = 0 ] || exit $rc
  rm -f $A
fi
if [ ! -e $D/${N}_viz.npz ]; then
  ( export CUDA_VISIBLE_DEVICES="" PYTHONPATH=$B/repo_rgvtag
    nice $PY $B/tmp/rgv_tag/rgv_values.py $D/$N.json $D/${N}_frames12.npz $D/${N}_dump $D/${N}_cf $D/${N}_viz.npz \
      $D/${N}_numbers.json > $D/prep_$N.log 2>&1 )
  rc=$?
  log "values rc $rc"
  [ $rc = 0 ] || exit $rc
  # RGV's own per-window panel numbers (agg, A, C, D, U by window band), the tag's RG97 ("old") against held ("new")
  ( export CUDA_VISIBLE_DEVICES="" PYTHONPATH=$B/repo_rgvtag
    nice $PY $B/tmp/rgv_tag/rgv_tag_check.py $D/tagcheck_tag_vs_held.json $B/output/gpu/rgv_tag/bunny300k_RG97_viz.npz \
      $D/${N}_viz.npz $D/${N}_dump > $D/tagcheck_tag_vs_held.log 2>&1 )
  log "tagcheck rc $?"
  sz=$(du -sm $D/${N}_dump $D/${N}_cf | awk '{s += $1} END {print s}')
  rm -rf $D/${N}_dump $D/${N}_cf
  echo "$(date '+%F %T') rgv_held: deleted $D/${N}_dump and _cf ($sz MB) after rgv_values (values in ${N}_viz.npz)" >> $B/output/gpu/cleanup.log
fi
export PYTHONPATH=$GPU_DEPS:$B/repo_rgvtag
cd $B/repo_rgvtag
render() {   # VIZ OUTD VIEWS GRIDFULL CODE
  rm -rf $2
  mkdir -p $2
  $PY $T/render_gradient_viz_flip.py $1 $B/output/gpu/d111/bunny_dense15.npz $2 +viz=A,B2,C,D +views=$3 +res=1080 \
    +interp=1 +grid=1 +gridfull=$4 +c_avg=5 +d_avg=5 "+code=$5" > $2.log 2>&1
}
render $D/${N}_viz.npz $D/render_held "35,90" 1 "freeze-2026-10-15 + --lambda_ema 0 (render weight held)"
log "render held rc $?"
render $B/output/gpu/rgv_tag/bunny300k_RG97_viz.npz $D/render_tag "35" 1 "freeze-2026-10-15"
log "render tag rc $?"
for pair in "render_held held" "render_tag tag"; do
  set -- $pair
  for d in $D/$1/gridB2_*_1080 $D/$1/gridB2half_*_1080; do
    [ -d $d ] || continue
    n=$(basename $d); k=${n%%_*}; az=$(echo $n | cut -d_ -f2)
    case $az in 35) vn=front ;; 90) vn=side ;; *) vn=az$az ;; esac
    case $k in gridB2) kn=AB2CD_2x2 ;; gridB2half) kn=AB2CD_2x2_half ;; esac
    ffmpeg -loglevel error -y -framerate 20 -pattern_type glob -i "$d/*.jpg" -c:v libx264 -crf 18 -pix_fmt yuv420p \
      $D/bunny300k_${2}_${kn}_${vn}_1080p.mp4 < /dev/null
    log "ffmpeg $2 $n rc $? $(ls $d | wc -l) frames"
    if [ $k = gridB2half ]; then
      ffmpeg -loglevel error -y -framerate 10 -pattern_type glob -i "$d/*.jpg" -c:v libx264 -crf 18 -pix_fmt yuv420p \
        $D/bunny300k_${2}_AB2CD_2x2_halfspeed_${vn}_1080p.mp4 < /dev/null
      log "ffmpeg $2 $n halfspeed rc $?"
    fi
  done
done
log "job 01 done"
