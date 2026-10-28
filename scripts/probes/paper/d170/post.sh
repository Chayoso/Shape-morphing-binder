#!/bin/bash
# post.sh MODE JOB -- D170: what a job's verdict arms (run by after.sh, detached, CPU):
#   carve_a / carve_b / wrinkle / dragon*: deliverables (output/gpu/d170/deliver/<scene>/): raking-light (84 deg) close-ups
#     for every scene whose content passes; for every PASS scene the twin | full video through the settle (4K + 1080p, 20 fps;
#     d163_render's true camera, so no flip) and studio stills of both ends (the default look, rake 72).
#   dragon40: if an arm keeps share >= 0.40 at both seeds, queue job 177 (the dragon at 300k, seed 97: that arm and the twin).
#   carve_a: if the plaque passes, arm the 500k stamp redux (job 179; tmp/d170/179_d170_plq500k.sh when written).
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d170; O=$B/output/gpu/d170; G=$B/output/gpu; Q=$B/tmp/gpu1_queue
MODE=$1; JOB=$2; D=$O/auto/$JOB; DL=$O/deliver
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=4 D157_THREADS=8 MPLCONFIGDIR=$B/cache/matplotlib
export PYTHONPATH=$B/repo_d166
log() { echo "$(date '+%m-%d %H:%M:%S') [post $JOB] $*" >> $O/status.txt; }
cd $B/repo_d166
# one line per scene: KEY STATUS CONTENT_OK FULL TWIN STOP_RAW
$PY - "$D/verdict.json" > $D/post_list.txt << 'EOP'
import json, sys
V = json.load(open(sys.argv[1]))["arms"]
for k, v in V.items():
    if v.get("status") in (None, "INCOMPLETE") or "full" not in v:
        continue
    stop = v.get("stop_raw")
    if stop is None:
        import numpy as np
        stop = -1
    print(k, v["status"], int(bool(v.get("content_ok"))), v["full"], v["twin"], stop)
EOP
declare -A BASE=( [plq]=$G/d163/assets/d163plaque.obj [rip]=$G/d161/assets/d161smooth_bunny.obj [rel]=$G/d161/assets/d161smooth_bunny.obj
                  [coin]=$G/d161/assets/d161coin.obj )
declare -A REL=( [plq]=$G/d163/assets/d163R300plaque.obj [rip]=$G/d162f/assets/d162fR300ripple_bunny.obj [rel]=$O/assets/d170L300_bunny.obj
                 [coin]=$O/assets/d170L300coin.obj )
declare -A VIEW=( [plq]="0.30 0.18" [rip]="1.5708 0.10" [rel]="1.5708 0.10" [coin]="0.0 0.18" )
deliver() {   # KEY FULL TWIN STOP STATUS
  local k=$1 full=$2 twin=$3 stop=$4 st=$5 base rel look out
  case $k in
    plq|rip|rel|coin) base=${BASE[$k]}; rel=${REL[$k]}; look=auto; [ $k = plq ] && look=plaque ;;
    *) return 0 ;;     # the detail scenes: close-ups / videos by deliver_detail
  esac
  [ $stop -lt 0 ] && stop=$($PY -c "import json; a=next(iter(json.load(open('$full.json'))['arms'].values())); print(max(h['frame_end'] for h in a['history'] if h.get('frame_end') and not h.get('settle')) - 1)")
  out=$DL/$k; mkdir -p $out/stills
  log "$k: close-ups (rake 84)"
  nice -n 10 $PY $W/d170_render.py close $out/close $base 300000 $rel ${twin}_frames12.npz ${full}_frames12.npz $stop $look auto 1.0 84 \
    > $out/close.log 2>&1; log "$k: close-ups exit $?"
  [ "$st" = PASS ] || return 0
  set -- ${VIEW[$k]}
  for side in full twin; do p=$full; [ $side = twin ] && p=$twin
    nice -n 10 $PY $W/d170_render.py still $out/stills/${side}_end.png $base 300000 ${p}_frames12.npz -1 $1 $2 3840 2160 72 \
      >> $out/stills.log 2>&1; done
  log "$k: twin | full video"
  s=$(date +%s)
  nice -n 10 $PY $W/d170_render.py sbs $out/frames $base 300000 ${twin}_frames12.npz ${full}_frames12.npz $stop 3840 2160 $1 $2 160 24 72 \
    > $out/sbs.log 2>&1; log "$k: frames exit $? ($(( $(date +%s) - s )) s)"
  ffmpeg -loglevel error -y -framerate 20 -i $out/frames/f%05d.jpg -c:v libx264 -crf 18 -pix_fmt yuv420p $out/${k}300k_twin_full_4k.mp4 < /dev/null
  ffmpeg -loglevel error -y -framerate 20 -i $out/frames/f%05d.jpg -vf scale=1920:1080:flags=lanczos -c:v libx264 -crf 18 \
    -pix_fmt yuv420p $out/${k}300k_twin_full_1080p.mp4 < /dev/null
  if [ -s $out/${k}300k_twin_full_4k.mp4 ] && [ -s $out/${k}300k_twin_full_1080p.mp4 ]; then
    mkdir -p $out/stills; n=$(ls $out/frames | wc -l)
    for f in f00000 f00040 f00080; do [ -e $out/frames/$f.jpg ] && cp $out/frames/$f.jpg $out/stills/sbs_$f.jpg; done
    cp $out/frames/$(printf f%05d $((n - 181))).jpg $out/stills/sbs_end.jpg 2>/dev/null
    sz=$(du -sm $out/frames | cut -f1); rm -rf $out/frames
    echo "$(date '+%Y-%m-%d %H:%M') D170: removed $out/frames ($sz MB; the frames of the $k twin | full video, both videos written, stills kept)" >> $G/cleanup.log
  fi
  log "$k: video done"
}
deliver_detail() {   # KEY FULL TWIN STATUS: the dragon's scales / the bunny's wrinkles (no pattern: D159's close-up cameras)
  local k=$1 full=$2 twin=$3 st=$4 base rel n=300000 out stop
  case $k in
    dsm40_*) base=$O/assets/d170smooth_dragon.obj; rel=$B/repo_d166/assets/dragon.obj; n=40000 ;;
    dsm300_*) base=$O/assets/d170smooth_dragon.obj; rel=$B/repo_d166/assets/dragon.obj ;;
    wrkd_*) base=$B/repo_d166/assets/dragon.obj; rel=$base ;;
    wrkb_*) base=$B/repo_d166/assets/bunny.obj; rel=$base ;;
    *) return 0 ;;
  esac
  stop=$($PY -c "import json; a=next(iter(json.load(open('$full.json'))['arms'].values())); print(max(h['frame_end'] for h in a['history'] if h.get('frame_end') and not h.get('settle')) - 1)")
  out=$DL/$k; mkdir -p $out
  log "$k: close-ups (rake 84)"
  nice -n 10 $PY $W/d170_render.py close $out/close_body $base $n $rel ${twin}_frames12.npz ${full}_frames12.npz $stop rel:0,0,0 0.30,0.18 1.0 84 \
    > $out/close.log 2>&1
  case $k in
    wrkb_*) nice -n 10 $PY $W/d170_render.py close $out/close_haunch $base $n $rel ${twin}_frames12.npz ${full}_frames12.npz $stop rel:0,0,0 1.449,0.0 2.0 84 >> $out/close.log 2>&1 ;;
    *) nice -n 10 $PY $W/d170_render.py close $out/close_scutes $base $n $rel ${twin}_frames12.npz ${full}_frames12.npz $stop rel:0.29,-0.299,0.29 2.18,0.087 3.3 84 >> $out/close.log 2>&1 ;;
  esac
  log "$k: close-ups exit $?"
  [ "$st" = PASS ] && [ $n = 300000 ] || return 0
  log "$k: twin | full video"
  nice -n 10 $PY $W/d170_render.py sbs $out/frames $base $n ${twin}_frames12.npz ${full}_frames12.npz $stop 3840 2160 0.30 0.18 160 24 72 > $out/sbs.log 2>&1
  ffmpeg -loglevel error -y -framerate 20 -i $out/frames/f%05d.jpg -c:v libx264 -crf 18 -pix_fmt yuv420p $out/${k}_twin_full_4k.mp4 < /dev/null
  ffmpeg -loglevel error -y -framerate 20 -i $out/frames/f%05d.jpg -vf scale=1920:1080:flags=lanczos -c:v libx264 -crf 18 \
    -pix_fmt yuv420p $out/${k}_twin_full_1080p.mp4 < /dev/null
  if [ -s $out/${k}_twin_full_4k.mp4 ] && [ -s $out/${k}_twin_full_1080p.mp4 ]; then
    sz=$(du -sm $out/frames | cut -f1); rm -rf $out/frames
    echo "$(date '+%Y-%m-%d %H:%M') D170: removed $out/frames ($sz MB; the frames of the $k twin | full video, both videos written)" >> $G/cleanup.log
  fi
  log "$k: video done"
}
while read k st ok full twin stop; do
  [ "$ok" = 1 ] || [ "$st" = PASS ] || [[ $k == dsm*_97 ]] || { log "$k: content $st, no deliverable"; continue; }
  case $k in
    plq|rip|rel|coin) deliver $k $full $twin $stop $st ;;
    *) deliver_detail $k $full $twin $st ;;
  esac
done < $D/post_list.txt
if [ "$MODE" = dragon40 ]; then
  arm=$($PY - "$D/verdict.json" "$D/read_dsm40.json" << 'EOP'
import json, sys
V = json.load(open(sys.argv[1]))["arms"]
R = json.load(open(sys.argv[2]))["refs"]
noise = {}
for x in (97, 99):   # the reading's own resolution: the spread of the detailed mesh's two N-samples, in share units
    a1, a2, f = (R[f"{k}{x}"]["relief"]["A"] for k in ("detail_s1", "detail_s2", "smooth_s2"))
    noise[x] = abs(a1 - a2) / max(abs(0.5 * (a1 + a2) - f), 1e-9)
best = None
for a in ("Rxm", "Rx", "H0m", "H0", "R0m"):
    v = [V.get(f"dsm40_{a}_{x}", {}) for x in (97, 99)]
    ok = all(q.get("share_end") is not None and q["share_end"] >= 0.40 and q["share_end"] - q["twin_share"] >= 2 * noise[x]
             for q, x in zip(v, (97, 99)))
    if ok and (best is None or min(q["share_end"] for q in v) > best[1]):
        best = (a, min(q["share_end"] for q in v))
print(best[0] if best else "", file=sys.stdout)
print("noise " + " ".join(f"{x}:{n:.2f}" for x, n in noise.items()), file=sys.stderr)
EOP
)
  if [ -n "$arm" ]; then
    log "dragon 40k: arm $arm keeps share >= 0.40 and above the twin by twice the reading's noise at both seeds: queue job 177 (300k)"
    case $arm in
      Rx) fl='$RX' ;; Rxm) fl='$RX --render_target mesh_shell' ;; H0) fl='--lambda_ema 0 $OFF' ;;
      H0m) fl='--lambda_ema 0 $OFF --render_target mesh_shell' ;; R0m) fl='$OFF --render_target mesh_shell' ;;
    esac
    cat > $W/177_d170_dragon300.sh << EOJ
#!/bin/bash
# 177_d170_dragon300.sh -- queue job (agent D170, armed by post.sh after the 40k read): the dragon with render-only relief at
# 300k, seed 97: arm $arm (the best 40k arm with share >= 0.40 at both seeds) and the twin, on repo_d170.
J=177_dragon300; TITLE="D170: dragon render-only relief at 300k"
source $W/job_common.sh
REPO=repo_d170; PAR=2; OFF="--settle_windows 30 --no-settle_relax --no-layer_relax"
DR="--tgt \$A/d170smooth_dragon.obj --render_tgt assets/dragon.obj --n 300000 --seed 97"
log "start (arm $arm)"
runp dsm300k_T_97 \$DR \$TW
runp dsm300k_${arm}_97 \$DR $fl
wait; log "done"
EOJ
    cp $W/177_d170_dragon300.sh $Q/177_d170_dragon300.sh.tmp && mv $Q/177_d170_dragon300.sh.tmp $Q/177_d170_dragon300.sh
    mkdir -p $O/auto/177_dragon300
    D170_DSM300_ARMS="T $arm" setsid nohup bash $W/after2.sh dragon300 177_dragon300 177_d170_dragon300 \
      > $O/auto/177_dragon300/after.out 2>&1 < /dev/null & echo $! > $O/auto/177_dragon300/after.pid
  else
    log "dragon 40k: no arm clears share >= 0.40 and the twin + twice the reading's noise (the spread of the detailed mesh's own samples) at both seeds: no 300k run (job 177 not queued)"
  fi
fi
if [ "$MODE" = carve_a ]; then
  st=$(awk '$1 == "plq" {print $2}' $D/post_list.txt)
  if [ "$st" = PASS ] && [ -e $W/179_d170_plq500k.sh ]; then
    cp $W/179_d170_plq500k.sh $Q/179_d170_plq500k.sh.tmp && mv $Q/179_d170_plq500k.sh.tmp $Q/179_d170_plq500k.sh
    log "plaque PASS: the 500k stamp redux queued (job 179)"
  else
    log "plaque ${st:-unread}: the 500k stamp redux not armed"
  fi
fi
if [ "$MODE" = band300 ] && [ "$(awk '$1 == "band" {print $2}' $D/post_list.txt)" = PASS ]; then
  cp $W/182_d170_band_video.sh $Q/182_d170_band_video.sh.tmp && mv $Q/182_d170_band_video.sh.tmp $Q/182_d170_band_video.sh
  log "band PASS: the colour video queued (job 182)"
fi
if [ "$MODE" = gal ]; then
  setsid nohup bash $W/d157_gal.sh > $O/auto/d157_gal.out 2>&1 < /dev/null &
  log "D157 gallery reader started (pid $!)"
fi
log "post done"
