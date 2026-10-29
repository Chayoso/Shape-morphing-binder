#!/bin/bash
# clip.sh GPU NAME ARCHIVE KIND FRAMES "BLENDER_KEYS" -- one demo clip on card GPU: mesh the kept frames (mesh_frames.py,
# four processes on the card), render them in Blender (blender_render.py), encode the mp4 (H.264, 20 fps; KIND morph
# holds the last frame for 1 s), make the 4-panel still (start, two in-betweens, end), then delete the per-frame meshes
# and images (logged in output/gpu/cleanup.log). KIND: morph | scene. FRAMES: a-b kept-frame indices or "all".
# A queue line named HOLD_x parks its runner (and keeps its card) while tmp/demo/hold_HOLD_x exists.
set -u
case $2 in HOLD*) while [ -f /data/relcfd/chayo/physmorph_v2/tmp/demo/hold_$2 ]; do sleep 10; done; exit 0;; esac
gpu=$1; name=$2; arc=$3; kind=$4; frames=$5; bkeys=$6
B=/data/relcfd/chayo/physmorph_v2; D=$B/tmp/demo; S=$D/status.txt
M=$D/mesh/$name; R=$D/frames/$name; O=$D/out
mkdir -p $M $R $O $D/blend $D/logs
log() { echo "$(date '+%m-%d %H:%M:%S') [gpu$gpu] $name $*" >> $S; }
fr=""; [ "$frames" != all ] && fr="+frames=$frames"
bfr=""; [ "$frames" != all ] && bfr="frames=$frames"
t0=$(date +%s)
(
  export PHYSMORPH_RUN_REPO=$B/repo_freeze1015 PHYSMORPH_OUT=$D/out
  source $B/repo_freeze1015/scripts/ops/hyde06_env.sh > /dev/null
  source $B/tmp/d146/cache_env.sh
  for w in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu $PY $D/scripts/mesh_frames.py $arc $M $fr +worker=$w/4 > $D/logs/${name}_mesh$w.log 2>&1 &
  done
  wait
)
t1=$(date +%s)
log "meshed $(ls $M/*.ply | wc -l) surfaces in $((t1 - t0)) s"
source $D/scripts/benv.sh
cd $D
CUDA_VISIBLE_DEVICES=$gpu $BL -b -noaudio --factory-startup --python $D/scripts/blender_render.py -- $M $R $bfr \
  blend=$D/blend/$name.blend $bkeys > $D/logs/${name}_blender.log 2>&1
rc=$?
t2=$(date +%s)
n=$(ls $R/*.png 2>/dev/null | wc -l)
log "rendered $n frames rc $rc in $((t2 - t1)) s ($(( (t2 - t1) / (n > 0 ? n : 1) )) s/frame)"
[ "$n" -gt 0 ] || exit 1
first=$(ls $R/*.png | head -1 | xargs basename | cut -c1-4)
hold=""; [ "$kind" = morph ] && hold="tpad=stop_mode=clone:stop_duration=1,"
ffmpeg -hide_banner -loglevel error -y -framerate 20 -start_number $((10#$first)) -i $R/%04d.png \
  -vf "${hold}format=yuv420p" -c:v libx264 -preset slow -crf 15 -movflags +faststart $O/$name.mp4
log "encoded $O/$name.mp4 rc $? ($(du -h $O/$name.mp4 | cut -f1))"
# the 4-panel still: start, two in-betweens, end. morph: the in-betweens where the body's box has gone 35 % and 70 %
# of its way from the start's to the end's; scene: at a third and two thirds of the clip.
panels=$(source $B/repo_freeze1015/scripts/ops/hyde06_env.sh > /dev/null; $PY - "$M" "$R" "$kind" <<'EOF'
import json, sys, glob, os
import numpy as np
M, R, kind = sys.argv[1:4]
ks = sorted(int(os.path.basename(p)[:4]) for p in glob.glob(f"{R}/*.png"))
if kind == "morph":
    e = {k: np.subtract(*[json.load(open(f"{M}/k{k:04d}_b0.json"))[s] for s in ("hi", "lo")]) for k in ks}
    a, b = e[ks[0]], e[ks[-1]]
    p = {k: np.linalg.norm(e[k] - a) / max(1e-9, np.linalg.norm(e[k] - a) + np.linalg.norm(e[k] - b)) for k in ks}
    pick = [ks[0]] + [min(ks[1:-1], key=lambda k: abs(p[k] - t)) for t in (0.35, 0.7)] + [ks[-1]]
else:
    pick = [ks[0], ks[len(ks) // 3], ks[2 * len(ks) // 3], ks[-1]]
print(" ".join(f"{k:04d}" for k in pick))
EOF
)
log "panels $panels"
ins=""; fl=""; i=0
for k in $panels; do
  ins="$ins -i $R/$k.png"
  if [ "$kind" = morph ]; then fl="$fl[$i:v]crop=2160:2160:840:0,scale=1080:1080[p$i];"; else fl="$fl[$i:v]scale=1920:1080[p$i];"; fi
  cp $R/$k.png $O/${name}_frame$k.png
  i=$((i + 1))
done
ffmpeg -hide_banner -loglevel error -y $ins -filter_complex "${fl}[p0][p1][p2][p3]hstack=inputs=4" $O/${name}_panels.png
ffmpeg -hide_banner -loglevel error -y -i $O/${name}_panels.png -q:v 2 $O/${name}_panels.jpg
log "panels written"
sz=$(du -sh $M $R | cut -f1 | tr '\n' ' ')
rm -rf $M $R
echo "$(date '+%F %T') DEMO: removed the per-frame meshes and frames of $name ($sz) after encoding $O/$name.mp4" >> $B/output/gpu/cleanup.log
log "done in $(( $(date +%s) - t0 )) s; scratch removed ($sz)"
