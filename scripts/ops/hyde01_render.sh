#!/bin/bash
# Draw a run's kept frames on hyde01 (2 x RTX A6000) instead of a hyde06 card. Run it on the LOCAL machine (Git Bash):
# hyde01's key is not on hyde06, so the files are relayed hyde06 -> this machine -> hyde01 in one tar stream (24-28 MB/s
# on campus, 9 MB/s once from a cold hyde06 disk: a 1.3 GB 300k frames file in ~50 s). hyde01 renders with the same code
# and versions as hyde06 (~/physmorph_render: torch 2.8.0+cu128, warp 1.16.0, cupy 14.0.0, diff_gauss rebuilt for sm_86;
# frames byte-identical to hyde06's), about 2.8x slower per frame than an RTX 6000 Ada (dragon 300k view: 2375 s against
# 861 s), deletes the frames, and the outputs come back to DEST on hyde06 (and --local). Two hyde01 connections per
# job: upload + render + wait (one long session), then fetch. The local machine must stay up until the job returns.
#
#   bash scripts/ops/hyde01_render.sh --npz HYDE06_NPZ --dest HYDE06_DIR --name NAME [options]
#     --views "side:+az=90 front:+az=35"   one mp4 per view, NAME_VIEW.mp4; a view's own + options follow ':' (joined
#                                          by ','), e.g. "close:+az=90,+look=0.3,0,0,+dist=0.5" (default side:+az=90)
#     --opts "+stride=1 +video=1 +fit=0 +field=zbs"   + options for every view (deliverable default; --scene default
#                                          "+stride=12 +field=zbs")
#     --mesh dragon        the dense samples output/gpu/d111/MESH_dense15{,_s9,_s10,_s11}.npz (default: the npz's name up
#                          to its first digit); --dense "a.npz,b.npz,..." other hyde06 paths, denses[0] first
#     --scene              a two-body scene archive (scripts/render_scene.py; no dense samples; default view "main:")
#     --fps 20             the mp4's frame rate
#     --repo HYDE06_DIR    ship that hyde06 checkout's physmorph/ and scripts/ with the job (default: hyde01's
#                          ~/physmorph_render/repo, commit e91b2a2)
#     --local DIR          also put the outputs in this local folder
#     --keep-frames        also return the jpg frames (DEST/NAME_VIEW/frames/*.jpg)
#     --gpus "1"           hyde01 GPUs the job may take (GPU 0 also drives another user's desktop session)
# DEST gets NAME_VIEW.mp4, NAME_VIEW.log, NAME_VIEW.rows.txt and NAME.hyde01_status; the script prints one
# "render NAME_VIEW rc=.. ffmpeg=.. frames=.. N s hyde01 gpu G" line per view and exits 0 when every view rendered.
set -u
H1=chayo@hyde01.dabh.io
SSHO="-o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=10"
G=/data/relcfd/chayo/physmorph_v2/output/gpu/d111
CACHED_DENSE="dragon bunny"                    # meshes whose dense samples hyde01 already holds
NPZ="" DEST="" NAME="" VIEWS="" OPTS="" MESH="" DENSE="" KIND=deliverable FPS=20 REPO="" LOCAL="" KEEP=0 GPUS=1
while [ $# -gt 0 ]; do
  case $1 in
    --npz) NPZ=$2; shift 2 ;;
    --dest) DEST=$2; shift 2 ;;
    --name) NAME=$2; shift 2 ;;
    --views) VIEWS=$2; shift 2 ;;
    --opts) OPTS=$2; shift 2 ;;
    --mesh) MESH=$2; shift 2 ;;
    --dense) DENSE=$2; shift 2 ;;
    --scene) KIND=scene; shift ;;
    --fps) FPS=$2; shift 2 ;;
    --repo) REPO=$2; shift 2 ;;
    --local) LOCAL=$2; shift 2 ;;
    --keep-frames) KEEP=1; shift ;;
    --gpus) GPUS=$2; shift 2 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
[ -n "$NPZ" ] && [ -n "$DEST" ] && [ -n "$NAME" ] || { sed -n 2,26p "$0"; exit 2; }
if [ $KIND = scene ]; then
  VIEWS=${VIEWS:-main:}; OPTS=${OPTS:-+stride=12 +field=zbs}
else
  VIEWS=${VIEWS:-side:+az=90}; OPTS=${OPTS:-+stride=1 +video=1 +fit=0 +field=zbs}
fi
# hyde06: direct on the campus network, else through hyde01's socat alias
if ssh -o ConnectTimeout=6 $SSHO hyde06d true 2>/dev/null; then H6=hyde06d; else H6=hyde06j; fi
# the members of the upload (tar -C dir name pairs) and the dense samples' names
members="-C $(dirname $NPZ) $(basename $NPZ)"
dense_bases=""
if [ $KIND = deliverable ]; then
  if [ -z "$DENSE" ]; then
    MESH=${MESH:-$(basename $NPZ | sed 's/[0-9].*//')}
    DENSE="$G/${MESH}_dense15.npz,$G/${MESH}_dense15_s9.npz,$G/${MESH}_dense15_s10.npz,$G/${MESH}_dense15_s11.npz"
    case " $CACHED_DENSE " in *" $MESH "*) ship_dense=0 ;; *) ship_dense=1 ;; esac
  else
    ship_dense=1
  fi
  for d in ${DENSE//,/ }; do
    dense_bases="$dense_bases${dense_bases:+,}$(basename $d)"
    [ $ship_dense = 1 ] && members="$members -C $(dirname $d) $(basename $d)"
  done
fi
SHIP_REPO=0
[ -n "$REPO" ] && { SHIP_REPO=1; members="$members -C $REPO physmorph scripts"; }
NPZ_SIZE=$(ssh $SSHO $H6 "stat -c %s $NPZ") || { echo "cannot read $NPZ on hyde06"; exit 1; }
JOB=${NAME}_$(date +%m%d_%H%M%S)_$$
ENV64=$(printf 'KIND=%q\nNAME=%q\nVIEWS=%q\nOPTS=%q\nFPS=%q\nGPUS=%q\nKEEP_FRAMES=%q\nNPZ_BASE=%q\nNPZ_SIZE=%q\nDENSE_BASES=%q\nSHIP_REPO=%q\n' \
  "$KIND" "$NAME" "$VIEWS" "$OPTS" "$FPS" "$GPUS" "$KEEP" "$(basename $NPZ)" "$NPZ_SIZE" "$dense_bases" "$SHIP_REPO" | base64 -w0)
TMPD=${TMPDIR:-/tmp}/hyde01_render_$JOB
mkdir -p $TMPD
echo "job $JOB: $(basename $NPZ) ($(( NPZ_SIZE / 1000000 )) MB) via $H6 -> hyde01, views [$VIEWS] opts [$OPTS]"
launched=0
for attempt in 1 2; do
  ssh $SSHO $H6 "tar cf - --exclude=__pycache__ $members" \
    | ssh $SSHO $H1 "bash ~/physmorph_render/bin/render_job.sh submit $JOB $ENV64" > $TMPD/session.log 2>&1
  rc=$?
  grep -q "^LAUNCHED" $TMPD/session.log && launched=1
  [ $launched = 1 ] && break
  cat $TMPD/session.log
  [ $rc = 4 ] || [ $rc = 5 ] && exit $rc
  echo "upload attempt $attempt failed (rc $rc); retrying in 150 s"
  sleep 150
done
[ $launched = 1 ] || { echo "job $JOB not launched"; exit 1; }
for attempt in 1 2 3 4 5; do                   # a dropped session (rc 255): the job runs on; wait again
  [ $rc != 255 ] && break
  sleep 150
  ssh $SSHO $H1 "bash ~/physmorph_render/bin/render_job.sh wait $JOB" > $TMPD/session.log 2>&1
  rc=$?
done
grep -E "^(upload|render)" $TMPD/session.log
ssh $SSHO $H1 "bash ~/physmorph_render/bin/render_job.sh fetch $JOB" > $TMPD/out.tar || { echo "fetch failed; outputs stay in hyde01:~/physmorph_render/queue/$JOB"; exit 1; }
ssh $SSHO $H6 "mkdir -p $DEST && tar xf - -C $DEST && \
  for t in \$(ls $DEST/${NAME}_*_frames.tar 2>/dev/null); do d=\${t%_frames.tar}; mkdir -p \$d && tar xf \$t -C \$d && rm \$t; done" < $TMPD/out.tar \
  || { echo "copy to hyde06 $DEST failed; the tar is $TMPD/out.tar"; exit 1; }
if [ -n "$LOCAL" ]; then
  mkdir -p "$LOCAL" && tar xf $TMPD/out.tar -C "$LOCAL"
  for t in $(ls "$LOCAL"/${NAME}_*_frames.tar 2>/dev/null); do d=${t%_frames.tar}; mkdir -p $d && tar xf $t -C $d && rm $t; done
fi
rm -rf $TMPD
echo "outputs in $H6:$DEST${LOCAL:+ and $LOCAL}"
exit $rc
