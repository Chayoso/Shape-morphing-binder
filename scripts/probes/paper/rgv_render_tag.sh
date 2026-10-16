#!/bin/bash
# rgv_render_tag.sh -- RGV's videos on the tag freeze-2026-10-15 for one mesh, run inside rgv_idle.sh's card claim
# (RGV_OUT=output/gpu/rgv_tag RGV_REPO=repo_rgvtag):
#   M=bunny RES=1080 VIEWS=35,90 [KINDS=A,B2,C,D,U] [GRIDFULL=1] [STRIDE=1] [SUB=render] [KEEP=0] bash rgv_render_tag.sh
# The fixed design: A, B2, C (cosine as a 5-window mean), D (5-window mean, +d_avg=5), U; the 2 x 2 A-B2-C-D at 2W x 2H
# (not with GRIDFULL=0, the 4K panels) and its half-size copy. Values: rgv_values.py on D145's post-tag B7 run RG97
# (output/gpu/rgv_tag/<mesh>300k_RG97_viz.npz; every non-render term counted in A and C). repo_rgvtag = the tag + the
# RGV renderer. mp4s at 20 fps (real time); the half-size 2 x 2 also at 10 fps (half speed). Four stills per frame
# folder are kept in OUTD/stills/; each frame folder is deleted once every mp4 made from it holds all its frames
# (ffprobe's packet count), logged in output/gpu/cleanup.log (KEEP=1 keeps the frames, for previews).
M=${M:-bunny}; RES=${RES:-1080}; V=${VIEWS:-35,90}; KINDS=${KINDS:-A,B2,C,D,U}
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/rgv_tag; R=$B/repo_rgvtag
OUTD=$O/${SUB:-render}_${M}_${RES}_az${V//,/-}
rm -rf $OUTD
mkdir -p $OUTD/stills
$PY $R/scripts/render_deliverable.py $O/${M}300k_RG97_viz.npz $B/output/gpu/d111/${M}_dense15.npz $OUTD \
  +viz=$KINDS +views=$V +res=$RES +interp=1 +grid=1 +gridfull=${GRIDFULL:-1} +stride=${STRIDE:-1} \
  +c_avg=5 +d_avg=5 +code=freeze-2026-10-15 > $OUTD.log 2>&1
rc=$?
echo "render $M $RES az $V stride ${STRIDE:-1} rc $rc $(date '+%m-%d %H:%M:%S')" >> $O/status.txt
[ $rc = 0 ] || exit $rc
rp=${RES}p; [ $RES = 2160 ] && rp=4K
packets() { ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "$1"; }
for d in $OUTD/*_*_$RES; do
  n=$(basename $d); k=${n%%_*}; az=$(echo $n | cut -d_ -f2)
  case $k in A) kn=A_where_render_acts ;; B) kn=B_how_strong_render_pull ;; C) kn=C_render_vs_physics_disagree ;;
    D) kn=D_what_render_did_to_dFc ;; U) kn=U_render_changes_u ;; grid) kn=ABCD_2x2 ;; gridhalf) kn=ABCD_2x2_half ;;
    B2) kn=B2_render_pull_vs_typical ;; gridB2) kn=AB2CD_2x2 ;; gridB2half) kn=AB2CD_2x2_half ;; esac
  case $az in 35) vn=front ;; 90) vn=side ;; *) vn=az$az ;; esac
  nf=$(ls $d | wc -l)
  mp4s="$OUTD/${M}300k_${kn}_${vn}_${rp}.mp4"
  ffmpeg -loglevel error -y -framerate 20 -pattern_type glob -i "$d/*.jpg" -c:v libx264 -crf 18 -pix_fmt yuv420p \
    $OUTD/${M}300k_${kn}_${vn}_${rp}.mp4 < /dev/null
  echo "ffmpeg $n rc $? $nf frames" >> $O/status.txt
  if [ $k = gridB2half ]; then                         # the half-size 2 x 2 also at half speed
    hs=$OUTD/${M}300k_AB2CD_2x2_halfspeed_${vn}_${rp}.mp4
    ffmpeg -loglevel error -y -framerate 10 -pattern_type glob -i "$d/*.jpg" -c:v libx264 -crf 18 -pix_fmt yuv420p \
      $hs < /dev/null
    echo "ffmpeg $n halfspeed rc $? $nf frames" >> $O/status.txt
    mp4s="$mp4s $hs"
  fi
  ls $d/*.jpg | awk -v n=$nf 'NR == int(.06 * n) + 1 || NR == int(.25 * n) + 1 || NR == int(.5 * n) + 1 || NR == n' |
    while read f; do cp $f $OUTD/stills/${n}_$(basename $f); done
  ok=1
  for f in $mp4s; do [ "$(packets $f 2>/dev/null)" = "$nf" ] || ok=0; done
  if [ "${KEEP:-0}" = 0 ] && [ $ok = 1 ]; then
    sz=$(du -sh $d | cut -f1)
    rm -rf $d
    echo "$(date '+%Y-%m-%d %H:%M') RGV: removed $d ($sz) -- RGV tag render frames, joined into $(basename -a $mp4s | tr '\n' ' ')($nf frames each, checked)" >> $B/output/gpu/cleanup.log
  elif [ $ok = 0 ]; then
    echo "frames kept: $n (an mp4 does not hold all $nf frames)" >> $O/status.txt
  fi
done
