#!/bin/bash
# sc_hflip_done.sh -- SHOWCASE (the coordinator, 2026-10-22 00:00): the studio raster draws a mirror image (StudioRaster:
# right = up x forward, then image.flip(0)), so the showcase's videos, stills and sheets are mirrored back at the encoding
# step (display only; the loss reads target and body through the same operator). sc_chain7.sh does it for new renders;
# this pass does it once for the renders encoded before (every video without its NAME_VIEW.hflip marker): both videos
# re-encoded from the mp4 with hflip (crf 16), the stills flipped in place, the sheet rebuilt from the stills (sc_sheet.py,
# the camera of the render's status line), then the marker. CPU only, nice 10; skips an item whose render is running.
B=/data/relcfd/chayo/physmorph_v2; D=$B/output/gpu/showcase; T=$B/tmp/showcase; V=$D/video; R=$D/render
source $B/repo_t1s/scripts/ops/hyde06_env.sh > /dev/null 2>&1
for v4 in $V/*_4k.mp4; do
  nv=$(basename $v4 _4k.mp4); [ -s $v4 ] || continue
  [ -e $V/$nv.hflip ] && continue
  [ -d $R/$nv/frames ] && { echo "skip $nv (render running)"; continue; }
  [ -s $V/${nv}_1080p.mp4 ] || continue
  for q in 4k 1080p; do
    nice -n 10 ffmpeg -loglevel error -y -i $V/${nv}_$q.mp4 -vf hflip -c:v libx264 -crf 16 -pix_fmt yuv420p $V/${nv}_$q.tmp.mp4 < /dev/null \
      && mv $V/${nv}_$q.tmp.mp4 $V/${nv}_$q.mp4
  done
  for s in $D/stills/$nv/*.jpg; do
    nice -n 10 ffmpeg -loglevel error -y -i $s -vf hflip -q:v 2 $s.tmp.jpg < /dev/null && mv $s.tmp.jpg $s
  done
  name=${nv%_*}; view=${nv##*_}
  cam=$(grep "render ${nv} az" $D/status.txt | tail -1 | sed 's/.* az \([^ ]*\) el \([^ ]*\).*/\1 \2/')
  az=${cam% *}; el=${cam#* }
  ks=$(ls $D/stills/$nv | sed 's/.jpg//' | sort -n | awk -v m="$($PY -c "import json; print(json.load(open('$D/mid/$name.json'))['mid'])" 2>/dev/null)" '$1+0 != m+0 || $1+0==0 || $1+0==5 || $1+0==10 || $1+0==15 || $1+0==20 || $1+0==30 || $1+0==50 {printf "%s%d", (n++ ? "," : ""), $1}')
  nice -n 10 $PY $T/sc_sheet.py $D/sheets/${nv}_sheet.jpg "$(cat $D/title_$name) ($view, az $az, el $el)" $D/stills/$nv $ks 4 960 > /dev/null 2>&1
  touch $V/$nv.hflip
  echo "hflip $nv (az $az el $el; sheet frames $ks)"
done
