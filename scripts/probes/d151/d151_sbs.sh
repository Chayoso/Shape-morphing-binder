#!/bin/bash
# d151_sbs.sh -- D151's side-by-side videos, base | layer_link, full length: D149's SL (base, the refreeze-cand render arm)
# beside NL (--layer_link), the side view at 20 frames a simulated second, each 1920 x 1080 and labelled with the
# simulated time; the shorter run held on its last frame. Also the twins (SP | NP). Output in output/gpu/d151/sbs.
D=/data/relcfd/chayo/physmorph_v2/output/gpu/d151; O=$D/sbs; mkdir -p $O
dur() { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1"; }
for mesh in dragon bunny; do
  for pair in "SL NL" "SP NP"; do
    set -- $pair; a=$D/${mesh}300k_$1_side.mp4; b=$D/${mesh}300k_$2_side.mp4
    [ -f "$a" ] && [ -f "$b" ] || { echo "missing $a or $b"; continue; }
    da=$(dur "$a"); db=$(dur "$b")
    pb=$(awk "BEGIN{d=$da-$db; print (d>0)?d:0}"); pa=$(awk "BEGIN{d=$db-$da; print (d>0)?d:0}")
    la="base (refreeze-cand, D149 $1)"; lb="--layer_link (D151 $2)"
    ffmpeg -loglevel error -y -i "$a" -i "$b" -filter_complex \
      "[0:v]scale=1920:1080,tpad=stop_mode=clone:stop_duration=$pa,drawtext=text='$la':fontsize=36:fontcolor=black:x=16:y=16,drawtext=text='%{pts\:flt} s':fontsize=36:fontcolor=black:x=16:y=60[a];[1:v]scale=1920:1080,tpad=stop_mode=clone:stop_duration=$pb,drawtext=text='$lb':fontsize=36:fontcolor=black:x=16:y=16,drawtext=text='%{pts\:flt} s':fontsize=36:fontcolor=black:x=16:y=60[b];[a][b]hstack=inputs=2" \
      -r 20 -c:v libx264 -crf 20 -pix_fmt yuv420p $O/${mesh}300k_side_$1_vs_$2.mp4
    echo "$mesh $1 vs $2 rc=$? ($da s, $db s)"
  done
done
