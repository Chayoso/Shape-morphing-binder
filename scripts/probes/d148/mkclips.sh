#!/bin/bash
# mkclips.sh A_LABEL A_MP4 B_LABEL B_MP4 OUT T0 T1 CROP SCALE -- D148's decision package: a side-by-side close-up clip
# of two side views at a quarter speed with the simulated time stamped (the side videos are 20 frames a simulated second).
la=$1; a=$2; lb=$3; b=$4; out=$5; t0=$6; t1=$7; crop=$8; sc=$9
f=$(mktemp -p /data/relcfd/chayo/physmorph_v2/tmp/d148 fg.XXXX)
cat > "$f" << FG
[0:v]trim=start=$t0:end=$t1,crop=$crop,scale=$sc,drawtext=text='$la':fontsize=28:fontcolor=black:x=12:y=12,drawtext=text='%{pts\\:flt} s':fontsize=28:fontcolor=black:x=12:y=48,setpts=4*(PTS-STARTPTS)[a];
[1:v]trim=start=$t0:end=$t1,crop=$crop,scale=$sc,drawtext=text='$lb':fontsize=28:fontcolor=black:x=12:y=12,drawtext=text='%{pts\\:flt} s':fontsize=28:fontcolor=black:x=12:y=48,setpts=4*(PTS-STARTPTS)[b];
[a][b]hstack=inputs=2
FG
ffmpeg -loglevel error -y -i "$a" -i "$b" -filter_complex_script "$f" -r 20 -c:v libx264 -crf 18 -pix_fmt yuv420p "$out"
rc=$?
rm -f "$f"
echo "$(basename "$out") rc=$rc"
