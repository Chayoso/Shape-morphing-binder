#!/bin/bash
# post_show.sh -- agent D175, CPU only (detached by the agent after job 209; PID in output/gpu/d175/deliver/show300/post.pid):
# the showcase deliverables in the studio look: the keyframe sheet (the keyframe | twin | full at t_k) and the twin | full
# video (4k + 1080p), then the frames removed (logged in output/gpu/cleanup.log).
J=post_show; TITLE="D175 showcase deliverables"
source /data/relcfd/chayo/physmorph_v2/tmp/d175/job_common.sh
TK=${TK:-1.0}; FK=$(awk "BEGIN{print int($TK * 240 + 0.5)}")
DL=$O/deliver/show300; mkdir -p $DL; F=$O/runs/show_S_97; T=$O/runs/show_T_97
envr repo_d166; export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
stop=$($PY -c "import json; a=next(iter(json.load(open('$F.json'))['arms'].values())); print(max(h['frame_end'] for h in a['history'] if h.get('frame_end') and not h.get('settle')) - 1)")
log "start: sheet + video (stop raw $stop, t_k frame $FK)"
nice -n 10 $PY $W/d175_video.py sheet $DL/keyframe_sheet.png $A/kf_S300.npz ${T}_frames12.npz ${F}_frames12.npz $FK > $DL/sheet.log 2>&1
log "sheet exit $?"
nice -n 10 $PY $W/d175_video.py sbs $DL/frames $A/kf_S300.npz ${T}_frames12.npz ${F}_frames12.npz $FK $stop 3840 2160 0.0 0.18 160 24 > $DL/sbs.log 2>&1
log "frames exit $?"
ffmpeg -loglevel error -y -framerate 20 -i $DL/frames/f%05d.jpg -c:v libx264 -crf 18 -pix_fmt yuv420p $DL/show300k_twin_full_4k.mp4 < /dev/null
ffmpeg -loglevel error -y -framerate 20 -i $DL/frames/f%05d.jpg -vf scale=1920:1080:flags=lanczos -c:v libx264 -crf 18 \
  -pix_fmt yuv420p $DL/show300k_twin_full_1080p.mp4 < /dev/null
if [ -s $DL/show300k_twin_full_4k.mp4 ] && [ -s $DL/show300k_twin_full_1080p.mp4 ]; then
  mkdir -p $DL/stills; for f in $(ls $DL/frames | awk 'NR % 40 == 1'); do cp $DL/frames/$f $DL/stills/; done
  sz=$(du -sm $DL/frames | cut -f1); rm -rf $DL/frames
  echo "$(date '+%Y-%m-%d %H:%M') D175: removed $DL/frames ($sz MB; the showcase video's frames, both videos and stills written)" >> $G/cleanup.log
fi
log "video done"
