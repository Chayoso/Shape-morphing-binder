#!/bin/bash
# after_show.sh -- agent D175, CPU only, detached (PID in output/gpu/d175/deliver/show300/after.pid): waits for the 300k
# showcase job (201_show300) to finish its reader, then runs post_show.sh (the keyframe sheet and the twin | full video,
# 1080p) and writes output/gpu/D175_SHOWCASE.md (the reader's table, KF-IoU / 3D IoU / the per-block render share, the
# deliverables' paths). Gives up after 6 h.
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/d175; DL=$O/deliver/show300; mkdir -p $DL
for k in $(seq 1 720); do grep -q "\[201_show300\] reader exit" $O/status.txt 2>/dev/null && break; sleep 30; done
grep -q "\[201_show300\] reader exit" $O/status.txt || { echo "$(date '+%m-%d %H:%M:%S') [after_show] no reader after 6 h" >> $O/status.txt; exit 1; }
bash $B/tmp/d175/post_show.sh
S=$O/auto/201_show300/summary.md
{ echo "# D175 showcase: bunny -> the bunny stretched 1.3x tall at t = 1 s -> dragon, 300k, seed 97"; echo
  echo "Settings: D175d (--keyframe_compare column, the soundness terms kept, the segment replay measure), Hu, repo_d175."
  echo "Deliverables: $DL/keyframe_sheet.png (the keyframe | twin | full at t_k, two cameras), $O/auto/201_show300/silhouettes.png"
  echo "(frame 240 in the run's views, the keyframe's outline in red), $DL/show300k_twin_full_1080p.mp4, stills in $DL/stills."; echo
  sed -n '/^| run /,/^$/p' $S; sed -n '/^## Render influence/,/^KF-IoU curve/p' $S | grep -v "^KF-IoU curve"; } > $B/output/gpu/D175_SHOWCASE.md
echo "$(date '+%m-%d %H:%M:%S') [after_show] D175_SHOWCASE.md written" >> $O/status.txt
