#!/bin/bash
# cleanup_d145.sh [go] -- D145's server cleanup (the user's GO, 2026-10-14): archive the logs/json/txt/csv of the touched
# directories first, then remove superseded kept frames, archives, bench states and old video frame folders. Without
# "go" it only lists. Kept: d90/d92 references, d111 dense samples, frozen_b, d139, d144, final, frozen, d135*, d146.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; cd $G || exit 1
FILES="d105/dragon300k_PV_frames12.npz d105/dragon300k_LV_frames12.npz d105/bunny300k_PV_frames12.npz d105/bunny300k_LV_frames12.npz
d120/dragon300k_LE_frames12.npz d120/bunny300k_LE_frames12.npz d121_proof/bunny300k_LVg_frames12.npz
d116/tail_bunny_LV_tangent.npz d116/tail_bunny_LV_orig.npz d116/tail_bunny_LV_normal.npz d116/t40k_render_full_dt_iso_nn.npz
d112/s40k_ours_render_full_dt_iso_nn.npz d112/s40k_xu_spray_paper_render_full_dt_iso_nn.npz d112/bunny300k_XP_frames12.npz
d112/dbg40k_paper_render_full_dt_iso_nn.npz d96/bunny300k_prof_frames12.npz show/bunny40000_render_full_dt_iso_nn.npz"
FILES="$FILES $(ls d118/*_frames12.npz rt/bench/*.pt 2>/dev/null | tr '\n' ' ')"
DIRS="$(ls -d d46/frames_* d17/frames_* d15r/frames_* d16r/frames_* d56/frames_* d52/frames_* 2>/dev/null | tr '\n' ' ')"
tot=0; for f in $FILES $DIRS; do [ -e "$f" ] || continue; s=$(du -sm "$f" | cut -f1); tot=$((tot+s)); echo "$s MB $f"; done
echo "total $tot MB"
[ "$1" = go ] || exit 0
A=$B/output/logs_archive_2026-10-14_d145.tgz
tar czf $A $(for d in d105 d118 d120 rt d121_proof d116 d112 d96 show d46 d17 d15r d16r d56 d52; do find $d -maxdepth 2 \( -name '*.log' -o -name '*.json' -o -name '*.txt' -o -name '*.csv' \) ; done) 2>/dev/null
echo "archive $A $(du -m $A | cut -f1) MB"
for f in $FILES $DIRS; do [ -e "$f" ] || continue; s=$(du -sm "$f" | cut -f1); rm -rf "$f"
  echo "$(date '+%Y-%m-%d %H:%M') d145: removed $G/$f ($s MB; superseded before the paper's evidence runs (D145 cleanup to <= 25 GB); logs/json/txt/csv in $(basename $A))" >> $G/cleanup.log; done
du -sh $B/output
