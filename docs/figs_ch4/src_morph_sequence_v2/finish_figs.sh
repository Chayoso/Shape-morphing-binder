#!/bin/bash
# finish_figs.sh -- FIGS_V2 (CPU only, idempotent; run_new.sh calls it after every finished run, or run it by hand):
# builds whatever final figure is complete, in output/gpu/figs_v2/final/.
#  Fig 4.2 v2: the five rows (bunny, dragon, armadillo, bob from the tag's B1 runs; spot from its re-run), every row with
#              the display rule (fig_render.py, pieces < 10 particles not drawn), columns t = 0, 0.5, 1, 1.5, 2.5 s, end;
#  Fig 4.7 v2: SHOWCASE's montage without sphere -> A, V, C and with the first three accepted new runs (accepted.txt:
#              no early stop, no serious piece at the end), in their place after fandisk.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/figs_v2; T=$B/tmp/figs_v2; F=$O/final; M=$O/morph
source $B/repo_t1s/scripts/ops/hyde06_env.sh > /dev/null 2>&1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
mkdir -p $F
log() { echo "$(date '+%m-%d %H:%M:%S') finish: $*" >> $O/status.txt; }
if [ -s $M/spot/pieces.jsonl ] && [ "$(ls $M/spot/frames/*.jpg 2>/dev/null | wc -l)" -ge 13 ]; then
  nice -n 10 $PY $T/compose_morph.py $F/fig_morph_sequence_v2.jpg 0,10,20,30,50,last bunny=$M/bunny/frames \
    dragon=$M/dragon/frames armadillo=$M/armadilo/frames bob=$M/bob/frames spot=$M/spot/frames > $F/fig_morph_sequence_v2.log 2>&1
  nice -n 10 $PY $T/rule_diff.py $M bunny dragon armadilo bob spot > $F/rule_diff.txt 2>&1
  log "Fig 4.2 v2 with the spot re-run: rc $? ($(tail -1 $F/fig_morph_sequence_v2.log))"
fi
if [ -s $O/accepted.txt ] && [ "$(wc -l < $O/accepted.txt)" -ge 3 ]; then
  new=()
  for nm in $(head -3 $O/accepted.txt); do new+=("$nm=sphere → ${nm%300k_L97}"); done
  nice -n 10 $PY $T/fig_montage_v2.py $F/fig_showcase_montage_v2.jpg 3 360 \
   "b1_bunny300k_L97=sphere → bunny" "b1_dragon300k_L97=sphere → dragon" "b1_armadilo300k_L97=sphere → armadillo" \
   "b1_homer300k_L97=sphere → homer" "b1_teapot300k_L97=sphere → teapot" "b1_bob300k_L97=sphere → bob" \
   "nefertiti300k_L97=sphere → nefertiti" "maxplanck300k_L97=sphere → max planck" "bimba300k_L97=sphere → bimba" \
   "spot300k_L97=sphere → spot" "cow300k_L97=sphere → cow" "cheburashka300k_L97=sphere → cheburashka" \
   "ogre300k_L97=sphere → ogre" "beast300k_L97=sphere → beast" "heart300k_L97=sphere → heart" \
   "fandisk300k_L97=sphere → fandisk" "${new[@]}" \
   "happy300k_L97=sphere → happy buddha" "bunny2dragon300k_L97=bunny → dragon" \
   "teapot2bob300k_L97=teapot → bob (genus 1 → 1)" "spotx2cow300k_L97=spot, yawed → cow" "spot2cow300k_L97=spot → cow (head over the back)" \
   "bunny2happy300k_L97=bunny → happy buddha" "armadilo2homer300k_L97=armadillo → homer" "armadilo2homer300k_CV97=armadillo → homer, cell from volume" \
   > $F/fig_showcase_montage_v2.log 2>&1
  log "Fig 4.7 v2 montage with $(head -3 $O/accepted.txt | tr '\n' ' '): rc $? ($(tail -1 $F/fig_showcase_montage_v2.log))"
fi
