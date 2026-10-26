#!/bin/bash
# SHOWCASE: the gallery montage of every example (start, progress-0.5 middle, end; the hflipped stills)
B=/data/relcfd/chayo/physmorph_v2; D=$B/output/gpu/showcase; T=$B/tmp/showcase
source $B/repo_t1s/scripts/ops/hyde06_env.sh >/dev/null 2>&1; export CUDA_VISIBLE_DEVICES=""
mkdir -p $D/montage; cd $D
nice -n 10 $PY $T/sc_montage.py $D/montage/showcase300k_montage.jpg 3 360 \
 "b1_bunny300k_L97=sphere → bunny" "b1_dragon300k_L97=sphere → dragon" "b1_armadilo300k_L97=sphere → armadillo" \
 "b1_homer300k_L97=sphere → homer" "b1_teapot300k_L97=sphere → teapot" "b1_bob300k_L97=sphere → bob" \
 "nefertiti300k_L97=sphere → nefertiti" "maxplanck300k_L97=sphere → max planck" "bimba300k_L97=sphere → bimba" \
 "spot300k_L97=sphere → spot" "cow300k_L97=sphere → cow" "cheburashka300k_L97=sphere → cheburashka" \
 "ogre300k_L97=sphere → ogre" "beast300k_L97=sphere → beast" "heart300k_L97=sphere → heart" \
 "fandisk300k_L97=sphere → fandisk" "A300k_L97=sphere → A" "V300k_L97=sphere → V" \
 "C300k_L97=sphere → C (stops early, still moving)" "happy300k_L97=sphere → happy buddha" "bunny2dragon300k_L97=bunny → dragon" \
 "teapot2bob300k_L97=teapot → bob (genus 1 → 1)" "spotx2cow300k_L97=spot, yawed → cow" "spot2cow300k_L97=spot → cow (head over the back)" \
 "bunny2happy300k_L97=bunny → happy buddha" "armadilo2homer300k_L97=armadillo → homer" "armadilo2homer300k_CV97=armadillo → homer, cell from volume"
