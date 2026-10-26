#!/bin/bash
# 33_figs_v2_bust300k_L97.sh -- gpu1 queue job (agent FIGS_V2, the dissertation's figures v2): Fig 4.7 v2: sphere -> bust (new montage example)
# tmp/figs_v2/run_new.sh: a 300k sphere -> mesh run on repo_t1s (tag freeze-2026-10-15-rt1), seed 97, the frozen defaults
# (render arm), SHOWCASE's call; then the progress (CPU) and studio stills (fig_render.py). Status output/gpu/figs_v2/status.txt.
# Exit 3 = gpu1 lacked memory (< 21000 MiB free): queued again (see WAIT_SLOT). Peak ~19.6 GB (SHOWCASE's 300k runs).
export WAIT_SLOT=96   # if gpu1 lacks memory: retried as 96_figs_v2_bust300k_L97_rNN.sh every 5 min, at most 8 h
exec bash /data/relcfd/chayo/physmorph_v2/tmp/figs_v2/run_new.sh bust300k_L97 /data/relcfd/chayo/physmorph_v2/tmp/figs_v2/assets/bust.obj 330 12 1.06 0 0,5,10,15,20,30,50,mid,last /data/relcfd/chayo/physmorph_v2/output/gpu/figs_v2/ind/isosphere_bust_dense15.npz /data/relcfd/chayo/physmorph_v2/output/gpu/figs_v2/render/bust300k_L97 0
