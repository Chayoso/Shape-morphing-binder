#!/bin/bash
# 30_figs_v2_spot300k_L97r.sh -- gpu1 queue job (agent FIGS_V2, the dissertation's figures v2): Fig 4.2's spot row: sphere -> spot again with kept frames (SHOWCASE's spot300k_L97 kept frames were removed after its readings), stills with the display rule (pieces < 10 particles not drawn) and without it
# tmp/figs_v2/run_new.sh: a 300k sphere -> mesh run on repo_t1s (tag freeze-2026-10-15-rt1), seed 97, the frozen defaults
# (render arm), SHOWCASE's call; then the progress (CPU) and studio stills (fig_render.py). Status output/gpu/figs_v2/status.txt.
# Exit 3 = gpu1 lacked memory (< 21000 MiB free): queued again (see WAIT_SLOT). Peak ~19.6 GB (SHOWCASE's 300k runs).
export WAIT_SLOT=93   # if gpu1 lacks memory: retried as 93_figs_v2_spot300k_L97r_rNN.sh every 5 min, at most 8 h
exec bash /data/relcfd/chayo/physmorph_v2/tmp/figs_v2/run_new.sh spot300k_L97r assets/spot.obj 35 18 1 10 0,5,10,15,20,25,30,40,50,60,80,100,last /data/relcfd/chayo/physmorph_v2/output/gpu/showcase/ind/isosphere_spot_dense15.npz /data/relcfd/chayo/physmorph_v2/output/gpu/figs_v2/morph/spot 1
