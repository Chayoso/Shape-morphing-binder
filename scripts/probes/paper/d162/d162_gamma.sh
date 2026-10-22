#!/bin/bash
# d162_gamma.sh -- D162, card beta's second list: once list B is done ("R done (d162_runs_B.txt)"), list A2 (the gallery's
# other half) on that card, from a private copy of d162_chain.sh. Card alpha reaches A2 only after A1 and X, by then
# every A2 run has its archive and is skipped; the first "R done (d162_runs_A2.txt)" line (the after-reader's cue) is
# this one's.
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d162; O=$B/output/gpu/d162; S=$O/status.txt
until grep -q "R done (d162_runs_B.txt)" $S; do sleep 60; done
C=$W/run_gamma_chain.sh
cp $W/d162_chain.sh $C
RUNS_FILE=d162_runs_A2.txt bash $C R
