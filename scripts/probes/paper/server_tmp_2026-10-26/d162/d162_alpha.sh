#!/bin/bash
# d162_alpha.sh -- D162, card alpha: the gates (V), the scene runs (list A1) and the arrangement read (X) on one claim, then
# the gallery's list A2 on a second claim. Runs a private copy of d162_chain.sh (bash reads a script while it runs it, so a
# chain file replaced under a running chain is read at its old offset); launch this wrapper from a copy as well.
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d162; O=$B/output/gpu/d162
C=$W/run_alpha_chain.sh
cp $W/d162_chain.sh $C
RUNS_FILE=d162_runs_A1.txt bash $C "V R X"
[ -e $O/gates/PASSED ] || exit 1
RUNS_FILE=d162_runs_A2.txt bash $C R
