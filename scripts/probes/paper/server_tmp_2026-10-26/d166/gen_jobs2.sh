#!/bin/bash
# gen_jobs2.sh -- the D166 / D167 rerun from scratch (2026-10-24, after the settle_relax capture fix): jobs 60-63 (D166),
# 100-129 (D167), 199 (the master) into tmp/gpu1_queue (agent D166).
B=/data/relcfd/chayo/physmorph_v2; Q=$B/tmp/gpu1_queue; W=$B/tmp/d166
: > $W/expected.txt
mk() { local n=$1 nm=$2 o=$3 t=$4 body=$5 f=$Q/${1}_${3}_${2}.sh
  cat > $f.tmp <<EOF
#!/bin/bash
# ${n}_${o}_${nm}.sh -- queue job (agent D166, rerun 2026-10-24): $t
O=$B/output/gpu/$o; J=${n}_${nm}; TITLE="$t"
source $W/job_common.sh
$body
log "done"
EOF
  mv $f.tmp $f; echo "${n}_${nm}" >> $W/expected.txt; }
G1='gate_wait; log "start"'
cp $W/job60_gate.sh $Q/60_d166_gate.sh.tmp && mv $Q/60_d166_gate.sh.tmp $Q/60_d166_gate.sh; echo 60_gate >> $W/expected.txt
mk 61 settle d166 "D166: held + settle 30, the settle as it is (u zero, relaxation on)" "$G1; scenes S0 \$HS; after scenes S0:T"
mk 62 relaxoff d166 "D166: held + settle 30 with the layer relaxation off in the settle (--no-settle_relax, capture fix)" \
  "$G1; scenes Rx \$HS --no-settle_relax; after scenes Rx:T"
mk 63 stiff3 d166 "D166: held + settle 30 with E x3 (4.2e5), against its own E x3 twin" \
  "$G1; scenes E3 \$HS --young 4.2e5; scenes TE3 \$TW --young 4.2e5; after scenes E3:TE3"
P="--placement_at_cell"; FL="--render_weight_ref floor"; SH="--render_weight_ref shape"; HD="--lambda_ema 0"
mk 100 twinP d167 "D167 scenes: the twin with placement_at_cell" "$G1; scenes TP \$TW $P; after scenes TP:T"
mk 101 floor d167 "D167 scenes: ref floor without / with placement_at_cell" "$G1; scenes F $FL; scenes FP $FL $P; after scenes F:T FP:TP"
mk 102 shape d167 "D167 scenes: ref shape without / with placement_at_cell" "$G1; scenes S $SH; scenes SP $SH $P; after scenes S:T SP:TP"
mk 103 held d167 "D167 scenes: held (--lambda_ema 0) without / with placement_at_cell" "$G1; scenes H $HD; scenes HP $HD $P; after scenes H:T HP:TP"
mk 104 rule d167 "D167 scenes: the rule (tag default) without / with placement_at_cell" "$G1; scenes R; scenes RP $P; after scenes R:T RP:TP"
mk 110 gal_twinP d167 "D167 gallery (6 meshes 40k seed 97): the twin with placement_at_cell" "$G1; gal TP \$TW $P; after gallery TP:T"
mk 111 gal_floor d167 "D167 gallery: ref floor without / with placement_at_cell" "$G1; gal F $FL; gal FP $FL $P; after gallery F:T FP:TP"
mk 112 gal_shape d167 "D167 gallery: ref shape without / with placement_at_cell" "$G1; gal S $SH; gal SP $SH $P; after gallery S:T SP:TP"
mk 113 gal_held d167 "D167 gallery: held without / with placement_at_cell" "$G1; gal H $HD; gal HP $HD $P; after gallery H:T HP:TP"
mk 114 gal_rule d167 "D167 gallery: the rule (tag default) without / with placement_at_cell" "$G1; gal R; gal RP $P; after gallery R:T RP:TP"
cp $W/job116_cross.sh $Q/116_d167_cross.sh.tmp && mv $Q/116_d167_cross.sh.tmp $Q/116_d167_cross.sh; echo 116_cross >> $W/expected.txt
for x in "120 T T:T|\$TW" "121 F F:T|$FL" "122 S S:T|$SH" "123 H H:T|$HD" "124 R R:T|" "125 TP TP:T|\$TW $P" \
         "126 FP FP:TP|$FL $P" "127 SP SP:TP|$SH $P" "128 HP HP:TP|$HD $P" "129 RP RP:TP|$P"; do
  n=${x%% *}; r=${x#* }; c=${r%% *}; r=${r#* }; spec=${r%%|*}; fl=${r#*|}
  mk $n 300k_$c d167 "D167 300k (bunny + dragon, seed 97): arm $c" "$G1; big $c $fl; after big $spec"
done
cp $W/job199_aggregate.sh $Q/199_d167_aggregate.sh.tmp && mv $Q/199_d167_aggregate.sh.tmp $Q/199_d167_aggregate.sh
ls $Q | grep -E '^[0-9]{2,3}_d16[67]_.*\.sh$' | sort -n | tr "\n" " "; echo; wc -l < $W/expected.txt
