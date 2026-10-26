#!/bin/bash
# gen_jobs.sh -- writes the D166 (60-67) / D167 (100-129) jobs and the aggregator (199) into tmp/gpu1_queue (agent D166).
B=/data/relcfd/chayo/physmorph_v2; Q=$B/tmp/gpu1_queue; W=$B/tmp/d166
: > $W/expected.txt
mk() { # mk NUM NAME OROOT TITLE BODY -- one job file, written as .tmp then moved
  local n=$1 nm=$2 o=$3 t=$4 body=$5 f=$Q/${1}_d16${3: -1}_${2}.sh
  cat > $f.tmp <<EOF
#!/bin/bash
# ${n}_d16${o: -1}_${nm}.sh -- queue job (agent D166): $t
O=$B/output/gpu/$o; J=${n}_${nm}; TITLE="$t"
source $W/job_common.sh
$body
log "done"
EOF
  mv $f.tmp $f; echo "${n}_${nm}" >> $W/expected.txt; }
G1='gate_wait; log "start"'

# ---- D166: rest under authority (40k; coin + bunny relief + band; seeds 97 / 99) ----
cp $W/job60_gate.sh $Q/60_d166_gate.sh.tmp && mv $Q/60_d166_gate.sh.tmp $Q/60_d166_gate.sh; echo 60_gate >> $W/expected.txt
mk 61 ulast d166 "D166 (a1): held + settle 30, u held at its last committed value (--settle_u last)" \
  "$G1; scenes Ul \$HS --settle_u last; after scenes Ul:T"
mk 62 relaxoff d166 "D166 (a2): held + settle 30, no layer relaxation in the settle windows (--no-settle_relax)" \
  "$G1; scenes Rx \$HS --no-settle_relax; after scenes Rx:T"
mk 63 assim10 d166 "D166 (b): held + settle 30, plastic assimilation 1.0 at every commit (default 0.5), against its own twin" \
  "$G1; scenes A10 \$HS --assim 1.0; scenes TA10 \$TW --assim 1.0; after scenes A10:TA10"
mk 64 assim08 d166 "D166 (b): held + settle 30, assimilation 0.8 (default 0.5), against its own twin" \
  "$G1; scenes A08 \$HS --assim 0.8; scenes TA08 \$TW --assim 0.8; after scenes A08:TA08"
mk 65 cross_a10 d166 "D166 (a x b): u held / no relaxation in the settle, each with assimilation 1.0" \
  "$G1; scenes UlA10 \$HS --settle_u last --assim 1.0; scenes RxA10 \$HS --no-settle_relax --assim 1.0; after scenes UlA10:TA10 RxA10:TA10"
mk 66 assim025 d166 "D166 (b): held + settle 30, assimilation 0.25 (the direction check), against its own twin" \
  "$G1; scenes A025 \$HS --assim 0.25; scenes TA025 \$TW --assim 0.25; after scenes A025:TA025"
mk 67 stiff3 d166 "D166 (d): E x3 (4.2e5) on the two settle fixes; job 51 (E x3 / x7 under plain settle) is in the master file" \
  "$G1; scenes UlE3 \$HS --settle_u last --young 4.2e5; scenes RxE3 \$HS --no-settle_relax --young 4.2e5; scenes TE3 \$TW --young 4.2e5; after scenes UlE3:TE3 RxE3:TE3"

# ---- D167: the weight definition ----
P="--placement_at_cell"; FL="--render_weight_ref floor"; SH="--render_weight_ref shape"; HD="--lambda_ema 0"
mk 100 twinP d167 "D167: the twin with placement_at_cell (the P arms' twin), 40k scenes" \
  "$G1; scenes TP \$TW $P; after scenes TP:T"
mk 101 floor d167 "D167: weight ref floor, without / with placement_at_cell, 40k scenes" \
  "$G1; scenes F $FL; scenes FP $FL $P; after scenes F:T FP:TP"
mk 102 shape d167 "D167: weight ref shape, without / with placement_at_cell, 40k scenes" \
  "$G1; scenes S $SH; scenes SP $SH $P; after scenes S:T SP:TP"
mk 103 held d167 "D167: held weight (--lambda_ema 0), without / with placement_at_cell, 40k scenes" \
  "$G1; scenes H $HD; scenes HP $HD $P; after scenes H:T HP:TP"
mk 104 rule d167 "D167: the rule (tag default), without / with placement_at_cell, 40k scenes" \
  "$G1; scenes R; scenes RP $P; after scenes R:T RP:TP"
mk 110 gal_twinP d167 "D167 gallery (6 meshes, 40k, seed 97): the twin with placement_at_cell" \
  "$G1; gal TP \$TW $P; after gallery TP:T"
mk 111 gal_floor d167 "D167 gallery: ref floor without / with placement_at_cell" \
  "$G1; gal F $FL; gal FP $FL $P; after gallery F:T FP:TP"
mk 112 gal_shape d167 "D167 gallery: ref shape without / with placement_at_cell" \
  "$G1; gal S $SH; gal SP $SH $P; after gallery S:T SP:TP"
mk 113 gal_held d167 "D167 gallery: held without / with placement_at_cell" \
  "$G1; gal H $HD; gal HP $HD $P; after gallery H:T HP:TP"
mk 114 gal_rule d167 "D167 gallery: the rule (tag default, rerun on repo_d166) without / with placement_at_cell" \
  "$G1; gal R; gal RP $P; after gallery R:T RP:TP"
cp $W/job116_cross.sh $Q/116_d167_cross.sh.tmp && mv $Q/116_d167_cross.sh.tmp $Q/116_d167_cross.sh; echo 116_cross >> $W/expected.txt
for x in "120 T T:T|\$TW" "121 F F:T|$FL" "122 S S:T|$SH" "123 H H:T|$HD" "124 R R:T|" "125 TP TP:T|\$TW $P" \
         "126 FP FP:TP|$FL $P" "127 SP SP:TP|$SH $P" "128 HP HP:TP|$HD $P" "129 RP RP:TP|$P"; do
  n=${x%% *}; r=${x#* }; c=${r%% *}; r=${r#* }; spec=${r%%|*}; fl=${r#*|}
  mk $n 300k_$c d167 "D167 300k (bunny + dragon, seed 97): arm $c" "$G1; big $c $fl; after big $spec"
done
cp $W/job199_aggregate.sh $Q/199_d167_aggregate.sh.tmp && mv $Q/199_d167_aggregate.sh.tmp $Q/199_d167_aggregate.sh
ls $Q | grep -E '^(6[0-9]|1[0-9][0-9])_d16' ; wc -l $W/expected.txt
