#!/bin/bash
# 116_d167_cross.sh -- queue job (agent D166, rerun 2026-10-24): if D166 reads the relaxation as the eraser (kept mean with
# --no-settle_relax >= the settle as it is + 0.3, and above E x3's gain), the best weight definition + settle 30 with
# --no-settle_relax on the 40k scenes (seeds 97 / 99), against the twin of that definition. Best = content share near held
# (>= 0.8 x H) and rest, the largest relief min; else rest only; else the largest relief min. Waits up to 3 h for the verdicts.
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d167; J=116_cross; TITLE="D167 cross: the best weight definition + settle 30 with --no-settle_relax"
source /data/relcfd/chayo/physmorph_v2/tmp/d166/job_common.sh
gate_wait; log "start: waiting for the D166 / D167 scene verdicts"
V="$G/d166/auto/61_settle $G/d166/auto/62_relaxoff $G/d166/auto/63_stiff3 $O/auto/101_floor $O/auto/102_shape $O/auto/103_held $O/auto/104_rule"
for k in $(seq 1 180); do n=0; for d in $V; do [ -e $d/verdict.json ] && n=$((n + 1)); done; [ $n -ge 7 ] && break; sleep 60; done
log "verdicts present: $n of 7"
read GO WD < <(/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python - $V <<'EOF'
import json, os, sys
vs = {}
for d in sys.argv[1:]:
    p = os.path.join(d, "verdict.json")
    if os.path.exists(p):
        vs.update(json.load(open(p))["arms"])
k = {a: (vs.get(a) or {}).get("kept_mean") for a in ("S0", "Rx", "E3")}
go = k["S0"] is not None and k["Rx"] is not None and k["Rx"] - k["S0"] >= 0.3 and (k["E3"] is None or k["Rx"] - k["S0"] >= k["E3"] - k["S0"])
h = vs.get("H") or {}
defs = {a: v for a, v in vs.items() if a in ("F", "FP", "S", "SP", "H", "HP", "R", "RP") and v.get("share_min") is not None}
near = lambda v: (h.get("share_min") is not None and h.get("colour_min") is not None and v.get("colour_min") is not None
                  and v["share_min"] >= 0.8 * h["share_min"] and v["colour_min"] >= 0.8 * h["colour_min"])
key = lambda a: defs[a]["share_min"]
wd = (max([a for a in defs if near(defs[a]) and defs[a].get("rest")], key=key, default=None)
      or max([a for a in defs if defs[a].get("rest")], key=key, default=None) or max(defs, key=key, default="F"))
print("yes" if go else "no", wd)
EOF
)
log "relaxation is the eraser: $GO; best definition: $WD"
if [ "$GO" != yes ]; then
  echo -e "# $TITLE\n\nD166 did not read the relaxation as the eraser (kept gain with --no-settle_relax < 0.3 or below E x3's): no cross run.\n\nVERDICT: no cross (best definition was $WD)" > $O/auto/$J/summary.md; exit 0; fi
declare -A WF=([R]="" [RP]="--placement_at_cell" [H]="--lambda_ema 0" [HP]="--lambda_ema 0 --placement_at_cell"
  [F]="--render_weight_ref floor" [FP]="--render_weight_ref floor --placement_at_cell" [S]="--render_weight_ref shape"
  [SP]="--render_weight_ref shape --placement_at_cell")
TWC=T; case $WD in *P) TWC=TP;; esac
if [ "$WD" = H ]; then
  echo -e "# $TITLE\n\nBest definition H (held): H + settle 30 + --no-settle_relax is D166's own arm Rx (job 62).\n\nVERDICT: Xr = Rx (no new run)" > $O/auto/$J/summary.md; exit 0; fi
echo "$WD" > $O/auto/$J/choice.txt
TITLE="$TITLE (chosen: $WD)"
scenes Xr ${WF[$WD]} --settle_windows 30 --no-settle_relax
[ $TWC = TP ] && for k in $(seq 1 120); do ls $O/runs/band_TP_99.rc > /dev/null 2>&1 && break; sleep 60; done
after scenes Xr:$TWC
