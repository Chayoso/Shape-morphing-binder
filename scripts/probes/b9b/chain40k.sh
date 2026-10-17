#!/bin/bash
# B9b chain 1, the 40k stage (pre-registered in docs/experiments.md, D145, "B9b", before any run):
#   nohup setsid bash scripts/probes/b9b/chain40k.sh REPO > /dev/null 2>&1 < /dev/null &
# On ONE card idle 10 min (lock protocol, agent B9b): the scene tests; one_equal.py (the one-body scene against the
# single-body window without gravity, 40k bunny s97, rollout level; arm B must equal A to the scene test's tolerance or
# the chain stops); the first optimised window, frozen twice and the scene once (w1_compare.py); the eight 40k runs
# (bunny / dragon x L97 L99 L101 P97 on specs/b9b_MESH.json); the GPU readings (gallery_ind.py in the body frame
# against the 40k independent sample, at the end and at B9's end time; momentum_probe.py). The card released, on the
# CPU: apart.py, read_b9b.py 40k (the gate). On "GATE 40k: PASS" it starts chain300k.sh (which claims its own card).
REPO=$1
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/b9b; R=$O/runs; LOG=$O/chain40k.log; T=$B/tmp
S=$REPO/scripts/probes/b9b; SP=scripts/probes/b9b/specs
mkdir -p $R
log() { echo "$(date '+%F %T') $*" >> $LOG; }
source $S/claim.sh
trap "release_card" EXIT
trap "exit 143" TERM INT HUP
source $B/tmp/d146/cache_env.sh
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
source $REPO/scripts/ops/hyde06_env.sh > /dev/null          # $PY; cwd = $REPO
echo $$ > $O/chain40k.pid
log "chain40k start (pid $$; $(head -1 $REPO/DEPLOYED_FROM))"
claim_card "b9b/chain40k: scene tests, the 40k equality check, 8 one-body 40k gravity runs, GPU readings (~60 min)"
log "claimed gpu$CARD"
$PY -m pytest -q -p no:cacheprovider tests/test_scene_rest.py tests/test_scene_bodies.py tests/test_scene_fixes.py \
  tests/test_scene_frame.py tests/test_scene_schedule.py tests/test_scene_assim.py > $R/tests.txt 2>&1
rc=$?
log "scene tests exit $rc: $(tail -1 $R/tests.txt)"
[ $rc = 0 ] || { log "STOP: the scene tests failed (exit $rc)"; exit 1; }
$PY $S/one_equal.py $R/one_equal.json 40000 97 bunny > $R/one_equal.txt 2>&1
log "one_equal exit $?"
$PY - $R/one_equal.json >> $R/one_equal.txt 2>&1 <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
p = r["pitch"]
# amended 2026-10-17 17:50 CDT (docs, B9b): the scene test's tolerance (positions 1e-5 wu; a term within 1e-4 of
# max(|A|, 1e-12)) OR, for a term, B's replay range overlapping the frozen window's own replay range (8 replays each;
# D145's convention): the first check had no floor and stopped on the kinetic terms of a body at rest (zero control)
for arm in ("B", "C"):
    bad, noted = [], []
    for c, v in r["arms"][arm].items():
        a = r["arms"]["A"][c]
        if v["dx_max_sp_all"] * p > 1e-5:
            bad.append(f"{c}: positions {v['dx_max_sp_all'] * p:.2e} wu")
        for k, br in v["replays"].items():
            a1, ar = a["terms"][k], a["replays"][k]
            if all(abs(b - a1) <= 1e-4 * max(abs(a1), 1e-12) for b in br):
                continue
            ov = min(br) <= max(ar) and min(ar) <= max(br)
            s = (f"{c}: {k} B [{min(br):.6e}, {max(br):.6e}] A [{min(ar):.6e}, {max(ar):.6e}]"
                 .replace("B [", f"{arm} [", 1))
            (noted if ov else bad).append(s + (" overlap" if ov else " APART"))
    print(f"ARM {arm}: replay ranges where the tolerance did not hold: " + ("; ".join(noted) or "none"))
    print(f"ARM {arm} ({'the same definition' if arm == 'B' else 'B9b without loads'}):",
          "EQUAL" if not bad else "DIFFERS " + "; ".join(bad))
    if arm == "B":
        verdict = 0 if not bad else 3
sys.exit(verdict)
EOF
rc=$?
log "one_equal: $(grep '^ARM' $R/one_equal.txt | tr '\n' ' ' | cut -c1-900)"
if [ $rc = 3 ]; then log "STOP: the one-body scene is not the frozen window at 40k (arm B)"; exit 1; fi
[ $rc = 0 ] || log "one_equal did not run to its end (rc $rc): the check is missing, the runs go on (read the log)"
$PY scripts/pipeline_run.py --tgt assets/bunny.obj --n 40000 --seed 97 --animations 1 --out $R/w1_frozen_a > $R/w1_frozen_a.log 2>&1
log "w1_frozen_a exit $?"
$PY scripts/pipeline_run.py --tgt assets/bunny.obj --n 40000 --seed 97 --animations 1 --out $R/w1_frozen_b > $R/w1_frozen_b.log 2>&1
log "w1_frozen_b exit $?"
$PY scripts/scene_run.py --scene $SP/b9b_g0.json --tgt assets/bunny.obj --n 40000 --seed 97 --windows 1 --out $R/w1_scene > $R/w1_scene.log 2>&1
log "w1_scene exit $?"
$PY $S/w1_compare.py $R w1_frozen_a w1_frozen_b w1_scene > $R/w1_compare.json 2>&1
log "w1_compare exit $?"
for mesh in bunny dragon; do
  for a in "L97 97" "L99 99" "L101 101" "P97 97 --render_weight_scale 0"; do
    set -- $a; arm=$1 seed=$2; shift 2; tag=${mesh}40k_$arm
    $PY scripts/scene_run.py --scene $SP/b9b_$mesh.json --tgt assets/$mesh.obj --n 40000 --seed $seed --save_stride 12 \
      "$@" --out $R/$tag > $R/$tag.log 2>&1
    log "$tag exit $?: $(tail -2 $R/$tag.log | head -1 | cut -c1-220)"
  done
done
# the GPU readings: the body-frame archives (tobody.py), then the 40k readings as B9's (repo_r77's gallery_ind.py and
# repo_r66's momentum_probe.py, the same code as B9's)
times() { [ $1 = bunny ] && echo 5.33333 || echo 8.16667; }        # B9's 40k G2L97 end (frame 1280 / 1960)
: > $R/ind40.log
for mesh in bunny dragon; do
  for arm in L97 L99 L101 P97; do
    tag=${mesh}40k_$arm
    [ -f $R/$tag.npz ] || continue
    $PY $S/tobody.py $R $tag $(times $mesh) > $R/tobody_$tag.txt 2>&1
    at=$(ls $R/${tag}_at*_body.npz 2>/dev/null | head -1)
    ( export PHYSMORPH_RUN_REPO=$B/repo_r77; source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null; cd $B/repo_r77
      $PY $T/gallery_ind.py $R/ind40.log "$tag=$R/${tag}_body12.npz,$G/d92/ind_$mesh.npz" \
        "$(basename $at .npz)=$at,$G/d92/ind_$mesh.npz" > $R/ind40_$tag.txt 2>&1 < /dev/null )
    $T/mom.sh $CARD $R/m_$tag.log "$mesh $arm=$R/${tag}_frames12.npz"
    log "$tag read: $(tail -1 $R/tobody_$tag.txt | cut -c1-200)"
  done
done
release_card
log "card released for the CPU readings"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
for mesh in bunny dragon; do
  for arm in L97 L99 L101 P97; do
    tag=${mesh}40k_$arm
    [ -f $R/$tag.npz ] && $PY $REPO/scripts/probes/d147/apart.py $R $tag > $R/apart_$tag.txt 2>&1 &
  done
done
wait
$PY $S/read_b9b.py 40k $R $R/read_40k > $R/read_40k.txt 2>&1
log "read_40k exit $?: $(grep '^GATE' $R/read_40k.md)"
for f in $R/*40k_*_frames12.npz $R/*40k_*_body12.npz $R/*40k_*_at*_body.npz; do
  [ -f $f ] || continue
  sz=$(du -m $f | cut -f1); rm -f $f
  echo "$(date '+%Y-%m-%d %H:%M') b9b: removed $f ($sz MB; B9b 40k reading copy, read; the run's own kept frames stay in its .npz)" >> $G/cleanup.log
done
if grep -q '^GATE 40k: PASS' $R/read_40k.md; then
  nohup setsid bash $S/chain300k.sh $REPO > /dev/null 2>&1 < /dev/null &
  log "GATE PASS: chain300k started (pid $!)"
else
  log "GATE not passed: no 300k (the rule); chain40k end"
fi
log "chain40k end"
