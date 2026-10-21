#!/bin/bash
# B9b option 1, the 40k stage (pre-registered in docs/experiments.md, D145, B9b, "Option 1", before any run):
#   setsid -f bash scripts/probes/b9b/chain40kw.sh REPO > /dev/null 2>&1 < /dev/null
# On ONE card idle 10 min (lock protocol, agent B9b): (1) the full suite (the pinned digest of the frozen single-body
# path, the scene's tests with the field off), its exit status recorded; not 0 stops the chain; (2) one_equal.py in the
# standing mode (the one-body scene in the world frame with B9's lifted target against the frozen window with it, no
# gravity): B must equal A (positions 1e-5 wu; each term within the test's tolerance or overlapping replay ranges) or
# the chain stops, C reported; (3) the eight 40k runs on specs/b9w_MESH.json (frame "standing"); (4) the GPU readings
# as B9's: the archive in the objective's frame (B9's world) against the 40k independent sample lifted as the run's
# target (ind_shift.py), at the end and at B9's end time; momentum_probe.py. The card released: tobody's tracks are
# already written; apart.py (at most three at once, niced), read_b9b.py 40k (the gate). No 300k here.
REPO=$1
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/b9b; R=$O/world; LOG=$O/chain40kw.log; T=$B/tmp
S=$REPO/scripts/probes/b9b; SP=scripts/probes/b9b/specs
mkdir -p $R
log() { echo "$(date '+%F %T') $*" >> $LOG; }
source $S/claim.sh
trap "release_card" EXIT
trap "exit 143" TERM INT HUP
source $B/tmp/d146/cache_env.sh
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
source $REPO/scripts/ops/hyde06_env.sh > /dev/null          # $PY; cwd = $REPO
echo $$ > $O/chain40kw.pid
log "chain40kw start (pid $$; $(tail -1 $REPO/DEPLOYED_FROM))"
claim_card "b9b/chain40kw: full suite, the standing equality check, 8 one-body 40k runs (option 1), GPU readings (~90 min)"
log "claimed gpu$CARD"
$PY -m pytest -q -p no:cacheprovider tests > $R/suite.txt 2>&1
rc=$?
log "full suite exit $rc: $(tail -1 $R/suite.txt)"
log "pinned references: $(grep -c 'test_refreeze_reference' $R/suite.txt) lines mention them; $(grep -E 'refreeze_reference.*(FAILED|ERROR)' $R/suite.txt | head -2 | tr '\n' ' ')"
[ $rc = 0 ] || { log "STOP: the full suite did not pass (exit $rc)"; exit 1; }
$PY -m pytest -q -p no:cacheprovider tests/test_refreeze_reference.py -v > $R/pinned.txt 2>&1
log "pinned reference tests exit $?: $(grep -E 'PASSED|FAILED|SKIPPED' $R/pinned.txt | sed 's/.*:://' | tr '\n' ' ' | cut -c1-300)"
$PY $S/one_equal.py $R/one_equal_stand.json 40000 97 bunny standing > $R/one_equal_stand.txt 2>&1
log "one_equal (standing) exit $?"
$PY - $R/one_equal_stand.json >> $R/one_equal_stand.txt 2>&1 <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
p = r["pitch"]
# B9b's amended check (2026-10-17): the scene test's tolerance (positions 1e-5 wu; a term within 1e-4 of max(|A|, 1e-12))
# or, for a term, the replay ranges overlapping (8 replays each); B stops the chain, C is reported
verdict = 3
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
            (noted if ov else bad).append(f"{c}: {k} {arm} [{min(br):.6e}, {max(br):.6e}] A [{min(ar):.6e}, {max(ar):.6e}]"
                                          + (" overlap" if ov else " APART"))
    print(f"ARM {arm}: replay ranges where the tolerance did not hold: " + ("; ".join(noted) or "none"))
    print(f"ARM {arm} ({'the same definition' if arm == 'B' else 'option 1 dynamics, no loads'}):",
          "EQUAL" if not bad else "DIFFERS " + "; ".join(bad))
    if arm == "B":
        verdict = 0 if not bad else 3
sys.exit(verdict)
PYEOF
rc=$?
log "one_equal (standing): $(grep '^ARM' $R/one_equal_stand.txt | tr '\n' ' ' | cut -c1-900)"
[ $rc = 3 ] && { log "STOP: the standing scene window is not the frozen window with the lifted target (arm B)"; exit 1; }
[ $rc = 0 ] || { log "STOP: the standing equality check did not run to its end (rc $rc)"; exit 1; }
for mesh in bunny dragon; do
  for a in "L97 97" "L99 99" "L101 101" "P97 97 --render_weight_scale 0"; do
    set -- $a; arm=$1 seed=$2; shift 2; tag=${mesh}40k_$arm
    $PY scripts/scene_run.py --scene $SP/b9w_$mesh.json --tgt assets/$mesh.obj --n 40000 --seed $seed --save_stride 12 \
      "$@" --out $R/$tag > $R/$tag.log 2>&1
    log "$tag exit $?: $(tail -2 $R/$tag.log | head -1 | cut -c1-220)"
  done
done
times() { [ $1 = bunny ] && echo 5.33333 || echo 8.16667; }        # B9's 40k G2L97 end (frame 1280 / 1960)
: > $R/ind40.log
for mesh in bunny dragon; do
  for arm in L97 L99 L101 P97; do
    tag=${mesh}40k_$arm
    [ -f $R/$tag.npz ] || continue
    OMP_NUM_THREADS=4 nice -n 10 $PY $S/tobody.py $R $tag $(times $mesh) > $R/tobody_$tag.txt 2>&1
    at=$(ls $R/${tag}_at*_body.npz 2>/dev/null | head -1)
    ( export PHYSMORPH_RUN_REPO=$B/repo_r77; source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null; cd $B/repo_r77
      $PY $T/d145/ind_shift.py $G/d92/ind_$mesh.npz $R/${tag}_body12.npz $R/ind_floor_$tag.npz > $R/ind_shift_$tag.txt 2>&1
      $PY $T/gallery_ind.py $R/ind40.log "$tag=$R/${tag}_body12.npz,$R/ind_floor_$tag.npz" \
        "$(basename $at .npz)=$at,$R/ind_floor_$tag.npz" > $R/ind40_$tag.txt 2>&1 < /dev/null )
    $T/mom.sh $CARD $R/m_$tag.log "$mesh $arm=$R/${tag}_frames12.npz"
    log "$tag read: $(tail -1 $R/tobody_$tag.txt | cut -c1-260)"
  done
done
release_card
log "card released for the CPU readings"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
n=0
for mesh in bunny dragon; do
  for arm in L97 L99 L101 P97; do
    tag=${mesh}40k_$arm
    [ -f $R/$tag.npz ] || continue
    nice -n 15 $PY $REPO/scripts/probes/d147/apart.py $R $tag > $R/apart_$tag.txt 2>&1 &
    n=$((n + 1)); [ $((n % 3)) = 0 ] && wait
  done
done
wait
nice -n 15 $PY $S/read_b9b.py 40k $R $R/read_40k "option 1: the scene path with the world-fixed standing target" > $R/read_40k.txt 2>&1
log "read_40k exit $?: $(grep '^GATE' $R/read_40k.md)"
for f in $R/*40k_*_frames12.npz $R/*40k_*_body12.npz $R/*40k_*_at*_body.npz; do
  [ -f $f ] || continue
  sz=$(du -m $f | cut -f1); rm -f $f
  echo "$(date '+%Y-%m-%d %H:%M') b9b: removed $f ($sz MB; B9b option 1 40k reading copy, read; the run's own kept frames stay in its .npz)" >> $G/cleanup.log
done
log "chain40kw end (no 300k here: pre-registered separately on a PASS)"
