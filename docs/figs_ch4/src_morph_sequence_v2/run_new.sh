#!/bin/bash
# run_new.sh NAME TGT_OBJ AZ EL DIST MIN_PIECE KS DENSE_NPZ OUT_DIR RAW -- FIGS_V2 (2026-10-22), called by the gpu1 queue
# jobs 3N_figs_v2_*.sh (foreground; the runner holds gpu1 and exports CUDA_VISIBLE_DEVICES=1).
# 1. A 300k run sphere -> TGT, seed 97, on repo_t1s (tag freeze-2026-10-15-rt1, kept frames), the frozen defaults (the
#    render arm "dF_c + layer u + render loss (full)"): exactly SHOWCASE's run_one call (sc_chain.sh). Skipped (exit 3)
#    if gpu1 has < 21000 MiB free (a 300k run peaks at ~19.6 GB; another user shares the card).
# 2. The progress per kept frame (SHOWCASE's sc_mid.py, CPU): `mid` = the first kept frame with progress >= 0.5.
# 3. Studio stills of the kept frames KS ("mid" and "last" allowed) by fig_render.py (render_deliverable.py's base
#    display, +field=zbs, fit 0), with MIN_PIECE (0: every disc drawn, as the SHOWCASE renders) into OUT_DIR, then
#    mirrored back (hflip, as SHOWCASE) into OUT_DIR/stills/.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/figs_v2; R=$O/runs; T=$B/tmp/figs_v2
name=$1 tgt=$2 az=$3 el=$4 dist=$5 rule=$6 ks=$7 dense=$8 outd=$9 raw=${10:-0}
mkdir -p $R $O/mid $T/tmpdir
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $O/status.txt; }
export PHYSMORPH_RUN_REPO=$B/repo_t1s PHYSMORPH_OUT=$G
source $B/repo_t1s/scripts/ops/hyde06_env.sh > /dev/null 2>&1
source $B/tmp/d146/cache_env.sh
export OMP_NUM_THREADS=8 TMPDIR=$T/tmpdir
cd $B/repo_t1s
F=$R/${name}_frames12.npz
if [ ! -e $R/rc_$name ]; then
  free=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "$free" -lt 21000 ]; then
    # not enough room beside the other user: wait 5 min, then queue this job again behind everything already queued
    # (name 9P_figs_v2_NAME_rNN.sh, P = its place: spot 3, kitten 4, falcon 5, bust 6), at most 96 tries (8 h), and exit 3
    n=$(cat $O/tries_$name 2>/dev/null || echo 0); n=$((n + 1)); echo $n > $O/tries_$name
    log "$name waits: gpu1 has $free MiB free (< 21000), try $n"
    [ $n -ge 96 ] && { log "$name GAVE UP after $n tries"; exit 3; }
    sleep 300
    Q=$B/tmp/gpu1_queue; P=${WAIT_SLOT:-9}; nm=$(printf "%s_figs_v2_%s_r%02d.sh" "$P" "$name" "$n")
    printf "#!/bin/bash
# FIGS_V2 retry %s of %s (gpu1 lacked memory); see tmp/figs_v2/run_new.sh
export WAIT_SLOT=%s
exec bash %q" "$n" "$name" "$P" "$B/tmp/figs_v2/run_new.sh" > $Q/$nm.tmp
    for a in "$@"; do printf " %q" "$a" >> $Q/$nm.tmp; done
    echo >> $Q/$nm.tmp; mv $Q/$nm.tmp $Q/$nm
    exit 3
  fi
  log "run $name start: sphere -> $tgt, gpu1 free $free MiB"
  ( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)"; sleep 10; done ) > $R/$name.mem &
  mem=$!
  t0=$(date +%s)
  $PY scripts/pipeline_run.py --src assets/isosphere.obj --tgt $tgt --n 300000 --seed 97 --out $R/$name > $R/$name.log 2>&1 < /dev/null
  rc=$?
  kill $mem
  echo "$rc $(( $(date +%s) - t0 )) $(sort -k2 -n $R/$name.mem | tail -1 | cut -d' ' -f2)" > $R/rc_$name
  log "run $name rc/wall/peak_card_mib $(cat $R/rc_$name); $(grep -m1 'early stop' $R/$name.log)"
fi
if [ "$(cut -d' ' -f1 $R/rc_$name)" != 0 ] || [ ! -e $F ]; then log "$name FAILED or no kept frames"; exit 1; fi
if [ ! -s $O/mid/$name.json ]; then
  CUDA_VISIBLE_DEVICES="" nice -n 10 $PY $B/tmp/showcase/sc_mid.py $O/mid/$name.json $F > $O/mid/$name.log 2>&1 < /dev/null
fi
mid=$($PY -c "import json; print(json.load(open('$O/mid/$name.json'))['mid'])")
ks=${ks//mid/$mid}
t0=$(date +%s)
$PY $T/fig_render.py $F $dense $outd +ks=$ks +az=$az +el=$el +dist=$dist +min_piece=$rule +raw=$raw > $outd.log 2>&1 < /dev/null
rc=$?
log "render $name ks $ks az $az el $el dist $dist min_piece $rule rc $rc $(( $(date +%s) - t0 )) s"
[ $rc = 0 ] || exit 1
mkdir -p $outd/stills
for f in $outd/frames/*.jpg; do ffmpeg -loglevel error -y -i $f -vf hflip -q:v 2 $outd/stills/$(basename $f) < /dev/null; done
log "stills $name: $(ls $outd/stills | wc -l) (hflipped)"
if [ "$raw" = 0 ] && [ ! -s $O/pieces_$name.jsonl ]; then
  # a new montage example: the pieces off the body over the run (D145's pieces145.py, every 4th kept frame and the end)
  # and the stop; not clean (an early stop, or a piece of >= 10 particles or >= 5 pitches off at the end) -> the next backup
  CUDA_VISIBLE_DEVICES="" nice -n 10 $PY $B/tmp/d145r/pieces145.py $O/pieces_$name.jsonl $name=$F --every 4 --threads 8 > $O/pieces_$name.log 2>&1 < /dev/null
  es=$(grep -ci "early stop" $R/$name.log)
  ser=$($PY -c "import json; r = [json.loads(s) for s in open('$O/pieces_$name.jsonl')][-1]; print(r['end_ge10'] + r['end_ge5a'])" 2>/dev/null || echo 99)
  if [ "$es" -gt 0 ] || [ "$ser" != 0 ]; then
    log "$name NOT CLEAN (early-stop lines $es, serious pieces at the end $ser): replaced by the next backup"
    echo $name >> $O/rejected.txt
    nb=$(head -1 $T/backups.txt)
    if [ -n "$nb" ]; then
      sed -i 1d $T/backups.txt
      set -- $nb
      Q=$B/tmp/gpu1_queue; jf=${7}_figs_v2_$1.sh
      printf "#!/bin/bash
# FIGS_V2 backup for a montage example that was not clean (see output/gpu/figs_v2/status.txt)
export WAIT_SLOT=%s
exec bash %s %s %s %s %s %s 0 0,5,10,15,20,30,50,mid,last %s %s 0
" "$7" "$T/run_new.sh" "$1" "$2" "$3" "$4" "$5" "$O/ind/${6}_dense15.npz" "$O/render/$1" > $Q/$jf.tmp
      mv $Q/$jf.tmp $Q/$jf
      log "queued backup $jf"
    fi
  else
    echo $name >> $O/accepted.txt
    log "$name clean (no early stop, no serious piece at the end): accepted for the montage"
  fi
fi
bash $T/finish_figs.sh
exit 0
