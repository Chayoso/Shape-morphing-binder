#!/bin/bash
# b12_pair.sh NAME METHOD CORR S -- D145 B12 (amendment 5, 2026-10-15): one external neural morph sphere -> NAME with the
# authors' own code. METHOD ISD (Sang et al., ICLR 2025, Implicit-surf-Deformation @ e7992a5) or FD (4Deform, CVPR 2025,
# @ 7f2274e); CORR none (matching weight 0, ISD separate_train 0) or radial (the shipped matching weights); S = 1, 2, 3 ->
# rng_seed = shipped + S - 1 (ISD 0, 1, 2; 4Deform 5, 6, 7). Runs as a D145 pool job: the pool claims the card, sets
# CUDA_VISIBLE_DEVICES and releases the card when this script exits. Training and evaluation (21 meshes at t = k/20,
# marching cubes 256, every component kept: eval_full.py) on the card; then the conversion to the pipeline frame, the
# 300k sample and the end check detached on the CPU (b12_convert.sh).
set -u
NAME=$1; METHOD=$2; CORR=$3; S=$4
B=/data/relcfd/chayo/physmorph_v2; BL=$B/baselines; W=$B/tmp/d145/b12; R=$BL/b12
TAG=${METHOD}${CORR:0:1}${S}
RUN=${NAME}300k_${TAG}
case $METHOD in
  ISD) REPO=$BL/Implicit-surf-Deformation; TARGS="--index 0 --subindex 1"; SEED=$((S - 1)); TPL=isd_pair.conf;;
  FD) REPO=$BL/4Deform; TARGS="--start_frame 0 --length 1"; SEED=$((4 + S)); TPL=fourdeform_pair.conf;;
  *) echo "METHOD ISD|FD"; exit 2;;
esac
if [ "$CORR" = none ]; then MI=0; MF=0; SEP=0; else MI=200; MF=100; SEP=1; fi
[ $METHOD = ISD ] && MW=$MI || MW=$MF
mkdir -p $R/logs $R/recon $R/results $R/cache/tmp $R/cache/xdg $R/cache/mpl
export TMPDIR=$R/cache/tmp XDG_CACHE_HOME=$R/cache/xdg MPLCONFIGDIR=$R/cache/mpl PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
ST=$R/status.txt; LOG=$R/logs/$RUN.log
say() { echo "$(date '+%Y-%m-%d %H:%M:%S %Z') $RUN $*" >> $ST; }
SRC=$BL/Implicit-surf-Deformation/data/b12_${NAME}_${CORR}
[ -s $SRC/pair_info.json ] || { say "FAILED: no prepared pair $SRC"; exit 1; }
source /home/chayo/miniforge3/etc/profile.d/conda.sh; conda activate nsd
cd $REPO
DR=data/b12r_$RUN                                   # this run's own copy: evaluation writes beside the data
rm -rf $DR exp_b12/$RUN; cp -rL $SRC $DR
sed -e "s#DATA_ROOT#./$DR#" -e "s#MATCH_W#$MW#" -e "s#SEP_TRAIN#$SEP#" -E -e "s/rng_seed *= *[0-9]+/rng_seed = $SEED/" \
  $B/baselines_prep/conf/$TPL > conf/b12_$RUN.conf
say "train start (gpu $CUDA_VISIBLE_DEVICES, rng_seed $SEED, match $MW, separate_train $SEP)"
t0=$(date +%s)
python train.py --conf ./conf/b12_$RUN.conf --savedir ./exp_b12 --expname $RUN $TARGS --reset > $LOG 2>&1 || { say "FAILED train"; exit 1; }
tt=$(( $(date +%s) - t0 ))
md=$(ls -d exp_b12/$RUN/*/ | sort | tail -1)
say "train done $tt s ($md)"
t0=$(date +%s)
python $W/eval_full.py --modeldir $md --steps 20 --mc_resolution 256 >> $LOG 2>&1 || { say "FAILED eval"; exit 1; }
te=$(( $(date +%s) - t0 ))
REC=$R/recon/$RUN; rm -rf $REC; mkdir -p $REC
find $REPO/$DR $REPO/$md -type f | grep -E "(step|time)_[0-9]{2}_mesh\.(ply|obj)$" | while read -r f; do
  k=$(echo "$f" | sed -E 's/.*(step|time)_([0-9]{2})_mesh\.(ply|obj)$/\2/'); ext=${f##*.}
  mv "$f" $REC/time_$((10#$k)).$ext
done
n=$(ls $REC | grep -c "^time_")
gname=$(nvidia-smi -i ${CUDA_VISIBLE_DEVICES%%,*} --query-gpu=name --format=csv,noheader)
echo "{\"run\": \"$RUN\", \"method\": \"$METHOD\", \"corr\": \"$CORR\", \"seed_index\": $S, \"rng_seed\": $SEED, \"train_s\": $tt, \"eval_s\": $te, \"meshes\": $n, \"modeldir\": \"$REPO/$md\", \"gpu\": \"$gname\"}" >> $R/results/times.jsonl
say "eval done $te s, $n meshes"
[ $n -eq 21 ] || { say "FAILED: $n meshes, expected 21"; exit 1; }
conda deactivate
setsid nohup bash $W/b12_convert.sh $NAME $TAG $RUN $REPO/$DR > /dev/null 2>&1 < /dev/null &
exit 0
