#!/bin/bash
# D165 follow-up (CPU, detached, outside the queue): once job c has run (STAMP_DONE, at most 12 h), the reading
# (d165_read.py), the imprint close-ups (d165_render.py), README_ko.md (5 lines) beside summary.md; DONE when through.
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/d165; W=$B/tmp/d165; D3=$B/output/gpu/d163
PAT=$D3/assets/d163R300plaque.obj.pattern.npz; CEIL=$D3/assets/ceiling300_frames12.npz
log() { echo "$(date '+%F %T') [follow] $*" >> $O/status.txt; }
for k in $(seq 1 1440); do [ -e $O/STAMP_DONE ] && break; sleep 30; done
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 D157_THREADS=8 MPLCONFIGDIR=$B/cache/matplotlib PYTHONPATH=$B/repo_d163
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
cd $B/repo_d163
A=()
[ -f $O/stamp/e_H_end.npz ] && A+=("full (dF_c + layer u + render loss):$O/stamp/e_H")
[ -f $O/stamp/e_T_end.npz ] && A+=("twin (dF_c + layer u) + post-hoc decal:$O/stamp/e_T")
[ -f $O/stamp/k_H_end.npz ] && A+=("full-endframe (end shape as rest shape):$O/stamp/k_H")
[ -f $O/stamp/k_T_end.npz ] && A+=("twin-endframe (end shape as rest shape):$O/stamp/k_T")
[ -f $O/stamp/k40_H_end.npz ] && A+=("full40 (40k, end shape as rest shape):$O/stamp/k40_H")
[ -f $O/stamp/k40_T_end.npz ] && A+=("twin40 (40k, end shape as rest shape):$O/stamp/k40_T")
log "reading ${#A[@]} runs"
nice -n 10 $PY $W/d165_read.py $O/read $PAT $CEIL "${A[@]}" > $O/read.log 2>&1
log "read exit $?"
nice -n 10 $PY $W/d165_render.py $O/read $PAT "${A[@]}" > $O/render_imprint.log 2>&1
log "imprint render exit $?"
$PY - > $O/read/README_ko.md 2>> $O/read.log <<PYEOF
import json
d = json.load(open("$O/read/read.json")); r = {x["label"].split(" ")[0]: x for x in d["runs"]}
f, t = r.get("full"), r.get("twin")
def g(x, k): return "n/a" if x is None else f"{x[k]:.2f}"
print("# D165 도장 테스트 (자동 요약)")
print(f"1. 판을 글자 면이 아래로 가게 부드러운 판(E/20)에 눌렀다 들어 올림. 판은 E×7, 끝 상태(F, Fp) 그대로 사용.")
print(f"2. full (dF_c + layer u + render loss): 판 글자 대비 {g(f,'plaque_settle_contrast_a0')} a0, 찍힌 자국 대비 {g(f,'slab_contrast_a0')} a0, 글자와의 상관 {g(f,'slab_corr')}.")
print(f"3. twin (dF_c + layer u) + post-hoc decal: 판 글자 대비 {g(t,'plaque_settle_contrast_a0')} a0, 자국 대비 {g(t,'slab_contrast_a0')} a0, 상관 {g(t,'slab_corr')} (decal은 화면에만 있어 판에 닿지 않음).")
print(f"4. 천장(relief mesh 300k): 글자 대비 {d['ceiling']['contrast_a0']:.2f} a0. a0 = 입자 하나 부피의 세제곱근.")
print("5. 파일: imprint_sheet.png(위에서 본 자국, 좌우 반전해 읽히게), imprint_depth.png(깊이 지도), stamp_full_vs_twin_*.mp4, decal/ 스틸.")
PYEOF
log "README exit $?"
touch $O/DONE
log "done"
