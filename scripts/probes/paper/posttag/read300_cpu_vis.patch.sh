#!/bin/bash
# D145 post-tag: read300_cpu.sh also runs D148's visible_apart.py (windows 3-10, and the whole run at every 4th kept
# frame) for the tag's runs (folders d145t_*), before the kept frames go; nothing changes for the other folders.
F=/data/relcfd/chayo/physmorph_v2/tmp/d145/read300_cpu.sh
grep -q "visible_apart" $F && { echo "already there"; exit 0; }
cp $F $F.bak_vis
awk '/^if \[ "\$keep" = 0 \]; then$/ && !d { print "case $dir in d145t_*)   # D145 post-tag: visible pieces (D148 visible_apart.py, the tag repo copy)";
  print "  ( export PHYSMORPH_RUN_REPO=$B/repo_t; source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1; cd $B/repo_t";
  print "    CUDA_VISIBLE_DEVICES=\"\" nice -n 10 $PY scripts/probes/paper/visible_apart.py $F --label $name --jsonl $D/vis_w3_10_$name.jsonl --threads 8 > $D/vis_$name.log 2>&1";
  print "    CUDA_VISIBLE_DEVICES=\"\" nice -n 10 $PY scripts/probes/paper/visible_apart.py $F --label $name --w0 0 --w1 100000 --every 4 --jsonl $D/vis_full_$name.jsonl --threads 8 >> $D/vis_$name.log 2>&1 ) < /dev/null;;";
  print "esac"; d=1 } { print }' $F > $F.new && mv $F.new $F && chmod +x $F && bash -n $F && echo patched
