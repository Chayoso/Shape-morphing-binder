#!/bin/bash
# pick_variant.sh BUILD_PID -- D172 (D): after build_scales.sh, write output/gpu/d172/assets/variant.txt: the first scale
# variant (A, then B) whose perfect 300k sample reads corr >= 0.25 in the pattern region (empty: none passes).
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/d172
while ps -p $1 > /dev/null 2>&1; do sleep 30; done
v=$(/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python - << 'EOP'
import json
for v in ("A", "B"):
    try:
        r = json.load(open(f"/data/relcfd/chayo/physmorph_v2/output/gpu/d172/size_probe/S{v}300.json"))["refs"]["detail_s197"]["relief"]
    except Exception:
        continue
    if (r.get("corr") or 0) >= 0.25:
        print(v)
        break
EOP
)
echo -n "$v" > $O/assets/variant.txt
echo "$(date '+%m-%d %H:%M:%S') [scales] variant chosen: '${v}' (corr >= 0.25 at 300k; empty = none)" >> $O/status.txt
